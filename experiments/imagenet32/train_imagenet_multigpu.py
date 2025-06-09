import os
import time
import copy

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from absl import app, flags, logging
import config_multigpu_imagenet32 as config  # config file for ImageNet32
config.define_flags()
FLAGS = flags.FLAGS

from torchvision import transforms

from utils_train import (
    create_timestamped_dir,
    generate_samples,
    gibbs_sampling_time_sweep,
    warmup_lr,
    ema,
    infiniteloop,
    save_pos_neg_grids
)

from network_transformer_vit import EBViTModelWrapper
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from dataset_imagenet32 import ImageNet32Dataset  # custom dataset


##############################################################################
def count_parameters(module: torch.nn.Module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


##############################################################################
def forward_all(model,
                flow_matcher,
                x_real_flow,
                x_real_cd,
                lambda_cd,
                cd_neg_clamp,
                n_gibbs,
                dt_gibbs,
                epsilon_max):
    """
    Computes flow_loss and optional CD loss in a single forward pass.
    Uses uniform weighting (1.0) for the flow MSE.
    
    Now does a one-sided trimmed mean (drop top 20%) on negative energies
    to stabilize the CD gradient.
    
    Returns: (total_loss, flow_loss, cd_loss, pos_energy, neg_energy)
    so we can log energy statistics outside.
    """
    device = x_real_flow.device

    # ----------------------
    # 1) Flow matching
    # ----------------------
    x0_flow = torch.randn_like(x_real_flow)
    t, xt, ut = flow_matcher.sample_location_and_conditional_flow(x0_flow, x_real_flow)

    vt = model(t, xt)
    flow_mse = (vt - ut).square()
    flow_loss = flow_mse.mean()

    # ----------------------
    # 2) Contrastive Divergence (CD)
    # ----------------------
    cd_loss = torch.tensor(0.0, device=device)
    pos_energy = torch.tensor(0.0, device=device)
    neg_energy = torch.tensor(0.0, device=device)

    if lambda_cd > 0.0:
        # positive energies (data)
        pos_energy = model.module.potential(x_real_cd, torch.ones_like(t))

        # initialize negatives
        if FLAGS.split_negative:
            B = x_real_cd.size(0)
            half_b = B // 2
            x_neg_init = torch.empty_like(x_real_cd)
            x_neg_init[:half_b] = x_real_cd[:half_b]
            x_neg_init[half_b:] = torch.randn_like(x_neg_init[half_b:])
            at_data_mask = torch.zeros(B, dtype=torch.bool, device=device)
            at_data_mask[:half_b] = True
        else:
            x_neg_init = torch.randn_like(x_real_cd)
            at_data_mask = torch.zeros(x_real_cd.size(0), dtype=torch.bool, device=device)

        # run Gibbs chain
        x_neg = gibbs_sampling_time_sweep(
            x_init=x_neg_init,
            model=model.module,
            at_data_mask=at_data_mask,
            n_steps=n_gibbs,
            dt=dt_gibbs
        )

        neg_energy = model.module.potential(x_neg, torch.ones_like(t))

        # trimmed-mean on neg energies: drop top 20%
        B = neg_energy.size(0)
        k = int(0.25 * B)
        if k > 0:
            neg_sorted, _ = neg_energy.sort()    # ascending
            neg_trimmed = neg_sorted[: B - k]    # drop highest k
            neg_stat = neg_trimmed.mean()
        else:
            neg_stat = neg_energy.mean()

        # CD value: data minus (trimmed) model
        cd_val = pos_energy.mean() - neg_stat

        # scale & clamp
        cd_val_scaled = lambda_cd * cd_val
        if cd_neg_clamp > 0:
            cd_val_scaled = torch.maximum(
                cd_val_scaled,
                torch.tensor(-cd_neg_clamp, device=device)
            )
        cd_loss = cd_val_scaled

    return flow_loss + cd_loss, flow_loss, cd_loss, pos_energy, neg_energy


##############################################################################
def train_loop(rank, world_size, argv):
    # -----------------------------------------------------------------------
    # 0) Init distributed
    # -----------------------------------------------------------------------
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # -----------------------------------------------------------------------
    # 1) Create output dir on rank=0
    # -----------------------------------------------------------------------
    savedir = None
    if rank == 0:
        savedir = create_timestamped_dir(FLAGS.output_dir, FLAGS.model)
        if not FLAGS.my_log_dir:
            FLAGS.my_log_dir = savedir

        logging.get_absl_handler().use_absl_log_file(
            program_name="train",
            log_dir=FLAGS.my_log_dir
        )
        logging.set_verbosity(logging.INFO)
        logging.info(f"[Rank 0] Using output directory: {savedir}\n")
        logging.info("========== Hyperparameters (FLAGS) ==========")
        for key, val in FLAGS.flag_values_dict().items():
            logging.info(f"{key} = {val}")
        logging.info("=============================================\n")

    # -----------------------------------------------------------------------
    # 2) ImageNet32 Dataset + distributed sampler
    # -----------------------------------------------------------------------
    dataset = ImageNet32Dataset(
        split="train",
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )
    dist.barrier()
    train_sampler = DistributedSampler(dataset, num_replicas=world_size,
                                       rank=rank, shuffle=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
        sampler=train_sampler,
        drop_last=True,
        pin_memory=True
    )
    datalooper = infiniteloop(dataloader)

    # -----------------------------------------------------------------------
    # 3) Model + DDP
    # -----------------------------------------------------------------------
    ch_mult = config.parse_channel_mult(FLAGS)
    net_model = EBViTModelWrapper(
        dim=(3, 32, 32),
        num_channels=FLAGS.num_channels,
        num_res_blocks=FLAGS.num_res_blocks,
        channel_mult=ch_mult,
        attention_resolutions=FLAGS.attention_resolutions,
        num_heads=FLAGS.num_heads,
        num_head_channels=FLAGS.num_head_channels,
        dropout=FLAGS.dropout,
        output_scale=FLAGS.output_scale,
        energy_clamp=FLAGS.energy_clamp,
        patch_size=4,
        embed_dim=FLAGS.embed_dim,
        transformer_nheads=FLAGS.transformer_nheads,
        transformer_nlayers=FLAGS.transformer_nlayers,
    ).to(device)

    # Enable find_unused_parameters only when CD loss is disabled. With CD loss
    # (lambda_cd > 0) all parameters participate in the backward pass.
    find_unused = False if FLAGS.lambda_cd > 0.0 else True
    net_model = DDP(net_model, device_ids=[rank],
                    output_device=rank, find_unused_parameters=find_unused)

    ema_model = copy.deepcopy(net_model.module).to(device)

    if rank == 0:
        total_params = count_parameters(net_model.module)
        logging.info(f"Total trainable params: {total_params}")

    # -----------------------------------------------------------------------
    # 4) Optimizer & scheduler
    # -----------------------------------------------------------------------
    optim = torch.optim.Adam(
        net_model.parameters(),
        lr=FLAGS.lr,
        betas=(0.9, 0.95)
    )
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)

    # -----------------------------------------------------------------------
    # 5) Load checkpoint if any
    # -----------------------------------------------------------------------
    start_step = 0
    checkpoint_data = None
    if rank == 0 and FLAGS.resume_ckpt and os.path.exists(FLAGS.resume_ckpt):
        logging.info(f"[Rank 0] Resuming from {FLAGS.resume_ckpt}")
        checkpoint_data = torch.load(FLAGS.resume_ckpt, map_location=device)

    dist.barrier()
    checkpoint_data = [checkpoint_data]
    dist.broadcast_object_list(checkpoint_data, src=0)
    checkpoint_data = checkpoint_data[0]

    if checkpoint_data is not None:
        net_model.module.load_state_dict(checkpoint_data["net_model"])
        ema_model.load_state_dict(checkpoint_data["ema_model"])
        optim.load_state_dict(checkpoint_data["optim"])
        for state in optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        sched.load_state_dict(checkpoint_data["sched"])
        start_step = checkpoint_data["step"]
        if rank == 0:
            logging.info(f"[Rank 0] Continuing from step {start_step}.\n")

    # -----------------------------------------------------------------------
    # 6) Flow matcher & logging setup
    # -----------------------------------------------------------------------
    sigma = 0.0
    flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    steps_per_log = 1
    last_log_time = time.time()

    # -----------------------------------------------------------------------
    # 7) Training Loop
    # -----------------------------------------------------------------------
    with torch.backends.cuda.sdp_kernel(enable_math=True,
                                        enable_flash=False,
                                        enable_mem_efficient=False):
        for step in range(start_step, FLAGS.total_steps + 1):
            train_sampler.set_epoch(step)
            optim.zero_grad()

            x_real_flow = next(datalooper).to(device)
            x_real_cd   = next(datalooper).to(device)

            total_loss, flow_loss, cd_loss, pos_energy, neg_energy = forward_all(
                model=net_model,
                flow_matcher=flow_matcher,
                x_real_flow=x_real_flow,
                x_real_cd=x_real_cd,
                lambda_cd=FLAGS.lambda_cd,
                cd_neg_clamp=FLAGS.cd_neg_clamp,
                n_gibbs=FLAGS.n_gibbs,
                dt_gibbs=FLAGS.dt_gibbs,
                epsilon_max=FLAGS.epsilon_max
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()
            ema(net_model.module, ema_model, FLAGS.ema_decay)

            if rank == 0 and step % steps_per_log == 0:
                now = time.time()
                elapsed = now - last_log_time
                sps = steps_per_log / elapsed if elapsed > 1e-9 else 0.0
                last_log_time = now
                curr_lr = sched.get_last_lr()[0]

                logging.info(
                    f"[Step {step}] flow={flow_loss.item():.5f}, "
                    f"cd={cd_loss.item():.5f}, "
                    f"pos_std={pos_energy.std().item():.5f}, pos_min={pos_energy.min().item():.5f}, pos_max={pos_energy.max().item():.5f}, "
                    f"neg_std={neg_energy.std().item():.5f}, neg_min={neg_energy.min().item():.5f}, neg_max={neg_energy.max().item():.5f}, "
                    f"LR={curr_lr:.6f}, {sps:.2f} it/s"
                )

            if rank == 0 and FLAGS.save_step > 0 and step % FLAGS.save_step == 0 and step > 0:
                real_batch = next(datalooper).to(device)[:8]
                generate_samples(net_model.module, savedir, step,
                                 net_="normal", real_data=real_batch)
                generate_samples(ema_model, savedir, step,
                                 net_="ema", real_data=real_batch)

                real_batch = next(datalooper).to(device)[:64]
                x_neg_init = torch.randn_like(real_batch)
                at_data_mask = torch.zeros(real_batch.size(0), dtype=torch.bool, device=device)
                x_neg = gibbs_sampling_time_sweep(
                    x_init=x_neg_init,
                    model=net_model.module,
                    at_data_mask=at_data_mask,
                    n_steps=FLAGS.n_gibbs,
                    dt=FLAGS.dt_gibbs
                )
                save_pos_neg_grids(real_batch, x_neg, savedir, step)

                ckpt_path = os.path.join(
                    savedir, f"{FLAGS.model}_imagenet32_weights_step_latest.pt"
                )
                torch.save(
                    {
                        "net_model": net_model.module.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "optim": optim.state_dict(),
                        "sched": sched.state_dict(),
                        "step": step,
                    },
                    ckpt_path
                )
                logging.info(f"[Rank 0] Saved checkpoint => {ckpt_path}")

    dist.barrier()
    dist.destroy_process_group()


def main(argv):
    world_size = torch.cuda.device_count()
    rank = int(os.environ.get("RANK", 0))
    train_loop(rank, world_size, argv)


if __name__ == "__main__":
    app.run(main)
