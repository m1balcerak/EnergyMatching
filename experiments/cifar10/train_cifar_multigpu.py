##############################################################################
# train_vit.py
# Now includes a --split_negative flag which controls if negative samples
# are 50/50 split between real and noise initialization or all noise.
##############################################################################
import os
import sys
import time
import copy
import datetime
import math  # <--- import for ceiling

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 1) Import absl + config
from absl import app, flags, logging
import config_multigpu as config  # your config file

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

config.define_flags()  # register all the flags
FLAGS = flags.FLAGS

# 2) Import your usual goodies
from torchvision import datasets, transforms

from utils_cifar_imagenet import (
    create_timestamped_dir,
    generate_samples,
    flow_weight,
    gibbs_sampling_time_sweep,
    warmup_lr,
    ema,
    infiniteloop,
    save_pos_neg_grids,
    sde_euler_maruyama
)


# 3) Import the EBM model (ViT version)
from network_transformer_vit import EBViTModelWrapper

# TorchCFM flow classes
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher


##############################################################################
# Helper: count_parameters
##############################################################################
def count_parameters(module: torch.nn.Module):
    """Count the total trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


##############################################################################
# Single forward function that computes flow_loss + cd_loss in one go,
# but now uses separate mini-batches: x_real_flow for flow, x_real_cd for CD.
##############################################################################
def forward_all(model,
                flow_matcher,
                x_real_flow,
                x_real_cd,       # separate CD batch
                lambda_cd,
                cd_neg_clamp,
                cd_trim_fraction,
                n_gibbs,
                dt_gibbs,
                epsilon_max,
                time_cutoff):
    """
    Do the entire forward pass (flow + optional CD) using the
    *DDP-wrapped* model. We have two mini-batches: one for flow,
    one for CD.

    Returns: ``total_loss, flow_loss, cd_loss, pos_energy, neg_energy`` so
    that the caller can log energy statistics similarly to the ImageNet
    training script. Optionally discards a fraction of highest negative
    energies (``cd_trim_fraction``) when computing the CD gradient.
    """
    device = x_real_flow.device

    # ----------------------------------------------------------
    # 1) Flow matching (using x_real_flow)
    # ----------------------------------------------------------
    x0_flow = torch.randn_like(x_real_flow)
    t, xt, ut = flow_matcher.sample_location_and_conditional_flow(x0_flow, x_real_flow)

    vt = model(t, xt)  # calls forward() in EBViTModelWrapper
    flow_mse = (vt - ut).square()
    w_flow = flow_weight(t, cutoff=time_cutoff)
    flow_loss = torch.mean(w_flow * flow_mse.mean(dim=[1, 2, 3]))

    # ----------------------------------------------------------
    # 2) Optional CD loss (using x_real_cd)
    # ----------------------------------------------------------
    cd_loss = torch.tensor(0.0, device=device)
    pos_energy = torch.tensor(0.0, device=device)
    neg_energy = torch.tensor(0.0, device=device)
    if lambda_cd > 0.0:
        pos_energy = model.module.potential(x_real_cd, torch.ones_like(t))

        ### NEW/CHANGED: Conditionally split negative samples based on flag.
        if FLAGS.split_negative:
            # 50/50 split: half from x_real_cd, half from noise
            B = x_real_cd.size(0)
            half_b = B // 2
            x_neg_init = torch.empty_like(x_real_cd)

            x_neg_init[:half_b] = x_real_cd[:half_b]
            x_neg_init[half_b:] = torch.randn_like(x_neg_init[half_b:])
            at_data_mask = torch.zeros(B, dtype=torch.bool, device=device)
            at_data_mask[:half_b] = True
        else:
            # Original approach: all negative samples from noise
            x_neg_init = torch.randn_like(x_real_cd)
            at_data_mask = torch.zeros(x_real_cd.size(0), dtype=torch.bool, device=device)

        if FLAGS.same_temperature_scheduler:
            at_data_mask = torch.zeros_like(at_data_mask)

        x_neg = gibbs_sampling_time_sweep(
            x_init=x_neg_init,
            model=model.module,
            at_data_mask=at_data_mask,
            n_steps=n_gibbs,
            dt=dt_gibbs
        )

        neg_energy = model.module.potential(x_neg, torch.ones_like(t))

        # Optionally use a trimmed mean for the negative energies
        if cd_trim_fraction > 0.0:
            B = neg_energy.size(0)
            k = int(cd_trim_fraction * B)
            if k > 0:
                neg_sorted, _ = neg_energy.sort()
                neg_trimmed = neg_sorted[: B - k]
                neg_stat = neg_trimmed.mean()
            else:
                neg_stat = neg_energy.mean()
        else:
            neg_stat = neg_energy.mean()

        cd_val = pos_energy.mean() - neg_stat

        cd_val_scaled = lambda_cd * cd_val
        if cd_neg_clamp > 0:
            cd_val_scaled = torch.maximum(
                cd_val_scaled,
                torch.tensor(-cd_neg_clamp, device=device)
            )
        cd_loss = cd_val_scaled

    total_loss = flow_loss + cd_loss
    return total_loss, flow_loss, cd_loss, pos_energy, neg_energy


##############################################################################
# Training loop
##############################################################################
def train_loop(rank, world_size, argv):
    # -----------------------------------------------------------------------
    # 0) Init distributed
    # -----------------------------------------------------------------------
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
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
    # 2) CIFAR10 dataset with distributed sampler
    # -----------------------------------------------------------------------
    data_root = os.environ.get("CIFAR10_PATH", "./data")
    if rank == 0:
        dataset = datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
        )
        dist.barrier()  # allow other ranks to see the downloaded data
    else:
        dist.barrier()  # wait for rank 0 to download
        dataset = datasets.CIFAR10(
            root=data_root,
            train=True,
            download=False,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
        )
    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
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
    # 3) Model + DDP (ViT-based EBM)
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
        # ViT-specific params:
        patch_size=4,
        embed_dim=FLAGS.embed_dim,
        transformer_nheads=FLAGS.transformer_nheads,
        transformer_nlayers=FLAGS.transformer_nlayers,
    ).to(device)

    # If we include the CD loss (lambda_cd > 0) then every parameter is used
    # in the backward pass and find_unused_parameters should be False. When the
    # CD loss is disabled some parameters are skipped and we set it to True to
    # avoid DDP errors.
    find_unused = False if FLAGS.lambda_cd > 0.0 else True
    net_model = DDP(net_model, device_ids=[rank], output_device=rank,
                    find_unused_parameters=find_unused)

    # EMA model (not DDP)
    ema_model = copy.deepcopy(net_model.module).to(device)

    # Log params count on rank=0
    if rank == 0:
        total_params = count_parameters(net_model.module)
        logging.info(f"Total trainable params: {total_params}")

    # -----------------------------------------------------------------------
    # 4) Optimizer, scheduler
    # -----------------------------------------------------------------------
    optim = torch.optim.Adam(
        net_model.parameters(),
        lr=FLAGS.lr,
        betas=(0.9, 0.95)
    )
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)

    # -----------------------------------------------------------------------
    # 5) Optional checkpoint resume
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
        sched.load_state_dict(checkpoint_data["sched"])
        optim.load_state_dict(checkpoint_data["optim"])
        # Ensure optimizer state tensors are on the correct device
        for state in optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        start_step = checkpoint_data["step"]
        if rank == 0:
            logging.info(f"[Rank 0] Resumed at step={start_step}")

    # -----------------------------------------------------------------------
    # 6) Setup flow matcher, etc.
    # -----------------------------------------------------------------------
    sigma = 0.0
    flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    steps_per_log = 10
    last_log_time = time.time()

    # -----------------------------------------------------------------------
    # 7) Actual Training Loop (with the difficulty-update logic + 2 data batches)
    # -----------------------------------------------------------------------
    with torch.backends.cuda.sdp_kernel(
        enable_math=True, enable_flash=False, enable_mem_efficient=False
    ):
        for step in range(start_step, FLAGS.total_steps + 1):
            train_sampler.set_epoch(step)  # shuffle each epoch in distributed

            optim.zero_grad()

            # Grab next batch for flow
            x_real_flow = next(datalooper).to(device)
            # Grab another batch for CD (independent from flow)
            x_real_cd = next(datalooper).to(device)

            # Forward pass (flow + optional CD) using both batches
            total_loss, flow_loss, cd_loss, pos_energy, neg_energy = forward_all(
                model=net_model,
                flow_matcher=flow_matcher,
                x_real_flow=x_real_flow,
                x_real_cd=x_real_cd,
                lambda_cd=FLAGS.lambda_cd,
                cd_neg_clamp=FLAGS.cd_neg_clamp,
                cd_trim_fraction=FLAGS.cd_trim_fraction,
                n_gibbs=FLAGS.n_gibbs,
                dt_gibbs=FLAGS.dt_gibbs,
                epsilon_max=FLAGS.epsilon_max,
                time_cutoff=FLAGS.time_cutoff
            )


            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()

            # Update EMA
            ema(net_model.module, ema_model, FLAGS.ema_decay)

            # -------------------------------------------------
            # Logging
            # -------------------------------------------------
            if rank == 0 and step % steps_per_log == 0:
                now = time.time()
                elapsed = now - last_log_time
                sps = steps_per_log / elapsed if elapsed > 1e-9 else 0.0
                last_log_time = now
                curr_lr = sched.get_last_lr()[0]
                logging.info(
                    f"[Step {step}] "
                    f"flow={flow_loss.item():.5f}, cd={cd_loss.item():.5f}, "
                    f"pos_std={pos_energy.std().item():.5f}, "
                    f"pos_min={pos_energy.min().item():.5f}, pos_max={pos_energy.max().item():.5f}, "
                    f"neg_std={neg_energy.std().item():.5f}, "
                    f"neg_min={neg_energy.min().item():.5f}, neg_max={neg_energy.max().item():.5f}, "
                    f"LR={curr_lr:.6f}, {sps:.2f} it/s"
                )

            # -------------------------------------------------
            # Save checkpoint occasionally (rank=0)
            # -------------------------------------------------
            if rank == 0 and FLAGS.save_step > 0 and step % FLAGS.save_step == 0 and step > 0:
                # generate a few samples for logging
                real_batch = next(datalooper).to(device)[:8]
                generate_samples(net_model.module, savedir, step, net_="normal", real_data=real_batch)
                generate_samples(ema_model, savedir, step, net_="ema", real_data=real_batch)

                # (a) create real data batch
                real_batch = next(datalooper).to(device)[:64]  # up to 64 for an 8x8 grid
                # (b) negative samples via MCMC (time sweep)
                x_neg_init = torch.randn_like(real_batch)
                at_data_mask = torch.zeros(real_batch.size(0), dtype=torch.bool, device=device)
                x_neg = gibbs_sampling_time_sweep(
                    x_init=x_neg_init,
                    model=net_model.module,
                    at_data_mask=at_data_mask,
                    n_steps=FLAGS.n_gibbs,
                    dt=FLAGS.dt_gibbs
                )
                # (c) Save side-by-side grids
                save_pos_neg_grids(real_batch, x_neg, savedir, step)

                ckpt_latest = os.path.join(savedir,
                                          f"{FLAGS.model}_cifar10_weights_step_latest.pt")
                ckpt_numbered = os.path.join(savedir, f"checkpoint_{step}.pt")

                checkpoint_data = {
                    "net_model": net_model.module.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "sched": sched.state_dict(),
                    "optim": optim.state_dict(),
                    "step": step,
                }

                torch.save(checkpoint_data, ckpt_latest)
                torch.save(checkpoint_data, ckpt_numbered)

                logging.info(f"[Rank 0] Saved checkpoint => {ckpt_latest}")
                logging.info(f"[Rank 0] Saved checkpoint => {ckpt_numbered}")

    dist.barrier()
    dist.destroy_process_group()


def main(argv):
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    rank = int(os.environ.get("RANK", 0))
    train_loop(rank, world_size, argv)


if __name__ == "__main__":
    app.run(main)
