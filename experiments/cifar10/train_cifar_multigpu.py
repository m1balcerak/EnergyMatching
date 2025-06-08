##############################################################################
# train_vit.py
# Now includes a --split_negative flag which controls if negative samples
# are 50/50 split between real and noise initialization or all noise.
##############################################################################
import os
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

config.define_flags()  # register all the flags
FLAGS = flags.FLAGS

# 2) Import your usual goodies
from torchvision import datasets, transforms

from utils_train import (
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

# Import single-GPU FID
from utils_fid import compute_fid_for_times

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
                n_gibbs,
                dt_gibbs,
                epsilon_max,
                time_cutoff):
    """
    Do the entire forward pass (flow + optional CD) using the
    *DDP-wrapped* model. We have two mini-batches: one for flow,
    one for CD.
    Returns: total_loss, flow_loss, cd_loss
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
        else:
            # Original approach: all negative samples from noise
            x_neg_init = torch.randn_like(x_real_cd)

        x_neg = gibbs_sampling_time_sweep(
            x_init=x_neg_init,
            model=model.module,
            n_steps=n_gibbs,
            dt=dt_gibbs
        )

        neg_energy = model.module.potential(x_neg, torch.ones_like(t))
        cd_val = pos_energy.mean() - neg_energy.mean()

        cd_val_scaled = lambda_cd * cd_val
        if cd_neg_clamp > 0:
            cd_val_scaled = torch.maximum(
                cd_val_scaled,
                torch.tensor(-cd_neg_clamp, device=device)
            )
        cd_loss = cd_val_scaled

    total_loss = flow_loss + cd_loss
    return total_loss, flow_loss, cd_loss


##############################################################################
# Training loop
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
    # 2) CIFAR10 dataset with distributed sampler
    # -----------------------------------------------------------------------
    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=(rank == 0),
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
    )
    dist.barrier()  # ensure download is done
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

    net_model = DDP(net_model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

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
            total_loss, flow_loss, cd_loss = forward_all(
                model=net_model,
                flow_matcher=flow_matcher,
                x_real_flow=x_real_flow,
                x_real_cd=x_real_cd,
                lambda_cd=FLAGS.lambda_cd,
                cd_neg_clamp=FLAGS.cd_neg_clamp,
                n_gibbs=FLAGS.n_gibbs,
                dt_gibbs=FLAGS.dt_gibbs,
                epsilon_max=FLAGS.epsilon_max,
                time_cutoff=FLAGS.time_cutoff
            )

            # -----------------------------------------------------------------
            # If cd_loss < -cd_loss_threshold, increase difficulty
            # (increase n_gibbs by 10%, decrease dt_gibbs by 10%)
            # -----------------------------------------------------------------
            cd_val = cd_loss.item()
            if rank == 0 and cd_val < -FLAGS.cd_loss_threshold:
                old_n_gibbs = FLAGS.n_gibbs
                old_dt_gibbs = FLAGS.dt_gibbs

                # Propose new values
                proposed_n_gibbs = math.ceil(old_n_gibbs * 1.10)
                proposed_dt_gibbs = old_dt_gibbs * 0.90

                # Clamp values
                new_n_gibbs = min(proposed_n_gibbs, 400)
                new_dt_gibbs = max(proposed_dt_gibbs, 0.01)

                # Check if any update actually occurs
                if new_n_gibbs != old_n_gibbs or abs(new_dt_gibbs - old_dt_gibbs) > 1e-12:
                    # Apply changes
                    FLAGS.n_gibbs = new_n_gibbs
                    FLAGS.dt_gibbs = new_dt_gibbs
                    logging.info(
                        f"cd_loss={cd_val:.4f} < -{FLAGS.cd_loss_threshold:.4f} => "
                        f"Increasing difficulty: n_gibbs {old_n_gibbs} -> {new_n_gibbs}, "
                        f"dt_gibbs {old_dt_gibbs:.4f} -> {new_dt_gibbs:.4f}"
                    )
                else:
                    # We’re at the boundary, no changes applied
                    logging.info(
                        f"cd_loss={cd_val:.4f} < -{FLAGS.cd_loss_threshold:.4f} => "
                        f"Already at boundary (n_gibbs={old_n_gibbs}, dt_gibbs={old_dt_gibbs}); no change."
                    )

            dist.barrier()  # synchronize before broadcast
            # Broadcast new n_gibbs, dt_gibbs to all ranks
            n_gibbs_tensor = torch.tensor(FLAGS.n_gibbs, dtype=torch.int, device=device)
            dt_gibbs_tensor = torch.tensor(FLAGS.dt_gibbs, dtype=torch.float, device=device)
            dist.broadcast(n_gibbs_tensor, src=0)
            dist.broadcast(dt_gibbs_tensor, src=0)
            FLAGS.n_gibbs = int(n_gibbs_tensor.item())
            FLAGS.dt_gibbs = float(dt_gibbs_tensor.item())
            # -----------------------------------------------------------------

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
                    f"LR={curr_lr:.6f}, {sps:.2f} it/s"
                )

            # -------------------------------------------------
            # Single-GPU FID on rank=0 (others wait)
            # -------------------------------------------------
            if (
                FLAGS.fid_freq > 0
                and step > 0
                and step % FLAGS.fid_freq == 0
                and step != start_step
            ):
                if rank == 0:
                    logging.info(f"[Rank 0] Computing single-GPU FID at step={step} ...")
                    fid_times = [float(t_str) for t_str in FLAGS.fid_times]
                    with torch.no_grad():
                        fid_scores = compute_fid_for_times(
                            model=ema_model,
                            device=device,
                            times=fid_times,
                            num_gen=FLAGS.fid_num_gen,
                            batch_size=FLAGS.batch_size,
                            dt=FLAGS.fid_dt
                        )
                    for T, fid_val in fid_scores.items():
                        logging.info(f"[Rank 0] FID at T={T}: {fid_val:.4f}")

                dist.barrier()  # block so all ranks stay in sync

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
                x_neg = gibbs_sampling_time_sweep(
                    x_init=x_neg_init,
                    model=net_model.module,
                    n_steps=FLAGS.n_gibbs,
                    dt=FLAGS.dt_gibbs
                )
                # (c) Save side-by-side grids
                save_pos_neg_grids(real_batch, x_neg, savedir, step)

                ckpt_path = os.path.join(savedir, f"{FLAGS.model}_cifar10_weights_step_latest.pt")
                torch.save(
                    {
                        "net_model": net_model.module.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                    },
                    ckpt_path
                )
                logging.info(f"[Rank 0] Saved checkpoint => {ckpt_path}")

    dist.barrier()
    dist.destroy_process_group()


def main(argv):
    world_size = torch.cuda.device_count()
    rank = int(os.environ.get("RANK", 0))  # or local_rank
    train_loop(rank, world_size, argv)


if __name__ == "__main__":
    app.run(main)
