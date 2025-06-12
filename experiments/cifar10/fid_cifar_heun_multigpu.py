#######################################################################
# File: fid_cifar_heun_multigpu.py
#
# Usage example:
#   torchrun --nproc_per_node=2 fid_cifar_heun_multigpu.py \
#       --resume_ckpt=/path/to/checkpoint.pt \
#       --batch_size=64 \
#       --dt_gibbs=0.01 \
#       --use_ema=True
#######################################################################

import os
import sys
import torch
import torch.distributed as dist

# absl flags
from absl import app, flags, logging

# Multi-GPU config (even if only using 1 GPU)
import config_multigpu as config
config.define_flags()
FLAGS = flags.FLAGS

flags.DEFINE_bool("use_ema", True,
                  "If True, load the EMA model from the checkpoint (default True).")

import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

# TorchMetrics FID
from torchmetrics.image.fid import FrechetInceptionDistance

# Our EBM + utilities
from network_transformer_vit import EBViTModelWrapper
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from utils_cifar_imagenet import (
    create_timestamped_dir,
    plot_epsilon
)

# Progress bar
from tqdm import tqdm

##############################################################################
# 1) CIFAR-10 Data (distributed)
##############################################################################
from torch.utils.data.distributed import DistributedSampler


def get_cifar10_train_loader(batch_size, num_workers, rank, world_size, root=None):
    """Returns a distributed DataLoader for CIFAR-10 train set."""
    if root is None:
        root = os.environ.get("CIFAR10_PATH", "./data")

    transform = T.ToTensor()
    dataset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        drop_last=False,
    )
    return loader

##############################################################################
# 2) SDE solver: Euler–Heun
##############################################################################
import torchsde

def solve_sde_heun(model, x, t_start, t_end, dt=0.01):
    """
    Integrates x from t_start..t_end using Stratonovich Euler–Heun
    (via torchsde) with no storing of entire trajectory in memory.
    Returns the final state at t_end, in [-1,1].
    """
    if t_end <= t_start:
        return x

    # Flatten for torchsde
    orig_shape = x.shape
    B = x.size(0)
    x_flat = x.view(B, -1)

    class FlattenSDE(torchsde.SDEStratonovich):
        def __init__(self, net):
            super().__init__(noise_type="diagonal")
            self.net = net

        def f(self, t, y):
            # Drift
            y_unflat = y.view(*orig_shape)
            # Ensure time tensor is on the same device and has a batch dimension
            t_batch = t.expand(B).to(y.device)
            v = self.net(t_batch, y_unflat)
            return v.view(B, -1)

        def g(self, t, y):
            # Diffusion
            e_val = plot_epsilon(float(t))
            if e_val <= 0:
                return torch.zeros_like(y)
            e_tensor = torch.tensor(e_val, device=y.device, dtype=y.dtype)
            scale = torch.sqrt(2.0 * e_tensor)
            return scale.expand_as(y) # Use expand_as for robustness

    sde = FlattenSDE(model)
    ts = torch.arange(t_start, t_end + 1e-9, dt, device=x.device)

    with torch.no_grad():
        # "heun" is the name in older torchsde for the Stratonovich Heun method
        x_sol = torchsde.sdeint(sde, x_flat, ts, method="heun", dt=dt)
        x_final = x_sol[-1].view(*orig_shape).clamp(-1, 1)

    return x_final


##############################################################################
# 3) Main FID computation
##############################################################################
def main(argv):
    # ------------------------------------------------------------
    # A) Initialize Distributed
    # ------------------------------------------------------------
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # ------------------------------------------------------------
    # B) Logging / Output
    # ------------------------------------------------------------
    if rank == 0:
        if not FLAGS.output_dir:
            FLAGS.output_dir = "./sampling_results"
        savedir = create_timestamped_dir(FLAGS.output_dir, FLAGS.model)
        logging.get_absl_handler().use_absl_log_file(program_name="fid_cifar10", log_dir=savedir)
        logging.set_verbosity(logging.INFO)
        logging.info(f"[Rank 0] Saving logs to: {savedir}")
    else:
        savedir = None
        logging.set_verbosity(logging.ERROR)

    # Add a barrier to ensure the output directory is created before proceeding
    dist.barrier()

    # ------------------------------------------------------------
    # 2) Build Model & Load Checkpoint
    # ------------------------------------------------------------
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
        # ViT-specific:
        patch_size=4,
        embed_dim=FLAGS.embed_dim,
        transformer_nheads=FLAGS.transformer_nheads,
        transformer_nlayers=FLAGS.transformer_nlayers,
    ).to(device)
    net_model.eval()

    if not FLAGS.resume_ckpt or not os.path.exists(FLAGS.resume_ckpt):
        raise ValueError(f"--resume_ckpt not found: {FLAGS.resume_ckpt}")

    logging.info(f"Loading checkpoint: {FLAGS.resume_ckpt}")
    ckpt_data = torch.load(FLAGS.resume_ckpt, map_location=device)
    if FLAGS.use_ema:
        net_model.load_state_dict(ckpt_data["ema_model"], strict=True)
        logging.info("Loaded EMA model.")
    else:
        net_model.load_state_dict(ckpt_data["net_model"], strict=True)
        logging.info("Loaded standard model.")
    net_model.eval()

    # ------------------------------------------------------------
    # 3) Real CIFAR => feed into multiple FID accumulators
    # ------------------------------------------------------------
    times_to_sample = [1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25]

    fid_dict = {}
    for t_val in times_to_sample:
        fid_dict[t_val] = FrechetInceptionDistance(feature=2048).to(device)

    if rank == 0:
        logging.info("[Rank 0] Updating FID with real images...")

    train_loader = get_cifar10_train_loader(
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
        rank=rank,
        world_size=world_size,
    )

    for real_imgs, _ in tqdm(train_loader, desc=f"Rank {rank} Real Data", disable=(rank != 0)):
        real_imgs = real_imgs.to(device)  # in [0,1]
        real_uint8 = (real_imgs * 255).clamp(0, 255).to(torch.uint8)
        for t_val in times_to_sample:
            fid_dict[t_val].update(real_uint8, real=True)

    dist.barrier(device_ids=[rank])

    # ------------------------------------------------------------
    # 4) Generate + feed fake images incrementally
    # ------------------------------------------------------------
    total_gen_global = 50000
    total_gen_local = total_gen_global // world_size
    if rank < (total_gen_global % world_size):
        total_gen_local += 1

    if rank == 0:
        logging.info(
            f"[Rank 0] Each rank generates ~{total_gen_local} of {total_gen_global} fakes."
        )

    n_batches = (total_gen_local + FLAGS.batch_size - 1) // FLAGS.batch_size
    num_generated_this_rank = 0

    for _ in tqdm(range(n_batches), desc=f"Rank {rank} Generating", disable=(rank != 0)):
        remaining_to_gen = total_gen_local - num_generated_this_rank
        curr_bsz = min(FLAGS.batch_size, remaining_to_gen)

        if curr_bsz <= 0:
            break

        # Start from standard normal in [B, 3, 32, 32]
        x = torch.randn(curr_bsz, 3, 32, 32, device=device)

        t_prev = 0.0
        for t_end in times_to_sample:
            # Integrate from t_prev..t_end
            x = solve_sde_heun(net_model, x, t_prev, t_end, dt=FLAGS.dt_gibbs)

            # Convert to [0,1] for FID
            x_01 = (x + 1.0) / 2.0
            x_uint8 = (x_01 * 255).clamp(0, 255).to(torch.uint8)

            # Update the corresponding FID
            fid_dict[t_end].update(x_uint8, real=False)

            t_prev = t_end

        num_generated_this_rank += curr_bsz

    dist.barrier(device_ids=[rank])

    # ------------------------------------------------------------
    # 5) Compute + Print final FIDs
    # ------------------------------------------------------------
    if rank == 0:
        logging.info("[Rank 0] Computing final FIDs...")

    for t_val in times_to_sample:
        fid_val = fid_dict[t_val].compute()
        if rank == 0:
            logging.info(f"FID at t={t_val:.2f} => {fid_val:.4f}")

    dist.barrier(device_ids=[rank])
    dist.destroy_process_group()


if __name__ == "__main__":
    app.run(main)