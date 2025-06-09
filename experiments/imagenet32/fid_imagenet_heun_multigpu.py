#######################################################################
# File: fid_imagenet_heun_multigpu.py
#
# Usage:
#   torchrun --nproc_per_node=2 fid_imagenet_heun_multigpu.py \
#       --resume_ckpt=./path/to/checkpoint.pt \
#       --batch_size=64 \
#       --dt_gibbs=0.01 \
#       --use_ema=True
#######################################################################

import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
from absl import app, flags, logging

import config_multigpu_imagenet32 as config
config.define_flags()
FLAGS = flags.FLAGS

flags.DEFINE_bool("use_ema", True, "If True, load the EMA model from the checkpoint.")

from torchmetrics.image.fid import FrechetInceptionDistance
from network_transformer_vit import EBViTModelWrapper
from dataset_imagenet32 import ImageNet32Dataset
from utils_cifar_imagenet import (
    create_timestamped_dir,
    plot_epsilon
)

import torchsde
from tqdm import tqdm


def get_imagenet32_train_loader(batch_size, num_workers, rank, world_size):
    """
    Returns a DataLoader for ImageNet32 (train), distributed among GPUs.
    Images in [-1,1].
    """
    dataset = ImageNet32Dataset(
        split="train",
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        drop_last=False
    )
    return loader


def solve_sde_heun(model, x, t_start, t_end, dt=0.01):
    """Integrates x from t_start..t_end using Stratonovich Euler–Heun."""
    if t_end <= t_start:
        return x

    orig_shape = x.shape
    B = x.size(0)
    x_flat = x.view(B, -1)

    class FlattenSDE(torchsde.SDEStratonovich):
        def __init__(self, net):
            super().__init__(noise_type="diagonal")
            self.net = net

        def f(self, t, y):
            y_unflat = y.view(*orig_shape)
            v = self.net(t.unsqueeze(0), y_unflat)
            return v.view(B, -1)

        def g(self, t, y):
            e_val = plot_epsilon(float(t))
            if e_val <= 0:
                return torch.zeros_like(y)
            e_tensor = torch.tensor(e_val, device=y.device, dtype=y.dtype)
            scale = torch.sqrt(2.0 * e_tensor)
            return scale * torch.ones_like(y)

    sde = FlattenSDE(model)
    ts = torch.arange(t_start, t_end + 1e-9, dt, device=x.device)

    with torch.no_grad():
        x_sol = torchsde.sdeint(sde, x_flat, ts, method="heun", dt=dt)
        x_final = x_sol[-1].view(*orig_shape).clamp(-1, 1)

    return x_final


def main(argv):
    # -------------------------------------------------------------------
    # A) Initialize Distributed
    # -------------------------------------------------------------------
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # -------------------------------------------------------------------
    # B) Logging / Output
    # -------------------------------------------------------------------
    if rank == 0:
        if not FLAGS.output_dir:
            FLAGS.output_dir = "./sampling_results"
        savedir = create_timestamped_dir(FLAGS.output_dir, FLAGS.model)
        logging.get_absl_handler().use_absl_log_file(program_name="fid_imagenet32", log_dir=savedir)
        logging.set_verbosity(logging.INFO)
        logging.info(f"[Rank 0] Saving logs to: {savedir}")
    else:
        savedir = None
        logging.set_verbosity(logging.ERROR)

    dist.barrier()

    # -------------------------------------------------------------------
    # C) Build Model & Load Checkpoint
    # -------------------------------------------------------------------
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
        raise ValueError(f"[Rank {rank}] --resume_ckpt not found: {FLAGS.resume_ckpt}")

    if rank == 0:
        logging.info(f"[Rank 0] Loading checkpoint: {FLAGS.resume_ckpt}")
    ckpt_data = torch.load(FLAGS.resume_ckpt, map_location=device)
    if FLAGS.use_ema:
        net_model.load_state_dict(ckpt_data["ema_model"], strict=True)
    else:
        net_model.load_state_dict(ckpt_data["net_model"], strict=True)
    net_model.eval()

    dist.barrier()

    # -------------------------------------------------------------------
    # D) Prepare DataLoader for Real Data
    # -------------------------------------------------------------------
    train_loader = get_imagenet32_train_loader(
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
        rank=rank,
        world_size=world_size
    )

    # -------------------------------------------------------------------
    # E) Build multiple FID objects (one per time),
    #    using TorchMetrics distributed sync
    # -------------------------------------------------------------------
    times_to_sample = [0.75, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0]
    fid_dict = {}
    for t_val in times_to_sample:
        fid_dict[t_val] = FrechetInceptionDistance(feature=2048).to(device)

    # -------------------------------------------------------------------
    # F) Add real images
    # -------------------------------------------------------------------
    if rank == 0:
        logging.info(f"[Rank 0] Updating FID with real images...")

    for real_imgs, _ in tqdm(train_loader, desc=f"Rank {rank} Real Data", disable=(rank != 0)):
        real_imgs = real_imgs.to(device)
        real_01 = (real_imgs + 1.0) / 2.0
        real_uint8 = (real_01 * 255).clamp(0, 255).byte()

        for t_val in times_to_sample:
            fid_dict[t_val].update(real_uint8, real=True)

    dist.barrier()

    # -------------------------------------------------------------------
    # G) Generate partial set of fake images on each rank
    # -------------------------------------------------------------------
    total_gen_global = 50000
    total_gen_local = total_gen_global // world_size
    if rank < (total_gen_global % world_size):
        total_gen_local += 1

    if rank == 0:
        logging.info(f"[Rank 0] Each rank generates ~{total_gen_local} of {total_gen_global} fakes.")

    n_batches = (total_gen_local + FLAGS.batch_size - 1) // FLAGS.batch_size

    for _ in tqdm(range(n_batches), desc=f"Rank {rank} Generating", disable=(rank != 0)):
        curr_bsz = min(FLAGS.batch_size, total_gen_local)
        total_gen_local -= curr_bsz
        if curr_bsz <= 0:
            break

        x = torch.randn(curr_bsz, 3, 32, 32, device=device)

        t_prev = 0.0
        for t_val in times_to_sample:
            x = solve_sde_heun(net_model, x, t_prev, t_val, dt=FLAGS.dt_gibbs)
            x_01 = (x + 1.0) / 2.0
            x_uint8 = (x_01 * 255).clamp(0, 255).byte()
            fid_dict[t_val].update(x_uint8, real=False)

            t_prev = t_val

    dist.barrier()

    # -------------------------------------------------------------------
    # H) Compute final FIDs (torchmetrics merges across ranks)
    # -------------------------------------------------------------------
    if rank == 0:
        logging.info("[Rank 0] Computing final FIDs...")

    for t_val in times_to_sample:
        fid_val = fid_dict[t_val].compute()  # merges across ranks internally
        if rank == 0:
            logging.info(f"FID at t={t_val:.2f} => {fid_val:.4f}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    app.run(main)
