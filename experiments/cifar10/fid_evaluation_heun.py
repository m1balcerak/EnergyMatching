#######################################################################
# File: fid_evaluation.py
#
# Usage example:
#   python fid_evaluation.py \
#       --resume_ckpt=/path/to/checkpoint.pt \
#       --batch_size=64 \
#       --dt_gibbs=0.01 \
#       --use_ema=True \
#       --output_dir=./fid_results
#######################################################################

import os
import sys
import torch

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
# 1) CIFAR-10 Data
##############################################################################
def get_cifar10_test_loader(root="./data", batch_size=64, num_workers=4):
    """
    Returns a DataLoader for the entire CIFAR-10 set
    with images in [0,1] float range.
    """
    transform = T.ToTensor()
    dataset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
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
            # ---- The important fix: pass the noise_type here. ----
            super().__init__(noise_type="diagonal")
            # If your torchsde requires sde_type as well, do:
            # super().__init__(sde_type="stratonovich", noise_type="diagonal")
            self.net = net

        def f(self, t, y):
            # Drift
            y_unflat = y.view(*orig_shape)
            v = self.net(t.unsqueeze(0), y_unflat)
            return v.view(B, -1)

        def g(self, t, y):
            # Diffusion
            e_val = plot_epsilon(float(t))
            if e_val <= 0:
                return torch.zeros_like(y)
            e_tensor = torch.tensor(e_val, device=y.device, dtype=y.dtype)
            scale = torch.sqrt(2.0 * e_tensor)
            return scale * torch.ones_like(y)

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
    # 1) Setup + Logging
    # ------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not FLAGS.output_dir:
        FLAGS.output_dir = "./fid_results"
    savedir = create_timestamped_dir(FLAGS.output_dir, FLAGS.model)
    logging.get_absl_handler().use_absl_log_file(program_name="fid_eval", log_dir=savedir)
    logging.set_verbosity(logging.INFO)
    logging.info(f"Saving results to: {savedir}")

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
    # Times at which to evaluate FID
    times_to_sample = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.50, 4.75, 5.0, 5.25, 5.5]
    times_to_sample = sorted(times_to_sample)

    # Create a separate FID object for each time
    fid_dict = {}
    for t_val in times_to_sample:
        fid_dict[t_val] = FrechetInceptionDistance(feature=2048).to(device)

    # Add real images to each FID aggregator
    logging.info("Feeding real CIFAR images to all FID accumulators...")
    test_loader = get_cifar10_test_loader(root="./data", batch_size=64)
    for real_imgs, _ in tqdm(test_loader, leave=False):
        real_imgs = real_imgs.to(device)  # in [0,1]
        real_uint8 = (real_imgs * 255).clamp(0, 255).to(torch.uint8)
        for t_val in times_to_sample:
            fid_dict[t_val].update(real_uint8, real=True)

    # ------------------------------------------------------------
    # 4) Generate + feed fake images incrementally
    # ------------------------------------------------------------
    total_gen = 50000  # or 50k if you want
    gen_batch_size = FLAGS.batch_size
    n_batches = (total_gen + gen_batch_size - 1) // gen_batch_size

    logging.info(f"Generating {total_gen} samples, Euler–Heun, dt={FLAGS.dt_gibbs}...")

    with torch.no_grad():
        for _ in tqdm(range(n_batches), desc="Generating samples", leave=False):
            curr_bsz = min(gen_batch_size, total_gen)
            total_gen -= curr_bsz
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

            if total_gen <= 0:
                break

    # ------------------------------------------------------------
    # 5) Compute + Print final FIDs
    # ------------------------------------------------------------
    logging.info("Computing FID for each time...")
    for t_val in times_to_sample:
        fid_val = fid_dict[t_val].compute()
        logging.info(f"FID at t={t_val:.2f} => {fid_val:.4f}")

    logging.info("FID evaluation complete!")


if __name__ == "__main__":
    app.run(main)
