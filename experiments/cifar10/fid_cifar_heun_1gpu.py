#######################################################################
# File: fid_cifar_heun_1gpu.py
#
# Description:
#   Computes the Frechet Inception Distance (FID) for a trained
#   energy-based model on CIFAR-10 using a single GPU. It generates
#   samples via an SDE solver (Heun's method) and compares them
#   against the real training data.
#
# Usage example:
#   python fid_cifar_heun_1gpu.py \
#       --resume_ckpt=/path/to/checkpoint.pt \
#       --batch_size=128 \
#       --dt_gibbs=0.01 \
#       --use_ema=True
#######################################################################

import os
import sys
import torch

# absl flags
from absl import app, flags, logging

# Single-GPU config
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
# 1) CIFAR-10 Data (single GPU)
##############################################################################

def get_cifar10_train_loader(batch_size, num_workers, root=None):
    """Returns a standard DataLoader for CIFAR-10 train set."""
    if root is None:
        root = os.environ.get("CIFAR10_PATH", "./data")

    transform = T.ToTensor()
    dataset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,  # Keep order consistent for evaluation
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
    # A) Initialize Device and Logging
    # ------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not FLAGS.output_dir:
        FLAGS.output_dir = "./sampling_results"
    savedir = create_timestamped_dir(FLAGS.output_dir, FLAGS.model)
    logging.get_absl_handler().use_absl_log_file(program_name="fid_cifar10", log_dir=savedir)
    logging.set_verbosity(logging.INFO)
    logging.info(f"Saving logs to: {savedir}")
    logging.info(f"Using device: {device}")

    # ------------------------------------------------------------
    # B) Build Model & Load Checkpoint
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
    # C) Process Real CIFAR Data for FID
    # ------------------------------------------------------------
    times_to_sample = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]

    # Create a separate FID calculator for each sampling time
    fid_dict = {t_val: FrechetInceptionDistance(feature=2048).to(device) for t_val in times_to_sample}

    logging.info("Updating FID with real images...")
    train_loader = get_cifar10_train_loader(
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
    )

    for real_imgs, _ in tqdm(train_loader, desc="Processing Real Data"):
        real_imgs = real_imgs.to(device)  # in [0,1]
        real_uint8 = (real_imgs * 255).clamp(0, 255).to(torch.uint8)
        # Update all FID calculators with the same real data
        for t_val in times_to_sample:
            fid_dict[t_val].update(real_uint8, real=True)

    # ------------------------------------------------------------
    # D) Generate and Process Fake Images
    # ------------------------------------------------------------
    total_samples_to_gen = 50000
    logging.info(f"Generating {total_samples_to_gen} fake samples for FID...")

    n_batches = (total_samples_to_gen + FLAGS.batch_size - 1) // FLAGS.batch_size
    num_generated = 0

    for _ in tqdm(range(n_batches), desc="Generating Fake Data"):
        remaining_to_gen = total_samples_to_gen - num_generated
        curr_bsz = min(FLAGS.batch_size, remaining_to_gen)

        if curr_bsz <= 0:
            break

        # Start from standard normal in [B, 3, 32, 32]
        x = torch.randn(curr_bsz, 3, 32, 32, device=device)

        t_prev = 0.0
        # Sequentially generate samples for each time point
        for t_end in times_to_sample:
            # Integrate from t_prev to t_end
            x = solve_sde_heun(net_model, x, t_prev, t_end, dt=FLAGS.dt_gibbs)

            # Convert to [0,1] range uint8 for FID
            x_01 = (x + 1.0) / 2.0
            x_uint8 = (x_01 * 255).clamp(0, 255).to(torch.uint8)

            # Update the corresponding FID metric with fake images
            fid_dict[t_end].update(x_uint8, real=False)

            # The end time of this step becomes the start time for the next
            t_prev = t_end

        num_generated += curr_bsz

    # ------------------------------------------------------------
    # E) Compute and Print Final FIDs
    # ------------------------------------------------------------
    logging.info("Computing final FID scores...")
    logging.info(f"Comparison is based on {len(train_loader.dataset)} real vs {num_generated} fake samples.")

    for t_val in times_to_sample:
        fid_val = fid_dict[t_val].compute()
        # Also log the sample counts to verify
        real_count = fid_dict[t_val].real_features_num_samples
        fake_count = fid_dict[t_val].fake_features_num_samples
        logging.info(f"FID at t={t_val:.2f} => {fid_val:.4f} (real: {real_count}, fake: {fake_count})")


if __name__ == "__main__":
    app.run(main)
