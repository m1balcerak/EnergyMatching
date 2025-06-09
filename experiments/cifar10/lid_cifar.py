#!/usr/bin/env python3
"""
compute_lid_cifar.py

Usage (example):
  python compute_lid_cifar.py \
      --resume_ckpt=/path/to/checkpoint.pt \
      --num_samples=8 \
      --threshold=10.0 \
      --output_dir=lid_cifar_output \
      --num_workers=4 \
      --[other flags from config_multigpu.py]
"""

import os
import io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.autograd.functional import hessian
from torchvision import datasets, transforms
from PIL import Image

# -- absl/config imports --
from absl import app, flags, logging
import config_multigpu as config  # your config file

# Import the EBM model (ViT version)
from network_transformer_vit import EBViTModelWrapper

# joblib for parallelization
from joblib import Parallel, delayed

# Spearman correlation
from scipy.stats import spearmanr

################################################################################
# 1) Register flags from config, plus any new local flags
################################################################################
config.define_flags()
FLAGS = flags.FLAGS

# NOTE: If config_multigpu.py already defines --num_workers, do NOT redefine it here.
# If it does NOT define --num_workers, you can uncomment this line, but ensure
# it doesn't clash with existing definitions.
#
# flags.DEFINE_integer("num_workers", 4, "Number of parallel worker processes to use.")

flags.DEFINE_integer("num_samples", 8, "Number of CIFAR test images to compute Hessians on.")
flags.DEFINE_float("threshold", 10.0, "Absolute value threshold for LID counting.")


################################################################################
# 2) Hessian-related routines (pure python, no absl flags references!)
################################################################################
def compute_hessian_spectrum(model, x):
    """
    Compute Hessian of potential V = model.potential(x, t=1) wrt x, return eigenvals.
    x: shape (1, 3, 32, 32) w/ requires_grad=True.
    """
    def potential_fn(flat_x):
        x_reshaped = flat_x.view_as(x)
        energy = model.potential(x_reshaped, torch.ones(1, device=x.device))
        return energy.squeeze()

    flat_x = x.flatten().requires_grad_(True)

    # Force fallback to standard (non-flash) attention so second derivatives work
    with torch.backends.cuda.sdp_kernel(enable_math=True,
                                        enable_flash=False,
                                        enable_mem_efficient=False):
        H = hessian(potential_fn, flat_x)

    # Symmetrize
    H = 0.5 * (H + H.t())
    # Move to CPU, compute eigenvalues
    evals = torch.linalg.eigvalsh(H.cpu())
    return evals.detach().numpy()


def estimate_intrinsic_dimension(evals, threshold):
    """
    Return # of eigenvalues whose absolute value is below `threshold`.
    """
    return np.sum(np.abs(evals) < threshold)


################################################################################
# 3) PNG size computation (pure python, no absl references)
################################################################################
def get_png_size_from_tensor_cifar(img_tensor):
    """
    Given a single CIFAR image tensor in [-1, 1], convert it to [0,255] range,
    save to an in-memory PNG, and return the PNG size in bytes.

    img_tensor shape: (3, 32, 32)
    """
    img_np = img_tensor.cpu().numpy()  # shape (3, 32, 32)
    img_np = (img_np + 1.0) * 127.5  # scale [-1,1] -> [0,255]
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    img_np = np.transpose(img_np, (1, 2, 0))  # shape (32,32,3)

    pil_img = Image.fromarray(img_np)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    png_size = len(buffer.getvalue())
    return png_size


################################################################################
# 4) Main function
################################################################################
def main(argv):
    """
    We do everything in main so we can avoid referencing FLAGS at the global scope.
    This prevents joblib from trying to pickle absl.flags.FlagValues.
    """
    # --------------------------------------------------------------------------
    # A) Read the flags once, store them in local variables
    # --------------------------------------------------------------------------
    resume_ckpt = FLAGS.resume_ckpt
    output_dir = FLAGS.output_dir
    threshold = FLAGS.threshold
    num_samples = FLAGS.num_samples

    # If config_multigpu.py defines `FLAGS.num_workers`, read from there:
    # If you want a fallback, do:
    if hasattr(FLAGS, 'num_workers'):
        num_workers = FLAGS.num_workers
    else:
        # fallback if not in config
        num_workers = 4

    if not resume_ckpt or not os.path.exists(resume_ckpt):
        raise FileNotFoundError(f"Checkpoint not found or not specified: {resume_ckpt}")
    os.makedirs(output_dir, exist_ok=True)

    # Create a log file in output_dir
    logging.get_absl_handler().use_absl_log_file(
        program_name="lid_cifar",
        log_dir=output_dir
    )
    logging.info("===== LID Computation for CIFAR EBM (Parallel) =====")
    logging.info(f"resume_ckpt = {resume_ckpt}")
    logging.info(f"output_dir  = {output_dir}")
    logging.info(f"num_samples = {num_samples}")
    logging.info(f"threshold   = {threshold}")
    logging.info(f"num_workers = {num_workers}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------------------------
    # B) Build model & load checkpoint
    # --------------------------------------------------------------------------
    ch_mult = config.parse_channel_mult(FLAGS)

    model = EBViTModelWrapper(
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
        patch_size=4,  # as in your script
        embed_dim=FLAGS.embed_dim,
        transformer_nheads=FLAGS.transformer_nheads,
        transformer_nlayers=FLAGS.transformer_nlayers,
    ).to(device)

    ckpt_dict = torch.load(resume_ckpt, map_location=device)
    if "ema_model" in ckpt_dict:
        model.load_state_dict(ckpt_dict["ema_model"])
    else:
        logging.warning("No 'ema_model' in checkpoint, using the entire dict directly.")
        model.load_state_dict(ckpt_dict)
    model.eval()

    # --------------------------------------------------------------------------
    # C) Load CIFAR-10 test set
    # --------------------------------------------------------------------------
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    cifar_test = datasets.CIFAR10(
        root="./data",
        train=True,
        download=not os.path.exists("./data/cifar-10-batches-py"),
        transform=transform_test
    )
    N = min(num_samples, len(cifar_test))
    logging.info(f"Computing Hessians on {N} test samples...")

    # --------------------------------------------------------------------------
    # D) Define a local worker function inside main (no global FLAGS references!)
    # --------------------------------------------------------------------------
    def process_one_image(i):
        """
        This function runs in a separate process via joblib.
        It returns a dictionary of computed results for image i.
        """
        # 1) Load the i-th image, shape (3,32,32)
        img, _ = cifar_test[i]

        # 2) Convert to batch dimension & put on device
        x = img.unsqueeze(0).to(device).requires_grad_(True)

        # 3) Hessian -> eigenvalues -> LID
        evals = compute_hessian_spectrum(model, x)
        lid = estimate_intrinsic_dimension(evals, threshold)

        # 4) PNG size (use original CPU tensor)
        png_size = get_png_size_from_tensor_cifar(img)

        return {
            'index': i,
            'img': img,    # store the CPU tensor for plotting extremes
            'evals': evals,
            'lid': lid,
            'png_size': png_size,
        }

    # --------------------------------------------------------------------------
    # E) Parallelize the computation with joblib
    # --------------------------------------------------------------------------
    from tqdm import tqdm
    indices = list(range(N))

    # We can show a progress bar by manually collecting results or with "verbose=10"
    # in Parallel. One approach is to do:
    results = []
    with tqdm(total=N, desc="Parallel Hessians") as pbar:
        # define a callback to increment pbar each time a job finishes
        def update(*_):
            pbar.update()

        # run joblib
        parallel_pool = Parallel(n_jobs=num_workers, prefer="processes")
        # "backend='loky'" is the default. You can also do backend='multiprocessing'.
        # We pass a generator of delayed calls to parallel_pool.
        tasks = (delayed(process_one_image)(i) for i in indices)
        # The trick to have a progress callback is a bit more elaborate if we want
        # a fully asynchronous approach. For simplicity, let's just collect them
        # in a list:
        results = parallel_pool(tasks)

    # Sort results by index just to be sure
    results.sort(key=lambda r: r['index'])

    # --------------------------------------------------------------------------
    # F) Post-processing: find min/max LID, correlation, etc.
    # --------------------------------------------------------------------------
    lids = [r['lid'] for r in results]
    png_sizes = [r['png_size'] for r in results]

    min_index = int(np.argmin(lids))
    max_index = int(np.argmax(lids))
    extremes = [results[min_index], results[max_index]]
    labels = ['lowest_lid', 'highest_lid']

    logging.info(f"Min LID => index={min_index}, LID={lids[min_index]}")
    logging.info(f"Max LID => index={max_index}, LID={lids[max_index]}")

    # Plot extremes
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for row, item in enumerate(extremes):
        img_cpu  = item['img']   # shape (3,32,32)
        evals    = item['evals']
        lid_val  = item['lid']
        png_size = item['png_size']

        # (a) Show image
        ax_img = axes[row, 0]
        img_np = img_cpu.permute(1,2,0).numpy()  # shape (32,32,3)
        img_disp = 0.5*(img_np + 1.0)            # from [-1,1] to [0,1]
        img_disp = np.clip(img_disp, 0, 1)
        ax_img.imshow(img_disp)
        ax_img.axis('off')
        ax_img.set_title(f"{labels[row]}: LID={lid_val}\nPNG size={png_size} bytes")

        # (b) Hessian spectrum
        ax_spec = axes[row, 1]
        evals_sorted = np.sort(evals)[::-1]
        ax_spec.plot(evals_sorted, 'o-')
        ax_spec.set_title('Hessian Eigenvalues')
        ax_spec.grid(True)

    fig.tight_layout()
    out_fig = os.path.join(output_dir, 'cifar_lid_extremes.png')
    plt.savefig(out_fig)
    plt.show()

    # Spearman correlation
    lid_arr = np.array(lids)
    png_arr = np.array(png_sizes)
    corr, pval = spearmanr(lid_arr, png_arr)

    logging.info('===== Spearman Correlation between LID & PNG size =====')
    logging.info(f'  corr = {corr:.4f}, p-value = {pval:.4e}')

    # (Optional) scatter
    plt.figure(figsize=(6, 5))
    plt.scatter(png_arr, lid_arr, alpha=0.7)
    plt.xlabel('PNG Size (bytes)')
    plt.ylabel('LID')
    plt.title(f'Spearman r={corr:.3f}, p={pval:.1e}')
    out_fig_scatter = os.path.join(output_dir, 'lid_vs_pngsize.png')
    plt.savefig(out_fig_scatter)
    plt.show()

    # Save extremes as CSV / images
    for label, item in zip(labels, extremes):
        img_cpu = item['img']
        evals   = item['evals']
        lid_val = item['lid']
        png_sz  = item['png_size']

        evals_sorted = np.sort(evals)[::-1]
        csv_name = os.path.join(output_dir, f"{label}_hessian.csv")
        np.savetxt(
            csv_name,
            np.column_stack((np.arange(len(evals_sorted)), evals_sorted)),
            header='index,eigenvalue', delimiter=',', comments=''
        )

        png_name = os.path.join(output_dir, f"{label}_image.png")
        img_np = img_cpu.permute(1,2,0).numpy()
        img_disp = 0.5*(img_np + 1.0)
        img_disp = np.clip(img_disp, 0, 1)
        plt.imsave(png_name, img_disp)

    logging.info('===== Final Results =====')
    logging.info(f"Min LID: index={min_index}, LID={lids[min_index]}")
    logging.info(f"Max LID: index={max_index}, LID={lids[max_index]}")
    logging.info(f"Plot of extremes saved to {out_fig}")
    logging.info(f"Scatter of LID vs PNG size saved to {out_fig_scatter}")
    logging.info('Done.')


if __name__ == '__main__':
    app.run(main)
