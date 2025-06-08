#!/usr/bin/env python3

import os
import io
from pathlib import Path
import numpy as np
import torch
from torch.autograd.functional import hessian
from torchvision import datasets, transforms
from PIL import Image
from scipy.stats import spearmanr
from joblib import Parallel, delayed

from tqdm import tqdm
import matplotlib.pyplot as plt

# Your MNIST model
from model_mnist import Network_2M_MNIST28x28  # Adjust to match your filename and class name


################################################################################
# 1) Hessian-related routines
################################################################################
def compute_hessian_spectrum(model, x):
    """
    Compute the Hessian of the scalar potential V = model(x) w.r.t. x
    and return its eigenvalues.
    """
    def potential_fn(flat_x):
        return model(flat_x.view_as(x)).squeeze()

    flat_x = x.flatten().requires_grad_(True)
    # For numerical stability, symmetrize H
    H = hessian(potential_fn, flat_x)
    H = 0.5 * (H + H.t())
    evals = torch.linalg.eigvalsh(H.cpu())
    return evals.detach().numpy()


def estimate_intrinsic_dimension(evals, threshold):
    """
    Return the count of eigenvalues whose absolute value is less than `threshold`.
    """
    return np.sum(np.abs(evals) < threshold)


################################################################################
# 2) PNG compression size
################################################################################
def get_png_size_from_tensor(img_tensor):
    """
    Given a single MNIST image tensor in [-1, 1], convert it to [0,255] range,
    save to an in-memory PNG, and return the PNG size in bytes.
    """
    img_np = img_tensor.squeeze().cpu().numpy()  # shape: (28, 28)
    img_np = (img_np + 1) * 127.5  # scale [-1,1] -> [0,255]
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(img_np)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    png_size = len(buffer.getvalue())
    return png_size


################################################################################
# 3) Utility: computing correlations for a set of images
################################################################################
def compute_correlations_for_thresholds(results, thresholds):
    """
    Given a list of dicts (index, evals, png_size) and a list of thresholds,
    returns the Spearman correlation for each threshold.
    """
    png_sizes = np.array([r["png_size"] for r in results])
    all_corrs = []
    for thr in thresholds:
        lids_arr = np.array([estimate_intrinsic_dimension(r["evals"], thr) for r in results])
        corr, _ = spearmanr(lids_arr, png_sizes)
        all_corrs.append(corr)
    return np.array(all_corrs)


################################################################################
# 4) Main script
################################################################################
def main():
    # Arguments / settings
    model_path = Path(__file__).resolve().parent / "assets" / "mnist_model.pth"
    thresholds = np.linspace(0.1, 40, 50)
    num_samples_select = 0
    num_samples_test = 4096

    num_workers = 4  # joblib concurrency
    chunk_size = 16  # how many images to process per chunk in parallel

    # Reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Check model file
    if not model_path.exists():
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = Network_2M_MNIST28x28().to(device)
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.eval()

    # MNIST transform: [0,1] -> [-1,1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),
    ])

    # MNIST train set
    dataset = datasets.MNIST(
        root="./data",
        train=True,
        transform=transform,
        download=True
    )

    # --------------------------------------------------------------------------
    # Pick 1024 + 4096 distinct random images from MNIST train, with a fixed seed
    # --------------------------------------------------------------------------
    total_needed = num_samples_select + num_samples_test
    if total_needed > len(dataset):
        raise ValueError("Requested more samples than exist in the MNIST train set.")

    all_indices = np.random.choice(len(dataset), size=total_needed, replace=False)
    select_indices = all_indices[:num_samples_select]
    test_indices = all_indices[num_samples_select:]

    # ----------------------------------------------------------------------------
    # Worker function for one image
    # ----------------------------------------------------------------------------
    def process_one_image(idx):
        """
        - Loads the idx-th MNIST image
        - Computes Hessian eigenvalues
        - Computes PNG size
        Returns a dict of the results
        """
        img_tensor, _ = dataset[idx]
        x = img_tensor.unsqueeze(0).to(device).requires_grad_(True)

        # Hessian spectrum
        evals = compute_hessian_spectrum(model, x)

        # PNG compression size
        png_size = get_png_size_from_tensor(img_tensor)

        return {
            "index": idx,
            "evals": evals,
            "png_size": png_size
        }

    # ----------------------------------------------------------------------------
    # Helper to process a given list of indices in parallel
    # ----------------------------------------------------------------------------
    def process_indices_in_parallel(indices):
        results_local = []
        with tqdm(total=len(indices), desc="Parallel Hessians") as pbar:
            for start_idx in range(0, len(indices), chunk_size):
                chunk = indices[start_idx:start_idx + chunk_size]
                chunk_results = Parallel(n_jobs=num_workers)(
                    delayed(process_one_image)(i) for i in chunk
                )
                results_local.extend(chunk_results)
                pbar.update(len(chunk))
        # Sort results by index for consistency
        results_local.sort(key=lambda r: r["index"])
        return results_local

    # 1) Process the 1024 "select" indices
    print("Computing Hessians/PNG sizes for the 1024 threshold-selection images ...")
    results_select = process_indices_in_parallel(select_indices)

    # 2) Process the 4096 "test" indices
    print("\nComputing Hessians/PNG sizes for the 4096 test images ...")
    results_test = process_indices_in_parallel(test_indices)

    # ----------------------------------------------------------------------------
    # STEP A: Sweep thresholds on the 1024 images to find the best threshold
    # ----------------------------------------------------------------------------
    print("\nSweeping thresholds on the 1024 selection set ...")
    corrs_select = compute_correlations_for_thresholds(results_select, thresholds)

    # Find the threshold that maximizes |correlation|
    best_idx = np.argmax(np.abs(corrs_select))
    best_thr = thresholds[best_idx]
    best_corr = corrs_select[best_idx]

    print("=====================================================")
    print("Threshold sweep results (selection set of 1024):")
    for thr_val, c_val in zip(thresholds, corrs_select):
        print(f"  threshold={thr_val:.3f}, correlation={c_val:.4f}")
    print("=====================================================")
    print(f"Best threshold (by |corr|) on 1024 = {best_thr:.3f}, corr={best_corr:.4f}")
    print("=====================================================")

    # Optional: Plot correlation vs threshold for the 1024 set
    # (If you only need the 4096 set’s plot, skip this block or comment it out.)
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, corrs_select, 'o-', label="Spearman correlation (1024 images)")
    plt.scatter([best_thr], [best_corr], color='red', zorder=5, label="best threshold")
    plt.title("Spearman correlation vs. threshold (1024 selection images)")
    plt.xlabel("Threshold")
    plt.ylabel("Spearman correlation")
    plt.grid(True)
    plt.legend()
    plt.savefig("spearman_corr_vs_threshold_1024.png", dpi=120, bbox_inches='tight')
    plt.close()
    print("Saved plot 'spearman_corr_vs_threshold_1024.png'")

    # ----------------------------------------------------------------------------
    # STEP B: On the 4096 images, we:
    #  - compute correlation vs threshold (and plot)
    #  - report correlation at best_thr from the 1024 set
    # ----------------------------------------------------------------------------
    print("\nSweeping thresholds on the 4096 test set ...")
    corrs_test = compute_correlations_for_thresholds(results_test, thresholds)

    # Correlation if we use the “best_thr” found on the 1024 images
    # (We do NOT re-pick the best threshold here.)
    lids_test_best = np.array([
        estimate_intrinsic_dimension(r["evals"], best_thr) for r in results_test
    ])
    png_test = np.array([r["png_size"] for r in results_test])
    final_corr, _ = spearmanr(lids_test_best, png_test)

    # Print summary
    print("=====================================================")
    print("Threshold sweep results (test set of 4096):")
    for thr_val, c_val in zip(thresholds, corrs_test):
        print(f"  threshold={thr_val:.3f}, correlation={c_val:.4f}")
    print("=====================================================")
    print("Note: We are NOT picking a new best threshold here, just reporting.")
    print(f"Correlation at the previously chosen best_thr={best_thr:.3f} is {final_corr:.4f}")
    print("=====================================================")

    # Plot correlation vs threshold for the 4096 test set
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, corrs_test, 'o-', label="Spearman correlation (4096 images)")
    # highlight correlation for best_thr
    # find the correlation in corrs_test at that threshold index
    test_best_idx = np.argwhere(thresholds == best_thr)
    # In case best_thr is not exactly one of the 50 linearly spaced thresholds
    # (floating precision issues), we can skip a direct dot. Otherwise, we might
    # do a small trick to find the nearest threshold in the array:
    nearest_idx = np.argmin(np.abs(thresholds - best_thr))
    plt.scatter([thresholds[nearest_idx]], [corrs_test[nearest_idx]], color='red', zorder=5,
                label="correlation at best_thr from the 1024-set")
    plt.title("Spearman correlation vs. threshold (4096 test images)")
    plt.xlabel("Threshold")
    plt.ylabel("Spearman correlation")
    plt.grid(True)
    plt.legend()

    plot_filename = "spearman_corr_vs_threshold_4096.png"
    plt.savefig(plot_filename, dpi=120, bbox_inches='tight')
    plt.close()

    print(f"Saved plot '{plot_filename}'")
    print("\nAll done.")


if __name__ == "__main__":
    main()
