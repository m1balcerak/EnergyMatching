import torch
import os
import matplotlib.pyplot as plt
import numpy as np


def linear_noise_schedule(t):
    """
    Identity schedule: returns t as is.
    t: [B,1,1,1]
    returns: [B,1,1,1]
    """
    return t


def add_noise(x, t):
    """
    Adds Gaussian noise to x based on t and clamps the result to [-1,1].

    Args:
        x: [B, C, H, W], e.g. [B,1,28,28] for MNIST. Values in [-1,1].
        t: [B,1,1,1] scalar in [0,1].

    Procedure:
      noise = Gaussian(0,1), shape [B,C,H,W]
      noisy_data = (1 - t)*noise + t*x
      clamp to [-1,1].
    Returns:
        noisy_data: [B,C,H,W], requires_grad=True
        noise: [B,C,H,W] the actual noise added
    """
    noise = torch.randn_like(x)
    noisy_data = (1 - t) * noise + t * x
    noisy_data = torch.clamp(noisy_data, -1.0, 1.0)
    noisy_data = noisy_data.requires_grad_(True)
    return noisy_data, noise


def compute_velocity(x, noise, t):
    """
    Target velocity = (x - noise).

    x: [B,C,H,W]
    noise: [B,C,H,W]
    t: [B,1,1,1] (currently unused)
    """
    return x - noise


def gibbs_sampling_n_steps_fast(x, model, config, n_steps):
    """
    Inlined version of the n-step Gibbs sampler. 
    Does the same thing as calling gibbs_sampling_step in a loop, 
    but with less Python overhead. 
    """
    # Make sure we're on the right device and in a proper state
    samples = x.clone().detach().to(config.device)
    
    dt_eff = config.dt * config.k
    noise_std = (2.0 * dt_eff * config.epsilon) ** 0.5
    
    for _ in range(n_steps):
        # Enable gradient for current samples
        samples.requires_grad_(True)
        
        # Forward pass
        V = model(samples)  # shape [B, 1]
        
        # Gradient wrt samples
        grad_V = torch.autograd.grad(
            outputs=V,
            inputs=samples,
            grad_outputs=torch.ones_like(V),
            create_graph=False,
            retain_graph=False
        )[0]
        
        # Update samples with no_grad
        with torch.no_grad():
            samples = samples - dt_eff * grad_V
            samples += noise_std * torch.randn_like(samples)
            samples = torch.clamp(samples, -1.0, 1.0)
            
    return samples.detach()


def gibbs_sampling_step(samples, model, config):
    """
    One Gibbs (Langevin-like) step:
      x_new = x - (dt*k)*grad_V(x) + sqrt(2*dt*k*epsilon)*N(0,I)
    Where dt=1.0, scaled by k. epsilon is the entropy regularization.
    """
    # Fresh copy with grad
    samples = samples.clone().detach().requires_grad_(True).to(config.device)

    # Forward pass
    V = model(samples)  # shape [B,1]
    grad_V = torch.autograd.grad(
        outputs=V,
        inputs=samples,
        grad_outputs=torch.ones_like(V),
        create_graph=False
    )[0]  # [B,C,H,W]

    dt_eff = config.dt * config.k
    noise_std = (2.0 * dt_eff * config.epsilon) ** 0.5

    with torch.no_grad():
        samples = samples - dt_eff * grad_V
        samples += noise_std * torch.randn_like(samples)
        samples = torch.clamp(samples, -1.0, 1.0)

    return samples.detach()


def gibbs_sampling_n_steps(x, model, config, n_steps):
    """
    Repeats gibbs_sampling_step n times.
    """
    samples = x.clone()
    for _ in range(n_steps):
        samples = gibbs_sampling_step(samples, model, config)
    return samples


def save_diagnostics_figure(model, data, config, epoch, results_dir, n_steps_list=[1, 10, 100]):
    """
    Creates a side-by-side figure:
      - Real data + potential
      - Negative samples from 1-step, 10-steps, 100-steps Gibbs + potential
    Saves as "diagnostics_epoch_{epoch}.png" in the same _results directory.
    """
    model.eval()
    B = data.shape[0]

    # Potential of real data
    # (We do NOT wrap in torch.no_grad because we only do a forward pass,
    #  which does not create a large graph anyway. It's safe to do no_grad
    #  here if you like, but it's not strictly necessary.)
    V_real = model(data).view(-1)

    # Negative samples & potentials
    neg_samples_list = []
    neg_potentials_list = []

    # Note: No "no_grad()" around gibbs_sampling_n_steps; it needs local grad
    for n_steps in n_steps_list:
        x_neg = gibbs_sampling_n_steps(data.clone(), model, config, n_steps)
        V_neg = model(x_neg).view(-1)

        neg_samples_list.append(x_neg)
        neg_potentials_list.append(V_neg)

    # Plot
    n_cols = 1 + len(n_steps_list)  # real + each negative
    fig, axes = plt.subplots(B, n_cols, figsize=(3*n_cols, 3*B))

    for row in range(B):
        # Real
        ax_real = axes[row, 0] if B > 1 else axes[0]
        img_real = data[row].cpu().numpy()  # [C,H,W]
        if config.num_channels == 1:
            ax_real.imshow(img_real.squeeze(), cmap='gray', vmin=-1, vmax=1)
        else:
            ax_real.imshow(np.transpose(img_real, (1,2,0)), vmin=-1, vmax=1)
        ax_real.set_title(f"Real (V={V_real[row].item():.2f})", fontsize=8)
        ax_real.axis('off')

        # Each negative sample
        for c_idx, n_steps_val in enumerate(n_steps_list, start=1):
            ax_neg = axes[row, c_idx] if B > 1 else axes[c_idx]
            img_neg = neg_samples_list[c_idx - 1][row].cpu().numpy()
            V_n = neg_potentials_list[c_idx - 1][row].item()

            if config.num_channels == 1:
                ax_neg.imshow(img_neg.squeeze(), cmap='gray', vmin=-1, vmax=1)
            else:
                ax_neg.imshow(np.transpose(img_neg, (1,2,0)), vmin=-1, vmax=1)
            ax_neg.set_title(f"{n_steps_val} steps (V={V_n:.2f})", fontsize=8)
            ax_neg.axis('off')

    fig.suptitle(f"Diagnostics at Epoch {epoch}", fontsize=14)
    plt.tight_layout()

    # Save figure
    os.makedirs(results_dir, exist_ok=True)
    out_filename = os.path.join(results_dir, f"diagnostics_epoch_{epoch:03d}.png")
    plt.savefig(out_filename)
    plt.close(fig)
    print(f"[Diagnostics] Saved to {out_filename}")


def compute_velocity_from_potential(model, x):
    """
    velocity = -grad(V). For ODE integration approach.
    """
    V = model(x)  # [B,1]
    grad_V = torch.autograd.grad(
        outputs=V,
        inputs=x,
        grad_outputs=torch.ones_like(V),
        create_graph=False,
        retain_graph=False
    )[0]  # [B,C,H,W]
    return -grad_V


def ode_integrate(model, x_T, config, capture_steps):
    """
    Integrates from t=1 to t=0 by repeatedly applying velocity from potential.
    """
    x = x_T.clone().detach().requires_grad_(True).to(config.device)
    captured_x = []
    captured_V = []

    for step in range(1, config.integration_steps + 1):
        velocity = compute_velocity_from_potential(model, x)
        x = x + velocity * config.dt
        x = torch.clamp(x, -1.0, 1.0)

        V = model(x)
        if step in capture_steps:
            captured_x.append(x.detach().cpu())
            captured_V.append(V.detach().cpu())

        x = x.detach().requires_grad_(True)

    return captured_x, captured_V


def generate_and_save_samples(model, config, epoch, results_dir):
    """
    ODE integration from random noise. Saves figure in _results directory.
    """
    model.eval()
    os.makedirs(results_dir, exist_ok=True)

    capture_steps = np.linspace(1, config.integration_steps, config.timepoints, dtype=int).tolist()
    fig, axes = plt.subplots(config.num_samples, config.timepoints,
                             figsize=(config.timepoints * 2, config.num_samples * 2))
    fig.suptitle(f'Samples at Epoch {epoch}', fontsize=16)

    for sample_idx in range(config.num_samples):
        # Start from random Gaussian in [-1,1]
        x_init = torch.randn(1, config.num_channels, config.image_size, config.image_size,
                             device=config.device).requires_grad_(True)

        captured_x, captured_V = ode_integrate(model, x_init, config, capture_steps)

        for time_idx, (img, pot) in enumerate(zip(captured_x, captured_V)):
            if config.num_samples > 1:
                ax = axes[sample_idx, time_idx]
            else:
                ax = axes[time_idx]

            img_np = img.squeeze().numpy()
            if config.num_channels == 1:
                ax.imshow(img_np, cmap='gray', vmin=-1, vmax=1)
            else:
                ax.imshow(np.transpose(img_np, (1,2,0)), vmin=-1, vmax=1)
            ax.axis('off')
            ax.set_title(f'V={pot.item():.4f}', fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_filename = os.path.join(results_dir, f"samples_epoch_{epoch:03d}.png")
    plt.savefig(out_filename)
    plt.close(fig)

    log_message(f"Samples saved to {out_filename}", config, results_dir)


def log_message(msg, config, results_dir):
    """
    Logs 'msg' to "<model_base_name>_results/training_log.txt".
    """
    results_dir = f"{os.path.splitext(config.model_save_path)[0]}_results"
    os.makedirs(results_dir, exist_ok=True)
    log_file = os.path.join(results_dir, "training_log.txt")
    with open(log_file, 'a') as f:
        f.write(msg + "\n")
