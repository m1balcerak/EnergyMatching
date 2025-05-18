import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from torchcfm.utils import sample_8gaussians, sample_moons


def temperature(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 2 and t.size(1) == 1:
        t = t.squeeze(-1)
    eps = torch.zeros_like(t)
    mask_mid = (t >= 0.8) & (t < 1.0)
    eps[mask_mid] = 0.2 * (t[mask_mid] - 0.8) / 0.2
    eps[t >= 1.0] = 0.15
    return eps


def velocity_training(model: nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    x = x.detach().requires_grad_(True)
    V = model(x, t)
    gradV = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
    return -gradV


def velocity_inference(model: nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    with torch.enable_grad():
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        V = model(x, t)
        gradV = torch.autograd.grad(V.sum(), x, create_graph=False)[0]
    return -gradV


def gibbs_sampler(model: nn.Module, x_init: torch.Tensor, t_start: torch.Tensor, *, steps: int = 10, dt: float = 0.01) -> torch.Tensor:
    x = x_init
    for step in range(steps):
        t_current = t_start + (1 - t_start) * ((step + 1) / steps)
        x.requires_grad_(True)
        V = model(x, torch.tensor(1.0, device=x.device))
        g = torch.autograd.grad(V.sum(), x, create_graph=False)[0]
        eps = temperature(t_current)
        noise_scale = torch.sqrt(2.0 * eps * dt).unsqueeze(-1)
        noise = noise_scale * torch.randn_like(x)
        x = (x - g * dt + noise).detach()
    return x


def simulate_piecewise_length(model: nn.Module, x0: torch.Tensor, *, dt: float = 0.01, max_length: float = 4.0):
    x = x0
    traj = [x0.cpu().numpy()]
    times = [0.0]
    t_now = 0.0
    cum_length = 0.0
    device = x0.device
    while cum_length < max_length:
        t_tensor = torch.tensor([t_now], dtype=x0.dtype, device=device)
        g = velocity_inference(model, x, t_tensor)
        eps_now = temperature(t_tensor).item()
        if t_now < 0.8:
            dx = g * dt
        else:
            noise = torch.sqrt(torch.tensor(2.0 * eps_now * dt, device=device)) * torch.randn_like(x)
            dx = g * dt + noise
        x = (x + dx).detach()
        step_length = torch.norm(dx).item()
        cum_length += step_length
        t_now += dt
        traj.append(x.cpu().numpy())
        times.append(t_now)
    return np.array(traj), np.array(times)


def plot_trajectories_custom(traj: np.ndarray) -> None:
    n = 2000
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=3, alpha=0.8, c='black', marker='s')
    plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.1, c='olive')
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1.0, c='blue', marker='*')
    for i in range(10):
        plt.plot(traj[:, i, 0], traj[:, i, 1], c='red', linewidth=1.2, alpha=1.0)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def train(model_class, *, device: torch.device, batch_size: int, lr: float, epochs_phase1: int, epochs_phase2: int, flow_weight: float, ebm_weight: float, sigma: float, save_dir: str) -> nn.Module:
    os.makedirs(save_dir, exist_ok=True)
    model = model_class(dim=2, w=128, time_varying=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)

    def x0_dist(n):
        return sample_8gaussians(n).to(device)

    def x1_dist(n):
        return sample_moons(n).to(device)

    for _ in range(epochs_phase1):
        optimizer.zero_grad()
        x0 = x0_dist(batch_size)
        x1 = x1_dist(batch_size)
        t_samp, x_t, u_t = FM.sample_location_and_conditional_flow(x0, x1)
        v_pred = velocity_training(model, x_t, t_samp.unsqueeze(-1))
        loss_flow = (v_pred - u_t).pow(2).mean()
        loss_flow.backward()
        optimizer.step()

    for _ in range(epochs_phase2):
        optimizer.zero_grad()
        x0 = x0_dist(batch_size)
        x1 = x1_dist(batch_size)
        t_flow, x_t_flow, u_t_flow = FM.sample_location_and_conditional_flow(x0, x1)
        v_pred_flow = velocity_training(model, x_t_flow, t_flow.unsqueeze(-1))
        loss_flow = (v_pred_flow - u_t_flow).pow(2).mean()
        x_data = x1_dist(batch_size)
        Epos = model(x_data, torch.tensor(1.0, device=device)).mean()
        half_bs = batch_size // 2
        x_data_init = x1_dist(half_bs)
        x_prior_init = x0_dist(half_bs)
        x_init_neg = torch.cat([x_data_init, x_prior_init], dim=0)
        t_start = torch.cat([torch.ones(half_bs, device=device), torch.zeros(half_bs, device=device)], dim=0)
        x_neg = gibbs_sampler(model, x_init_neg, t_start, steps=200, dt=0.01)
        Eneg = model(x_neg, torch.tensor(1.0, device=device)).mean()
        loss_ebm = Epos - Eneg
        loss = flow_weight * loss_flow + ebm_weight * loss_ebm
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), os.path.join(save_dir, 'final_V_model.pth'))
    return model

