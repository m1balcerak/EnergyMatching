import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# == Your local modules ==
from model import (
    ModifiedResNet50,
    DeeperPureCNN,
    EBM_ResNet_1200K,
    Network_2M_MNIST28x28
)

# Import from utils.py
from utils import (
    linear_noise_schedule,
    add_noise,
    compute_velocity,
    gibbs_sampling_step,
    gibbs_sampling_n_steps,
    gibbs_sampling_n_steps_fast,  # ### CHANGES ###
    save_diagnostics_figure,
    generate_and_save_samples
)
from ot_utils import compute_ot_couplings_and_velocity


def log_message(msg, log_dir):
    """
    Appends a log message to the file "<log_dir>/training_log.txt".
    Creates 'log_dir' if it doesn't exist.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training_log.txt")
    with open(log_file, 'a') as f:
        f.write(msg + "\n")


class Config:
    def __init__(
        self, alpha, batch_size, lr, hidden_dim, epsilon,
        lambda_model, lambda_cd, use_ot, k,
        grad_clip, cd_flow_ratio,
        n_gibbs=1  # ### CHANGES ###
    ):
        """
        alpha: unused param, but kept for interface consistency
        batch_size: images per batch
        lr: learning rate
        hidden_dim: dimension in your EBM/potential model
        epsilon: used in OT or gibbs sampling
        lambda_model: weight on the flow (velocity) loss
        lambda_cd: weight on the CD portion
        use_ot: whether to use OT-based velocity
        k: scale factor for the gibbs step
        grad_clip: gradient clipping norm
        cd_flow_ratio: ratio-based clamp so |CD| <= cd_flow_ratio * |Flow|
        n_gibbs: number of Gibbs sampling steps for negative samples  ### CHANGES ###

        For MNIST (28x28, single channel).
        """
        self.batch_size = batch_size
        self.epochs = 10000
        self.lr = lr

        # MNIST specifics
        self.image_size = 28
        self.num_channels = 1

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.alpha = alpha
        self.hidden_dim = hidden_dim
        self.lambda_model = lambda_model
        self.lambda_cd = lambda_cd
        self.use_ot = use_ot

        # JKO-related
        self.dt = 1.0
        self.epsilon = epsilon
        self.k = k

        # Clipping & ratio
        self.grad_clip = grad_clip
        self.cd_flow_ratio = cd_flow_ratio

        # We'll name the model file with "_mnist"
        self.model_save_path = (
            f'Network_2M_MNIST28x28_alpha{self.alpha}_bs{self.batch_size}_lr{self.lr}_hd{self.hidden_dim}_mnist.pth'
        )
        self.num_workers = 4

        # For ODE-based sample generation
        self.num_samples = 8
        self.integration_steps = 100
        self.timepoints = 5
        self.dt_sampling = 1.0 / self.integration_steps

        # ### CHANGES ###
        self.n_gibbs = n_gibbs


def train(config):
    """
    Train the Network_2M_MNIST28x28 on MNIST with JKO + CD.
    Key features:
      - Gradient clipping
      - Ratio-based clamp of CD
      - CD term clipped >= 0
      - Diagnostics every 10 epochs
      - ODE-based sample generation
      - Saving best model (by total train loss) as 'best_model.pth'
      - Saving latest model every 10 epochs as 'latest_model.pth'
      - Timestamp in the _results directory name
    """
    # 1) Create the main results dir with timestamp in its name
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(config.model_save_path)[0]
    results_dir = f"{base_name}_{time_str}_results"
    os.makedirs(results_dir, exist_ok=True)

    # -- Log the hyperparameters at the very beginning --
    hp_msg = (
        f"Hyperparameters used:\n"
        f"  alpha={config.alpha}\n"
        f"  batch_size={config.batch_size}\n"
        f"  lr={config.lr}\n"
        f"  hidden_dim={config.hidden_dim}\n"
        f"  epsilon={config.epsilon}\n"
        f"  lambda_model={config.lambda_model}\n"
        f"  lambda_cd={config.lambda_cd}\n"
        f"  use_ot={config.use_ot}\n"
        f"  k={config.k}\n"
        f"  grad_clip={config.grad_clip}\n"
        f"  cd_flow_ratio={config.cd_flow_ratio}\n"
        f"  n_gibbs={config.n_gibbs}\n"  # ### CHANGES ###
    )
    log_message(hp_msg, results_dir)
    # ----------------------------------------------------

    # We'll have subdirectories for storing just two model files:
    #  - best_model.pth    (always overwritten when a new best total loss is found)
    #  - latest_model.pth  (overwritten every 10 epochs)
    best_model_path = os.path.join(results_dir, "best_model.pth")
    latest_model_path = os.path.join(results_dir, "latest_model.pth")

    # 2) Dataloader (MNIST)
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True
    )

    # 3) Model, optimizer, scheduler
    model = Network_2M_MNIST28x28().to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=1e-6
    )

    # Track best total training loss
    best_loss = float('inf')

    # 4) Training loop
    for epoch in range(1, config.epochs + 1):
        model.train()

        epoch_loss = 0.0
        epoch_flow_loss = 0.0
        epoch_pos_loss = 0.0
        epoch_neg_loss = 0.0
        last_gamma = None  # for OT stats

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(config.device)  # shape [B,1,28,28]

            # 1) random t
            B = data.size(0)
            t = torch.rand(B, 1, 1, 1, device=config.device)
            t_linear = linear_noise_schedule(t)

            # 2) Add noise
            noisy_data, noise_data = add_noise(data, t_linear)

            # 3) Flow velocity
            V_noisy = model(noisy_data)
            grad_V = torch.autograd.grad(
                outputs=V_noisy,
                inputs=noisy_data,
                grad_outputs=torch.ones_like(V_noisy),
                create_graph=True
            )[0]
            pred_velocity = -grad_V

            # 4) target velocity
            if config.use_ot:
                data_flat = data.view(B, -1)
                noise_flat = noise_data.view(B, -1)
                gamma, v_data_flat = compute_ot_couplings_and_velocity(
                    x=data_flat,
                    z=noise_flat,
                    epsilon=config.epsilon,
                    max_iter=10000,
                    tol=1e-4,
                    device=config.device
                )
                target_velocity = v_data_flat.view(B, 1, config.image_size, config.image_size)
                last_gamma = gamma.detach()
            else:
                target_velocity = compute_velocity(data, noise_data, t_linear)

            flow_loss_val = criterion(pred_velocity, target_velocity)

            # 5) Positive part
            V_real = model(data)
            pos_loss_val = torch.mean(V_real)

            # 6) Negative part (CD) => REPLACED single step with n-step
            # ### CHANGES ###
            x_neg = gibbs_sampling_n_steps_fast(
                data, model, config, config.n_gibbs
            )
            V_neg = model(x_neg)
            neg_loss_val = -torch.mean(V_neg)

            # Weighted sum
            loss_flow = config.lambda_model * flow_loss_val

            # We do a raw CD
            loss_cd_raw = config.lambda_cd * (pos_loss_val + neg_loss_val)
            # Clip at 0 => no negative CD allowed
            loss_cd_raw = torch.clamp(loss_cd_raw, min=0.0)

            # ratio-based clamp => |CD| <= cd_flow_ratio * |Flow|
            with torch.no_grad():
                abs_flow = torch.abs(loss_flow)
                abs_cd_raw = torch.abs(loss_cd_raw)
                if abs_cd_raw > config.cd_flow_ratio * abs_flow:
                    scale_factor = (config.cd_flow_ratio * abs_flow) / (abs_cd_raw + 1e-9)
                else:
                    scale_factor = 1.0

            loss_cd = loss_cd_raw * scale_factor
            loss = loss_flow + loss_cd

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            # Stats
            epoch_loss += loss.item()
            epoch_flow_loss += flow_loss_val.item()
            epoch_pos_loss += pos_loss_val.item()
            epoch_neg_loss += neg_loss_val.item()

        # Scheduler step
        scheduler.step()

        # Averages
        avg_loss = epoch_loss / len(train_loader)
        avg_flow = epoch_flow_loss / len(train_loader)
        avg_pos = epoch_pos_loss / len(train_loader)
        avg_neg = epoch_neg_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]

        # Logging
        msg = (
            f"Epoch [{epoch}/{config.epochs}] "
            f"LR={current_lr:.6f}, "
            f"Train Loss={avg_loss:.6f}, Flow_Loss={avg_flow:.6f}, "
            f"Pos_Loss={avg_pos:.6f}, Neg_Loss={avg_neg:.6f}, "
            f"Use_OT={config.use_ot}"
        )
        log_message(msg, results_dir)

        if config.use_ot and last_gamma is not None:
            gamma_stats = (
                f"  OT Gamma stats: mean={torch.mean(last_gamma):.6f}, "
                f"std={torch.std(last_gamma):.6f}, "
                f"min={torch.min(last_gamma):.6f}, "
                f"max={torch.max(last_gamma):.6f}"
            )
            log_message(gamma_stats, results_dir)

        # ------------- Save best model (total loss) if improved -------------
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)
            log_message(
                f"New best training loss: {best_loss:.6f} at epoch {epoch}. "
                f"Overwriting best_model.pth",
                results_dir
            )

        # ------------- Save "latest" + diagnostics every 10 epochs -------------
        if epoch % 10 == 0 or epoch == config.epochs:
            torch.save(model.state_dict(), latest_model_path)
            log_message(f"Overwrote 'latest_model.pth' at epoch {epoch}", results_dir)

            # ODE-based sampling
            original_dt = config.dt
            config.dt = config.dt_sampling
            generate_and_save_samples(model, config, epoch, results_dir)
            config.dt = original_dt

            # Diagnostics
            model.eval()
            data_iter = iter(train_loader)
            diag_data, _ = next(data_iter)
            diag_data = diag_data[:4].to(config.device)
            save_diagnostics_figure(model, diag_data, config, epoch, results_dir, [1, 10, 100])


def main():
    parser = argparse.ArgumentParser(
        description="Train an EBM on MNIST (28×28) with JKO + CD. "
                    "Saves best_model.pth (based on total train loss) and latest_model.pth."
    )
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--lambda_model', type=float, default=1.0)
    parser.add_argument('--lambda_cd', type=float, default=0.025)
    parser.add_argument('--use_ot', action='store_true')
    parser.add_argument('--k', type=float, default=0.001,
                        help="Scale factor for the Gibbs step.")
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help="Clip gradient norm to this value.")
    parser.add_argument('--cd_flow_ratio', type=float, default=1.0,
                        help="Ratio clamp so |CD| <= cd_flow_ratio * |Flow|.")

    # ### CHANGES ###
    parser.add_argument('--n_gibbs', type=int, default=10,
                        help="Number of Gibbs sampling steps for negative samples.")

    args = parser.parse_args()

    config = Config(
        alpha=args.alpha,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        epsilon=args.epsilon,
        lambda_model=args.lambda_model,
        lambda_cd=args.lambda_cd,
        use_ot=args.use_ot,
        k=args.k,
        grad_clip=args.grad_clip,
        cd_flow_ratio=args.cd_flow_ratio,
        n_gibbs=args.n_gibbs  # ### CHANGES ###
    )

    train(config)


if __name__ == "__main__":
    main()
