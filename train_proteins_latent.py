# File: train.py

import os
import time
import copy
import torch
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

# 1) Import absl + config
from absl import app, flags, logging
import config  # <-- your config file
config.define_flags()  # register all the flags
FLAGS = flags.FLAGS

# 2) Import everything else
from torchvision import datasets, transforms

# Our newly-created utilities
from utils_proteins import (
    ProteinDataset,
    ProteinDatasetOrig,
    Encoder,
)

from utils_train_proteins import (
    create_timestamped_dir,
    generate_samples,
    flow_weight,
    gibbs_sampling_n_steps_fast,
    gibbs_sampling_time_sweep,
    warmup_lr,
    ema,
    infiniteloop
)

# Our FID utilities
# from utils_fid import compute_fid_for_times

# ─────────────────────────────────────────────────────────────────────────
# 3) Import the *Static* EBM model that uses patches + ViT:
# ─────────────────────────────────────────────────────────────────────────
#   >> Make sure this file is the one from the snippet above.
#   >> If you want time dependence, import EBModelWrapper instead.
# # ─────────────────────────────────────────────────────────────────────────
# from network_transformer_vit import EBViTModelWrapper
from model_proteins import Unet1DModelWrapper, VAE

# TorchCFM flow classes
from torchcfm.conditional_flow_matching import (
    ExactOptimalTransportConditionalFlowMatcher,
    # Or whichever FlowMatcher you prefer
)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

cwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(cwd) 


##############################################################################
# Helper: count_parameters
##############################################################################
def count_parameters(module: torch.nn.Module):
    """Count the total trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def train(argv):
    # -----------------------------------------------------------------------
    # 1) Create the output directory
    # -----------------------------------------------------------------------
    savedir = create_timestamped_dir(FLAGS.output_dir, FLAGS.model, "_latent") 
    
    # If user didn't specify --my_log_dir, set it to savedir
    if not FLAGS.my_log_dir:
        FLAGS.my_log_dir = savedir

    # Configure Abseil logging
    logging.get_absl_handler().use_absl_log_file(
        program_name="train",
        log_dir=FLAGS.my_log_dir
    )
    logging.set_verbosity(logging.INFO)
    logging.info(f"Using output directory: {savedir}\n")

    # -----------------------------------------------------------------------
    # 2) Log all hyperparameters
    # -----------------------------------------------------------------------
    logging.info("========== Hyperparameters (FLAGS) ==========")
    for key, val in FLAGS.flag_values_dict().items():
        logging.info(f"{key} = {val}")
    logging.info("=============================================\n")

    # -----------------------------------------------------------------------
    # 3) Setup dataset / dataloader
    # -----------------------------------------------------------------------
    scenario = "aav" # "aav", "gfp"
    task = "hard" # "medium", "hard"
    load_csv_name = scenario + '_' + task + '.csv'
    seq_len = 28 if scenario == "aav" else 240 # padding to have //8 
    latent_dim = 16 if scenario == "aav" else 32
    step_switch_to_CD = 6000 #10000 for AAV medium, for AAV hard try 5000

    if scenario == "aav":
        train_df = pd.read_csv(os.getcwd() + '/data/' + load_csv_name)
    elif scenario == "gfp":
        train_df = pd.read_csv(os.getcwd() + '/data/' + load_csv_name)
    else:
        raise NotImplementedError

    tokenizer = Encoder()
    dataset = ProteinDatasetOrig(train_df, scenario, task, tokenizer, seq_len)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True,num_workers=FLAGS.num_workers,drop_last=True,)
    datalooper = infiniteloop(dataloader)

    # VAE model: Load VAE encoder/decoder
    encoder_path = Path(__file__).resolve().parents[0] / 'vae/' / f"vae_{scenario}_{task}.pt"
    vae_model = VAE(input_dim=seq_len,latent_dim=latent_dim).to(device)
    vae_model.load_state_dict(torch.load(encoder_path, map_location=device)["state_dict"], strict=True)
    vae_model.eval()
    # -----------------------------------------------------------------------
    # 4) Initialize our model + EMA model
    # -----------------------------------------------------------------------
    # EBStaticModelWrapper => always uses t=0.5 internally
    ch_mult = (1, 2) # (1,2) 124
    dim = 28 # 28 # seq_len # adapt channels to 28, 240 TODO 
    net_model = Unet1DModelWrapper(
        dim=dim,
        channels=1, # 1d vector in latent space
        dim_mults=ch_mult,
        dropout=FLAGS.dropout,
        output_scale=FLAGS.output_scale,
    ).to(device) 

    # ─────────────────────────────────────────────────────────────────────
    # Print separate parameter counts: UNet portion vs. ViT portion
    # ─────────────────────────────────────────────────────────────────────
    total_params = count_parameters(net_model)
    # For the patch-based head, check submodules: patch_embed, transformer_encoder, final_linear
    transformer_params = 0
    for submodule_name in ["patch_embed", "transformer_encoder", "final_linear"]:
        submod = getattr(net_model, submodule_name, None)
        if submod is not None:
            transformer_params += count_parameters(submod)
    unet_params = total_params - transformer_params

    logging.info("=== Parameter Counts ===")
    logging.info(f"UNet portion:         {unet_params}")
    logging.info(f"Transformer portion:  {transformer_params}")
    logging.info(f"Total:                {total_params}")
    logging.info(f"Total (in millions):  {total_params / 1e6:.2f} M\n")

    # Clone for EMA
    ema_model = copy.deepcopy(net_model).to(device)

    # Optimizer & LR scheduler
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)

    start_step = 0
    # Optionally resume from a checkpoint
    if FLAGS.resume_ckpt and os.path.exists(os.path.join(os.getcwd(),'results',FLAGS.resume_ckpt)):
        print('CHECKPOINT RESUMED')
        resume_ckpt = os.path.join(os.getcwd(),'results',FLAGS.resume_ckpt)
        logging.info(f"Resuming from checkpoint: {resume_ckpt}")
        checkpoint = torch.load(resume_ckpt, map_location=device, weights_only=True)
        net_model.load_state_dict(checkpoint["net_model"])
        ema_model.load_state_dict(checkpoint["ema_model"])
        sched.load_state_dict(checkpoint["sched"])
        optim.load_state_dict(checkpoint["optim"])
        start_step = checkpoint["step"]
        logging.info(f"Resumed at step {start_step}")

    # -----------------------------------------------------------------------
    # 5) Setup Flow Matcher for (t, x_t, u_t)
    # -----------------------------------------------------------------------
    sigma = 0.0
    flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)

    # -----------------------------------------------------------------------
    # 6) TRAINING LOOP
    # -----------------------------------------------------------------------
    steps_per_log = 100
    last_log_time = time.time()

    with torch.backends.cuda.sdp_kernel(
        enable_flash=False,
        enable_mem_efficient=False,
        enable_math=True
    ):
        for step in range(start_step, FLAGS.total_steps + 1):
            optim.zero_grad()

            # -------------------------------------------------
            # (a) Sample real data & random noise
            # -------------------------------------------------
            x1 = next(datalooper).to(device)  # (B,seq_len,20)
            # Get token embeddings 
            with torch.no_grad():
                x1 = vae_model.forward(x1)[1].unsqueeze(1) # 1 is mu, 3 is z

            x0 = torch.randn_like(x1)         # (B,3,32,32)

            # -------------------------------------------------
            # (b) Flow matching => (t, x_t, u_t)
            # -------------------------------------------------
            t, xt, ut = flow_matcher.sample_location_and_conditional_flow(x0, x1)

            # -------------------------------------------------
            # (1) FLOW LOSS
            # -------------------------------------------------
            # EBStaticModelWrapper ignores 't' in potential/velocity,
            # but we keep the code for clarity.
            vt = net_model(t, xt.permute(0,2,1)).permute(0,2,1)  # predicted velocity
            flow_mse = (vt - ut).square()
            w_flow_ = flow_weight(t, cutoff=FLAGS.time_cutoff)
            flow_loss = torch.mean(w_flow_ * flow_mse.mean(dim=[1, 2]))

            # -------------------------------------------------
            # (2) CD LOSS at t=1 only
            # -------------------------------------------------
            cd_loss = torch.tensor(0.0, device=device)
            if step > step_switch_to_CD: # TODO 
                FLAGS.lambda_cd = 1e-4
            if FLAGS.lambda_cd > 0.0:
                pos_energy_1 = net_model.potential(x1.permute(0,2,1), torch.ones_like(t))

                n = x1.shape[0]
                half = n // 2
                at_data_mask = torch.cat([torch.ones(half), torch.zeros(n - half)])
                at_data_mask = at_data_mask[torch.randperm(n)]
                x_neg_1 = gibbs_sampling_time_sweep(
                    x_init=x1.permute(0,2,1),
                    model=net_model,
                    at_data_mask=at_data_mask,
                    n_steps=FLAGS.n_gibbs,  # e.g. 150 steps
                    dt=FLAGS.dt_gibbs,    # each step is 0.01 => total time 1.5
                    clamp=False
                )
                # x_neg_1 = gibbs_sampling_n_steps_fast(
                #     x_init=x1,
                #     model=net_model,
                #     t=torch.ones_like(t),
                #     n_steps=FLAGS.n_gibbs,
                #     dt=FLAGS.dt_gibbs,
                #     epsilon=FLAGS.epsilon_max,
                # )
                neg_energy_1 = net_model.potential(x_neg_1, torch.ones_like(t))
                cd_val = pos_energy_1.mean() - neg_energy_1.mean()
                cd_loss = FLAGS.lambda_cd * cd_val

                # Optionally clamp negative side
                if FLAGS.cd_neg_clamp > 0:
                    cd_loss = torch.maximum(
                        cd_loss,
                        torch.tensor(-FLAGS.cd_neg_clamp, device=device)
                    )

            # -------------------------------------------------
            # Combine total loss: (flow + cd)
            # -------------------------------------------------
            total_loss = flow_loss + cd_loss
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()

            # EMA
            ema(net_model, ema_model, FLAGS.ema_decay)

            # -------------------------------------------------
            # Logging
            # -------------------------------------------------
            if step % steps_per_log == 0:
                current_time = time.time()
                elapsed = current_time - last_log_time
                steps_per_sec = steps_per_log / elapsed if elapsed > 1e-9 else 0.0
                last_log_time = current_time
                current_lr = sched.get_last_lr()[0]

                logging.info(
                    f"[Step {step}] "
                    f"flow={flow_loss.item():.6f}, "
                    f"cd={cd_loss.item():.6f}, "
                    f"LR={current_lr:.6f}, "
                    f"{steps_per_sec:.2f} it/s"
                )

            # -------------------------------------------------
            # Save, Generate Samples, Checkpoints
            # -------------------------------------------------
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:# and step > 0:
                real_batch = next(datalooper).to(device)[:8]
                with torch.no_grad():
                    real_batch = vae_model.forward(real_batch)[1].unsqueeze(1)
                seq_len = real_batch.shape[2] # dim for latent: 16 or 32

                generate_samples(
                    net_model,
                    savedir,
                    step,
                    seq_len,
                    net_="normal",
                    real_data=real_batch,
                    vae=vae_model
                )
                generate_samples(
                    ema_model,
                    savedir,
                    step,
                    seq_len,
                    net_="ema",
                    real_data=real_batch, 
                    vae=vae_model
                )

                ckpt_path = os.path.join(savedir, f"{FLAGS.model}_{scenario}_{task}_weights_step_{step}.pt")
                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                    },
                    ckpt_path,
                )
                logging.info(f"Checkpoint saved to {ckpt_path}")


def main(argv):
    train(argv)


if __name__ == "__main__":
    app.run(main)
