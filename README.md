# Energy Matching 
<img align="right" src="media/EM_2D.png" width="30%" alt="Energy Matching Illustration" />
Energy Matching unifies flow matching and energy-based models in a single time-independent scalar field, enabling efficient transport between the source and target distributions while retaining explicit likelihood information for flexible, high-quality generation. [NeurIPS 2025]

**Version 1.0** – This is the official repository for the paper [Energy Matching](https://arxiv.org/abs/2504.10612).

### Checkpoints
- **CIFAR-10** (Image → Scalar, 50M parameters): warm-up and main-training checkpoints on [Hugging Face](https://huggingface.co/m1balcerak/energy_matching) reach **FID ≈ 3.3** around `T=3.25`.
- **ImageNet32** (Image → Scalar, 50M parameters): warm-up and main-training checkpoints on [Hugging Face](https://huggingface.co/m1balcerak/energy_matching) reach **FID ≈ 6.6** around `T=2.50`.

### Setup (CUDA)
1. Create and activate a Python environment (conda example):
   ```bash
   conda create -n energy-matching python=3.10 -y
   conda activate energy-matching
   ```
2. Install PyTorch with CUDA support and the project requirements:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install -r requirements.txt
   ```

## Running the examples

### 2D Playground – Eight Gaussians to Two Moons
Experiment with the core idea in a lightweight setting using the notebook at `experiments/toy2d/tutorial_2D.ipynb`. It visualizes how the potential energy field transports particles in 2D from eight Gaussians to two moons.

### CIFAR‑10 Training and Evaluation

<p align="center">
  <strong>Langevin MCMC (unconditional) <br> Trajectory from T = 0 to T = 4 (FID = 3.3)</strong><br>
  <img src="media/cifar10_FID_3_3.gif" width="60%" alt="Trajectory Animation">
</p>

Initial training (warm-up, Algorithm 1):
```bash
torchrun --nproc_per_node=4 experiments/cifar10/train_cifar_multigpu.py \
    --total_steps 145000 \
    --lr 1.2e-3 \
    --batch_size 128 \
    --time_cutoff 1.0 \
    --epsilon_max 0.0 \
    --lambda_cd 0. \
    --n_gibbs 0 \
    --ema_decay 0.9999 \
    --save_step 5000
```
Main training with contrastive divergance (Algorithm 2):
```bash
torchrun --nproc_per_node=4 experiments/cifar10/train_cifar_multigpu.py \
    --resume_ckpt /PATH/TO/warm_up_checkpoint.pt \
    --total_steps 147000 \
    --lr 1.2e-3 \
    --batch_size 128 \
    --time_cutoff 1.0 \
    --epsilon_max 0.01 \
    --lambda_cd 1e-3 \
    --n_gibbs 200 \
    --ema_decay 0.99 \
    --save_step 100 \
    --dt_gibbs 0.01 \
    --cd_neg_clamp 0.02  \
    --split_negative True \
    --same_temperature_scheduler True
```
Evaluation FID across trajectories at times `T=1.0` to `T=5.0` (Heun solver):
```bash
python experiments/cifar10/fid_cifar_heun_1gpu.py \
    --resume_ckpt=/PATH/TO/main_training_checkpoint.pt \
    --output_dir=./sampling_results \
    --use_ema True \
    --time_cutoff 1.0 \
    --epsilon_max 0.01 \
    --batch_size 64 \
    --dt_gibbs 0.01
```
Pretrained CIFAR-10 checkpoints are available at [Hugging Face](https://huggingface.co/m1balcerak/energy_matching_cifar10).
Use `cifar10_warm_up_145000.pt` for the warm-up phase and `cifar10_main_training_147000.pt` after the main training. The latter obtains an **FID of 3.3** at around `T=3.25`.

To generate CIFAR-10 images using unconditional Langevin Monte Carlo sampling from the trained Energy Matching model, run:

```bash
python experiments/cifar10/sample_cifar_heun_1gpu.py \
    --resume_ckpt=/PATH/TO/main_training_checkpoint.pt \
    --batch_size 128 \
    --time_cutoff 1.0 \
    --epsilon_max 0.01 \
    --dt_gibbs 0.01 \
    --use_ema True \
    --t_end=3.25
```
Here, `t_end` corresponds to the sampling time $\tau_s$.

### ImageNet32 Training and Evaluation

Download the downsampled ImageNet32 training batches (`train_data_batch_1` ... `train_data_batch_10`) from the [official release](https://patrykchrabaszcz.github.io/Imagenet32/) and place them under `experiments/imagenet/data/Imagenet32_train/` (or point the `IMAGENET32_PATH` environment variable to that folder) before launching training.

Initial training (Algorithm 1):
```bash
torchrun --nproc_per_node=7 experiments/imagenet/train_imagenet_multigpu.py \
    --total_steps 640000 \
    --lr 6e-4 \
    --batch_size 128 \
    --time_cutoff 1.0 \
    --epsilon_max 0. \
    --lambda_cd 0. \
    --ema_decay 0.9999 \
    --save_step 80000
```
Main training with contrastive divergence (Algorithm 2):
```bash
torchrun --nproc_per_node=7 experiments/imagenet/train_imagenet_multigpu.py \
    --resume_ckpt=/PATH/TO/warm_up_imagenet_checkpoint.pt \
    --total_steps 641000 \
    --lr 6e-4 \
    --batch_size 128 \
    --time_cutoff 1.0 \
    --epsilon_max 0.01 \
    --lambda_cd 0.001 \
    --n_gibbs 200 \
    --ema_decay 0.99 \
    --save_step 100 \
    --cd_neg_clamp 0.02 \
    --split_negative True \
    --same_temperature_scheduler True
```
FID evaluation across trajectory times `T=0.75` to `T=4.0` (Heun solver):
```bash
torchrun --nproc_per_node=1 experiments/imagenet/fid_imagenet_heun_multigpu.py \
    --resume_ckpt=/PATH/TO/main_training_imagenet_checkpoint.pt \
    --output_dir=./sampling_results \
    --use_ema True \
    --time_cutoff 1.0 \
    --epsilon_max 0.01 \
    --batch_size 128 \
    --dt_gibbs 0.01
```
Pretrained ImageNet32 checkpoints (warm-up and main training) are hosted on [Hugging Face](https://huggingface.co/m1balcerak/energy_matching).


### Protein inverse design
Train the model with:
```bash
python experiments/proteins/train_proteins.py \
    --epsilon_max 0.1 \
    --time_cutoff 0.9 \
    --n_gibbs 200 \
    --dt_gibbs 0.01
```
Pretrained AAV medium/hard checkpoints are available at [Hugging Face](https://huggingface.co/m1balcerak/energy_matching). Run conditional sampling with:
```bash
python experiments/proteins/sampling.py
```
The VAE used for the continuous latent space and the dataset is already provided. 



## Citation

If you find our work useful, please consider citing:

```bibtex
@article{balcerak2025energy,
  title={Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling},
  author={Balcerak, Michal and Amiranashvili, Tamaz and Terpin, Antonio and Shit, Suprosanna and Bogensperger, Lea and Kaltenbach, Sebastian and Koumoutsakos, Petros and Menze, Bjoern},
  journal={arXiv preprint arXiv:2504.10612},
  year={2025}
}
```
