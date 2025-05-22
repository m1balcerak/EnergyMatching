# EnergyMatching
Energy Matching unifies flow matching and energy-based models in a single time-independent scalar field, enabling efficient transport from noise to data while retaining explicit likelihood information for flexible, high-quality generation.

## Setup (CUDA)
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
- A simple 2D playground is provided in `tutorial_2D.ipynb`.
- Checkpoints for MNIST and CIFAR‑10 are included under `assets/`.

### LID on MNIST
Evaluate the local intrinsic dimension on MNIST by running:
```bash
python lid_mnist.py
```

### Protein inverse design
Train the model and sample sequences with:
```bash
python train_proteins_latent.py
python sampling_latent.py
```
The VAE used for the continuous latent space is already provided.

### CIFAR‑10 training and evaluation
Initial training:
```bash
torchrun --nproc_per_node=4 train_cifar_multigpu.py --lr 8e-4 --batch_size 128 --epsilon_max 0.0 --n_gibbs 0 --lambda_cd 0. --time_cutoff 1.0 --ema_decay 0.9999 --save_step 5000
```
Fine‑tuning with Algorithm 2:
```bash
torchrun --nproc_per_node=4 train_cifar_multigpu2.py --lr 8e-4 --batch_size 64 --resume_ckpt 'path_to_pretrained' --epsilon_max 0.01 --n_gibbs 201 --lambda_cd 1e-4 --time_cutoff 0.9 --ema_decay 0.999 --save_step 100 --dt_gibbs 0.01 --cd_loss_threshold 1.0 --split_negative=True
```
Evaluation:
```bash
python fid_evaluation_heun.py --resume_ckpt=PATH --output_dir=./sampling_results --use_ema True --time_cutoff 0.9 --epsilon_max 0.01
```
A pretrained CIFAR‑10 checkpoint is available in `assets/`, so training from scratch is optional.
