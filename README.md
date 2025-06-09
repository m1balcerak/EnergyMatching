# Energy Matching
<img align="right" src="EM_2D.png" width="300" alt="Energy Matching Illustration" />
Energy Matching unifies flow matching and energy-based models in a single time-independent scalar field, enabling efficient transport between the source and target distributions while retaining explicit likelihood information for flexible, high-quality generation.

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
- A simple 2D playground is provided in `experiments/toy2d/tutorial_2D.ipynb`.
- A checkpoint for MNIST is provided under `experiments/mnist/assets/`. The CIFAR‑10 checkpoint is too large to include but can be trained using the commands below.

### LID on MNIST
Evaluate the local intrinsic dimension on MNIST by running:
```bash
python experiments/mnist/lid_mnist.py
```

### Train the MNIST model
You can also train the MNIST energy model yourself. A minimal command looks like:
```bash
python experiments/mnist/train_mnist.py --lr 1e-4 --batch_size 128
```

### Protein inverse design
Train the model and sample sequences with:
```bash
python experiments/proteins/train_proteins_latent.py
python experiments/proteins/sampling_latent.py
```
The VAE used for the continuous latent space is already provided.

### CIFAR‑10 training and evaluation
Initial training:
```bash
torchrun --nproc_per_node=4 experiments/cifar10/train_cifar_multigpu.py --lr 8e-4 --batch_size 128 --epsilon_max 0.0 --n_gibbs 0 --lambda_cd 0. --time_cutoff 1.0 --ema_decay 0.9999 --save_step 5000
```
Fine‑tuning with Algorithm 2 (same script):
```bash
torchrun --nproc_per_node=4 experiments/cifar10/train_cifar_multigpu.py --lr 8e-4 --batch_size 64 --resume_ckpt 'path_to_pretrained' --epsilon_max 0.01 --n_gibbs 201 --lambda_cd 1e-4 --time_cutoff 0.9 --ema_decay 0.999 --save_step 100 --dt_gibbs 0.01 --cd_loss_threshold 1.0 --split_negative=True
```
Evaluation (multi‑GPU):
```bash
torchrun --nproc_per_node=2 experiments/cifar10/fid_cifar_heun_multigpu.py \
    --resume_ckpt=PATH \
    --output_dir=./sampling_results \
    --use_ema True \
    --time_cutoff 0.9 \
    --epsilon_max 0.01
```
The dataset path defaults to `./data` or can be overridden with the
`CIFAR10_PATH` environment variable.
No CIFAR‑10 checkpoint is included because the file is large, so training from scratch is required or you may provide your own checkpoint.

### LID on CIFAR‑10
Run the local intrinsic dimension experiment with:
```bash
python experiments/cifar10/lid_cifar.py --chunk_size 64 --resume_ckpt=/path/to/checkpoint.pt --output_dir results_lid_merged --num_samples_test 1024 --num_samples_select 64 "$@"
```

### ImageNet32 training
Train an ImageNet32 model with:
```bash
torchrun --nproc_per_node=8 experiments/imagenet32/train_imagenet_multigpu.py \
    --lr 6e-4 --batch_size 128
```
The dataset path defaults to `./data/imagenet32` or can be overridden with the
`IMAGENET32_PATH` environment variable.

### ImageNet32 FID evaluation
Evaluate the model using the multi-GPU FID script:
```bash
torchrun --nproc_per_node=2 experiments/imagenet32/fid_imagenet_heun_multigpu.py \
    --resume_ckpt='/path/to/checkpoint.pt' \
    --output_dir=./sampling_results \
    --use_ema True \
    --time_cutoff 0.9 \
    --epsilon_max 0.01 \
    --batch_size 128 \
    --dt_gibbs 0.005
```
