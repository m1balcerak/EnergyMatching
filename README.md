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

The repository contains `HelloWorld.ipynb` which now includes a toy example of energy matching for quick experimentation.
