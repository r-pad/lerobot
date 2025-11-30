#!/bin/bash

set -e

echo "--------------------------------------------"
echo " Installing NEW isolated Miniconda for lerobot"
echo "--------------------------------------------"

# Install path
CONDA_PATH="$HOME/miniconda-lerobot"

# Remove any existing partial installs
rm -rf $CONDA_PATH

# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

# Install non-interactively
bash miniconda.sh -b -p $CONDA_PATH
rm miniconda.sh

# Initialize conda
source $CONDA_PATH/bin/activate

echo "--------------------------------------------"
echo " Creating fresh conda environment: lerobot"
echo "--------------------------------------------"

conda create -y -n lerobot python=3.10
conda activate lerobot

echo "--------------------------------------------"
echo " Installing PyTorch 2.1.0 + CUDA 11.8"
echo "--------------------------------------------"

conda install -y pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia

echo "--------------------------------------------"
echo " Verifying PyTorch installation"
echo "--------------------------------------------"

python - << 'EOF'
import torch
print("Torch version:", torch.__version__)
print("CUDA runtime version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
EOF

echo "--------------------------------------------"
echo " Installing PyTorch3D (matching CUDA 11.8 + PyTorch 2.1)"
echo "--------------------------------------------"

pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt21/download.html

echo "--------------------------------------------"
echo " Verifying PyTorch3D installation"
echo "--------------------------------------------"

python - << 'EOF'
import pytorch3d
import pytorch3d.transforms as T
print("PyTorch3D installed correctly!")
print("Transforms module:", T)
EOF

echo "--------------------------------------------"
echo " SUCCESS!"
echo " New environment located at: $CONDA_PATH"
echo " Use this environment with:"
echo "     source $CONDA_PATH/bin/activate"
echo "     conda activate lerobot"
echo "--------------------------------------------"
