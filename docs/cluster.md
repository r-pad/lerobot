# Cluster (Orchard)

Notes for running this fork on the CMU Orchard SLURM cluster.

## pixi cache on shared filesystems

Point pixi's cache and environments at your project space so they aren't recreated per node:

```bash
export PIXI_CACHE_DIR=/project/flame/$USER/.cache/pixi
echo 'detached-environments = "/project/flame/$USER/envs"' > ~/.pixi/config.toml
```

## Install

```bash
git clone git@github.com:r-pad/lerobot.git
cd lerobot
CC=/usr/bin/gcc CXX=/usr/bin/g++ pixi install

# Install pytorch3d inside a SLURM job (can take ~20 min)
./cluster/launch-slurm.py -J install_pytorch3d --gpus 1 install-pytorch3d
```

## Launch a job

```bash
./cluster/launch-slurm.py -J train --gpus 1 --sync-logs $MY_TRAINING_SCRIPT
```

## Multi-GPU training

High-level policies are trained with `torchrun` in the companion repo. A typical multi-GPU launch sets `HF_HOME` to scratch and disables NCCL P2P where needed:

```bash
NCCL_P2P_LEVEL=NVL HF_HOME="/scratch/$USER/lerobot" \
    torchrun --nproc_per_node=8 scripts/train.py \
    model=dino_3dgp dataset=rpadLerobot \
    dataset.repo_id=<repo_id> \
    resources.num_workers=32 \
    resources.gpus=-1
```

See [training.md](training.md) for the full set of training commands.
