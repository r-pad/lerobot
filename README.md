# LeRobot (r-pad fork)

**Fork of [huggingface/lerobot](https://github.com/huggingface/lerobot) maintained by [r-pad](https://r-pad.github.io) (CMU)**

---

## Overview

This is a **research fork** of [HuggingFace LeRobot](https://github.com/huggingface/lerobot) focused on goal-conditioned manipulation with **Aloha** and **DROID/Franka** hardware. It diverged from upstream around April 2024 and has accumulated substantial additions across hardware drivers, policies, simulation environments, and dataset tooling.

> **Warning:** This fork is **not compatible with upstream** LeRobot. The two should not be used interchangeably.

## What's Different from Upstream

### Hardware
- **DROID / Franka Panda** robot support (via [Deoxys](https://github.com/UT-Austin-RPL/deoxys_control))
- **Franka + LEAP hand** — 16-DOF dexterous hand support
- **Azure Kinect** camera driver (RGB + transformed depth)
- **Intel RealSense** camera support
- **ZED** camera support
- **GELLO** teleoperation controller
- **Robotiq** gripper integration

### Policies
- **Hierarchical goal-conditioned policies**: MimicPlay, DINO-3DGP, ViT-3DGP, ArticuBot
- **DP3** (3D Diffusion Policy)
- Goal conditioning with heatmap-based subgoal prediction
- Text-conditioned policies (SigLIP embeddings)

### Environments
- **LIBERO** simulation environment integration

### Dataset Tools
- `merge_datasets.py` — merge multiple LeRobotDatasets
- `subsample_dataset.py` — downsample to a target fps
- `upgrade_dataset.py` — add new features, phantomize/humanize demos
- `annotate_events.py` — annotate ground-truth events
- `create_droid_dataset.py` — convert DROID data to LeRobot format
- `create_libero_dataset.py` — convert LIBERO data to LeRobot format

### Other
- Forward kinematics / EEF pose storage in datasets
- **Phantom** human-to-robot video retargeting pipeline
- Relative action spaces

## Installation

```bash
git clone git@github.com:r-pad/lerobot.git
cd lerobot
pixi install
```

Install optional dependencies as needed:

```bash
pixi run install-pytorch3d    # PyTorch3D (required for 3D policies)
pixi run install-pynput       # pynput (keyboard control during teleop)
pixi run install-open3d       # Open3D (point cloud visualization)
pixi run install-k4a          # Azure Kinect (pyk4a, built from source for numpy 2)
pixi run install-gello        # GELLO teleoperation controller
pixi run install-deoxys       # Deoxys (Franka/DROID control)
pixi run install-zed          # ZED camera SDK
```

### Weights & Biases

```bash
wandb login
```

Then enable WandB during training with `--wandb.enable=true`.

## Usage

### Teleoperation

```bash
python lerobot/scripts/control_robot.py \
    --robot.type=aloha \
    --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}}' \
    --control.type=teleoperate \
    --control.display_data=true
```

Per-robot teleop, camera configs, and calibration are documented in the hardware docs: [Aloha](docs/aloha.md), [DROID / Franka](docs/droid.md), [Franka + LEAP](docs/franka_leap.md).

### Recording Episodes

```bash
python lerobot/scripts/control_robot.py \
    --robot.type=aloha \
    --control.type=record \
    --control.single_task="Grasp mug and place it on the table." \
    --control.repo_id=<your_hf_user>/<dataset_name> \
    --control.num_episodes=100 \
    --robot.cameras='{ ... }' \
    --robot.use_eef=true \
    --control.push_to_hub=true \
    --control.fps=30 \
    --control.reset_time_s=5 \
    --control.warmup_time_s=3 \
    --control.resume=true
```

Key flags:
- `--control.repo_id` — dataset name (and HuggingFace repo if pushing to hub)
- `--control.resume` — append to an existing dataset
- `--robot.use_eef=true` — run forward kinematics and store EEF pose
- Use left/right arrow keys to finish / reset the current episode

### Training & Evaluation

```bash
python lerobot/scripts/train.py \
    --dataset.repo_id=<repo_id> \
    --policy.type=diffusion \
    --output_dir=outputs/train/<run_name> \
    --job_name=<run_name> \
    --policy.device=cuda \
    --wandb.enable=true
```

See [docs/training.md](docs/training.md) for goal-conditioned, multi-task, multi-view, high-level, and evaluation/rollout commands.

## Documentation

- **Hardware** — [Aloha](docs/aloha.md), [DROID / Franka](docs/droid.md), [Franka + LEAP](docs/franka_leap.md)
- **[Training & Evaluation](docs/training.md)** — policy training and on-robot rollout
- **[Human Demonstrations (GHOST)](docs/ghost.md)** — processing human videos into training data
- **[Dataset Utilities](docs/dataset_utilities.md)** — visualize, merge, subsample, upgrade, annotate
- **[LIBERO](docs/libero.md)** — simulation dataset creation, training, and evaluation
- **[Cluster (Orchard)](docs/cluster.md)** — running on the CMU SLURM cluster

## Projects Using This Repo

- **GHOST: Hierarchical Sub-Goal Policies for Generalizing Robot Manipulation** (RSS 2026) — [project page](https://ghost-human-demo.github.io/)

## Citation

If you use this codebase, please cite the original LeRobot project:

```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```

If you use any specific policy architecture, please also cite the original authors (ACT, Diffusion Policy, DP3, MimicPlay, etc.).

## Acknowledgments

This fork builds on top of [HuggingFace LeRobot](https://github.com/huggingface/lerobot). We thank the original authors for their excellent open-source robotics framework.
