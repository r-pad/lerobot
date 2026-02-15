<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="media/lerobot-logo-thumbnail.png">
    <source media="(prefers-color-scheme: light)" srcset="media/lerobot-logo-thumbnail.png">
    <img alt="LeRobot (r-pad fork)" src="media/lerobot-logo-thumbnail.png" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>

<div align="center">

[![Python versions](https://img.shields.io/pypi/pyversions/lerobot)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/r-pad/lerobot/blob/main/LICENSE)

**Fork of [huggingface/lerobot](https://github.com/huggingface/lerobot) maintained by [r-pad](https://r-pad.github.io) (CMU)**

</div>

---

## Overview

This is a **research fork** of [HuggingFace LeRobot](https://github.com/huggingface/lerobot) focused on goal-conditioned manipulation with **Aloha** and **DROID/Franka** hardware. It diverged from upstream around April 2024 and has accumulated substantial additions across hardware drivers, policies, simulation environments, and dataset tooling.

> **Warning:** This fork is **not compatible with upstream** LeRobot. The two should not be used interchangeably.

## What's Different from Upstream

### Hardware
- **DROID / Franka Panda** robot support (via [Deoxys](https://github.com/UT-Austin-RPL/deoxys_control))
- **Azure Kinect** camera driver (RGB + transformed depth)
- **Intel RealSense** camera support
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
- Camera calibration pipeline (ArUco-based)
- Relative action spaces

## Installation

### With pixi (recommended)

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
```

### With conda (alternative)

```bash
git clone git@github.com:r-pad/lerobot.git
cd lerobot
conda create -y -n lerobot python=3.10
conda activate lerobot
conda install ffmpeg -c conda-forge
pip install -e .
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

See [docs/aloha.md](docs/aloha.md) for Aloha-specific camera configs and details.

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
    --control.fps=60 \
    --control.reset_time_s=5 \
    --control.warmup_time_s=3 \
    --control.resume=true
```

Key flags:
- `--control.repo_id` — dataset name (and HuggingFace repo if pushing to hub)
- `--control.resume` — append to an existing dataset
- `--robot.use_eef=true` — run forward kinematics and store EEF pose
- Use left/right arrow keys to finish / reset the current episode

### Visualizing Datasets

```bash
python lerobot/scripts/visualize_dataset.py \
    --repo-id <repo_id> \
    --episode-index 0
```

### Training

```bash
python lerobot/scripts/train.py \
    --dataset.repo_id=<repo_id> \
    --policy.type=diffusion \
    --output_dir=outputs/train/<run_name> \
    --job_name=<run_name> \
    --policy.device=cuda \
    --wandb.enable=true
```

Notable training config options:
- `--policy.horizon` / `--policy.n_action_steps` — higher than default for high-frequency data
- `--policy.crop_shape="[600, 600]"` — center crop to avoid workspace edges
- `--policy.use_separate_rgb_encoder_per_camera=true` — separate encoders (useful with heatmap + RGB inputs)
- `--policy.enable_goal_conditioning=true` — enable goal conditioning
- `--policy.use_text_embedding=true` — condition on SigLIP text features
- `--num_workers` — increase if GPU utilization is low (video decoding benefits from parallelism)

### Evaluation / Rollout

```bash
python lerobot/scripts/control_robot.py \
    --robot.type=aloha \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Grasp mug and place it on the table." \
    --control.repo_id=<eval_dataset_id> \
    --control.num_episodes=1 \
    --robot.cameras='{ ... }' \
    --robot.use_eef=true \
    --control.push_to_hub=false \
    --control.policy.path=outputs/train/<run_name>/checkpoints/last/pretrained_model/ \
    --control.display_data=true \
    --control.episode_time_s=120
```

Some config parameters are saved during training and cannot be overridden at runtime. To change them, modify `outputs/train/<run_name>/checkpoints/last/pretrained_model/config.json` directly.

### LIBERO

Create datasets from LIBERO source demos:

```bash
# Single task
python lerobot/scripts/create_libero_dataset.py \
    --hdf5_list libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5

# Full suites
python lerobot/scripts/create_libero_dataset.py \
    --suite_names libero_object libero_goal libero_spatial libero_90 libero_10
```

Train on LIBERO:

```bash
python lerobot/scripts/train.py \
    --dataset.repo_id=<libero_dataset_id> \
    --policy.type=diffusion \
    --policy.robot_type=libero_franka \
    --env.type=libero \
    --env.task=libero_object_0 \
    --eval.batch_size=10 \
    --policy.use_separate_rgb_encoder_per_camera=true \
    --policy.use_text_embedding=true
```

Evaluate on a full suite:

```bash
python lerobot/scripts/eval_suite.py \
    --policy.path=<checkpoint_path> \
    --env.type=libero \
    --suite_name=libero_90 \
    --task_ids=0,1,2,3,4,5,6,7,8,9 \
    --eval.batch_size=10 \
    --eval.n_episodes=20
```

## Dataset Utilities

### `upgrade_dataset.py`

Create a new dataset from an existing one, adding/modifying features:

```bash
python lerobot/scripts/upgrade_dataset.py \
    --source_repo_id <source_id> \
    --target_repo_id <target_id> \
    --new_features goal_gripper_proj gripper_pcds next_event_idx \
    --push_to_hub
```

Supports `--phantomize` (retarget human demo to robot via [Phantom](https://phantom-human-videos.github.io/)) and `--humanize` (keep human in video). Use `--discard_episodes` to skip problematic episodes.

### `merge_datasets.py`

```bash
python lerobot/scripts/merge_datasets.py \
    --datasets DATASET1_ID DATASET2_ID DATASET3_ID \
    --target_repo_id MERGED_DATASET_ID \
    --push_to_hub
```

All input datasets must have compatible features and fps.

### `subsample_dataset.py`

```bash
python lerobot/scripts/subsample_dataset.py \
    --source_repo_id SOURCE_ID \
    --target_repo_id TARGET_ID \
    --target_fps <fps>
```

## Preparing Human Demonstrations

Pipeline for processing human (non-robot) demonstration videos:

1. **Hand pose estimation** — Process videos with [WiLoR](https://github.com/sriramsk1999/wilor)
2. **Event annotation** — Manually annotate ground-truth events with `lerobot/scripts/annotate_events.py`
3. **Upgrade dataset** — Run `upgrade_dataset.py` with either:
   - `--humanize` to keep the human in the video
   - `--phantomize` to retarget to a robot (requires [GSAM-2](https://github.com/sriramsk1999/Grounded-SAM-2) masks, [E2FGVI](https://github.com/MCG-NKU/E2FGVI) inpainting, and Phantom rendering)

See [docs/aloha.md](docs/aloha.md) for the detailed step-by-step pipeline.

## Cluster (Orchard)

Optional: Configure pixi cache for shared filesystems:

```bash
export PIXI_CACHE_DIR=/project/flame/$USER/.cache/pixi
echo 'detached-environments = "/project/flame/$USER/envs"' > ~/.pixi/config.toml
```

Install:

```bash
git clone git@github.com:r-pad/lerobot.git
cd lerobot
CC=/usr/bin/gcc CXX=/usr/bin/g++ pixi install

# Install pytorch3d inside a SLURM job (can take ~20 min)
./cluster/launch-slurm.py -J install_pytorch3d --gpus 1 install-pytorch3d
```

Launch training:

```bash
./cluster/launch-slurm.py -J train --gpus 1 --sync-logs $MY_TRAINING_SCRIPT
```

## Project Structure

```
.
├── lerobot
│   ├── configs              # Draccus config classes (policy, env, robot)
│   ├── common
│   │   ├── datasets         # LeRobotDataset, video utils, sampling
│   │   ├── envs             # Sim environments (aloha, pusht, xarm, libero)
│   │   ├── policies
│   │   │   ├── act          # ACT policy
│   │   │   ├── diffusion    # Diffusion Policy (with goal conditioning, text, etc.)
│   │   │   ├── dp3          # 3D Diffusion Policy
│   │   │   ├── high_level   # Hierarchical policies (MimicPlay, DINO-3DGP, ViT-3DGP, ArticuBot)
│   │   │   ├── tdmpc        # TD-MPC
│   │   │   └── vqbet        # VQ-BeT
│   │   ├── robot_devices
│   │   │   ├── cameras      # opencv, intelrealsense, azure_kinect
│   │   │   ├── motors       # dynamixel, feetech
│   │   │   └── robots       # manipulator (Aloha), droid (Franka), stretch, lekiwi
│   │   └── utils
│   └── scripts
│       ├── train.py                 # Train a policy
│       ├── eval.py                  # Evaluate a policy in sim
│       ├── eval_suite.py            # Evaluate across a LIBERO suite
│       ├── control_robot.py         # Teleop, record, replay, rollout
│       ├── upgrade_dataset.py       # Add features / phantomize / humanize
│       ├── merge_datasets.py        # Merge multiple datasets
│       ├── subsample_dataset.py     # Subsample to target fps
│       ├── create_droid_dataset.py  # DROID → LeRobot conversion
│       ├── create_libero_dataset.py # LIBERO → LeRobot conversion
│       ├── annotate_events.py       # Annotate GT events
│       └── visualize_dataset.py     # Visualize with rerun
├── third_party
│   ├── gello                # GELLO teleoperation (git submodule)
│   └── deoxys_control       # Deoxys Franka control (git submodule)
├── cluster                  # SLURM launcher for Orchard
├── docs                     # Additional documentation
│   └── aloha.md             # Aloha-specific setup and usage
├── outputs                  # Training logs, checkpoints, videos
└── tests
```

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
