# Training & Evaluation

Canonical commands for training low-level policies in this fork and evaluating them on hardware. Hardware-specific teleop/recording lives in the per-robot docs ([aloha.md](aloha.md), [droid.md](droid.md), [franka_leap.md](franka_leap.md)).

## Low-level policy (Diffusion Policy)

### Non-goal-conditioned baseline

```bash
python lerobot/scripts/train.py \
    --dataset.repo_id=<repo_id> \
    --policy.type=diffusion \
    --output_dir=outputs/train/<run_name> \
    --job_name=<run_name> \
    --policy.device=cuda \
    --wandb.enable=true \
    --policy.use_separate_rgb_encoder_per_camera=true \
    --policy.use_text_embedding=true \
    --policy.crop_shape="[600, 600]" \
    --policy.crop_is_random=false \
    --dataset.image_transforms.enable=true
```

### Goal-conditioned (heatmap subgoal)

Add `--policy.enable_goal_conditioning=true` and drop the text embedding:

```bash
python lerobot/scripts/train.py \
    --dataset.repo_id=<gc_repo_id> \
    --policy.type=diffusion \
    --output_dir=outputs/train/<run_name> \
    --job_name=<run_name> \
    --policy.device=cuda \
    --wandb.enable=true \
    --policy.use_separate_rgb_encoder_per_camera=true \
    --policy.use_text_embedding=false \
    --policy.crop_shape="[600, 600]" \
    --policy.crop_is_random=false \
    --dataset.image_transforms.enable=true \
    --policy.enable_goal_conditioning=true
```

### Multi-task

Pass a list of dataset repo IDs:

```bash
python lerobot/scripts/train.py \
    --dataset.repo_id='["<repo_a>", "<repo_b>", "<repo_c>"]' \
    --policy.type=diffusion \
    --output_dir=outputs/train/<run_name> \
    --job_name=<run_name> \
    --wandb.enable=true \
    --policy.use_text_embedding=true \
    --steps=300_000 \
    --policy.crop_shape="[600, 600]" \
    --policy.crop_is_random=false
```

### Multi-view

With multiple fixed cameras, use a larger random crop and select the image feature keys explicitly:

```bash
python lerobot/scripts/train.py \
    --dataset.repo_id='["<repo_a>", "<repo_b>"]' \
    --policy.type=diffusion \
    --output_dir=outputs/train/<run_name> \
    --job_name=<run_name> \
    --wandb.enable=true \
    --policy.use_text_embedding=true \
    --steps=300_000 \
    --policy.crop_shape="[700, 700]" \
    --policy.crop_is_random=true \
    --policy.input_image_feature_keys='["observation.images.cam_azure_kinect_back.color", "observation.images.cam_azure_kinect_front.color", "observation.images.cam_wrist"]' \
    --batch_size=4
```

### Useful flags

- `--policy.horizon` / `--policy.n_action_steps` — increase for high-frequency data
- `--policy.crop_shape` / `--policy.crop_is_random` — center/random crop to avoid workspace edges
- `--policy.use_separate_rgb_encoder_per_camera=true` — separate encoders per camera (useful with heatmap + RGB inputs)
- `--policy.enable_goal_conditioning=true` — enable goal conditioning
- `--policy.use_text_embedding=true` — condition on SigLIP text features
- `--policy.phantomize=true` — train on phantomized (human→robot) images (see [ghost.md](ghost.md))
- `--num_workers` — increase if GPU utilization is low (video decoding parallelism)

DROID/Franka data requires `--policy.robot_type=droid`. To run on a shared cluster, prefix with `HF_HOME=/scratch/$USER/lerobot` and see [cluster.md](cluster.md).

## Evaluation / Rollout

Roll out a trained policy on hardware via `control_robot.py` in `record` mode with `--control.policy.path`:

```bash
python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record \
    --control.fps=15 \
    --control.single_task="Grasp mug and place it on the table." \
    --control.repo_id=<eval_dataset_id> \
    --control.num_episodes=20 \
    --control.reset_time_s=5 --control.warmup_time_s=3 \
    --robot.cameras='{ ... }' \
    --robot.use_eef=true \
    --control.push_to_hub=false \
    --control.policy.path=outputs/train/<run_name>/checkpoints/last/pretrained_model/ \
    --control.display_data=true \
    --control.episode_time_s=120
```

- The camera config must match what the policy was trained on.
- Some config parameters are baked in at training time and cannot be overridden at runtime. To change them, edit `outputs/train/<run_name>/checkpoints/last/pretrained_model/config.json` directly.
- For LIBERO sim evaluation see [libero.md](libero.md).
