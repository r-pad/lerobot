# LIBERO

[LIBERO](https://libero-project.github.io/) simulation environment integration for this fork.

## Creating Datasets

Convert LIBERO source demos into the LeRobot format with `create_libero_dataset.py`:

```bash
# Single task
python lerobot/scripts/create_libero_dataset.py \
    --hdf5_list libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5

# Full suites
python lerobot/scripts/create_libero_dataset.py \
    --suite_names libero_object libero_goal libero_spatial libero_90 libero_10
```

## Training

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

## Evaluation

Evaluate across a full suite with `eval_suite.py`:

```bash
python lerobot/scripts/eval_suite.py \
    --policy.path=<checkpoint_path> \
    --env.type=libero \
    --suite_name=libero_90 \
    --task_ids=0,1,2,3,4,5,6,7,8,9 \
    --eval.batch_size=10 \
    --eval.n_episodes=20
```
