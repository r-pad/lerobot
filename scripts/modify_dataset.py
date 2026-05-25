import numpy as np
import os

traj_path = "/data/yufei/lerobot/data/articubot_format/pap_three_subsampled_0520_v2_with_goal_rgb_dino_1_crop_table/traj_0015"

release_blue_idx = 350
grasp_purple_idx = 450
release_purple_idx = 505

grasp_purple_path = os.path.join(traj_path, f"{grasp_purple_idx}.npz")
with open(grasp_purple_path, "rb") as f:
    grasp_purple_data = np.load(f)
    grasp_purple_gripper_pcd = grasp_purple_data["gripper_pcd"]

release_purple_path = os.path.join(traj_path, f"{release_purple_idx}.npz")
with open(release_purple_path, "rb") as f:
    release_purple_data = np.load(f)
    release_purple_gripper_pcd = release_purple_data["gripper_pcd"]

for idx in range(release_blue_idx + 1, grasp_purple_idx):
    step_path = os.path.join(traj_path, str(idx) + ".npz")
    with np.load(step_path, allow_pickle=False) as step_data:
        data = {key: step_data[key] for key in step_data.files}
    data["goal_gripper_pcd"] = np.asarray(grasp_purple_gripper_pcd).copy()
    np.savez_compressed(step_path, **data)

for idx in range(grasp_purple_idx + 1, release_purple_idx):
    step_path = os.path.join(traj_path, str(idx) + ".npz")
    with np.load(step_path, allow_pickle=False) as step_data:
        data = {key: step_data[key] for key in step_data.files}
    data["goal_gripper_pcd"] = np.asarray(release_purple_gripper_pcd).copy()
    np.savez_compressed(step_path, **data)