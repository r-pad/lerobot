import numpy as np
import os

traj_path = "/data/yufei/lerobot/data/plate_new_rot/traj_0018"
all_np_files = [f for f in os.listdir(traj_path) if f.endswith(".npz")]
traj_len = len(all_np_files)

actions = []
eef_poses = []
for t in range(traj_len):
    np_file = os.path.join(traj_path, f"{t}.npz")
    data = np.load(np_file, allow_pickle=True)
    action = data['action']
    actions.append(action)
    eef_poses.append(data['state'])

delta_finger = [a[0][9] for a in actions]
cur_finger = [e[0][9] for e in eef_poses]

from matplotlib import pyplot as plt

plt.plot(range(traj_len), delta_finger, color='blue', label='delta_finger')
# plt.plot(range(traj_len), cur_finger, color='red', label='cur_finger')
plt.xlabel('Timestep')
plt.ylabel('Finger Value')
plt.show()