import pickle as pkl 
import numpy as np
import matplotlib.pyplot as plt

file_path = "/project_data/held/ziyuw2/Robogen-sim2real/local_exps/temp/parallel_output_dict.pkl"
with open(file_path, "rb") as f:
    data = pkl.load(f)

pcd = np.array(data["point_cloud"][0])
gripper_pcd = np.array(data["gripper_pcd"][0])
predicted_goal = np.array(data["predicted_goal"])
if len(predicted_goal.shape) == 3:
    predicted_goal = predicted_goal[0]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c='b', marker='o', s=0.5)
ax.scatter(gripper_pcd[:, 0], gripper_pcd[:, 1], gripper_pcd[:, 2], c='b', marker='o', s=3)
ax.scatter(predicted_goal[:, 0], predicted_goal[:, 1], predicted_goal[:, 2], c='r', marker='o', s=3)
plt.show()