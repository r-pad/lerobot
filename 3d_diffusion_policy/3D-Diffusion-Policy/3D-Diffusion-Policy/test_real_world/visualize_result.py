import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

original_data = "test_pcd_microwave_1"

for i in range(3):
    data_path = f"/project_data/held/ziyuw2/Robogen-sim2real/local_exps/{original_data}/result_pointnet_{i}.pkl"
    with open(data_path, "rb") as f:
        data = pkl.load(f)
    pcd = np.array(data["pcd"])
    gripper_pcd = np.array(data["gripper_pcd"])
    if len(gripper_pcd.shape) == 3:
        gripper_pcd = gripper_pcd[0]
    agent_pos = np.array(data["agent_pos"])
    predicted_goal = np.array(data["predicted_goal"])

    pcd = np.concatenate([pcd, gripper_pcd], axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c='b', marker='o', s=1)
    ax.scatter(predicted_goal[:, 0], predicted_goal[:, 1], predicted_goal[:, 2], c='r', marker='o')

    # data_path = f"/project_data/held/ziyuw2/Robogen-sim2real/local_exps/test_pcd_microwave_0/result_200_{i}.pkl"
    # with open(data_path, "rb") as f:
    #     data = pkl.load(f)  
    # pcd = np.array(data["pcd"])
    # gripper_pcd = np.array(data["gripper_pcd"])
    # agent_pos = np.array(data["agent_pos"])
    # predicted_goal = np.array(data["predicted_goal"])

    # pcd = np.concatenate([pcd, gripper_pcd], axis=0)
    # ax = fig.add_subplot(122, projection='3d')
    # ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c='b', marker='o', s=1)
    # ax.scatter(predicted_goal[:, 0], predicted_goal[:, 1], predicted_goal[:, 2], c='r', marker='o')

    plt.show()
