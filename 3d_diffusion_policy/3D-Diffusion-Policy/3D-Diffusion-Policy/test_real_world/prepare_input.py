import pickle
import numpy as np
import matplotlib.pyplot as plt
# import open3d as o3d
import fpsample


# original_data = "test_pcd_microwave_2"

# pcd_path = f"/project_data/held/yufeiw2/RoboGen_sim2real/data/{original_data}.pkl"
# with open(pcd_path, "rb") as f:
#     pcd = pickle.load(f)


original_data = "temp"
input_path = "/project_data/held/ziyuw2/Robogen-sim2real/data/parallel_input_dict.pkl"
with open(input_path, "rb") as f:
    input_dict = pickle.load(f)
pcd = np.array(input_dict["high_level_point_cloud"][0])

# min_z = np.min(pcd[:, 2])
# # crop the pcd
# pcd = pcd[pcd[:, 2] > min_z + 0.047]


# # min_y = np.min(pcd[:, 1])
# # pcd = pcd[pcd[:, 1] > min_y + 0.08]

# max_y = np.max(pcd[:, 1])
# pcd = pcd[pcd[:, 1] < max_y - 0.15]



x_range = np.max(pcd[:, 0]) - np.min(pcd[:, 0])
y_range = np.max(pcd[:, 1]) - np.min(pcd[:, 1])
x_min = np.min(pcd[:, 0])
y_min = np.min(pcd[:, 1])
x_max = np.max(pcd[:, 0])
y_max = np.max(pcd[:, 1])
z_min = np.min(pcd[:, 2])
z_max = np.max(pcd[:, 2])

print(f"x_range: {x_range}, y_range: {y_range}")
print(f"x_min: {x_min}, x_max: {x_max}")
print(f"y_min: {y_min}, y_max: {y_max}")
print(f"z_min: {z_min}, z_max: {z_max}")

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], s=0.5)
# plt.show()
# import pdb; pdb.set_trace()

import open3d as o3d

# ================= for fridge =================
# # make pcd twice bigger
# pcd[:, 0] = (pcd[:, 0] - x_min) * 2
# pcd[:, 1] = (pcd[:, 1] - y_min) * 2
# pcd[:, 2] = (pcd[:, 2] - z_min) * 2

# pcd[:, 0] = pcd[:, 0] + 0.65
# pcd[:, 1] = pcd[:, 1] - 0.4
# pcd[:, 2] = pcd[:, 2] + 0.0

# ================= for microwave =================
# pcd = pcd[pcd[:, 2] < 0.26]

# pcd[:, 0] = pcd[:, 0] - x_min
# pcd[:, 1] = pcd[:, 1] - y_min
# pcd[:, 2] = pcd[:, 2] - z_min

# pcd[:, 0] = pcd[:, 0] + 0.7
# pcd[:, 1] = pcd[:, 1] - 0.4
# pcd[:, 2] = pcd[:, 2] + 0.3

# # remove the outliers of pcd
# pointcloud = o3d.geometry.PointCloud()
# pointcloud.points = o3d.utility.Vector3dVector(pcd[:, :3])
# cl, ind = pointcloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
# pcd = np.asarray(pointcloud.points)



# kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(pcd[:, :3], 4500, h=1, start_idx=0)
# pcd = pcd[kdline_fps_samples_idx]

agent_pos_1 = np.array([ 0.60897565,  0.05900308,  0.55199987, -0.103153  , -0.5509229 , -0.82815665,  0.92665505,  0.24935651, -0.28130358,  0.04      ])
gripper_pcd_1 = np.array([[0.58547926, 0.11077122, 0.5204883 ],
        [0.64569294, 0.09265602, 0.5250392 ],
        [0.55302745, 0.06772038, 0.55316955],
        [0.60897565, 0.05900308, 0.55199987]])

agent_pos_2 = np.array([ 0.4694055 , -0.08446881,  0.4990294 , -0.2915407 , -0.7561545 ,
         0.5858621 ,  0.87238276,  0.04104537,  0.4870971 ,  0.04      ])
gripper_pcd_2 = np.array([[ 0.4949094 , -0.12692061,  0.45692956],
        [ 0.52346164, -0.09978913,  0.5061557 ],
        [ 0.43622333, -0.10389367,  0.457446  ],
        [ 0.4694055 , -0.08446881,  0.4990294 ]])

agent_pos_3 = np.array([ 0.4818811 ,  0.19097114,  0.2687416 , -0.90904826, -0.03154957,
        -0.41549474,  0.06629866,  0.9734763 , -0.21897133,  0.04      ])
gripper_pcd_3 = np.array([[0.45514122, 0.20570028, 0.32612655],
        [0.47425324, 0.24567257, 0.2812767 ],
        [0.46762338, 0.14832495, 0.30317384],
        [0.4818811 , 0.19097114, 0.2687416 ]])

agent_pos = np.array([agent_pos_1, agent_pos_2, agent_pos_3])
gripper_pcd = np.array([gripper_pcd_1, gripper_pcd_2, gripper_pcd_3])

for i in range(3):
    # save pcd and gripper_pcd
    save_path = f"/project_data/held/ziyuw2/Robogen-sim2real/local_exps/{original_data}/{i}.pkl"
    save_data = {
        "pcd": pcd,
        "gripper_pcd": gripper_pcd[i], 
        "agent_pos": agent_pos[i]
    }
    with open(save_path, "wb") as f:
        pickle.dump(save_data, f)

    # pcd = np.concatenate([pcd, gripper_pcd[0]], axis=0)

    # # show pcd in plot 
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], s=0.5)
    # plt.show()

# one sample in simulation data 
# x: 0.7 ~ 1.3
# y: -0.4 ~ 0.4
# z: 0.0 ~ 0.6



# pcd = np.concatenate([pcd, gripper_pcd[0]], axis=0)

# # show pcd in plot 
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], s=0.5)
# plt.show()