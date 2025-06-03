import mink
from robot_descriptions.loaders.mujoco import load_robot_description
import numpy as np
import pytorch3d.transforms as transforms
from tqdm import tqdm
from torch import pi
import torch

ALOHA_MODEL = load_robot_description("aloha_mj_description")
ALOHA_CONFIGURATION = mink.Configuration(ALOHA_MODEL)

def map_real2sim(Q):
    """
    The real robot joints and the sim robot don't map exactly to each other.
    Some joints are offset, some joints rotate the opposite direction.
    This mapping converts real robot joint angles (in radians) to the sim version.

    sim = real*sign + offset
    """
    # Set gripper fingers to 0, we don't care about them for IK
    sign = torch.tensor([-1, -1, 1, 1, 1, 1, 0, 0,    
                      -1, -1, 1, 1, 1, 1, 0, 0])
    offset = torch.tensor([pi/2, 0, -pi/2, 0, 0, 0, 0, 0,
                       pi/2, 0, -pi/2, 0, 0, 0, 0, 0])
    Q = sign*Q + offset

    # We handle the shoulder joint separately, x*-1 + np.pi/2 brings it close but just outside joint limits for some reason....
    # Remap this joint range using real observed min/max and sim min/max
    real_min, real_max = -3.59, -0.23
    sim_min, sim_max = -1.85, 1.26 
    Q[1] = (Q[1] - real_min)*((sim_max-sim_min)/(real_max-real_min)) + sim_min
    Q[9] = (Q[9] - real_min)*((sim_max-sim_min)/(real_max-real_min)) + sim_min

    return Q

def forward_kinematics(aloha_configuration, real_joints):
    # FK for the Aloha
    # Mapping from LeRobot (real) to robot_descriptions (sim)
    # Check LeRobot joint names in lerobot/common/robot_devices/robots/configs.py
    # Check robot_descriptions joint names with `print([model.joint(i).name for i in range(model.njnt)])`
    Q = torch.deg2rad(torch.tensor(
        [
            real_joints[0],
            real_joints[1],
            real_joints[3],
            real_joints[5],
            real_joints[6],
            real_joints[7],
            0,
            0,

            real_joints[9],
            real_joints[10],
            real_joints[12],
            real_joints[14],
            real_joints[15],
            real_joints[16],
            0,
            0,
        ]
    ))
    Q = map_real2sim(Q)
    aloha_configuration.update(Q)
    eef_pose_se3 = aloha_configuration.get_transform_frame_to_world("right/gripper", "site")
    rot_6d, trans = transforms.matrix_to_rotation_6d(torch.from_numpy(eef_pose_se3.as_matrix()[None, :3, :3])).squeeze(), torch.from_numpy(eef_pose_se3.as_matrix()[:3,3])
    eef_pose = torch.cat([rot_6d, trans], axis=0)
    return eef_pose, eef_pose_se3

def map_sim2real(vec):
    """
    inverse of map_real2sim
    sim = real*sign + offset
    real = (sim - offset)*sign
    """
    # Set gripper fingers to 0, we don't care about them for IK
    sign = torch.tensor([-1, -1, -1, 1, 1, 1, 1, 1, 0,    
                      -1, -1, -1, 1, 1, 1, 1, 1, 0])
    offset = torch.tensor([pi/2, 0, 0, -pi/2, -pi/2, 0, 0, 0, 0,
                       pi/2, 0, 0, -pi/2, -pi/2, 0, 0, 0, 0])
    vec = (vec - offset)*sign

    # Inverted from real2sim
    real_min, real_max = 0.23, 3.59 
    sim_min, sim_max = -1.26, 1.85 

    vec[1] = (vec[1] - sim_min)*((real_max-real_min)/(sim_max-sim_min)) + real_min
    vec[2] = (vec[2] - sim_min)*((real_max-real_min)/(sim_max-sim_min)) + real_min
    vec[10] = (vec[10] - sim_min)*((real_max-real_min)/(sim_max-sim_min)) + real_min
    vec[11] = (vec[11] - sim_min)*((real_max-real_min)/(sim_max-sim_min)) + real_min
    return vec


def inverse_kinematics(configuration, ee_pose):
    rot_6d, trans, articulation = ee_pose[:6], ee_pose[6:9], ee_pose[9]
    pose_matrix = torch.eye(4, device=ee_pose.device)
    pose_matrix[:3,3] = trans
    pose_matrix[:3,:3] = transforms.rotation_6d_to_matrix(torch.tensor(rot_6d)[None]).squeeze()
    ee_pose_se3 = mink.lie.se3.SE3.from_matrix(pose_matrix.cpu().numpy())
    
    ee_task = mink.FrameTask(frame_name="right/gripper", frame_type="site", position_cost=1., orientation_cost=1.)
    ee_task.set_target(ee_pose_se3)
    n_iter = 200
    dt = 0.01
    thresh = 1e-3
    for i in range(n_iter):
        vel = mink.solve_ik(configuration, [ee_task], dt=dt, solver='daqp')
        configuration.integrate_inplace(vel, dt)

        err = ee_task.compute_error(configuration)
        print(i, np.linalg.norm(err))
        if np.linalg.norm(err) < thresh: break
    
    Q = configuration.q
    vec = torch.tensor(
        [
            Q[0],
            Q[1],
            Q[1],
            Q[2],
            Q[2],
            Q[3],
            Q[4],
            Q[5],
            0,

            Q[8],
            Q[9],
            Q[9],
            Q[10],
            Q[10],
            Q[11],
            Q[12],
            Q[13],
            0,
        ]
    )
    vec = torch.rad2deg(map_sim2real(vec))
    vec[-1] = articulation
    return vec
