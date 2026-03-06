from abc import ABC, abstractmethod
from typing import Dict
import torch
import mujoco
from lerobot.common.utils.aloha_utils import ALOHA_CONFIGURATION, ALOHA_REST_STATE, ALOHA_MODEL, convert_real_joints, \
    get_aloha_visual_meshes, combine_meshes, get_aloha_pc_from_cache
import numpy as np

import open3d as o3d
import numpy as np

def dense_surface_samples(mesh: o3d.geometry.TriangleMesh,
                          n_points: int = 200000,
                          method: str = "poisson",   # "uniform" or "poisson"
                          init_factor: int = 5,
                          with_normals: bool = True):
    # Clean-ish mesh helps sampling quality
    mesh = mesh.remove_duplicated_vertices().remove_degenerate_triangles()
    mesh = mesh.remove_duplicated_triangles().remove_non_manifold_edges()
    if with_normals:
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()

    if method == "uniform":
        pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    elif method == "poisson":
        # Good for more even spacing than uniform
        pcd = mesh.sample_points_poisson_disk(
            number_of_points=n_points,
            init_factor=init_factor
        )
    else:
        raise ValueError("method must be 'uniform' or 'poisson'")

    # if with_normals and not pcd.has_normals():
    #     # (usually already has normals if mesh had them)
    #     pcd.estimate_normals()

    return pcd  # open3d.geometry.PointCloud

def sample_mesh_by_spacing(mesh: o3d.geometry.TriangleMesh,
                           spacing: float,
                           method: str = "poisson"):
    # Surface area (Open3D uses triangle areas internally; we can estimate total area)
    area = mesh.get_surface_area()
    n_points = int(np.ceil(area / (spacing ** 2)))  # rough heuristic

    return dense_surface_samples(mesh, n_points=n_points, method=method)

class RobotAdapter(ABC):
    """Abstract interface for robot-specific logic"""

    @abstractmethod
    def get_obs_key(self) -> str:
        pass

    @abstractmethod
    def get_act_key(self) -> str:
        pass

    @abstractmethod
    def transform_action(self, action: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Transform policy output to robot-executable action"""
        pass

    @abstractmethod
    def get_eef_action(self, action: torch.Tensor) -> torch.Tensor:
        """Get eef action info """
        pass

class AlohaAdapter(RobotAdapter):
    def __init__(self, action_space: str):
        assert action_space in ["right_eef", "joint"]
        self.action_space = action_space
        self.config = ALOHA_CONFIGURATION
        self.rest_state = ALOHA_REST_STATE
        with open("data/aloha_visual_pcd_cache.pkl", "rb") as f:
            import pickle
            self.aloha_cache = pickle.load(f)

    def get_obs_key(self) -> str:
        if self.action_space == "right_eef":
            return "observation.right_eef_pose"
        return "observation.state"

    def get_act_key(self) -> str:
        if self.action_space == "right_eef":
            return "action.right_eef_pose"
        return "action"

    def transform_action(self, action: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        from lerobot.common.utils.aloha_utils import forward_kinematics, inverse_kinematics

        if self.action_space == "right_eef":
            # FK on current state
            # forward_kinematics(self.config, state[0])
            forward_kinematics(self.config, state)

            # IK to get joint action
            action_joint = inverse_kinematics(self.config, action)[None].float()
            # Force left arm to rest at predefined pose
            action_joint[:, :9] = self.rest_state[:, :9]
            return action_joint.squeeze()
        return action

    def get_eef_action(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_space == "right_eef":
            return action  # eef action is primary
        # For joint space, return dummy eef
        return torch.zeros(10, device=action.device, dtype=action.dtype)
    

    
    def get_aloha_pcd(self, real_joint_angle):
        data = mujoco.MjData(ALOHA_MODEL)
        # import pdb; pdb.set_trace()
        sim_joint_angle = convert_real_joints(real_joint_angle)
        data.qpos = sim_joint_angle
        all_points = get_aloha_pc_from_cache(
            ALOHA_MODEL,
            data,
            self.aloha_cache,
            return_by_body=False,
        )

        # meshes = get_right_gripper_mesh(ALOHA_MODEL, data)
        # meshes = get_aloha_visual_meshes(ALOHA_MODEL, data)  # whole robot visuals
        # mesh = combine_meshes(meshes)

        # # import open3d as o3d
        # # o3d_mesh = o3d.geometry.TriangleMesh()
        # # o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        # # o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles))
        # # o3d_mesh.compute_vertex_normals()   
        # # ### add a coordinate frame
        # # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
        # # o3d_mesh += coord_frame
        # # o3d.visualization.draw_geometries([o3d_mesh])

        # # all_points = np.asarray(mesh.vertices).astype(np.float32)
        # all_points = dense_surface_samples(o3d_mesh, n_points=20000)
        # all_points = np.asarray(all_points.points).astype(np.float32)

        return all_points

class LiberoFrankaAdapter(RobotAdapter):
    """Libero Franka adapter"""
    def __init__(self, obs_key: str, act_key: str):
        self.obs_key = obs_key
        self.act_key = act_key

    def get_obs_key(self) -> str:
        return self.obs_key

    def get_act_key(self) -> str:
        return self.act_key

    def transform_action(self, action: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return action  # No transformation

    def get_eef_action(self, action: torch.Tensor) -> torch.Tensor:
        return action  # Return same action as auxiliary
