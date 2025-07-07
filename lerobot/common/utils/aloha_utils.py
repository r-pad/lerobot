import mink
from robot_descriptions.loaders.mujoco import load_robot_description
import numpy as np
import pytorch3d.transforms as transforms
from torch import pi
import torch
import mujoco
import open3d as o3d
import trimesh


ALOHA_MODEL = load_robot_description("aloha_mj_description")
ALOHA_CONFIGURATION = mink.Configuration(ALOHA_MODEL)
ALOHA_REST_STATE = torch.tensor([[ 91.9336, 191.8652, 191.7773, 174.2871, 174.3750,   5.4492,  17.4023, -2.5488,  11.5245,  
                            92.1094, 193.5352, 193.0078, 169.6289, 169.6289, -3.7793,  21.0059,   2.2852, 100.7582]])

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

def convert_real_joints(real_joints):
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
    return Q

def forward_kinematics(aloha_configuration, real_joints):
    # FK for the Aloha
    Q = convert_real_joints(real_joints)
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


def render_gripper_pcd(cam_to_world, joint_state, sample_n_points=500):
    """
    Run FK, extract gripper pcd from robot URDF, transform to camera frame
    """
    data = mujoco.MjData(ALOHA_MODEL)
    Q = convert_real_joints(joint_state)
    data.qpos = Q
    mujoco.mj_forward(ALOHA_MODEL, data)

    world_to_cam = np.linalg.inv(cam_to_world)
    meshes = get_right_gripper_mesh(ALOHA_MODEL, data)
    mesh = combine_meshes(meshes)
    mesh_ = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles)
    )
    points_, _ = trimesh.sample.sample_surface(mesh_, sample_n_points, seed=42)

    gripper_urdf_3d_pos = np.concatenate(
        [points_, np.ones((sample_n_points, 1))], axis=-1
    )[:, :, None]
    urdf_cam3dcoords = (world_to_cam @ gripper_urdf_3d_pos)[:, :3].squeeze(2)
    return urdf_cam3dcoords

def get_right_gripper_mesh(mj_model, mj_data):
    """
    Extract the visual meshes of the right gripper from the Aloha MJCF model.

    Args:
        mj_model: MuJoCo model object.
        mj_data: MuJoCo data object containing the current simulation state.

    Returns:
        List of open3d.geometry.TriangleMesh objects representing the right gripper's visual meshes
        in world coordinates.
    """
    meshes = []

    # Define the bodies that make up the right gripper
    right_gripper_body_names = [
        "right/gripper_base",
        "right/left_finger_link",
        "right/right_finger_link",
    ]
    exclude_mesh_ids = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_MESH, "d405_solid"),
        mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_MESH, "vx300s_7_gripper_wrist_mount"
        ),
    ]

    # Get body IDs for the right gripper components
    right_gripper_body_ids = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in right_gripper_body_names
    ]

    # Iterate over all geoms in the model
    for geom_id in range(mj_model.ngeom):
        # Check if the geom belongs to the right gripper, is visual (group 2), and is a mesh, and is not camera
        if (
            mj_model.geom_bodyid[geom_id] in right_gripper_body_ids
            and mj_model.geom_group[geom_id] == 2
            and mj_model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
            and mj_model.geom_dataid[geom_id] not in exclude_mesh_ids
        ):
            geom_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)

            # Get the geom's world position and orientation from the simulation state
            geom_pos = mj_data.geom_xpos[geom_id]  # 3D position in world coordinates
            geom_mat = mj_data.geom_xmat[geom_id].reshape(3, 3)  # 3x3 rotation matrix

            # Get the mesh ID associated with this geom
            mesh_id = mj_model.geom_dataid[geom_id]
            if mesh_id >= 0:  # Ensure the geom has a valid mesh
                # Extract mesh vertex and face data
                mesh_vert_adr = mj_model.mesh_vertadr[
                    mesh_id
                ]  # Start index of vertices
                mesh_vert_num = mj_model.mesh_vertnum[mesh_id]  # Number of vertices
                mesh_face_adr = mj_model.mesh_faceadr[mesh_id]  # Start index of faces
                mesh_face_num = mj_model.mesh_facenum[mesh_id]  # Number of faces

                vertices_local = mj_model.mesh_vert[
                    mesh_vert_adr : mesh_vert_adr + mesh_vert_num
                ].copy()
                faces = mj_model.mesh_face[
                    mesh_face_adr : mesh_face_adr + mesh_face_num
                ].copy()

                # Transform local vertices to world coordinates
                vertices_world = vertices_local @ geom_mat.T + geom_pos

                # Create an Open3D mesh
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertices_world)
                mesh.triangles = o3d.utility.Vector3iVector(faces)
                meshes.append(mesh)

    return meshes

def combine_meshes(meshes):
    """
    Combine multiple Open3D meshes into a single mesh with proper vertex and triangle indexing.

    Args:
        meshes: List of Open3D triangle meshes

    Returns:
        combined_mesh: A single Open3D triangle mesh
    """
    # Initialize vertices and triangles lists
    vertices = []
    triangles = []
    vertex_offset = 0

    # Combine meshes
    for mesh in meshes:
        # Convert mesh vertices and triangles to numpy arrays
        mesh_vertices = np.asarray(mesh.vertices)
        mesh_triangles = np.asarray(mesh.triangles)

        # Add vertices to the combined list
        vertices.append(mesh_vertices)

        # Adjust triangle indices and add to the combined list
        adjusted_triangles = mesh_triangles + vertex_offset
        triangles.append(adjusted_triangles)

        # Update vertex offset for the next mesh
        vertex_offset += len(mesh_vertices)

    # Concatenate all vertices and triangles
    all_vertices = np.vstack(vertices)
    all_triangles = np.vstack(triangles)

    # Create the combined mesh
    combined_mesh = o3d.geometry.TriangleMesh()
    combined_mesh.vertices = o3d.utility.Vector3dVector(all_vertices)
    combined_mesh.triangles = o3d.utility.Vector3iVector(all_triangles)

    combined_mesh.compute_vertex_normals()

    combined_mesh.remove_duplicated_vertices()
    combined_mesh.remove_duplicated_triangles()
    combined_mesh.remove_degenerate_triangles()

    return combined_mesh