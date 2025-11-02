import mink
from robot_descriptions.loaders.mujoco import load_robot_description
import numpy as np
import pytorch3d.transforms as transforms
from torch import pi
import torch
import mujoco
import trimesh
from scipy.spatial.transform import Rotation as R
import cv2


ALOHA_MODEL = load_robot_description("aloha_mj_description")
ALOHA_CONFIGURATION = mink.Configuration(ALOHA_MODEL)
ALOHA_REST_STATE = torch.tensor([[ 91.9336, 191.8652, 191.7773, 174.2871, 174.3750,   5.4492,  17.4023, -2.5488,  11.5245,  
                            92.1094, 193.5352, 193.0078, 169.6289, 169.6289, -3.7793,  21.0059,   2.2852, 100.7582]])
ALOHA_REST_QPOS = np.array(
    [0, -1.73, 1.49, 0, 0, 0, 0, 0, 0, -1.73, 1.49, 0, 0, 0, 0, 0]
)


def map_real2sim(Q):
    """
    The real robot joints and the sim robot don't map exactly to each other.
    Some joints are offset, some joints rotate the opposite direction.
    This mapping converts real robot joint angles (in radians) to the sim version.

    sim = real*sign + offset
    """
    sign = torch.tensor([-1, -1, 1, 1, 1, 1, 1, 1,    
                      -1, -1, 1, 1, 1, 1, 1, 1])
    offset = torch.tensor([pi/2, 0, -pi/2, 0, 0, 0, 0, 0,
                       pi/2, 0, -pi/2, 0, 0, 0, 0, 0])
    Q = sign*Q + offset

    # We handle the shoulder joint separately, x*-1 + np.pi/2 brings it close but just outside joint limits for some reason....
    # Remap this joint range using real observed min/max and sim min/max
    real_shoulder_min, real_shoulder_max = -3.59, -0.23
    sim_shoulder_min, sim_shoulder_max = -1.85, 1.26 
    Q[1] = (Q[1] - real_shoulder_min)*((sim_shoulder_max-sim_shoulder_min)/(real_shoulder_max-real_shoulder_min)) + sim_shoulder_min
    Q[9] = (Q[9] - real_shoulder_min)*((sim_shoulder_max-sim_shoulder_min)/(real_shoulder_max-real_shoulder_min)) + sim_shoulder_min

    # same for gripper
    real_gripper_min, real_gripper_max = -0.11, 1.7262
    sim_gripper_min, sim_gripper_max = 0, 0.041
    Q[6] = (Q[6] - real_gripper_min)*((sim_gripper_max-sim_gripper_min)/(real_gripper_max-real_gripper_min)) + sim_gripper_min
    Q[7] = (Q[7] - real_gripper_min)*((sim_gripper_max-sim_gripper_min)/(real_gripper_max-real_gripper_min)) + sim_gripper_min
    Q[14] = (Q[14] - real_gripper_min)*((sim_gripper_max-sim_gripper_min)/(real_gripper_max-real_gripper_min)) + sim_gripper_min
    Q[15] = (Q[15] - real_gripper_min)*((sim_gripper_max-sim_gripper_min)/(real_gripper_max-real_gripper_min)) + sim_gripper_min

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
            real_joints[8],
            real_joints[8],

            real_joints[9],
            real_joints[10],
            real_joints[12],
            real_joints[14],
            real_joints[15],
            real_joints[16],
            real_joints[17],
            real_joints[17],
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
    sign = torch.tensor([-1, -1, -1, 1, 1, 1, 1, 1, 1,    
                      -1, -1, -1, 1, 1, 1, 1, 1, 1])
    offset = torch.tensor([pi/2, 0, 0, -pi/2, -pi/2, 0, 0, 0, 0,
                       pi/2, 0, 0, -pi/2, -pi/2, 0, 0, 0, 0])
    vec = (vec - offset)*sign

    # Inverted from real2sim
    real_shoulder_min, real_shoulder_max = 0.23, 3.59 
    sim_shoulder_min, sim_shoulder_max = -1.26, 1.85 

    vec[1] = (vec[1] - sim_shoulder_min)*((real_shoulder_max-real_shoulder_min)/(sim_shoulder_max-sim_shoulder_min)) + real_shoulder_min
    vec[2] = (vec[2] - sim_shoulder_min)*((real_shoulder_max-real_shoulder_min)/(sim_shoulder_max-sim_shoulder_min)) + real_shoulder_min
    vec[10] = (vec[10] - sim_shoulder_min)*((real_shoulder_max-real_shoulder_min)/(sim_shoulder_max-sim_shoulder_min)) + real_shoulder_min
    vec[11] = (vec[11] - sim_shoulder_min)*((real_shoulder_max-real_shoulder_min)/(sim_shoulder_max-sim_shoulder_min)) + real_shoulder_min

    real_gripper_min, real_gripper_max = -1.7262, 0.11
    sim_gripper_min, sim_gripper_max = -0.041, 0
    vec[8] = (vec[8] - sim_gripper_min)*((real_gripper_max-real_gripper_min)/(sim_gripper_max-sim_gripper_min)) + real_gripper_min
    vec[17] = (vec[17] - sim_gripper_min)*((real_gripper_max-real_gripper_min)/(sim_gripper_max-sim_gripper_min)) + real_gripper_min
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
    vec = torch.rad2deg(map_sim2real(vec)) # IK ignoring gripper
    vec[-1] = articulation # overwrite with predicted gripper
    return vec


def get_rest_aloha_mesh(sample_n_points=500):
    """
    Aloha mesh when resting, rest eef pos, rest eef rot
    """

    site = "right/gripper"
    site_id = mujoco.mj_name2id(ALOHA_MODEL, mujoco.mjtObj.mjOBJ_SITE, site)

    data = mujoco.MjData(ALOHA_MODEL)
    data.qpos = ALOHA_REST_QPOS

    mujoco.mj_forward(ALOHA_MODEL, data)

    rest_pos = data.site_xpos[site_id].copy()
    rest_rot = data.site_xmat[site_id].reshape(3, 3).copy()

    meshes = get_right_gripper_mesh(ALOHA_MODEL, data)
    mesh = combine_meshes(meshes)
    mesh_ = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles)
    )
    rest_mesh, _ = trimesh.sample.sample_surface(mesh_, sample_n_points, seed=42)

    return rest_pos, rest_rot, rest_mesh


def retarget_aloha_gripper_pcd(cam_to_world, eef_data, sample_n_points=500):
    """
    For goal gripper pcd, retarget gripper instead of ik
    """
    gripper_rot = transforms.rotation_6d_to_matrix(eef_data[:6])
    gripper_eef_pos = eef_data[6:9] # world frame

    rest_pos, rest_rot, rest_mesh = get_rest_aloha_mesh(sample_n_points)
    
    H_rest_to_world = np.eye(4)
    H_rest_to_world[:3,:3] = gripper_rot @ rest_rot.T
    H_rest_to_world[:3,3] = gripper_eef_pos - (gripper_rot @ rest_rot.T) @ rest_pos

    world_to_cam = np.linalg.inv(cam_to_world)
    homo_rest_mesh = np.concatenate(
        [rest_mesh, np.ones((sample_n_points, 1))], axis=-1
    )[:, :, None]
    
    home_retarget_mesh = H_rest_to_world @ homo_rest_mesh
    
    urdf_cam3dcoords = (world_to_cam @ home_retarget_mesh)[:, :3].squeeze(2)

    return urdf_cam3dcoords.astype(np.float32)


def render_aloha_gripper_pcd(cam_to_world, joint_state, sample_n_points=500):
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
    return urdf_cam3dcoords.astype(np.float32)

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
    import open3d as o3d # Local import to avoid cluster install issues
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
    import open3d as o3d # Local import to avoid cluster install issues
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


def setup_renderer(model, intrinsics_txt, extrinsics_txt, downsample_factor, width, height):
    """
    Setup mujoco renderer for Aloha.
    Re-use the teleoperator_pov camera and configure as required
    Downsample the rendered image because we run into opengl framebuffer issues otherwise
    """
    cam_to_world = np.loadtxt(extrinsics_txt)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "teleoperator_pov")
    K = np.loadtxt(intrinsics_txt)
    width, height = int(width * downsample_factor), int(height * downsample_factor)
    K[0,0] *= downsample_factor
    K[0,2] *= downsample_factor
    K[1,1] *= downsample_factor
    K[1,2] *= downsample_factor
    setup_camera(model, cam_id, cam_to_world, width, height, K)
    renderer = mujoco.Renderer(model, width=width, height=height)
    return renderer

def setup_camera(model, cam_id, cam_to_world, width, height, K):
    """
    Configure camera to match real world setup. """
    model.cam_pos[cam_id] = cam_to_world[:3, 3]
    R_flip = np.diag([1, -1, -1])
    R_cam = R.from_matrix(cam_to_world[:3, :3] @ R_flip)
    cam_quat = R_cam.as_quat()  # [x, y, z, w]
    cam_quat = cam_quat[[3, 0, 1, 2]]  # Reorder to [w, x, y, z] for MuJoCo
    model.cam_quat[cam_id] = cam_quat
    fovy = np.degrees(2 * np.arctan((height / 2) / K[1, 1]))
    model.cam_fovy[cam_id] = fovy


def render_rightArm_images(renderer, data, camera="teleoperator_pov", use_seg=False):
    """
    Render RGB, depth, and segmentation images with right arm masking from MuJoCo simulation.
    Args:
        renderer (mujoco.Renderer): MuJoCo renderer instance configured for the scene
        data (mujoco.MjData): MuJoCo data object containing current simulation state
        camera (str, optional): Name of the camera to render from.

    Returns:
        tuple: A tuple containing:
            - rgb (np.ndarray): Masked RGB image of shape (H, W, 3), dtype uint8.
                               Right arm pixels retain original colors, background pixels are black.
            - depth (np.ndarray): Masked depth image of shape (H, W), dtype float32.
                                 Right arm pixels contain depth values, background pixels are zero.
            - seg (np.ndarray): Binary segmentation mask of shape (H, W), dtype bool.
                               True for right arm pixels, False for background.
    """
    renderer.update_scene(data, camera=camera)
    rgb = renderer.render()

    # Depth rendering
    renderer.enable_depth_rendering()
    depth = renderer.render()
    renderer.disable_depth_rendering()

    # Segmentation rendering
    renderer.enable_segmentation_rendering()
    seg = renderer.render()
    renderer.disable_segmentation_rendering()

    seg = seg[:, :, 0]  # channel 1 is foreground/background
    # NOTE: Classes for the right arm excluding the camera mount. Handpicked
    target_classes = set(range(65, 91)) - {81, 82, 83}

    seg = np.isin(seg, list(target_classes)).astype(bool)
    if use_seg:
        rgb[~seg] = 0
        depth[~seg] = 0

    return rgb, depth, seg


def render_and_overlay(renderer, ALOHA_MODEL, joint_state, real_rgb, downsample_factor):
    """
    Place sim-aloha in the same location as the real one
    Render rgb/depth/seg and then overlay the rendered robot on the real one
    """
    upsample_factor = 1/downsample_factor
    Q = convert_real_joints(joint_state)
    data = mujoco.MjData(ALOHA_MODEL)
    data.qpos = Q
    mujoco.mj_forward(ALOHA_MODEL, data)

    rgb, depth, seg = render_rightArm_images(renderer, data)
    rgb_ = cv2.resize(rgb, (0,0), fx=upsample_factor, fy=upsample_factor)
    seg_ = cv2.resize(seg.astype(np.uint8), (0,0), fx=upsample_factor, fy=upsample_factor).astype(bool)
    
    real_rgb[seg_] = rgb_[seg_]
    return real_rgb
