import os
import pickle

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def _to_homogeneous(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def _split_pose(T):
    return T[:3, :3], T[:3, 3]


def _se3_log(T):
    """Map SE(3) -> R^6 (rotation vector + translation in tangent space at identity)."""
    rotvec = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    return np.concatenate([rotvec, T[:3, 3]])


def _se3_exp(xi):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(xi[:3]).as_matrix()
    T[:3, 3] = xi[3:]
    return T


def _poses_to_rt(poses):
    R_list, t_list = [], []
    for pose in poses:
        R, t = _split_pose(pose)
        R_list.append(R)
        t_list.append(t.reshape(3, 1))
    return R_list, t_list


def estimate_tag_pose(gripper_pose, tag_to_gripper=None):
    """
    Estimate the tag pose in the robot base frame using a known tag-to-gripper transform.

    Args:
        gripper_pose: 4x4 transform from gripper to robot base (T^B_G).
        tag_to_gripper: optional 4x4 transform from gripper to tag (T^G_T). Defaults to identity.
    Returns:
        (None, tag_pose): tag pose in base frame (T^B_T = T^B_G @ T^G_T).
    """
    if tag_to_gripper is None:
        tag_to_gripper = np.eye(4)
    tag_pose = gripper_pose @ tag_to_gripper
    return None, tag_pose


def solve_rigid_transformation(inpts, outpts):
    """
    Takes in two sets of corresponding points, returns the rigid transformation matrix from the first to the second.
    """
    assert inpts.shape == outpts.shape
    inpts, outpts = np.copy(inpts), np.copy(outpts)
    inpt_mean = inpts.mean(axis=0)
    outpt_mean = outpts.mean(axis=0)
    outpts -= outpt_mean
    inpts -= inpt_mean
    X = inpts.T
    Y = outpts.T
    covariance = np.dot(X, Y.T)
    U, _, Vt = np.linalg.svd(covariance)
    V = Vt.T
    idmatrix = np.identity(3)
    idmatrix[2, 2] = np.linalg.det(V @ U.T)
    R = V @ idmatrix @ U.T
    t = outpt_mean - R @ inpt_mean
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def solve_extrinsic_known_tag_offset(gripper_poses, target_poses_in_camera, tag_to_gripper=None):
    """
    Solve camera-to-base using known tag-to-gripper and point correspondences (SVD).
    """
    tag_poses = [estimate_tag_pose(pose, tag_to_gripper)[1] for pose in gripper_poses]
    tag_pos = np.array([pose[:3, 3] for pose in tag_poses])
    target_pos = np.array([pose[:3, 3] for pose in target_poses_in_camera])
    T_base_from_camera = solve_rigid_transformation(target_pos, tag_pos)
    return T_base_from_camera, tag_to_gripper if tag_to_gripper is not None else np.eye(4)


def solve_extrinsic_hand_eye(
    gripper_poses,
    target_poses_in_camera,
    tag_to_gripper_init=None,
    refine=True,
    method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI,
):
    """
    Solve camera-to-base (T^B_C) and tag-to-gripper (T^G_T) jointly for eye-to-hand calibration.

    Uses the constraint at each pose i:
        T^B_{G,i} @ T^G_T = T^B_C @ T^C_{T,i}

    Args:
        gripper_poses: list of 4x4 transforms from gripper to robot base (T^B_G).
        target_poses_in_camera: list of 4x4 transforms from tag to camera (T^C_T).
        tag_to_gripper_init: optional 4x4 initial guess for T^G_T used only for nonlinear refinement.
        refine: whether to run nonlinear refinement after the closed-form OpenCV solve.
        method: OpenCV RobotWorldHandEyeCalibrationMethod.

    Returns:
        T_base_from_camera: 4x4 transform from camera to robot base (T^B_C).
        T_gripper_from_tag: 4x4 transform from gripper to tag (T^G_T).
    """
    if len(gripper_poses) < 3:
        raise ValueError(f"Need at least 3 poses for hand-eye calibration, got {len(gripper_poses)}")

    R_target_in_camera, t_target_in_camera = _poses_to_rt(target_poses_in_camera)
    # OpenCV names this R_base2gripper, but expects gripper-to-base (T^B_G) in practice.
    R_gripper_in_base, t_gripper_in_base = _poses_to_rt(gripper_poses)

    R_base_to_tag, t_base_to_tag, R_gripper_to_camera, t_gripper_to_camera = cv2.calibrateRobotWorldHandEye(
        R_target_in_camera,
        t_target_in_camera,
        R_gripper_in_base,
        t_gripper_in_base,
        method=method,
    )

    T_tag_from_gripper = _to_homogeneous(R_base_to_tag, t_base_to_tag)
    T_gripper_from_tag = np.linalg.inv(T_tag_from_gripper)
    T_gripper_to_camera = _to_homogeneous(R_gripper_to_camera, t_gripper_to_camera)
    T_base_from_camera = np.linalg.inv(T_gripper_to_camera)

    if refine:
        T_base_from_camera, T_gripper_from_tag = refine_hand_eye_calibration(
            gripper_poses,
            target_poses_in_camera,
            T_base_from_camera,
            T_gripper_from_tag if tag_to_gripper_init is None else tag_to_gripper_init,
        )

    return T_base_from_camera, T_gripper_from_tag


def refine_hand_eye_calibration(
    gripper_poses,
    target_poses_in_camera,
    T_base_from_camera_init,
    T_gripper_from_tag_init,
):
    """Nonlinear refinement of T^B_C and T^G_T using full SE(3) residuals."""
    xi_base_from_camera = _se3_log(T_base_from_camera_init)
    xi_gripper_from_tag = _se3_log(T_gripper_from_tag_init)

    def residuals(xi):
        T_base_from_camera = _se3_exp(xi[:6])
        T_gripper_from_tag = _se3_exp(xi[6:])
        errors = []
        for T_gripper_in_base, T_tag_in_camera in zip(gripper_poses, target_poses_in_camera):
            T_predicted_tag_in_base = T_gripper_in_base @ T_gripper_from_tag
            T_measured_tag_in_base = T_base_from_camera @ T_tag_in_camera
            T_error = np.linalg.inv(T_measured_tag_in_base) @ T_predicted_tag_in_base
            errors.append(_se3_log(T_error))
        return np.concatenate(errors)

    result = least_squares(
        residuals,
        np.concatenate([xi_base_from_camera, xi_gripper_from_tag]),
        method="lm",
    )
    T_base_from_camera = _se3_exp(result.x[:6])
    T_gripper_from_tag = _se3_exp(result.x[6:])
    return T_base_from_camera, T_gripper_from_tag


def calculate_pose_consistency_error(gripper_poses, target_poses_in_camera, T_base_from_camera, T_gripper_from_tag):
    """
    Compute per-sample SE(3) consistency errors for T^B_G @ T^G_T = T^B_C @ T^C_T.
    """
    errors = []
    for T_gripper_in_base, T_tag_in_camera in zip(gripper_poses, target_poses_in_camera):
        T_predicted = T_gripper_in_base @ T_gripper_from_tag
        T_measured = T_base_from_camera @ T_tag_in_camera
        T_error = np.linalg.inv(T_measured) @ T_predicted
        errors.append(np.linalg.norm(_se3_log(T_error)))
    return np.array(errors)


def load_calibration_results(calib_path):
    """Load camera-to-base and optional tag-to-gripper transforms from an .npz file."""
    data = np.load(calib_path)
    T_base_from_camera = data["T"]
    T_gripper_from_tag = data["T_gripper_from_tag"] if "T_gripper_from_tag" in data else None
    return T_base_from_camera, T_gripper_from_tag


def calculate_reprojection_error(tag_poses, target_poses, T_matrix):
    """Position-only reprojection error (legacy helper)."""
    errors = []
    for tag_pose, target_pose in zip(tag_poses, target_poses):
        transformed_target = T_matrix @ target_pose
        tag_pos = tag_pose[:3, 3]
        transformed_pos = transformed_target[:3, 3]
        errors.append(np.linalg.norm(tag_pos - transformed_pos))
    return np.mean(errors)


def solve_extrinsic(
    gripper_poses,
    target_poses_in_camera,
    eye_to_hand=True,
    unknown_tag_offset=True,
    tag_to_gripper=None,
    tag_to_gripper_init=None,
    refine=True,
):
    """
    Solve camera-to-base extrinsic calibration.

    Args:
        unknown_tag_offset: if True, jointly estimate tag-to-gripper and camera-to-base.
            if False, use known tag_to_gripper (or identity) with point-only SVD.
    """
    if eye_to_hand and unknown_tag_offset:
        T_base_from_camera, T_gripper_from_tag = solve_extrinsic_hand_eye(
            gripper_poses,
            target_poses_in_camera,
            tag_to_gripper_init=tag_to_gripper_init,
            refine=refine,
        )
        pose_errors = calculate_pose_consistency_error(
            gripper_poses, target_poses_in_camera, T_base_from_camera, T_gripper_from_tag
        )
        print(f"Transformation matrix T (base from camera):\n{T_base_from_camera}")
        print(f"Tag-to-gripper transform T^G_T:\n{T_gripper_from_tag}")
        print(
            "Pose consistency error (m, rad equiv): "
            f"mean={pose_errors.mean():.6f}, max={pose_errors.max():.6f}, std={pose_errors.std():.6f}"
        )
        return T_base_from_camera, T_gripper_from_tag

    tag_to_gripper = tag_to_gripper if tag_to_gripper is not None else np.eye(4)
    T_base_from_camera, _ = solve_extrinsic_known_tag_offset(
        gripper_poses, target_poses_in_camera, tag_to_gripper
    )
    tag_poses = [estimate_tag_pose(pose, tag_to_gripper)[1] for pose in gripper_poses]
    print(f"Transformation matrix T (base from camera):\n{T_base_from_camera}")
    avg_error = calculate_reprojection_error(tag_poses, target_poses_in_camera, T_base_from_camera)
    print(f"Average position reprojection error: {avg_error}")
    return T_base_from_camera


if __name__ == "__main__":
    cam_name = "cam_azure_kinect_front"
    # cam_name = "cam_azure_kinect_back"
    data_dirname = "/data/yufei/lerobot/data/calibration"
    data_filepath = os.path.join(data_dirname, f"cam{cam_name}_data.pkl")
    with open(data_filepath, "rb") as f:
        data = pickle.load(f)
    gripper_poses, target_poses_in_camera = zip(*data)

    if cam_name == 'cam_azure_kinect_back':
        tag_to_gripper = np.array(
            [
                [ 0.01588481, -0.39287454, -0.91945488, -0.11363858],
                [ 0.99985234, 0.01226962, 0.01203109, -0.00169885],
                [ 0.00655465, -0.91951023, 0.39301143, 0.10392223],
                [ 0.          , 0.          , 0.          , 1.          ]
            ]
        )   
    else:
        tag_to_gripper = np.array(
            [
                [ 0.03466124, -0.36009281, -0.93227237, -0.12392002],
                [ 0.99923518, 0.02938247, 0.02580181, -0.00538151],
                [ 0.01810142, -0.93245367, 0.36083584, 0.10930598],
                [ 0.          , 0.          , 0.          , 1.          ]
            ]
        )

    T_base_from_camera, T_gripper_from_tag = solve_extrinsic(
        gripper_poses,
        target_poses_in_camera,
        unknown_tag_offset=True,
        tag_to_gripper_init=tag_to_gripper,
    )

    calib_dirname = os.path.join(data_dirname, "calibration_results")
    os.makedirs(calib_dirname, exist_ok=True)
    filepath = os.path.join(calib_dirname, f"cam{cam_name}_calibration.npz")
    np.savez(
        filepath,
        T=T_base_from_camera,
        T_gripper_from_tag=T_gripper_from_tag,
    )
    print(f"Saved calibration to {filepath}")
