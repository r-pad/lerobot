"""
NOTE: we use deoxys for controlling our franka arm, you can feel free to switch it to a different robot controller. 
https://github.com/UT-Austin-RPL/deoxys_control
"""

import os
import time
import numpy as np
from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.ik_utils import IKWrapper
import pybullet as p
from scipy.spatial.transform import Rotation as R
from termcolor import cprint
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.utils.log_utils import get_project_logger

try:
    from manipulation.panda import Panda
    from gym.utils import seeding
except ImportError:
    Panda = None
    seeding = None

# Way to call the controller
# ### NOTE: command the robot to move.
#         ### the controller should expect a target eef position, target eef orientation, and a target gripper width
#         if self.robot_controller_type == 'joint':
#             cprint(f"Delta pos norm {np.linalg.norm(delta_pos)} Delta rotation norm {np.linalg.norm(new_delta_axis_angle)}", "green")
#             if self.binary_grasping and self.first_grasp_t == self.time_step:
#                 res = self.robot_controller.control(target_pos=new_pos, target_rot=after_rotate_matrix, grasping_action=grasping_action, only_grasping=True)
#             else:
#                 res = self.robot_controller.control(target_pos=new_pos, target_rot=after_rotate_matrix, grasping_action=grasping_action)

logger = get_project_logger()

class PyBulletIKWrapper():
    def __init__(self, urdf_path: str | None = None, right_eef_idx: int = 11):
        self.id = p.connect(p.DIRECT)
        if urdf_path is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
            urdf_path = os.path.join(repo_root, "assets", "panda_real_world.urdf")
        self.urdf_path = urdf_path
        self.panda = p.loadURDF(
            self.urdf_path,
            useFixedBase=True,
            basePosition=[0, 0, 0],
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=self.id,
        )
        
        self.all_joint_indices = list(range(p.getNumJoints(self.panda, physicsClientId=self.id)))
        self.ik_lower_limits = []
        self.ik_upper_limits = []
        for j in self.all_joint_indices:
            joint_info = p.getJointInfo(self.panda, j, physicsClientId=self.id)
            joint_name = joint_info[1]
            joint_type = joint_info[2]
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            if joint_type != p.JOINT_FIXED:
                self.ik_lower_limits.append(lower_limit)
                self.ik_upper_limits.append(upper_limit)

        self.ik_lower_limits = np.array(self.ik_lower_limits)
        self.ik_upper_limits = np.array(self.ik_upper_limits)
        joint_range = self.ik_upper_limits - self.ik_lower_limits
        self.ik_lower_limits += 0.1 * joint_range
        self.ik_upper_limits -= 0.1 * joint_range

        self.right_eef_idx = right_eef_idx

    def inverse_kinematics(self, target_world_orient, target_world_position, last_q):
        ik_indices = [0, 1, 2, 3, 4, 5, 6]

        for idx in ik_indices:
            p.resetJointState(self.panda, idx, last_q[idx], physicsClientId=self.id)

        try_times = 25
        original_joint_angles = np.array(last_q)
        all_possible_solutions = []
        for try_idx in range(try_times):
            if try_idx > 0: 
                disturbed_joint_angles = original_joint_angles[ik_indices] + np.random.uniform(-0.3, 0.3, size=len(ik_indices))
                disturbed_joint_angles = np.clip(disturbed_joint_angles, self.ik_lower_limits[ik_indices], self.ik_upper_limits[ik_indices])
            else:
                disturbed_joint_angles = original_joint_angles[ik_indices]

            for idx in ik_indices:
                p.resetJointState(self.panda, idx, disturbed_joint_angles[idx], physicsClientId=self.id)

            # print("targetOrientation: ", target_world_orient)
            new_joint_angle = p.calculateInverseKinematics(
                self.panda,
                self.right_eef_idx, 
                targetPosition=target_world_position, 
                targetOrientation=target_world_orient, 
                maxNumIterations=100000, 
                residualThreshold=1e-5,
                physicsClientId=self.id
            )            

            new_joint_angle = new_joint_angle[:7]

            if np.all(new_joint_angle >= self.ik_lower_limits[ik_indices]) and np.all(new_joint_angle <= self.ik_upper_limits[ik_indices]):
                all_possible_solutions.append(new_joint_angle)


        return all_possible_solutions
    
    def forward_kinematics(self, q):
        for idx in range(len(q)):
            p.resetJointState(self.panda, idx, q[idx], physicsClientId=self.id)

        eef_pos, eef_orient = p.getLinkState(self.panda, self.right_eef_idx, physicsClientId=self.id)[0:2]
        
        return eef_pos, eef_orient


class PybulletTrackIKWrapper():
    def __init__(self):
        if Panda is None or seeding is None:
            raise ImportError(
                "PybulletTrackIKWrapper requires the mentor manipulation/gym dependencies. "
                "Use PyBulletIKWrapper for the local URDF IK path."
            )
        self.id = p.connect(p.DIRECT)
        self.robot_class = Panda
        self.robot = self.robot_class(slider=False)
        self.asset_dir = "/data/robogen/RoboGen-sim2real/manipulation/assets"
        self.np_random, _ = seeding.np_random()
        self.robot.init(self.asset_dir, self.id, self.np_random, fixed_base=True, use_suction=False)

        self.ik_indices = [0, 1, 2, 3, 4, 5, 6]

    def inverse_kinematics(self, target_world_orient, target_world_position, last_q):
        agent = self.robot

        original_joint_angles = np.array(last_q)
        for idx in self.ik_indices:
            p.resetJointState(self.robot.body, idx, last_q[idx])
        tracIK_solutions = agent.ik_tracik_franka(target_world_position, target_world_orient, self.ik_indices)

        return tracIK_solutions



class RobotIKController():
    def __init__(
            self,
            visualizer: bool = False,
            interface_cfg: str = "charmander.yml",
            impedance_control=False,
            use_bullet=False,
            binary_grasping=False,
            robot_interface=None,
            controller_cfg=None,
            controller_type: str | None = None,
    ):
        if robot_interface is None:
            self.robot_interface = FrankaInterface(
                        config_root + f"/{interface_cfg}",
                        control_freq=20,
                        use_visualizer=visualizer,
                        automatic_gripper_reset=False)
        else:
            self.robot_interface = robot_interface

        if controller_cfg is not None and controller_type is not None:
            self.controller_cfg = controller_cfg
            self.controller_type = controller_type
            cprint(f"Using provided {self.controller_type} controller", "green")
        elif not impedance_control:
            cprint("Using joint position controller", "green")
            self.controller_cfg = YamlConfig(os.path.join(config_root, "joint-position-controller.yml")).as_easydict()
            self.controller_type = "JOINT_POSITION"
        else:
            cprint("Using joint impedance controller", "green")
            self.controller_cfg = YamlConfig(os.path.join(config_root, "joint-impedance-controller.yml")).as_easydict()
            self.controller_type = "JOINT_IMPEDANCE"
        self.impedance_control = impedance_control

        self.ik_wrapper = IKWrapper()
        self.bullet_ik_wrapper = PyBulletIKWrapper()
        # self.bullet_tracIK_wrapper = PybulletTrackIKWrapper()

        self.use_bullet = use_bullet
        self.binary_grasping = binary_grasping

        self.not_minimized = False
        self.last_joint_target = None
        self.last_action = None
        self.last_control_success = False

        self.reset_joint_positions = [
            -0.5493463,
            0.18639661,
            0.04967389,
            -1.92004654,
            -0.01182675,
            2.10698001,
            0.27106661]

    def control(self, target_pos, target_rot, grasping_action, only_grasping=False, wait_times=50, joint_threshold=0.5):

        while self.robot_interface.state_buffer_size == 0:
            time.sleep(0.1)

        if not only_grasping:

            last_q = np.array(self.robot_interface.last_q)
            target_rot_quat = R.from_matrix(target_rot).as_quat()
            bullet_joints = self.bullet_ik_wrapper.inverse_kinematics(target_rot_quat, target_pos, last_q.tolist())

            original_joint_angles = np.array(last_q)
            all_possible_solutions = bullet_joints

            if len(all_possible_solutions) > 0:
                all_possible_solutions = np.array(all_possible_solutions).reshape(-1, 7)

                distance_to_cur_angle = np.linalg.norm(all_possible_solutions - original_joint_angles.reshape(1, -1), axis=1)
                min_idx = np.argmin(distance_to_cur_angle)
                best_joint_angles = all_possible_solutions[min_idx]

                joint = best_joint_angles
                self.not_minimized = False
                cprint(
                    "PyBullet IK selected joint target, distance_to_current={:.6f}".format(
                        float(distance_to_cur_angle[min_idx])
                    ),
                    "green",
                )
            else:
                cprint("No valid PyBullet joint solution found", 'red')
                joint = self.robot_interface.last_q
        else:
            joint = self.robot_interface.last_q

        ### compute the robot finger movement
        if self.binary_grasping:
            if not grasping_action:
                action = joint.tolist() + [-1.0]
            else:
                action = joint.tolist() + [1.0]
        else:
            action = joint.tolist() + [grasping_action]
        self.last_joint_target = np.asarray(joint, dtype=np.float64)
        self.last_action = np.asarray(action, dtype=np.float64)

        ### execute the robot action till it reaches the joint angle or exceeds a time limit
        t = 0
        while True:

            self.robot_interface.control(
                controller_type=self.controller_type,
                action=action,
                controller_cfg=self.controller_cfg,
                binary_grasping=self.binary_grasping,
            )
            joint_delta = np.array(self.robot_interface._state_buffer[-1].q) - np.array(joint)
            # print("joint_delta", np.round(joint_delta, 3))
            if (
                np.max(
                    np.abs(
                        np.array(self.robot_interface._state_buffer[-1].q)
                        - np.array(joint)
                    )
                )
                < 2e-3 
            ):
                cprint("Joint diff minimized: {}".format(np.linalg.norm(joint - self.robot_interface.last_q)), 'green')
                self.not_minimized = False
                self.last_control_success = True
                return True

            if t >= wait_times:
                cprint("Joint diff not minimized after {} steps: {}".format(wait_times, np.linalg.norm(joint - self.robot_interface.last_q)), 'red')
                self.not_minimized = True
                self.last_control_success = False
                return False

            t += 1

    @property
    def eef_pose(self):
        # pose of the gripper tip
        last_eef_pose = self.robot_interface.last_eef_pose
        return last_eef_pose
    
    @property
    def eef_base_pose(self):
        # pose of the gripper tip base
        last_eef_pose = self.robot_interface.last_eef_pose
        return last_eef_pose
    
    @property
    def eef_rot_and_pos(self):
        pose = self.eef_pose
        rot, pos = pose[:3, :3], pose[:3, 3:]
        return rot, pos

    @property
    def joint_positions(self):
        return self.robot_interface.last_q
    
    @property
    def is_grasped(self):
        gripper_q = self.robot_interface.last_gripper_q
        last_gripper_action = self.robot_interface.last_gripper_action
        # return (gripper_q > 0.01) and (last_gripper_action >= 0.0)
        return (last_gripper_action >= 0.0)
    

    def reset(self, joint_positions=None, **kwargs):
        joint_positions = joint_positions if joint_positions is not None else self.reset_joint_positions
        reset_joints_to(self.robot_interface, joint_positions, gripper_open=True, impedance=self.impedance_control)
        while self.robot_interface.state_buffer_size == 0:
            logger.warn("Robot state not received")
            time.sleep(0.5)

if __name__ == "__main__":
    robot_controller = RobotIKController(
        visualizer=False,
        impedance_control=False,
        use_bullet=True,
        binary_grasping=False,
    )

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, default=0)
    args = parser.parse_args()

    # orient, pos = robot_controller.eef_rot_and_pos
    # pos = pos.flatten()
    # robot_controller.control(target_pos=pos, target_rot=orient, grasping_action=0.02)

    reset_joints = {
        "high": [-1.4742674741296573, -0.36589019469359607, -0.03307992528795409, -2.6484214394607144, 1.480606553680367, 1.8101963592063728, 0.25680279963390107],
        "low": [-1.2128674596496416, 0.08782837270146263, 0.18413104419657292, -2.7950245061120027, 2.0446445442503616, 2.2405066723474603, -0.5691467293280373],
        "tabletop": [-1.0875715552596874, -0.5290805282592773, 0.021696630244976596, -2.7972190923523486, 0.47477878718502975, 2.372381427679463, -0.6324331577326928],
        "paper_low": [-1.8065071721521402, 0.10268306770973037, 0.0646789007634473, -2.865263553058171, 1.6498028561296285, 2.307841544409731, -0.605628443530028,], 
        # "paper_low_2": [-1.959786736103529, -0.023414187760990964, 0.06126829040207361, -2.8760041275347423, 1.414490511543221, 1.8048159386979457, 0.7344229830042663,], 
        # "paper_low_3": [-1.959786736103529, -0.023414187760990964, 0.06126829040207361, -2.8760041275347423, 1.414490511543221, 1.8048159386979457, 0.7344229830042663,], 
        "netural": [-0.8206274549700872, -0.04409709098742481, 0.6051782829051007, -2.8740565410564525, 0.25495421942852553, 3.053153913269703, 0.45680335530904603,],
    }
    
    robot_controller.reset(
        reset_joints[args.name]   
    )
