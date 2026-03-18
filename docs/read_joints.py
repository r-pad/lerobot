"""Read and print joint positions from both GELLO and Franka in a loop.

Usage:
    python docs/read_joints.py --config lerobot/common/robot_devices/robots/franka_configs/charmander_droid.yml

Use this to verify that GELLO joint offsets are correct. When the GELLO and
Franka are physically in the same pose, their printed joint values should match.
If they don't, adjust gello_joint_offsets and gello_joint_signs in DroidRobotConfig.
"""

import argparse
import glob
import time

import numpy as np

DEFAULT_CONFIG = "lerobot/common/robot_devices/robots/franka_configs/charmander_droid.yml"


def main():
    parser = argparse.ArgumentParser(description="Read GELLO and Franka joint positions.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to deoxys YAML config file")
    args = parser.parse_args()

    from deoxys.franka_interface import FrankaInterface
    from gello.robots.dynamixel import DynamixelRobot

    # --- Franka setup ---
    interface = FrankaInterface(
        args.config,
        use_visualizer=False,
    )
    print("Waiting for Franka state buffer...")
    while len(interface._state_buffer) == 0:
        time.sleep(0.1)
    print("Franka connected.")

    # --- GELLO setup (uses defaults from DroidRobotConfig) ---
    from lerobot.common.robot_devices.robots.configs import DroidRobotConfig

    cfg = DroidRobotConfig()
    port = glob.glob("/dev/serial/by-id/*")[0]
    gello = DynamixelRobot(
        joint_ids=list(cfg.gello_joint_ids),
        joint_offsets=list(cfg.gello_joint_offsets),
        real=True,
        joint_signs=list(cfg.gello_joint_signs),
        port=port,
        gripper_config=(
            cfg.gello_gripper_joint_id,
            cfg.gello_gripper_open_degrees,
            cfg.gello_gripper_close_degrees,
        ),
    )
    print("GELLO connected.")

    # --- Print loop ---
    print("\nReading joints (Ctrl+C to stop)...\n")
    try:
        while True:
            franka_q = np.array(interface._state_buffer[-1].q)
            gello_q = np.array(gello.get_joint_state())[:7]
            diff = franka_q - gello_q

            print(f"Franka: {np.array2string(franka_q, precision=4, suppress_small=True)}")
            print(f"GELLO:  {np.array2string(gello_q, precision=4, suppress_small=True)}")
            print(f"Diff:   {np.array2string(diff, precision=4, suppress_small=True)}")
            print(f"Max diff: {np.max(np.abs(diff)):.4f} rad")
            print("-" * 70)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nDone.")


if __name__ == "__main__":
    main()
