"""Read and print joint positions from both GELLO and Franka in a loop.

Usage:
    python docs/read_joints.py --config lerobot/common/robot_devices/robots/franka_configs/charmander_droid.yml

Use this to verify that GELLO joint offsets are correct. When the GELLO and
Franka are physically in the same pose, their printed joint values should match.
If they don't, adjust gello_joint_offsets and gello_joint_signs in lerobot/common/robot_devices/robots/configs.py.
"""

import argparse
import glob
import time

import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Read GELLO and Franka joint positions.")
    parser.add_argument("--config", type=str, required=True, help="Path to deoxys YAML config file")
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

    # --- GELLO setup (uses defaults from DroidRobotConfig), modify as necessary ---
    gello_joint_ids: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
    gello_joint_offsets: tuple[float, ...] = (
        3 * 3.141592653589793 / 2,
        0 * 3.141592653589793 / 2,
        4 * 3.141592653589793 / 2,
        2 * 3.141592653589793 / 2,
        2 * 3.141592653589793 / 2,
        2 * 3.141592653589793 / 2,
        0 * 3.141592653589793 / 2,
    )
    gello_joint_signs: tuple[int, ...] = (1, 1, 1, 1, 1, -1, 1)
    gello_gripper_joint_id: int = 8
    gello_gripper_open_degrees: int = 195
    gello_gripper_close_degrees: int = 152

    matches = [p for p in glob.glob("/dev/serial/by-id/*") if "Serial_Converter" in p]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one GELLO serial device, found {matches}.")
    port = matches[0]
    gello = DynamixelRobot(
        joint_ids=list(gello_joint_ids),
        joint_offsets=list(gello_joint_offsets),
        real=True,
        joint_signs=list(gello_joint_signs),
        port=port,
        gripper_config=(
            gello_gripper_joint_id,
            gello_gripper_open_degrees,
            gello_gripper_close_degrees,
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
