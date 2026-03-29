"""LEAP Hand control node.

Modified from the LEAP Hand API: https://github.com/leap-hand/LEAP_Hand_API

Controls and queries the LEAP Hand via Dynamixel motors. Supports allegro
convention joint angles (0.0 = fully open, positive = closing) and raw
LEAP hand convention (pi = home pose).

Joint layout (16 DOF):
  Index (0-3), Middle (4-7), Ring (8-11), Thumb (12-15)
  Each finger: MCP Side, MCP Forward, PIP, DIP

Recommended query rate: below 90 Hz. Use combined read commands when possible.
"""

import numpy as np

from lerobot.common.robot_devices.robots.leap_hand.dynamixel_client import DynamixelClient
from lerobot.common.robot_devices.robots.leap_hand import leap_hand_utils as lhu


class LeapNode:
    def __init__(self, port: str | None = None):
        """Initialize connection to the LEAP hand.

        Args:
            port: Serial port for the LEAP hand. If None, tries /dev/ttyUSB0
                then /dev/ttyUSB1.
        """
        # PID and current limit parameters.
        # Keep current limit at 350 for lite motors, 550 for full motors.
        self.kP = 600
        self.kI = 0
        self.kD = 200
        self.curr_lim = 350
        self.prev_pos = self.pos = self.curr_pos = lhu.allegro_to_LEAPhand(np.zeros(16))
        self.motors = motors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        if port is not None:
            self.dxl_client = DynamixelClient(motors, port, 4000000)
            self.dxl_client.connect()
        else:
            try:
                self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB0', 4000000)
                self.dxl_client.connect()
            except Exception:
                try:
                    self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB1', 4000000)
                    self.dxl_client.connect()
                except Exception:
                    self.dxl_client = DynamixelClient(motors, 'COM13', 4000000)
                    self.dxl_client.connect()

        # Position-current control mode with default PID parameters
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * 5, 11, 1)
        self.dxl_client.set_torque_enabled(motors, True)
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kP, 84, 2)
        self.dxl_client.sync_write([0, 4, 8], np.ones(3) * (self.kP * 0.75), 84, 2)
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kI, 82, 2)
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kD, 80, 2)
        self.dxl_client.sync_write([0, 4, 8], np.ones(3) * (self.kD * 0.75), 80, 2)
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.curr_lim, 102, 2)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    def set_leap(self, pose):
        """Command hand directly in LEAP convention (pi = home)."""
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    def set_allegro(self, pose):
        """Command hand in allegro convention (0 = open, positive = close)."""
        pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    def set_ones(self, pose):
        """Command hand in [-1, 1] range (for RL policies)."""
        pose = lhu.sim_ones_to_LEAPhand(np.array(pose))
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    def read_pos(self):
        """Read current joint positions (LEAP convention)."""
        return self.dxl_client.read_pos()

    def read_vel(self):
        """Read current joint velocities."""
        return self.dxl_client.read_vel()

    def read_cur(self):
        """Read current motor currents."""
        return self.dxl_client.read_cur()

    def pos_vel(self):
        """Read positions and velocities (faster than separate calls)."""
        return self.dxl_client.read_pos_vel()

    def pos_vel_eff_srv(self):
        """Read positions, velocities, and currents (faster than separate calls)."""
        return self.dxl_client.read_pos_vel_cur()
