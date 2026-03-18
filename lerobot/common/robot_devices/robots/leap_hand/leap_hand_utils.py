"""Utilities for LEAP Hand joint angle conversions between conventions.

Embodiments:
  LEAPhand: Real LEAP hand (180 degrees for the motor is actual zero)
  LEAPsim:  LEAP hand in sim (has allegro-like zero positions)
  one_range: [-1, 1] for all joints to facilitate RL
  allegro:  Allegro hand in real or sim
"""

import numpy as np


def LEAPsim_limits():
    sim_min = np.array([
        -1.047, -0.314, -0.506, -0.366,
        -1.047, -0.314, -0.506, -0.366,
        -1.047, -0.314, -0.506, -0.366,
        -0.349, -0.47, -1.20, -1.34,
    ])
    sim_max = np.array([
        1.047, 2.23, 1.885, 2.042,
        1.047, 2.23, 1.885, 2.042,
        1.047, 2.23, 1.885, 2.042,
        2.094, 2.443, 1.90, 1.88,
    ])
    return sim_min, sim_max


def angle_safety_clip(joints):
    """Safety clips all joints so nothing unsafe can happen."""
    sim_min, sim_max = LEAPsim_limits()
    real_min = LEAPsim_to_LEAPhand(sim_min)
    real_max = LEAPsim_to_LEAPhand(sim_max)
    return np.clip(joints, real_min, real_max)


def scale(x, lower, upper):
    """Goes from [-1, 1] to [lower, upper]."""
    return 0.5 * (x + 1.0) * (upper - lower) + lower


def unscale(x, lower, upper):
    """Goes from [lower, upper] to [-1, 1]."""
    return (2.0 * x - upper - lower) / (upper - lower)


def sim_ones_to_LEAPhand(joints):
    """Isaac [-1, 1] range to LEAPHand real world."""
    sim_min, sim_max = LEAPsim_limits()
    joints = scale(joints, sim_min, sim_max)
    joints = LEAPsim_to_LEAPhand(joints)
    return joints


def LEAPhand_to_sim_ones(joints):
    """LEAPHand real world to Isaac [-1, 1] range."""
    joints = LEAPhand_to_LEAPsim(joints)
    sim_min, sim_max = LEAPsim_limits()
    joints = unscale(joints, sim_min, sim_max)
    return joints


def LEAPsim_to_LEAPhand(joints):
    """Sim LEAP hand to real LEAP hand."""
    joints = np.array(joints)
    return joints + 3.14159


def LEAPhand_to_LEAPsim(joints):
    """Real LEAP hand to sim LEAP hand."""
    joints = np.array(joints)
    return joints - 3.14159


def allegro_to_LEAPhand(joints, teleop=False, zeros=True):
    """Converts allegro hand radians to LEAP hand radians."""
    joints = np.array(joints)
    ret_joints = joints + 3.14159
    if zeros:
        ret_joints[0] = ret_joints[4] = ret_joints[8] = 3.14
    if teleop:
        ret_joints[12] = joints[12] + 0.2
        ret_joints[14] = joints[14] - 0.2
    return ret_joints


def LEAPhand_to_allegro(joints, teleop=False, zeros=True):
    """Converts LEAP hand radians to allegro hand radians."""
    joints = np.array(joints)
    ret_joints = joints - 3.14159
    if zeros:
        ret_joints[0] = ret_joints[4] = ret_joints[8] = 0
    if teleop:
        ret_joints[12] = joints[12] - 0.2
        ret_joints[14] = joints[14] + 0.2
    return ret_joints
