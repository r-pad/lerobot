# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# From https://github.com/YingYuan0414/GeoRT/

import json
import os
from pathlib import Path

import numpy as np
import torch

from lerobot.common.robot_devices.robots.geort.formatter import HandFormatter
from lerobot.common.robot_devices.robots.geort.model import IKModel

def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_keypoint_info(config):
    joint_order = config["joint_order"]
    keypoint_joints = []
    keypoint_human_ids = []

    for info in config["fingertip_link"]:
        keypoint_human_ids.append(info["human_hand_id"])
        keypoint_joints.append([joint_order.index(j) for j in info["joint"]])

    return {"joint": keypoint_joints, "human_id": keypoint_human_ids}


def _parse_joint_limits(config):
    return np.array(config["joint"]["lower"]), np.array(config["joint"]["upper"])


class GeoRTRetargetingModel:
    """Inference wrapper: human hand keypoints -> robot joint positions."""

    def __init__(self, model_path, config_path):
        config = _load_json(config_path)
        keypoint_info = _parse_keypoint_info(config)
        joint_lower, joint_upper = _parse_joint_limits(config)

        self.human_ids = keypoint_info["human_id"]
        self.model = IKModel(keypoint_joints=keypoint_info["joint"]).cuda()
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()
        self.qpos_normalizer = HandFormatter(joint_lower, joint_upper)

    def forward(self, keypoints):
        """Map human hand keypoints to robot joint angles.

        Args:
            keypoints: [N, 3] array of hand keypoints from mocap.
        Returns:
            [DOF] array of robot joint angles.
        """
        keypoints = keypoints[self.human_ids]
        joint_normalized = self.model.forward(
            torch.from_numpy(keypoints).unsqueeze(0).reshape(1, -1, 3).float().cuda()
        )
        joint_raw = self.qpos_normalizer.unnormalize(joint_normalized.detach().cpu().numpy())
        return joint_raw[0]


def load_model(tag="", epoch=0, checkpoint_root=None):
    """Load a GeoRT retargeting model by checkpoint tag.

    Args:
        tag: Substring to match against checkpoint directory names.
        epoch: Specific epoch to load. 0 (default) loads 'last.pth'.
        checkpoint_root: Path to checkpoint directory (required).

    Returns:
        GeoRTRetargetingModel ready for inference.
    """
    if checkpoint_root is None:
        raise ValueError(
            "checkpoint_root is required. Set geort_checkpoint_root in FrankaLeapRobotConfig "
            "to point to your GeoRT checkpoint directory."
        )
    checkpoint_root = Path(checkpoint_root)

    checkpoint_name = ""
    for name in os.listdir(checkpoint_root):
        if tag in name:
            checkpoint_name = name
            break

    if not checkpoint_name:
        raise FileNotFoundError(
            f"No checkpoint matching tag '{tag}' found in {checkpoint_root}"
        )

    checkpoint_dir = checkpoint_root / checkpoint_name
    model_path = checkpoint_dir / (f"epoch_{epoch}.pth" if epoch > 0 else "last.pth")
    config_path = checkpoint_dir / "config.json"

    return GeoRTRetargetingModel(model_path=model_path, config_path=config_path)
