# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# From https://github.com/YingYuan0414/GeoRT/

import numpy as np


class HandFormatter:
    """Simple joint position normalizer ([-1, 1] <-> joint limits)."""

    def __init__(self, joint_lower_limit, joint_upper_limit):
        self.joint_lower_limit = np.array(joint_lower_limit)
        self.joint_upper_limit = np.array(joint_upper_limit)

    def normalize(self, x):
        return ((x - self.joint_lower_limit) / (self.joint_upper_limit - self.joint_lower_limit) - 0.5) * 2

    def unnormalize(self, x):
        return (x / 2 + 0.5) * (self.joint_upper_limit - self.joint_lower_limit) + self.joint_lower_limit
