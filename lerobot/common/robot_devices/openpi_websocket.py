# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Minimal OpenPI policy WebSocket client and LeRobot -> pi05 observation formatting.
# Protocol matches openpi.serving.websocket_policy_server / openpi_client.websocket_client_policy.

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as ScipyRotation
import websockets.sync.client

from lerobot.common.robot_devices._msgpack_numpy import Packer, unpackb


def _convert_to_uint8(img: np.ndarray) -> np.ndarray:
    if np.issubdtype(img.dtype, np.floating):
        img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    return img


def _resize_with_pad_hwc(image: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Single image HWC uint8 or float -> HWC resized with pad (same behavior as openpi_client.image_tools)."""
    if image.shape[-3:-1] == (height, width):
        return image
    pil = Image.fromarray(_convert_to_uint8(image) if image.dtype != np.uint8 else image)
    cur_width, cur_height = pil.size
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = pil.resize((resized_width, resized_height), resample=method)
    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    return np.asarray(zero_image)


def _lerobot_image_to_uint8_hwc(arr: Any) -> np.ndarray:
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    x = np.asarray(arr)
    if x.ndim == 3 and x.shape[0] == 3:
        x = np.transpose(x, (1, 2, 0))
    return _convert_to_uint8(x)


class OpenPiWebsocketClient:
    """Sync WebSocket client for OpenPI policy servers (metadata on connect, then infer loop)."""

    def __init__(self, host: str = "127.0.0.1", port: int | None = 8000, *, api_key: str | None = None) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> dict:
        return self._server_metadata

    def _wait_for_server(self):
        logging.info("Waiting for OpenPI policy server at %s...", self._uri)
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("OpenPI server not ready; retrying in 5s...")
                time.sleep(5)

    def infer(self, obs: dict) -> dict:
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"OpenPI inference server error:\n{response}")
        return unpackb(response)

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass


def lerobot_observation_to_openpi_dict(
    observation: dict,
    *,
    prompt: str,
    resize_size: int,
    base_image_key: str,
    wrist_image_key: str,
) -> dict:
    """Build pi05-style observation dict (see openpi-mimicgen/examples/mimicgen/main.py)."""
    from lerobot.scripts.yufei_policy_utils import get_gripper_4_points_from_sriram_data

    if "observation.right_eef_pose" not in observation:
        raise KeyError("openpi_websocket expects observation.right_eef_pose (set robot.use_eef=True).")

    right_eef = observation["observation.right_eef_pose"]
    _eef_pos, _eef_rot_6d, _gw, eef_pos_robot_base, eef_rot_matrix_robot_base, _r6d_rb, eef_gripper_width_franka = (
        get_gripper_4_points_from_sriram_data(right_eef)
    )

    # rotvec = ScipyRotation.from_matrix(eef_rot_matrix_robot_base).as_rotvec()
    # norm = np.linalg.norm(rotvec)
    # axis = rotvec / norm
    # neg_axis = -axis
    # neg_rotation = 2 * np.pi - norm
    # neg_rotvec = neg_axis * neg_rotation
    
    gripper_qpos = np.array([eef_gripper_width_franka, -eef_gripper_width_franka], dtype=np.float32).reshape(2)
    # state = np.concatenate([eef_pos_robot_base.astype(np.float32), rotvec.astype(np.float32), gripper_qpos])
    # state = np.concatenate([eef_pos_robot_base.astype(np.float32), neg_rotvec.astype(np.float32), gripper_qpos])

    gripper_qpos = eef_gripper_width_franka.astype(np.float32)
    state = np.concatenate([eef_pos_robot_base.astype(np.float32), _r6d_rb.astype(np.float32), gripper_qpos])

    base_rgb = _lerobot_image_to_uint8_hwc(observation[base_image_key])
    wrist_rgb = _lerobot_image_to_uint8_hwc(observation[wrist_image_key])
    img = _resize_with_pad_hwc(base_rgb, resize_size, resize_size)
    wrist_img = _resize_with_pad_hwc(wrist_rgb, resize_size, resize_size)
    img = _convert_to_uint8(img)
    wrist_img = _convert_to_uint8(wrist_img)


    # import pdb; pdb.set_trace()

    # from matplotlib import pyplot as plt
    # plt.imshow(img)
    # plt.show()
    # plt.imshow(wrist_img)
    # plt.show()
    # plt.close()

    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": state.astype(np.float32),
        "prompt": prompt,
    }


def make_openpi_websocket_policy_dict(
    *,
    robot_adapter: Any,
    host: str = "127.0.0.1",
    port: int = 8000,
    prompt: str,
    replan_steps: int = 5,
    resize_size: int = 224,
    base_image_key: str = "observation.images.cam_azure_kinect_front.color",
    wrist_image_key: str = "observation.images.cam_wrist",
    gripper_delta_scale: float = 1.0,
    api_key: str | None = None,
) -> dict:
    """Construct the policy dict consumed by control_utils `type == openpi_websocket` branch."""
    client = OpenPiWebsocketClient(host, port, api_key=api_key)
    return {
        "type": "openpi_websocket",
        "client": client,
        "robot_adapter": robot_adapter,
        "action_queue": deque(),
        "prompt": prompt,
        "replan_steps": replan_steps,
        "resize_size": resize_size,
        "base_image_key": base_image_key,
        "wrist_image_key": wrist_image_key,
        "gripper_delta_scale": gripper_delta_scale,
    }
