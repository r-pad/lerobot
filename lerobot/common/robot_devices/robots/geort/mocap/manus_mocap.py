# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# From https://github.com/YingYuan0414/GeoRT/

import threading
import time

import numpy as np
import zmq

LEFT_GLOVE_SN = "6fb94ce0"
RIGHT_GLOVE_SN = "cd9db816"


def parse_full_skeleton(data, right_glove_sn=RIGHT_GLOVE_SN, left_glove_sn=LEFT_GLOVE_SN):
    if data[0] == left_glove_sn:
        return None
    elif data[0] == right_glove_sn:
        data = np.array(list(map(float, data[1:]))).reshape(-1, 7)
        # Transform from Manus glove coords to canonical hand coords.
        T = np.array([
            [ 0, -1, 0],
            [-1,  0, 0],
            [ 0,  0, 1],
        ])
        points_transformed = data[:, :3] @ T.T
        keep = [i for i in range(len(points_transformed)) if i not in (5, 10, 15, 20)]
        return points_transformed[keep]
    else:
        return None


class ManusMocap:
    """Receives Manus glove skeleton data over ZMQ PULL socket.

    Runs a background thread to continuously receive and update latest data.
    """

    def __init__(self, host="localhost", port=8000,
                 right_glove_sn=RIGHT_GLOVE_SN, left_glove_sn=LEFT_GLOVE_SN):
        self._right_glove_sn = right_glove_sn
        self._left_glove_sn = left_glove_sn
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        socket.setsockopt(zmq.CONFLATE, True)
        socket.connect(f"tcp://{host}:{port}")
        self.socket = socket

        self._latest_data = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def _recv_loop(self):
        while self._running:
            try:
                msg = self.socket.recv(flags=zmq.NOBLOCK)
                msg = msg.decode("utf-8")
                data = msg.split(",")
                if len(data) == 176:
                    arr = parse_full_skeleton(data, self._right_glove_sn, self._left_glove_sn)
                    if arr is None:
                        continue
                    assert arr.shape == (21, 3)
                    with self._lock:
                        self._latest_data = arr
            except zmq.Again:
                time.sleep(0.001)

    def get(self):
        with self._lock:
            if self._latest_data is not None:
                return {"result": self._latest_data.copy(), "status": "recording"}
            else:
                return {"result": None, "status": "no data"}

    def close(self):
        self._running = False
        self._thread.join()
        self.socket.close()
