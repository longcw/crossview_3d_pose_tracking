import os
from typing import List
import glob
import numpy as np
import cv2

try:
    import ujson as json
except ImportError:
    import json


class FrameLoader:
    def __init__(self, frame_root: str, camera_names: List[str] = None):
        self.frame_root = frame_root

        # load cameras
        self.camera_names = camera_names
        if not self.camera_names:
            self.camera_names = os.listdir(self.frame_root)

        # load image files
        self.full_names = []
        for camera in self.camera_names:
            self.full_names.extend(
                glob.glob(os.path.join(self.frame_root, camera, "*.jpg"))
            )

        # sort by timestamp
        self.timestamps = [
            float(os.path.splitext(os.path.basename(filename))[0])
            for filename in self.full_names
        ]
        self._sorted_indices = np.argsort(self.timestamps)

        print(
            "Load {} frames from {}".format(
                len(self._sorted_indices), self.camera_names
            )
        )

    def __len__(self):
        return len(self._sorted_indices)

    def __getitem__(self, i):
        idx = self._sorted_indices[i]
        filename = self.full_names[idx]
        timestamp = self.timestamps[idx]
        frame_name = os.path.relpath(filename, self.frame_root)
        camera_name = os.path.dirname(frame_name)
        return {
            "image": cv2.imread(filename),
            "full_name": filename,
            "frame_name": frame_name,
            "camera_name": camera_name,
            "timestamp": timestamp,
        }


class Pose2DLoader:
    def __init__(self, data_file: str):
        with open(data_file, "r") as f:
            self.data = json.load(f)

    def get_data(self, frame_name):
        frame_data = self.data["frames"][frame_name]
        frame_data["image_wh"] = self.data["image_wh"]
        return frame_data


class Pose3DLoader:
    def __init__(self, data_file: str):
        with open(data_file, "r") as f:
            self.data = json.load(f)
        self.timestamps = [item["timestamp"] for item in self.data]

        # sort by timestamp
        sorted_indices = np.argsort(self.timestamps)
        self.data = [self.data[i] for i in sorted_indices]
        self.timestamps = np.asarray([item["timestamp"] for item in self.data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def get_data(self, timestamp, delta_time_threshold=0.1):
        idx = np.searchsorted(self.timestamps, timestamp)
        if idx >= len(self.timestamps) or (
            idx > 0
            and (
                abs(self.timestamps[idx - 1] - timestamp)
                < abs(self.timestamps[idx] - timestamp)
            )
        ):
            idx = idx - 1

        if (
            idx < 0
            or idx >= len(self.timestamps)
            or abs(self.timestamps[idx] - timestamp) > delta_time_threshold
        ):
            return None

        return self.data[idx]
