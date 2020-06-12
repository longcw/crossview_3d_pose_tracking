import cv2
import numpy as np

from .color_generator import ColorGenerator

color_generator = ColorGenerator()


class OpenCVWindow:
    def __init__(self, name, location, size, disable_imshow):
        self.name = name
        self.location = location
        self.size = size
        self.disable_imshow = disable_imshow
        if not self.disable_imshow:
            cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
            cv2.moveWindow(self.name, *location)
            cv2.resizeWindow(self.name, *self.size)

    def update(self, data):
        raise NotImplementedError

    def show(self, rendering):
        if not self.disable_imshow:
            cv2.imshow(self.name, rendering)


class CameraWindow(OpenCVWindow):
    def __init__(self, name, location, disable_imshow, image_size=(320, 280)):
        super().__init__(name, location, image_size, disable_imshow)
        self.newest_time = -np.inf
        self.point_radius = 2
        self.score_thresh = 0.1

    def update(self, frame, frame_data):
        rendering = frame["image"].copy()
        scale = np.sqrt(
            self.size[0] * self.size[1] / (rendering.shape[0] * rendering.shape[1])
        )
        rendering = cv2.resize(rendering, None, fx=scale, fy=scale)
        time_str = "{}: {:07.3f}".format(frame["camera_name"], frame["timestamp"])
        cv2.putText(
            rendering, time_str, (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )

        # plot poses
        if frame_data is not None:
            render_h, render_w = rendering.shape[0:2]
            scale_x = render_w / frame_data["image_wh"][0]
            scale_y = render_h / frame_data["image_wh"][1]
            for i, pose in enumerate(frame_data["poses"]):
                color = color_generator.get_color(pose["id"])

                raw_points = np.copy(pose["points_2d"])
                keypoints_score = pose["scores"]
                raw_points[:, 0] *= scale_x
                raw_points[:, 1] *= scale_y
                keypoints = [tuple(map(int, point)) for point in raw_points]
                for k, (point, score) in enumerate(zip(keypoints, keypoints_score)):
                    if score < self.score_thresh:
                        continue
                    cv2.circle(rendering, point, self.point_radius, color, -1)

        self.show(rendering)


class Visualizer(object):
    def __init__(
        self,
        image_size=(320, 280),
        screen_size=(1920, 1080),
        vis_3d=False,
        world_ltrb=None,
    ):
        self.vis3d = None
        if vis_3d:
            from .visualizer_3d import Visualizer3D

            self.vis3d = Visualizer3D(world_ltrb, color_generator)

        self._horizontal_gap = 5
        # self.x_ori = new_world_size[0] + self.horizontal_gap
        self._x_ori = self._horizontal_gap
        self._location = (self._x_ori, 100)
        self.image_size = image_size
        self.screen_size = screen_size
        self.camera_name_to_window = {}

    def create_camera_window(self, name):
        window = CameraWindow(
            name, self._location, disable_imshow=False, image_size=self.image_size
        )
        self.camera_name_to_window[name] = window

        self._location = [
            self._location[0] + self.image_size[0] + self._horizontal_gap,
            self._location[1],
        ]  # x, y
        if self._location[0] > self.screen_size[0] - self.image_size[0]:
            self._location[0] = self._x_ori
            self._location[1] += self.image_size[1] + 30
        return window

    def update(self, frame, frame_data):
        window = self.camera_name_to_window.get(frame["camera_name"])
        if window is None:
            window = self.create_camera_window(frame["camera_name"])
        window.update(frame, frame_data)

    def update_3d(self, data_3d):
        if self.vis3d is None:
            return
        self.vis3d.update(data_3d)

    def stop_v3d(self):
        if self.vis3d is not None:
            self.vis3d.stop()
