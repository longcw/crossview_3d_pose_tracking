import json
import numpy as np
import cv2
from .camera import Camera


class Calibration(object):
    def __init__(self, cameras, selected_camera_ids=None):
        self.cameras = {k: Camera.from_dict(v) for k, v in cameras.items()}
        if selected_camera_ids is not None:
            self.cameras = {
                camera_id: self.cameras[camera_id] for camera_id in selected_camera_ids
            }

        self.camera_ids = tuple(self.cameras.keys())
        self.world_ltrb = self.compute_world_ltrb()

    def get_projection_matrix(self, camera_id):
        camera = self.cameras[camera_id]
        P = camera.K @ np.eye(3, 4) @ np.linalg.inv(camera.Tw)
        return P

    def compute_world_ltrb(self, increment=0):
        cameras = self.cameras
        camera_ids = list(dict(cameras).keys())

        xs = []
        ys = []
        for camera_id in camera_ids:
            Tw = cameras[camera_id].Tw
            xs.append(Tw[0, 3])
            ys.append(Tw[1, 3])

        x1, y1 = min(xs) - increment, min(ys) - increment
        x2, y2 = max(xs) + increment, max(ys) + increment

        return [x1, y1, x2, y2]

    def undistort(self, point_2d, camera_id):
        camera = self.cameras[camera_id]
        point_2d = cv2.undistortPoints(
            point_2d[None, None, :], camera.K, camera.dist_coeffs, P=camera.K
        ).squeeze(axis=1)[0]
        return point_2d

    def project(self, points_3d, camera_id):
        camera = self.cameras[camera_id]  # type: Camera
        return camera.project(points_3d)

    def triangulate(self, points_2d, camera_ids):
        """
        Triangulation on multiple points from different cameras.
        args:
            points_2d: N x 2 np.ndarray of 2D points,
                       the points should be normalized by the image width and height,
                       i.e. the inputed x, y should be in the range of [0, 1]
            camera_ids: camera id for each point comes from
        """
        assert len(points_2d) >= 2, "triangulation requires at least two cameras"

        points_2d = np.asarray(points_2d)
        A = np.zeros([len(points_2d) * 2, 4], dtype=float)
        for i, point in enumerate(points_2d):
            camera_id = camera_ids[i]
            upoint = self.undistort(point, camera_id)
            P = self.get_projection_matrix(camera_id)
            P3T = P[2]
            A[2 * i, :] = upoint[0] * P3T - P[0]
            A[2 * i + 1, :] = upoint[1] * P3T - P[1]
        u, s, vh = np.linalg.svd(A)
        error = s[-1]
        X = vh[len(s) - 1]
        point_3d = X[:3] / X[3]

        return error, point_3d

    @classmethod
    def from_json(cls, filename, camera_ids=None):
        """Returns a Calibration intialized from a json file

        Args:
            filename (str): A json file containing calibration info.
        """
        with open(filename, "r") as f:
            data = json.loads(f.read())

        cameras = data["cameras"]
        return Calibration(cameras=cameras, selected_camera_ids=camera_ids)

    def save(self, filename):
        def listify(d):
            """Converts all ndarrays in dict d to lists"""
            if isinstance(d, np.ndarray):
                return d.tolist()
            if not isinstance(d, dict):
                return d
            return {k: listify(v) for k, v in d.items()}

        def jsonify(d):
            return json.dumps({str(id_): x for id_, x in listify(d).items()})

        data = {
            self.BOUNDS_KEY: self.world_ltrb,
            "cameras": {
                id_: camera.to_dict(legacy_format=False)
                for id_, camera in self.cameras.items()
            },
        }
        with open(filename, "w") as f:
            f.write(jsonify(data))
