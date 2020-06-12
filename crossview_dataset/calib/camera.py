import json
import cv2
import numpy as np
from numpy.linalg import inv


def distortion_coeffs(k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
    """Composes a 5x1 cv2 compatible matrix of distortion coefficients

    Args:
        k1 (float): First radial distortion coefficient.
        k2 (float): Second radial distortion coefficient.
        p1 (float): First tangential distortion coefficient.
        p2 (float): Second tangential distortion coefficient.
        k3 (float): Third radial distortion coefficient.
    """
    return np.array([k1, k2, p1, p2, k3])


def intrinsic_matrix(fx=1.0, fy=1.0, cx=0.5, cy=0.5):
    """Composes a 3x3 intrinsic matrix from the provided information

    Args:
        fx (float): Focal length of the camera in x.
        fy (float): Focal length of the camera in y.
        cx (float): X-coordinate of the principal point.
        cy (float): Y-coordinate of the principal point.
    """
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def pose_matrix(R=None, t=None):
    """Composes the 4x4 pose matrix from R and t

    Args:
        R (ndarray): Either a 3x3 rotation matrix or a Rodrigues rotation
                     vector. Defaults to no rotation.
        t (ndarray): A 3 element vector denoting the translation. Defaults to
                     [0, 0, 0].
    """
    # Default to no rotation
    if R is None:
        R = np.eye(3)
    # Default to origin
    if t is None:
        t = [0, 0, 0]
    # Convert lists to ndarrays
    R = Marshal.ndarrayify(R)
    t = Marshal.ndarrayify(t)
    # Convert from Rodrigues notation if necessary
    if R.shape == (3,):
        R, _ = cv2.Rodrigues(R)
    # Construct the matrix
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t
    return H


def pose_to_vectors(Tw):
    """Extracts the rotation and translation vector from a pose matrix

    Args:
        Tw (ndarray): A 4x4 pose matrix.

    Returns: A tuple (r, t)
        r (ndarray): Rodrigues rotation vector.
        t (ndarray): Translation vector.
    """
    r, _ = cv2.Rodrigues(Tw[:3, :3])
    r = np.squeeze(r)
    t = Tw[:3, 3]
    return r, t


class Marshal(object):
    """A collection of useful marshalling and demarshalling functions"""

    @staticmethod
    def listify(d):
        """Converts all ndarrays in dict d to lists"""
        if isinstance(d, np.ndarray):
            return d.tolist()
        if not isinstance(d, dict):
            return d
        return {k: Marshal.listify(v) for k, v in d.items()}

    @staticmethod
    def ndarrayify(d):
        """Converts all lists in d to ndarrays"""
        if isinstance(d, np.ndarray):
            return d
        if isinstance(d, list):
            return np.array(d)
        if not isinstance(d, dict):
            return d
        return {k: Marshal.ndarrayify(v) for k, v in d.items()}

    @staticmethod
    def jsonify(d):
        """Converts the data into a json compatible string.

        This opperation converts the ids into strings and the ndarrays into
        lists.

        Args:
            d (dict): A dictionary to make json compatible.
        """
        return json.dumps({str(id_): x for id_, x in Marshal.listify(d).items()})


def back_project(points_2d, z_worlds, K, Tw, dist_coeffs):
    """Back project points in the image plane to 3D

    A single point in the image plane correspods to a ray in 3D space. This
    method determines the 3D cooridates of the points where rays cast out
    of the image plane intersect with the provided heights.

    Args:
        points_2d (ndarray): An Nx2 array of image coordinates to back
                                project.
        z_worlds (ndarray): A list-like object of N heights (assuming z=0
                            is the ground plane) to back project to.
        K (ndarray): A 3x3 intrinsic matrix.
        Tw (ndarray): A 4x4 pose matrix.
        dist_coeffs (ndarray): An array of distortion coefficients of the form
                               [k1, k2, [p1, p2, [k3]]], where ki is the ith
                               radial distortion coefficient and pi is the ith
                               tangential distortion coeff.
    """
    # Unpack the intrinsics we are going to need for this calculation.
    fx, fy = K[0, 0], K[1, 1]
    ccx, ccy = K[0, 2], K[1, 2]
    points_2d = Marshal.ndarrayify(points_2d)
    points_2d = cv2.undistortPoints(
        points_2d[:, np.newaxis], K, dist_coeffs, P=K
    ).squeeze(axis=1)
    points_3d = []
    # TODO: Vectorize
    for (x_image, y_image), z_world in zip(points_2d, z_worlds):
        kx = (x_image - ccx) / fx
        ky = (y_image - ccy) / fy
        # get point position in camera coordinates
        z3d = (z_world - Tw[2, 3]) / np.dot(Tw[2, :3], [kx, ky, 1])
        x3d = kx * z3d
        y3d = ky * z3d
        # transform the point to world coordinates
        x_world, y_world = (Tw @ [x3d, y3d, z3d, 1])[:2]
        points_3d.append((x_world, y_world, z_world))
    return np.array(points_3d)


def distortion(points_2d, K, dist_coeffs=None):
    if dist_coeffs is None:
        return points_2d

    k1, k2, p1, p2, k3 = dist_coeffs
    cx, cy = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]

    # To relative coordinates
    x = (points_2d[:, 0] - cx) / fx
    y = (points_2d[:, 1] - cy) / fy
    r2 = x * x + y * y

    # Radial distorsion
    xdistort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
    ydistort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)

    # Tangential distorsion
    xdistort += 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    ydistort += p1 * (r2 + 2 * y * y) + 2 * p2 * x * y

    # Back to absolute coordinates.
    xdistort = xdistort * fx + cx
    ydistort = ydistort * fy + cy

    return np.stack([xdistort, ydistort]).T


def project(points_3d, K, Tw, dist_coeffs=None):
    def make_3x4(K, Tw):
        tmp = np.append(np.eye(3), np.zeros((3, 1)), axis=1)
        return K @ tmp @ np.linalg.inv(Tw)

    P = make_3x4(K, Tw)
    p3d_ = np.hstack((points_3d, np.ones([len(points_3d), 1], dtype=points_3d.dtype)))
    p2d_ = p3d_ @ P.T
    p2d = p2d_[:, 0:2] / p2d_[:, 2:3]

    # only valid point needs distortion
    valid = np.all(p2d > 0, axis=1) & np.all(p2d < 1, axis=1)
    valid_p2d = p2d[valid]
    if len(valid_p2d) > 0:
        p2d[valid] = distortion(valid_p2d, K, dist_coeffs)

    return np.squeeze(p2d)


class Camera(object):
    """Data class that models a single camera's intrinsics and extrinsics"""

    def __init__(self, K=None, Tw=None, dist_coeffs=None):
        """Contruct a Camera

        Args:
            K (ndarray): A 3x3 intrinsic matrix
            Tw (ndarray): A 4x4 pose matrix
            dist_coeffs (ndarray): An array of distortion coefficients of the
                                   form [k1, k2, [p1, p2, [k3]]], where k_i is
                                   the ith radial_distortion coefficient and
                                   p_i is the ith tangential distortion coeff.
        """
        self.K = intrinsic_matrix() if K is None else Marshal.ndarrayify(K)
        if self.K.shape != (3, 3):
            raise ValueError("Intrinsic Matrix K should be 3x3.")
        self.Tw = pose_matrix() if Tw is None else Marshal.ndarrayify(Tw)
        if self.Tw.shape != (4, 4):
            raise ValueError("Pose Matrix K should be 4x4.")
        dist_coeffs = [] if dist_coeffs is None else dist_coeffs
        self.dist_coeffs = distortion_coeffs(*dist_coeffs)

    def update_camera_location(self, new_location):
        self.Tw[:3, -1] = new_location

    def update_euler_angles(self, new_angles):
        rx, ry, rz = map(lambda r: r * np.pi / 180, new_angles)

        sa = np.sin(rx)
        ca = np.cos(rx)
        sb = np.sin(ry)
        cb = np.cos(ry)
        sg = np.sin(rz)
        cg = np.cos(rz)

        r11 = cb * cg
        r12 = cg * sa * sb - ca * sg
        r13 = sa * sg + ca * cg * sb
        r21 = cb * sg
        r22 = sa * sb * sg + ca * cg
        r23 = ca * sb * sg - cg * sa
        r31 = -sb
        r32 = cb * sa
        r33 = ca * cb

        R = np.asarray([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
        self.Tw[0:3, 0:3] = inv(R)

    @property
    def euler_angles(self):
        Tw = np.linalg.inv(self.Tw)
        rx = np.arctan2(Tw[2, 1], Tw[2, 2])
        ry = np.arctan2(-Tw[2, 0], np.sqrt(Tw[2, 1] ** 2 + Tw[2, 2] ** 2))
        rz = np.arctan2(Tw[1, 0], Tw[0, 0])

        rx, ry, rz = map(lambda r: r * 180 / np.pi, [rx, ry, rz])
        return rx, ry, rz

    @property
    def aspect(self):
        """Returns the aspect ratio of the camera"""
        return self.K[1, 1] / self.K[0, 0]

    @property
    def location(self):
        """Returns the 3D location of the camera"""
        return self.Tw[:3, -1]

    @property
    def look_at(self):
        """Returns the intersection of the optical axis and the floor"""
        return self.back_project(points_2d=[[0.5, 0.5]], z_worlds=[0.0])[0]

    def unnormalized(self, h):
        """Returns an unnormalized version of the intrinsic matrix K"""
        w = self.aspect * h
        return np.diag([w, h, 1.0]) @ self.K

    def back_project(self, points_2d, z_worlds):
        """Back project points in the image plane to 3D

        A single point in the image plane correspods to a ray in 3D space. This
        method determines the 3D cooridates of the points where rays cast out
        of the image plane intersect with the provided heights.

        Args:
            points_2d (ndarray): An Nx2 array of image coordinates to back
                                    project.
            z_worlds (ndarray): A list-like object of N heights (assuming z=0
                                is the ground plane) to back project to.
        """
        return back_project(
            points_2d=points_2d,
            z_worlds=z_worlds,
            K=self.K,
            Tw=self.Tw,
            dist_coeffs=self.dist_coeffs,
        )

    def get_distance(self, points_3d):
        """Get distande of the 3D points to camera

        Args:
            points_3d (ndarray): An Nx3 array of 3D points to calculate
            distance
        """
        return np.linalg.norm(points_3d - self.location, axis=-1)

    def project(self, points_3d):
        """Project the 3D points into the image plane of this camera

        Args:
            points_3d (ndarray): An Nx3 array of 3D points to project.
        """
        return project(
            points_3d=points_3d, K=self.K, Tw=self.Tw, dist_coeffs=self.dist_coeffs
        )

    @classmethod
    def from_dict(cls, d):
        """Contruct a camera from a dict

        Args:
            d (dict): A dictionary containing entries for:
                      - 'K': A 3x3 intrinsic matrix
                      - 'Tw': A 4x4 pose matrix
                      - 'dist_coeffs': A (5,) vector of distortion coefficients
        """
        w, h = d.get("image_wh", [1.0, 1.0])
        return Camera(
            K=cls.normalize(d["K"], w, h) if "K" in d else None,
            Tw=d.get("Tw", d.get("pose")),
            dist_coeffs=d.get("dist_coeffs"),
        )

    @staticmethod
    def normalize(K, w, h):
        """Normalizes the intrinsic matrix K by the given width and height"""
        return np.diag([1.0 / w, 1.0 / h, 1.0]) @ K

    def to_dict(self, legacy_format=False):
        """Returns a dict representation of this camera"""
        # We include a look_at point to make life easier on the visualiztion
        # team. They need to know where the optical axis intersects with the
        # groundplane to properly visualize the cameras. If we don't serialize
        # this value, they would need to write a javascript implementation of
        # backprojection. This saves them some trouble.
        d = {
            "K": self.K,
            "Tw": self.Tw,
            "dist_coeffs": self.dist_coeffs,
            "look_at": self.look_at,
        }
        if legacy_format:
            # Arbitrarily choose a height to be 1
            h = 1
            w = self.aspect * h
            d["K"] = self.unnormalized(h)
            d["image_wh"] = [w, h]
        return d

    def __eq__(self, other):
        """Override default __eq__ because ndarrays are not comparable"""
        if not isinstance(other, Camera):
            return False
        return all(
            [
                np.allclose(self.K, other.K),
                np.allclose(self.Tw, other.Tw),
                np.allclose(self.dist_coeffs, other.dist_coeffs),
            ]
        )

    def __repr__(self):
        return "Camera:\n\tK={}\n\tTw={}\n\tdist_coeffs={}".format(
            self.K, self.Tw, self.dist_coeffs
        )
