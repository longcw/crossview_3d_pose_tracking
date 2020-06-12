import argparse
import cv2
import numpy as np
from tqdm import tqdm

from crossview_dataset import data_utils
from crossview_dataset.visualization.visualizer import Visualizer
from crossview_dataset.calib.calibration import Calibration


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frame-root", type=str, default="/data/3DPose_pub/Campus_Seq1/frames",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default="/data/3DPose_pub/Campus_Seq1/calibration.json",
    )
    parser.add_argument(
        "--pose-file",
        type=str,
        default="/data/3DPose_pub/Campus_Seq1/annotation_2d.json",
    )
    parser.add_argument("--pose-type", type=str, default="2d", help="2d or 3d poses")
    parser.add_argument("--cameras", nargs="+", default=None, help="Camera names")
    return parser.parse_args()


def project_poses_2d(camera_name, data_3d, calibration: Calibration):
    if data_3d is None:
        return None
    # the projected points are scaled by the image width and height
    # so the image_wh is [1, 1] here
    frame_data = {
        "camera": camera_name,
        "timestamp": data_3d["timestamp"],
        "poses": [],
        "image_wh": [1, 1],
    }
    for pose_3d in data_3d["poses"]:
        points_3d = np.asarray(pose_3d["points_3d"], dtype=float)
        points_2d = calibration.project(points_3d, camera_name)
        frame_data["poses"].append(
            {"id": pose_3d["id"], "points_2d": points_2d, "scores": pose_3d["scores"]}
        )
    return frame_data


def main(args):
    # load data
    frame_loader = data_utils.FrameLoader(args.frame_root, args.cameras)
    if args.pose_type == "2d":
        pose_loader = data_utils.Pose2DLoader(args.pose_file)
    else:
        pose_loader = data_utils.Pose3DLoader(args.pose_file)

    # load calibration
    calibration = Calibration.from_json(args.calibration)

    # visualizer
    visualizer = Visualizer(
        vis_3d=args.pose_type == "3d", world_ltrb=calibration.world_ltrb
    )

    wait_time = 1
    for frame in tqdm(frame_loader, total=len(frame_loader)):
        if args.pose_type == "2d":
            data_3d = None
            frame_data = pose_loader.get_data(frame["frame_name"])
        else:
            # get the nearest 3d poses
            data_3d = pose_loader.get_data(frame["timestamp"])
            frame_data = project_poses_2d(frame["camera_name"], data_3d, calibration)

        visualizer.update(frame, frame_data)
        visualizer.update_3d(data_3d)
        key = cv2.waitKey(wait_time) % 128
        if key == ord("q"):
            break
        elif key == ord("a"):
            wait_time = int(not wait_time)
    visualizer.stop_v3d()


if __name__ == "__main__":
    main(parse_args())
