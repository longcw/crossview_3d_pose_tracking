import argparse
import random
import numpy as np
from tqdm import tqdm
from collections import OrderedDict, defaultdict
from prettytable import PrettyTable

from crossview_dataset import data_utils

np.set_printoptions(precision=3, suppress=True, linewidth=200)  # numpy printing options
random.seed(234)
np.random.seed(234)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation",
        type=str,
        default="/data/3DPose_pub/Shelf_Seq1/annotation_3d.json",
    )
    parser.add_argument(
        "--result", type=str, default="/data/3DPose_pub/Shelf_Seq1/result_3d.json"
    )
    args = parser.parse_args()
    return args


def print_results(check_result):
    bone_group = OrderedDict(
        [
            ("Head", np.array([8])),
            ("Torso", np.array([9])),
            ("Upper arms", np.array([5, 6])),
            ("Lower arms", np.array([4, 7])),
            ("Upper legs", np.array([1, 2])),
            ("Lower legs", np.array([0, 3])),
        ]
        # + [(str(i), np.array([i])) for i in range(check_result.shape[2])]
    )

    # total_avg = np.sum(check_result > 0) / np.sum(np.abs(check_result))
    person_wise_avg = np.sum(check_result > 0, axis=(0, 2)) / np.sum(
        np.abs(check_result), axis=(0, 2)
    )

    bone_wise_result = OrderedDict()
    bone_person_wise_result = OrderedDict()
    for k, v in bone_group.items():
        bone_wise_result[k] = np.sum(check_result[:, :, v] > 0) / np.sum(
            np.abs(check_result[:, :, v])
        )
        bone_person_wise_result[k] = np.sum(
            check_result[:, :, v] > 0, axis=(0, 2)
        ) / np.sum(np.abs(check_result[:, :, v]), axis=(0, 2))

    tb = PrettyTable()
    tb.field_names = (
        ["Bone Group"]
        + [f"Actor {i}" for i in range(bone_person_wise_result["Head"].shape[0])]
        + ["Average"]
    )
    list_tb = [tb.field_names]
    for k, v in bone_person_wise_result.items():
        this_row = (
            [k]
            + [np.char.mod("%.4f", i) for i in v]
            + [np.char.mod("%.4f", np.sum(v) / len(v))]
        )
        list_tb.append(
            [
                i.astype(float).tolist() if isinstance(i, np.ndarray) else i
                for i in this_row
            ]
        )
        tb.add_row(this_row)
    this_row = (
        ["Total"]
        + [np.char.mod("%.4f", i) for i in person_wise_avg]
        + [np.char.mod("%.4f", np.sum(person_wise_avg) / len(person_wise_avg))]
    )
    tb.add_row(this_row)
    list_tb.append(
        [i.astype(float).tolist() if isinstance(i, np.ndarray) else i for i in this_row]
    )
    return tb, list_tb


def coco2shelf3D(coco_pose):
    """
    transform coco order(our method output) 3d pose to shelf dataset order with interpolation
    :param coco_pose: np.array with shape 17x3
    :return: 3D pose in shelf order with shape 14x3
    """
    shelf_pose = np.zeros((14, 3))
    coco2shelf = np.array([16, 14, 12, 11, 13, 15, 10, 8, 6, 5, 7, 9])
    shelf_pose[0:12] += coco_pose[coco2shelf]

    # Use middle of shoulder to init
    shelf_pose[12] = (shelf_pose[8] + shelf_pose[9]) / 2
    # shelf_pose[13] = coco_pose[0]  # use nose to init
    shelf_pose[13] = shelf_pose[12] + (coco_pose[0] - shelf_pose[12]) * np.array(
        [0.78, 0.5, 1.5]
    )
    shelf_pose[12] = shelf_pose[12] + (coco_pose[0] - shelf_pose[12]) * np.array(
        [0.3, 0.4, 0.6]
    )

    return shelf_pose


def _match_gts(model_poses, gt_poses):
    if len(model_poses) == 0 or len(gt_poses) == 0:
        return [-1] * len(gt_poses)
    dists = np.zeros([len(model_poses), len(gt_poses)], dtype=float)
    for i, model_pose in enumerate(model_poses):
        for j, gt_pose in enumerate(gt_poses):
            dists[i, j] = np.linalg.norm(model_pose - gt_pose)
    indices = np.argmin(dists, axis=0)
    return indices


def _eval_pose(model_pose, gt_pose):
    def is_right(
        model_start_point, model_end_point, gt_strat_point, gt_end_point, alpha=0.5
    ):
        bone_lenth = np.linalg.norm(gt_end_point - gt_strat_point)
        start_difference = np.linalg.norm(gt_strat_point - model_start_point)
        end_difference = np.linalg.norm(gt_end_point - model_end_point)
        return ((start_difference + end_difference) / 2) <= alpha * bone_lenth

    bones = [
        [0, 1],
        [1, 2],
        [3, 4],
        [4, 5],
        [6, 7],
        [7, 8],
        [9, 10],
        [10, 11],
        [12, 13],
    ]
    check_result = np.zeros([10], dtype=int)
    for i, bone in enumerate(bones):
        start_point, end_point = bone
        if is_right(
            model_pose[start_point],
            model_pose[end_point],
            gt_pose[start_point],
            gt_pose[end_point],
        ):
            check_result[i] = 1
        else:
            check_result[i] = -1

    gt_hip = (gt_pose[2] + gt_pose[3]) * 0.5
    tracked_hip = (model_pose[2] + model_pose[3]) * 0.5
    if is_right(tracked_hip, model_pose[12], gt_hip, gt_pose[12]):
        check_result[-1] = 1
    else:
        check_result[-1] = -1

    return check_result


def evaluate(args):
    anno_loader = data_utils.Pose3DLoader(args.annotation)
    res_loader = data_utils.Pose3DLoader(args.result)

    if "Campus" in args.annotation:
        test_range = list(range(350, 471)) + list(range(650, 751))
    elif "Shelf" in args.annotation:
        test_range = range(300, 600)
    else:
        raise ValueError(args.annotation)

    # Evaluate
    check_results = defaultdict(list)  # id -> N x 10
    for fid in tqdm(test_range, total=len(test_range)):
        anno = anno_loader[fid]
        gt_poses = [
            np.asarray(pose["points_3d"], dtype=float) for pose in anno["poses"]
        ]
        gt_ids = [pose["id"] for pose in anno["poses"]]

        res = res_loader.get_data(anno["timestamp"], delta_time_threshold=0.1)
        if res is None:
            model_poses = []
            # model_ids = []
        else:
            model_poses = [
                coco2shelf3D(np.asarray(pose["points_3d"], dtype=float))
                for pose in res["poses"]
            ]
            # model_ids = [pose["id"] for pose in res["poses"]]

        # Match targets and gts
        indices = _match_gts(model_poses, gt_poses)
        # eval
        for i, gt_pose in enumerate(gt_poses):
            idx = indices[i]
            gt_id = gt_ids[i]
            if idx < 0:
                check_results[gt_id].append([-1] * 10)
            else:
                check_results[gt_id].append(_eval_pose(model_poses[idx], gt_pose))

    # filter out zero results
    if "Shelf" in args.annotation:
        del check_results[3]  # follow the orginal mvpose

    n_frames = np.max([len(check_res) for check_res in check_results.values()])
    n_targets = len(check_results)
    total_check_result = np.zeros([n_frames, n_targets, 10], dtype=int)
    for i, gt_id in enumerate(sorted(list(check_results.keys()))):
        check_res = check_results[gt_id]
        total_check_result[0 : len(check_res), i, :] = np.asarray(check_res)

    tb, list_tb = print_results(total_check_result)
    print(tb)


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
