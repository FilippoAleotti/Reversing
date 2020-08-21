"""
This code is based on https://github.com/mrharicot/monodepth/blob/master/utils/evaluate_kitti.py
We would like to thank C. Godard and other authors for sharing their code
"""
from __future__ import division
import os
import numpy as np
import cv2
import argparse
from tqdm import tqdm

width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351

parser = argparse.ArgumentParser(description="Evaluation on the KITTI dataset")
parser.add_argument(
    "--prediction", type=str, help="path to estimated disparities", required=True
)
parser.add_argument(
    "--gt", type=str, help="path to ground truth disparities", required=True
)
parser.add_argument(
    "--min_depth", type=float, help="minimum depth for evaluation", default=1e-3
)
parser.add_argument(
    "--max_depth", type=float, help="maximum depth for evaluation", default=80
)
parser.add_argument("--noc", action="store_true", help="evaluate kitti noc ")
parser.add_argument("--test_file", type=str, default=None)
args = parser.parse_args()


def compute_errors(gt, pred):
    """Compute error between ground truth and prediction
    Args:
        gt: ground truth depth
        pred: predicted depth

    Return:
        rmse, rmse_log
        a1,a2,a3: delta accuracies
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    return rmse, rmse_log, a1, a2, a3


def load_gt_disp_kitti(path, is_noc):
    """Load ground truth disparity
    Args:
        path: path to folder that contains ground truth disp
        is_noc: True if you want to test noc disparities, False for occ

    Return:
        list with ground truth disparities as np arrays
    """
    ground_truth_disparities = []
    split = "/disp_noc_0/" if is_noc else "/disp_occ_0/"

    for index in range(200):
        disp_name = path + split + str(index).zfill(6) + "_10.png"
        if not os.path.exists(name):
            raise ValueError("Image {} not found: {}".format(i, disp_name))
        ground_truth = cv2.imread(disp_name, -1)
        ground_truth = ground_truth.astype(np.float32) / 256
        ground_truth_disparities.append(ground_truth)
    return ground_truth_disparities


def convert_disps_to_depths_kitti(gts, preds):
    """Turn disparities into depths, using baseline and focal length
    Args:
        gts: list with ground truth disparities
        preds: list with predicted disparities

    Return:
        gt_depths: list with ground truth depths
        pred_depths: list with predicted depths
    """
    depths_gt = []
    depths_pred = []

    for index, gt_disparity in enumerate(gts):
        _, width = gt_disparity.shape

        validity_mask = gt_disparity > 0
        depth_gt = width_to_focal[width] * 0.54 / (gt_disparity + (1.0 - validity_mask))
        depth_pred = width_to_focal[width] * 0.54 / preds[index]

        depths_gt.append(depth_gt)
        depths_pred.append(depth_pred)
    return depths_gt, depths_pred


if __name__ == "__main__":

    pred_disparities = []
    NUM_SAMPLES = 200

    for t_id in range(NUM_SAMPLES):
        name = os.path.join(args.prediction, str(t_id).zfill(6) + "_10.png")
        if not os.path.exists(name):
            raise ValueError("Prediction {}/200 not found:{}".format(t_id, name))
        disp = cv2.imread(name, -1) / 256.0
        pred_disparities.append(disp)

    gt_disparities = load_gt_disp_kitti(args.gt, args.noc)
    gt_depths, pred_depths = convert_disps_to_depths_kitti(
        gt_disparities, pred_disparities
    )
    rms = np.zeros(NUM_SAMPLES, np.float32)
    log_rms = np.zeros(NUM_SAMPLES, np.float32)
    d1_all = np.zeros(NUM_SAMPLES, np.float32)
    a1 = np.zeros(NUM_SAMPLES, np.float32)
    a2 = np.zeros(NUM_SAMPLES, np.float32)
    a3 = np.zeros(NUM_SAMPLES, np.float32)
    epe = np.zeros(NUM_SAMPLES, np.float32)

    with tqdm(total=NUM_SAMPLES) as pbar:
        for i in range(NUM_SAMPLES):
            gt_depth = gt_depths[i]
            pred_depth = pred_depths[i]

            pred_depth[pred_depth < args.min_depth] = args.min_depth
            pred_depth[pred_depth > args.max_depth] = args.max_depth

            gt_disp = gt_disparities[i]
            mask = gt_disp > 0
            pred_disp = pred_disparities[i]

            disp_diff = np.abs(gt_disp[mask] - pred_disp[mask])
            bad_pixels = np.logical_and(
                disp_diff >= 3, (disp_diff / gt_disp[mask]) >= 0.05
            )
            d1_all[i] = 100.0 * bad_pixels.sum() / mask.sum()
            epe[i] = np.abs(gt_disp[mask] - pred_disp[mask]).mean()

            rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(
                gt_depth[mask], pred_depth[mask]
            )
            pbar.update(1)
    print(
        "{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(
            "RMSE", "RMSE_LOG", "D1", "EPE", "A1", "A2", "A3"
        )
    )
    print(
        "{:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(
            rms.mean(),
            log_rms.mean(),
            d1_all.mean(),
            epe.mean(),
            a1.mean(),
            a2.mean(),
            a3.mean(),
        )
    )
