import shutil
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
import torch
import yaml
import argparse
import datetime
import ast
import cv2
from tools.semi_base3d import SemiBase3DDetector
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu


"""
有没有可能时分类分数的伪标签不能反应教师预测的包围盒的质量，可以试试统计下预训练模型的依据分类分数筛选的boxes和gt的iou与该boxes的分类分数是不是正比，做一个散点图看看
只用统计val上的就行，正好有gt在
"""


parser = argparse.ArgumentParser(description='Depth-aware Transformer for Monocular 3D Object Detection')
parser.add_argument('--config', dest='config', help='settings of detection in yaml format')
parser.add_argument('-e', '--evaluate_only', action='store_true', default=False, help='evaluation only')
parser.add_argument(
    '--launcher',
    choices=['none', 'pytorch', 'slurm', 'mpi'],
    default='none',
    help='job launcher')
parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
args = parser.parse_args()
if 'LOCAL_RANK' not in os.environ:
    os.environ['LOCAL_RANK'] = str(args.local_rank)


def main():
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config_name, _ = os.path.splitext(os.path.basename(args.config))
    save_dir = "statistics"
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = cfg["trainer"].get("pretrain_model", None)

    print("start statistics:")
    print(f"loading from {checkpoint}")
    unlabeled_dataset = KITTI_Dataset(split=cfg["dataset"]["inference_split"], cfg=cfg['dataset'])
    subset = Subset(unlabeled_dataset, range(20))     # 3712 3769 14940 40404  4,5
    loader = DataLoader(dataset=subset,
                        batch_size=1,
                        num_workers=1,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False,
                        persistent_workers=True)
    model = SemiBase3DDetector(cfg, cfg['model'], loader, cfg["semi_train_cfg"], cfg["semi_test_cfg"], inference_set=subset.dataset).to(
        'cuda')
    if checkpoint is not None:
        ckpt = torch.load(checkpoint)
        model.load_state_dict(ckpt['state_dict'])

    all_scores = []
    all_max_ious = []
    all_l2_distance = []
    all_depth_score = []
    for inputs, calib, targets, info in tqdm(loader):
        input_teacher = inputs[1]
        input_teacher = input_teacher.to("cuda")
        calib = calib.to("cuda")
        id = int(info['img_id'])
        # print(id)
        info['img_size'] = info['img_size'].to("cuda")
        calibs_from_file = subset.dataset.get_calib(id)
        # img = subset.dataset.get_image(id)
        # pc_velo = subset.dataset.get_lidar(id)
        boxes_lidar, score, loc_list, depth_score_list = model.teacher(input_teacher, calib, targets, info, mode='statistics')
        if boxes_lidar is None:
            continue
        # print(boxes_lidar.shape, score.shape)
        gt_objects = unlabeled_dataset.get_label(id)
        gts = []
        for i in range(len(gt_objects)):
            # filter objects by writelist
            if gt_objects[i].cls_type not in subset.dataset.writelist:
                continue
            # filter inappropriate samples
            if gt_objects[i].level_str == 'UnKnown' or gt_objects[i].pos[-1] < 2:
                continue
            # ignore the samples beyond the threshold [hard encoding]
            threshold = 65
            if gt_objects[i].pos[-1] > threshold:
                continue
            # filter 3d center out of img
            proj_inside_img = True
            center_3d = gt_objects[i].pos + [0, -gt_objects[i].h / 2, 0]  # real 3D center in 3D space
            center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
            center_3d, _ = calibs_from_file.rect_to_img(center_3d)  # project 3D center to image plane
            center_3d = center_3d[0]  # shape adjustment
            if center_3d[0] < 0 or center_3d[0] >= subset.dataset.resolution[0]:
                proj_inside_img = False
            if center_3d[1] < 0 or center_3d[1] >= subset.dataset.resolution[1]:
                proj_inside_img = False
            if proj_inside_img == False:
                continue
            gts.append(gt_objects[i])

        gt_boxes = []
        loc_gts = []
        l2_distances = []
        calibs_gt = [subset.dataset.get_calib(index) for index in info['img_id']]
        calib_gt = calibs_gt[0]
        for gt in gts:
            if gt.cls_type == "Car":
                loc = gt.pos.reshape((1, -1))
                loc_gts.append(torch.tensor(loc))
                h = np.array([gt.h]).reshape((1, -1))
                w = np.array([gt.w]).reshape((1, -1))
                l = np.array([gt.l]).reshape((1, -1))
                ry = np.array([gt.ry]).reshape((1, -1))
                loc_lidar = calib_gt.rect_to_lidar(loc)
                loc_lidar[:, 2] += h[:, 0] / 2
                heading = -(np.pi / 2 + ry)
                gt_lidar = np.concatenate([loc_lidar, l, w, h, heading], axis=1)
                gt_lidar = torch.from_numpy(gt_lidar)
                gt_boxes.append(gt_lidar)
        if gt_boxes:
            gt_boxes = torch.stack(gt_boxes).to('cuda')
            gt_boxes = gt_boxes.squeeze(1).float()
        else:
            continue


        boxes_lidar = boxes_lidar.float().to('cuda')
        iou3D = boxes_iou3d_gpu(boxes_lidar, gt_boxes)  # [num_pre, num_gt]
        # max_iou_per_pred = torch.max(iou3D, dim=1)[0]
        max_iou_values, max_iou_indices = torch.max(iou3D, dim=1)

        valid_indices = [idx for idx, val in enumerate(max_iou_values.cpu().numpy()) if val > 0]
        filtered_scores = [score.cpu().numpy()[i] for i in valid_indices]
        pred_depth_scores = [depth_score_list.cpu().numpy()[i] for i in valid_indices]
        filtered_max_ious = [max_iou_values.cpu().numpy()[i] for i in valid_indices]
        filtered_l2_distances = []

        for idx in valid_indices:
            pred_loc = loc_list[idx].cpu()  # 预测中心点，确保已移到CPU上
            gt_loc = loc_gts[max_iou_indices[idx]].cpu()  # 真实中心点，确保已移到CPU上
            l2_distance = torch.norm(pred_loc - gt_loc, p=2).item()  # 计算L2距离
            filtered_l2_distances.append(l2_distance)

        # print(iou3D)
        # print(filtered_max_ious)
        # print(filtered_scores)
        # print(filtered_l2_distances)
        all_scores.extend(filtered_scores)
        all_max_ious.extend(filtered_max_ious)
        all_l2_distance.extend(filtered_l2_distances)
        all_depth_score.extend(pred_depth_scores)
        # print(id)
        # print(len(all_scores))
        # print(len(all_max_ious))
        # print(len(all_l2_distance))

    print(len(all_scores))
    print(len(all_max_ious))
    print(len(all_l2_distance))
    print(len(all_depth_score))
    plt.figure(figsize=(10, 6))
    plt.scatter(all_scores, all_max_ious, s=0.5)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.xlabel('Score')

    plt.ylabel('Max IoU with GT')
    plt.title('Score vs. Max IoU')
    plt.grid(True)
    save_path1 = os.path.join(save_dir,
                              f'score_vs_iou_{cfg["dataset"]["inference_split"]}_{id}_{cfg["semi_train_cfg"]["cls_pseudo_thr"]}.png')
    plt.savefig(save_path1)

    plt.figure(figsize=(10, 6))
    plt.scatter(all_scores, all_l2_distance, s=0.5)
    plt.xlim(0, 1)
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.xlabel('Score')

    plt.ylabel('l2_distance with GT')
    plt.title('Score vs. l2_distance')
    plt.grid(True)
    save_path2 = os.path.join(save_dir,
                              f'Score_vs_l2_distance_{cfg["dataset"]["inference_split"]}_{id}_{cfg["semi_train_cfg"]["cls_pseudo_thr"]}.png')

    plt.savefig(save_path2)

    plt.figure(figsize=(10, 6))
    plt.scatter(all_depth_score, all_max_ious, s=0.5)
    plt.xlim(0, 1)
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.xlabel('Depth Score')

    plt.ylabel('IOU')
    plt.title('Depth Score vs. IOU')
    plt.grid(True)
    save_path3 = os.path.join(save_dir,
                              f'Depth Score vs. IOU_{cfg["dataset"]["inference_split"]}_{id}_{cfg["semi_train_cfg"]["cls_pseudo_thr"]}.png')

    plt.savefig(save_path3)

if __name__ == '__main__':
    main()