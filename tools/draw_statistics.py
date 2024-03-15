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
    os.makedirs("outputs_visual", exist_ok=True)

    checkpoint = cfg["trainer"].get("pretrain_model", None)

    print("start statistics:")
    print(f"loading from {checkpoint}")
    unlabeled_dataset = KITTI_Dataset(split=cfg["dataset"]["inference_split"], cfg=cfg['dataset'])
    subset = Subset(unlabeled_dataset, range(3769))     # 3712 3769 14940 40404
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
    for inputs, calib, targets, info in tqdm(loader):
        input_teacher = inputs[1]
        input_teacher = input_teacher.to("cuda")
        calib = calib.to("cuda")
        id = int(info['img_id'])
        # print(id)
        info['img_size'] = info['img_size'].to("cuda")
        # img = subset.dataset.get_image(id)
        # pc_velo = subset.dataset.get_lidar(id)
        boxes_lidar, score, loc_list = model.teacher(input_teacher, calib, targets, info, mode='statistics')
        if boxes_lidar is None:
            continue
        # print(boxes_lidar.shape, score.shape)
        gts = unlabeled_dataset.get_label(id)
        gt_boxes = []
        loc_gts = []
        calibs_gt = [subset.dataset.get_calib(index) for index in info['img_id']]
        calib_gt = calibs_gt[0]
        for gt in gts:
            if gt.cls_type == "Car":
                loc = gt.pos.reshape((1, -1))
                loc_gts.append(loc)
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
        if loc_gts:
            loc_gts = torch.stack(loc_gts).to('cuda')
        boxes_lidar = boxes_lidar.float().to('cuda')
        iou3D = boxes_iou3d_gpu(boxes_lidar, gt_boxes)  # [num_pre, num_gt]
        num_pre = boxes_lidar.shape[0]
        num_gt = gt_boxes.shape[0]
        # print(num_pre, num_gt)
        if num_pre > num_gt:
            iou3D = iou3D.T
        max_iou_per_pred = torch.max(iou3D, dim=1)[0]

        all_scores.extend(score.cpu().numpy())
        all_max_ious.extend(max_iou_per_pred.cpu().numpy())
        print(id)
        print(len(all_scores))
        print(len(all_max_ious))


    print(len(all_scores))
    print(len(all_max_ious))
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
    plt.savefig(f'score_vs_iou_{cfg["dataset"]["inference_split"]}_{id}_{cfg["semi_train_cfg"]["cls_pseudo_thr"]}.png')


if __name__ == '__main__':
    main()