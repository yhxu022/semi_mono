
import warnings

import numpy as np
import os
import torch.utils.data as data
from PIL import Image, ImageFile
import random
warnings.filterwarnings("ignore")

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
import torch
import yaml
import argparse
# import shutil
# import datetime
# import ast
# import cv2
# import time
from tools.semi_base3d import SemiBase3DDetector
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
from lib.helpers.utils_helper import set_random_seed


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
    seed=1
    set_random_seed(seed)
    os.makedirs('mask_vis',exist_ok=True)
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config_name, _ = os.path.splitext(os.path.basename(args.config))

    # checkpoint = cfg["trainer"].get("pretrain_model", None)
    prefix = cfg["dataset"]["inference_split"]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    print("Visualize mask:")
    # print(f"loading from {checkpoint}")
    unlabeled_dataset = KITTI_Dataset(split=cfg["dataset"]["inference_split"], cfg=cfg['dataset'])
    subset = Subset(unlabeled_dataset, range(1,50))  # 3712 3769 14940 40404  4,5
    loader = DataLoader(dataset=subset,
                        batch_size=1,
                        num_workers=1,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False,
                        persistent_workers=True)
    model = SemiBase3DDetector \
        (cfg, cfg['model'], loader, cfg["semi_train_cfg"], cfg["semi_test_cfg"], inference_set=subset.dataset).to(
        'cuda')
    # if checkpoint is not None:
    #     ckpt = torch.load(checkpoint)
    #     model.load_state_dict(ckpt['state_dict'])
    for inputs, calib, targets, info in tqdm(loader):
        input_teacher = inputs[1]
        input_student = inputs[0]
        # print(input_teacher.shape,input_student.shape)
        input_teacher = input_teacher.to("cuda")
        calib = calib.to("cuda")
        id = int(info['img_id'])
        # print(f"image idx:  {id}")
        info['img_size'] = info['img_size'].to("cuda")
        # calibs_from_file = subset.dataset.get_5calib(id)
        pseudo_targets_list, mask, cls_score_list ,topk_boxes, pseudo_targets_to_mask_list, mask2dbox_list = model.teacher(
            input_teacher, calib, targets, info, mode='get_pseudo_targets')

        student_inputs_with_2dmask = model.mask_input_for_student(input_student, pseudo_targets_to_mask_list,
                                                                 mask2dbox_list, info)
        img = input_student[0].cpu().numpy().transpose(1, 2, 0)
        img = (img * std + mean) * 255
        img = Image.fromarray(img.astype(np.uint8))
        path1 = os.path.join("mask_vis",f"{prefix}_{id}.png")
        img.save(path1)

        img = student_inputs_with_2dmask[0].cpu().numpy().transpose(1, 2, 0)
        img = (img * std + mean) * 255
        img = Image.fromarray(img.astype(np.uint8))
        path2 = os.path.join("mask_vis", f"{prefix}_{id}_mask.png")
        img.save(path2)

if __name__ == '__main__':
    main()
