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
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.tester_helper import Tester
from tools.KITTI_METRIC import KITTI_METRIC
from mmengine.runner import Runner
from mmengine.logging import MMLogger
from tools.semi_base3d import SemiBase3DDetector
from tools.Mono_DETR import Mono_DETR
from lib.helpers.model_helper import build_model
from tools.mean_teacher_hook import MeanTeacherHook
from loops import TeacherStudentValLoop
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
from visual.kitti_util import Calibration
from visual.Object_pred import Object3d_pred
from visual.kitti_object import show_image_with_boxes, show_lidar_topview_with_boxes
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import time
from tqdm import tqdm
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
    os.mkdir("statistics")

    checkpoint = cfg["trainer"].get("pretrain_model", None)

    print("start inference and visualize")
    print(f"loading from {checkpoint}")
    unlabeled_dataset = KITTI_Dataset(split=cfg["dataset"]["inference_split"], cfg=cfg['dataset'])
    subset = Subset(unlabeled_dataset, range(100))
    loader = DataLoader(dataset=subset,
                        batch_size=1,
                        num_workers=1,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False,
                        persistent_workers=True)
    model = SemiBase3DDetector(cfg, cfg['model'], loader, cfg["semi_train_cfg"], inference_set=subset.dataset).to(
        'cuda')
    if checkpoint is not None:
        ckpt = torch.load(checkpoint)
        model.load_state_dict(ckpt['state_dict'])
