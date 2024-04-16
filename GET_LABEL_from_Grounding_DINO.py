from tools.glip_kitti import Glip_Kitti
import torch
from PIL import Image
from tqdm import tqdm
import os
import sys
from groundingdino.util.inference import load_model, load_image, predict, annotate,predict_with_tokenized
import cv2
from utils.iou2d_utils import bbox_iou
from torchvision.ops import box_convert
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
import torch
import yaml
import argparse

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

if __name__ == "__main__":
    inference_on = 'train'
    device = 'cuda'
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config_name, _ = os.path.splitext(os.path.basename(args.config))
    save_dir = f"/home/xyh/MonoDETR_ori/data/KITTI/{inference_on}ing/label_from_GD"
    os.makedirs(save_dir, exist_ok=True)
    assert (cfg["dataset"]["inference_split"] is inference_on)
    GLIP = Glip_Kitti()

    unlabeled_dataset = KITTI_Dataset(split=cfg["dataset"]["inference_split"], cfg=cfg['dataset'])
    subset = Subset(unlabeled_dataset, range(3712))  # 3712 3769 14940 40404  (4,5)->id=
    # subset = Subset(unlabeled_dataset, range(100))
    loader = DataLoader(dataset=subset,
                        batch_size=1,
                        num_workers=1,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False,
                        persistent_workers=True)
    progress_bar = tqdm(loader)
    for inputs, calib, targets, info in progress_bar:
        inputs = inputs.to(device)
        boxes_from_gd, logits, phrases = GLIP.predict(inputs, device=inputs.device)
        obj_num = boxes_from_gd.shape[0]
        idx = int(info['img_id'])
        file_path = os.path.join(save_dir, '%06d.txt' % idx)
        cls_type_list = phrases
        w = 1280
        h = 384

        if obj_num == 0:
            open(file_path, 'w').close()
            continue
        else:
            size = torch.tensor([w, h, w, h], device=device)
            size = size.to(device)
            boxes_from_gd = boxes_from_gd.to(device)
            bbox = boxes_from_gd * size

            with open(file_path, 'w') as file:
                for i in range(obj_num):
                    cls_type = cls_type_list[i]
                    trucation = -1
                    occluded = -1
                    alpha = data['alpha'][i]
                    bbox2d = data['bbox'][i]
                    l = data['dimensions'][i][0]
                    h = data['dimensions'][i][1]
                    w = data['dimensions'][i][2]
                    pos = data['location'][i]
                    ry = data['rotation_y'][i]
                    score = data['score'][i]
                    file.write(
                        f"{cls_type} {trucation} {occluded} {alpha} {bbox2d[0]} {bbox2d[1]} {bbox2d[2]} {bbox2d[3]} {h} {w} {l} {pos[0]} {pos[1]} {pos[2]} {ry} {score}\n")












