
import warnings

import cv2
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
# import shutil
# import datetime
# import ast
# import cv2
# import time

from tools.semi_base3d_glip import SemiBase3DDetector
# from tools.semi_base3d_clip import SemiBase3DDetector
# from tools.semi_base3d import SemiBase3DDetector

from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
# from utils.iou2d_utils import bbox_iou
from utils.box_ops import box_iou
from visual.Object_pred import Object3d_pred
from visual.kitti_object import show_image_with_boxes, show_lidar_topview_with_boxes
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
    save_TNFP_dir = 'TNFP'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_TNFP_dir, exist_ok=True)
    if cfg.get("visualize",False):
        # 检查文件夹是否已经存在
        if not os.path.exists('outputs_visual'):
        # 创建新的文件夹
            os.mkdir('outputs_visual')
    checkpoint = cfg["trainer"].get("pretrain_model", None)
    IOU_thr_glip =  cfg["semi_train_cfg"].get("IOU_thr", 0.7)
    print("start statistics:")
    print(f"loading from CONFIG {checkpoint}")
    unlabeled_dataset = KITTI_Dataset(split=cfg["dataset"]["inference_split"], cfg=cfg['dataset'])
    subset = Subset(unlabeled_dataset, range(3769))     # 3712 3769 14940 40404  (4,5)->id=
    # subset = Subset(unlabeled_dataset, range(100))
    loader = DataLoader(dataset=subset,
                        batch_size=1,
                        num_workers=1,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False,
                        persistent_workers=True)
    model = SemiBase3DDetector\
        (cfg, cfg['model'], loader, cfg["semi_train_cfg"], cfg["semi_test_cfg"], inference_set=subset.dataset).to('cuda')
    # if checkpoint is not None:
    #     ckpt = torch.load(checkpoint)
    #     model.load_state_dict(ckpt['state_dict'])

    all_gts = 0
    all_preds = 0
    # GT中的正负面对象：
    pos_in_gt = 0
    neg_in_gt = 0
    # 预测的正负面对象：
    pos_in_pred = 0
    neg_in_pred = 0

    all_TP = 0
    all_TP_2d = 0
    all_scores = []
    all_max_ious = []
    all_l2_distance = []
    all_depth_score = []
    all_pred_depth_and_cls_scores = []

    progress_bar = tqdm(loader)
    for inputs, calib, targets, info in progress_bar:
        TP_pic = 0
        GT_pic = 0
        FP_pic = 0
        FN_pic = 0
        wrong_labels = []
        input_teacher = inputs[1]
        input_teacher = input_teacher.to("cuda")
        calib = calib.to("cuda")
        id = int(info['img_id'])
        # print(f"image idx:  {id}")
        info['img_size'] = info['img_size'].to("cuda")
        calibs_from_file = subset.dataset.get_calib(id)
        boxes_lidar, score, loc_list, depth_score_list, score_list, pseudo_labels_list, boxes_2d_from_model,dets_img = \
            model.teacher(input_teacher, calib, targets, info, mode='statistics')
        pseudo_labels_list = pseudo_labels_list[0].tolist()
        gt_objects = unlabeled_dataset.get_label(id)
        gts = []
        labels_gt = []
        labels_pred = pseudo_labels_list
        if cfg.get("visualize",False):
            img = subset.dataset.get_image(id)
            img_from_file = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            calibs_from_file = subset.dataset.get_calib(id)
            pc_velo = subset.dataset.get_lidar(id)
        
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
        gt_boxes_2d = []
        loc_gts = []
        calibs_gt = [subset.dataset.get_calib(index) for index in info['img_id']]
        calib_gt = calibs_gt[0]
        for gt in gts:
            if gt.cls_type == "Car":
                # pos_in_gt = pos_in_gt + 1
                loc = gt.pos.reshape((1, -1))
                loc_gts.append(torch.tensor(loc))
                h = np.array([gt.h]).reshape((1, -1))
                w = np.array([gt.w]).reshape((1, -1))
                l = np.array([gt.l]).reshape((1, -1))
                ry = np.array([gt.ry]).reshape((1, -1))
                gt_box_2d = gt.box2d
                loc_lidar = calib_gt.rect_to_lidar(loc)
                loc_lidar[:, 2] += h[:, 0] / 2
                heading = -(np.pi / 2 + ry)
                gt_lidar = np.concatenate([loc_lidar, l, w, h, heading], axis=1)
                gt_lidar = torch.from_numpy(gt_lidar)
                gt_boxes.append(gt_lidar)
                gt_box_2d = torch.from_numpy(gt_box_2d)
                gt_boxes_2d.append(gt_box_2d)
                labels_gt.append(1.)
        if gt_boxes_2d:
            gt_boxes_2d = torch.stack(gt_boxes_2d).to('cuda')
            gt_boxes_2d = gt_boxes_2d.squeeze(1).float()


        if gt_boxes:
            gt_boxes = torch.stack(gt_boxes).to('cuda')
            gt_boxes = gt_boxes.squeeze(1).float()
        else:
            # 该图为负样本图
            num_gt = 0
            all_gts = all_gts + num_gt
            if boxes_lidar is None:
                num_pre = 0
            else:
                num_pre = boxes_lidar.shape[0]
            all_preds = all_preds + num_pre
            description = f"image idx: {id} | all_gts: {all_gts} | all_preds: {all_preds} | all_TP: {all_TP} | all_TP_2d: {all_TP_2d}"
            progress_bar.set_description(description)
            if cfg.get("visualize",False):
                objects = []
                for det in dets_img:
                    object = Object3d_pred(det.detach().cpu().numpy())
                    objects.append(object)
                img_bbox2d = show_image_with_boxes(img_from_file, objects, calibs_from_file, color=(0, 0, 255), mode="2D")
                img_bbox3d = show_image_with_boxes(img_from_file, objects, calibs_from_file, color=(0, 0, 255), mode="3D")
                if (cfg["dataset"]["inference_split"] in ['test']):
                    img_bev = show_lidar_topview_with_boxes(pc_velo, objects, calibs_from_file)
                if (cfg["dataset"]["inference_split"] not in ['test', 'eigen_clean', 'raw_mix']):
                    img_bbox2d = show_image_with_boxes(img_bbox2d,gts, calibs_from_file, color=(0, 255, 0),
                                                    mode="2D")
                    img_bbox3d = show_image_with_boxes(img_bbox3d,gts, calibs_from_file, color=(0, 255, 0),
                                                    mode="3D")
                    img_bev = show_lidar_topview_with_boxes(pc_velo,gts, calibs_from_file,
                                                            objects_pred=objects)
                cv2.imwrite(f'outputs_visual/KITTI_{cfg["dataset"]["inference_split"]}_{id}_2d.png', img_bbox2d)
                cv2.imwrite(f'outputs_visual/KITTI_{cfg["dataset"]["inference_split"]}_{id}_3d.png', img_bbox3d)
                if (cfg["dataset"]["inference_split"] not in ['eigen_clean', 'raw_mix']):
                    cv2.imwrite(f'outputs_visual/KITTI_{cfg["dataset"]["inference_split"]}_{id}_bev.png', img_bev)
            continue
        if boxes_lidar is None:
            # 预测为0 ， gt不是0
            num_pre = 0
            num_gt = gt_boxes.shape[0]
            all_preds = all_preds + num_pre
            all_gts = all_gts + num_gt
            description = f"image idx: {id} | all_gts: {all_gts} | all_preds: {all_preds} | all_TP: {all_TP} | all_TP_2d: {all_TP_2d}"
            progress_bar.set_description(description)
            if cfg.get("visualize",False):
                objects = []
                for det in dets_img:
                    object = Object3d_pred(det.detach().cpu().numpy())
                    objects.append(object)
                img_bbox2d = show_image_with_boxes(img_from_file, objects, calibs_from_file, color=(0, 0, 255), mode="2D")
                img_bbox3d = show_image_with_boxes(img_from_file, objects, calibs_from_file, color=(0, 0, 255), mode="3D")
                if (cfg["dataset"]["inference_split"] in ['test']):
                    img_bev = show_lidar_topview_with_boxes(pc_velo, objects, calibs_from_file)
                if (cfg["dataset"]["inference_split"] not in ['test', 'eigen_clean', 'raw_mix']):
                    img_bbox2d = show_image_with_boxes(img_bbox2d,gts, calibs_from_file, color=(0, 255, 0),
                                                    mode="2D")
                    img_bbox3d = show_image_with_boxes(img_bbox3d,gts, calibs_from_file, color=(0, 255, 0),
                                                    mode="3D")
                    img_bev = show_lidar_topview_with_boxes(pc_velo,gts, calibs_from_file,
                                                            objects_pred=objects)
                cv2.imwrite(f'outputs_visual/KITTI_{cfg["dataset"]["inference_split"]}_{id}_2d.png', img_bbox2d)
                cv2.imwrite(f'outputs_visual/KITTI_{cfg["dataset"]["inference_split"]}_{id}_3d.png', img_bbox3d)
                if (cfg["dataset"]["inference_split"] not in ['eigen_clean', 'raw_mix']):
                    cv2.imwrite(f'outputs_visual/KITTI_{cfg["dataset"]["inference_split"]}_{id}_bev.png', img_bev)
            continue

        boxes_lidar = boxes_lidar.float().to('cuda')
        iou3D = boxes_iou3d_gpu(boxes_lidar, gt_boxes)  # [num_pre, num_gt]
        # iou3D = boxes_iou3d_gpu(gt_boxes, boxes_lidar)

        iou2D,_= box_iou(boxes_2d_from_model, gt_boxes_2d)
        # iou2D = bbox_iou(boxes_2d_from_model, gt_boxes_2d)
        max_iou_values_2d, max_iou_indices_2d = torch.max(iou2D, dim=1)
        valid_indices_2d = [idx for idx, val in enumerate(max_iou_values_2d.cpu().numpy()) if val > 0.7]  # [1,2,0]
        idx_selected_2d = []
        for idx in valid_indices_2d:
            pred_label_2d = pseudo_labels_list[idx]
            gt_label_2d = labels_gt[max_iou_indices_2d[idx]]
            if max_iou_indices_2d[idx] not in idx_selected_2d:
                if pred_label_2d == gt_label_2d:
                    all_TP_2d = all_TP_2d + 1
                    TP_pic = TP_pic + 1
                    idx_selected_2d.append(max_iou_indices_2d[idx])
                else:
                    wrong_labels.append(pred_label_2d)


        num_pre = boxes_lidar.shape[0]
        num_gt = gt_boxes.shape[0]
        all_preds = all_preds + num_pre
        all_gts = all_gts + num_gt
        # max_iou_per_pred = torch.max(iou3D, dim=1)[0]
        max_iou_values, max_iou_indices = torch.max(iou3D, dim=1)
        valid_indices = [idx for idx, val in enumerate(max_iou_values.cpu().numpy()) if val > 0.7]   # [1,2,0]
        filtered_scores = [score.cpu().numpy()[i] for i in valid_indices]
        pred_depth_scores = [depth_score_list.cpu().numpy()[i] for i in valid_indices]
        pred_depth_and_cls_scores = [score_list.cpu().numpy()[i] for i in valid_indices]
        filtered_max_ious = [max_iou_values.cpu().numpy()[i] for i in valid_indices]
        idx_selected = []
        for idx in valid_indices:
            pred_label = pseudo_labels_list[idx]
            gt_label = labels_gt[max_iou_indices[idx]]
            if max_iou_indices[idx] not in idx_selected:
                if pred_label == gt_label:
                    all_TP = all_TP + 1
                    idx_selected.append(max_iou_indices[idx])

        all_scores.extend(filtered_scores)
        all_max_ious.extend(filtered_max_ious)
        all_depth_score.extend(pred_depth_scores)
        all_pred_depth_and_cls_scores.extend(pred_depth_and_cls_scores)
        description = f"image idx: {id} | all_gts: {all_gts} | all_preds: {all_preds} | all_TP: {all_TP} | all_TP_2d: {all_TP_2d}"
        progress_bar.set_description(description)
        PRED_pic = num_pre
        GT_pic = num_gt
        FP_pic = num_pre - TP_pic
        FN_pic = num_gt - TP_pic
        IOU_thr_glip=cfg["semi_train_cfg"].get("IOU_thr", 0.5)
        filename = f"{save_TNFP_dir}/TNFP_in_one_picture_IOU_{IOU_thr_glip}.txt"
        with open(filename, "a") as file:
            file.write(f"Image Index---{id}    GT---{GT_pic}    PRED---{PRED_pic}    TP: {TP_pic}    FP: {FP_pic}    FN: {FN_pic}    wronglist:{wrong_labels}\n")

        if cfg.get("visualize",False):
            objects = []
            for det in dets_img:
                object = Object3d_pred(det.detach().cpu().numpy())
                objects.append(object)
            img_bbox2d = show_image_with_boxes(img_from_file, objects, calibs_from_file, color=(0, 0, 255), mode="2D")
            img_bbox3d = show_image_with_boxes(img_from_file, objects, calibs_from_file, color=(0, 0, 255), mode="3D")
            if (cfg["dataset"]["inference_split"] in ['test']):
                img_bev = show_lidar_topview_with_boxes(pc_velo, objects, calibs_from_file)
            if (cfg["dataset"]["inference_split"] not in ['test', 'eigen_clean', 'raw_mix']):
                img_bbox2d = show_image_with_boxes(img_bbox2d,gts, calibs_from_file, color=(0, 255, 0),
                                                mode="2D")
                img_bbox3d = show_image_with_boxes(img_bbox3d,gts, calibs_from_file, color=(0, 255, 0),
                                                mode="3D")
                img_bev = show_lidar_topview_with_boxes(pc_velo,gts, calibs_from_file,
                                                        objects_pred=objects)
            cv2.imwrite(f'outputs_visual/KITTI_{cfg["dataset"]["inference_split"]}_{id}_2d.png', img_bbox2d)
            cv2.imwrite(f'outputs_visual/KITTI_{cfg["dataset"]["inference_split"]}_{id}_3d.png', img_bbox3d)
            if (cfg["dataset"]["inference_split"] not in ['eigen_clean', 'raw_mix']):
                cv2.imwrite(f'outputs_visual/KITTI_{cfg["dataset"]["inference_split"]}_{id}_bev.png', img_bev)


    print(f"all_gts  --  {all_gts}")
    print(f"all_preds  --  {all_preds}")
    print(f"all_TP  --  {all_TP} -- all_TP_2d  --   {all_TP_2d}")
    all_FP = all_preds - all_TP
    all_FP_2d = all_preds - all_TP_2d
    print(f"all_FP  --  {all_FP} -- all_FP_2d  --  {all_FP_2d}")
    all_FN = all_gts - all_TP
    all_FN_2d = all_gts - all_TP_2d
    print(f"all_FN  --  {all_FN} -- all_FN_2d  --  {all_FN_2d}")
    print(len(all_depth_score))

if __name__ == '__main__':
    main()