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
# Mask掉歧义的物体
from tools.semi_base3d_mask import SemiBase3DDetector
# from tools.semi_base3d import SemiBase3DDetector
from tools.Mono_DETR import Mono_DETR
from lib.helpers.model_helper import build_model
from tools.hook.mean_teacher_hook import MeanTeacherHook
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
parser.add_argument('-t', '--to_minute', action='store_true', default=False, help='down to the minute')
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
    # output_path = os.path.join('./' + cfg["trainer"]['save_path'], model_name)
    if args.to_minute is False:
        output_path = os.path.join('./' + 'outputs', config_name + '@' + datetime.datetime.now().strftime('%m%d%H'))
    else:
        output_path = os.path.join('./' + 'outputs', config_name + '@' + datetime.datetime.now().strftime('%m%d%H_%M'))
    os.makedirs(output_path, exist_ok=True)
    shutil.copy(args.config, output_path)
    shutil.copy(os.path.join('tools', 'semi_base3d_mask.py'), output_path)
    shutil.copy(os.path.join('tools', 'Semi_Mono_DETR_mask.py'), output_path)
    log_file = os.path.join(output_path, 'train.log')
    logger = MMLogger.get_instance('mmengine', log_file=log_file, log_level='INFO')
    checkpoint = cfg["trainer"].get("pretrain_model", None)

    if cfg.get('evaluate_only', False):
        os.makedirs("outputs_visual", exist_ok=True)
        print("start inference and visualize")
        print(f"loading from {checkpoint}")
        unlabeled_dataset = KITTI_Dataset(split=cfg["dataset"]["inference_split"], cfg=cfg['dataset'])
        subset = Subset(unlabeled_dataset, range(40404))  # 3712 3769 14940 40404
        loader = DataLoader(dataset=subset,
                            batch_size=1,
                            num_workers=1,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=False,
                            persistent_workers=True)
        model = SemiBase3DDetector(cfg, cfg['model'], loader, cfg["semi_train_cfg"], cfg["semi_test_cfg"],
                                   inference_set=subset.dataset).to('cuda')
        if checkpoint is not None:
            ckpt = torch.load(checkpoint)
            model.load_state_dict(ckpt['state_dict'])
        gt_sum = sum = 0
        for inputs, calib, targets, info in tqdm(loader):
            input_teacher = inputs[1]
            input_teacher = input_teacher.to("cuda")
            calib = calib.to("cuda")
            # targets = targets.to("cuda")
            id = int(info['img_id'])
            info['img_size'] = info['img_size'].to("cuda")
            img = subset.dataset.get_image(id)
            img_from_file = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            calibs_from_file = subset.dataset.get_calib(id)
            pc_velo = subset.dataset.get_lidar(id)
            dets = model.teacher(input_teacher, calib, targets, info, mode='inference')
            if (cfg["dataset"]["inference_split"] not in ['test', 'eigen_clean', 'raw_mix']):
                gt_objects = subset.dataset.get_label(id)
                gt_objects_filtered = []
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
                    gt_objects_filtered.append(gt_objects[i])
            objects = []
            for det in dets:
                object = Object3d_pred(det)
                objects.append(object)
            if len(objects) == 0:
                sum += 1
            if (cfg["dataset"]["inference_split"] not in ['test', 'eigen_clean', 'raw_mix']):
                if len(gt_objects_filtered) == 0:
                    gt_sum += 1
            img_bbox2d = show_image_with_boxes(img_from_file, objects, calibs_from_file, color=(0, 0, 255), mode="2D")
            img_bbox3d = show_image_with_boxes(img_from_file, objects, calibs_from_file, color=(0, 0, 255), mode="3D")
            if (cfg["dataset"]["inference_split"] in ['test']):
                img_bev = show_lidar_topview_with_boxes(pc_velo, objects, calibs_from_file)
            if (cfg["dataset"]["inference_split"] not in ['test', 'eigen_clean', 'raw_mix']):
                img_bbox2d = show_image_with_boxes(img_bbox2d, gt_objects_filtered, calibs_from_file, color=(0, 255, 0),
                                                   mode="2D")
                img_bbox3d = show_image_with_boxes(img_bbox3d, gt_objects_filtered, calibs_from_file, color=(0, 255, 0),
                                                   mode="3D")
                img_bev = show_lidar_topview_with_boxes(pc_velo, gt_objects_filtered, calibs_from_file,
                                                        objects_pred=objects)
            cv2.imwrite(f'outputs_visual/KITTI_{cfg["dataset"]["inference_split"]}_{id}_2d.png', img_bbox2d)
            cv2.imwrite(f'outputs_visual/KITTI_{cfg["dataset"]["inference_split"]}_{id}_3d.png', img_bbox3d)
            if (cfg["dataset"]["inference_split"] not in ['eigen_clean', 'raw_mix']):
                cv2.imwrite(f'outputs_visual/KITTI_{cfg["dataset"]["inference_split"]}_{id}_bev.png', img_bev)
        print("number of no predictions images:", sum)
        if (cfg["dataset"]["inference_split"] not in ['test', 'eigen_clean', 'raw_mix']):
            print("number of no gt images:", gt_sum)
        print("number of total images:", len(subset))
        return
    # build dataloader
    train_set, test_loader, sampler = build_dataloader(cfg['dataset'])
    if cfg['dataset']["train_split"] in ["semi", 'semi_eigen_clean', 'semi_raw_mix']:
        if checkpoint is not None:
            model = SemiBase3DDetector(cfg, cfg['model'], test_loader, cfg["semi_train_cfg"], cfg["semi_test_cfg"],
                                       init_cfg=dict(type='Pretrained', checkpoint=checkpoint))
        else:
            model = SemiBase3DDetector(cfg, cfg['model'], test_loader, cfg["semi_train_cfg"], cfg["semi_test_cfg"])
        custom_hooks = [
            dict(type="MeanTeacherHook", momentum=cfg["mean_teacher_hook"]["momentum"],
                 interval=cfg["mean_teacher_hook"]["interval"], skip_buffer=cfg["mean_teacher_hook"]["skip_buffer"])
        ]
        if cfg.get('two_stages', False) == True:
            if cfg['lr_scheduler'].get('type', None) == 'step':
                if cfg['lr_scheduler'].get('warmup', None) is True:
                    param_scheduler = [dict(type='LinearLR',
                                            start_factor=0.001,
                                            by_epoch=False,
                                            begin=0,
                                            end=500),
                                       dict(type='MultiStepLR',
                                            by_epoch=False,
                                            milestones=cfg["lr_scheduler"]["decay_list"],
                                            gamma=cfg["lr_scheduler"]["decay_rate"]),
                                       ]
                else:
                    param_scheduler = [
                                       dict(type='MultiStepLR',
                                            by_epoch=False,
                                            milestones=cfg["lr_scheduler"]["decay_list"],
                                            gamma=cfg["lr_scheduler"]["decay_rate"]),
                                       ]

            else:
                param_scheduler = [  # 在 [0, 232*10) 迭代时使用线性学习率
                    dict(type='LinearLR',
                         start_factor=0.001,
                         by_epoch=False,
                         begin=0,
                         end=500),
                    # 在 [232*10, cfg["trainer"]["max_iteration"]) 迭代时使用余弦学习率
                    dict(type='CosineAnnealingLR',
                         T_max=cfg["trainer"]["max_iteration"] - 500,
                         by_epoch=False,
                         begin=500,
                         end=cfg["trainer"]["max_iteration"],
                         eta_min_ratio=0.001)
                ]
        else:
            if cfg['lr_scheduler'].get('type', None) == 'step':
                print("use MultiStepLR")
                param_scheduler = dict(type='MultiStepLR',
                                       by_epoch=False,
                                       milestones=cfg["lr_scheduler"]["decay_list"],
                                       gamma=cfg["lr_scheduler"]["decay_rate"]),
            elif cfg['lr_scheduler'].get('type', None) == 'cos':
                print("use CosineAnnealingLR")
                param_scheduler = dict(type='CosineAnnealingLR',
                                       T_max=cfg["trainer"]["max_iteration"] - 10000,
                                       by_epoch=False,
                                       begin=10000,
                                       end=cfg["trainer"]["max_iteration"]),
            else:
                raise RuntimeError("No lr scheduler")
        runner = Runner(model=model,
                        work_dir=output_path,
                        custom_hooks=custom_hooks,
                        train_dataloader=dict(
                            batch_size=cfg["dataset"]['batch_size'],
                            sampler=sampler,
                            dataset=train_set,
                            pin_memory=True,
                            # num_workers=cfg["dataset"]['batch_size'],
                            num_workers=2 * cfg["dataset"]['batch_size'],
                            collate_fn=dict(type='default_collate'),
                            persistent_workers=True
                        ),
                        # train_dataloader = train_loader,
                        optim_wrapper=dict(optimizer=dict(type=cfg["optimizer"]["type"],
                                                          lr=cfg["optimizer"]["lr"],
                                                          weight_decay=cfg["optimizer"]["weight_decay"]
                                                          ),
                                           paramwise_cfg=dict(bias_decay_mult=0,
                                                              norm_decay_mult=0,
                                                              bypass_duplicate=True)),
                        param_scheduler=param_scheduler,
                        train_cfg=dict(by_epoch=False,
                                       max_iters=cfg["trainer"]["max_iteration"],
                                       val_begin=cfg["trainer"].get('val_begin', 1),
                                       val_interval=cfg["trainer"]["val_iterval"],
                                       dynamic_intervals=ast.literal_eval(
                                           (cfg["trainer"].get('dynamic_intervals', 'None')))
                                       ),
                        val_dataloader=test_loader,
                        val_cfg=dict(type='TeacherStudentValLoop'),
                        val_evaluator=dict(type=KITTI_METRIC,
                                           output_dir=output_path,
                                           dataloader=test_loader,
                                           logger=logger,
                                           cfg=cfg),
                        test_dataloader=test_loader,
                        test_cfg=dict(type='TeacherStudentValLoop'),
                        test_evaluator=dict(type=KITTI_METRIC,
                                            output_dir=output_path,
                                            dataloader=test_loader,
                                            logger=logger,
                                            cfg=cfg),
                        default_hooks=dict(
                            logger=dict(type='LoggerHook',
                                        log_metric_by_epoch=False,
                                        interval=50),
                            checkpoint=dict(type='CheckpointHook',
                                            by_epoch=False,
                                            save_best="teacher/Car_3d_moderate_R40",
                                            rule='greater')),
                        log_processor=dict(window_size=50,
                                           by_epoch=False,
                                           custom_cfg=[
                                               dict(data_src='batch_regression_unsup_pseudo_instances_num',
                                                    method_name='mean',
                                                    window_size=50),
                                               dict(data_src='batch_masked_pseudo_instances_num',
                                                    method_name='mean',
                                                    window_size=50),
                                               dict(data_src='batch_cls_unsup_pseudo_instances_num',
                                                    method_name='mean',
                                                    window_size=50),
                                               dict(data_src='batch_unsup_gt_instances_num',
                                                    method_name='mean',
                                                    window_size=50),
                                           ]),
                        randomness=dict(seed=cfg.get('random_seed', 444),
                                        diff_rank_seed=True,
                                        deterministic=cfg.get('deterministic', False)),

                        launcher=args.launcher,
                        cfg=dict(
                            model_wrapper_cfg=dict(
                                type='MMDistributedDataParallel', find_unused_parameters=True))
                        )
    else:
        model, loss = build_model(cfg['model'])
        if (checkpoint is not None):
            model = Mono_DETR(model, loss, cfg, test_loader, init_cfg=dict(type='Pretrained', checkpoint=checkpoint))
        else:
            model = Mono_DETR(model, loss, cfg, test_loader)
        runner = Runner(model=model,
                        work_dir=output_path,
                        train_dataloader=dict(
                            batch_size=cfg["dataset"]['batch_size'],
                            sampler=sampler,
                            dataset=train_set,
                            pin_memory=True,
                            # num_workers=cfg["dataset"]['batch_size'],
                            num_workers=2 * cfg["dataset"]['batch_size'],
                            collate_fn=dict(type='default_collate'),
                            persistent_workers=True
                        ),
                        optim_wrapper=dict(optimizer=dict(type=cfg["optimizer"]["type"],
                                                          lr=cfg["optimizer"]["lr"],
                                                          weight_decay=cfg["optimizer"]["weight_decay"]
                                                          ),
                                           paramwise_cfg=dict(bias_decay_mult=0,
                                                              norm_decay_mult=0,
                                                              bypass_duplicate=True)),
                        param_scheduler=dict(type='MultiStepLR',
                                             by_epoch=False,
                                             milestones=cfg["lr_scheduler"]["decay_list"],
                                             gamma=cfg["lr_scheduler"]["decay_rate"]),
                        train_cfg=dict(by_epoch=False,
                                       max_iters=cfg["trainer"]["max_iteration"],
                                       val_begin=cfg["trainer"].get('val_begin', 1),
                                       val_interval=cfg["trainer"]["val_iterval"],
                                       dynamic_intervals=ast.literal_eval(
                                           (cfg["trainer"].get('dynamic_intervals', 'None')))
                                       ),
                        val_dataloader=test_loader,
                        val_cfg=dict(),
                        val_evaluator=dict(type=KITTI_METRIC,
                                           output_dir=output_path,
                                           dataloader=test_loader,
                                           logger=logger,
                                           cfg=cfg),
                        test_dataloader=test_loader,
                        test_cfg=dict(),
                        test_evaluator=dict(type=KITTI_METRIC,
                                            output_dir=output_path,
                                            dataloader=test_loader,
                                            logger=logger,
                                            cfg=cfg),
                        default_hooks=dict(
                            logger=dict(type='LoggerHook',
                                        log_metric_by_epoch=False,
                                        interval=50),
                            checkpoint=dict(type='CheckpointHook',
                                            by_epoch=False,
                                            save_best="Car_3d_moderate_R40",
                                            rule='greater')),
                        log_processor=dict(window_size=50,
                                           by_epoch=False),
                        randomness=dict(seed=cfg.get('random_seed', 444),
                                        diff_rank_seed=True,
                                        deterministic=cfg.get('deterministic', False)),
                        launcher=args.launcher,
                        cfg=dict(
                            model_wrapper_cfg=dict(
                                type='MMDistributedDataParallel', find_unused_parameters=True))
                        )
    runner.train()

    if cfg['dataset']['test_split'] == 'test':
        return

    logger.info('###################  Testing  ##################')
    logger.info('Batch Size: %d' % (cfg['dataset']['batch_size']))
    logger.info('Split: %s' % (cfg['dataset']['test_split']))
    runner.test()


if __name__ == '__main__':
    main()
