import warnings

warnings.filterwarnings("ignore")

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import argparse
import datetime

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
    # output_path = os.path.join('./' + cfg["trainer"]['save_path'], model_name)
    output_path = os.path.join('./' + 'outputs', config_name)
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(output_path, 'train.log')
    logger = MMLogger.get_instance('mmengine', log_file=log_file, log_level='INFO')
    # build dataloader
    train_set, test_loader, sampler = build_dataloader(cfg['dataset'])

    # if args.evaluate_only:
    #     logger.info('###################  Evaluation Only  ##################')
    #     tester = Tester(cfg=cfg['tester'],
    #                     model=model,
    #                     dataloader=test_loader,
    #                     logger=logger,
    #                     train_cfg=cfg['trainer'],
    #                     model_name=model_name)
    #     tester.test()
    #     return
    # ipdb.set_trace()
    checkpoint = cfg["trainer"].get("pretrain_model", None)
    if cfg['dataset']["train_split"] == "semi":
        if checkpoint is not None:
            model = SemiBase3DDetector(cfg, cfg['model'], test_loader, cfg["semi_train_cfg"], cfg["semi_test_cfg"],
                                       init_cfg=dict(type='Pretrained', checkpoint=checkpoint))
        else:
            model = SemiBase3DDetector(cfg, cfg['model'], test_loader, cfg["semi_train_cfg"], cfg["semi_test_cfg"])
        custom_hooks = [
            dict(type="MeanTeacherHook", momentum=cfg["mean_teacher_hook"]["momentum"],
                 interval=cfg["mean_teacher_hook"]["interval"], skip_buffer=cfg["mean_teacher_hook"]["skip_buffer"])
        ]
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
                                                              bypass_duplicate=True)),
                        param_scheduler=dict(type='MultiStepLR',
                                             by_epoch=False,
                                             milestones=cfg["lr_scheduler"]["decay_list"],
                                             gamma=cfg["lr_scheduler"]["decay_rate"]),
                        train_cfg=dict(by_epoch=False,
                                       max_iters=cfg["trainer"]["max_iteration"],
                                       val_interval=cfg["trainer"]["val_iterval"]),
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
                                            save_best="teacher/car_moderate",
                                            rule='greater')),
                        log_processor=dict(window_size=50,
                                           by_epoch=False,
                                           custom_cfg=[
                                               dict(data_src='batch_unsup_pseudo_instances_num',
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
                                                              bypass_duplicate=True)),
                        param_scheduler=dict(type='MultiStepLR',
                                             by_epoch=False,
                                             milestones=cfg["lr_scheduler"]["decay_list"],
                                             gamma=cfg["lr_scheduler"]["decay_rate"]),
                        train_cfg=dict(by_epoch=False,
                                       max_iters=cfg["trainer"]["max_iteration"],
                                       val_interval=cfg["trainer"]["val_iterval"]),
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
                                            save_best="car_moderate",
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
