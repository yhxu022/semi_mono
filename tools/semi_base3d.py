# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.transforms import ToPILImage
import numpy as np
from .misc import (filter_gt_instances, rename_loss_dict,
                   reweight_loss_dict)
# from mmdet3d.registry import MODELS
from mmengine import MessageHub
# from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig, SampleList
from mmengine.model import BaseModel
# from mmdet3d.models import Base3DDetector
from lib.helpers.model_helper import build_model
from tools.Semi_Mono_DETR import Semi_Mono_DETR


def prepare_targets(targets, batch_size):
    targets_list = []
    mask = targets['mask_2d']

    key_list = ['labels', 'boxes', 'calibs', 'depth', 'size_3d', 'heading_bin', 'heading_res', 'boxes_3d']
    for bz in range(batch_size):
        target_dict = {}
        for key, val in targets.items():
            if key in key_list:
                target_dict[key] = val[bz][mask[bz]]
        targets_list.append(target_dict)
    return targets_list


# @MODELS.register_module()
class SemiBase3DDetector(BaseModel):
    """Base class for semi-supervised detectors.

    Semi-supervised detectors typically consisting of a teacher model
    updated by exponential moving average and a student model updated
    by gradient descent.

    Args:
        detector (:obj:`ConfigDict` or dict): The detector config.
        semi_train_cfg (:obj:`ConfigDict` or dict, optional):
            The semi-supervised training config.
        semi_test_cfg (:obj:`ConfigDict` or dict, optional):
            The semi-supervised testing config.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 cfg,
                 model_cfg,
                 test_loader,
                 semi_train_cfg=None,
                 semi_test_cfg=None,
                 init_cfg=None,
                 inference_set=None) -> None:
        super().__init__(data_preprocessor=None, init_cfg=init_cfg)
        # build model
        student_model, student_loss = build_model(model_cfg)
        teacher_model, teacher_loss = build_model(model_cfg)
        #支持加载MonoDETR官方训练权重
        student_model.load_state_dict(torch.load("/home/xyh/MonoDETR_semi_baseline_33/ckpts/MonoDETR_pretrained_100.pth")['model_state'])
        teacher_model.load_state_dict(torch.load("/home/xyh/MonoDETR_semi_baseline_33/ckpts/MonoDETR_pretrained_100.pth")['model_state'])
        # check_point=torch.load('/data/ipad_3d/monocular/semi_mono/outputs/monodetr_4gpu_origin_30pc/best_car_moderate_iter_33408.pth')["state_dict"]
        # ckpt={k.replace('model.', ''): v for k, v in check_point.items()}
        # student_model.load_state_dict(ckpt)
        # teacher_model.load_state_dict(ckpt)
        self.student = Semi_Mono_DETR(student_model, student_loss, cfg, test_loader, inference_set)
        self.teacher = Semi_Mono_DETR(teacher_model, teacher_loss, cfg, test_loader, inference_set)
        self.semi_train_cfg = semi_train_cfg
        self.semi_test_cfg = semi_test_cfg
        self.sup_size = semi_train_cfg["sup_size"]
        self.student.pseudo_label_group_num = self.teacher.pseudo_label_group_num = self.semi_train_cfg.get(
            "pseudo_label_group_num", 1)
        self.student.val_nms = self.teacher.val_nms = self.semi_test_cfg.get('nms', False)
        if self.semi_train_cfg.get('freeze_teacher', True) is True:
            self.freeze(self.teacher)

    def forward(self, inputs, calibs, targets, info, mode):
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`Det3DDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs  (dict | list[dict]): When it is a list[dict], the
                outer list indicate the test time augmentation. Each
                dict contains batch inputs
                which include 'points' and 'imgs' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor): Image tensor has shape (B, C, H, W).
            data_samples (list[:obj:`Det3DDataSample`],
                list[list[:obj:`Det3DDataSample`]], optional): The
                annotation data of every samples. When it is a list[list], the
                outer list indicate the test time augmentation, and the
                inter list indicate the batch. Otherwise, the list simply
                indicate the batch. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`Det3DDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, calibs, targets, info)
        elif mode == 'predict':
            # if isinstance(data_samples[0], list):
            #     # aug test
            #     assert len(data_samples[0]) == 1, 'Only support ' \
            #                                       'batch_size 1 ' \
            #                                       'in mmdet3d when ' \
            #                                       'do the test' \
            #                                       'time augmentation.'
            #     return self.aug_test(inputs, data_samples, **kwargs)
            # else:
            return self.predict(inputs, calibs, targets, info)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    @staticmethod
    def freeze(model: nn.Module):
        """Freeze the model."""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def loss(self, inputs, calibs, targets, info):
        """Calculate losses from multi-branch inputs and data samples.

        Args:
            multi_batch_inputs (Dict[str, Tensor]): The dict of multi-branch
                input images, each value with shape (N, C, H, W).
                Each value should usually be mean centered and std scaled.
            multi_batch_data_samples (Dict[str, List[:obj:`Det3DDataSample`]]):
                The dict of multi-branch data samples.

        Returns:
            dict: A dictionary of loss components
        """
        sup_inputs, student_inputs, teacher_inputs = inputs[0][:self.sup_size], inputs[0][self.sup_size:], inputs[1][
                                                                                                           self.sup_size:]
        # student_image=student_inputs[0].cpu().numpy().transpose(1, 2, 0)
        # teacher_image=teacher_inputs[0].cpu().numpy().transpose(1, 2, 0)
        # mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        # std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        # student_image = (student_image * std + mean) * 255.0
        # teacher_image = (teacher_image * std + mean) * 255.0
        # student_image=ToPILImage()(np.round(student_image).astype(np.uint8))
        # teacher_image=ToPILImage()(np.round(teacher_image).astype(np.uint8))
        # student_image.save("student_image.jpg")
        # teacher_image.save("teacher_image.jpg")
        sup_calibs, unsup_calibs = calibs[:self.sup_size], calibs[self.sup_size:]
        sup_targets = {k: v[:self.sup_size] for k, v in targets.items()}
        unsup_targets = {k: v[self.sup_size:] for k, v in targets.items()}
        sup_info = {k: v[:self.sup_size] for k, v in info.items()}
        unsup_info = {k: v[self.sup_size:] for k, v in info.items()}
        losses = dict()
        losses.update(**self.loss_by_gt_instances(
            sup_inputs, sup_calibs, sup_targets, sup_info))
        labels = unsup_targets["labels"]
        masks = unsup_targets["mask_2d"]
        unsup_gt_instances_num = sum([
            len(label[mask])
            for label, mask in zip(labels, masks)
        ])
        message_hub = MessageHub.get_current_instance()
        message_hub.update_scalar('train/batch_unsup_gt_instances_num', unsup_gt_instances_num)
        pseudo_targets_list, mask, cls_score, topk_boxes = self.get_pseudo_targets(
            teacher_inputs, unsup_calibs, unsup_targets, unsup_info)
        # 用伪标签监督
        losses.update(**self.loss_by_pseudo_instances(
            student_inputs, unsup_calibs, pseudo_targets_list, mask, cls_score, topk_boxes, unsup_info))

        # 用GT监督
        # unsup_gt_targets_list = prepare_targets(unsup_targets, student_inputs.shape[0])
        # losses.update(**self.loss_by_pseudo_instances(
        #     student_inputs, unsup_calibs, unsup_gt_targets_list, mask, cls_score, unsup_info))
        return losses

    def loss_by_gt_instances(self,
                             sup_inputs,
                             sup_calibs,
                             sup_targets,
                             sup_info):
        """Calculate losses from a batch of inputs and ground-truth data
        samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """

        losses = self.student.forward(sup_inputs, sup_calibs, sup_targets, sup_info, mode='loss')
        sup_weight = self.semi_train_cfg.get('sup_weight', 1.)
        return rename_loss_dict('sup_', reweight_loss_dict(losses, sup_weight))

    def loss_by_pseudo_instances(self,
                                 unsup_inputs, unsup_calibs, pseudo_targets_list, mask, cls_score, topk_boxes,
                                 unsup_info, unsupweight_from_hook=None) -> dict:
        """Calculate losses from a batch of inputs and pseudo data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`,
                which are `pseudo_instance` or `pseudo_panoptic_seg`
                or `pseudo_sem_seg` in fact.
            batch_info (dict): Batch information of teacher model
                forward propagation process. Defaults to None.

        Returns:
            dict: A dictionary of loss components
        """
        losses = self.student.forward(unsup_inputs, unsup_calibs, pseudo_targets_list, unsup_info, mode='unsup_loss')
        unsup_pseudo_instances_num = sum([
            len(pseudo_targets["labels"])
            for pseudo_targets in pseudo_targets_list
        ])
        message_hub = MessageHub.get_current_instance()
        message_hub.update_scalar('train/batch_unsup_pseudo_instances_num', unsup_pseudo_instances_num)
        if unsupweight_from_hook is None:
            unsup_weight = self.semi_train_cfg.get(
                'unsup_weight', 1.) if unsup_pseudo_instances_num > 0 else 0.
        losses = reweight_loss_dict(losses, unsup_weight)

        # 与教师模型每一层的输出计算一致性损失
        # consistency_loss = self.consistency_loss(self.student.model.hs,self.teacher.model.hs,mask,cls_score,topk_boxes,self.student.loss.indices)
        # 不加一致性损失
        consistency_loss = torch.tensor(0.).to(self.student.model.hs.device)
        # 与教师模型最后一层的输出计算一致性损失
        # consistency_loss = self.consistency_loss(self.student.model.hs[[2]], self.teacher.model.hs[[2]], mask,
        #  cls_score, topk_boxes,self.student.loss.indices)

        losses.update({"consistency_loss": consistency_loss * self.semi_train_cfg.get(
            'consistency_weight', 1.)})
        unsup_loss_dict = rename_loss_dict('unsup_',
                                           losses)
        for name, loss in unsup_loss_dict.items():
            # 所有unsup深度loss置零
            # if 'loss_depth' in name:
            #     unsup_loss_dict[name] = unsup_loss_dict[name] * 0.
            # unsup深度loss置零,保留depth_map loss
            # if 'loss_depth' in name and "loss_depth_map" not in name:
            #     unsup_loss_dict[name] = unsup_loss_dict[name] * 0.
            #将unsup分类损失和中心点损失置零
            # if 'loss_ce' in name:
            #     unsup_loss_dict[name] = unsup_loss_dict[name] * 0.
            #将unsup分类损失置零
            if 'loss_ce' in name and 'loss_center' not in name:
                unsup_loss_dict[name] = unsup_loss_dict[name] * 0.
        return unsup_loss_dict

    def consistency_loss(self,
                         student_decoder_outputs,
                         teacher_decoder_outputs,
                         masks,
                         cls_score_list,
                         topk_boxes,
                         idx):
        consistency_loss = torch.tensor(0.).to(student_decoder_outputs.device)
        batchsize = student_decoder_outputs.shape[1]
        levels = student_decoder_outputs.shape[0]
        cls_score_list = cls_score_list.gather(1, topk_boxes.squeeze(2))
        teacher_decoder_outputs = teacher_decoder_outputs.gather(2, topk_boxes.repeat(levels, 1, 1, 256))
        for i in range(batchsize):
            student_decoder_output = student_decoder_outputs[:, i, :, :]
            teacher_decoder_output = teacher_decoder_outputs[:, i, :, :]
            mask = masks[i]
            cls_score = cls_score_list[i]
            if mask.sum() == 0:
                consistency_loss += 0.0
            else:
                teacher_decoder_output = teacher_decoder_output[:, mask, :]
                for level in range(levels):
                    indices = idx[level][i]
                    student_decoder_output_level = student_decoder_output[level]
                    teacher_decoder_output_level = teacher_decoder_output[level]
                    student_decoder_output_level = student_decoder_output_level[indices[0]]
                    teacher_decoder_output_level = teacher_decoder_output_level[indices[1]]
                    # cls_score_level=torch.ones_like(cls_score[mask][indices[1]].unsqueeze(1))
                    cls_score_level = cls_score[mask][indices[1]].unsqueeze(1)
                    delta = student_decoder_output_level - teacher_decoder_output_level.detach()
                    delta = delta.square()
                    consistency = delta * cls_score_level
                    a = consistency.mean()
                    b = torch.nn.functional.mse_loss(student_decoder_output_level,
                                                     teacher_decoder_output_level.detach())
                    consistency_loss += b
        consistency_loss = consistency_loss / levels / batchsize
        return consistency_loss

    @torch.no_grad()
    def get_pseudo_targets(
            self, unsup_inputs, unsup_calibs, unsup_targets, unsup_info
    ):
        """Get pseudo targets from teacher model."""
        self.teacher.eval()
        pseudo_targets_list, mask, cls_score, topk_boxes = self.teacher.forward(
            unsup_inputs, unsup_calibs, unsup_targets, unsup_info, mode='get_pseudo_targets')
        return pseudo_targets_list, mask, cls_score, topk_boxes

    def project_pseudo_instances(self, batch_pseudo_instances,
                                 batch_data_samples):
        """Project pseudo instances."""
        for pseudo_instances, data_samples in zip(batch_pseudo_instances,
                                                  batch_data_samples):
            data_samples.gt_instances = copy.deepcopy(
                pseudo_instances.gt_instances)
            data_samples.gt_instances_3d = copy.deepcopy(
                pseudo_instances.pred_instances_3d)
            # data_samples.gt_instances.bboxes = bbox_project(
            #     data_samples.gt_instances.bboxes,
            #     torch.tensor(data_samples.homography_matrix).to(
            #         self.data_preprocessor.device), data_samples.img_shape)
        wh_thr = self.semi_train_cfg.get('min_pseudo_bbox_wh', (1e-2, 1e-2))
        return filter_gt_instances(batch_data_samples, wh_thr=wh_thr)

    def predict(self, inputs, calibs, targets, info):
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`Det3DDataSample`]: Return the detection results of the
            input images. The returns value is Det3DDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        if self.semi_test_cfg.get('predict_on', 'teacher') == 'teacher':
            return self.teacher(
                inputs[0], calibs, targets, info, mode='predict')
        else:
            return self.student(
                inputs[0], calibs, targets, info, mode='predict')

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples):
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
        if self.semi_test_cfg.get('forward_on', 'teacher') == 'teacher':
            return self.teacher(
                batch_inputs, batch_data_samples, mode='tensor')
        else:
            return self.student(
                batch_inputs, batch_data_samples, mode='tensor')

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        if self.semi_test_cfg.get('extract_feat_on', 'teacher') == 'teacher':
            return self.teacher.extract_feat(batch_inputs)
        else:
            return self.student.extract_feat(batch_inputs)

    # 加载从状态字典中加载参数
    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Add teacher and student prefixes to model parameter names."""
        # 判断state_dict中是否包含'student'或'teacher'
        if not any([
            'student' in key or 'teacher' in key
            for key in state_dict.keys()
        ]):
            state_dict = state_dict["model_state"]
            # 将state_dict中的参数名添加'teacher.'前缀
            keys = list(state_dict.keys())
            state_dict.update({'teacher.' + "model." + k: state_dict[k] for k in keys})
            # 将state_dict中的参数名添加'student.'前缀
            state_dict.update({'student.' + "model." + k: state_dict[k] for k in keys})
            # 删除state_dict中不包含'teacher.'或'student.'的参数名
            for k in keys:
                state_dict.pop(k)
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
