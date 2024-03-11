import os
import sys
import warnings
import torch
import numpy as np
from numba import jit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

# from mmdet3d.core.bbox.iou_calculators.iou3d_calculator import BboxOverlaps3D

# 初始化UncertaintyEstimator类
class UncertaintyEstimator():
    def __init__(self):
        super(UncertaintyEstimator, self).__init__()
        # self.iou_calculator = BboxOverlaps3D(coordinate="camera")

    def find_matching_box(self, boxes_list, new_box, match_iou):
        '''
        根据IOU阈值，查找与新框匹配的框
        '''
        best_iou = match_iou
        best_index = -1
        for i in range(len(boxes_list)):
            box = boxes_list[i]
            if box[0] != new_box[0]:
                continue
            iou = bb_intersection_over_union(box[4:8], new_box[4:8])
            if iou > best_iou:
                best_index = i
                best_iou = iou
        return best_index, best_iou

    def get_weighted_box(self, boxes, conf_type='avg'):
        """
        Create weighted box for set of boxes
        :param boxes: set of boxes to fuse
        :param conf_type: type of confidence one of 'avg' or 'max'
        :return: weighted box (label, score, weight, x1, y1, x2, y2)
        """
        box = np.zeros(18, dtype=np.float32)
        conf = 0
        conf_list = []
        w = 0

        box3d = boxes.copy()
        box3d = np.array(box3d)
        box3d = box3d[:, [8,9,10,13,11,12,14]]
        box3d = torch.tensor(box3d)
        bbox_ious_3d = self.iou_calculator(box3d, box3d, "iou")
        bbox_ious_3d = bbox_ious_3d.numpy()

        matrix_w = np.ones((len(self.model_list), len(self.model_list)))
        tempalte_ious_3d = np.zeros((len(self.model_list), len(self.model_list)))
        if tempalte_ious_3d.shape[0] >= bbox_ious_3d.shape[0]:
            tempalte_ious_3d[:bbox_ious_3d.shape[0], :bbox_ious_3d.shape[1]] = bbox_ious_3d
        else:
            tempalte_ious_3d = bbox_ious_3d[:tempalte_ious_3d.shape[0], :tempalte_ious_3d.shape[1]]
        matrix_w[range(matrix_w.shape[0]), range(matrix_w.shape[1])] = 1.0
        weighted_ious_3d = tempalte_ious_3d * matrix_w
        geo_conf = np.sum(weighted_ious_3d) / (np.sum(matrix_w) + 10e-6)

        for b in boxes:
            # [label, score, weight, model_index, x1, y1, x2, y2]
            # ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'lx', 'ly', 'lz','dh', 'dw', 'dl', 'ry']
            # + ['score', 'weights', 'model index']
            conf += b[-3]
            conf_list.append(b[-3])
            w += b[-2]

        if conf_type == 'avg':
            box[-3] = conf / len(boxes)
        elif conf_type == 'max':
            box[-3] = np.array(conf_list).max()
        elif conf_type in ['box_and_model_avg', 'absent_model_aware_avg']:
            box[-3] = conf / len(boxes)
        
        box[-2] = w
        box[-1] = -1 # model index field is retained for consistensy but is not used.
        box[:-2] = boxes[0][:-2]
        if box[10] < 4.2:
            box[-2] = max(0.85, box[-2])
        else:
            box[-2] = geo_conf
        box[-1] = len(boxes)
        # ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'lx', 'ly', 'lz','dh', 'dw', 'dl', 'ry']
        # + ['score', 'geo_conf', 'model index']
        return box

    def boxes_cluster(self, pseudo_target_dict, dets, iou_thr=0.55, conf_type='avg', allows_overflow=False):
        '''
        :param boxes_list: list of boxes predictions from each model, each box is 4 numbers.
        It has 3 dimensions (models_number, model_preds, 15)
        Order of boxes: 'type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'lx', 'ly', 'lz','dh', 'dw', 'dl' 'ry', 'score'. 
        We expect float normalized coordinates [0; 1]
        :param scores_list: list of scores for each model
        :param labels_list: list of labels for each model
        :param iou_thr: IoU value for boxes to be a match
        :param skip_box_thr: exclude boxes with score lower than this variable
        :param conf_type: how to calculate confidence in weighted boxes. 'avg': average value, 'max': maximum value, 'box_and_model_avg': box and model wise hybrid weighted average, 'absent_model_aware_avg': weighted average that takes into account the absent model.
        :param allows_overflow: false if we want confidence score not exceed 1.0
        :return: boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2).
        :return: scores: confidence scores
        :return: labels: boxes labels
        '''

        if conf_type not in ['avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg']:
            print('Unknown conf_type: {}. Must be "avg", "max" or "box_and_model_avg", or "absent_model_aware_avg"'.format(conf_type))
            exit()

        filtered_boxes = self.prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
        if len(filtered_boxes) == 0:
            return np.zeros((0, 18)), np.zeros((0,)), np.zeros((0,)), np.zeros((0,))

        overall_boxes = []
        overall_new_boxes = {}
        # 遍历每一类
        for label in filtered_boxes:
            boxes = filtered_boxes[label]
            new_boxes = []
            weighted_boxes = []
            # Clusterize boxes
            # 遍历每一个box
            for j in range(0, len(boxes)):
                # 查找与当前box最匹配的box
                index, best_iou = self.find_matching_box(weighted_boxes, boxes[j], iou_thr)
                # 如果找到匹配的box，则将当前box添加到匹配的box的cluster中
                if index != -1:
                    new_boxes[index].append(boxes[j])
                    weighted_boxes[index] = self.get_weighted_box(new_boxes[index], conf_type)
                # 如果没有找到匹配的box，则新建一个cluster
                else:
                    new_boxes.append([boxes[j].copy()])
                    candidate = boxes[j].copy()
                    # 如果当前box的置信度小于4.2，则将其置信度设置为0.85
                    if candidate[10] < 4.2:
                        candidate[-2] = max(0.85, candidate[-2])
                    # 否则，置信度设置为1/len(self.model_list)
                    else:
                        candidate[-2] = 1 / len(self.model_list)
                    weighted_boxes.append(candidate)
            # 将每一个cluster的box添加到overall_boxes中
            overall_boxes.append(np.array(weighted_boxes))
            # 将每一个cluster的box添加到overall_new_boxes中
            overall_new_boxes[label] = new_boxes
        
        # 将overall_boxes中的每一个cluster的box按照置信度降序排列
        overall_boxes = np.concatenate(overall_boxes, axis=0)
        overall_boxes = overall_boxes[overall_boxes[:, -3].argsort()[::-1]]
        boxes = overall_boxes[:, :]
        scores = overall_boxes[:, -3]
        labels = overall_boxes[:, 0]

        #   ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'lx', 'ly', 'lz','dh', 'dw', 'dl' 'ry']
        # + ['score', 'weights', 'model index']
        return boxes, scores, labels, overall_new_boxes

