import numpy as np
import torch
import torch.nn as nn
from lib.datasets.utils import class2angle
from utils import box_ops
import torch





def decode_detections_GPU(dets, info, calibs, cls_mean_size, threshold):
    '''
    NOTE: THIS IS A TENSOR FUNCTION
    Assume calibs is a list of objects that have a method img_to_rect accepting and returning Tensors,
    and get_heading_angle is modified to work with Tensors.
    '''
    results = {}
    cls_scores = {}
    depth_score = {}
    scores = {}
    if len(info['img_size'].shape) != 2:
        info['img_size'] = info['img_size'].unsqueeze(0)
        info['img_id'] = info['img_id'].unsqueeze(0)
    for i in range(dets.shape[0]):  # batch
        preds = []
        cls_scores_list = []
        depth_score_list = []
        score_list = []
        for j in range(dets.shape[1]):  # max_dets
            cls_id = int(dets[i, j, 0].item())
            cls_score = dets[i, j, 1]
            if cls_score < threshold:
                continue

            # 2d bboxs decoding
            x = dets[i, j, 2] * info['img_size'][i][0]
            y = dets[i, j, 3] * info['img_size'][i][1]
            w = dets[i, j, 4] * info['img_size'][i][0]
            h = dets[i, j, 5] * info['img_size'][i][1]
            bbox = torch.tensor([x - w / 2, y - h / 2, x + w / 2, y + h / 2], device=dets.device)

            # 3d bboxs decoding
            depth = dets[i, j, 6]
            # dimensions decoding
            dimensions = dets[i, j, 31:34]
            dimensions += cls_mean_size[int(cls_id)]

            # positions decoding
            x3d = dets[i, j, 34] * info['img_size'][i][0]
            y3d = dets[i, j, 35] * info['img_size'][i][1]
            locations = calibs[i].img_to_rect_gpu(x3d, y3d, depth).reshape(-1)
            locations[1] += dimensions[0] / 2

            # heading angle decoding
            alpha = get_heading_angle_gpu(dets[i, j, 7:31])
            ry = calibs[i].alpha2ry_gpu(alpha, x)

            score = cls_score * dets[i, j, -1]
            pred = torch.cat([torch.tensor([cls_id, alpha], device=dets.device), bbox, dimensions, locations,
                              torch.tensor([ry, score], device=dets.device)])
            preds.append(pred)
            cls_scores_list.append(cls_score)
            depth_score_list.append(dets[i, j, -1])
            score_list.append(score)
        results[int(info['img_id'][i])] = torch.stack(preds)
        cls_scores[int(info['img_id'][i])] = torch.tensor(cls_scores_list, device=dets.device)
        depth_score[int(info['img_id'][i])] = torch.tensor(depth_score_list, device=dets.device)
        scores[int(info['img_id'][i])] = torch.tensor(score_list, device=dets.device)
    return results, cls_scores, depth_score, scores


def decode_detections(dets, info, calibs, cls_mean_size, threshold):
    '''
    NOTE: THIS IS A NUMPY FUNCTION
    input: dets, numpy array, shape in [batch x max_dets x dim]
    input: img_info, dict, necessary information of input images
    input: calibs, corresponding calibs for the input batch
    output:
    '''
    results = {}
    cls_scores = {}
    depth_score = {}
    scores = {}
    for i in range(dets.shape[0]):  # batch
        preds = []
        cls_scores_list = []
        depth_score_list = []
        score_list = []
        for j in range(dets.shape[1]):  # max_dets
            cls_id = int(dets[i, j, 0])
            cls_score = dets[i, j, 1]
            if cls_score < threshold:
                continue

            # 2d bboxs decoding
            x = dets[i, j, 2] * info['img_size'][i][0]
            y = dets[i, j, 3] * info['img_size'][i][1]
            w = dets[i, j, 4] * info['img_size'][i][0]
            h = dets[i, j, 5] * info['img_size'][i][1]
            bbox = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]

            # 3d bboxs decoding
            # depth decoding
            depth = dets[i, j, 6]

            # dimensions decoding
            dimensions = dets[i, j, 31:34]
            dimensions += cls_mean_size[int(cls_id)]

            # positions decoding
            x3d = dets[i, j, 34] * info['img_size'][i][0]
            y3d = dets[i, j, 35] * info['img_size'][i][1]
            locations = calibs[i].img_to_rect(x3d, y3d, depth).reshape(-1)
            locations[1] += dimensions[0] / 2

            # heading angle decoding
            alpha = get_heading_angle(dets[i, j, 7:31])
            ry = calibs[i].alpha2ry(alpha, x)

            score = cls_score * dets[i, j, -1]
            preds.append([cls_id, alpha] + bbox + dimensions.tolist() + locations.tolist() + [ry, score])
            cls_scores_list.append(cls_score)
            depth_score_list.append(dets[i, j, -1])
            score_list.append(score)
        results[info['img_id'][i]] = preds
        cls_scores[info['img_id'][i]] = cls_scores_list
        depth_score[info['img_id'][i]] = depth_score_list
        scores[info['img_id'][i]] = score_list
    return results, cls_scores, depth_score, scores


def extract_dets_from_outputs(outputs, K=50, topk=50):
    # get src outputs

    # b, q, c
    out_logits = outputs['pred_logits']
    out_bbox = outputs['pred_boxes']

    prob = out_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), topk, dim=1)

    # final scores
    scores = topk_values
    # final indexes
    topk_boxes = (topk_indexes // out_logits.shape[2]).unsqueeze(-1)
    # final labels
    labels = topk_indexes % out_logits.shape[2]

    heading = outputs['pred_angle']
    size_3d = outputs['pred_3d_dim']
    depth = outputs['pred_depth'][:, :, 0: 1]
    sigma = outputs['pred_depth'][:, :, 1: 2]
    sigma = torch.exp(-sigma)

    # decode
    boxes = torch.gather(out_bbox, 1, topk_boxes.repeat(1, 1, 6))  # b, q', 4

    xs3d = boxes[:, :, 0: 1]
    ys3d = boxes[:, :, 1: 2]

    heading = torch.gather(heading, 1, topk_boxes.repeat(1, 1, 24))
    depth = torch.gather(depth, 1, topk_boxes)
    sigma = torch.gather(sigma, 1, topk_boxes)
    size_3d = torch.gather(size_3d, 1, topk_boxes.repeat(1, 1, 3))

    corner_2d = box_ops.box_cxcylrtb_to_xyxy(boxes)

    xywh_2d = box_ops.box_xyxy_to_cxcywh(corner_2d)
    size_2d = xywh_2d[:, :, 2: 4]

    xs2d = xywh_2d[:, :, 0: 1]
    ys2d = xywh_2d[:, :, 1: 2]

    batch = out_logits.shape[0]
    labels = labels.view(batch, -1, 1)
    scores = scores.view(batch, -1, 1)
    xs2d = xs2d.view(batch, -1, 1)
    ys2d = ys2d.view(batch, -1, 1)
    xs3d = xs3d.view(batch, -1, 1)
    ys3d = ys3d.view(batch, -1, 1)

    detections = torch.cat([labels, scores, xs2d, ys2d, size_2d, depth, heading, size_3d, xs3d, ys3d, sigma], dim=2)

    return detections, topk_boxes


############### auxiliary function ############


def _nms(heatmap, kernel=3):
    padding = (kernel - 1) // 2
    heatmapmax = nn.functional.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=padding)
    keep = (heatmapmax == heatmap).float()
    return heatmap * keep


def _topk(heatmap, K=50):
    batch, cat, height, width = heatmap.size()

    # batch * cls_ids * 50
    topk_scores, topk_inds = torch.topk(heatmap.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    # batch * cls_ids * 50
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_cls_ids = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_cls_ids, topk_xs, topk_ys


def _gather_feat(feat, ind, mask=None):
    '''
    Args:
        feat: tensor shaped in B * (H*W) * C
        ind:  tensor shaped in B * K (default: 50)
        mask: tensor shaped in B * K (default: 50)

    Returns: tensor shaped in B * K or B * sum(mask)
    '''
    dim = feat.size(2)  # get channel dim
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)  # B*len(ind) --> B*len(ind)*1 --> B*len(ind)*C
    feat = feat.gather(1, ind)  # B*(HW)*C ---> B*K*C
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)  # B*50 ---> B*K*1 --> B*K*C
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    '''
    Args:
        feat: feature maps shaped in B * C * H * W
        ind: indices tensor shaped in B * K
    Returns:
    '''
    feat = feat.permute(0, 2, 3, 1).contiguous()  # B * C * H * W ---> B * H * W * C
    feat = feat.view(feat.size(0), -1, feat.size(3))  # B * H * W * C ---> B * (H*W) * C
    feat = _gather_feat(feat, ind)  # B * len(ind) * C
    return feat


def get_heading_angle(heading):
    heading_bin, heading_res = heading[0:12], heading[12:24]
    cls = np.argmax(heading_bin)
    res = heading_res[cls]
    return class2angle(cls, res, to_label_format=True)

def get_heading_angle_gpu(heading):
    heading_bin, heading_res = heading[0:12], heading[12:24]
    cls = torch.argmax(heading_bin)
    res = heading_res[cls]
    return class2angle_gpu(cls, res, to_label_format=True)

def class2angle_gpu(cls, residual, to_label_format=False, num_heading_bin=12):
    angle_per_class = 2 * torch.pi / float(num_heading_bin)
    angle_center = cls.float() * angle_per_class  # Ensure cls is float for multiplication
    angle = angle_center + residual
    if to_label_format:
        angle = torch.where(angle > torch.pi, angle - 2 * torch.pi, angle)
    return angle