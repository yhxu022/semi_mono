from mmengine.model import BaseModel
import torch
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections, decode_detections_GPU
import numpy as np
from uncertainty_estimator import UncertaintyEstimator
from pcdet.ops.iou3d_nms.iou3d_nms_utils import nms_gpu
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu


def class2angle_gpu(cls, residual, to_label_format=False, num_heading_bin=12):

    angle_per_class = 2 * torch.pi / float(num_heading_bin)
    angle_center = cls.float() * angle_per_class  # Ensure cls is float for multiplication
    angle = angle_center + residual

    if to_label_format:
        # Using torch.where to handle condition across tensors
        angle = torch.where(angle > torch.pi, angle - 2 * torch.pi, angle)

    return angle

def rect_to_lidar_gpu(calib, pts_rect):
    """
    Convert rectified camera coordinates to LiDAR coordinates.
    Assume pts_rect is a Tensor [N, 3] and on the correct device (GPU).
    """
    # Matrix inverse and dot product
    R0_inv = torch.inverse(calib.R0)
    pts_ref = torch.mm(pts_rect, R0_inv.T)  # Using torch.mm for matrix multiplication

    # Convert Cartesian to homogeneous coordinates
    pts_ref_hom = calib.cart_to_hom(pts_ref)  # Assuming cart_to_hom is adapted for Tensors

    # Dot product to transform to LiDAR coordinates
    lidar_coords = torch.mm(pts_ref_hom, calib.C2V.T)

    return lidar_coords

def alpha2ry_gpu(calib, alpha, u):
    """
    Convert alpha (observation angle) to rotation_y (rotation around Y-axis in camera coordinates),
    considering the object center 'u' and camera calibration parameters 'cu' and 'fu'.

    Parameters:
    alpha (Tensor): Observation angle of object, ranging [-pi..pi]
    u (Tensor): Object center x to the camera center (x-W/2), in pixels

    Returns:
    Tensor: rotation_y around Y-axis in camera coordinates [-pi..pi]
    """
    # Ensure alpha, u, cu, and fu are on the same device
    # device = alpha.device
    calib.cu = calib.cu
    calib.fu = calib.fu

    # Calculate rotation_y
    ry = alpha + torch.atan2(u - calib.cu, calib.fu)

    # Adjust rotation_y to be within [-pi, pi]
    ry = torch.where(ry > torch.pi, ry - 2 * torch.pi, ry)
    ry = torch.where(ry < -torch.pi, ry + 2 * torch.pi, ry)

    return ry

class Semi_Mono_DETR(BaseModel):
    def __init__(self, model, loss, cfg, dataloader, inference_set=None):
        super().__init__()
        self.uncertainty_estimator = UncertaintyEstimator()
        self.model = model
        self.loss = loss
        self.cfg = cfg
        self.writelist = cfg["dataset"]["writelist"]
        self.resolution = np.array([1280, 384])  # W * H
        self.id2cls = {0: 'Pedestrian', 1: 'Car', 2: 'Cyclist'}
        self.dataloader = dataloader
        # self.max_objs = dataloader.dataset.max_objs    # max objects per images, defined in dataset
        # self.max_objs = dataloader["dataset"].max_objs
        self.max_objs = 50
        self.inference_set = inference_set

    def forward(self, inputs, calibs, targets, info, mode):
        self.model.mode = mode
        self.model.pseudo_label_group_num = self.pseudo_label_group_num
        self.model.val_nms = self.val_nms
        if mode == 'loss':
            img_sizes = targets['img_size']
            ##dn
            dn_args = None
            if self.cfg["model"]["use_dn"]:
                dn_args = (targets, self.cfg["model"]['scalar'], self.cfg["model"]['label_noise_scale'],
                           self.cfg["model"]['box_noise_scale'], self.cfg["model"]['num_patterns'])
            ###
            # train one batch
            targets = self.prepare_targets(targets, inputs.shape[0])
            outputs = self.model(inputs, calibs, img_sizes, dn_args=dn_args)
            mask_dict = None
            # ipdb.set_trace()
            detr_losses_dict = self.loss(outputs, targets, mask_dict)
            weight_dict = self.loss.weight_dict
            detr_losses_dict_weighted = {k: detr_losses_dict[k] * weight_dict[k] for k in detr_losses_dict.keys() if
                                         k in weight_dict}
            return detr_losses_dict_weighted

        elif mode == 'predict':
            img_sizes = info['img_size']
            ###dn
            targets = self.prepare_targets(targets, inputs.shape[0])
            outputs = self.model(inputs, calibs, img_sizes, dn_args=0)
            if (self.val_nms == False):
                dets, topk_boxes = extract_dets_from_outputs(outputs=outputs, K=self.max_objs,
                                                             topk=self.cfg["tester"]['topk'])
            else:
                dets, topk_boxes = extract_dets_from_outputs(outputs=outputs,
                                                             K=self.pseudo_label_group_num * self.max_objs,
                                                             topk=self.pseudo_label_group_num * self.cfg["tester"][
                                                                 'topk'])
                device = dets.device
            dets = dets.detach().cpu().numpy()
            # get corresponding calibs & transform tensor to numpy
            calibs = [self.dataloader["dataset"].get_calib(index) for index in info['img_id']]
            info = {key: val.detach().cpu().numpy() for key, val in info.items()}
            cls_mean_size = self.dataloader["dataset"].cls_mean_size
            dets, cls_scores, depth_score_list, scores = decode_detections(
                dets=dets,
                info=info,
                calibs=calibs,
                cls_mean_size=cls_mean_size,
                threshold=self.cfg["tester"].get('threshold', 0.2))
            if (self.val_nms == True):
                if len(dets) > 0:
                    for i, id in enumerate(info['img_id']):
                        calib = calibs[i]
                        dets_img = dets[int(id)]
                        cls_score = cls_scores[int(id)]
                        if self.pseudo_label_group_num > 1:
                            if len(dets_img) >= 1:
                                dets_img = torch.tensor(dets_img, dtype=torch.float32).to(device)
                                cls_score = torch.tensor(cls_score, dtype=torch.float32).to(device)
                                scores = dets_img[:, -1]
                                loc = dets_img[:, 9:12]
                                h = dets_img[:, 6:7]
                                w = dets_img[:, 7:8]
                                l = dets_img[:, 8:9]
                                ry = dets_img[:, 12:13]
                                loc_lidar = torch.tensor(calib.rect_to_lidar(loc.detach().cpu().numpy()),
                                                         dtype=torch.float32).to(device)
                                loc_lidar[:, 2] += h[:, 0] / 2
                                heading = -(torch.pi / 2 + ry)
                                boxes_lidar = torch.concatenate([loc_lidar, l, w, h, heading], axis=1)
                                dets_after_nms, _ = nms_gpu(boxes_lidar, scores, thresh=0.55)
                                dets_img = dets_img[dets_after_nms].detach().cpu().numpy()
                                dets[int(id)] = dets_img
            return dets, targets


        elif mode == 'get_pseudo_targets':
            img_sizes = info['img_size']
            outputs = self.model(inputs, calibs, img_sizes, dn_args=0)
            if self.pseudo_label_group_num == 1:
                dets, topk_boxes = extract_dets_from_outputs(outputs=outputs, K=self.max_objs,
                                                             topk=self.cfg["semi_train_cfg"]['topk'])
                cls_pseudo_targets_list, cls_mask, cls_cls_score = self.get_pseudo_targets_list(dets, calibs,
                                                                                                dets.shape[0],
                                                                                                self.cfg[
                                                                                                    "semi_train_cfg"][
                                                                                                    "cls_cls_pseudo_thr"],
                                                                                                self.cfg[
                                                                                                    "semi_train_cfg"][
                                                                                                    "cls_score_pseudo_thr"],
                                                                                                self.cfg[
                                                                                                    "semi_train_cfg"].get(
                                                                                                    "cls_depth_score_thr",
                                                                                                    0, ),
                                                                                                batch_targets=targets)

                regression_pseudo_targets_list, regression_mask, regression_cls_score = self.get_pseudo_targets_list(
                    dets, calibs, dets.shape[0],
                    self.cfg["semi_train_cfg"][
                        "regression_cls_pseudo_thr"],
                    self.cfg["semi_train_cfg"][
                        "regression_score_pseudo_thr"],
                    self.cfg["semi_train_cfg"].get("regression_depth_score_thr", 0, ),
                    batch_targets=targets)

                mask_pseudo_targets_list, mask_mask, mask_cls_score = self.get_pseudo_targets_list(dets, calibs,
                                                                                                   dets.shape[0],
                                                                                                   self.cfg[
                                                                                                       "semi_train_cfg"][
                                                                                                       "mask_cls_pseudo_floor"],
                                                                                                   self.cfg[
                                                                                                       "semi_train_cfg"][
                                                                                                       "mask_score_pseudo_floor"],
                                                                                                   self.cfg[
                                                                                                       "semi_train_cfg"].get(
                                                                                                       "mask_depth_score_floor",
                                                                                                       0),
                                                                                                   self.cfg[
                                                                                                       "semi_train_cfg"][
                                                                                                       "mask_cls_pseudo_ceiling"],
                                                                                                   self.cfg[
                                                                                                       "semi_train_cfg"][
                                                                                                       "mask_score_pseudo_ceiling"],
                                                                                                   self.cfg[
                                                                                                       "semi_train_cfg"].get(
                                                                                                       "mask_depth_score_ceiling",
                                                                                                       0),
                                                                                                   batch_targets=targets)

                mask_topk_boxes = cls_topk_boxes = regression_topk_boxes = topk_boxes
            else:
                dets, topk_boxes = extract_dets_from_outputs(outputs=outputs,
                                                             K=self.pseudo_label_group_num * self.max_objs,
                                                             topk=self.pseudo_label_group_num *
                                                                  self.cfg["semi_train_cfg"]['topk'])
                cls_pseudo_targets_list, cls_mask, cls_cls_score = self.get_pseudo_targets_list(dets, calibs,
                                                                                                dets.shape[0],
                                                                                                self.cfg[
                                                                                                    "semi_train_cfg"][
                                                                                                    "cls_cls_pseudo_thr"],
                                                                                                self.cfg[
                                                                                                    "semi_train_cfg"][
                                                                                                    "cls_score_pseudo_thr"],
                                                                                                self.cfg[
                                                                                                    "semi_train_cfg"].get(
                                                                                                    "cls_depth_score_thr",
                                                                                                    0)
                                                                                                )
                regression_pseudo_targets_list, regression_mask, regression_cls_score = self.get_pseudo_targets_list(
                    dets, calibs, dets.shape[0],
                    self.cfg["semi_train_cfg"][
                        "regression_cls_pseudo_thr"],
                    self.cfg["semi_train_cfg"][
                        "regression_score_pseudo_thr"],
                    self.cfg["semi_train_cfg"].get("regression_depth_score_thr", 0)
                    )
                mask_pseudo_targets_list, mask_mask, mask_cls_score = self.get_pseudo_targets_list(dets, calibs,
                                                                                                   dets.shape[0],
                                                                                                   self.cfg[
                                                                                                       "semi_train_cfg"][
                                                                                                       "mask_cls_pseudo_floor"],
                                                                                                   self.cfg[
                                                                                                       "semi_train_cfg"][
                                                                                                       "mask_score_pseudo_floor"],
                                                                                                   self.cfg[
                                                                                                       "semi_train_cfg"].get(
                                                                                                       "mask_depth_score_floor",
                                                                                                       0),
                                                                                                   self.cfg[
                                                                                                       "semi_train_cfg"][
                                                                                                       "mask_cls_pseudo_ceiling"],
                                                                                                   self.cfg[
                                                                                                       "semi_train_cfg"][
                                                                                                       "mask_score_pseudo_ceiling"],
                                                                                                   self.cfg[
                                                                                                       "semi_train_cfg"].get(
                                                                                                       "mask_depth_score_ceiling",
                                                                                                       0)
                                                                                                   )
                mask_topk_boxes = cls_topk_boxes = regression_topk_boxes = topk_boxes

            return cls_pseudo_targets_list, cls_mask, cls_cls_score, cls_topk_boxes, regression_pseudo_targets_list, \
                regression_mask, regression_cls_score, regression_topk_boxes, \
                mask_pseudo_targets_list, mask_mask, mask_cls_score, mask_topk_boxes
        elif mode == 'inference':
            img_sizes = info['img_size']
            outputs = self.model(inputs, calibs, img_sizes, dn_args=0)
            if self.pseudo_label_group_num == 1:
                dets, topk_boxes = extract_dets_from_outputs(outputs=outputs, K=self.max_objs,
                                                             topk=self.cfg["semi_train_cfg"]['topk'])
                dets = self.get_pseudo_targets_list_inference(dets, calibs, dets.shape[0],
                                                              self.cfg["semi_train_cfg"]["cls_pseudo_thr"],
                                                              self.cfg["semi_train_cfg"]["score_pseudo_thr"], info)
            else:
                dets, topk_boxes = extract_dets_from_outputs(outputs=outputs,
                                                             K=self.pseudo_label_group_num * self.max_objs,
                                                             topk=self.pseudo_label_group_num *
                                                                  self.cfg["semi_train_cfg"]['topk'])
                dets = self.get_pseudo_targets_list_inference(dets, calibs, dets.shape[0],
                                                              self.cfg["semi_train_cfg"]["cls_pseudo_thr"],
                                                              self.cfg["semi_train_cfg"]["score_pseudo_thr"], info)
            return dets

        elif mode == 'unsup_loss':
            img_sizes = info['img_size']
            ##dn
            dn_args = None
            if self.cfg["model"]["use_dn"]:
                dn_args = (targets, self.cfg["model"]['scalar'], self.cfg["model"]['label_noise_scale'],
                           self.cfg["model"]['box_noise_scale'], self.cfg["model"]['num_patterns'])
            ###
            # train one batch
            outputs = self.model(inputs, calibs, img_sizes, dn_args=dn_args)
            mask_dict = None
            detr_losses_dict = self.loss(outputs, targets, mask_dict)
            weight_dict = self.loss.weight_dict
            detr_losses_dict_weighted = {k: detr_losses_dict[k] * weight_dict[k] for k in detr_losses_dict.keys() if
                                         k in weight_dict}
            return detr_losses_dict_weighted

        elif mode == 'statistics':
            img_sizes = info['img_size']
            outputs = self.model(inputs, calibs, img_sizes, dn_args=0)
            if self.pseudo_label_group_num == 1:
                dets, topk_boxes = extract_dets_from_outputs(outputs=outputs, K=self.max_objs,
                                                             topk=self.cfg["semi_train_cfg"]['topk'])
                boxes_lidar, score, loc_list, depth_score_list, scores, pseudo_labels_list = self.get_boxes_lidar_and_clsscore(
                    dets, calibs, dets.shape[0],
                    self.cfg["semi_train_cfg"]["cls_pseudo_thr"],
                    self.cfg["semi_train_cfg"]["score_pseudo_thr"],
                    self.cfg["semi_train_cfg"].get("depth_score_thr", 0),
                    info)
            else:
                dets, topk_boxes = extract_dets_from_outputs(outputs=outputs,
                                                             K=self.pseudo_label_group_num * self.max_objs,
                                                             topk=self.pseudo_label_group_num *
                                                                  self.cfg["semi_train_cfg"]['topk'])
                boxes_lidar, score, loc_list, depth_score_list, scores, pseudo_labels_list = self.get_boxes_lidar_and_clsscore(
                    dets, calibs, dets.shape[0],
                    self.cfg["semi_train_cfg"]["cls_pseudo_thr"],
                    self.cfg["semi_train_cfg"]["score_pseudo_thr"],
                    self.cfg["semi_train_cfg"].get("depth_score_thr", 0),
                    info)
            return boxes_lidar, score, loc_list, depth_score_list, scores, pseudo_labels_list

    def prepare_targets(self, targets, batch_size):
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

    def get_pseudo_targets_list(self, batch_dets, batch_calibs, batch_size, cls_pseudo_thr, score_pseudo_thr,
                                depth_score_thr,
                                cls_pseudo_thr_ceiling=10000, score_pseudo_thr_ceiling=10000,
                                depth_score_thr_ceiling=10000, batch_targets=None):
        pseudo_targets_list = []
        mask_list = []
        cls_score_list = batch_dets[:, :, 1]
        for bz in range(batch_size):
            dets = batch_dets[bz]
            calibs = batch_calibs[bz]
            target = batch_targets[bz]
            pseudo_labels = dets[:, 0]
            mask_cls_type = np.zeros((len(pseudo_labels)), dtype=bool)
            mask_cls_pseudo_thr = np.zeros((len(pseudo_labels)), dtype=bool)
            mask_score_pseudo_thr = np.zeros((len(pseudo_labels)), dtype=bool)
            mask_depth_score_pseudo_thr = np.zeros((len(pseudo_labels)), dtype=bool)
            for i in range(len(pseudo_labels)):
                if self.id2cls[int(pseudo_labels[i])] in self.writelist:
                    mask_cls_type[i] = True
                if dets[i, 1] > cls_pseudo_thr and dets[i, 1] < cls_pseudo_thr_ceiling:
                    mask_cls_pseudo_thr[i] = True
                score = dets[i, 1] * dets[i, -1]
                if score > score_pseudo_thr and score < score_pseudo_thr_ceiling:
                    mask_score_pseudo_thr[i] = True
                if dets[i, -1] > depth_score_thr and dets[i, -1] < depth_score_thr_ceiling:
                    mask_depth_score_pseudo_thr[i] = True
            mask = mask_cls_type & mask_cls_pseudo_thr & mask_score_pseudo_thr & mask_depth_score_pseudo_thr
            mask_list.append(mask)
            dets = dets[mask]


            # 把target和伪标签做iou，使得分类一定正确
            mask_cls_gt = np.zeros((len(dets[:, 0])), dtype=bool)
            cls_mean_size = torch.zeros((3, 3), device=device)
            dets_decode, cls_scores, depth_score_list, scores = decode_detections_GPU(
                dets=dets,
                info=target,
                calibs=calibs,
                cls_mean_size=cls_mean_size,
                threshold=self.cfg["tester"].get('threshold', 0.2))

            calib = calibs[0]
            dets_img = dets_decode[int(target['img_id'])]
            if len(dets_img) >= 1:
                device = dets_img.device
                dets_img = torch.tensor(dets_img, dtype=torch.float32).to(device)
                loc = dets_img[:, 9:12]
                h = dets_img[:, 6:7]
                w = dets_img[:, 7:8]
                l = dets_img[:, 8:9]
                ry = dets_img[:, 12:13]
                loc_lidar = rect_to_lidar_gpu(calib,loc)
                loc_lidar[:, 2] += h[:, 0] / 2
                heading = -(torch.pi / 2 + ry)
                boxes_lidar = torch.concatenate([loc_lidar, l, w, h, heading], axis=1)

                hwl = target["src_size_3d"]
                h_gt = hwl[0]
                w_gt = hwl[1]
                l_gt = hwl[2]
                x_3d_gt = target['boxes_3d'][0] * 1280
                y_3d_gt = target['boxes_3d'][1] * 384
                z_3d_gt = target['depth']
                alpha_gt = class2angle_gpu(cls=target['heading_bin'],residual=target['heading_res'])
                x_gt = target['boxes'][0]
                ry_gt = alpha2ry_gpu(calib, alpha_gt, x_gt)
                loc_gt = torch.stack([x_3d_gt, y_3d_gt, z_3d_gt], dim=0).to(device)
                loc_lidar_gt = rect_to_lidar_gpu(calib, loc_gt)
                loc_lidar_gt[:, 2] += h_gt[:, 0] / 2
                heading_gt = -(torch.pi / 2 + ry_gt)
                boxes_lidar_gt = torch.concatenate([loc_lidar_gt, l_gt, w_gt, h_gt, heading_gt], axis=1)
                pass
            else:
                boxes_lidar = None
                loc = None

            boxes_lidar = boxes_lidar.float()
            iou3D = boxes_iou3d_gpu(boxes_lidar, boxes_lidar_gt)  # [num_pre, num_gt]

            num_pre = boxes_lidar.shape[0]
            num_gt = boxes_lidar_gt.shape[0]
            max_iou_values, max_iou_indices = torch.max(iou3D, dim=1)
            valid_indices = [idx for idx, val in enumerate(max_iou_values) if val > 0]  # [1,2,0]
            for idx in valid_indices:
                pred_label = dets[:, 0][idx]
                gt_label = target['labels'][max_iou_indices[idx]]
                if pred_label == gt_label:
                    mask_cls_gt[idx]=True

            dets = dets[mask_cls_gt]


            pseudo_target_dict = {}
            pseudo_labels = dets[:, 0].to(torch.int8)
            pseudo_target_dict["labels"] = pseudo_labels
            if (len(pseudo_labels)) == 0:
                device = pseudo_labels.device
                pseudo_target_dict["calibs"] = torch.zeros(size=(0, 3, 4), dtype=torch.float32, device=device)
                pseudo_target_dict["boxes"] = torch.zeros(size=(0, 4), dtype=torch.float32, device=device)
                pseudo_target_dict["depth"] = torch.zeros(size=(0, 1), dtype=torch.float32, device=device)
                pseudo_target_dict["size_3d"] = torch.zeros(size=(0, 3), dtype=torch.float32, device=device)
                pseudo_target_dict["boxes_3d"] = torch.zeros(size=(0, 6), dtype=torch.float32, device=device)
                pseudo_target_dict["heading_bin"] = torch.zeros(size=(0, 1), dtype=torch.int64, device=device)
                pseudo_target_dict["heading_res"] = torch.zeros(size=(0, 1), dtype=torch.float32, device=device)
                pseudo_targets_list.append(pseudo_target_dict)
                continue
            for i in range(len(pseudo_labels)):
                if (i == 0):
                    calibs = calib.unsqueeze(0)
                else:
                    calibs = torch.cat((calibs, calib.unsqueeze(0)))
            pseudo_target_dict["calibs"] = calibs
            boxes = dets[:, 2:6].to(torch.float32)
            pseudo_target_dict["boxes"] = boxes
            depth = dets[:, 6:7].to(torch.float32)
            pseudo_target_dict["depth"] = depth
            size_3d = dets[:, 31:34].to(torch.float32)
            pseudo_target_dict["size_3d"] = size_3d
            x3d = dets[:, 34:35].to(torch.float32)
            y3d = dets[:, 35:36].to(torch.float32)
            x = boxes[:, 0:1]
            y = boxes[:, 1:2]
            w = boxes[:, 2:3]
            h = boxes[:, 3:4]
            corner_2d = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
            l, r = x3d - corner_2d[0], corner_2d[2] - x3d
            t, b = y3d - corner_2d[1], corner_2d[3] - y3d
            boxes_3d = torch.cat((x3d, y3d, l, r, t, b), dim=1)
            pseudo_target_dict["boxes_3d"] = boxes_3d
            for i in range(len(pseudo_labels)):
                heading = dets[i, 7:31]
                heading_bin, heading_res = heading[0:12], heading[12:24]
                heading_bin = torch.argmax(heading_bin).unsqueeze(0)
                heading_res = heading_res[heading_bin]
                if (i == 0):
                    heading_bins = heading_bin.unsqueeze(0)
                    heading_ress = heading_res.unsqueeze(0)
                else:
                    heading_bins = torch.cat((heading_bins, heading_bin.unsqueeze(0)), dim=0)
                    heading_ress = torch.cat((heading_ress, heading_res.unsqueeze(0)), dim=0)
            pseudo_target_dict["heading_bin"] = heading_bins
            pseudo_target_dict["heading_res"] = heading_ress
            # if self.pseudo_label_group_num>1:
            #     self.uncertainty_estimator.boxes_cluster(pseudo_target_dict,dets)
            pseudo_targets_list.append(pseudo_target_dict)
        return pseudo_targets_list, mask_list, cls_score_list

    def get_pseudo_targets_list_inference(self, batch_dets, batch_calibs, batch_size, cls_pseudo_thr, score_pseudo_thr,
                                          depth_score_thr, info):
        mask_list = []
        cls_score_list = batch_dets[:, :, 1]
        for bz in range(batch_size):
            dets = batch_dets[bz]
            calib = batch_calibs[bz]
            # target=batch_targets[bz]
            pseudo_labels = dets[:, 0]
            mask_cls_type = np.zeros((len(pseudo_labels)), dtype=bool)
            mask_cls_pseudo_thr = np.zeros((len(pseudo_labels)), dtype=bool)
            mask_score_pseudo_thr = np.zeros((len(pseudo_labels)), dtype=bool)
            for i in range(len(pseudo_labels)):
                if self.id2cls[int(pseudo_labels[i])] in self.writelist:
                    mask_cls_type[i] = True
                if dets[i, 1] > cls_pseudo_thr:
                    mask_cls_pseudo_thr[i] = True
                score = dets[i, 1] * dets[i, -1]
                if score > score_pseudo_thr:
                    mask_score_pseudo_thr[i] = True
            mask = mask_cls_type & mask_cls_pseudo_thr & mask_score_pseudo_thr
            mask_list.append(mask)
            dets = dets[mask]
            if len(dets) > 0:
                scores = dets[:, 1]
            dets = dets.unsqueeze(0)
            if self.pseudo_label_group_num > 1:
                device = dets.device
            dets = dets.detach().cpu().numpy()
            calibs = [self.inference_set.get_calib(index) for index in info['img_id']]
            info = {key: val.detach().cpu().numpy() for key, val in info.items()}
            cls_mean_size = self.inference_set.cls_mean_size
            dets, cls_scores, depth_score_list, scores = decode_detections(
                dets=dets,
                info=info,
                calibs=calibs,
                cls_mean_size=cls_mean_size,
                threshold=self.cfg["tester"].get('threshold', 0.2))
            calib = calibs[0]
            dets_img = dets[int(info['img_id'])]
            if self.pseudo_label_group_num > 1:
                if len(dets_img) >= 1:
                    dets_img = torch.tensor(dets_img, dtype=torch.float32).to(device)
                    loc = dets_img[:, 9:12]
                    h = dets_img[:, 6:7]
                    w = dets_img[:, 7:8]
                    l = dets_img[:, 8:9]
                    ry = dets_img[:, 12:13]
                    loc_lidar = torch.tensor(calib.rect_to_lidar(loc.detach().cpu().numpy()), dtype=torch.float32).to(
                        device)
                    loc_lidar[:, 2] += h[:, 0] / 2
                    heading = -(torch.pi / 2 + ry)
                    boxes_lidar = torch.concatenate([loc_lidar, l, w, h, heading], axis=1)
                    dets_after_nms, _ = nms_gpu(boxes_lidar, scores, thresh=0.55)
                    dets_img = dets_img[dets_after_nms].detach().cpu().numpy()
                    pass

        return dets_img

    def get_boxes_lidar_and_clsscore(self, batch_dets, batch_calibs, batch_size, cls_pseudo_thr,
                                     score_pseudo_thr, depth_score_thr, info):
        cls_score_list = batch_dets[:, :, 1]
        score_list = []
        depth_score_list = []
        scores_list = []
        pseudo_labels_list = []
        # print(f"cls_scroe_list:      {cls_score_list.shape}")
        for bz in range(batch_size):
            dets = batch_dets[bz]
            calib = batch_calibs[bz]
            # target=batch_targets[bz]
            pseudo_labels = dets[:, 0]
            mask_cls_type = np.zeros((len(pseudo_labels)), dtype=bool)
            mask_cls_pseudo_thr = np.zeros((len(pseudo_labels)), dtype=bool)
            mask_score_pseudo_thr = np.zeros((len(pseudo_labels)), dtype=bool)
            mask_depth_score_pseudo_thr = np.zeros((len(pseudo_labels)), dtype=bool)
            for i in range(len(pseudo_labels)):
                if self.id2cls[int(pseudo_labels[i])] in self.writelist:
                    mask_cls_type[i] = True
                if dets[i, 1] > cls_pseudo_thr:
                    mask_cls_pseudo_thr[i] = True
                    score_list.append(dets[i, 1])
                score = dets[i, 1] * dets[i, -1]
                if score > score_pseudo_thr:
                    mask_score_pseudo_thr[i] = True
                if dets[i, -1] > depth_score_thr:
                    mask_depth_score_pseudo_thr[i] = True
            mask = mask_cls_type & mask_cls_pseudo_thr & mask_score_pseudo_thr & mask_depth_score_pseudo_thr
            # print(mask.shape)
            dets = dets[mask]
            pseudo_labels_list.append(dets[:, 0])
            if len(dets) > 0:
                scores = dets[:, 1]
            dets = dets.unsqueeze(0)
            if self.pseudo_label_group_num > 1:
                device = dets.device
            else:
                device = 'cpu'
            dets = dets.detach().cpu().numpy()
            calibs = [self.inference_set.get_calib(index) for index in info['img_id']]
            info = {key: val.detach().cpu().numpy() for key, val in info.items()}
            cls_mean_size = self.inference_set.cls_mean_size
            dets, cls_scores, depth_score_list1, scores_list1 = decode_detections(
                dets=dets,
                info=info,
                calibs=calibs,
                cls_mean_size=cls_mean_size,
                threshold=self.cfg["tester"].get('threshold', 0.1))
            calib = calibs[0]
            dets_img = dets[int(info['img_id'])]
            depth_score = depth_score_list1[int(info['img_id'])]
            score_from_list = scores_list1[int(info['img_id'])]
            depth_score_list.append(depth_score)
            scores_list.append(score_from_list)
            if len(dets_img) >= 1:
                dets_img = torch.tensor(dets_img, dtype=torch.float32).to(device)
                loc = dets_img[:, 9:12]
                h = dets_img[:, 6:7]
                w = dets_img[:, 7:8]
                l = dets_img[:, 8:9]
                ry = dets_img[:, 12:13]
                loc_lidar = torch.tensor(calib.rect_to_lidar(loc.detach().cpu().numpy()),
                                         dtype=torch.float32).to(device)
                loc_lidar[:, 2] += h[:, 0] / 2
                heading = -(torch.pi / 2 + ry)
                boxes_lidar = torch.concatenate([loc_lidar, l, w, h, heading], axis=1)
                pass
            else:
                boxes_lidar = None
                loc = None

        score_list = torch.tensor(score_list)
        depth_score_list = torch.tensor(depth_score_list)
        depth_score_list = torch.squeeze(depth_score_list, dim=0)
        scores_list = torch.tensor(scores_list)
        scores_list = torch.squeeze(scores_list, dim=0)
        # pseudo_labels_list = torch.tensor(pseudo_labels_list)
        return boxes_lidar, score_list, loc, depth_score_list, scores_list, pseudo_labels_list