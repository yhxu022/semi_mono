from mmengine.model import BaseModel
import torch
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections, decode_detections_GPU
import numpy as np
from uncertainty_estimator import UncertaintyEstimator
from pcdet.ops.iou3d_nms.iou3d_nms_utils import nms_gpu
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
from torchvision.transforms import ToPILImage

def class2angle_gpu(cls, residual, to_label_format=False, num_heading_bin=12):

    angle_per_class = 2 * torch.pi / float(num_heading_bin)
    angle_center = cls.float() * angle_per_class  # Ensure cls is float for multiplication
    angle = angle_center + residual

    if to_label_format:
        # Using torch.where to handle condition across tensors
        angle = torch.where(angle > torch.pi, angle - 2 * torch.pi, angle)

    return angle

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

def crop(input, bbox2d):
    bbox2d[0] = bbox2d[0] * 1280
    bbox2d[1] = bbox2d[1] * 384
    bbox2d[2] = bbox2d[2] * 1280
    bbox2d[3] = bbox2d[3] * 384
    x = bbox2d[0]
    y = bbox2d[1]
    w = bbox2d[2]
    h = bbox2d[3]
    corner_2d = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
    corner_2d = torch.stack(corner_2d)
    x1 = torch.round(corner_2d[0]).int()
    y1 = torch.round(corner_2d[1]).int()
    x2 = torch.round(corner_2d[2]).int()
    y2 = torch.round(corner_2d[3]).int()
    input_croped = input[:,y1:y2, x1:x2]
    return input_croped
    
class Semi_Mono_DETR(BaseModel):
    def __init__(self, model, loss, cfg, dataloader, inference_set=None, unlabeled_set=None):
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
        self.unlabeled_set = unlabeled_set
        self.use_clip = cfg.get("use_clip", True)
        if self.use_clip:
            print("------USE CLIP TO HELP------")
        self.decouple = cfg["semi_train_cfg"].get('decouple', False)

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
            if self.use_clip is True:
                clip_inputs = inputs
            else:
                clip_inputs = None
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
                                                                                                    0),
                                                                                                batch_targets=targets,
                                                                                                batch_inputs=clip_inputs,
                                                                                                cls_clip_threshold=self.cfg[
                                                                                                    "semi_train_cfg"].get(
                                                                                                    "cls_clip_thr",
                                                                                                    0.0))
                if self.decouple:
                    regression_pseudo_targets_list, regression_mask, regression_cls_score = self.get_pseudo_targets_list(
                        dets, calibs, dets.shape[0],
                        self.cfg["semi_train_cfg"][
                            "regression_cls_pseudo_thr"],
                        self.cfg["semi_train_cfg"][
                            "regression_score_pseudo_thr"],
                        self.cfg["semi_train_cfg"].get("regression_depth_score_thr", 0),
                        batch_targets=targets,
                        batch_inputs=None)
                else:
                    regression_pseudo_targets_list = regression_mask = regression_cls_score = None
                cls_topk_boxes = regression_topk_boxes = topk_boxes
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
                                                                                                    0),
                                                                                                batch_targets=targets,
                                                                                                batch_inputs=clip_inputs,
                                                                                                cls_clip_threshold=self.cfg[
                                                                                                    "semi_train_cfg"].get(
                                                                                                    "cls_clip_thr",
                                                                                                    0.0)
                                                                                                )
                if self.decouple:
                    regression_pseudo_targets_list, regression_mask, regression_cls_score = self.get_pseudo_targets_list(
                        dets, calibs, dets.shape[0],
                        self.cfg["semi_train_cfg"][
                            "regression_cls_pseudo_thr"],
                        self.cfg["semi_train_cfg"][
                            "regression_score_pseudo_thr"],
                        self.cfg["semi_train_cfg"].get("regression_depth_score_thr", 0),
                        batch_targets=targets,
                        batch_inputs=None)
                else:
                    regression_pseudo_targets_list = regression_mask = regression_cls_score = None
                cls_topk_boxes = regression_topk_boxes = topk_boxes

            return cls_pseudo_targets_list, cls_mask, cls_cls_score, cls_topk_boxes, regression_pseudo_targets_list, \
                regression_mask, regression_cls_score, regression_topk_boxes
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
                    info,batch_inputs=inputs,cls_clip_threshold=self.cfg["semi_train_cfg"].get("cls_clip_thr",0.0))
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
                    info,batch_inputs=inputs,cls_clip_threshold=self.cfg["semi_train_cfg"].get("cls_clip_thr",
                                                                                                    0.0))
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
                                depth_score_thr,batch_targets=None,batch_inputs=None,cls_clip_threshold=0.0):
        pseudo_targets_list = []
        mask_list = []
        cls_score_list = batch_dets[:, :, 1]
        for bz in range(batch_size):
            dets = batch_dets[bz]
            calib = batch_calibs[bz]
            pseudo_labels = dets[:, 0]
            mask_cls_type = np.zeros((len(pseudo_labels)), dtype=bool)
            mask_cls_pseudo_thr = np.zeros((len(pseudo_labels)), dtype=bool)
            mask_score_pseudo_thr = np.zeros((len(pseudo_labels)), dtype=bool)
            mask_depth_score_pseudo_thr = np.zeros((len(pseudo_labels)), dtype=bool)
            for i in range(len(pseudo_labels)):
                if self.id2cls[int(pseudo_labels[i])] in self.writelist:
                    mask_cls_type[i] = True
                if dets[i, 1] > cls_pseudo_thr :
                    if batch_inputs is not None:
                        #如果初筛通过,将2dbbox对应的图片区域裁剪下来送入clip模型精筛,若为正样本则保留,否则舍弃
                        boxes = dets[i, 2:6].to(torch.float32)
                        img = batch_inputs[bz]
                        img_croped = crop(img, boxes.clone())
                        if img_croped.shape[1]>0 and img_croped.shape[2]>0:
                            croped_image=img_croped.cpu().numpy().transpose(1, 2, 0)
                            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                            croped_image = (croped_image * std + mean) * 255.0
                            croped_image=ToPILImage()(np.round(croped_image).astype(np.uint8))
                            # croped_image.save("croped_image.jpg")
                            probs, pred = self.clip_kitti.predict(croped_image, device=img.device)
                            if int(pred)==pseudo_labels[i] and probs.max() > cls_clip_threshold:
                                mask_cls_pseudo_thr[i] = True
                    else:
                        mask_cls_pseudo_thr[i] = True
                score = dets[i, 1] * dets[i, -1]
                if score > score_pseudo_thr :
                    mask_score_pseudo_thr[i] = True
                if dets[i, -1] > depth_score_thr :
                    mask_depth_score_pseudo_thr[i] = True
            mask = mask_cls_type & mask_cls_pseudo_thr & mask_score_pseudo_thr & mask_depth_score_pseudo_thr
            mask_list.append(mask)
            dets = dets[mask]
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
                                     score_pseudo_thr, depth_score_thr, info,batch_inputs=None,cls_clip_threshold=0.0):
        cls_score_list = batch_dets[:, :, 1]
        score_list = []
        depth_score_list = []
        scores_list = []
        pseudo_labels_list = []
        prob_from_clip = []
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
                    if batch_inputs is not None:
                        # 如果初筛通过,将2dbbox对应的图片区域裁剪下来送入clip模型精筛,若为正样本则保留,否则舍弃
                        boxes = dets[i, 2:6].to(torch.float32)
                        img = batch_inputs[bz]
                        img_croped = crop(img, boxes.clone())
                        if img_croped.shape[1] > 0 and img_croped.shape[2] > 0:
                            croped_image = img_croped.cpu().numpy().transpose(1, 2, 0)
                            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                            croped_image = (croped_image * std + mean) * 255.0
                            croped_image = ToPILImage()(np.round(croped_image).astype(np.uint8))
                            # croped_image.save("croped_image.jpg")
                            probs, pred = self.clip_kitti.predict(croped_image, device=img.device)
                            # if int(pred) == pseudo_labels[i] or int(pred) == len(probs[0])-1:
                            # if int(pred) == pseudo_labels[i]:
                            if self.clip_kitti.analyze_pred_result(prob=probs, pred=pred, label=pseudo_labels[i],thr=cls_clip_threshold):
                                mask_cls_pseudo_thr[i] = True
                                prob_from_clip.append(probs[0][pred])
                            # else:
                            #     if int(pred) != pseudo_labels[i] and pseudo_labels[i] == 1:
                            #         with open(file="your_file_path.txt", mode="a") as f:
                            #             f.write(f"{info['img_id'][bz]}_{i} -- {pred} -- {probs}\n")
                            #             croped_image.save(f"wrong/{info['img_id'][bz]}_{i}.jpg")
                    else:
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
        prob_from_clip = torch.tensor(prob_from_clip)
        depth_score_list = torch.tensor(depth_score_list)
        depth_score_list = torch.squeeze(depth_score_list, dim=0)
        scores_list = torch.tensor(scores_list)
        scores_list = torch.squeeze(scores_list, dim=0)
        # pseudo_labels_list = torch.tensor(pseudo_labels_list)
        return boxes_lidar, score_list, loc, depth_score_list, scores_list, pseudo_labels_list
        # return boxes_lidar, score_list, loc, prob_from_clip, scores_list, pseudo_labels_list

    # score_list：分类分    depth_score_list：深度分    scores_list：分类分和深度分相乘
