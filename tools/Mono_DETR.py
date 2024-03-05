from mmengine.model import BaseModel
import torch
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections
import numpy as np


class Mono_DETR(BaseModel):
    def __init__(self, model, loss, cfg, dataloader):
        super().__init__()
        self.model = model
        self.loss = loss
        self.cfg = cfg
        self.writelist = cfg["dataset"]["writelist"]
        self.resolution = np.array([1280, 384])  # W * H
        self.id2cls = {0: 'Pedestrian', 1: 'Car', 2: 'Cyclist'}
        self.dataloader = dataloader
        # self.max_objs = dataloader.dataset.max_objs    # max objects per images, defined in dataset
        self.max_objs = dataloader["dataset"].max_objs

    def forward(self, inputs, calibs, targets, info, mode):
        inputs=inputs[0]
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
            dets = extract_dets_from_outputs(outputs=outputs, K=self.max_objs, topk=self.cfg["tester"]['topk'])
            dets = dets.detach().cpu().numpy()
            # get corresponding calibs & transform tensor to numpy
            calibs = [self.dataloader["dataset"].get_calib(index) for index in info['img_id']]
            info = {key: val.detach().cpu().numpy() for key, val in info.items()}
            cls_mean_size = self.dataloader["dataset"].cls_mean_size
            dets = decode_detections(
                dets=dets,
                info=info,
                calibs=calibs,
                cls_mean_size=cls_mean_size,
                threshold=self.cfg["tester"].get('threshold', 0.2))
            return dets, targets

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
