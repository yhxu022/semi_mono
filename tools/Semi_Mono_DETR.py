from mmengine.model import BaseModel
import torch
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections
import numpy as np
class Semi_Mono_DETR(BaseModel):
    def __init__(self,model,loss,cfg,dataloader):
        super().__init__()
        self.model=model
        self.loss=loss
        self.cfg=cfg
        self.writelist=cfg["dataset"]["writelist"]
        self.resolution = np.array([1280, 384])  # W * H
        self.id2cls = {0: 'Pedestrian', 1: 'Car', 2: 'Cyclist'}
        self.dataloader = dataloader
        #self.max_objs = dataloader.dataset.max_objs    # max objects per images, defined in dataset
        self.max_objs = dataloader["dataset"].max_objs

    def forward(self, inputs, calibs, targets, info, mode):
        if mode == 'loss':
            img_sizes = targets['img_size']
            ##dn
            dn_args = None
            if self.cfg["model"]["use_dn"]:
                dn_args=(targets, self.cfg["model"]['scalar'], self.cfg["model"]['label_noise_scale'], self.cfg["model"]['box_noise_scale'], self.cfg["model"]['num_patterns'])
            ###
            # train one batch
            targets = self.prepare_targets(targets, inputs.shape[0])
            outputs = self.model(inputs, calibs,img_sizes, dn_args=dn_args)
            mask_dict=None
            #ipdb.set_trace()
            detr_losses_dict = self.loss(outputs, targets, mask_dict)
            weight_dict = self.loss.weight_dict
            detr_losses_dict_weighted = {k:detr_losses_dict[k] * weight_dict[k] for k in detr_losses_dict.keys() if k in weight_dict}
            return detr_losses_dict_weighted
        elif mode == 'predict':
            img_sizes = info['img_size']
            ###dn
            targets = self.prepare_targets(targets, inputs.shape[0])
            outputs = self.model(inputs, calibs,img_sizes, dn_args = 0)
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
        elif mode == 'get_pseudo_targets':
            img_sizes = info['img_size']
            outputs = self.model(inputs, calibs,img_sizes, dn_args = 0)
            dets = extract_dets_from_outputs(outputs=outputs, K=self.max_objs, topk=self.cfg["semi_train_cfg"]['topk'])
            pseudo_targets_list,mask,cls_score_list=self.get_pseudo_targets_list(dets,calibs, dets.shape[0],self.cfg["semi_train_cfg"]["cls_pseudo_thr"],self.cfg["semi_train_cfg"]["score_pseudo_thr"])
            return pseudo_targets_list,mask,cls_score_list
        elif mode == 'unsup_loss':
            img_sizes = info['img_size']
            ##dn
            dn_args = None
            if self.cfg["model"]["use_dn"]:
                dn_args=(targets, self.cfg["model"]['scalar'], self.cfg["model"]['label_noise_scale'], self.cfg["model"]['box_noise_scale'], self.cfg["model"]['num_patterns'])
            ###
            # train one batch
            outputs = self.model(inputs, calibs,img_sizes, dn_args=dn_args)
            mask_dict=None
            detr_losses_dict = self.loss(outputs, targets, mask_dict)
            weight_dict = self.loss.weight_dict
            detr_losses_dict_weighted = {k:detr_losses_dict[k] * weight_dict[k] for k in detr_losses_dict.keys() if k in weight_dict}
            return detr_losses_dict_weighted
        
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
    
    def get_pseudo_targets_list(self, batch_dets, batch_calibs, batch_size, cls_pseudo_thr, score_pseudo_thr):
        pseudo_targets_list = []
        mask_list = []
        cls_score_list = batch_dets[:,:,1]
        for bz in range(batch_size):
            dets=batch_dets[bz]
            calib=batch_calibs[bz]
            #target=batch_targets[bz]
            pseudo_labels=dets[:,0]
            mask_cls_type=np.zeros((len(pseudo_labels)), dtype=bool)
            mask_cls_pseudo_thr=np.zeros((len(pseudo_labels)), dtype=bool)
            mask_score_pseudo_thr=np.zeros((len(pseudo_labels)), dtype=bool)
            for i in range(len(pseudo_labels)):
                if self.id2cls[int(pseudo_labels[i])] in self.writelist:
                    mask_cls_type[i]=True
                if dets[i,1]>cls_pseudo_thr:
                    mask_cls_pseudo_thr[i]=True
                score = dets[i,1] * dets[i,-1]
                if score >score_pseudo_thr:
                    mask_score_pseudo_thr[i]=True
            mask = mask_cls_type & mask_cls_pseudo_thr & mask_score_pseudo_thr
            mask_list.append(mask)
            dets=dets[mask]
            pseudo_target_dict = {}     
            pseudo_labels=dets[:,0].to(torch.int8)
            pseudo_target_dict["labels"]=pseudo_labels
            if(len(pseudo_labels))==0:
                device=pseudo_labels.device
                pseudo_target_dict["calibs"]=torch.zeros(size=(0,3,4),dtype=torch.float32,device=device)
                pseudo_target_dict["boxes"]=torch.zeros(size=(0,4),dtype=torch.float32,device=device)
                pseudo_target_dict["depth"]=torch.zeros(size=(0,1),dtype=torch.float32,device=device)
                pseudo_target_dict["size_3d"]=torch.zeros(size=(0,3),dtype=torch.float32,device=device)
                pseudo_target_dict["boxes_3d"]=torch.zeros(size=(0,6),dtype=torch.float32,device=device)
                pseudo_target_dict["heading_bin"]=torch.zeros(size=(0,1),dtype=torch.int64,device=device)
                pseudo_target_dict["heading_res"]=torch.zeros(size=(0,1),dtype=torch.float32,device=device)
                pseudo_targets_list.append(pseudo_target_dict)
                continue
            for i in range(len(pseudo_labels)):
                if (i==0):
                    calibs=calib.unsqueeze(0)
                else:
                    calibs=torch.cat((calibs,calib.unsqueeze(0)))
            pseudo_target_dict["calibs"]=calibs
            boxes=dets[:,2:6].to(torch.float32)
            pseudo_target_dict["boxes"]=boxes
            depth=dets[:,6:7].to(torch.float32)
            pseudo_target_dict["depth"]=depth
            size_3d=dets[:,31:34].to(torch.float32)
            pseudo_target_dict["size_3d"]=size_3d
            x3d=dets[:,34:35].to(torch.float32)
            y3d=dets[:,35:36].to(torch.float32)
            x=boxes[:,0:1]
            y=boxes[:,1:2]
            w=boxes[:,2:3]
            h=boxes[:,3:4]
            corner_2d = [x-w/2, y-h/2, x+w/2, y+h/2]
            l, r = x3d - corner_2d[0], corner_2d[2] - x3d
            t, b = y3d - corner_2d[1], corner_2d[3] - y3d
            boxes_3d=torch.cat((x3d,y3d,l,r,t,b),dim=1)
            pseudo_target_dict["boxes_3d"]=boxes_3d

            for i in range(len(pseudo_labels)):
                heading=dets[i,7:31]
                heading_bin, heading_res = heading[0:12], heading[12:24]
                heading_bin = torch.argmax(heading_bin).unsqueeze(0)
                heading_res = heading_res[heading_bin]
                if(i==0):
                    heading_bins=heading_bin.unsqueeze(0)
                    heading_ress=heading_res.unsqueeze(0)
                else:
                    heading_bins=torch.cat((heading_bins,heading_bin.unsqueeze(0)),dim=0)
                    heading_ress=torch.cat((heading_ress,heading_res.unsqueeze(0)),dim=0)
            pseudo_target_dict["heading_bin"]=heading_bins
            pseudo_target_dict["heading_res"]=heading_ress
            pseudo_targets_list.append(pseudo_target_dict)
        return pseudo_targets_list,mask_list,cls_score_list