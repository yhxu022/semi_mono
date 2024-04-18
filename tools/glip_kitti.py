import torch
import clip
from PIL import Image
from tqdm import tqdm
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from groundingdino.util.inference import load_model, load_image, predict, annotate,preprocess_caption
import cv2
from utils.iou2d_utils import bbox_iou
#from utils.box_ops import box_iou
from torchvision.ops import box_convert
from groundingdino.models.GroundingDINO.bertwarper import (
    BertModelWarper,
    generate_masks_with_special_tokens,
    generate_masks_with_special_tokens_and_transfer_map,
)
import bisect
import torch.nn.functional as F
from groundingdino.util.utils import get_phrases_from_posmap
from groundingdino.util.misc import (
    NestedTensor,
    accuracy,
    get_world_size,
    interpolate,
    inverse_sigmoid,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)

class Glip_Kitti(object):
    def __init__(self):
        self.device = None
        self.TEXT_PROMPT = "Van . Car . Truck ."
        self.BOX_TRESHOLD = 0.35
        self.TEXT_TRESHOLD = 0.25
        self.tokenized = None

        print(f"{len(set(self.TEXT_PROMPT.replace('.', '').split()))} classes")

    def forward_with_tokenized(self, model, samples, targets, **kw):

        bert_output = self.bert_output
        tokenized = self.tokenized_in_forward
        position_ids = self.position_ids
        text_self_attention_masks = self.text_self_attention_masks

        encoded_text = model.feat_map(bert_output["last_hidden_state"])  # bs, 195, d_model
        text_token_mask = tokenized.attention_mask.bool()  # bs, 195
        # text_token_mask: True for nomask, False for mask
        # text_self_attention_masks: True for nomask, False for mask

        if encoded_text.shape[1] > model.max_text_len:
            encoded_text = encoded_text[:, : model.max_text_len, :]
            text_token_mask = text_token_mask[:, : model.max_text_len]
            position_ids = position_ids[:, : model.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                                        :, : model.max_text_len, : model.max_text_len
                                        ]

        text_dict = {
            "encoded_text": encoded_text,  # bs, 195, d_model
            "text_token_mask": text_token_mask,  # bs, 195
            "position_ids": position_ids,  # bs, 195
            "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
        }

        # import ipdb; ipdb.set_trace()
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        if not hasattr(model, 'features') or not hasattr(model, 'poss'):
            model.set_image_tensor(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(self.model.features):
            src, mask = feat.decompose()
            srcs.append(model.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if model.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, model.num_feature_levels):
                if l == _len_srcs:
                    src = model.input_proj[l](model.features[-1].tensors)
                else:
                    src = model.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = model.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                model.poss.append(pos_l)

        input_query_bbox = input_query_label = attn_mask = dn_meta = None
        hs, reference, hs_enc, ref_enc, init_box_proposal = model.transformer(
            srcs, masks, input_query_bbox, model.poss, input_query_label, attn_mask, text_dict
        )

        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
                zip(reference[:-1], model.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        # output
        outputs_class = torch.stack(
            [
                layer_cls_embed(layer_hs, text_dict)
                for layer_cls_embed, layer_hs in zip(model.class_embed, hs)
            ]
        )
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord_list[-1]}

        # # for intermediate outputs
        # if self.aux_loss:
        #     out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list)

        # # for encoder output
        # if hs_enc is not None:
        #     # prepare intermediate outputs
        #     interm_coord = ref_enc[-1]
        #     interm_class = self.transformer.enc_out_class_embed(hs_enc[-1], text_dict)
        #     out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
        #     out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': init_box_proposal}
        unset_image_tensor = kw.get('unset_image_tensor', True)
        if unset_image_tensor:
            self.model.unset_image_tensor()  ## If necessary
        return out


    def predict(self, image, device=None,**kw):
        if self.device is None:
            self.device = device
            print('LOADING Grounding DINO......')
            self.model = load_model("thirdparty/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
                       "thirdparty/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth",device=device)
            self.model = self.model.to(device)
            print('LOADING OVER')
            tokenizer = self.model.tokenizer
            samples = image[None]
            caption = preprocess_caption(caption=self.TEXT_PROMPT)

            captions = [caption]
            self.tokenized_in_predict = tokenizer(caption)
            self.tokenized_in_forward = self.model.tokenizer(captions, padding="longest", return_tensors="pt").to(self.device)

            tokenized = self.tokenized_in_forward
            (
                text_self_attention_masks,
                position_ids,
                cate_to_token_mask_list,
            ) = generate_masks_with_special_tokens_and_transfer_map(
                tokenized, self.model.specical_tokens, self.model.tokenizer
            )

            if text_self_attention_masks.shape[1] > self.model.max_text_len:
                text_self_attention_masks = text_self_attention_masks[
                                            :, : self.model.max_text_len, : self.model.max_text_len
                                            ]
                position_ids = position_ids[:, : self.model.max_text_len]
                tokenized["input_ids"] = tokenized["input_ids"][:, : self.model.max_text_len]
                tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.model.max_text_len]
                tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.model.max_text_len]

            # extract text embeddings
            if self.model.sub_sentence_present:
                tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
                tokenized_for_encoder["attention_mask"] = text_self_attention_masks
                tokenized_for_encoder["position_ids"] = position_ids
            else:
                # import ipdb; ipdb.set_trace()
                tokenized_for_encoder = tokenized
            self.bert_output = self.model.bert(**tokenized_for_encoder)  # bs, 195, 768
            self.position_ids = position_ids
            self.text_self_attention_masks = text_self_attention_masks

        with torch.no_grad():
            model = self.model
            image = image
            caption = self.TEXT_PROMPT
            tokenized = self.tokenized_in_predict
            box_threshold = self.BOX_TRESHOLD
            text_threshold = self.TEXT_TRESHOLD
            device = device
            remove_combined = True

            model = model.to(device)
            image = image.to(device)

            with torch.no_grad():
                outputs = model(image[None], captions=[caption])

            prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
            prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

            mask = prediction_logits.max(dim=1)[0] > box_threshold
            logits = prediction_logits[mask]  # logits.shape = (n, 256)
            boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

            tokenizer = model.tokenizer
            # tokenized = tokenizer(caption)

            if remove_combined:
                sep_idx = [i for i in range(len(tokenized['input_ids'])) if
                           tokenized['input_ids'][i] in [101, 102, 1012]]

                phrases = []
                for logit in logits:
                    max_idx = logit.argmax()
                    insert_idx = bisect.bisect_left(sep_idx, max_idx)
                    right_idx = sep_idx[insert_idx]
                    left_idx = sep_idx[insert_idx - 1]
                    phrases.append(
                        get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer, left_idx,
                                                right_idx).replace('.',
                                                                   ''))
            else:
                phrases = [
                    get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
                    for logit
                    in logits
                ]
        mask_class=phrases=="car"
        return boxes[mask_class], logits[mask_class], phrases[mask_class]

    def analyze_pred_result(self, boxes_from_glip, boxes_from_preds, phrases, IOU_thr=0.7):
        w = 1280
        h = 384

        size = torch.tensor([w, h, w, h],device=self.device)
        # print(size.device)
        size = size.to(self.device)
        # print(f"{size.device}")
        boxes_from_glip = boxes_from_glip.to(self.device)
        boxes_from_preds = boxes_from_preds.to(self.device)

        bbox_from_glip_orisize = boxes_from_glip * size
        bbox_from_preds_orisize = boxes_from_preds * size
        if len(bbox_from_glip_orisize) == 0 or len(bbox_from_preds_orisize) == 0:
            return []
        mask_height_glip=bbox_from_glip_orisize[:,-1]>=25
        mask_height_preds=bbox_from_preds_orisize[:,-1]>=25
        bbox_from_glip_orisize_height_filtered=bbox_from_glip_orisize[mask_height_glip]
        bbox_from_preds_orisize_height_filtered=bbox_from_preds_orisize[mask_height_preds]
        if len(bbox_from_glip_orisize_height_filtered) == 0 or len(bbox_from_preds_orisize_height_filtered) == 0:
            return []

        IOUs = bbox_iou(bbox_from_glip_orisize_height_filtered, bbox_from_preds_orisize_height_filtered)
        #IOUs,_ = box_iou(bbox_from_preds_orisize, bbox_from_glip_orisize)
        max_iou, max_indices = torch.max(IOUs, dim=1)
        preds_indexes = torch.where(max_iou > IOU_thr)[0]
        glip_indexes = max_indices[preds_indexes]

        preds_indexes_filtered=[]
        idx_selected = []
        # print(phrases)
        for pred_idx, glip_idx in zip(preds_indexes, glip_indexes):
            if max_indices[pred_idx] not in idx_selected:
                if phrases[glip_idx] == 'car':
                    preds_indexes_filtered.append(pred_idx.item())
                    idx_selected.append(max_indices[pred_idx])

        # preds_indexes_filtered = [pred_idx.item() for pred_idx, glip_idx in zip(preds_indexes, glip_indexes) if
        #                           phrases[glip_idx] == 'car']

        return preds_indexes_filtered


if __name__ == "__main__":


    model = load_model("thirdparty/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
                       "thirdparty/GroundingDINO/groundingdino/weights/groundingdino_swinb_cogcoor.pth")
    IMAGE_PATH = "/data/ipad_3d/monocular/semi_mono/data/KITTIDataset/training/image_2/000052.png"
    TEXT_PROMPT = "van . car . truck ."
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    image_source, image = load_image(IMAGE_PATH)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
        remove_combined=True
    )
    print(boxes.shape)
    print(logits.shape)
    print(phrases)
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    cv2.imwrite("annotated_image.jpg", annotated_frame)