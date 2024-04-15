import torch
import clip
from PIL import Image
from tqdm import tqdm
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
from utils.iou2d_utils import bbox_iou
class Glip_Kitti(object):
    def __init__(self):
        self.device = None
        self.TEXT_PROMPT = "van . car . truck ."
        self.BOX_TRESHOLD = 0.35
        self.TEXT_TRESHOLD = 0.25
        print(f"{len(self.kitti_classes)} classes, {len(self.kitti_templates)} templates")

    def zeroshot_classifier(self, classnames, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames):
                texts = [template.format(classname) for template in templates]  # format with class
                texts = clip.tokenize(texts).cuda()  # tokenize
                class_embeddings = self.model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights

    def predict(self, image, device=None):
        if self.device is None:
            self.device = device
            self.model = load_model("thirdparty/GroundingDINO/config/GroundingDINO_SwinB_cfg.py",
                       "thirdparty/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth")


        with torch.no_grad():
            boxes, logits, phrases = predict(
                model=self.model,
                image=image,
                caption=self.TEXT_PROMPT,
                box_threshold=self.BOX_TRESHOLD,
                text_threshold=self.TEXT_TRESHOLD,
                device=self.device,
                remove_combined=True
            )

        return boxes,logits

    def analyze_pred_result(self, boxes_from_glip, boxes_from_pres):
        IOU = bbox_iou(boxes_from_glip, boxes_from_pres)



if __name__ == "__main__":
    model = load_model("thirdparty/GroundingDINO/config/GroundingDINO_SwinB_cfg.py",
                       "thirdparty/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth")
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