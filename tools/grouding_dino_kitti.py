from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

#SwinT
#model = load_model("/data/ipad_3d/monocular/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "/data/ipad_3d/monocular/GroundingDINO/weights/groundingdino_swint_ogc.pth")
#SwinB
model = load_model("thirdparty/GroundingDINO/config/GroundingDINO_SwinB_cfg.py",
                   "thirdparty/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth")
IMAGE_PATH = "/home/xyh/MonoDETR_ori/data/KITTI/training/image_2/000052.png"
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
    device='cuda',
    remove_combined=True
)
print(boxes.shape)
print(logits.shape)
print(phrases)
annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("annotated_image.jpg", annotated_frame)