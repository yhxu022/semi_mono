from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

#SwinT
#model = load_model("/data/ipad_3d/monocular/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "/data/ipad_3d/monocular/GroundingDINO/weights/groundingdino_swint_ogc.pth")
#SwinB
model = load_model("/data/ipad_3d/monocular/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py", "/data/ipad_3d/monocular/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth")
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

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("annotated_image.jpg", annotated_frame)