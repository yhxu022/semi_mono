import torch
import clip
from PIL import Image
from tqdm import tqdm

def zeroshot_classifier(classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14@336px', device=device)

kitti_classes = ["Car", "Pedestrian", "Cyclist","Van","Truck","Person_sitting","Tram","Misc","Backgournd"]
kitti_templates = ["This is a photo of a {}."]
print(f"{len(kitti_classes)} classes, {len(kitti_templates)} templates")
zeroshot_weights = zeroshot_classifier(kitti_classes, kitti_templates)
image = preprocess(Image.open("/data/ipad_3d/monocular/semi_mono/data/KITTIDataset/training/image_2/000008.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram","a car","a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    # predict
    image_features /= image_features.norm(dim=-1, keepdim=True)
    logits = 100. * image_features @ zeroshot_weights
    pred = logits.topk(max(topk=(1,)), 1, True, True)[1].t()
    print("Predictions:", pred)  

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]