import torch
import clip
from PIL import Image
from tqdm import tqdm

class Clip_Kitti(object):
    def __init__(self,device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model, self.preprocess = clip.load('ViT-L/14@336px', device=self.device)
        self.kitti_classes = ['Pedestrian', 'Car', 'Cyclist',"Van","Truck","Person_sitting","Tram","Background","asphalt road","road","tree","sky","wall"]
        self.kitti_templates = ["This is a photo of a {}."]
        print(f"{len(self.kitti_classes)} classes, {len(self.kitti_templates)} templates")
        self.zeroshot_weights = self.zeroshot_classifier(self.kitti_classes, self.kitti_templates)

    def zeroshot_classifier(self,classnames, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames):
                texts = [template.format(classname) for template in templates] #format with class
                texts = clip.tokenize(texts).cuda() #tokenize
                class_embeddings = self.model.encode_text(texts) #embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights

    def predict(self, image):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            # predict
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ self.zeroshot_weights
            probs = logits.softmax(dim=-1).cpu().numpy()
            pred = logits.topk(max((1,)), 1, True, True)[1].t()
        return probs, pred
    
if __name__ == "__main__":
    clip_kitti = Clip_Kitti()
    probs, pred = clip_kitti.predict(Image.open("/data/ipad_3d/monocular/semi_mono/000043.png"))
    print(probs, pred)