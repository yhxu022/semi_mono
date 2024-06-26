import torch
import clip
from PIL import Image
from tqdm import tqdm


class Clip_Kitti(object):
    def __init__(self):
        self.device = None
        # self.kitti_classes = ['Pedestrian', 'Car', 'Cyclist',"Van","Truck","Person_sitting","Tram","Background","asphalt road","road","tree","sky","wall"]
        # "Black Car","Black Van","Black Truck","Gray Car","Gray Van","Gray Truck", "SUV"]
        # self.kitti_classes = ["Background",'Car', 'Cyclist',"Van","Truck","asphalt road","road","tree","sky","wall","SUV"]
        # self.kitti_classes = ["Background", 'Car', 'Cyclist', "Van", "Truck", "asphalt road", "road", "tree", "sky", "wall", "Sedan"]
        self.kitti_classes = ["Van", 'Car', "Truck"]
        # self.kitti_classes = ["Car", "Truck","Pickup Truck","Sedan","Saloon","SUV","Coupe","Sports Car"]
        # self.kitti_classes = ["Car","Delivery Truck", "Dump Truck", "Semi Truck", "Pickup Truck", "Sedan", "Saloon", "SUV", "Coupe", "Sports Car"]
        # self.kitti_classes = ["Delivery Truck", "Dump Truck", "Semi Truck", "Pickup Truck", "Sedan", "Saloon", "SUV", "Coupe", "Sports Car","Luxury Car"]
        self.kitti_templates = ["This is a photo of a {}."]
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
            self.model, self.preprocess = clip.load('ViT-L/14@336px', device=self.device)
            # self.model, self.preprocess = clip.load('ViT-B/32', device=self.device)

            self.zeroshot_weights = self.zeroshot_classifier(self.kitti_classes, self.kitti_templates)
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            # predict
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ self.zeroshot_weights
            probs = logits.softmax(dim=-1).cpu().numpy()
            pred = logits.topk(max((1,)), 1, True, True)[1].t()
        return probs, pred

    def analyze_pred_result(self, prob, pred, label, thr=0):
        if label == 1 and prob.max() > thr:
            if "Luxury Car" in self.kitti_classes and self.kitti_classes[int(pred)] == "Luxury Car":
                return True
            if "Sports Car" in self.kitti_classes and self.kitti_classes[int(pred)] == "Sports Car":
                return True
            if "Coupe" in self.kitti_classes and self.kitti_classes[int(pred)] == "Coupe":
                return True
            if "Saloon" in self.kitti_classes and self.kitti_classes[int(pred)] == "Saloon":
                return True
            if "SUV" in self.kitti_classes and self.kitti_classes[int(pred)] == "SUV":
                return True
            if "Sedan" in self.kitti_classes and self.kitti_classes[int(pred)] == "Sedan":
                return True

            if self.kitti_classes[int(pred)] == "Car":
                return True
        else:
            return False


if __name__ == "__main__":
    clip_kitti = Clip_Kitti()
    probs, pred = clip_kitti.predict(Image.open("/data/ipad_3d/monocular/semi_mono/outputs/Clip/19_1.jpg"), device='cuda')
    print(probs, pred, clip_kitti.kitti_classes[int(pred)])
