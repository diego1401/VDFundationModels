import torch
import os
from torchvision import transforms as T
from . import FeatureExtractor

PATCH_H = 57
PATCH_W = 57

SIZE_TO_MODEL = {
    "small": ('dinov2_vits14',384),
    "big": ('dinov2_vitg14',1536)
}


def load_model(model_name:str):
    dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', model_name)
    dinov2_vits14.eval()
    return dinov2_vits14

def get_feature_file_namer(path):
    return lambda idx: os.path.join(path,f"feature_{idx}.pt")



class DinoFeatureExtractor(FeatureExtractor):
    def __init__(self,device,args):
        super().__init__(device,args)
        model_name, _ = SIZE_TO_MODEL[args.model_size]
        self.model = load_model(model_name)
        self.model = self.model.to(device)
        self.img_wh = (PATCH_H * 14, PATCH_W * 14)
        self.transform = T.Compose([
            T.Resize(self.img_wh),
            T.CenterCrop(self.img_wh),
            T.ToTensor(),
            lambda x: x[:3], # Discard alpha component
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        

    def compute_features(self,image):
        image = image.reshape(1,3,PATCH_H*14,PATCH_W*14)
        with torch.inference_mode():
            features_dict = self.model.forward_features(image.to(self.device))
            features_batch = features_dict['x_norm_patchtokens']
        return features_batch.reshape(PATCH_H*PATCH_W,-1)