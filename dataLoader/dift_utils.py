import torch
import os
from torchvision import transforms as T
from torchvision.transforms import PILToTensor
from . import FeatureExtractor
from .dift.src.models.dift_sd import SDFeaturizer

def custom_transform(img,img_wh):
    img = img.convert('RGB')
    img = img.resize(img_wh)
    img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
    return img_tensor

class DiftFeatureExtractor(FeatureExtractor):
    def __init__(self,device,args):
        super().__init__(device,args)
        self.model = SDFeaturizer(device)
        self.img_wh = (768,768)
        self.transform = lambda img: custom_transform(img,self.img_wh)
        self.forward_function = lambda img: self.model.forward(img,prompt=args.prompt)
        
    def compute_features(self,image,prompt=None):
        image = image.reshape(1,3,self.img_wh[0],self.img_wh[1])
        with torch.inference_mode():
            if prompt is not None:
                ft = self.model.forward(image,prompt=prompt)
            else:
                ft = self.forward_function(image)
        return ft.reshape(self.img_wh[0],self.img_wh[1],-1)