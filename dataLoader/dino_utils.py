from distutils.dir_util import copy_tree
import torch,math,itertools
import torch.nn.functional as F
from torchvision import transforms as T
from tqdm import tqdm
import os

PATCH_H = 40
PATCH_W = 40
FEAT_DIM = 384 # vits14

def load_model():
    dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    #dinov2_vits14.register_forward_pre_hook(lambda _, x: CenterPadding(dinov2_vits14.patch_size)(x[0]))
    dinov2_vits14.eval()
    return dinov2_vits14

class FeatureExtractor:
    def __init__(self,model,batch_size,device):
        self.device = device
        self.batch_size = batch_size
        self.indices_changed = []
        self.model = model

    def compute_dino_features(self,image):
        with torch.inference_mode():
            features_dict = self.model.forward_features(image.to(self.device))
            features_batch = features_dict['x_norm_patchtokens']
        return features_batch

    def get_dino_features(self,dataset,path,save=False):
        os.makedirs(path,exist_ok=True)
        all_features = []
        indices_changed = []
        length_dataset = len(dataset)
        #If the file already exists skip it
        existing_index = 0
        get_feature_path = lambda idx: os.path.join(path,f"feature_{idx}.pt")
        while os.path.isfile(get_feature_path(existing_index)):
            if existing_index %2 == 0: 
                print("reading file number",existing_index)
                features_batch = torch.load(get_feature_path(existing_index)).unsqueeze(0)
                all_features.append(features_batch)
            existing_index += 1
        if existing_index: print("Cached",existing_index,"files from",path)
        #Start from unknown
        batch_start_idx = existing_index//self.batch_size
        batch_end_idx = length_dataset//self.batch_size
        for idx_sample in tqdm(range(batch_start_idx,batch_end_idx)):
            #TODO: Could implement to fix eventual corruptions
            #Otherwise compute the features and save them
            indices,image_batch = dataset.get_batch(idx_sample,self.batch_size)
            if not len(indices): break
            features_batch = self.compute_dino_features(image_batch)
            #Keep tracked of the ones we compute
            all_features.append(features_batch.detach().cpu())
            indices_changed += indices

        if len(indices_changed): print("Computed",len(indices_changed),"features")
        all_features = torch.concatenate(all_features,dim=0)
        if save: self.save_dino_features(all_features,indices_changed,get_feature_path)
        return all_features

    def save_dino_features(self,features,indices_changed,name_map):
        print("features shape",features.shape)
        if len(indices_changed)==0:
            print("All feature already existed")
            return
        for current_index in indices_changed:
            feature = features[current_index]
            torch.save(feature,f=name_map(current_index))



DINO_DEPTH_TRANSFORM = T.Compose([
        T.ToTensor(),
        lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
        T.Normalize(
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
        ),
    ])

GET_FEATURES_TRANSFORM = T.Compose([
    T.GaussianBlur(9, sigma=(0.1, 2.0)),
    T.Resize((PATCH_H * 14, PATCH_W * 14)),
    T.CenterCrop((PATCH_H * 14, PATCH_W * 14)),
    T.ToTensor(),
    lambda x: x[:3], # Discard alpha component
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])