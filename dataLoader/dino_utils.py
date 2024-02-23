import torch
from torchvision import transforms as T
from tqdm import tqdm
import os
from .image_loader import ImageLoader


def load_model(model_name:str):
    dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', model_name)
    dinov2_vits14.eval()
    return dinov2_vits14

def get_feature_file_namer(path):
    return lambda idx: os.path.join(path,f"feature_{idx}.pt")

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

    def get_dino_features_from_id(self,dataset:ImageLoader,path:str,idx:list):
        '''
        Output dim: [batch_size,PATCH_H,PATCH_W,FEAT_DIM]
        '''
        all_features = []
        get_feature_path = get_feature_file_namer(path)
        for i in idx:
            #If the file already exists load it
            if os.path.isfile(get_feature_path(i)):
                features_batch = torch.load(get_feature_path(i)).unsqueeze(0)
            else:
                #Compute it
                image_batch = dataset[i]['rgbs'].unsqueeze(0)
                features_batch = self.compute_dino_features(image_batch).detach().cpu()
            all_features.append(features_batch)
        return torch.concatenate(all_features,dim=0)

    def get_dino_features(self,dataset,path,save=False):
        os.makedirs(path,exist_ok=True)
        all_features = []
        indices_changed = []
        length_dataset = len(dataset)
        #If the file already exists skip it
        existing_index = 0
        get_feature_path = get_feature_file_namer(path)
        while os.path.isfile(get_feature_path(existing_index)):
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
        print("Saved",len(indices_changed),"features")