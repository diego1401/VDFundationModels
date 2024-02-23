import numpy as np
from PIL import Image
import json
import torch,cv2, os
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms as T
from dataLoader import POSSIBLE_SPLITS, GET_FEATURES_TRANSFORM


class ImageLoader(Dataset):
    def __init__(self,datadir,split="train"):
        self.root_dir = datadir
        self.split_dir = os.path.join(datadir,split)
        self.name = lambda idx: f"r_{idx}.png"
        self.split = split
        assert self.split in POSSIBLE_SPLITS
        self.H = None
        self.W = None
        self.define_transforms()
        self.read_data()

    def define_transforms(self):
        self.transform = GET_FEATURES_TRANSFORM

    def get_number_of_files(self):
        #NeRF dataset   
        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            meta = json.load(f)
        number_of_files = len(meta['frames'])# number of images in split

        return number_of_files

    def read_data(self):
        number_of_files = self.get_number_of_files()
        self.all_rgbs = []
        for i in tqdm(range(number_of_files)):
            image_path = os.path.join(self.split_dir, self.name(i))
            
            img = Image.open(image_path)
            img = self.transform(img)  # (4, h, w)
            if not (self.H or self.W):
                _ , self.H,self.W = img.shape
            
            if img.shape[0] == 4:
                img = img.view(4, -1)  # (h*w, 4) RGBA
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            elif img.shape[0] == 3:
                img = img.view(3,-1)
            self.all_rgbs += [img]
        self.img_wh = self.all_rgbs[-1].shape
        self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,3,self.H,self.W)

    def get_batch(self,idx,batch_size):
        start_batch = idx*batch_size
        end_batch = min((idx+1)*batch_size,self.__len__())
        indices = list(range(start_batch,end_batch))
        return indices,self.all_rgbs[start_batch:end_batch]

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):
        img = self.all_rgbs[idx]
        sample = {'rgbs': img}
        return sample