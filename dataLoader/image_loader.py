import numpy as np
from PIL import Image
import json_ as json
import torch,cv2, os
from torch.utils.data import Dataset
from torchvision import transforms as T

POSSIBLE_SPLITS = ["train","test","val"]

class ImageLoader(Dataset):
    def __init__(self,datadir,split="train"):
        self.root_dir = os.path.join(datadir,split)
        self.name = lambda idx: f"r_{idx}.png"
        self.split = split
        assert self.split in POSSIBLE_SPLITS
        self.define_transforms()
        self.read_data()

    def define_transforms(self):
        self.transform = T.ToTensor()

    def get_number_of_files(self):
        #NeRF dataset   
        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            meta = json.load(f)
        number_of_files = len(meta['frames'])# number of images in split

        return number_of_files

    def read_data(self):
        number_of_files = self.get_number_of_files()
        self.all_rgbs = []
        for i in range(number_of_files):
            image_path = os.path.join(self.root_dir, self.name(i))
            
            img = Image.open(image_path)
            img = self.transform(img)  # (4, h, w)
            img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
            img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            self.all_rgbs += [img]

        self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):
        img = self.all_rgbs[idx]
        sample = {'rgbs': img}
        return sample