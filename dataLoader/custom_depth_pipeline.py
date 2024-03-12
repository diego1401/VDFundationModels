import numpy as np
from PIL import Image
import json
import torch,cv2, os
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms as T
from dataLoader import PATCH_H, PATCH_W
import json
from skimage.measure import block_reduce
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from .ray_utils import get_ray_directions, get_rays

def downsample_mask(original_mask):
    # Perform downsampling using the "or" operation
    original_mask = original_mask.view(PATCH_H*14,PATCH_W*14)
    downsampled_mask = block_reduce(original_mask, block_size=(14,14), func=np.any,cval=0)
    return downsampled_mask

class PositionLoader(Dataset):
    def __init__(self,datadir):
        self.root_dir = datadir
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        #self.h,self.w = PATCH_H*14, PATCH_W*14
        self.h,self.w = 800,800
        self.img_hw = (self.h, self.w)
        with open(os.path.join(self.root_dir, "transforms.json"), 'r') as f:
            meta = json.load(f)
            self.F = 0.5 /np.tan(0.5 * meta['camera_angle_x'])
            self.F *= self.img_hw[0] / 800  # modify focal length to match size self.img_wh
            
            self.focal = 0.5 * 800 / np.tan(0.5 * meta['camera_angle_x'])  # original focal length
            self.focal *= self.img_hw[0] / 800  # modify focal length to match size self.img_wh
            
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(self.h, self.w, [self.focal,self.focal])  # (h, w, 3)\
        # We do not normalize the directions, as to not wrap the rays around the camera
        self.define_transforms()

    def define_transforms(self):
        raise ValueError("Not implemented")
        self.transform = None

    def __len__(self):
        return 100 #HARDCODED
    
    def _get_prefix_from_index(self,idx):
        return f"r_{idx:03}"
    
    def get_rgb_name(self,idx):
        return self._get_prefix_from_index(idx) + ".png"
    
    def get_depth_name(self,idx):
        return self._get_prefix_from_index(idx) + "_depth_0000.exr"  
    
    def get_rgb(self,idx):
        img_name = os.path.join(self.root_dir, self.get_rgb_name(idx))
        img = Image.open(img_name)
        img = self.transform(img)
        if (self.h is None) or (self.w is None):
            self.h,self.w = img.shape[1],img.shape[2]
        return img.view(img.shape[0],self.w,self.h)

    def get_coordinates(self,idx):
        depth_img_name = os.path.join(self.root_dir, self.get_depth_name(idx))
        depth = cv2.imread(depth_img_name, cv2.IMREAD_ANYDEPTH)
        #depth = np.ones_like(depth)
        #depth = cv2.resize(depth, (PATCH_H*14,PATCH_W*14), interpolation=cv2.INTER_NEAREST)
        depth = torch.from_numpy(depth).view(-1,1)

        c_o, d = self.get_camera_coordinates(idx)
        # f = self.F/2
        # print(f"f: {f}")
        
        dirs = d * depth
        position = c_o  + dirs 
        mask = depth < 10
        
        # Expand mask along the last dimension
        position = position[mask.expand(-1, 3)].view(-1,3)
        #return downsample_mask(mask),position
        return mask, position
    
    def get_camera_coordinates(self,idx):
        with open(os.path.join(self.root_dir, "transforms.json"), 'r') as f:
            meta = json.load(f)
        frame = meta["frames"][idx]
        pose = np.array(frame['transform_matrix']) @ self.blender2opencv
        c2w = torch.FloatTensor(pose)
        return get_rays(self.directions, c2w)  # both (h*w, 3)
        
    def __getitem__(self, idx):
        #img = self.get_rgb(idx)
        img = None
        mask,position = self.get_coordinates(idx)
        sample = {'rgbs': img,
                  'coordinates': position,
                  'mask':mask}
        return sample