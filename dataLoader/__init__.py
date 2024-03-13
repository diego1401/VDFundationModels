from torchvision import transforms as T
POSSIBLE_SPLITS = ["train","test","val"]

class FeatureExtractor:
    def __init__(self,device,args):
        self.device = device
        self.transform = None

    def compute_features(self,image):
        raise ValueError("Cannot Use Abstract Class")

from .dift_utils import DiftFeatureExtractor
from .dino_utils import DinoFeatureExtractor, SIZE_TO_MODEL, PATCH_H, PATCH_W
from .image_loader import ImageLoader
from .custom_depth_pipeline import PositionLoader