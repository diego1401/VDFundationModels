from torchvision import transforms as T

PATCH_H = 57
PATCH_W = 57

SIZE_TO_MODEL = {
    "small": ('dinov2_vits14',384),
    "big": ('dinov2_vitg14',1536)
}

# DINO_DEPTH_TRANSFORM = T.Compose([
#         T.ToTensor(),
#         lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
#         T.Normalize(
#             mean=(123.675, 116.28, 103.53),
#             std=(58.395, 57.12, 57.375),
#         ),
#     ])

GET_FEATURES_TRANSFORM = T.Compose([
    #T.GaussianBlur(9, sigma=(0.1, 2.0)),
    T.Resize((PATCH_H * 14, PATCH_W * 14)),
    T.CenterCrop((PATCH_H * 14, PATCH_W * 14)),
    T.ToTensor(),
    lambda x: x[:3], # Discard alpha component
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

POSSIBLE_SPLITS = ["train","test","val"]

from .image_loader import ImageLoader
from .dino_utils import load_model,FeatureExtractor