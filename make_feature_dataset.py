import torch

def load_model():
    dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    return dinov2_vits14
