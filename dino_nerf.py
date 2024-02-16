import torch

def main():
    dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

main()