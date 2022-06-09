import torch
import torch.nn as nn
import torch.nn.functional as F
from vit_pytorch import ViT
from vit_pytorch import SimpleViT

import timm
from functools import partial
from timm.models.vision_transformer import VisionTransformer, _cfg

def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

class EncoderViT(nn.Module):
    def __init__(self, args):
        super().__init__()

        # This is ViT-Ti settings
        self.regressor = ViT(
                    image_size=int(args.img_dim),
                    patch_size=16,
                    num_classes=50,
                    dim=192,
                    depth=12,
                    heads=3,
                    mlp_dim=768,
                    dropout=0.1,
                    emb_dropout=0.1
                )       
        
        # self.regressor = deit_small_patch16_224(pretrained=True)
        

    def forward(self, x):
        x = self.regressor(x)
        # print(x.shape)
        
        return x

class RegressorViT(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(50, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        sa = self.fc2(x)

        return sa

class DecoderViT(nn.Module):
    def __init__(self, args, in_dim=50, out_dim=64*5*13):
        super().__init__()

        self.args = args
        self.img_dim = int(args.img_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.decFC1 = nn.Linear(in_dim, out_dim)
        self.decConv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1)
        self.decConv3 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1)
        self.decConv4 = nn.ConvTranspose2d(16, 8, 3, stride=2, padding=(1,0))
        self.decConv5 = nn.ConvTranspose2d(8, 3, 3, stride=2, padding=(1,0), output_padding=1)
        self.decFC2 = nn.Linear(3*66*200, (3*self.img_dim*self.img_dim))
    
    def forward(self, x):
        x = self.relu(self.decFC1(x))
        
        x = x.reshape(-1, 64, 5, 13)

        x = self.relu((self.decConv2(x)))
        x = self.relu((self.decConv3(x)))
        x = self.relu((self.decConv4(x)))     
        x = self.sigmoid((self.decConv5(x)))

        x = x.reshape(x.shape[0], -1)
        x = self.decFC2(x)

        x = x.reshape(x.shape[0], 3, self.img_dim, self.img_dim)

        return x