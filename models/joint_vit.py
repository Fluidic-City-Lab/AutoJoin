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

        # This is ViT-S settings
        # self.regressor = ViT(
        #             image_size=int(args.img_dim),
        #             patch_size=16,
        #             num_classes=1000,
        #             dim=384,
        #             depth=12,
        #             heads=6,
        #             mlp_dim=1536,
        #             dropout=0.1,
        #             emb_dropout=0.1
        #         )
        
        # self.regressor.load_state_dict(torch.load("deit_small.pth"))
        
        # This is SimpleViT with ViT-S settings
        # self.regressor = SimpleViT(
        #             image_size=int(args.img_dim),
        #             patch_size=16,
        #             num_classes=1000,
        #             dim=384,
        #             depth=12,
        #             heads=6,
        #             mlp_dim=1536,
        #         )        
        
        self.regressor = deit_small_patch16_224(pretrained=True)
        

    def forward(self, x):
        x = self.regressor(x)
        # print(x.shape)
        
        return x

class RegressorViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1000, 1)
        # self.fc2 = nn.Linear(64, 1)
        

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        sa = self.fc1(x)

        return sa

class DecoderViT(nn.Module):
    def __init__(self, args, in_dim=1000):
        super().__init__()

        self.img_dim = int(args.img_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.decFC1 = nn.Linear(in_dim, 256)
        self.decFC2 = nn.Linear(256, 512)
        self.decFC3 = nn.Linear(512, 1024)
        self.decFC4 = nn.Linear(1024, (3 * self.img_dim * self.img_dim))
    
    def forward(self, x):
        x = self.relu(self.decFC1(x))
        
        # x = x.view(-1, 64, 4, 4)

        x = self.relu((self.decFC2(x)))
        x = self.relu((self.decFC3(x)))
        x = self.sigmoid((self.decFC4(x)))    

        x = x.view(-1, 3, self.img_dim, self.img_dim) 
        # print(x.shape)

        return x