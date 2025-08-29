from model import MODEL

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ._base_model import BaseModel
from .inp_former_utils import vit_encoder
from .inp_former_utils.vision_transformer import Mlp, Aggregation_Block, Prototype_Block
from .inp_former_utils.uad import INP_Former
from functools import partial
from torch.nn.init import trunc_normal_

class INP_Former_Model(BaseModel):
    def __init__(self, encoder_arch, INP_num):
        super(INP_Former_Model, self).__init__()

        assert encoder_arch in ['dinov2reg_vit_base_14', 'dinov2reg_vit_small_14', 'dinov2reg_vit_large_14']

        # Adopting a grouping-based reconstruction strategy similar to Dinomaly
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
        #target_layers = [2, 3, 4, 5, 6, 7, 8, 9,10,11]
        fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        #fuse_layer_decoder = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        fuse_layer_decoder = [[0,1,2,3],[4,5,6,7]]
        # Encoder info
        self.encoder = vit_encoder.load(encoder_arch)
        if 'small' in encoder_arch:
            embed_dim, num_heads = 384, 6
        elif 'base' in encoder_arch:
            embed_dim, num_heads = 768, 12
        elif 'large' in encoder_arch:
            embed_dim, num_heads = 1024, 16
            target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
        else:
            raise "Architecture not in small, base, large."

        # Model Preparation
        Bottleneck = []
        INP_Guided_Decoder = []
        INP_Extractor = []

        # bottleneck
        Bottleneck.append(Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.))
        self.Bottleneck = nn.ModuleList(Bottleneck)

        # INP
        self.INP = nn.ParameterList(
                        [nn.Parameter(torch.randn(INP_num, embed_dim))
                         for _ in range(1)])

        # # INP Extractor
        # for i in range(1):
        #     blk = Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
        #                             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        #     INP_Extractor.append(blk)
        # self.INP_Extractor = nn.ModuleList(INP_Extractor)

        # # INP_Guided_Decoder
        # for i in range(8):
        #     blk = Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
        #                           qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        #     INP_Guided_Decoder.append(blk)
        # self.INP_Guided_Decoder = nn.ModuleList(INP_Guided_Decoder)

        #加深聚合与解码堆叠，并加入轻度正则（dropout/drop_path）
        # INP Extractor
        for i in range(1):
            blk = Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                                    qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
                                    # drop = 0.1,drop_path = 0.1)
            INP_Extractor.append(blk)
        self.INP_Extractor = nn.ModuleList(INP_Extractor)

        # INP_Guided_Decoder
        for i in range(8):
            blk = Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                                  qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
                                #   drop = 0.1,drop_path = 0.1)
            INP_Guided_Decoder.append(blk)
        self.INP_Guided_Decoder = nn.ModuleList(INP_Guided_Decoder)

        self.model = INP_Former(encoder=self.encoder, bottleneck=self.Bottleneck, aggregation=self.INP_Extractor, decoder=self.INP_Guided_Decoder,
                                 target_layers=target_layers,  remove_class_token=True, fuse_layer_encoder=fuse_layer_encoder,
                                 fuse_layer_decoder=fuse_layer_decoder, prototype_token=self.INP)

        self.set_frozen_layers(["encoder"])

        trainable = nn.ModuleList([self.Bottleneck, self.INP_Guided_Decoder, self.INP_Extractor, self.INP])
        for m in trainable.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, img):
        en, de, g_loss = self.model(img)

        return en, de, g_loss

@MODEL.register_module
def inpformer(pretrained=False, **kwargs):
    model = INP_Former_Model(**kwargs)
    return model
