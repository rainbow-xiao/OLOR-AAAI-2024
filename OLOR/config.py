import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

_C.MODEL = CN()
_C.MODEL.ft_mode = 'linear'
_C.MODEL.type = 'vit'
_C.MODEL.img_size = 224
_C.MODEL.num_classes = 7
_C.MODEL.finetune = None


_C.MODEL.output_dir = '/root/autodl-tmp'
_C.MODEL.backbone = CN()

# convnext_base.clip_laion2b_augreg_ft_in12k_in1k out_dim = 1024
# vit_base_patch16_clip_224.laion2b_ft_in12k_in1k out_dim = 768
# vit_base_patch16_224.augreg_in21k out_dim = 768
# vit_base_patch16_clip_224.laion2b out_dim = 768
# vit_base_patch16_224.mae out_dim = 768

_C.MODEL.backbone.model_name = 'vit_base_patch16_clip_224.laion2b_ft_in12k_in1k'
_C.MODEL.backbone.out_dim = 768

_C.MODEL.backbone.VPT = CN()
_C.MODEL.backbone.VPT.p_num = 10

#Optimizer
_C.Optimizer = CN()
_C.Optimizer.name = 'Adam'
_C.Optimizer.back_level_max = 0.01
_C.Optimizer.back_level_min = 0
_C.Optimizer.back_pow = 1
def get_config():
    config = _C.clone()
    return config