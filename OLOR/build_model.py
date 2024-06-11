import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import timm
from functools import partial

class Model_Cls(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_type = config.MODEL.type
        self.ft_mode = config.MODEL.ft_mode
        self.backbone = timm.create_model(config.MODEL.backbone.model_name, pretrained=False, num_classes=0)
        if self.ft_mode in ['linear', 'vpt']:
            for param in self.backbone.parameters():
                param.requires_grad=False
        # if self.model_type == 'vit':
        #     self.backbone.set_ft_mode(self.ft_mode, p_num=config.MODEL.backbone.VPT.p_num)

        elif self.ft_mode == 'vpt' and self.model_type == 'conv':
            self.num_tokens = config.MODEL.backbone.VPT.p_num
            self.prompt_embeddings_tb = nn.Parameter(torch.zeros(
                    1, 3, 2 * self.num_tokens,
                    224 + 2 * self.num_tokens
            ))
            self.prompt_embeddings_lr = nn.Parameter(torch.zeros(
                    1, 3, 224,
                    2 * self.num_tokens
            ))
        self.head = nn.Linear(config.MODEL.backbone.out_dim, config.MODEL.num_classes)

    def forward(self, x):
        if self.model_type == 'conv':
            if self.ft_mode == 'vpt':
                prompt_emb_lr = self.prompt_embeddings_lr.expand(x.shape[0], -1, -1, -1)
                prompt_emb_tb = self.prompt_embeddings_tb.expand(x.shape[0], -1, -1, -1)
                x = torch.cat((
                prompt_emb_lr[:, :, :, :self.num_tokens],
                x, prompt_emb_lr[:, :, :, self.num_tokens:]
                ), dim=-1)
                x = torch.cat((
                    prompt_emb_tb[:, :, :self.num_tokens, :],
                    x, prompt_emb_tb[:, :, self.num_tokens:, :]
                ), dim=-2)
            x = self.backbone(x)
        elif self.model_type == 'vit':
            x = self.backbone(x)
        x = self.head(x)
        return x
