import torch
import torch.nn as nn
class L2_SP_Loss(nn.Module):
    def __init__(self, weight_ori):
        super().__init__()
        self.weight_ori = weight_ori
        self.criterion = nn.MSELoss()

    def forward(self, weight_cur):
        loss = 0
        for k_ori, v_cur in zip(self.weight_ori, weight_cur):
            loss += self.criterion(v_cur, self.weight_ori[k_ori])
        return loss
