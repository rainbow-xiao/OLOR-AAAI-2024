import math
import torch
from torch.optim import Optimizer

class SGDB(Optimizer):
    def __init__(
            self, params, lr=1e-3, beta=0.0, eps=1e-8, pretrained=True,
            back_level_max=1, back_level_min=0, back_pow=2):
        assert 0<= back_level_max <= 1, "back_level_max should be in [0, 1]"
        defaults = dict(
            lr=lr, beta=beta, eps=eps, back_level_max=back_level_max, back_level_min=back_level_min, 
            pretrained=pretrained, back_pow=back_pow)
        super().__init__(params, defaults)
        self.init_all()
        
    @torch.no_grad()
    def init_all(self):
        for group in self.param_groups:
            group['num_layers'] = -1
            for p in group['params']:
                state = self.state[p]
                if group['pretrained']:
                    group['num_layers'] += 1
                    state['D_value'] = torch.zeros_like(p)
                    state['p_index'] = group['num_layers']
                    state['momentum'] = torch.zeros_like(p)
                else:
                    state['momentum'] = torch.zeros_like(p)
                    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta = group['beta']
            pretrained = group['pretrained']

            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                momentum = state['momentum']
                momentum.mul_(beta).add_(grad, alpha=(1-beta))  # m_t
                update = momentum * group['lr']
                if pretrained:
                    D_value = state['D_value']
                    p_index, num_layers, back_level_max, back_level_min = state['p_index'], group['num_layers'], group['back_level_max'], group['back_level_min']
                    back_level = back_level_min+((1 - (p_index / num_layers + group['eps']))**group['back_pow']) * (back_level_max-back_level_min)
                    update.add_(D_value, alpha=back_level)
                    p.add_(update, alpha=-1)
                    D_value.add_(update, alpha=-1)
                    
                else:
                    p.add_(update, alpha=-1)
        return 