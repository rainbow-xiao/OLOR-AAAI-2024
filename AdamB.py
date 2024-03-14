import math
import torch
from torch.optim import Optimizer

class AdamB(Optimizer):
    def __init__(
            self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, pretrained=True,
            back_level_max=1, back_level_min=0, back_pow=2):
        assert 0 <= back_level_max <= 1, "back_level_max should be in [0, 1]"
        defaults = dict(
            lr=lr, betas=betas, eps=eps, back_level_max=back_level_max, back_level_min=back_level_min, 
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
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                else:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            pretrained = group['pretrained']

            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            bias_correction1 = 1 - beta1 ** group['step']
            bias_correction2 = 1 - beta2 ** group['step']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # m_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # v_t
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                update = (exp_avg / bias_correction1).div_(denom)

                if pretrained:
                    D_value = state['D_value']
                    p_index, num_layers, back_level_max, back_level_min = state['p_index'], group['num_layers'], group['back_level_max'], group['back_level_min']
                    back_level = back_level_min+((1 - (p_index / num_layers + group['eps']))**group['back_pow']) * (back_level_max-back_level_min)
                    update.mul_((1-group['lr']*back_level))
                    update.add_(D_value, alpha=back_level)
                    p.add_(update, alpha=-group['lr'])
                    D_value.add_(update, alpha=-group['lr'])
                else:
                    p.add_(update, alpha=-group['lr'])
