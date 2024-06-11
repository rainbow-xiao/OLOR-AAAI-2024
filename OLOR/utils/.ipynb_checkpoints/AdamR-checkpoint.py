import math

import torch
from torch.optim import Optimizer


class AdamR(Optimizer):
    def __init__(
            self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, enable_recover=True,
            recover_level=1, weight_decay=0.01, recover_pow=2):
        assert 0<= recover_level <= 1, "recover_level should be in [0, 1]"
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_recover_max=weight_recover_max, 
            weight_decay=weight_decay, enable_recover=enable_recover, recover_mode=recover_mode)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        device = self.param_groups[0]['params'][0].device
        one_tensor = torch.tensor(1.0, device=device)  # because torch.where doesn't handle scalars correctly
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            enable_recover = group['enable_recover']
            
            if 'step' in group:
                group['step'] += 1
            else:
                group['num_layers'] = 0
                group['step'] = 1

            bias_correction1 = 1 - beta1 ** group['step']
            bias_correction2 = 1 - beta2 ** group['step']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient valuesa
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    if enable_recover:
                        state['weight_dist'] = torch.zeros_like(p)
                        state['p_index'] = group['num_layers']
                        group['num_layers'] += 1              
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                if enable_recover:
                    weight_dist = state['weight_dist']
                    p_i, n_p, r_level = state['p_index'], group['num_layers']-1, group['recover_level']
                    weight_recover = (1 - (p_i / n_p)**group['recover_pow']) * r_level
                else:
                    weight_decay = group['weight_decay']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # m_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # v_t

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                update = (exp_avg / bias_correction1).div_(denom)
                
                if enable_recover:
                    update.mul_(group['lr'])
                    update.mul_(1-weight_recover)
                    update.add_(weight_dist, alpha=weight_recover)
                    weight_dist.add_(update, alpha=-1)
                    p.add_(update, alpha=-1)
                    
                elif weight_decay != 0:
                    update.add_(p, alpha=weight_decay)
                    p.add_(update, alpha=-group['lr'])
                
        return 