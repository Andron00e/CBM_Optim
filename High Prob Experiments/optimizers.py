import torch
import scipy.stats as sps
from torch.optim import Optimizer
from typing import List, Optional


class HeavyTailedNoise(sps.rv_continuous):
    """The noise we want to generate."""
    def _pdf(self, x):
        return 1 / (1 + abs(x)) ** 3
    
class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"
    
required = _RequiredParameter()

class HeavyTailedSGD(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr <0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        super(HeavyTailedSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(HeavyTailedSGD, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                noise = self.generate_noise(distr=HeavyTailedNoise())
                d_p.add_(noise, p.data)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                    else:
                        buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(1 - dampening, d_p)

                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss
    
    def generate_noise(self, distr):
        noise = None
        while (noise is None):
            try:
                noise = distr.rvs(size=1)[0]
            except Exception:
                pass
        return noise

                
class HeavyTailedAdagrad(Optimizer):
    def __init__(self, params, lr=required, lr_decay=0, weight_decay=0, initial_accumulator_value=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError("Invalid initial_accumulator_value value: {}".format(initial_accumulator_value))
        
        defaults = dict(lr=lr, lr_decay=lr_decay, weight_decay=weight_decay, initial_accumulator_value=initial_accumulator_value)
        super(HeavyTailedAdagrad, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['sum'] = torch.full_like(p.data, initial_accumulator_value)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                state = self.state[p]

                state['step'] += 1

                if group['weight_decay'] != 0:
                    d_p = d_p.add_(group['weight_decay'], p.data)

                clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])

                noise = self.generate_noise(distr=HeavyTailedNoise())
                d_p = d_p.add_(noise, p.data)

                state['sum'].addcmul_(1, d_p, d_p)
                std = state['sum'].sqrt().add_(1e-10)
                p.data.addcdiv_(-clr, d_p, std)

        return loss
    
    def generate_noise(self, distr):
        noise = None
        while (noise is None):
            try:
                noise = distr.rvs(size=1)[0]
            except Exception:
                pass
        return noise