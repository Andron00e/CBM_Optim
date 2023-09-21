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
    r"""Class which implements a custom version of SGD optimizer.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float): SGD momentum factor
        dampening (float):  dampening for momentum
        weight_decay (float): weight decay (L2 penalty)
        nesterov (bool): enables Nesterov momentum
        is_clipped (bool): enables gradient clipping according to 2-norm
        clipping_level (float): indicates the max_norm of clipping
            (more information about clipping)
            we use a simple clipping rule
            for x, \lambda: clip(x, \lambda) = min{1, \lambda / ||x||} * x
            then the gradient updates look like:
            x^{k+1} = x^{k} - lr * clip(gradient, \lambda) 
            where \lambda is clipping_level
    Example:
        >>> optimizer = HeavyTailedSGD(model.parameters(), lr=0.001, 
                                       momentum=0, is_clipped=False, 
                                       clipping_level=1.0)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, is_clipped=False, 
                 clipping_level=1.0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        self.is_clipped = is_clipped
        self.clipping_level = clipping_level
        
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, 
                        is_clipped=is_clipped, clipping_level=clipping_level)
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
            is_clipped = self.is_clipped
            clipping_level = self.clipping_level

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                noise = self.generate_noise(distr=HeavyTailedNoise())
                d_p.add_(noise, p.data)

                if is_clipped:
                    torch.nn.utils.clip_grad_norm_(d_p, clipping_level)

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
    r"""Class which implements a custom version of AdaGrad optimizer.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float): SGD momentum factor
        dampening (float):  dampening for momentum
        weight_decay (float): weight decay (L2 penalty)
        nesterov (bool): enables Nesterov momentum
        is_clipped (bool): enables gradient clipping according to 2-norm
        clipping_level (float): indicates the max_norm of clipping
        b_0 (float): constant parameter in AdaGrad when performing denominator
            update
    Example:
        >>> optimizer = HeavyTailedAdagrad(model.parameters(), lr=0.001, 
                                           momentum=0, is_clipped=False, 
                                           clipping_level=1.0, b_0=1.0)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """
    def __init__(self, params, lr=required, 
                 lr_decay=0, weight_decay=0, 
                 initial_accumulator_value=0, 
                 is_clipped=False, 
                 clipping_level=1.0, b_0=1.0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError("Invalid initial_accumulator_value value: {}".format(initial_accumulator_value))
        self.is_clipped = is_clipped
        self.clipping_level = clipping_level
        self.b_0 = b_0
        
        defaults = dict(lr=lr, lr_decay=lr_decay, weight_decay=weight_decay, 
                        initial_accumulator_value=initial_accumulator_value, 
                        is_clipped=is_clipped, clipping_level=clipping_level,
                        b_0=b_0)
        super(HeavyTailedAdagrad, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['sum'] = torch.full_like(p.data, initial_accumulator_value)
                is_clipped = group.get('is_clipped', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            is_clipped = self.is_clipped
            clipping_level = self.clipping_level
            b_0 = self.b_0
            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                state = self.state[p]

                state['step'] += 1

                if group['weight_decay'] != 0:
                    d_p = d_p.add_(group['weight_decay'], p.data)

                clr = group['lr'] / (b_0 + (state['step'] - 1) * group['lr_decay'])

                noise = self.generate_noise(distr=HeavyTailedNoise())
                d_p = d_p.add_(noise, p.data)

                if is_clipped:
                    torch.nn.utils.clip_grad_norm_(d_p, clipping_level)

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