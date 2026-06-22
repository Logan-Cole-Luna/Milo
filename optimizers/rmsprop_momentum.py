"""
RMSprop with Momentum

Enhanced variant of RMSprop with built-in momentum for better convergence.
Combines adaptive learning rates with momentum acceleration.
"""

import torch
from torch.optim.optimizer import Optimizer


class RMSpropMomentum(Optimizer):
    r"""Implements RMSprop with Momentum.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        alpha (float, optional): smoothing constant (default: 0.99)
        momentum (float, optional): momentum factor (default: 0.9)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        centered (bool, optional): if True, gradients are normalized by the estimated variance (default: False)
    """

    def __init__(self, params, lr=1e-3, alpha=0.99, momentum=0.9, eps=1e-8,
                 weight_decay=0, centered=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, alpha=alpha, momentum=momentum, eps=eps,
                       weight_decay=weight_decay, centered=centered)
        super(RMSpropMomentum, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RMSpropMomentum does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['square_avg'] = torch.zeros_like(p_data_fp32)
                    state['momentum_buffer'] = torch.zeros_like(p_data_fp32)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p_data_fp32)

                square_avg = state['square_avg']
                momentum_buffer = state['momentum_buffer']
                alpha = group['alpha']

                # Decay the average of the squared gradient
                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
                    avg = square_avg - grad_avg.pow(2)
                    avg.clamp_(min=group['eps'])
                else:
                    avg = square_avg.add(group['eps'])

                # Add momentum
                momentum_buffer.mul_(group['momentum']).add_(grad / avg.sqrt(), alpha=1)

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(p_data_fp32, alpha=-group['weight_decay'] * group['lr'])

                p_data_fp32.add_(momentum_buffer, alpha=-group['lr'])
                p.data.copy_(p_data_fp32)

        return loss
