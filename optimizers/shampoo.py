"""
Shampoo: Preconditioned Stochastic Gradient Method

Paper: "Shampoo: Preconditioned Stochastic Gradient Method"

Second-order optimization using block-wise diagonal approximation of the Hessian.
Provides faster convergence especially for neural networks.
"""

import torch
from torch.optim.optimizer import Optimizer
import math


class Shampoo(Optimizer):
    r"""Implements Shampoo algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        eps (float, optional): regularization constant for numerical stability (default: 1e-10)
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        update_freq (int, optional): update frequency for inverse approximation (default: 1)
    """

    def __init__(self, params, lr=1e-3, eps=1e-10, momentum=0,
                 weight_decay=0, update_freq=1):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, eps=eps, momentum=momentum,
                       weight_decay=weight_decay, update_freq=update_freq)
        super(Shampoo, self).__init__(params, defaults)

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
                    raise RuntimeError('Shampoo does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['H'] = torch.eye(grad.size(-1), device=grad.device, dtype=grad.dtype)
                    if group['momentum'] > 0:
                        state['v'] = torch.zeros_like(p.data)

                state['step'] += 1
                H = state['H']
                lr = group['lr']
                eps = group['eps']

                # Reshape gradient for computation
                grad_flat = grad.view(-1, grad.size(-1)) if grad.dim() > 1 else grad.unsqueeze(0)

                # Update H matrix (Hessian approximation)
                if state['step'] % group['update_freq'] == 0:
                    for i in range(grad_flat.size(0)):
                        g = grad_flat[i].unsqueeze(-1)  # Column vector
                        H.add_(g @ g.t(), alpha=1.0)

                # Compute update with regularization
                try:
                    H_inv = torch.linalg.inv(H + eps * torch.eye(H.size(0), device=H.device, dtype=H.dtype))
                except:
                    # Fallback to pseudoinverse if singular
                    H_inv = torch.linalg.pinv(H)

                # Apply update
                if grad.dim() == 1:
                    update = H_inv @ grad
                else:
                    grad_flat = grad.view(-1, grad.size(-1))
                    update = grad @ H_inv
                    update = update.view_as(grad)

                # Weight decay
                if group['weight_decay'] != 0:
                    update = update + group['weight_decay'] * p.data

                # Momentum
                if group['momentum'] > 0:
                    v = state['v']
                    v.mul_(group['momentum']).add_(update, alpha=1)
                    p.data.add_(v, alpha=-lr)
                else:
                    p.data.add_(update, alpha=-lr)

        return loss
