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

                # Update H matrix (Hessian approximation).
                # Sum_i g_i g_i^T over all rows == G^T G; the vectorized matmul
                # is mathematically identical to the per-row loop but avoids
                # millions of tiny GPU ops per step (the loop made Shampoo hang
                # on large conv/linear layers).
                # Recompute the (expensive) inverse only when H is updated, and
                # cache it between updates. With update_freq=1 this is identical
                # to inverting every step; with update_freq>1 it amortizes the
                # O(d^3) inverse (standard Shampoo "preconditioner frequency"),
                # which is essential for large matrices (e.g. BERT FFN 3072^2).
                if state['step'] % group['update_freq'] == 0:
                    H.add_(grad_flat.t() @ grad_flat)
                if 'H_inv' not in state or state['step'] % group['update_freq'] == 0:
                    # linalg.inv needs fp32+ (bf16 unsupported); compute in float,
                    # cast back to the parameter dtype.
                    Hf = (H + eps * torch.eye(H.size(0), device=H.device, dtype=H.dtype)).float()
                    try:
                        state['H_inv'] = torch.linalg.inv(Hf).to(H.dtype)
                    except Exception:
                        state['H_inv'] = torch.linalg.pinv(Hf).to(H.dtype)
                H_inv = state['H_inv']

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
