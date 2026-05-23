"""
PyTorch implementation of AdEMAMix optimizer.
Based on the paper: "The AdEMAMix Optimizer: Better, Faster, Older"
"""
import torch
from torch.optim.optimizer import Optimizer
import math


class AdEMAMix(Optimizer):
    """
    AdEMAMix optimizer implementation in PyTorch.
    
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999, 0.9999))
        alpha (float, optional): mixing coefficient for fast and slow EMAs (default: 8.0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 0.0)
        beta3_warmup (int, optional): warmup steps for beta3 scheduler (default: 0)
        alpha_warmup (int, optional): warmup steps for alpha scheduler (default: 0)
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.9999), alpha=8.0, 
                 eps=1e-8, weight_decay=0.0, beta3_warmup=0, alpha_warmup=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 2: {betas[2]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha value: {alpha}")
            
        defaults = dict(lr=lr, betas=betas, alpha=alpha, eps=eps, 
                       weight_decay=weight_decay, beta3_warmup=beta3_warmup, 
                       alpha_warmup=alpha_warmup)
        super(AdEMAMix, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(AdEMAMix, self).__setstate__(state)
    
    def _beta3_scheduler(self, step, beta3_end, beta1, warmup):
        """Beta3 scheduler with warmup as described in the paper."""
        if step < warmup:
            # Interpolate in transformed space
            def f(beta):
                return math.log(0.5) / math.log(beta) - 1
            
            def f_inv(t):
                return pow(0.5, 1 / (t + 1))
            
            alpha_sched = step / float(warmup)
            return f_inv((1.0 - alpha_sched) * f(beta1) + alpha_sched * f(beta3_end))
        else:
            return beta3_end
    
    def _alpha_scheduler(self, step, alpha_end, alpha_start, warmup):
        """Alpha scheduler with warmup."""
        if step < warmup:
            alpha_sched = step / float(warmup)
            return (1.0 - alpha_sched) * alpha_start + alpha_sched * alpha_end
        else:
            return alpha_end

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

        for group in self.param_groups:
            beta1, beta2, beta3_end = group['betas']
            alpha_end = group['alpha']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m1'] = torch.zeros_like(p, memory_format=torch.preserve_format)  # fast EMA
                    state['m2'] = torch.zeros_like(p, memory_format=torch.preserve_format)  # slow EMA
                    state['nu'] = torch.zeros_like(p, memory_format=torch.preserve_format)  # second moment
                
                m1, m2, nu = state['m1'], state['m2'], state['nu']
                
                state['step'] += 1
                step = state['step']
                
                # Get current beta3 and alpha values (with scheduling)
                current_beta3 = self._beta3_scheduler(step, beta3_end, beta1, group['beta3_warmup'])
                current_alpha = self._alpha_scheduler(step, alpha_end, 0.0, group['alpha_warmup'])
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Update biased first moment estimate (fast EMA)
                m1.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update biased slow moment estimate
                m2.mul_(current_beta3).add_(grad, alpha=1 - current_beta3)
                
                # Update biased second raw moment estimate
                nu.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Compute bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                # Bias corrected first moment
                m1_hat = m1.div(bias_correction1)
                
                # Bias corrected second moment
                nu_hat = nu.div(bias_correction2)
                
                # Combine fast and slow EMAs
                combined_m = m1_hat.add(m2, alpha=current_alpha)
                
                # Compute denominator
                denom = nu_hat.sqrt().add_(group['eps'])
                
                # Apply update
                step_size = group['lr']
                p.addcdiv_(combined_m, denom, value=-step_size)
        
        return loss
