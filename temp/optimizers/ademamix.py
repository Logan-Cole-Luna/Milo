"""AdEMAMix (Pagliardini, Ablin, Grangier, 2024 -- arXiv:2409.03137).

AdamW with a mixture of two momentum EMAs: a fast one (beta1) and a very slow
one (beta3, e.g. 0.9999) weighted by alpha. Faithful single-file
implementation kept locally so the benchmark does not depend on upstream
repo layout. Includes the paper's warmup schedulers for beta3 and alpha.
"""

import math
import torch
from torch.optim.optimizer import Optimizer


def _linear_warmup(step, warmup, end_value, start_value=0.0):
    if warmup is None or step >= warmup:
        return end_value
    return start_value + (end_value - start_value) * step / warmup


def _beta3_warmup(step, warmup, beta_end, beta_start=0.9):
    if warmup is None or step >= warmup:
        return beta_end
    # interpolate in 1/log(beta) space as in the paper
    def f(beta): return math.log(0.5) / math.log(beta) - 1
    def f_inv(t): return math.pow(0.5, 1 / (t + 1))
    return f_inv(f(beta_start) + (f(beta_end) - f(beta_start)) * step / warmup)


class AdEMAMix(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.9999), alpha=5.0,
                 eps=1e-8, weight_decay=0.0, beta3_warmup=None, alpha_warmup=None):
        if not 0.0 <= lr:
            raise ValueError(f"invalid lr: {lr}")
        defaults = dict(lr=lr, betas=betas, alpha=alpha, eps=eps,
                        weight_decay=weight_decay,
                        beta3_warmup=beta3_warmup, alpha_warmup=alpha_warmup)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2, b3_final = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m1"] = torch.zeros_like(p)
                    state["m2"] = torch.zeros_like(p)
                    state["nu"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                b3 = _beta3_warmup(t, group["beta3_warmup"], b3_final)
                alpha = _linear_warmup(t, group["alpha_warmup"], group["alpha"])

                m1, m2, nu = state["m1"], state["m2"], state["nu"]
                m1.mul_(b1).add_(g, alpha=1 - b1)
                m2.mul_(b3).add_(g, alpha=1 - b3)
                nu.mul_(b2).addcmul_(g, g, value=1 - b2)

                bc1 = 1 - b1 ** t
                bc2 = 1 - b2 ** t
                denom = (nu.sqrt() / math.sqrt(bc2)).add_(group["eps"])
                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                update = (m1 / bc1 + alpha * m2) / denom
                p.add_(update, alpha=-group["lr"])
        return loss
