"""
Milo v2: two upgraded variants of the Milo normalized-SGD optimizer.

MiloM ("Milo, momentum-first")
------------------------------
Minimal-surgery upgrade of the original optimizer:
  1. Nesterov momentum is applied FIRST; group standardization operates on
     the smoothed update direction, not the raw minibatch gradient.
  2. Groups are aligned to model structure: per output row for ndim>=2
     tensors (per-neuron for Linear, per-channel for Conv), sqrt(N) groups
     for vectors (unchanged from original Milo).
  3. The AdaGrad accumulator is removed: group standardization already
     fixes the update scale, and a growing sum_sq imposes an implicit
     1/sqrt(t) LR decay that fights the normalization.
  4. Weight decay is decoupled (AdamW style). Coupled decay added before
     standardization is largely cancelled by the normalization.
  5. scale_aware blending is kept, but blends with the RMS-normalized raw
     direction so total update RMS stays ~rms_target (grafting-style).

Mion ("Milo + Muon")
--------------------
A single self-contained optimizer:
  * ndim>=2 "hidden" weights: Newton-Schulz orthogonalization of the
    Nesterov momentum buffer (Muon update). Conv kernels are flattened to
    (out_channels, -1).
  * everything else (vectors; embeddings/lm_head if routed via a param
    group with spectral=False): MiloM group standardization.
  Both paths are rescaled to a common update RMS (rms_target, default 0.2,
  matching typical Adam update RMS), so ONE learning rate works for all
  parameters. This removes Muon's need for an auxiliary AdamW instance --
  Milo's normalization is the fallback.

  Two optional knobs (both default to plain MION's behavior):
  * ortho_strength in [0, 1]: gates the spectral path between full
    Newton-Schulz orthogonalization (1.0, default) and row group-standardization
    (0.0, MiloM-like on that path) -- lets a single optimizer interpolate
    between "MION" and "MiloM" on hidden matrices. Motivated by orthogonalization
    helping from-scratch pretraining but hurting fine-tuning (MION-Gated/MION-FT).
  * row_norm: NorMuon-style per-row second-moment normalization applied after
    the update is formed (on either path), equalizing per-neuron/per-token
    update norms instead of leaving a few rows dominant post-orthogonalization
    (MION-Nor). Applies uniformly across param groups, including the
    non-spectral embedding group.

Usage:
    opt = MiloM(model.parameters(), lr=0.02, momentum=0.95, weight_decay=0.01)

    # Mion: route embeddings / output head away from the spectral path.
    hidden  = [p for n, p in model.named_parameters()
               if p.ndim >= 2 and "embed" not in n and "lm_head" not in n]
    other   = [p for n, p in model.named_parameters() if p not in set(hidden)]
    opt = Mion([{"params": hidden, "spectral": True},
                {"params": other,  "spectral": False}],
               lr=0.02, momentum=0.95, weight_decay=0.01)
"""

import math
from typing import Optional

import torch
from torch.optim.optimizer import Optimizer


# --------------------------------------------------------------------------
# Newton-Schulz quintic iteration (Jordan et al., Muon). Computes an
# approximate semi-orthogonalization of G: replaces singular values with ~1.
# --------------------------------------------------------------------------
@torch.no_grad()
def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16) if G.is_cuda else G.to(torch.float32)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


def _rms(t: torch.Tensor, eps: float) -> torch.Tensor:
    return t.norm() / math.sqrt(t.numel()) + eps


# --------------------------------------------------------------------------
# NorMuon-style per-row (per-neuron / per-token) second-moment normalization,
# applied AFTER the main update direction is formed. Equalizes row norms
# (Newton-Schulz orthogonalization leaves some neurons dominant) while
# preserving the pre-normalization global RMS, so the caller's single-LR
# rescale is unaffected.
# --------------------------------------------------------------------------
@torch.no_grad()
def row_second_moment_normalize(
    u: torch.Tensor, state: dict, beta2: float = 0.999, eps: float = 1e-8,
) -> torch.Tensor:
    if u.ndim < 2 or u.shape[0] < 2:
        return u
    mat = u.reshape(u.shape[0], -1)
    target_rms = mat.pow(2).mean().sqrt() + eps
    row_ms = mat.pow(2).mean(dim=1)
    v = state.get("row_v")
    if v is None or v.shape != row_ms.shape:
        v = state["row_v"] = row_ms.clone()
    else:
        v.mul_(beta2).add_(row_ms, alpha=1 - beta2)
    row_rms = v.sqrt() + eps
    out = mat / row_rms.unsqueeze(1)
    out = out * (target_rms / (out.pow(2).mean().sqrt() + eps))
    return out.view_as(u)


# --------------------------------------------------------------------------
# Shared: structure-aligned group standardization of an update direction.
# Returns a tensor with RMS ~= 1 (before blending).
# --------------------------------------------------------------------------
@torch.no_grad()
def group_standardize(
    d: torch.Tensor,
    eps: float = 1e-8,
    scale_factor: float = 0.2,
    group_size: Optional[int] = None,
) -> torch.Tensor:
    if d.numel() < 2:
        return d / _rms(d, eps)

    if d.ndim >= 2 and group_size is None:
        # one group per output row (neuron / channel)
        mat = d.reshape(d.shape[0], -1)
        if mat.shape[1] >= 2:
            mean = mat.mean(dim=1, keepdim=True)
            std = mat.std(dim=1, keepdim=True) + eps
            standardized = ((mat - mean) / std).view_as(d)
        else:
            standardized = d / _rms(d, eps)
    else:
        flat = d.reshape(-1)
        N = flat.numel()
        gs = group_size or max(2, math.ceil(N / max(1, int(math.sqrt(N)))))
        rem = N % gs
        if rem:
            flat = torch.cat([flat, flat.new_zeros(gs - rem)])
        resh = flat.view(-1, gs)
        mean = resh.mean(dim=1, keepdim=True)
        std = resh.std(dim=1, keepdim=True) + eps
        standardized = ((resh - mean) / std).reshape(-1)[:N].view_as(d)

    if scale_factor > 0:
        # grafting-style blend: direction info from the raw update, but
        # RMS-normalized so the result keeps unit scale.
        standardized = scale_factor * (d / _rms(d, eps)) + (1.0 - scale_factor) * standardized
    return standardized


class MiloM(Optimizer):
    """Momentum-first, structure-aligned Milo. Drop-in replacement."""

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        scale_factor: float = 0.2,
        rms_target: float = 0.2,
        group_size: Optional[int] = None,
        group_mode: str = "row",   # "row" (structure-aligned) | "flat" (original sqrt-N) -- ablation knob
    ):
        if group_mode not in ("row", "flat"):
            raise ValueError("group_mode must be 'row' or 'flat'")
        if lr < 0 or momentum < 0 or weight_decay < 0 or eps < 0:
            raise ValueError("negative hyperparameter")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        weight_decay=weight_decay, eps=eps,
                        scale_factor=scale_factor, rms_target=rms_target,
                        group_size=group_size, group_mode=group_mode)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mu = group["lr"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = state["momentum_buffer"] = torch.zeros_like(g)
                buf.mul_(mu).add_(g)
                d = g.add(buf, alpha=mu) if group["nesterov"] else buf

                if group["group_mode"] == "flat" and d.ndim >= 2:
                    # ablation: original Milo's structure-blind sqrt-N grouping
                    u = group_standardize(d.reshape(1, -1), eps=group["eps"],
                                          scale_factor=group["scale_factor"],
                                          group_size=group["group_size"]
                                          or max(2, math.ceil(d.numel() / max(1, int(math.sqrt(d.numel())))))
                                          ).view_as(d)
                else:
                    u = group_standardize(d, eps=group["eps"],
                                          scale_factor=group["scale_factor"],
                                          group_size=group["group_size"])
                if group["weight_decay"] != 0:
                    p.mul_(1.0 - lr * group["weight_decay"])
                p.add_(u, alpha=-lr * group["rms_target"])
        return loss


class Mion(Optimizer):
    """Newton-Schulz spectral update for 2D hidden weights, Milo group
    standardization for everything else, unified to a common update RMS."""

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        scale_factor: float = 0.0,   # grafting blend on the group path
        rms_target: float = 0.2,
        ns_steps: int = 5,
        spectral: bool = True,       # per param-group routing flag
        momentum_mode: str = "heavy_ball",  # "heavy_ball" (orig) | "ema" (Muon-style lerp)
        aux_mode: str = "groupstd",  # non-matrix path: "groupstd" | "adam"
        aux_lr: float = None,        # adam aux LR; None => unified (lr * rms_target)
        aux_betas: tuple = (0.9, 0.95),
        aux_eps: float = 1e-8,
        ortho_strength: float = 1.0,  # spectral-path gate: 1.0=full NS (MION), 0.0=row-groupstd (MiloM-like)
        row_norm: bool = False,       # NorMuon-style per-row 2nd-moment norm, applied post-update on any path
        row_beta2: float = 0.999,
        row_eps: float = 1e-8,
    ):
        if aux_mode not in ("groupstd", "adam"):
            raise ValueError("aux_mode must be 'groupstd' or 'adam'")
        if momentum_mode not in ("heavy_ball", "ema"):
            raise ValueError("momentum_mode must be 'heavy_ball' or 'ema'")
        if not (0.0 <= ortho_strength <= 1.0):
            raise ValueError("ortho_strength must be in [0, 1]")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        weight_decay=weight_decay, eps=eps,
                        scale_factor=scale_factor, rms_target=rms_target,
                        ns_steps=ns_steps, spectral=spectral, momentum_mode=momentum_mode,
                        aux_mode=aux_mode, aux_lr=aux_lr,  # None => unified single-LR
                        aux_betas=aux_betas, aux_eps=aux_eps,
                        ortho_strength=ortho_strength, row_norm=row_norm,
                        row_beta2=row_beta2, row_eps=row_eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mu = group["lr"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                use_spec = (group["spectral"] and p.ndim >= 2
                            and min(p.shape[0], p[0].numel()) >= 2)

                # Non-matrix params with AdamW aux (Muon-style); separate aux_lr.
                # Used only for the controlled MION-vs-Muon ablation; the default
                # aux_mode="groupstd" leaves MION's single-LR behavior unchanged.
                if (not use_spec) and group["aux_mode"] == "adam":
                    # Per-coordinate adaptive aux. aux_lr=None => unified single LR
                    # (lr * rms_target), matching the matrix path's scale so one LR
                    # serves the whole model while embeddings get Adam adaptivity.
                    alr = group["aux_lr"] if group["aux_lr"] is not None else lr * group["rms_target"]
                    b1, b2 = group["aux_betas"]; aeps = group["aux_eps"]
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(g)
                        state["exp_avg_sq"] = torch.zeros_like(g)
                        state["adam_step"] = 0
                    state["adam_step"] += 1; t = state["adam_step"]
                    m_, v_ = state["exp_avg"], state["exp_avg_sq"]
                    m_.mul_(b1).add_(g, alpha=1 - b1)
                    v_.mul_(b2).addcmul_(g, g, value=1 - b2)
                    mhat = m_ / (1 - b1 ** t); vhat = v_ / (1 - b2 ** t)
                    if group["weight_decay"] != 0:
                        p.mul_(1.0 - alr * group["weight_decay"])
                    p.addcdiv_(mhat, vhat.sqrt().add_(aeps), value=-alr)
                    continue

                # Spectral (matrix) or group-std (vector) path: shared momentum.
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = state["momentum_buffer"] = torch.zeros_like(g)
                if group["momentum_mode"] == "ema":
                    buf.lerp_(g, 1 - mu)                      # Muon-style EMA
                    d = g.lerp(buf, mu) if group["nesterov"] else buf
                else:
                    buf.mul_(mu).add_(g)                      # heavy-ball (original)
                    d = g.add(buf, alpha=mu) if group["nesterov"] else buf

                if use_spec:
                    mat = d.reshape(d.shape[0], -1)
                    beta = group["ortho_strength"]
                    u = 0
                    if beta > 0.0:
                        O = newton_schulz5(mat, steps=group["ns_steps"])
                        # semi-orthogonal (m,n) matrix has RMS = 1/sqrt(max(m,n));
                        # rescale to unit RMS so one lr serves both paths.
                        u = u + beta * (O * math.sqrt(max(mat.shape)))
                    if beta < 1.0:
                        u = u + (1.0 - beta) * group_standardize(
                            mat, eps=group["eps"], scale_factor=group["scale_factor"])
                    u = u.view_as(d)
                else:
                    u = group_standardize(d, eps=group["eps"],
                                          scale_factor=group["scale_factor"])
                if group["row_norm"]:
                    u = row_second_moment_normalize(u, state, group["row_beta2"], group["row_eps"])
                if group["weight_decay"] != 0:
                    p.mul_(1.0 - lr * group["weight_decay"])
                p.add_(u, alpha=-lr * group["rms_target"])
        return loss
