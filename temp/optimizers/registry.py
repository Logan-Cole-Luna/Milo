"""
Unified optimizer registry.

Every optimizer is constructed through build_optimizer(name, model, lr, wd, **kw),
which returns (optimizer, meta). `meta` tells the training loop how to drive it:

    meta = {
      "use_schedule":   bool,   # apply warmup+cosine LR schedule
      "sf_train_eval":  bool,   # schedule-free: call opt.train()/opt.eval()
      "hessian_every":  int|None,  # Sophia: Gauss-Newton-Bartlett interval
      "step_kwargs":    dict,   # extra kwargs for opt.step()
    }

Parameter routing conventions (standard in the Muon/SOAP literature):
  * "hidden" = ndim>=2 weights excluding embeddings & output head
  * matrix-preconditioned optimizers (muon, mion) route embeddings/head/1D
    params to their auxiliary path
  * decay/no-decay split (no WD on biases, norms, embeddings) for Adam-family

Vendored single-file optimizers live in optimizers/vendored/ (see
scripts/vendor_optimizers.sh, which pins exact upstream files). pip-installed
ones are imported lazily; missing ones raise a clear error naming the package.
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "vendored"))
sys.path.insert(0, str(Path(__file__).parent))

from milo2 import MiloM, Mion  # noqa: E402

META_DEFAULT = dict(use_schedule=True, sf_train_eval=False,
                    hessian_every=None, step_kwargs={})


# --------------------------------------------------------------- routing
def split_decay(model):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    return decay, no_decay


def split_muon_style(model):
    """hidden 2D+ matrices vs (embeddings, head, scalars/vectors)."""
    hidden, aux = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_embed = any(k in n.lower() for k in ("embed", "wte", "wpe", "tok_emb", "pos_emb"))
        is_head = any(k in n.lower() for k in ("head", "lm_head", "classifier", "fc.weight")) \
            and p.ndim >= 2
        if p.ndim >= 2 and not is_embed and not is_head:
            hidden.append(p)
        else:
            aux.append(p)
    return hidden, aux


# --------------------------------------------------------------- builders
def build_optimizer(name, model, lr, weight_decay=0.0, **kw):
    name = name.lower()
    meta = dict(META_DEFAULT)
    decay, no_decay = split_decay(model)
    grouped = [{"params": decay, "weight_decay": weight_decay},
               {"params": no_decay, "weight_decay": 0.0}]
    allp = [p for p in model.parameters() if p.requires_grad]

    # ---------------- torch built-ins ----------------
    if name == "sgdm":
        return torch.optim.SGD(grouped, lr=lr, momentum=kw.get("momentum", 0.9),
                               nesterov=True), meta
    if name == "adamw":
        return torch.optim.AdamW(grouped, lr=lr,
                                 betas=kw.get("betas", (0.9, 0.95)),
                                 eps=kw.get("eps", 1e-8)), meta
    if name == "nadamw":
        return torch.optim.NAdam(grouped, lr=lr, betas=kw.get("betas", (0.9, 0.95)),
                                 decoupled_weight_decay=True), meta
    if name == "adafactor":
        return torch.optim.Adafactor(grouped, lr=lr), meta

    # ---------------- pip packages ----------------
    if name == "lion":
        from lion_pytorch import Lion  # pip install lion-pytorch
        return Lion(grouped, lr=lr, betas=kw.get("betas", (0.9, 0.99))), meta
    if name == "schedulefree":
        import schedulefree  # pip install schedulefree
        meta.update(use_schedule=False, sf_train_eval=True)
        return schedulefree.AdamWScheduleFree(
            grouped, lr=lr, warmup_steps=kw.get("warmup_steps", 1000),
            betas=kw.get("betas", (0.9, 0.95))), meta
    if name == "prodigy":
        from prodigyopt import Prodigy  # pip install prodigyopt
        # lr is a multiplier here; 1.0 is the parameter-free setting
        return Prodigy(grouped, lr=lr, weight_decay=weight_decay,
                       decouple=True, safeguard_warmup=True), meta
    if name == "kron":
        from kron_torch import Kron  # pip install kron-torch  (PSGD-Kron)
        return Kron(allp, lr=lr, weight_decay=weight_decay), meta
    if name == "shampoo":
        # pip install git+https://github.com/facebookresearch/optimizers.git
        from distributed_shampoo import AdamGraftingConfig, DistributedShampoo
        return DistributedShampoo(
            allp, lr=lr, betas=(0.9, 0.999), epsilon=1e-12,
            weight_decay=weight_decay, use_decoupled_weight_decay=True,
            max_preconditioner_dim=kw.get("max_precond_dim", 2048),
            precondition_frequency=kw.get("precond_freq", 25),
            grafting_config=AdamGraftingConfig(beta2=0.999, epsilon=1e-8)), meta

    # ---------------- vendored ----------------
    if name == "soap":
        from soap import SOAP  # vendored nikhilvyas/SOAP
        return SOAP(allp, lr=lr, betas=(0.95, 0.95), weight_decay=weight_decay,
                    precondition_frequency=kw.get("precond_freq", 10)), meta
    if name == "sophia":
        from sophia import SophiaG  # vendored Liuhong99/Sophia
        meta.update(hessian_every=kw.get("hessian_every", 10),
                    step_kwargs={"bs": kw["sophia_bs"]})  # tokens per step
        return SophiaG(grouped, lr=lr, betas=(0.965, 0.99),
                       rho=kw.get("rho", 0.05)), meta
    if name == "muon":
        from muon import SingleDeviceMuonWithAuxAdam  # vendored KellerJordan/Muon
        hidden, aux = split_muon_style(model)
        groups = [dict(params=hidden, lr=lr, momentum=0.95,
                       weight_decay=weight_decay, use_muon=True),
                  dict(params=aux, lr=kw.get("aux_lr", lr * kw.get("aux_lr_mult", 0.15)),
                       betas=(0.9, 0.95), eps=1e-10,
                       weight_decay=weight_decay, use_muon=False)]
        return SingleDeviceMuonWithAuxAdam(groups), meta
    if name == "ademamix":
        from ademamix import AdEMAMix  # local implementation (optimizers/ademamix.py)
        return AdEMAMix(grouped, lr=lr, betas=kw.get("betas", (0.9, 0.999, 0.9999)),
                        alpha=kw.get("alpha", 5.0)), meta

    # ---------------- ours ----------------
    if name == "milo":
        from milo import milo as MiloOrig  # original implementation (drop in optimizers/milo.py)
        return MiloOrig(allp, lr=lr, momentum=0.9, weight_decay=weight_decay,
                        normalize=True, adaptive=True, use_cuda_kernels=False), meta
    if name == "milo_m":
        return MiloM(allp, lr=lr, momentum=kw.get("momentum", 0.95),
                     weight_decay=weight_decay,
                     scale_factor=kw.get("scale_factor", 0.2),
                     rms_target=kw.get("rms_target", 0.2),
                     group_mode=kw.get("group_mode", "row")), meta
    if name == "mion":
        hidden, aux = split_muon_style(model)
        return Mion([{"params": hidden, "spectral": True},
                     {"params": aux, "spectral": False}],
                    lr=lr, momentum=kw.get("momentum", 0.95),
                    weight_decay=weight_decay,
                    scale_factor=kw.get("scale_factor", 0.0),
                    rms_target=kw.get("rms_target", 0.2),
                    ns_steps=kw.get("ns_steps", 5)), meta

    raise ValueError(f"unknown optimizer: {name}")


ALL_OPTIMIZERS = ["sgdm", "adamw", "nadamw", "adafactor", "lion", "schedulefree",
                  "prodigy", "sophia", "muon", "soap", "shampoo", "kron",
                  "ademamix", "milo", "milo_m", "mion"]


def parse_opt_kwargs(s):
    """--opt-kwargs '{"ns_steps": 3}' on any task CLI."""
    return json.loads(s) if s else {}
