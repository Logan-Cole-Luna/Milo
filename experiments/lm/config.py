"""Model-size presets and optimizer construction for LM pretraining."""
from experiments.lm.model import GPTConfig

# Common benchmark scales (non-embedding params approximate).
MODEL_PRESETS = {
    "small":  GPTConfig(n_layer=12, n_head=12, n_embd=768,  block_size=1024),  # ~124M
    "medium": GPTConfig(n_layer=24, n_head=16, n_embd=1024, block_size=1024),  # ~350M
    "large":  GPTConfig(n_layer=24, n_head=16, n_embd=1536, block_size=1024),  # ~770M
}

# All optimizers under comparison (MILO family + modern baselines).
OPTIMIZERS = [
    "MILO", "MILO_LW", "MILOM", "MION",
    "MION_ADAM",  # MION + AdamW aux (separate aux LR) for embeddings/head
    "MION_V2",    # MION + EMA momentum + per-coord adaptive embeddings under ONE LR
    "MION_EMB",   # MION (group-std) + larger embedding LR via per-group multiplier
    "MION_NOR",   # MION + NorMuon-style per-row 2nd-moment norm (spectral + embed path)
    "MION_GATED", # MION with a tunable ortho_strength (beta) blend, for FT regime
    "ADAMW", "LION", "MUON", "SOAP", "SHAMPOO", "ADAM_MINI", "SGD",
]

# Per-optimizer kwargs (LR is set separately / tuned). Matrix-method optimizers
# route 2D hidden weights to their spectral path; embeddings/head/1D to the aux.
OPTIMIZER_PARAMS = {
    "MILO":    {"normalize": True, "layer_wise": False, "scale_aware": True, "scale_factor": 0.2,
                 "momentum": 0.9, "adaptive": True, "max_group_size": 5000, "use_cuda_kernels": False},
    "MILO_LW": {"normalize": True, "layer_wise": True,  "scale_aware": True, "scale_factor": 0.2,
                 "momentum": 0.9, "adaptive": True, "max_group_size": 5000, "use_cuda_kernels": False},
    "MILOM":   {"momentum": 0.95, "nesterov": True, "weight_decay": 0.0, "scale_factor": 0.2, "rms_target": 0.2},
    "MION":    {"momentum": 0.95, "nesterov": True, "weight_decay": 0.0, "scale_factor": 0.0,
                 "rms_target": 0.2, "ns_steps": 5},
    "MION_ADAM": {"momentum": 0.95, "nesterov": True, "weight_decay": 0.0, "scale_factor": 0.0,
                 "rms_target": 0.2, "ns_steps": 5, "aux_mode": "adam", "aux_lr": 3e-4},
    "MION_V2": {"momentum": 0.95, "nesterov": True, "weight_decay": 0.0, "scale_factor": 0.0,
                 "rms_target": 0.2, "ns_steps": 5, "momentum_mode": "ema",
                 "aux_mode": "adam", "aux_lr": None},   # aux_lr None => unified single LR
    "MION_EMB": {"momentum": 0.95, "nesterov": True, "weight_decay": 0.0, "scale_factor": 0.0,
                 "rms_target": 0.2, "ns_steps": 5},      # group-std aux; emb LR mult via EMB_LR_MULT
    "MION_NOR": {"momentum": 0.95, "nesterov": True, "weight_decay": 0.0, "scale_factor": 0.0,
                 "rms_target": 0.2, "ns_steps": 5, "row_norm": True},
    "MION_GATED": {"momentum": 0.95, "nesterov": True, "weight_decay": 0.0, "scale_factor": 0.0,
                 "rms_target": 0.2, "ns_steps": 5},      # ortho_strength set via ORTHO_STRENGTH env var
    "ADAMW":   {"betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.1},
    "LION":    {"betas": (0.9, 0.99), "weight_decay": 0.1},
    "ADAM_MINI": {"betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.1},
    "SGD":     {"momentum": 0.9, "nesterov": True, "weight_decay": 0.0},
    "SOAP":    {"betas": (0.95, 0.95), "weight_decay": 0.1, "precondition_frequency": 10},
    "SHAMPOO": {"eps": 1e-10, "momentum": 0.9, "weight_decay": 0.0, "update_freq": 16},
    "MUON":    {"weight_decay": 0.0},
}

# Tuned LRs filled after the LM Optuna sweep; defaults are literature starting points.
LEARNING_RATES = {
    # Optuna-tuned on FineWeb-Edu (60M-token trials, minimize val loss)
    "MILO": 1.68e-3, "MILO_LW": 4.73e-3, "MILOM": 1.31e-3, "MION": 5.61e-3,
    "MION_ADAM": 5.18e-3, "MION_V2": 5.61e-3, "MION_EMB": 5.61e-3,
    "MION_NOR": 5.61e-3,   # placeholder = MION's tuned LR; retune via optuna_sweep.py domain=lm
    "MION_GATED": 5.61e-3,  # placeholder; (lr, ortho_strength) jointly tuned on the FT domain
    "ADAMW": 1.21e-3, "LION": 2.45e-4, "ADAM_MINI": 1.90e-3, "SGD": 6.25e-1,
    "SOAP": 1.12e-3, "SHAMPOO": 2.39e-2, "MUON": 1.12e-2,
}


def build_optimizer(name, model, lr, params):
    """Construct an optimizer; route matrix vs non-matrix params for Muon/MION."""
    import torch
    from milo import milo
    from optimizers.milo2 import MiloM, Mion
    from optimizers.lion import Lion
    from optimizers.adam_mini import AdamMini
    from optimizers.soap import SOAP
    from optimizers.shampoo import Shampoo
    from optimizers.muon import MuonWithAuxAdam

    p = dict(params)
    n = name.upper()
    if n in ("MILO", "MILO_LW"):
        return milo(model.parameters(), lr=lr, **p)
    if n == "MILOM":
        return MiloM(model.parameters(), lr=lr, **p)
    if n in ("MION", "MION_ADAM", "MION_V2", "MION_NOR", "MION_GATED"):
        # Route hidden matrices -> spectral (Newton-Schulz); embeddings/head/1D ->
        # non-spectral path (group-std for MION, AdamW aux for MION_ADAM).
        # Orthogonalizing the large vocab embedding badly hurts LM training.
        hidden, other = [], []
        for nm, prm in model.named_parameters():
            if prm.ndim >= 2 and "embed" not in nm.lower() and "lm_head" not in nm.lower():
                hidden.append(prm)
            else:
                other.append(prm)
        groups = [{"params": hidden, "spectral": True},
                  {"params": other, "spectral": False}]
        if n == "MION_GATED":
            import os
            p = dict(p, ortho_strength=float(os.getenv("ORTHO_STRENGTH", "1.0")))
        return Mion(groups, lr=lr, **p)
    if n == "MION_EMB":
        # group-std everywhere, but embeddings/head get a larger LR (multiplier),
        # since the single-LR design under-trains the large vocab embedding.
        import os
        alpha = float(os.getenv("EMB_LR_MULT", "10"))
        hidden, embed, scalar = [], [], []
        for nm, prm in model.named_parameters():
            is_emb = ("embed" in nm.lower() or "lm_head" in nm.lower())
            if is_emb:
                embed.append(prm)
            elif prm.ndim >= 2:
                hidden.append(prm)
            else:
                scalar.append(prm)
        groups = [{"params": hidden, "spectral": True,  "lr_mult": 1.0},
                  {"params": embed,  "spectral": False, "lr_mult": alpha},
                  {"params": scalar, "spectral": False, "lr_mult": 1.0}]
        return Mion(groups, lr=lr, **p)
    if n == "ADAMW":
        return torch.optim.AdamW(model.parameters(), lr=lr, **p)
    if n == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr, **p)
    if n == "LION":
        return Lion(model.parameters(), lr=lr, **p)
    if n == "ADAM_MINI":
        return AdamMini(model.parameters(), lr=lr, **p)
    if n == "SOAP":
        return SOAP(model.parameters(), lr=lr, **p)
    if n == "SHAMPOO":
        return Shampoo(model.parameters(), lr=lr, **p)
    if n == "MUON":
        wd = p.get("weight_decay", 0.0)
        # hidden matrices -> Muon; embeddings/head/1D -> AdamW aux
        hidden, aux = [], []
        for nm, prm in model.named_parameters():
            if prm.ndim >= 2 and "embed" not in nm.lower() and "lm_head" not in nm.lower():
                hidden.append(prm)
            else:
                aux.append(prm)
        groups = [
            dict(params=hidden, use_muon=True, lr=lr, momentum=0.95, weight_decay=wd),
            dict(params=aux, use_muon=False, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=wd),
        ]
        return MuonWithAuxAdam(groups)
    raise ValueError(f"Unknown optimizer {name}")
