"""Fast correctness tests: every available optimizer must (a) construct,
(b) take 30 steps on a tiny over-parameterized regression without NaN,
(c) reduce the loss. Optimizers whose packages are missing are skipped with
a notice. Run: python -m pytest tests/ -q   (or just: python tests/test_optimizers.py)
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimizers.registry import ALL_OPTIMIZERS, build_optimizer  # noqa: E402

LRS = {"sgdm": 0.05, "adamw": 1e-2, "nadamw": 1e-2, "adafactor": 1e-2,
       "lion": 1e-3, "schedulefree": 1e-2, "prodigy": 1.0, "sophia": 1e-3,
       "muon": 0.02, "soap": 1e-2, "shampoo": 1e-2, "kron": 1e-3,
       "ademamix": 1e-2, "milo": 0.05, "milo_m": 0.01, "mion": 0.01}


def tiny_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 64), nn.ReLU(),
                         nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))


def run_one(name):
    model = tiny_model()
    opt, meta = build_optimizer(name, model, LRS[name], weight_decay=0.0,
                                **({"sophia_bs": 256} if name == "sophia" else {}))
    if meta["sf_train_eval"]:
        opt.train()
    torch.manual_seed(1)
    x = torch.randn(256, 16)
    y = (x[:, :4].sum(1, keepdim=True) + 0.1 * torch.randn(256, 1))
    loss0 = None
    for step in range(30):
        loss = ((model(x) - y) ** 2).mean()
        if loss0 is None:
            loss0 = loss.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(**meta["step_kwargs"])
        assert torch.isfinite(loss), f"{name}: NaN/Inf at step {step}"
        if meta["hessian_every"] and step % meta["hessian_every"] == 0:
            loss_h = ((model(x)) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss_h.backward()
            opt.update_hessian()
            opt.zero_grad(set_to_none=True)
    assert loss.item() < loss0, f"{name}: loss did not decrease ({loss0:.4f} -> {loss.item():.4f})"
    return loss0, loss.item()


def main():
    failed = []
    for name in ALL_OPTIMIZERS:
        try:
            l0, l1 = run_one(name)
            print(f"PASS {name:13s} {l0:.4f} -> {l1:.4f}")
        except (ImportError, ModuleNotFoundError) as e:
            print(f"SKIP {name:13s} ({e})")
        except Exception as e:
            print(f"FAIL {name:13s} {type(e).__name__}: {e}")
            failed.append(name)
    if failed:
        sys.exit(f"failures: {failed}")


# pytest entry points
def test_all():
    main()


if __name__ == "__main__":
    main()
