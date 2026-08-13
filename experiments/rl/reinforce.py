"""
REINFORCE policy-gradient on standard Gymnasium control tasks, comparing optimizers.
A fourth domain (sequential decision-making) for breadth beyond vision/NLP/LM.

Metrics: reward curve, final mean reward (last 10% of episodes), episodes-to-threshold,
across seeds. All optimizers built via the shared builder.

  python -m experiments.rl.reinforce --env CartPole-v1 --optimizer MION --episodes 600 --seeds 3
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "hpc", "experiments"))
from smoke_quick import build_optimizer  # shared optimizer construction  # noqa: E402

DEVICE = "cpu"  # tiny policy; CPU is fine and avoids GPU contention
THRESHOLD = {"CartPole-v1": 475.0, "Acrobot-v1": -100.0, "LunarLander-v2": 200.0}

OPTIMIZERS = ["MILO", "MILO_LW", "MILOM", "MION", "SGD", "ADAMW", "ADAGRAD",
              "LION", "ADAM_MINI", "RMSPROP_MOMENTUM", "SHAMPOO", "SOAP", "MUON"]
RL_LR = {"SGD": 0.019, "ADAMW": 0.000518, "ADAGRAD": 0.0121, "LION": 0.000473, "ADAM_MINI": 0.000518, "RMSPROP_MOMENTUM": 0.00112, "SHAMPOO": 0.0254, "SOAP": 0.00841, "MUON": 0.00357, "MILO": 0.00766, "MILO_LW": 0.00766, "MILOM": 0.0017, "MION": 0.0111}
RL_PARAMS = {  # minimal, mirror the supervised configs
    "SGD": {"momentum": 0.9}, "ADAMW": {}, "ADAGRAD": {}, "LION": {},
    "ADAM_MINI": {}, "RMSPROP_MOMENTUM": {"momentum": 0.9}, "SHAMPOO": {"momentum": 0.9},
    "SOAP": {}, "MUON": {"weight_decay": 0.0},
    "MILO": {"normalize": True, "scale_aware": True, "momentum": 0.9, "use_cuda_kernels": False},
    "MILO_LW": {"normalize": True, "layer_wise": True, "scale_aware": True, "momentum": 0.9, "use_cuda_kernels": False},
    "MILOM": {"momentum": 0.95, "nesterov": True}, "MION": {"momentum": 0.95, "nesterov": True},
}


class Policy(nn.Module):
    def __init__(self, obs_dim, n_act, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, n_act))

    def forward(self, x):
        return torch.distributions.Categorical(logits=self.net(x))


def run(env_name, optimizer, lr, episodes, seed, gamma=0.99):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]; n_act = env.action_space.n
    torch.manual_seed(seed); np.random.seed(seed)
    pol = Policy(obs_dim, n_act).to(DEVICE)
    opt = build_optimizer(optimizer, pol, lr, RL_PARAMS.get(optimizer.upper(), {}))
    rewards = []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        logps, rs, done, trunc = [], [], False, False
        while not (done or trunc):
            dist = pol(torch.tensor(s, dtype=torch.float32))
            a = dist.sample()
            logps.append(dist.log_prob(a))
            s, r, done, trunc, _ = env.step(int(a))
            rs.append(r)
        # discounted returns, normalized
        R, returns = 0.0, []
        for r in reversed(rs):
            R = r + gamma * R; returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        loss = -(torch.stack(logps) * returns).sum()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step()
        rewards.append(sum(rs))
    env.close()
    return rewards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="CartPole-v1")
    ap.add_argument("--optimizer", required=True)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out-dir", default="results/rl")
    args = ap.parse_args()
    lr = args.lr if args.lr is not None else RL_LR.get(args.optimizer.upper(), 3e-3)
    thr = THRESHOLD.get(args.env, None)

    print(f"RL {args.env} | {args.optimizer} lr={lr:g} | {args.episodes} ep × {args.seeds} seeds", flush=True)
    all_curves, finals, ep2thr = [], [], []
    t0 = time.time()
    for sd in range(args.seeds):
        rw = run(args.env, args.optimizer, lr, args.episodes, seed=sd)
        all_curves.append(rw)
        finals.append(float(np.mean(rw[-max(1, args.episodes // 10):])))
        hit = next((i for i in range(len(rw)) if np.mean(rw[max(0, i-9):i+1]) >= (thr or 1e9)), None)
        ep2thr.append(hit)
        print(f"  seed {sd}: final={finals[-1]:.1f} ep->thr={hit}", flush=True)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    log = {"env": args.env, "optimizer": args.optimizer, "lr": lr, "seeds": args.seeds,
           "episodes": args.episodes, "threshold": thr,
           "final_reward_mean": float(np.mean(finals)), "final_reward_std": float(np.std(finals)),
           "episodes_to_threshold": [e for e in ep2thr],
           "mean_curve": list(np.mean(np.array(all_curves), axis=0)),
           "secs": round(time.time() - t0, 1)}
    out = Path(args.out_dir) / f"rl_{args.env.replace('-','').lower()}_{args.optimizer.lower()}.json"
    json.dump(log, open(out, "w"), indent=2)
    print(f"\n✓ {args.optimizer} on {args.env}: final {log['final_reward_mean']:.1f}"
          f" ± {log['final_reward_std']:.1f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
