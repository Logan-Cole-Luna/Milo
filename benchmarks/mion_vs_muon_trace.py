"""
Step-by-step compute trace of MION vs MUON on IDENTICAL gradients.

For one real hidden matrix and the token embedding, mirror each optimizer's
update math exactly (from optimizers/milo2.py and optimizers/muon.py), capture
every intermediate's statistics, the singular-value spectrum, the direction
agreement (cosine similarity) between the two updates, per-op timing, and the
optimizer-state memory. Tiny model -> millisecond diagnostic.

  python -m benchmarks.mion_vs_muon_trace
"""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
from experiments.lm.model import GPT, GPTConfig

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"

def st(t):  # summary stats of a tensor
    t = t.float()
    return (f"shape={tuple(t.shape)} rms={t.norm()/math.sqrt(t.numel()):.4f} "
            f"max|.|={t.abs().max():.4f}")

def spectral(t):
    sv = torch.linalg.svdvals(t.float().reshape(t.shape[0], -1))
    return sv

# ---- tiny model + one real backward ----
cfg = GPTConfig(vocab_size=4096, n_layer=2, n_head=4, n_embd=256, block_size=64)
m = GPT(cfg).to(dev)
x = torch.randint(0, 4096, (4, 64), device=dev); y = torch.randint(0, 4096, (4, 64), device=dev)
_, loss = m(x, y); loss.backward()

W = m.blocks[0].attn.qkv.weight          # hidden matrix (768x256)
E = m.tok_emb.weight                     # embedding (4096x256) -- tied to head
gW, gE = W.grad.clone(), E.grad.clone()

# ================= MATRIX PATH =================
print("="*70); print(f"  MATRIX PATH — qkv.weight {tuple(W.shape)}"); print("="*70)
beta = 0.95; ns = 5

# ---- MUON matrix update (mirror muon_update) ----
def muon_ns(G, steps):
    a,b,c = 3.4445,-4.7750,2.0315
    X = G.bfloat16() if G.is_cuda else G.float()
    if X.size(-2) > X.size(-1): X = X.mT
    X = X / (X.norm(dim=(-2,-1), keepdim=True)+1e-7)
    for _ in range(steps):
        A = X@X.mT; B = b*A + c*A@A; X = a*X + B@X
    if G.size(-2) > G.size(-1): X = X.mT
    return X.to(G.dtype)
buf_mu = torch.zeros_like(gW); buf_mu.lerp_(gW, 1-beta)        # EMA momentum
upd_mu = gW.lerp(buf_mu, beta)                                # nesterov lerp
o_mu = muon_ns(upd_mu, ns)
o_mu = o_mu * max(1, o_mu.size(-2)/o_mu.size(-1))**0.5        # aspect scale
print("MUON : EMA-momentum then nesterov-lerp ->", st(upd_mu))
print("       NS(orthogonalized) * max(1,m/n)^.5 ->", st(o_mu))

# ---- MION matrix update (mirror Mion spectral path) ----
mu = 0.95; rms_target = 0.2
buf_mi = torch.zeros_like(gW); buf_mi.mul_(mu).add_(gW)        # heavy-ball accumulation
d_mi = gW.add(buf_mi, alpha=mu)                               # nesterov
def mion_ns(G, steps, eps=1e-7):
    a,b,c=3.4445,-4.7750,2.0315
    X=G.bfloat16() if G.is_cuda else G.float()
    tr = X.size(0)>X.size(1)
    if tr: X=X.T
    X = X/(X.norm()+eps)
    for _ in range(steps):
        A=X@X.T; B=b*A+c*(A@A); X=a*X+B@X
    if tr: X=X.T
    return X.to(G.dtype)
o_mi = mion_ns(d_mi.reshape(d_mi.shape[0],-1), ns)
u_mi = o_mi * math.sqrt(max(o_mi.shape))                      # rescale to RMS~1
print("MION : heavy-ball momentum then nesterov  ->", st(d_mi))
print("       NS * sqrt(max(m,n))  (pre-rms_target) ->", st(u_mi))

# singular-value spectra (the heart of the difference)
svW = spectral(gW); sv_mu = spectral(o_mu); sv_mi = spectral(u_mi)
def svinfo(sv): return f"min={sv.min():.3f} med={sv.median():.3f} max={sv.max():.3f} (n={len(sv)})"
print("\nsingular values:")
print("  raw grad   :", svinfo(svW))
print("  MUON update:", svinfo(sv_mu), " <- ~uniform => spectral-norm units")
print("  MION update:", svinfo(sv_mi), " <- ~uniform * sqrt(max) => RMS units")

# direction agreement after each rescales to unit RMS
nu_mu = (o_mu/(o_mu.norm())).flatten(); nu_mi = (u_mi/(u_mi.norm())).flatten()
print(f"\ncosine(MUON dir, MION dir) on matrix = {torch.dot(nu_mu.float(), nu_mi.float()):.4f}")

# ================= EMBEDDING / AUX PATH =================
print("\n"+"="*70); print(f"  AUX PATH — token embedding {tuple(E.shape)}"); print("="*70)
# MUON aux = Adam
b1,b2,eps=0.9,0.95,1e-10
m1=torch.zeros_like(gE); m2=torch.zeros_like(gE)
m1.lerp_(gE,1-b1); m2.lerp_(gE.square(),1-b2)
adam = (m1/(1-b1))/((m2/(1-b2)).sqrt()+eps)
# MION aux = group-standardize (per-row, scale_factor blend)
def _rms(t,eps=1e-8): return t.norm()/math.sqrt(t.numel())+eps
def group_std(d, sf=0.0, eps=1e-8):
    mat=d.reshape(d.shape[0],-1); mean=mat.mean(1,keepdim=True); s=mat.std(1,keepdim=True)+eps
    out=((mat-mean)/s).view_as(d)
    if sf>0: out = sf*(d/_rms(d)) + (1-sf)*out
    return out
buf_e=torch.zeros_like(gE); buf_e.mul_(mu).add_(gE); de=gE.add(buf_e,alpha=mu)
gs = group_std(de)
print("MUON aux (Adam, per-coordinate adaptive):", st(adam))
print("MION aux (group-std, per-row, no 2nd-moment):", st(gs))
nad=(adam/adam.norm()).flatten(); ngs=(gs/gs.norm()).flatten()
print(f"cosine(MUON-Adam dir, MION-groupstd dir) on embedding = {torch.dot(nad.float(),ngs.float()):.4f}")
# per-coordinate adaptivity: how much Adam rescales columns vs groupstd
print(f"Adam per-coord scale spread (max/min of |update|/|grad| over coords): "
      f"{(adam.abs()/(gE.abs()+1e-12)).max()/(adam.abs()/(gE.abs()+1e-12)).clamp_min(1e-12).min():.1e}")

# ================= TIMING + MEMORY =================
print("\n"+"="*70); print("  PER-OP TIMING (matrix path, avg of 50)"); print("="*70)
def timeit(fn, n=50):
    for _ in range(3): fn()
    if dev=="cuda": torch.cuda.synchronize()
    t=time.time()
    for _ in range(n): fn()
    if dev=="cuda": torch.cuda.synchronize()
    return (time.time()-t)/n*1e3
t_ns   = timeit(lambda: muon_ns(upd_mu, ns))
t_mom  = timeit(lambda: torch.zeros_like(gW).lerp_(gW,1-beta))
t_adam = timeit(lambda: (m1/(1-b1))/((m2/(1-b2)).sqrt()+eps))
t_gs   = timeit(lambda: group_std(de))
print(f"  Newton-Schulz (shared, dominates): {t_ns:.3f} ms")
print(f"  momentum update              : {t_mom:.3f} ms")
print(f"  MUON aux step (Adam)         : {t_adam:.3f} ms")
print(f"  MION aux step (group-std)    : {t_gs:.3f} ms")
print("\nOptimizer-state buffers per parameter:")
print("  MUON: hidden=1 (momentum); aux=2 (exp_avg, exp_avg_sq)")
print("  MION: ALL params=1 (momentum)            <- memory advantage")
