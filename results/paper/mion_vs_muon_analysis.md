# MION vs MUON — step-by-step compute comparison

Mirrored from `optimizers/milo2.py` (Mion) and `optimizers/muon.py` (MuonWithAuxAdam),
with an instrumented single-step trace (`benchmarks/mion_vs_muon_trace.py`).

## Side-by-side update rule

### Hidden matrix W (m×n)
| step | MUON | MION |
|---|---|---|
| 1 momentum | `buf ← (1-β)·buf + β?` **EMA** (`lerp_`, β=.95) | `buf ← μ·buf + g` **heavy-ball** (μ=.95), gain 1/(1-μ)≈20× |
| 2 nesterov | `u = (1-β)g + β·buf` | `d = g + μ·buf` |
| 3 orthogonalize | Newton–Schulz5 (bf16, quintic) | Newton–Schulz5 (same coeffs) |
| 4 scale | `× max(1, m/n)^½` → **spectral-norm units** (σ≈1) | `× √max(m,n)` → **RMS units** (RMS≈1), then `× rms_target(0.2)` |
| 5 weight decay | decoupled `p·=(1-lr·wd)` | decoupled `p·=(1-lr·wd)` |
| 6 update | `p -= lr·u` | `p -= lr·0.2·u` |

### Non-matrix params (embedding / head / norms / bias)
| | MUON | MION |
|---|---|---|
| rule | **AdamW**: per-coordinate `m̂/(√v̂+ε)`, bias-corrected | **group-standardize**: per-row (x-μ)/σ, blend `sf·(d/rms)+(1-sf)·std` |
| LR | **separate** `lr_aux` (often very different: head .22 / embed .6 / scalar .04) | **same single LR** as matrices (unified by rms_target) |
| state | 2 buffers (exp_avg, exp_avg_sq) | 1 buffer (momentum), shared design |

## Empirical trace (one step, identical gradients)
- **Matrix update direction: cosine(MUON, MION) = 1.0000.** The matrix path is the
  *same algorithm* — orthogonalized nesterov momentum — differing only by an overall
  scale (MUON singular values ≈1.5–2, MION ≈23–33 = ×√max(m,n)). **That scale is
  absorbed by the LR**, so on matrices MION ≡ MUON up to LR reparameterization.
  (Caveat: identical at step 1; over training the EMA-vs-heavy-ball momentum memory
  makes directions drift slightly.)
- **Embedding update direction: cosine(MUON-Adam, MION-groupstd) = 0.79.** This is the
  *real* divergence. Adam's per-coordinate second moment rescales coordinates over a
  ~5e8 dynamic range (rare-token embedding rows), which group-standardization cannot
  reproduce. **This 0.79 (vs 1.0 on matrices) is the quantitative source of MION's LM gap.**

## Where each excels
- **MUON**: large vocab embeddings / heavy-tailed-gradient params — Adam's per-coord
  adaptivity + a dedicated aux LR. → wins LM (3.238 vs 3.281).
- **MION**: everything where the aux path matters less — wins all vision tasks; uses
  **1 optimizer buffer for all params** (vs MUON's 2 on aux) → half AdamW memory,
  less than MUON; **single LR** (no aux-LR tuning), and the controlled study showed
  group-std+1LR matches a tuned Adam-aux on vision.

## MION's limitations vs MUON, and how to address
1. **Non-adaptive aux path** (the LM gap). Embeddings need per-coordinate adaptivity.
   *Fix options:* (a) add a cheap per-row/second-moment EMA to group-std for embeddings
   only (keeps 1 LR, adds adaptivity without full Adam state); (b) co-tune MION_ADAM's
   aux LR (we saw fixed aux_lr hurt — 3.548); (c) accept a 2-group LR (matrices vs
   embeddings) — middle ground between MUON's many LRs and MION's one.
2. **Single global LR across matrix + aux + layer shapes.** *Fix:* shape-aware
   rms_target, or muP-style width scaling so one LR transfers; or the 2-group LR above.
3. **Heavy-ball (non-EMA) momentum** → effective time-constant differs from MUON's
   well-tuned EMA and isn't magnitude-stable. *Fix:* switch to `lerp_` EMA momentum
   (match MUON) — cheap, likely strictly better.
4. **RMS-units scaling (`√max(m,n)`·rms_target) vs MUON's spectral-norm units.** Mion's
   per-update spectral norm grows with √max(m,n); MUON's is shape-invariant. *Fix:*
   adopt MUON's `max(1,m/n)^½` scaling for the matrix path for better cross-layer LR
   transfer.

## Empirical follow-up: the obvious improvements backfire (MION is near-optimal)
Two principled attempts to close the 0.043 LM gap both made MION *worse*:
- **MION_V2** (EMA momentum + per-coord adaptive embeddings, unified LR): 3.692.
- **MION_EMB** (group-std embeddings + larger embedding LR): α=4→3.449, 12→3.667,
  36→3.985 — monotonically worse with higher embedding LR.
So the embedding is **not** under-trained and adaptivity is **not** missing — MION's
group-std + single LR is already well-balanced. The residual 0.043 gap (3.281 vs
3.238, single seed) is small and likely partly noise; the only untested structural
difference is the matrix-update scaling (RMS-units vs spectral-norm units). Practical
verdict: report MION as-is; the design resists the complexity Muon needs.

## One-line takeaway
On matrices MION and MUON are the same method (cosine 1.0, scale folds into LR); MION's
only real deficit is the **non-adaptive embedding path** (cosine 0.79). Adding light
embedding adaptivity while keeping the single-LR/1-buffer design is the highest-value
improvement and directly targets the LM gap without sacrificing MION's memory/simplicity edge.
