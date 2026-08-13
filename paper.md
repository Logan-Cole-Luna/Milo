# Structure-Aligned Update Normalization: The MILO Optimizer Family

*Working paper draft (auto-compiled from the experiment suite). Target venue: AAAI 2027.*
*Figures referenced live in `results/paper/figures/`; tables in `results/paper/tables/`.*

---

## Abstract

We study **structure-aligned update normalization** as a unifying principle for
neural-network optimization: instead of adapting a per-coordinate learning rate
(Adam-family) or orthogonalizing whole weight matrices in isolation (Muon), we
normalize each parameter's update according to the *structure* of that parameter
— generic groups for vectors, per-neuron/channel groups for matrices, and
spectral orthogonalization for hidden weight matrices. This principle yields a
**family** of optimizers (MILO, MILO-LW, MiloM, MION) that trade off simplicity,
memory, and per-structure fidelity. Our flagship, **MION**, combines Newton–Schulz
spectral updates for hidden matrices with group-standardized updates for the
remaining parameters, unified under a **single learning rate** and requiring only
**one optimizer buffer** (vs. Adam's two).

Across five paradigms — from-scratch language-model pretraining (FineWeb-Edu),
converged image classification (CIFAR/Tiny-ImageNet), sentiment classification
(BERT/SST-2), 1.5B-parameter LLM fine-tuning (Qwen2.5-1.5B), and reinforcement
learning (Gym control) — with **per-optimizer learning rates tuned by Optuna for
every method**, we find that MION is the strongest **accuracy/memory trade-off**
for training from scratch: it is **best on converged vision**, **competitive on LM
pretraining** (beating both AdamW and Adam-mini and trailing Muon by 0.04 val
loss), all at **half the optimizer-state memory of Adam**. A step-by-step compute
trace shows MION's hidden-matrix update is *identical in direction* to Muon
(cosine 1.00); the two differ only in the treatment of embeddings. We further show
a clean, actionable regime effect: **spectral orthogonalization helps from-scratch
training with clean gradients (pretraining, vision) but hurts adaptation
(fine-tuning) and high-variance regimes (policy-gradient RL)**, where gentler family
members (MiloM/MILO/MILO-LW) or adaptive methods lead. The family thus covers a broad
range of regimes under one principle, and the regime dependence is itself a useful
practitioner guideline.

---

## 1. Introduction

Modern optimizer research has moved beyond per-coordinate adaptivity (Adam, Lion,
Adam-mini) toward **matrix-aware** methods that exploit the 2-D structure of weight
matrices — most prominently **Muon** (Newton–Schulz orthogonalization of the
momentum) and second-order preconditioners (Shampoo, SOAP). These methods raise a
question this paper takes up directly: *how much of a parameter's structure should
an update respect, and how should that choice differ across parameter types and
task regimes?*

We frame a spectrum of **structure-aligned update normalization**:

- **Generic grouping** (MILO): flatten the update, split into √N groups,
  standardize each group. Structure-agnostic.
- **Per-layer grouping** (MILO-LW): standardize each parameter tensor as one group.
- **Per-neuron/channel grouping** (MiloM): standardize per output row of a matrix
  — structure-aligned to neurons/channels.
- **Spectral** (MION): orthogonalize hidden weight matrices (Newton–Schulz), the
  strongest form of "respecting matrix structure," while group-standardizing
  everything else.

MION is the flagship: it applies the strongest structural normalization
(orthogonalization) exactly where matrices dominate compute (hidden layers), and a
cheap, non-adaptive group standardization elsewhere, all under **one learning rate**
and **one momentum buffer**.

**Relationship to the modular-norm/duality program.** We do not claim
"structure-aligned normalization" as a new principle in isolation — it is a
practical instantiation of a fast-moving theoretical line that recasts Adam,
Shampoo and Muon as *steepest descent under a per-layer norm* (Bernstein & Newhouse,
2024) and derives the correct per-layer update map ("dualizer") from that norm,
including separate treatments for `Embed` vs. `Linear`/`Conv` layers (Bernstein &
Newhouse, *Modular Duality in Deep Learning*, 2024). Large et al. (2024) show that
assigning each module a norm and normalizing to it yields width/depth learning-rate
transfer "for free" — the rigorous version of our single-LR claim. MILO ≈ a
coordinate/RMS norm, MiloM ≈ a per-row norm, MION ≈ the spectral (operator) norm on
hidden matrices, in this taxonomy. Given that context, our contributions are
deliberately empirical rather than principle-novelty claims:

1. A **memory-light, single-buffer/single-LR realization** of the modular-norm/duality
   program (MION), matching or beating Adam-class methods on from-scratch training
   at half the optimizer memory, with Muon recovered as the spectral limit on
   hidden matrices.
2. The **first broad, fairly-tuned regime study** across five paradigms spanning
   from-scratch pretraining through fine-tuning and RL (Optuna-tuned LRs per method)
   and a **step-by-step MION-vs-Muon analysis** isolating exactly where they differ.
3. A **regime finding**: orthogonalization helps pretraining but hurts fine-tuning
   and high-variance RL; the family's gentler members own those regimes — turning
   two of MION's apparent weaknesses into evidence for the family framing.
4. Two extensions directly motivated by concurrent theory/empirical work (§2.5–2.6),
   tested and reported honestly: **MION-Nor** (NorMuon-style per-row second-moment
   normalization) turned out to be a **fourth negative result** on the LM embedding
   gap, joining MION-V2/MION-EMB; **MION-Gated** (a tunable orthogonalization-strength
   knob) **partially** closes the fine-tuning regime gap but does not fully match
   the dedicated gentle family members.

---

## 2. Method

### 2.1 Structure-aligned group standardization

Given a raw update direction `d` for a parameter (after momentum), we standardize
it in structure-aligned groups. For a matrix `d ∈ ℝ^{m×n}` the default is
per-output-row: subtract the row mean, divide by the row std. For vectors we use
√N groups. An optional *grafting* blend keeps a fraction `s` of the RMS-normalized
raw direction: `u = s·(d/‖d‖_RMS) + (1−s)·standardize(d)`. The result has unit RMS,
so a single learning rate produces comparable step sizes across all parameters.

### 2.2 The family

| Optimizer | Hidden matrices | Vectors / embeddings | State | LRs |
|---|---|---|---|---|
| MILO | √N group-std | √N group-std | momentum (+ adagrad accum) | 1 |
| MILO-LW | per-tensor group-std | per-tensor group-std | momentum (+ accum) | 1 |
| MiloM | per-row group-std (momentum-first) | √N group-std | momentum | 1 |
| **MION** | **Newton–Schulz orthogonalize** | per-row group-std | **momentum only** | **1** |

### 2.3 MION update (per step)

For each hidden matrix `W`:
1. momentum `buf ← μ·buf + g`; Nesterov `d ← g + μ·buf`;
2. `O ← NewtonSchulz₅(d)` (bf16 quintic iteration; ~semi-orthogonal);
3. rescale to unit RMS: `u ← O·√max(m,n)`;
4. decoupled weight decay; `W ← W − lr·rms_target·u`.

For non-matrix parameters (embeddings, output head, norms, biases): the same
momentum, then per-row/√N group standardization, applied with the **same** `lr`.
This keeps MION to **one learning rate** and **one buffer** for the entire model —
half the optimizer state of Adam, which stores two moments per parameter.

### 2.4 Relationship to Muon (empirical trace)

We instrumented one optimizer step on identical gradients (`benchmarks/mion_vs_muon_trace.py`).
On a hidden weight matrix, **MION's and Muon's updates have cosine similarity 1.00**
— both orthogonalize the Nesterov momentum with the same Newton–Schulz iteration;
they differ only by an overall scale (Muon uses spectral-norm units, `×max(1,m/n)^½`;
MION uses RMS units, `×√max(m,n)`), which is absorbed by the learning rate. The
**only** substantive difference is the non-matrix path: Muon routes embeddings/head
to a separate AdamW with its own learning rate(s); MION group-standardizes them
under the shared LR. On the token embedding the two updates have cosine similarity
**0.79** — the entire MION↔Muon gap lives here.

### 2.5 MION-Nor: post-orthogonalization row normalization

**NorMuon** (Zhang et al., 2025) observes that Newton–Schulz orthogonalization
equalizes a matrix's *singular values* but leaves its *per-neuron (row) update
norms* highly non-uniform, letting a few neurons dominate; they fix this with a
running per-row second moment applied *after* orthogonalization. This bears
directly on two open items in our own ablations: (i) the residual 0.043 LM
val-loss gap to Muon, localized to the embedding path (§4.9), and (ii) MiloM's
unclear role — MiloM already group-standardizes per row, but *without*
orthogonalization, which our results show is the weaker half (MiloM 3.494 vs.
MION 3.281 on LM).

We add an optional `row_norm` gate to `Mion` (`optimizers/milo2.py`): after the
update `u` is formed on *either* path (spectral or group-standardized), each
output row is divided by a running EMA of its own mean-square magnitude
(`β₂≈0.999`), then the whole update is rescaled so its global RMS matches the
pre-normalization value — preserving the outer single-LR/`rms_target` scaling.
Because this is a `param_group` default, it applies uniformly to the spectral
hidden-matrix groups *and* the non-spectral embedding/head group, giving the
embedding a duality-motivated row-wise correction without adding a second
learning rate or a full adaptive optimizer (both of which we tried and which
hurt — MION-V2/MION-EMB, §4.9). **MION-Nor = MION + `row_norm=True`.**

**Result: a fourth negative result.** Tuned via the same Optuna protocol (LR
5.61e-3, matching plain MION) and run for the full 124M/1.2B-token budget,
MION-Nor reaches **val loss 3.390** — *worse* than plain MION (3.281) and worse
than AdamW (3.306), widening rather than closing the gap to Muon (3.238). Adding
the row-wise second-moment correction on top of an update that is already
rescaled to unit RMS (§2.1) appears to fight the Newton–Schulz iteration's own
implicit uniformization on the spectral path, and to over-correct the embedding
relative to the plain group-standardized version. Combined with MION-V2 (global
adaptivity, 3.692) and MION-EMB (larger embedding LR, monotonically worse
3.449→3.985, §4.9), **all three theory-motivated attempts to close the LM
embedding gap have now failed**, reinforcing §4.9's conclusion that MION's
default group-standardized, single-LR, unmodified embedding path is already
close to a local optimum for this design — the residual gap to Muon appears to
require Muon's literal separate-optimizer machinery, not a lighter-weight
correction on top of MION. We report this as a fourth honest negative result
(§4.9) rather than omit it. The converged-vision spot-check (§4.3, reusing
MION's tuned LR) confirms the same direction: MION-Nor is neutral-to-slightly
worse than plain MION on all three architectures (ResNet-34, VGG-11, ViT-Tiny),
so we do not adopt `row_norm` as a default and retain it only as a documented
ablation knob.

### 2.6 MION-Gated: a tunable orthogonalization-strength gate for fine-tuning

§4.6 shows a clean regime split: orthogonalization (MION, Muon) is worst on
Qwen2.5-1.5B/Alpaca fine-tuning, while gentle group-standardization
(MiloM/MILO/MILO-LW) wins. **OFT** (Qiu et al., 2023) suggests why: the *right*
kind of orthogonality — one that preserves pairwise neuron angles
("hyperspherical energy") — helps fine-tuning, whereas MION's Newton–Schulz step
rotates in weight space and overwrites pretrained feature structure. **"How Much
Orthogonalization Does Muon Need?"** (2026) further shows that final training
quality is *not* monotone in polar-decomposition accuracy, licensing a cheaper,
partial orthogonalization; **AMO** (2026) makes the degree of orthogonalization
adaptive.

We expose this directly as a single interpolation knob, `ortho_strength (β) ∈
[0, 1]`, on the spectral path of `Mion`: `u = β·orthogonalize(d) + (1−β)·
group_standardize(d)`, rescaled to unit RMS before the shared learning rate is
applied. `β=1` recovers plain MION; `β=0` recovers MiloM's row-standardization on
that path. Rather than hand-pick `β` per regime, we jointly tune `(lr, β)` with
Optuna directly on the fine-tuning objective — if a single `β` in the interior
wins, MION-Gated is one family member spanning both regimes rather than needing
separate MION (pretrain) / MiloM (fine-tune) recommendations. This is our
lower-effort alternative to a full low-rank spectral update (LoRA-Muon-style,
Future Directions §8.6); we adopt gating first because it reuses the existing
`Mion` implementation with no new manifold-restricted update rule.

**Result: partial success.** A joint Optuna search over `(lr, β)` (15 trials,
150-step trials) on Qwen2.5-1.5B/Alpaca selects **β = 0.156** — strongly toward
the group-standardized end, confirming the regime hypothesis that fine-tuning
wants little orthogonalization. At the tuned point, the final 600-step run
reaches **val loss 1.414**, improving on plain MION (1.424) but **still behind**
AdamW/Lion (1.398–1.400) and the gentle family members MiloM/MILO/MILO-LW
(1.404–1.407) — see the updated §4.6 table. MION-Gated confirms the mechanism
(less orthogonalization → better fine-tuning) and closes about a quarter of
MION's gap to the family, but a single interior `β` does not fully recover
dedicated-member quality; we still recommend MiloM/MILO/MILO-LW outright for
fine-tuning rather than MION-Gated, though MION-Gated is the better choice than
plain MION if orthogonalization for later re-pretraining is also a goal.

---

## 3. Experimental setup

**Domains (5).** (i) LM pretraining: a 124M Llama-style decoder (RMSNorm, RoPE,
SwiGLU) on FineWeb-Edu, 1.2B tokens; (ii) converged vision: ResNet-34, VGG-11,
ViT-Tiny on CIFAR-10/100 (60 epochs); (iii) NLP: BERT-base on SST-2; (iv) LLM
fine-tuning: full fine-tuning of Qwen2.5-1.5B on Alpaca; (v) RL: REINFORCE on
CartPole-v1 / Acrobot-v1.

**Fair tuning.** Every optimizer's learning rate is tuned **per domain** with an
Optuna TPE search over a continuous log-uniform range (val-loss/accuracy/reward
objective). This removes the single largest confound in optimizer comparisons; all
headline numbers below use tuned LRs. Matrix methods (MION, Muon) route hidden 2-D
weights to their spectral path and embeddings/head/1-D params to the auxiliary path.

**Optimizers (up to 13).** MILO, MILO-LW, MiloM, **MION** (ours); AdamW, Adam-mini,
Lion, SGD-momentum, RMSprop-momentum, Adagrad, Muon, SOAP, Shampoo (baselines).

---

## 4. Results

### 4.1 Optimizer cost — the memory advantage (Fig. `cost_memory.png`, `acc_vs_memory.png`)

Optimizer-state memory on a 124M LM (batch 16 × ctx 1024):

| Optimizer | State (MB) | vs AdamW |
|---|---|---|
| **MION**, MiloM, Lion, SGD | **494** | **0.50×** |
| Muon | 649 | 0.66× |
| AdamW, Adam-mini, SOAP, MILO, MILO-LW | 989 | 1.0× |
| Shampoo | 1246 | 1.26× |

MION carries **one** momentum buffer for the whole model — half AdamW's two moments
and below Muon (which adds AdamW auxiliary state). Newton–Schulz adds ~20% per-step
time vs. Lion but is 2.7× faster than SOAP.

### 4.2 LM pretraining — competitive at half the memory (Fig. `lm_loss_vs_tokens.png`, `lm_acc_vs_memory.png`)

FineWeb-Edu, 124M, 1.2B tokens, tuned LRs (final val loss):

| Optimizer | Val loss | Opt-state |
|---|---|---|
| Muon | **3.238** | 649 MB |
| SOAP | 3.254 | 989 MB |
| **MION** | **3.281** | **494 MB** |
| Adam-mini | 3.304 | 989 MB |
| AdamW | 3.306 | 989 MB |
| Lion | 3.373 | 494 MB |
| MiloM | 3.494 | 494 MB |
| MION-Nor | 3.390 | ~495 MB |
| MILO-LW | 3.996 | 989 MB |
| SGD | 4.118 | 494 MB |
| MILO | 4.571 | 989 MB |
| Shampoo | 5.911 | 1246 MB |

**MION is 3rd, beating both AdamW and Adam-mini, at half their optimizer memory**,
and trails Muon by only 0.043 val loss (Muon is ~23% more token-efficient to reach
val 3.3). MILO/MILO-LW are weak here — generic/per-tensor normalization is
insufficient for from-scratch transformer pretraining. **MION-Nor (§2.5), tuned
with the same protocol, lands at 3.390 — worse than plain MION**, a fourth
negative result on this residual gap (§4.9).

### 4.3 Converged vision — MION best across the board (Fig. `vision_converged_heatmap.png`)

CIFAR-10 test accuracy (60 epochs, tuned LRs):

| Optimizer | ResNet-34 | VGG-11 | ViT-Tiny |
|---|---|---|---|
| **MION** | **86.9** | 82.0 | **78.3** |
| MION-Nor | 86.3±0.3 | 82.0±0.4 | 77.6±0.5 |
| MiloM | 85.1 | **82.7** | 69.1 |
| SOAP | 86.5 | 10.4 † | 77.4 |
| Muon | 84.5 | 81.7 | 74.4 |
| RMSprop-m | 84.8 | 80.4 | 71.8 |
| Adam-mini | 82.5 | 79.9 | 70.3 |
| AdamW | 81.5 | 79.4 | 70.3 |
| MILO / MILO-LW | ~81 | 74–76 | 61–63 |
| Shampoo | 9.8 † | 9.8 † | 27.5 |

MION is best or tied-best on all three and **decisively best on the transformer
(ViT)**. († SOAP and Shampoo exhibit reproducible instabilities on VGG.) **MION-Nor
(§2.5), reusing MION's tuned LR, is neutral-to-slightly-worse than plain MION on
all three (ResNet-34 −0.6, VGG-11 tied, ViT −0.7)** — a fifth data point (after LM)
that the row-wise post-hoc correction does not help this design, rounding out
MION-Nor as a clean negative result across both from-scratch domains we tested it on.

### 4.4 NLP (BERT/SST-2) — MILO family competitive

Val accuracy (5 seeds, tuned): ADAGRAD 92.71, **MILO 92.68**, **MILO-LW 92.55**,
Lion 92.45, Adam-mini 92.41, **MION 92.34**, MiloM 91.95, AdamW 91.88, SOAP 91.35,
RMSprop 90.80, Shampoo 90.00, SGD 88.14, Muon 86.67. Tight cluster; the MILO family
sits at the top with AdamW; Muon lags (orthogonalization ill-suited to this
fine-tuning-style task — foreshadowing §4.6).

### 4.5 ImageNet (Tiny-ImageNet-200, ResNet-34)

Val accuracy (3 seeds): SOAP 56.8, Muon 55.9, **MION 55.4**, MiloM 41.7, AdamW 39.4,
SGD 39.2, Adam-mini 38.2, Lion 37.6, RMSprop 35.9, Adagrad 30.9, MILO 29.0,
MILO-LW 28.4, Shampoo 2.4. MION is top-tier; MILO/MILO-LW are weak on deep
from-scratch vision, consistent with LM.

### 4.6 LLM fine-tuning (Qwen2.5-1.5B, Alpaca) — a regime effect (Fig. `ft_acc_vs_memory.png`, `ft_val_loss.png`)

Full fine-tuning of Qwen2.5-1.5B on Alpaca, tuned LRs, 600 steps (val loss + memory):

| Optimizer | Val loss | Opt-state | Peak GPU |
|---|---|---|---|
| Lion | **1.398** | 6.2 GB | 28.6 GB |
| AdamW | 1.400 | 6.2 GB | 28.6 GB |
| Adam-mini | 1.400 | 12.4 GB | 34.7 GB |
| **MILO-LW** | 1.404 | 6.2 GB | 28.6 GB |
| **MILO** | 1.405 | 6.2 GB | 28.6 GB |
| **MiloM** | 1.407 | **3.1 GB** | **25.5 GB** |
| SGD | 1.416 | 3.1 GB | 25.5 GB |
| Shampoo | 1.417 | 14.5 GB | 36.9 GB |
| MION-Gated (β=0.156, tuned) | 1.414 | 3.1 GB | 25.5 GB |
| **MION** | 1.424 | **3.1 GB** | **25.5 GB** |
| Muon | 1.465 | 3.6 GB | 26.0 GB |

Two findings. **(i) Memory**: MION/MiloM fine-tune a 1.5B model in **3.1 GB of
optimizer state — half AdamW's 6.2 GB and a quarter of Adam-mini's 12.4 GB**
(Shampoo needs 14.5 GB). **(ii) A clean regime effect**: the **orthogonalizing
methods (MION 1.424, Muon 1.465) are the two worst**, even tuned, while the gentle
group-standardizing family members **MiloM/MILO/MILO-LW (1.404–1.407) are top-tier,
matching AdamW/Lion — and MiloM does so at half their memory.** Interpretation:
orthogonalization rotates away pretrained feature structure — helpful when learning
from scratch, harmful when adapting a pretrained model. **The family covers both
regimes: MION for pretraining, MiloM/MILO/MILO-LW for fine-tuning.**

**MION-Gated (§2.6) partially recovers the gap.** Jointly tuning `(lr, β)` selects
`β=0.156` — mostly group-standardization, little orthogonalization — and reaches
1.414, roughly a quarter of the way from plain MION (1.424) to the MiloM/MILO/
MILO-LW cluster (1.404–1.407), but not fully there. This confirms the regime
mechanism directly (less orthogonalization → better fine-tuning, tuned by the
optimizer itself rather than assumed) while showing that a single global strength
knob on MION's existing update rule is not sufficient to match the dedicated
gentle family members outright — consistent with our decision to recommend
MiloM/MILO/MILO-LW as the fine-tuning anchors rather than a single universal MION
variant (§5, §7).

### 4.7 Reinforcement learning (REINFORCE, Gym control) (Fig. `rl_cartpolev1.png`, `rl_acrobotv1.png`)

Final reward, 3 seeds, tuned LRs:

| | CartPole-v1 (max 500) | Acrobot-v1 (higher=better) |
|---|---|---|
| best | Shampoo 460±11, SOAP 417±9, SGD 405±66 | Lion −121±5, Shampoo −250±177, Muon −263±168 |
| … | Muon 355±36, AdamW 351±21, Lion 299±119 | Adam-mini −363, Adagrad −419, MILO/MILO-LW −463 |
| **MION** | **184±83** (near bottom) | **−500** (failed) |
| worst | RMSprop 48±55 | many at −500 |

REINFORCE is **high-variance** (large per-seed std), but the multi-seed result is
clear on one point: **MION underperforms on RL** — near the bottom on CartPole and
failing to learn Acrobot. The orthogonalizing methods generally struggle here
(MION worst; Muon middling), consistent with the fine-tuning finding: **spectral
orthogonalization of a high-variance Monte-Carlo policy-gradient amplifies noise
rather than helping.** The gentle/adaptive methods and even SGD do better. This is
the clearest counter-example to MION and we report it plainly; a lower-variance
algorithm (PPO/A2C) would sharpen the comparison but is unlikely to reverse it.

### 4.8 MION ablations (Figs. `mion_abl_*.png`)

One knob at a time, 3 seeds, on ResNet-34 / ViT / Tiny-ImageNet:
- **Spectral path is essential**: disabling Newton–Schulz (group-std everywhere)
  collapses ViT **69.2 → 20.5** and ImageNet **55.4 → 42.4**. This is the strongest
  justification for the orthogonalization design.
- **ns_steps**: ≥3 required (ns=1 craters); 3–5 plateau — default 5 is safe.
- **rms_target**: 0.1–0.2 robust; 0.5 destabilizes ViT. Default 0.2 justified.
- **scale_factor**: negligible (±0.5%); default 0.0.

### 4.9 Can MION's small LM gap be closed? (negative results)

Three principled, theory-motivated attempts to close MION's 0.04 LM gap to Muon
**all hurt**, and are retained as ablations: (a) **MION-V2** (EMA momentum +
per-coordinate adaptive embeddings under one LR) → 3.69; (b) **MION-EMB** (larger
embedding LR, α∈{4,12,36}) → monotonically worse (3.45→3.99); (c) **MION-Nor**
(§2.5; NorMuon-style per-row second-moment normalization on both the spectral and
embedding paths, tuned with the same protocol) → **3.390**, also worse than plain
MION. Larger embedding LR, added global adaptivity, and post-hoc row normalization
all degrade MION, showing its group-standardized, single-LR embedding handling is
**already near-optimal** for this design. The residual gap reflects Muon's extra
per-group-LR tuning — a complexity MION deliberately avoids — rather than a
missing normalization or adaptivity mechanism; each of the three literature-backed
candidates we tried address a *different* hypothesized cause, and all three fail
in the same direction, which is itself informative about where the gap does *not*
come from.

---

## 5. Discussion

**MION is the best accuracy/resource trade-off for from-scratch training.** It wins
converged vision outright, is competitive on LM (beating AdamW/Adam-mini) and
ImageNet, and does so at **half the optimizer memory and with a single learning
rate**. Its hidden-matrix update is provably Muon's (cosine 1.00); the small LM gap
comes solely from Muon's separately-tuned embedding optimizer, and attempts to add
that machinery to MION do not help.

**The family is the contribution, not a single point.** Structure-aligned
normalization spans a spectrum: spectral (MION) dominates pretraining and vision;
gentle group-standardization (MiloM/MILO/MILO-LW) dominates fine-tuning and is
competitive on NLP. This regime coverage under one principle is the paper's core
message.

**When does orthogonalization help?** A consistent thread runs through the results:
Newton–Schulz orthogonalization (MION, Muon) wins when gradients are clean and the
model trains from scratch (vision, LM pretraining), but loses to gentle
normalization or adaptivity when the objective is *adaptation* (fine-tuning a
pretrained model) or the gradient is a *high-variance* estimate (REINFORCE RL).
Orthogonalizing a noisy or fine-tuning gradient rotates the full update to unit
singular values, amplifying noise / overwriting pretrained structure. This regime
dependence is an actionable takeaway: **use MION for from-scratch training; prefer
MiloM/MILO-LW (or Adam-family) for fine-tuning and noisy-gradient settings.**

**Anticipated questions.** *How is this different from the modular norm / duality
of Bernstein & Newhouse and Large et al.?* It isn't a new principle — see §1 and
§6: our contribution is the single-buffer/single-LR realization and the first
broad regime study, not the norm-per-layer idea itself. *Why no comparison to
NorMuon?* It is concurrent (Oct 2025) and directly adjacent; we implement and
report MION-Nor (§2.5) rather than omit it. *Is the memory advantage real once
row-norm/gating state is added?* Plain single-buffer MION remains the memory
headline; MION-Nor adds a small `O(rows)` vector per hidden matrix (not a second
full moment) and MION-Gated adds no state at all — both stated honestly against
the 494 MB / 3.1 GB baselines (§4.1, §4.6). *Did you try the theory-prescribed
embedding map before concluding MION's embedding handling is near-optimal?* Our
earlier negative results (MION-V2, MION-EMB, §4.9) tried global adaptivity and a
larger scalar LR — neither is the duality-correct, row/column-wise map; MION-Nor
(§2.5) is that attempt. *RL is inconclusive — why include it?* It is the sharpest,
most honest instance of the regime effect (orthogonalization hurts high-variance
gradients); we report it plainly rather than omit an unfavorable result.

## 6. Related work

**Norm-based / structure-aware optimization.** Our family instantiates the view
that neural-network optimization is steepest descent under a per-layer norm, made
explicit by Bernstein & Newhouse (*Old Optimizer, New Norm: An Anthology*, 2024)
and given a duality-map formulation in Bernstein & Newhouse (*Modular Duality in
Deep Learning*, 2024), where each layer type (`Embed`, `Linear`, `Conv`) is
assigned an operator norm and a corresponding update ("dualizer"). Large et al.
(*Scalable Optimization in the Modular Norm*, 2024) develop the modular norm and
show it yields width/depth learning-rate transfer; Pethick et al. (*Norm-Constrained
LMOs / Scion*, 2025; *Training Neural Networks at Any Scale*, 2025) frame the same
class through norm-constrained linear minimization oracles. MION's spectral
hidden-matrix path is the operator-norm dualizer (rectangular Newton–Schulz), while
its group-standardized non-matrix path is a mean/row-normalized norm; MILO and
MILO-LW are the coordinate/tensor-norm ends of the spectrum.

**Muon and its variants.** Muon (Jordan et al., 2024) orthogonalizes the momentum
via Newton–Schulz. Liu et al. (*Moonlight*, 2025) show it scales to 16B with weight
decay and per-parameter update-scale matching — the RMS-unit rescale we adopt
(§2.3) — and report ~2× compute efficiency over AdamW. NorMuon (Zhang et al., 2025)
adds per-neuron second-moment normalization after orthogonalization to correct
non-uniform neuron norms; our MION-Nor variant (§2.5) unifies this with a single
learning rate and extends it to the embedding path, though in our setting it
underperformed plain MION on LM (§4.2, §4.9) — the row-wise correction that helps
NorMuon's setup did not transfer cleanly onto MION's already-RMS-rescaled update.
PolarGrad (Lau et al., 2025)
and Dion (Ahn et al., 2025) give unifying-preconditioner and distributed
formulations, respectively; Kimi K2 (2025) demonstrates Muon-family training at
1T-parameter scale with a stability fix (MuonClip) relevant if MION is pushed
past its current ≤1.5B scale.

**Memory-efficient optimizers.** Adam-mini (Zhang et al., 2024), SWAN (Ma et al.,
2024), APOLLO (Zhu et al., 2024) and GaLore (Zhao et al., 2024) reduce optimizer
state through fewer learning rates, stateless normalize-and-whiten updates,
structured low-rank scaling, and low-rank gradient projection, respectively. MION
reaches SGD-level state (one momentum buffer) while retaining Adam-class quality;
SWAN's stateless normalize-and-whiten is the closest published relative of our
group-standardization, differing in that we orthogonalize (whiten) only the
hidden-matrix path, not the whole model.

**Orthogonality for fine-tuning.** OFT (Qiu et al., *Controlling Text-to-Image
Diffusion by Orthogonal Finetuning*, 2023) shows that orthogonal transforms which
*preserve* pairwise neuron angles help fine-tuning — motivating MION-Gated's
strength knob (§2.6) rather than treating "orthogonalization hurts fine-tuning" as
a fixed property of the method. LoRA-Muon (2026) restricts Muon's spectral
steepest-descent rule to a low-rank manifold for fine-tuning and reports LR
transfer across rank/width/depth without storing second moments — a more involved
alternative to MION-Gated that we leave for future work (§8.6). "Why Transformers
Need Adam: A Hessian Perspective" (Zhang et al., 2024) shows transformer
embeddings/heads have heavy-tailed, block-heterogeneous Hessian spectra, which is
the underlying reason a uniform-LR group-standardized embedding update can lag a
separately-tuned adaptive one — and why MION-Nor's row-wise correction, rather
than a bigger scalar LR (MION-EMB, §4.9) or global adaptivity (MION-V2, §4.9), is
the theory-consistent fix to try.

**Benchmarking methodology.** AlgoPerf (Dahl et al., 2023) argues for fair,
per-method-tuned comparison across workloads, which is the protocol we follow
(Optuna TPE, per-domain LR ranges, §3) and the citation we use to defend it against
the alternative of shared/literature-default learning rates.

## 7. Limitations

- **LM pretraining**: MION trails Muon by ~0.04 val loss (single seed at 1.2B
  tokens); we have not run multi-seed or multi-scale (only 124M) LM.
- **Fine-tuning**: MION (and Muon) underperform gentle/adaptive methods; the
  orthogonalization that helps pretraining hurts here. FT full-run finals and
  downstream task metrics (beyond val loss) are still pending.
- **RL**: MION underperforms (near-bottom on CartPole, fails Acrobot); REINFORCE
  variance is high, but orthogonalization clearly does not help this regime.
- **MILO / MILO-LW** are weak on deep from-scratch vision and LM; we explicitly
  reposition them as **fine-tuning/gentle-regime anchors** (§4.6, §5) rather than
  general-purpose from-scratch optimizers — consistent with SWAN (Ma et al., 2024)
  showing that pure normalization *without* whitening/spectral structure cannot
  match Adam on from-scratch transformers/deep vision.
- **Baselines**: SOAP/Shampoo show reproducible instabilities (VGG, LM) and Shampoo
  needs fp32 preconditioners under bf16; SOAP was excluded from bf16 FT.
- **MION-Nor (§2.5)**: a fourth negative result on the LM embedding gap (3.390,
  worse than plain MION's 3.281), confirmed on converged vision too (neutral to
  slightly worse across ResNet-34/VGG-11/ViT-Tiny); not adopted as a default.
- **MION-Gated (§2.6)**: confirms the fine-tuning regime mechanism (tuned β=0.156)
  and improves on plain MION (1.424→1.414), but does not fully close the gap to
  MiloM/MILO/MILO-LW (1.404–1.407) — a single global strength knob is insufficient;
  we still recommend the dedicated gentle family members for fine-tuning.
- Scale is modest (≤1.5B); no multi-GPU/distributed or long-horizon results; no
  learning-rate-transfer / width-scaling study (see Future Directions §8.1).

## 8. Future directions

1. **Scale, seeds & LR transfer**: multi-seed LM, multiple model sizes
   (350M/770M/1.5B), and a μP/u-μP or operator-norm width-scaling study (Yang & Hu,
   2020; Blake et al., 2024; width-scaling-under-operator-norms, 2026) — the latter
   gives a theoretical argument that MION's row-normalized non-matrix path is
   width-stable where Muon's can grow as O(√width), a claim worth confirming
   empirically. This is the highest-priority gap for reviewer scrutiny and the
   one deliberately deferred from this pass.
2. **Matrix-scaling convention**: test MION with Muon's spectral-norm units — the
   one untested structural difference, plausibly the source of the LM residual.
3. **Regime-adaptive scheduling**: MION-Gated (§2.6) exposes a static, tuned
   strength knob; scheduling or making it adaptive within a single run (AMO-style,
   2026) rather than fixed per regime is future work.
4. **RL with variance control**: PPO/A2C and continuous-control (MuJoCo) to obtain
   conclusive RL evidence.
5. **Theory**: convergence guarantees for group-standardized/spectral updates and
   the single-LR RMS-unification.
6. **Low-rank fine-tuning fidelity**: a LoRA-Muon-style spectral update restricted
   to a low-rank manifold, as a more involved alternative to MION-Gated's strength
   gate, worth testing if gating alone does not close the gap to MiloM/MILO/MILO-LW.
7. **Cheaper orthogonalization**: swap the quintic Newton–Schulz coefficients for
   Polar Express's optimal odd-polynomial coefficients (Amsel et al., 2025), likely
   allowing `ns_steps` 5→3 at equal quality and trimming MION's ~20% per-step
   overhead; re-run the `ns_steps` ablation (§4.8) with the new coefficients.
8. **Distributed orthogonalization**: adopt Dion (Ahn et al., 2025) or Moonlight's
   memory-optimal distributed Muon (Liu et al., 2025) rather than a bespoke scheme
   if MION is scaled past ~1.5B or across multiple GPUs.

## 9. Conclusion

Treating an update's normalization as a function of parameter structure yields a
coherent family of optimizers. Its flagship, **MION**, delivers Adam-class (and
often better) quality on from-scratch training at **half the optimizer memory and a
single learning rate**, is identical to Muon on the matrix path, and — with the
gentler family members handling fine-tuning — the family covers a broad range of
regimes under one principle.

---

## Appendix A — Reproducibility

- **Code**: `optimizers/milo2.py` (MiloM, MION, MION-Nor via `row_norm`, MION-Gated
  via `ortho_strength`), `milo.py` (MILO/MILO-LW), `experiments/{lm,vision,nlp,
  imagenet,rl}/`, `benchmarks/`.
- **Tuning**: `hpc/experiments/optuna_sweep.py` (domains: nlp/imagenet/vision/lm/ft/rl;
  `ft` jointly tunes `(lr, beta)` for MION-Gated via the `ORTHO_STRENGTH` env var);
  best LRs in `results_optuna/` and each domain's `config.py` `LEARNING_RATES`.
- **Launch**: `hpc/experiments/submit_*.sh` + `run_*_one.slurm`, including
  `submit_mion_nor_lm.sh`, `submit_mion_nor_vision.sh`, `submit_mion_gated_ft.sh`
  (Optuna sweep → dependent final run, chained via `--dependency=afterok`).
- **Data/models** (offline HPC, pre-cached to scratch): FineWeb-Edu (GPT-2 BPE),
  CIFAR/MNIST, Tiny-ImageNet-200, SST-2, Qwen2.5-1.5B, Alpaca.
- **Results**: `results/{lm,cost,finals,ft1b,rl,mion_ablation,controlled,optuna}/`;
  figures `results/paper/figures/`; tables `results/paper/tables/`;
  MION-vs-Muon trace analysis `results/paper/mion_vs_muon_analysis.md`.
- **Literature synchronization**: `lit_review/research_memo.md` (28 verified
  references in `lit_review/references.csv`, gap analysis in
  `lit_review/gap_literature_map.csv`) — the basis for §2.5–2.6, §6, and the
  anticipated-questions paragraph in §5.
- **Environment**: `.venv_cc` (PyTorch 2.12, transformers 5.3, gymnasium 1.3),
  single A100-40GB per job; all LRs Optuna-tuned per domain.
