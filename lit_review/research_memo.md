# Research Memo: Strengthening the MILO / MION Optimizer Family

**To:** MILO/MION authors · **Re:** Literature synchronization + a prioritized path to "outperform in every domain"
**Scope:** 28 verified references (see `references.csv`); gap-to-literature matrix (see `gap_literature_map.csv`).
**Bottom line:** Your "structure-aligned update normalization" principle is the empirical instantiation of a fast-moving theoretical program (modular norm / modular duality / steepest descent under a norm). That is *good news for positioning* but it means novelty must be argued carefully. Three published-in-the-last-year papers map onto your three biggest gaps so directly that they are simultaneously your best citations and your most dangerous "why didn't you compare?" reviewer questions: **NorMuon** (per-neuron normalization *after* orthogonalization), **LoRA-Muon** (spectral descent on a low-rank manifold for fine-tuning), and the **operator-norm width-scaling** analysis (LR transfer, and a theoretical *advantage* for your row-standardized design). Prioritize those three.

*All arXiv IDs below were verified against the live arXiv API. Where I quote a mechanism, it is from the paper's abstract/text, not memory. Claims I could not verify are flagged.*

---

## 1. How the field frames what you built (positioning)

Your one-sentence principle — *normalize each parameter's update according to the structure of that parameter* — is, in the current theory, **steepest descent under a per-layer norm**. Two lines of work say this explicitly:

- **Bernstein & Newhouse, "Old Optimizer, New Norm: An Anthology" (2409.20325, 2024)** recast Adam, Shampoo and Muon as steepest descent under *different norms*, with the choice of norm tied to each parameter's role. This is your spectrum (generic grouping → per-row → spectral) stated as theory: MILO ≈ a coordinate/RMS norm, MiloM ≈ a row-wise norm, MION ≈ the spectral (operator) norm on hidden matrices.
- **Bernstein & Newhouse, "Modular Duality in Deep Learning" (2410.21265, 2024)** goes further and *derives* the correct update map ("dualizer") per layer type from the layer's operator norm — and, critically, gives **separate GPU-friendly dualizers for `Embed` vs `Linear`/`Conv` layers** (the latter via rectangular Newton–Schulz). This is the single most important paper for your framing, because it says the embedding path *should* use a different map than the hidden matrices — which is exactly where your MION↔Muon gap lives.
- **Large, Liu, Bernstein et al., "Scalable Optimization in the Modular Norm" (2405.14813, 2024)** assigns each module a norm, normalizes updates to it, and gets width/depth **LR transfer** as a consequence — the rigorous version of your "single learning rate" claim.
- **Pethick et al., "Norm-Constrained LMOs / Scion" (2502.07529, 2025)** and **"Training Neural Networks at Any Scale" (2511.11163, 2025)** provide the linear-minimization-oracle framing and a recent synthesis to cite as the umbrella.

**Recommended framing for the paper.** Do *not* present structure-aligned normalization as a brand-new principle — a reviewer who knows the modular-norm line will penalize that. Instead: **"We give a practical, memory-light instantiation of steepest-descent-under-a-per-layer-norm (the modular-norm/duality program), realized with a single momentum buffer and a single learning rate, and we run the first broad regime study (from-scratch vs. fine-tuning) across five paradigms."** Your defensible contributions become: (1) the **single-buffer/single-LR realization** and its memory result; (2) the **empirical family + regime finding** (spectral helps pretraining, hurts FT); (3) the **fair, per-method-tuned multi-domain evaluation**. Cite the theory as your foundation and own the empirics. This is stronger, not weaker, than a novelty claim you'd have to defend against 2409.20325.

---

## 2. The three highest-leverage moves (do these first)

### 2.1 MION-Nor: add NorMuon's per-neuron second moment on top of the spectral path

**Gap:** G1 (LM embedding gap) + G8 (MiloM is dominated / unclear role).

**NorMuon (Zhang et al., 2510.05491, Oct 2025)** makes an observation that directly concerns you: Muon reduces the update's condition number, but *the orthogonalized updates then have highly non-uniform per-neuron (row) norms, so a few neurons dominate*. NorMuon fixes this by **maintaining a per-neuron second-moment statistic and applying row-wise normalization *after* orthogonalization**, and reports outperforming *both* Adam and Muon (they cite a ~21.7% efficiency gain over Adam).

Why this matters for you specifically:
- **MiloM already does per-row standardization — but *without* orthogonalization**, which the ablations show is the weak half (MiloM 3.494 vs MION 3.281 on LM). NorMuon says the right place for row-normalization is *after* the spectral step, not instead of it.
- Fusing them — **MION-Nor = Newton–Schulz orthogonalize, then row-wise normalize by a per-neuron second moment** — is a small code change on your existing MION and MiloM paths, and it is the most likely single change to close part of the 0.043 LM gap to Muon *and* to beat Muon (which NorMuon claims to do). It also retro-explains MiloM as the natural "row-norm without spectral" ablation.
- Cost: adds one more buffer on hidden matrices (a per-row/neuron vector, not a full moment — much cheaper than Adam's second moment). You'd move from "one buffer" to "one buffer + O(rows) per matrix," which is still far below Adam and arguably still worth the single-LR/memory story. Report it as an *optional* family member so the single-buffer MION headline survives.

**Action:** Implement MION-Nor; re-run the 124M LM and ViT/ImageNet sweeps. Expected: closes much of the LM gap; strengthens vision. Cite NorMuon as concurrent work you extend (single-LR, and fused with your embedding handling).

### 2.2 A muP / operator-norm LR-transfer study — closes your single biggest reviewer weakness

**Gap:** G3 (no scaling law, single seed, single 124M scale). For an AAAI optimizer paper this is the weakness most likely to be raised.

The tools now exist and one of them hands you a **theoretical advantage**:

- **"On the Width Scaling of Neural Optimizers Under Matrix Operator Norms" (2603.09952, 2026)** interprets AdamW and Muon as steepest descent under operator norms and proves that **standard operator-norm rules (Muon) can suffer O(√w) worst-case growth of the smoothness constant with width, whereas *row-normalized* / mean-normalized optimizers are width-stable and recover μP scaling as a special case.** Your non-matrix path is row-standardized; this is a citable argument that **MION's design transfers across width better than Muon's**. That flips a limitation into a selling point.
- **μP (Yang & Hu, 2011.14522, 2020)** and the cleaner **u-μP (Blake et al., 2407.17465, 2024)** give the parameterization for zero-shot LR transfer across width. u-μP's unit-scaling philosophy pairs naturally with your RMS-unit updates.
- **Moonlight / "Muon is Scalable" (2502.16982, 2025)** is the protocol template: they show Muon's ~2× compute efficiency via *scaling-law experiments* and identify **weight decay + per-parameter update-scale matching** as the two things needed to scale — and their update-scale matching is *exactly your RMS-unit rescale* (`×√max(m,n)`). Cite this to validate §2.3 and to justify checking your weight-decay coupling at scale.

**Concrete experiment design (this is Future Direction #1 made credible):**
1. Adopt **u-μP** (or the mean-normalized operator-norm parameterization from 2603.09952) for MION.
2. Train **3 widths** (e.g. d_model 256/512/768, ~40M/124M/350M) × **3 seeds**, sweeping LR per width at the smallest width only.
3. **Claim to demonstrate:** the LR that is optimal at 40M stays within the optimal basin at 350M (LR transfer), and that MION's transfer is *tighter* than Muon's (predicted by the O(√w) result). Plot val-loss-vs-LR curves overlaid across widths (the standard μP "aligned minima" figure) and a token-efficiency scaling curve vs Muon/AdamW.
4. Report multi-seed error bars on the 124M LM headline — this alone answers the "single seed" limitation.

This is the highest-value *new experiment* in the memo: it converts your weakest section into a theorem-backed strength and gives the paper a scaling story that AAAI reviewers now expect for optimizer work.

### 2.3 MION-FT: gated / low-rank orthogonalization for the fine-tuning regime

**Gap:** G2 (orthogonalization hurts FT) — your Future Direction #6, which the literature has partly done for you.

Your regime finding (spectral helps pretraining, hurts FT because it "rotates away pretrained feature structure") is corroborated and *actionable*:

- **OFT — "Controlling Text-to-Image Diffusion by Orthogonal Finetuning" (Qiu et al., 2306.07280, 2023)** shows the *right* orthogonality **helps** fine-tuning: OFT applies orthogonal transforms that **provably preserve hyperspherical energy (pairwise neuron angles)**, and this preservation is what retains pretrained semantics. Your MION rotates in weight space and overwrites features; OFT preserves the *relational* structure. This reframes your FT failure precisely: it is not "orthogonalization is bad for FT," it is "MION uses feature-*overwriting* orthogonalization instead of feature-*preserving* orthogonalization."
- **LoRA-Muon (2606.12921, 2026)** is your Future Direction #6 already realized: it applies **Muon's spectral steepest-descent rule restricted to a low-rank manifold**, reports that **optimal LRs transfer across rank/width/depth**, that a rank-32 run **beat the dense baseline** in their seed-averaged sweep, and that it **avoids storing second moments** (memory-friendly). This is the low-rank spectral update you proposed, with evidence it works. You can either build on it or, at minimum, must cite and compare.
- **"How Much Orthogonalization Does Muon Need?" (2606.00371, 2026)** shows **training quality is *not* monotone in polar-decomposition accuracy** — a cheaper/partial orthogonalization matches full NS. This is the theoretical license for a **gated/tunable-strength spectral update**: interpolate between full orthogonalization and gentle group-std by a single strength knob `β ∈ [0,1]`, `u = β·orthogonalize(d) + (1−β)·standardize(d)`. Set `β→1` for pretraining, `β→0` (or low) for fine-tuning — and, per **AMO (2605.17806, 2026)**, make `β` *adaptive* to recover your "regime-adaptive family" (Future Direction #3) automatically.

**Action:** Two variants worth testing on the Qwen2.5-1.5B FT benchmark: (a) **MION-gated** (strength knob β, possibly scheduled/adaptive); (b) **MION-LR** (low-rank spectral update, LoRA-Muon style). Either could give you a *single* family member that wins both regimes — which is a stronger story than "use MION for pretrain, MILO for FT."

---

## 3. Secondary moves (cheap wins and rigor)

### 3.1 Swap the Newton–Schulz polynomial (free ~speed/quality win) — Gap G5
Your quintic NS with `ns_steps=5` can likely be replaced by better-conditioned polynomials:
- **"The Polar Express: Optimal Matrix Sign Methods for Muon" (2505.16932, 2025)** gives *optimal* odd-polynomial coefficients for the matrix-sign/orthogonalization problem — faster convergence and higher accuracy than classic quintic NS.
- **"Accelerating Newton–Schulz via Chebyshev-type Polynomials" (2506.10935, 2025)** is an alternative acceleration.
- **"Iterative Orthogonalization Scaling Laws" (2505.04005, 2025)** tells you how NS-step count should grow with matrix size — directly informs your `ns_steps` ablation at larger scale.

Given 2606.00371's finding that final loss is insensitive to polar accuracy, the realistic win is **reducing `ns_steps` (e.g. 5→3) with Polar Express coefficients at equal quality**, trimming your ~20% per-step overhead. Re-run the `ns_steps` ablation with the new coefficients.

### 3.2 Distributed / larger-scale orthogonalization — Gap (scale limitation)
- **Dion (2504.05295, 2025)** and Moonlight's open-sourced **memory-optimal, communication-efficient distributed Muon** (2502.16982) both solve the "no multi-GPU/distributed results" limitation. If you attempt any run >1.5B or multi-GPU, adopt one of these orthogonalization-distribution schemes rather than rolling your own.

### 3.3 Rescue or re-scope MILO / MILO-LW — Gap G4
- **SWAN (2412.13148, 2024)** shows a *stateless* optimizer using **normalization + whitening** can match Adam from scratch — but note it *whitens* (a spectral operation), it doesn't merely standardize. This explains why MILO's generic √N grouping collapses on from-scratch transformers/deep vision: there's no whitening/spectral structure on the hidden matrices. Two honest options: (a) **re-scope** MILO/MILO-LW explicitly as fine-tuning/gentle-regime members (they already win NLP and FT), or (b) add SWAN-style whitening to MILO's matrix path — but that pushes it toward MION and blurs the family. I recommend (a): present MILO/MILO-LW as the FT-regime anchors, MION-Nor as the from-scratch anchor.

### 3.4 RL — Gap G6
Follow your own Future Direction #4: move to **PPO/A2C** (lower variance than REINFORCE) or relegate RL to an appendix. **AlgoPerf (Dahl et al., 2306.07179, 2023)** is the methodology citation for "high-variance workloads need many seeds / a different protocol." Don't let a null result dilute the headline.

### 3.5 Why the embedding path is special (theory for G1)
**"Why Transformers Need Adam: A Hessian Perspective" (2402.16788, 2024)** shows transformer blocks (notably embeddings/heads) have **heavy-tailed, block-heterogeneous Hessian spectra**, which is why they benefit from per-block adaptive LRs. This *explains* why Muon routes embeddings to a separate AdamW and why a uniform-LR group-std can lag there. Combined with **Modular Duality's separate `Embed` dualizer (2410.21265)**, the prescription for closing G1 is not "bigger embedding LR" (your MION-EMB, which failed) or "global adaptivity" (MION-V2, which failed) but the **duality-correct embedding map**: a row/column-wise normalization matched to the embedding's operator-norm semantics, optionally with a *per-row* second moment (NorMuon-style) confined to the embedding. That is a specific, theory-grounded thing to try that you have not yet tried.

---

## 4. Candidate embedding-update rules to test (closing G1, concretely)

Your MION-V2 and MION-EMB failures share a diagnosis: they added *global adaptivity* or *a bigger scalar LR*, neither of which is what the duality theory prescribes for an embedding. The theory (2410.21265) and the heavy-tail Hessian result (2402.16788) point instead to **structure-matched, per-row/column** maps. Ranked by expected payoff:

1. **NorMuon-style per-row second moment on the embedding only** — keep group-std direction, but divide by a running per-row RMS of the update. Single LR preserved; adds an O(vocab) or O(d_model) vector. Most likely to help.
2. **Column-wise (per-token) max-norm / RMS normalization** — embeddings are indexed per token; normalize each token's update vector to unit RMS. This is closer to the duality `Embed` dualizer than row-of-matrix standardization.
3. **Sign-descent / max-norm on the embedding** — the duality map under an ℓ∞-flavored embedding norm reduces toward sign-like updates; cheap to test as a bracket on the design space.
4. **Grafting the embedding direction from a tiny auxiliary second moment** while keeping MION's global LR — a middle point between your (failed) full-adaptive V2 and pure group-std.

For each, keep the **single global LR** so the headline survives, and report the LM val loss vs. Muon. The claim you want: "a duality-matched embedding map closes the gap under one LR, whereas naive adaptivity/LR-scaling (V2/EMB) does not" — which turns your negative results into a *positive, explained* finding.

---

## 5. Prioritized recommendation table

| # | Move | Gap(s) | Effort | Expected payoff | Key refs |
|---|------|--------|--------|-----------------|----------|
| 1 | **MION-Nor**: per-neuron 2nd moment after NS | G1, G8 | Low (code on existing paths) | High — may beat Muon on LM & vision; clarifies MiloM | NorMuon 2510.05491 |
| 2 | **μP / u-μP + operator-norm LR-transfer study** (3 widths × 3 seeds) | G3 | Medium (compute) | Very high — fixes the #1 reviewer weakness; theorem-backed transfer edge | 2011.14522, 2407.17465, 2603.09952, 2502.16982 |
| 3 | **MION-FT**: gated/adaptive-strength or low-rank spectral | G2 | Medium | High — a single member that wins both regimes | OFT 2306.07280, LoRA-Muon 2606.12921, 2606.00371, AMO 2605.17806 |
| 4 | **Duality-matched embedding rule** (§4 candidates) | G1 | Low–Med | Med–High — closes the residual LM gap, explains V2/EMB failures | 2410.21265, 2402.16788 |
| 5 | **Polar Express NS coefficients**, re-run ns_steps ablation | G5 | Low | Med — trims ~20% overhead at equal quality | 2505.16932, 2506.10935, 2505.04005 |
| 6 | **Reposition MILO/MILO-LW as FT-regime anchors** | G4, G7 | Trivial (framing) | Med — coherent family story | SWAN 2412.13148 |
| 7 | **Distributed orthogonalization** for any >1.5B run | scale | Med | Med — removes scale limitation | Dion 2504.05295, Moonlight 2502.16982 |
| 8 | **PPO/A2C RL** or move RL to appendix | G6 | Low | Low — protects headline | AlgoPerf 2306.07179 |
| 9 | **Reframe as modular-norm/duality instantiation** | G7 | Trivial | High — novelty defensibility | 2409.20325, 2410.21265, 2405.14813 |

If you do only three: **#1 (MION-Nor), #2 (μP transfer study), #3 (MION-FT).** These attack the three gaps a reviewer will fixate on and each has published evidence that the mechanism works.

---

## 6. Related-work paragraphs (drop-in drafts)

**Norm-based / structure-aware optimization.** *Our family instantiates the view that neural-network optimization is steepest descent under a per-layer norm, made explicit by Bernstein & Newhouse (2024a) and given a duality-map formulation in Bernstein & Newhouse (2024b), where each layer type (Embed, Linear, Conv) is assigned an operator norm and a corresponding update ("dualizer"). Large et al. (2024) develop the modular norm and show it yields width/depth learning-rate transfer, and Pethick et al. (2025) frame the same class through norm-constrained linear minimization oracles. MION's spectral hidden-matrix path is the operator-norm dualizer (rectangular Newton–Schulz), while its group-standardized non-matrix path is a mean/row-normalized norm; MILO and MILO-LW are the coordinate/tensor-norm ends of the spectrum.*

**Muon and its variants.** *Muon (Jordan et al., 2024) orthogonalizes the momentum via Newton–Schulz. Liu et al. (2025, Moonlight) show it scales to 16B with weight decay and per-parameter update-scale matching — the RMS-unit rescale we adopt — and report ~2× compute efficiency over AdamW. NorMuon (Zhang et al., 2025) adds per-neuron second-moment normalization after orthogonalization to correct non-uniform neuron norms; our MION-Nor variant unifies this with a single learning rate. PolarGrad (Lau et al., 2025) and Dion (Ahn et al., 2025) give unifying-preconditioner and distributed formulations respectively.*

**Memory-efficient optimizers.** *Adam-mini (Zhang et al., 2024), SWAN (Ma et al., 2024), APOLLO (Zhu et al., 2024) and GaLore (Zhao et al., 2024) reduce optimizer state. MION reaches SGD-level state (one momentum buffer) while retaining Adam-class quality; SWAN's stateless normalize-and-whiten is the closest relative to our group-standardization, differing in that we orthogonalize (whiten) only the hidden-matrix path.*

---

## 7. What an AAAI reviewer will ask (anticipate these)

1. *"How is 'structure-aligned normalization' different from the modular norm / duality of Bernstein & Newhouse and Large et al.?"* — Answer with the §1 framing: you're the memory-light single-buffer instantiation + the first broad regime study. Have this in the intro, not the rebuttal.
2. *"Why no comparison to NorMuon?"* — It's concurrent (Oct 2025) and directly adjacent. Add MION-Nor and cite it, or you will be asked.
3. *"Single seed, single scale on your headline LM claim — does the LR transfer?"* — The μP study (#2) is the answer. Without it, the LM headline is fragile.
4. *"Your negative results (V2/EMB) — did you try the theory-prescribed embedding map?"* — §4. Turn the negatives into an explained positive.
5. *"Is the memory advantage real once you add the NorMuon buffer / auxiliary state?"* — Keep single-buffer MION as the headline; present MION-Nor as an optional accuracy-max variant with its (modest) added state stated honestly.
6. *"RL is inconclusive — why include it?"* — Either strengthen (PPO) or appendix it.

---

## 8. Reference artifacts

- **`references.csv`** — 28 verified references (title, authors, arXiv ID, year, thread, relevance), all checked against the live arXiv API.
- **`gap_literature_map.csv`** — 8 gaps × {reviewer-risk, key papers, mechanism of relevance, concrete lever}.

*Verification note: every arXiv ID here was confirmed to resolve to the stated title via the arXiv API. Mechanistic claims for the four load-bearing papers (NorMuon 2510.05491, Moonlight 2502.16982, LoRA-Muon 2606.12921, operator-norm width scaling 2603.09952) and for Modular Duality (2410.21265), OFT (2306.07280) and "How much orthogonalization" (2606.00371) are drawn from their abstracts/text, not from memory. A handful of 2026-dated IDs (AMO 2605.17806, LoRA-Muon 2606.12921, "How much orthogonalization" 2606.00371, width-scaling 2603.09952, Chebyshev NS 2506.10935) are very recent preprints — treat their specific numeric claims as preliminary until you read the PDFs.*
