# HPC Submission Guide

Three options for running experiments on HPC:

## Option 1: Baseline Only (Recommended First Run)

```bash
bash hpc/submit_all.sh
```

**What runs:**
- Vision baseline (6 models × 11 optimizers × 5 runs = 330 runs)
- ImageNet baseline (1 model × 11 optimizers × 3 runs = 33 runs)
- NLP baseline (BERT × 11 optimizers × 5 runs = 55 runs)
- **All three run in PARALLEL**

**Timeline:** ~24 hours (longest job determines total time)

**When to use:** Initial optimizer comparison, quick iteration

---

## Option 2: Baseline Then Tuning (Full Evaluation)

```bash
bash hpc/submit_baseline_then_tuning.sh
```

**What runs:**
1. **Phase 1 (Parallel):** All baselines from Option 1
2. **Phase 2 (After Phase 1):** Hyperparameter tuning on best performers

**Timeline:**
- Phase 1: ~24 hours
- Phase 2: ~8-12 hours after Phase 1 completes
- **Total: ~32-36 hours**

**When to use:** Finding optimal hyperparameters, publication-quality results

---

## Option 3: Individual Experiments

Submit each experiment separately:

```bash
# Vision only
sbatch hpc/experiments/run_vision_experiments.slurm

# ImageNet only
sbatch hpc/experiments/run_imagenet_experiments.slurm

# NLP only
sbatch hpc/experiments/run_nlp_experiments.slurm

# Tuning only (if baselines already done)
sbatch hpc/experiments/run_hyperparameter_tuning.slurm
```

**When to use:** Rerunning specific domains, debugging, parallel submissions on multiple allocation accounts

---

## Understanding Job Dependencies

In `submit_baseline_then_tuning.sh`, the tuning job uses SLURM dependencies:

```bash
--dependency=afterok:${VISION_JID}:${IMAGENET_JID}:${NLP_JID}
```

This means tuning will NOT start until:
- Vision baseline succeeds (`afterok:`)
- AND ImageNet baseline succeeds
- AND NLP baseline succeeds

If any baseline **fails**, tuning will not run (stays pending). You must:
1. Fix the failed job
2. Resubmit manually: `sbatch hpc/experiments/run_hyperparameter_tuning.slurm`

---

## Monitoring Jobs

```bash
# View all your jobs
squeue -u $(whoami)

# Watch specific job
swatch -j <JOB_ID>

# Tail output in real-time
tail -f logs/vision_experiments_*.out
tail -f logs/imagenet_experiments_*.out
tail -f logs/nlp_experiments_*.out

# Check job details
sinfo
sacct -j <JOB_ID> --format=JobID,JobName,State,Elapsed,TimeLimit
```

---

## Common Patterns

### "I want baseline results fast"
```bash
bash hpc/submit_all.sh
```

### "I want tuning after baselines (fire and forget)"
```bash
bash hpc/submit_baseline_then_tuning.sh
```

### "I'm debugging Vision, submit ImageNet/NLP independently"
```bash
sbatch hpc/experiments/run_imagenet_experiments.slurm
sbatch hpc/experiments/run_nlp_experiments.slurm
# Fix vision code locally, then resubmit
```

### "I want to manually chain experiments (custom logic)"
```bash
# Submit baselines
VISION=$(sbatch hpc/experiments/run_vision_experiments.slurm | awk '{print $NF}')
IMAGENET=$(sbatch hpc/experiments/run_imagenet_experiments.slurm | awk '{print $NF}')
NLP=$(sbatch hpc/experiments/run_nlp_experiments.slurm | awk '{print $NF}')

# Then analysis after all complete
sbatch --dependency=afterok:${VISION}:${IMAGENET}:${NLP} my_analysis_script.slurm
```

---

## Performance Expectations

| Phase | Job | Runtime | GPUs | Memory |
|-------|-----|---------|------|--------|
| Baseline | Vision | ~24h | 2 | 48GB |
| Baseline | ImageNet | ~12h | 2 | 48GB |
| Baseline | NLP | ~22h | 2 | 48GB |
| Tuning | All | ~8-12h | 2 | 48GB |

**Notes:**
- All baselines run in parallel, so total is ~24h (not 24+12+22)
- Tuning is sequential per domain, so total is ~8-12h after baselines
- GPU utilization is high; CPU overhead minimal

---

## Troubleshooting

### Job stuck in PENDING
```bash
squeue -j <JOB_ID> --format=JobID,Name,State,Reason
```

Check `Reason` column. Common issues:
- `QOSMaxGRESPerUser` → Too many GPU jobs; wait for some to finish
- `Dependency` → Waiting for parent job (check `--dependency` chain)
- `Resources` → Insufficient resources available

### Job failed silently
```bash
tail -f logs/<domain>_experiments_<JOB_ID>.err
```

Look for:
- CUDA out of memory
- Dataset not found
- Missing dependencies (transformers, etc.)

### Results missing
```bash
ls -lh experiments/{vision,nlp,imagenet}/results_nt*/
```

Check if results dir exists and contains JSON files.

---

## Performance Tips

1. **Submit during off-peak hours** if possible (allocation less congested)
2. **Run baselines first**, then decide if tuning is needed
3. **Monitor early** — catch errors in first 5 minutes
4. **Use `--dependency` for complex chains** rather than manual resubmission
