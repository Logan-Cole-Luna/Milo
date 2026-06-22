# Milo Experiments

Comprehensive optimizer evaluation suite with modular experiment structure.

## Directory Structure

```
experiments/
├── vision/              # CIFAR-10/100 vision experiments
│   ├── config.py       # Vision experiment configuration
│   ├── vision_experiment.py
│   └── results_nt/     # Vision results (auto-generated)
├── nlp/                # BERT fine-tuning experiments
│   ├── config.py       # NLP experiment configuration
│   ├── nlp_experiment.py
│   └── results_nt_bert/# NLP results (auto-generated)
├── imagenet/           # Large-scale vision (Tiny ImageNet-200)
│   ├── config.py       # ImageNet experiment configuration
│   ├── imagenet_experiment.py
│   └── results_nt_imagenet200/  # ImageNet results (auto-generated)
└── train_utils.py, network.py, etc.  # Shared utilities
```

## Running Experiments

### Local Mode

```bash
# Vision experiments (CIFAR-10/100 with ResNet34, VGG11, ViT-Tiny)
python experiments/vision/vision_experiment.py

# NLP experiments (BERT on SST-2)
python experiments/nlp/nlp_experiment.py

# Large-scale vision (Tiny ImageNet-200)
python experiments/imagenet/imagenet_experiment.py
```

### HPC Mode

```bash
# Submit all experiments in parallel
bash hpc/submit_all.sh

# Or submit individually
sbatch hpc/experiments/run_vision_experiments.slurm
sbatch hpc/experiments/run_nlp_experiments.slurm
sbatch hpc/experiments/run_imagenet_experiments.slurm
```

## Configuration

Each experiment has its own `config.py`:

- **Vision** (`experiments/vision/config.py`)
  - 8 experiments (Logistic, MLP, ResNet34×2, VGG11×2, ViT-Tiny×2)
  - 11 optimizers
  - 5 runs per optimizer

- **NLP** (`experiments/nlp/config.py`)
  - BERT-base on SST-2 sentiment classification
  - 11 optimizers
  - 5 runs per optimizer

- **ImageNet** (`experiments/imagenet/config.py`)
  - Tiny ImageNet-200 (64×64, 200 classes)
  - ResNet34
  - 11 optimizers
  - 3 runs per optimizer

## Optimizers Evaluated

All 11 optimizers across all experiments:
- **MILO** (network-wide, scale-aware)
- **MILO_LW** (layer-wise, scale-aware)
- **SGD** (baseline)
- **AdamW** (baseline)
- **AdaGrad** (baseline)
- **Lion** (modern, sign-based)
- **Adam-mini** (parameter-efficient)
- **RMSprop-Momentum** (adaptive)
- **Shampoo** (second-order)
- **SOAP** (preconditioned)
- **Muon** (distributed-aware)

## Results

Results are saved in JSON format with per-optimizer files:
```
experiments/{domain}/results_{suffix}/{domain}_{config}_{optimizer}.json
```

Each result contains:
- Run number
- Optimizer name
- Best validation accuracy
- Best validation loss (NLP/ImageNet)
- Training time

## Modifying Configuration

To change hyperparameters for any experiment:

1. Edit the corresponding `config.py` (e.g., `experiments/vision/config.py`)
2. Update `LEARNING_RATES`, `OPTIMIZER_PARAMS`, or training parameters
3. Re-run the experiment script

Example:
```python
# experiments/vision/config.py
LEARNING_RATES = {
    "MILO": 0.001,      # Change as needed
    "ADAMW": 0.0001,    # per optimizer
    ...
}
```

## Notes

- **Offline HPC**: Models and datasets must be pre-downloaded on login node
- **GPU Memory**: Vision requires 48GB RAM, NLP/ImageNet require 48GB RAM
- **Reproducibility**: All experiments use fixed seeds (seed=42)
