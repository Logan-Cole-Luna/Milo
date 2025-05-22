\
# Milo: A Magnitude-Invariant Learning Optimizer with Group-Wise Gradient Normalization

This repository provides the official implementation and experimental code for "Milo: A Magnitude-Invariant Learning Optimizer with Group-Wise Gradient Normalization". MILO is a novel optimization algorithm designed to address the challenges of gradient scale variability in deep learning.

## Installation

```bash
# Clone the repository
git clone https://github.com/username/Milo
cd Milo

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

Requirements:
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

For reproducibility, we recommend creating a conda environment:

```bash
conda create -n milo python=3.10
conda activate milo
pip install -r requirements.txt
```

### Repository Structure

```
Milo/
├── milo.py                  # Core MILO optimizer implementation
├── experiments/             # Experimental code and configurations
│   ├── supervised_learning/ # Vision and other supervised learning tasks
│   │   ├── supervised_learning_experiment.py
│   │   └── config.py
│   ├── ablation_study/      # Ablation experiments
│   │   ├── milo_ablation.py
│   │   └── config.py
│   ├── RL/                  # Reinforcement learning experiments
│   │   ├── rl_experiment.py
│   │   └── config.py
│   ├── loss_maps/           # Loss landscape analysis
│   │   ├── loss_map.py
│   │   └── config.py
│   └── train-llm-from-scratch/  # Language model training experiments
│       ├── config/
│       │   └── config.py    # LLM experiment configuration
│       ├── scripts/         # Training and evaluation scripts
│       │   └── train_multi_optimizer.py
│       └── visuals/        # Generated plots and visualizations
├── data/                    # Placeholder for datasets (downloaded automatically)
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Using the MILO Optimizer

MILO can be used as a drop-in replacement for standard PyTorch optimizers:

```python
from milo import milo

# Basic usage
optimizer = milo(model.parameters(), lr=1e-3)

# With recommended settings for many common tasks
optimizer = milo(
    model.parameters(),
    lr=1e-3,
    momentum=0.9,
    weight_decay=1e-4,
    layer_wise=True,    # Enable layer-wise normalization
    scale_aware=True,   # Preserve some gradient scale
    scale_factor=0.2,   # Mixing factor for scale-awareness
    adaptive=True       # Enable adaptive gradient scaling
)
```

### Key Parameters

- `lr` (float, Tensor): Learning rate.
- `momentum` (float): Momentum factor.
- `weight_decay` (float): Weight decay (L2 penalty).
- `normalize` (bool): Enable gradient normalization (default: True).
- `layer_wise` (bool): Group parameters by layer for normalization (default: True). If False, uses `group_size`.
- `group_size` (int, optional): Fixed number of parameters per group when `layer_wise=False`.
- `eps` (float): Small constant for numerical stability in normalization (default: 1e-5).
- `scale_aware` (bool): Preserve some gradient scale information during normalization (default: True).
- `scale_factor` (float): Mix factor for scale-aware normalization (default: 0.2). When `scale_aware=True`, the normalized gradient `g_norm` is mixed with the original gradient `g_orig` as `(1-scale_factor)*g_norm + scale_factor*g_orig`.
- `adaptive` (bool): Use adaptive gradient scaling similar to RMSprop (default: True).
- `clip_norm` (float, optional): Maximum norm for gradient clipping.


## Reproducing Experiments

We provide scripts to reproduce the experiments presented in the paper. Detailed configurations for each experiment type can be found in their respective `config.py` files within the `experiments/` subdirectories.

### Supervised Learning Experiments
To run all supervised learning experiments (e.g., image classification on CIFAR, MNIST):
```bash
python experiments/supervised_learning/supervised_learning_experiment.py
```
To run specific experiments (e.g., ResNet18 and MLP only):
```bash
python experiments/supervised_learning/supervised_learning_experiment.py --experiments RESNET18 MLP
```

### Ablation Studies
To run the full ablation study on MILO's components:
```bash
python experiments/ablation_study/milo_ablation.py
```

### Reinforcement Learning Experiments
To run reinforcement learning experiments (e.g., Point Navigation):
```bash
python experiments/RL/rl_experiment.py
```

### Loss Map Analysis
To generate visualizations of optimizer trajectories on benchmark functions:
```bash
python experiments/loss_maps/loss_map.py
```

### LLM Training Experiments
To run comparative optimizer experiments on transformer language models:
```bash
python experiments/train-llm-from-scratch/scripts/train_multi_optimizer.py
```

Configuration for the LLM experiments can be found in `experiments/LLM/config/config.py`.


### Acknowledgements

Our work utilizes or is inspired by concepts from the following repositories:

1. **[train-llm-from-scratch](https://github.com/FareedKhan-dev/train-llm-from-scratch)** (MIT License)  
  - Citation: \citep{FareedKhandev2025trainLLM}  
  - Adaptations:  
    Our LLM training code was adapted to support multiple optimizer runs for comparative analysis. The script allows customization of training runs with the following arguments:
    - `--optimizers`: List of optimizers to compare (default: `MILO MILO_LW SGD ADAMW ADAGRAD NOVOGRAD`)
    - `--num_runs`: Number of training runs per optimizer for statistical analysis (default: `5`)
    - `--experiment_title`: Custom title for the experiment plots

    The script performs the following tasks:
    1. Trains a small transformer model with each specified optimizer.
    2. Generates comparative visualizations, including:
      - Training and validation loss curves
      - Loss vs. wall time analysis
      - Layer-wise gradient norm analysis
      - Gradient imbalance ratio plots
    3. Saves detailed metrics to CSV for further analysis.

2. **[NovoGrad-pytorch](https://github.com/lonePatient/NovoGrad-pytorch)** (MIT License)  
  - Citation: \citep{lonePatient2019novograd}

## Citing MILO

If you use MILO or this codebase in your research, please cite our paper:

```bibtex
@inproceedings{milo2025neurips,
  title={MILO: Magnitude-Invariant Learning Optimizer},
  author={},
  booktitle={},
  year={2025}
}
```

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).

You are free to:
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material for any purpose, even commercially

Under the following terms:
- Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made.

This is a human-readable summary of (and not a substitute for) the [license](http://creativecommons.org/licenses/by/4.0/legalcode).