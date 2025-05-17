"""
Configuration file for Loss Map experiments.

Purpose:
This file defines settings for visualizing 2D loss landscapes of mathematical benchmark
functions and plotting the optimization paths taken by different optimizers. It helps
in understanding and comparing the behavior of these algorithms in simple, controlled
environments.

Options:
- OUTPUT_DIR: Directory where generated visualizations (loss maps, trajectory plots)
  will be saved.
- RUNS: Number of times each optimization process is run (e.g., for averaging or
  overlaying trajectories if stochasticity is involved, though less common for these
  deterministic benchmarks).
- GRID_POINTS: Resolution of the grid used to compute and plot the loss surface.
  Higher values give smoother, more detailed maps but take longer to compute.
- DEFAULT_OPTIMIZER_PARAMS: Default hyperparameter settings (e.g., learning rate `eta`,
  `iterations`, `momentum`) for each optimizer ("SGD", "ADAM", "milo", "NOVOGRAD")
  being tested.
- BENCHMARKS: A list defining the mathematical functions to be used. Each benchmark
  entry includes:
    - `name`: The name of the function (e.g., "Himmelblau", "Camel6").
    - `domain`: The (x_min, x_max, y_min, y_max) boundaries for plotting the surface.
    - `minima`: Coordinates of known global minima of the function, used for reference.
    - `init_point`: The starting (x, y) coordinates for all optimization algorithms.
    - Optimizer-specific parameters (e.g., `milo_params`) that can override the
      defaults for that particular benchmark function if needed.
- PLOT_SETTINGS: Parameters for customizing the appearance of 2D and 3D plots,
  including figure size, colormaps, contour styles, and marker styles/colors for
  different optimizers and points of interest (start, minima).

Experiments:
The primary goal is to visually analyze and compare the behavior of different
optimization algorithms on well-known 2D mathematical functions. This involves:
1. Generating a 2D or 3D plot of the function's loss surface within the specified domain.
2. Running each configured optimizer starting from the `init_point` for a set number
   of iterations.
3. Overlaying the step-by-step paths (trajectories) these optimizers take on the loss
   surface.
This allows for a qualitative assessment of their convergence properties, speed,
ability to navigate complex landscapes, and proximity to known minima.
"""

import numpy as np
import math

# General settings
OUTPUT_DIR = "experiments/loss_maps/visuals"
RUNS = 5
GRID_POINTS = 400

# Default optimizer parameters
DEFAULT_OPTIMIZER_PARAMS = {
    "SGD": {
        "eta": 0.1,
        "iterations": 100,
        "momentum": 0.09,
        "grad_clip": 1.0
    },
    "ADAM": {
        "eta": 0.1,
        "iterations": 100,
        "amsgrad": True
    },
    "milo": {
        "eta": 0.1,
        "iterations": 100,
        "group_size": 150,
        "momentum": 0.30,
        "adaptive": True,
        "adaptive_eps": 1e-8,
        "weight_decay": 0.05,
        "layer_wise": False
    },
    "NOVOGRAD": {
        "eta": 0.1,
        "iterations": 100,
        "grad_averaging": True 
    }
}

# Benchmark function configurations
BENCHMARKS = [
    {
        "name": "Himmelblau",
        "domain": (-6, 6, -6, 6),
        "minima": [(3.0, 2.0)],
        "init_point": (0,0),
        "sgd_params": {"iterations": 150},
        "adam_params": {"iterations": 150},
        "milo_params": {"iterations": 150},
        "novograd_params": {"iterations": 150}
    },
    {
        "name": "Camel6",
        "domain": (-3, 3, -3, 3),
        "minima": [(0.0898, -0.7126)],
        "init_point": (-1, -2.8),
        "sgd_params": {"iterations": 150},
        "adam_params": {"iterations": 150},
        "milo_params": {"iterations": 150},
        "novograd_params": {"iterations": 150}
    },
    {
        "name": "De Jong5",
        "domain": (-10, 10, -10, 10),
        "minima": [(0, 0)],
        "init_point": (2.4, -4.3),
        "sgd_params": {"iterations": 250},
        "adam_params": {"iterations": 250},
        "milo_params": {"iterations": 250},
        "novograd_params": {"iterations": 250}
    },
    {
        "name": "Michalewicz",
        "domain": (0, np.pi, 0, np.pi),
        "minima": [(2.20, 1.57)],
        "init_point": (1.2, 1.9),
        "sgd_params": {"iterations": 150},
        "adam_params": {"iterations": 150},
        "milo_params": {"iterations": 150},
        "novograd_params": {"iterations": 150}
    }
]

# Plotting settings
PLOT_SETTINGS = {
    "2D": {
        "figsize": (10, 8),
        "levels": 50,
        "cmap": "viridis",
        "contour_colors": "k",
        "contour_linewidths": 0.5,
        "contour_alpha": 0.3
    },
    "3D": {
        "figsize": (12, 10),
        "cmap": "viridis",
        "surface_alpha": 0.7
    },
    "markers": {
        "SGD": {"line": "r-o", "marker": "ro", "label": "SGD"},
        "ADAM": {"line": "m-^", "marker": "m^", "label": "ADAM"},
        "milo": {"line": "b-s", "marker": "bs", "label": "milo"},
        "NOVOGRAD": {"line": "g-d", "marker": "gd", "label": "NovoGrad"},
        "start": {"marker": "ko", "size": 10, "label": "Start"},
        "minima": {"marker": "k*", "size": 15, "label": "Ideal Minimum"}
    }
}
