"""
Unified supervised learning experiment runner.
This script combines the functionality of logistic regression, multilayer neural network,
and deep CNN experiments into a single framework.
"""
import sys, os, argparse
# Prevent local scripts from shadowing installed optimizer packages
if '' in sys.path:
    sys.path.remove('')
if os.getcwd() in sys.path:
    sys.path.remove(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))  

import random
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
import time
from matplotlib import rcParams
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.distributed as dist

# Imports from existing experiments
from experiments.train_utils import run_training, get_layer_names, evaluate_model
from experiments.experiment_runner import run_experiments
from experiments.hyperparameter_tuning_utils import tune_hyperparameters

# Import network models
from experiments.supervised_learning.network import (
    LogisticRegressionModel, MLP, DeepCNN, ResNet18, ResNet34, VGG11, ViT_Tiny,
)

# Import optimizers and utilities
from milo import milo
from optimizers.novograd import NovoGrad
#from adalayer import Adalayer
# from adam_mini import Adam_mini
from optimizers.muon import MuonWithAuxAdam
from optimizers.ademamix_pytorch import AdEMAMix
from optimizers.soap import SOAP

# Import configuration
from experiments.supervised_learning.config import (
    EXPERIMENTS,
    BATCH_SIZE,
    EPOCHS,
    LR,
    PARAM_GRID,
    OPTIMIZERS,
    OPTIMIZER_PARAMS,
    SCHEDULER_PARAMS,
    RUNS_PER_OPTIMIZER,
    EXPERIMENT_CONFIGS,
    TRIALS,
    VAL_SPLIT_RATIO,
    TEST_SPLIT_RATIO,
    PERFORM_HYPERPARAMETER_TUNING,
    RESULTS_DIR_TUNING,
    VISUALS_DIR_TUNING,
    RESULTS_DIR_NO_TUNING,
    VISUALS_DIR_NO_TUNING,
)  

# Note: Avoid globally monkey-patching torch.distributed on Windows, as it can
# cause hangs in unrelated code paths. If Muon is selected, we'll handle any
# necessary single-process shims locally in that branch only.

# --- Setup Visualization Style ---
sns.set(style="whitegrid", context="paper")
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 10
rcParams['axes.titlesize'] = 12
rcParams['axes.labelsize'] = 11
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 10
rcParams['figure.titlesize'] = 14

# Set color palette
palette = sns.color_palette("colorblind")

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Reproducibility: Fix random seeds globally ---
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
# Ensure deterministic behavior where possible
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- Helper Functions ---

def get_model(model_name, model_args={}):
    """Instantiates a model based on its name."""
    if model_name == "MLP":
        return MLP(**model_args)
    elif model_name == "DeepCNN":
        return DeepCNN(**model_args)
    elif model_name == "LogisticRegressionModel":
        return LogisticRegressionModel(**model_args)
    elif model_name == "ResNet18":
        return ResNet18(**model_args)
    elif model_name == "ResNet34":
        return ResNet34(**model_args)
    elif model_name == "VGG11":
        return VGG11(**model_args)
    elif model_name == "ViT_Tiny":
        return ViT_Tiny(**model_args)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def get_dataloader(dataset_name, transform, batch_size, train=True):
    """Loads a dataset based on its name."""
    import os
    # Use pre-downloaded datasets from scratch directory (offline HPC)
    root = os.path.expanduser('~/scratch/datasets')
    os.makedirs(root, exist_ok=True)

    if dataset_name == "MNIST":
        dataset = datasets.MNIST(root=root, train=train, download=False, transform=transform)
    elif dataset_name == "CIFAR10":
        dataset = datasets.CIFAR10(root=root, train=train, download=False, transform=transform)
    elif dataset_name == "CIFAR100":
        dataset = datasets.CIFAR100(root=root, train=train, download=False, transform=transform)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)

# --- Training Function Factory ---

def create_train_experiment_fn(experiment_type, train_loader_instance):
    """Create a training function closure for the specified experiment."""
    # Split train_loader_instance into train, validation, and test subsets
    train_dataset = train_loader_instance.dataset
    num_total = len(train_dataset)
    num_test = int(num_total * TEST_SPLIT_RATIO)
    num_remaining = num_total - num_test
    num_val = int(num_remaining * VAL_SPLIT_RATIO)  # Val ratio applied to remaining data
    num_train_subset = num_remaining - num_val

    # Ensure splits add up
    if num_train_subset + num_val + num_test != num_total:
        # Adjust train subset size slightly due to rounding
        num_train_subset = num_total - num_val - num_test
        print(f"Adjusting split sizes: Train={num_train_subset}, Val={num_val}, Test={num_test}")

    train_subset, val_subset, test_subset = random_split(
        train_dataset,
        [num_train_subset, num_val, num_test],
        generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
    )

    # Create DataLoaders for the subsets
    train_subset_loader = DataLoader(
        train_subset,
        batch_size=train_loader_instance.batch_size,
        shuffle=True,
        num_workers=train_loader_instance.num_workers if hasattr(train_loader_instance, 'num_workers') else 0
    )
    val_loader_instance = DataLoader(
        val_subset,
        batch_size=train_loader_instance.batch_size,
        shuffle=False,  # No need to shuffle validation data
        num_workers=train_loader_instance.num_workers if hasattr(train_loader_instance, 'num_workers') else 0
    )
    test_subset_loader = DataLoader(        test_subset,
        batch_size=train_loader_instance.batch_size,
        shuffle=False,        num_workers=train_loader_instance.num_workers if hasattr(train_loader_instance, 'num_workers') else 0
    )

    def train_experiment(optimizer_name, return_settings=False):
        """Train a model with a specified optimizer and return metrics."""
        # Get experiment-specific configurations
        config = EXPERIMENT_CONFIGS[experiment_type]
        model_name = config["model_name"]
        model_args = config["model_args"]
        
        # Instantiate the model
        print("Training Device:", device)
        model = get_model(model_name, model_args).to(device)
        # Get layer names for gradient tracking
        layer_names = get_layer_names(model)

        # Define loss function
        criterion = nn.CrossEntropyLoss()
        
        # Base learning rate for the experiment
        base_lr = LR[experiment_type]
        
        # Optimizer parameters (exclude 'lr' to avoid passing it twice)
        orig_optimizer_params = OPTIMIZER_PARAMS.get(optimizer_name, {})
        optimizer_params = {k: v for k, v in orig_optimizer_params.items() if k != 'lr'}

        # Select and instantiate the optimizer
        muon_shim_applied = False
        muon_dist_originals = {}
        if optimizer_name == "MILO":
            optimizer = milo(model.parameters(), lr=base_lr, **optimizer_params)
        elif optimizer_name == "MILO_LW":
            optimizer = milo(model.parameters(), lr=base_lr, **optimizer_params)
        elif optimizer_name == "MILO_TUNED":
            optimizer = milo(model.parameters(), lr=base_lr, **optimizer_params)
        elif optimizer_name == "MILO_LW_TUNED":
            optimizer = milo(model.parameters(), lr=base_lr, **optimizer_params)
        elif optimizer_name == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, **optimizer_params)
        elif optimizer_name == "ADAGRAD":
            optimizer = torch.optim.Adagrad(model.parameters(), lr=base_lr, **optimizer_params)
        elif optimizer_name == "ADAMW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, **optimizer_params)
        elif optimizer_name == "NOVOGRAD":
            optimizer = NovoGrad(model.parameters(), lr=base_lr, **optimizer_params)
        #elif optimizer_name == "ADALAYER":
        #    optimizer = Adalayer(model.parameters(), lr=base_lr, **optimizer_params)
        elif optimizer_name == "ADAM_MINI":
            # Adam-mini expects named parameters; architecture-aware safe filtering.
            is_vit_like = any(hasattr(model, attr) for attr in ("cls_token", "pos_embed"))
            def _adam_mini_named_params(m):
                for n, p in m.named_parameters():
                    if is_vit_like:
                        # For ViT, only allow 1D tensors (bias, LayerNorm weights) to avoid reshape assumptions.
                        if p.ndim != 1:
                            try:
                                p.requires_grad = False
                            except Exception:
                                pass
                            print(f"ADAM_MINI: [ViT] Skipping {n} shape {tuple(p.shape)} (ndim={p.ndim})")
                            continue
                        yield n, p
                    else:
                        # Non-ViT: allow 1D/2D; skip >=3D (conv/embeddings)
                        if p.ndim >= 3:
                            try:
                                p.requires_grad = False
                            except Exception:
                                pass
                            print(f"ADAM_MINI: Skipping {n} shape {tuple(p.shape)} (ndim={p.ndim})")
                            continue
                        yield n, p
            optimizer = Adam_mini(named_parameters=_adam_mini_named_params(model), lr=base_lr, **optimizer_params)
            # Avoid transformer-specific annotations for ViT to prevent internal head reshaping
            try:
                if not is_vit_like:
                    if hasattr(optimizer, 'output_names'):
                        optimizer.output_names.add('head')
                    if hasattr(optimizer, 'wqk_names'):
                        optimizer.wqk_names.update({'qkv', 'attn.qkv', 'q', 'k'})
                    if hasattr(optimizer, 'wv_names'):
                        optimizer.wv_names.update({'v', 'qkv'})
            except Exception:
                pass
        elif optimizer_name == "MUON":
            # Muon optimizer: param grouping varies by model architecture
            # Provide minimal single-process shims only for required collectives
            # without globally changing torch.distributed state.
            if dist.is_available():
                try:
                    muon_dist_originals['get_world_size'] = getattr(dist, 'get_world_size', None)
                    muon_dist_originals['get_rank'] = getattr(dist, 'get_rank', None)
                    muon_dist_originals['all_gather'] = getattr(dist, 'all_gather', None)
                    muon_dist_originals['is_initialized'] = getattr(dist, 'is_initialized', None)
                    dist.get_world_size = lambda group=None: 1
                    dist.get_rank = lambda group=None: 0
                    # Provide a local, no-op all_gather for single-process
                    def _fake_all_gather(tensor_list, tensor, group=None, async_op=False):
                        # Copy the input tensor into each slot of the list (world_size=1 -> one slot)
                        for i in range(len(tensor_list)):
                            tensor_list[i].copy_(tensor)
                        return None
                    dist.all_gather = _fake_all_gather
                    # Report not-initialized to discourage other code paths
                    dist.is_initialized = lambda group=None: False
                    muon_shim_applied = True
                except Exception:
                    muon_shim_applied = False
            if hasattr(model, 'body') and hasattr(model, 'head') and hasattr(model, 'embed'):
                hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2]
                hidden_gains_biases = [p for p in model.body.parameters() if p.ndim < 2]
                nonhidden_params = [*model.head.parameters(), *model.embed.parameters()]
                param_groups = [
                    dict(params=hidden_weights, use_muon=True, lr=0.02, weight_decay=0.01),
                    dict(params=hidden_gains_biases + nonhidden_params, use_muon=False, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
                ]
            else:
                hidden_weights = [p for _, p in model.named_parameters() if p.ndim >= 2]
                other_params = [p for _, p in model.named_parameters() if p.ndim < 2]
                param_groups = [
                    dict(params=hidden_weights, use_muon=True, lr=base_lr, weight_decay=optimizer_params.get('weight_decay', 0)),
                    dict(params=other_params, use_muon=False, lr=base_lr, betas=optimizer_params.get('betas', (0.9, 0.999)), weight_decay=optimizer_params.get('weight_decay', 0)),
                ]
            optimizer = MuonWithAuxAdam(param_groups)
        elif optimizer_name == "ADEMAMIX":
            optimizer = AdEMAMix(model.parameters(), lr=base_lr, **optimizer_params)
        elif optimizer_name == "SOAP":
            optimizer = SOAP(model.parameters(), lr=base_lr, **optimizer_params)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Scheduler parameters
        scheduler_info = SCHEDULER_PARAMS.get(optimizer_name, {"scheduler": "None"})
        scheduler = None
        if scheduler_info["scheduler"] != "None":
            scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_info["scheduler"])
            scheduler = scheduler_class(optimizer, **scheduler_info["params"])

        # Collect experimental settings
        settings = {
            "model": config["model_name"],
            "model_architecture": {layer: list(param.shape) for layer, param in model.named_parameters()},
            "optimizer_params": optimizer_params,
            "batch_size": BATCH_SIZE,
            "dataset": config["dataset_name"],
            "criterion": criterion.__class__.__name__,
            "device": str(device),
            "scheduler_params": scheduler_info,
            "validation_split_ratio": VAL_SPLIT_RATIO,
            "test_split_ratio": TEST_SPLIT_RATIO 
        }

        # Run training using the train and validation subset loaders
        # Unpack steps_per_epoch and train_metrics_hist from the returned tuple
        # Run training and capture layer-wise runtime history
        try:
            val_metrics, norm_walltimes, gradient_norms, iter_costs, \
            layer_runtime_history, layer_bwd_runtime_history, component_runtime_history, \
            trained_model, steps_per_epoch, train_metrics_hist, \
            iteration_logs, validation_walltimes, epoch_end_walltimes = run_training(
                model, train_subset_loader, val_loader_instance, optimizer, criterion, device, EPOCHS,
                scheduler=scheduler, layer_names=layer_names
            )
        finally:
            # Restore any MUON distributed shims if applied
            if muon_shim_applied and dist.is_available():
                try:
                    if muon_dist_originals.get('get_world_size') is not None:
                        dist.get_world_size = muon_dist_originals['get_world_size']
                    if muon_dist_originals.get('get_rank') is not None:
                        dist.get_rank = muon_dist_originals['get_rank']
                    if muon_dist_originals.get('all_gather') is not None:
                        dist.all_gather = muon_dist_originals['all_gather']
                    if muon_dist_originals.get('is_initialized') is not None:
                        dist.is_initialized = muon_dist_originals['is_initialized']
                except Exception:
                    pass

        # Final Test Evaluation
        print(f"Evaluating final model for {optimizer_name} on test subset...")
        test_start_time = time.time()
        test_metrics = evaluate_model(trained_model, test_subset_loader, criterion, device)
        test_eval_time = time.time() - test_start_time
        print(f"Test Subset Evaluation Time: {test_eval_time:.2f}s")
        print(f"Test Subset Results - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.2f}%, F1: {test_metrics['f1_score']:.4f}, AUC: {test_metrics['auc']:.4f}")

        test_metrics['eval_time_seconds'] = test_eval_time

        # Return metrics including layer-wise forward and backward runtime history and layer names
        result = (
            val_metrics['val_loss'],
            val_metrics['val_accuracy'],
            val_metrics['val_f1_score'],
            val_metrics['val_auc'],
            iter_costs,
            norm_walltimes,
            gradient_norms,
            layer_runtime_history,
            layer_bwd_runtime_history,
            component_runtime_history,
            layer_names,
            test_metrics,
            steps_per_epoch,
            train_metrics_hist,
            iteration_logs,
            validation_walltimes,
            epoch_end_walltimes
        )

        if return_settings:
            return result + (settings,)
        return result

    return train_experiment

# --- Warm-up Function ---
def perform_warmup(experiment_type, batch_size, device):
    """Performs a warm-up run for a specific experiment type."""
    print(f"--- Performing warm-up for {experiment_type} ---")
    try:
        config = EXPERIMENT_CONFIGS[experiment_type]
        # Use a minimal version of the model if possible, or the actual model
        warmup_model = get_model(config["model_name"], config["model_args"]).to(device)
        warmup_criterion = nn.CrossEntropyLoss()
        # Use a simple optimizer like SGD for warm-up
        warmup_optimizer = torch.optim.SGD(warmup_model.parameters(), lr=0.01)

        # Get a single batch from the training loader
        # Note: We create a temporary loader here to avoid affec
        # 
        # ting the main one
        temp_loader = get_dataloader(
            config["dataset_name"], config["transform"], batch_size, train=True
        )
        inputs, targets = next(iter(temp_loader))
        inputs, targets = inputs.to(device), targets.to(device)

        # Perform one forward/backward pass and step
        warmup_model.train()
        warmup_optimizer.zero_grad()
        outputs = warmup_model(inputs)
        loss = warmup_criterion(outputs, targets)
        loss.backward()
        warmup_optimizer.step()

        # Synchronize GPU if available
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Clean up memory
        del warmup_model, warmup_optimizer, temp_loader, inputs, targets, outputs, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"--- Warm-up for {experiment_type} complete ---")
        return True
    except Exception as e:
        print(f"Warning: Warm-up for {experiment_type} failed: {e}")
        return False

# --- Main Function ---

def run_supervised_experiments(experiments_to_run=None):
    """
    Run specified supervised learning experiments.
    
    Args:
        experiments_to_run (list, optional): List of experiment names to run.
            Defaults to EXPERIMENTS if None.
    """
    if experiments_to_run is None:
        experiments_to_run = EXPERIMENTS
    
    # Flatten experiments list if it contains nested lists
    flattened_experiments = []
    for item in experiments_to_run:
        if isinstance(item, list):
            flattened_experiments.extend(item)
        else:
            flattened_experiments.append(item)
    experiments_to_run = flattened_experiments
    
    # Validate experiment names
    valid_experiments = set(EXPERIMENT_CONFIGS.keys())
    for exp in experiments_to_run[:]:
        if exp not in valid_experiments:
            print(f"Warning: Experiment '{exp}' is not valid. Skipping.")
            experiments_to_run.remove(exp)
    
    if not experiments_to_run:
        print("No valid experiments to run. Exiting.")
        return
    
    # --- Perform Warm-up Before Starting Experiments ---
    first_experiment_type = experiments_to_run[0]
    perform_warmup(first_experiment_type, BATCH_SIZE, device)
    # --- End Warm-up ---

    # Run each experiment
    for experiment_type in experiments_to_run:
        print(f"\n{'='*50}")
        print(f"Starting {experiment_type} experiment")
        print(f"{'='*50}")
        
        # Get experiment configuration
        config = EXPERIMENT_CONFIGS[experiment_type]
        
        # Create the FULL training dataloader (will be split inside create_train_experiment_fn)
        train_loader_instance = get_dataloader(
            config["dataset_name"], config["transform"], BATCH_SIZE, train=True
        )
        
        # Create directories for results and visuals based on tuning flag
        base_dir = os.path.dirname(__file__)
        if PERFORM_HYPERPARAMETER_TUNING:
            results_dir_name = RESULTS_DIR_TUNING
            visuals_dir_name = VISUALS_DIR_TUNING
        else:
            results_dir_name = RESULTS_DIR_NO_TUNING
            visuals_dir_name = VISUALS_DIR_NO_TUNING
            print(f"Hyperparameter tuning is OFF. Using pre-defined parameters or defaults.")
            print(f"Results will be saved to '{results_dir_name}', visuals to '{visuals_dir_name}'.")

        results_dir = os.path.join(base_dir, experiment_type.lower(), results_dir_name)
        visuals_dir = os.path.join(base_dir, experiment_type.lower(), visuals_dir_name)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(visuals_dir, exist_ok=True)
        
        # Perform hyperparameter tuning if enabled
        if PERFORM_HYPERPARAMETER_TUNING:
            print(f"Tuning hyperparameters for {experiment_type}...")
            # Tune on train_loader_instance, val split handled inside function
            best_hps = tune_hyperparameters(
                model_fn=lambda: get_model(config['model_name'], config['model_args']),
                optimizer_names=OPTIMIZERS,
                param_grid=PARAM_GRID,
                device=device,
                experiment_name=experiment_type,
                task_type='classification',
                epochs=EPOCHS,
                num_trials=TRIALS,
                train_loader=train_loader_instance,
                val_ratio=VAL_SPLIT_RATIO,
                criterion=nn.CrossEntropyLoss()
            )
            # Update optimizer defaults for subsequent runs
            for opt_name, hp in best_hps.items():
                if isinstance(hp, dict) and hp:
                    OPTIMIZER_PARAMS[opt_name] = hp

        # Create training function for this experiment
        train_fn = create_train_experiment_fn(
            experiment_type, train_loader_instance
        )
        
        # Run experiments
        run_experiments(
            train_fn, results_dir, visuals_dir, EPOCHS,
            optimizer_names=OPTIMIZERS,
            loss_title=f"Validation {config['plot_titles']['loss']}",  # Update titles
            acc_title=f"Validation {config['plot_titles']['accuracy']}",
            plot_filename=f"{experiment_type.lower()}_validation_curves",  # Update filename
            csv_filename=f"{experiment_type.lower()}_validation_metrics.csv",  # Update filename
            experiment_title=f"{experiment_type} Experiment ",  # Update title
            cost_xlimit=config.get("cost_xlimit"),
            f1_title=f"Validation {config['plot_titles']['f1']}",
            num_runs=RUNS_PER_OPTIMIZER
        )
        
        print(f"\n{'-'*50}")
        print(f"Completed {experiment_type} experiment")
        print(f"{'-'*50}")

# --- Script Entry Point ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run supervised learning experiments.")
    parser.add_argument("--experiments", nargs='+', default=EXPERIMENTS,
                        help=f"List of experiments to run. Choices: {list(EXPERIMENT_CONFIGS.keys())}")
    args = parser.parse_args()
    experiments = args.experiments

    # Run specified experiments or all by default
    run_supervised_experiments(experiments)
