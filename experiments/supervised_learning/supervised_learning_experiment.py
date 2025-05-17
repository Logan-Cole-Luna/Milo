"""
Unified supervised learning experiment runner.
This script combines the functionality of logistic regression, multilayer neural network,
and deep CNN experiments into a single framework.
"""
import sys, os, argparse
import random
import numpy as np  
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))  

# Imports from existing experiments
from experiments.train_utils import run_training, get_layer_names, evaluate_model
from experiments.experiment_runner import run_experiments
from experiments.hyperparameter_tuning_utils import tune_hyperparameters

# Import network models
from network import LogisticRegressionModel, MLP, DeepCNN, ResNet18, ResNet34

# Import optimizers and utilities
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split  
from torchvision import datasets, transforms
import seaborn as sns
from matplotlib import rcParams
from milo import milo
from novograd import NovoGrad
import torch.optim  
import time

# Import configuration
from config import (EXPERIMENTS, BATCH_SIZE, EPOCHS, LR, PARAM_GRID, OPTIMIZERS,
                    OPTIMIZER_PARAMS, SCHEDULER_PARAMS, RUNS_PER_OPTIMIZER,
                    EXPERIMENT_CONFIGS, TRIALS, VAL_SPLIT_RATIO, TEST_SPLIT_RATIO,
                    PERFORM_HYPERPARAMETER_TUNING,  
                    RESULTS_DIR_TUNING, VISUALS_DIR_TUNING,  
                    RESULTS_DIR_NO_TUNING, VISUALS_DIR_NO_TUNING)  

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
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def get_dataloader(dataset_name, transform, batch_size, train=True):
    """Loads a dataset based on its name."""
    root = './data'  
    if dataset_name == "MNIST":
        dataset = datasets.MNIST(root=root, train=train, download=True, transform=transform)
    elif dataset_name == "CIFAR10":
        dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
    elif dataset_name == "CIFAR100":
        dataset = datasets.CIFAR100(root=root, train=train, download=True, transform=transform)
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
    test_subset_loader = DataLoader(  # Create loader for the test subset
        test_subset,
        batch_size=train_loader_instance.batch_size,
        shuffle=False,  # No need to shuffle test data
        num_workers=train_loader_instance.num_workers if hasattr(train_loader_instance, 'num_workers') else 0
    )

    def train_experiment(optimizer_name, return_settings=False):
        """Training function for the current experiment with specified optimizer."""
        # Use the subset loaders created above
        nonlocal train_subset_loader, val_loader_instance, test_subset_loader, experiment_type

        # Create a new model instance for each call
        config = EXPERIMENT_CONFIGS[experiment_type]
        model = get_model(config["model_name"], config["model_args"]).to(device)
        criterion = nn.CrossEntropyLoss()
        layer_names = get_layer_names(model)

        optimizer_upper = optimizer_name.upper()

        # Hyperparameter tuning logic remains the same, but uses train_subset_loader and val_loader_instance
        params = OPTIMIZER_PARAMS.get(optimizer_upper, {}).copy()
        if 'lr' not in params:
            if experiment_type in LR:
                params['lr'] = LR[experiment_type]
            else:
                print(f"Warning: Learning rate for experiment type '{experiment_type}' not found. Using default.")

        if PERFORM_HYPERPARAMETER_TUNING and optimizer_upper in PARAM_GRID and PARAM_GRID[optimizer_upper]: 
            best_hyperparams = tune_hyperparameters(
                model_fn=lambda: get_model(config["model_name"], config["model_args"]),
                optimizer_names=[optimizer_name],
                param_grid={optimizer_name: PARAM_GRID[optimizer_upper]},
                train_loader=train_loader_instance,  # Pass original full train loader here
                device=device,
                experiment_name=experiment_type,
                task_type="classification",
                epochs=8,
                num_trials=TRIALS,
                val_ratio=VAL_SPLIT_RATIO,  # Use the config ratio for tuning validation
                criterion=criterion 
            )[optimizer_name]

            if best_hyperparams:
                print(f"Using best hyperparameters for {optimizer_name}: {best_hyperparams}")
                params.update(best_hyperparams)
            else:
                print(f"No best hyperparameters found for {optimizer_name}, using defaults/dynamic LR.")

        print(f"Final parameters for {optimizer_name} in {experiment_type}: {params}")

        # Create optimizer
        if optimizer_upper == "ADAGRAD":
            optimizer = torch.optim.Adagrad(model.parameters(), **params)
        elif optimizer_upper == "ADAMW":
            optimizer = torch.optim.AdamW(model.parameters(), **params)
        elif optimizer_upper == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), **params)
        elif optimizer_upper == "MILO":
            optimizer = milo(model.parameters(), **params)
        elif optimizer_upper == "MILO_LW":
            optimizer = milo(model.parameters(), **params)
            
            
        elif optimizer_upper == "NOVOGRAD":
            optimizer = NovoGrad(model.parameters(), **params)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Set up scheduler
        scheduler = None
        scheduler_config = SCHEDULER_PARAMS.get(optimizer_upper, {})
        if scheduler_config and scheduler_config.get("scheduler") != "None":
            scheduler_type = scheduler_config["scheduler"]
            scheduler_params = scheduler_config["params"]
            scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_type)
            scheduler = scheduler_class(optimizer, **scheduler_params)

        # Collect experimental settings
        settings = {
            "model": config["model_name"],
            "model_architecture": {layer: list(param.shape) for layer, param in model.named_parameters()},
            "optimizer_params": params,
            "batch_size": BATCH_SIZE,
            "dataset": config["dataset_name"],
            "criterion": criterion.__class__.__name__,
            "device": str(device),
            "scheduler_params": scheduler_config,
            "validation_split_ratio": VAL_SPLIT_RATIO,
            "test_split_ratio": TEST_SPLIT_RATIO 
        }

        # Run training using the train and validation subset loaders
        # Unpack steps_per_epoch and train_metrics_hist from the returned tuple
        val_metrics, norm_walltimes, gradient_norms, iter_costs, trained_model, steps_per_epoch, train_metrics_hist = run_training(
            model, train_subset_loader, val_loader_instance, optimizer, criterion, device, EPOCHS,
            scheduler=scheduler, layer_names=layer_names
        )

        # Final Test Evaluation
        print(f"Evaluating final model for {optimizer_name} on test subset...")
        test_start_time = time.time()
        test_metrics = evaluate_model(trained_model, test_subset_loader, criterion, device) 
        test_eval_time = time.time() - test_start_time
        print(f"Test Subset Evaluation Time: {test_eval_time:.2f}s")
        print(f"Test Subset Results - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.2f}%, F1: {test_metrics['f1_score']:.4f}, AUC: {test_metrics['auc']:.4f}") 

        test_metrics['eval_time_seconds'] = test_eval_time

        # Return validation metrics history AND final test metrics AND steps_per_epoch
        result = (
            val_metrics['val_loss'],
            val_metrics['val_accuracy'],
            val_metrics['val_f1_score'],
            val_metrics['val_auc'],
            iter_costs,
            norm_walltimes,
            gradient_norms,
            layer_names,
            test_metrics,
            steps_per_epoch,
            train_metrics_hist 
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
