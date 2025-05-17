import sys, os
import copy
sys.path.append(os.path.join(os.path.dirname(__file__), '../..')) # Add project root

# Imports from existing experiments
from experiments.train_utils import run_training, get_layer_names
from experiments.experiment_runner import run_experiments
# Import all network types
from network import MLP, DeepCNN, LogisticRegressionModel, ResNet18, ResNet34
from milo import milo

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import seaborn as sns
from matplotlib import rcParams

# --- Import Configuration ---
from config import (BATCH_SIZE, EPOCHS, RUNS_PER_OPTIMIZER,
                    OPTIMIZER_PARAMS, ABLATION_PARAMS, EXPERIMENT_CONFIGS,
                    SKIP_EXISTING_ABLATION_GROUPS)

# --- Setup ---
sns.set(style="whitegrid", context="paper")
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
palette = sns.color_palette("colorblind")

base_dir = os.path.dirname(__file__)
results_dir_root = os.path.join(base_dir, 'results')
visuals_dir_root = os.path.join(base_dir, 'visuals')
os.makedirs(results_dir_root, exist_ok=True)
os.makedirs(visuals_dir_root, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Helper Functions ---

def get_model(model_name, model_args={}):
    """Instantiates a model based on its name."""
    if model_name == "MLP":
        return MLP(**model_args)
    elif model_name == "DeepCNN":
        return DeepCNN(**model_args)
    elif model_name == "LogisticRegression":
        return LogisticRegressionModel(**model_args) # Assuming it exists
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
    # elif dataset_name == "CIFAR100":
    #     dataset = datasets.CIFAR100(root=root, train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)

# Store generated ablation configurations globally - reset per model/dataset pair
ABLATION_CONFIGS_STORE = {}

# --- Training Function Factory ---

def create_train_experiment_fn(model_instance, train_loader_instance):
    """
    Creates a training function closure for the current model and dataloader.
    This function will be passed to run_experiments.
    
    Args:
        model_instance: The model instance to use for training.
        train_loader_instance: The DataLoader for the training data.
        
    Returns:
        function: A training function for the experiment.
    """
    current_model = model_instance
    current_loader = train_loader_instance

    def train_experiment_ablation(config_name, return_settings=False):
        """
        Trains the captured model using parameters defined by the config_name.
        config_name refers to a key in the global ABLATION_CONFIGS_STORE.
        
        Args:
            config_name (str): The name of the configuration to use for training.
            return_settings (bool): Whether to return the settings used for this training run.
            
        Returns:
            tuple: Contains training results and optionally the settings used.
        """
        nonlocal current_model, current_loader # Access model/loader from outer scope

        if config_name not in ABLATION_CONFIGS_STORE:
            raise ValueError(f"Configuration '{config_name}' not found in ABLATION_CONFIGS_STORE.")

        config = ABLATION_CONFIGS_STORE[config_name]
        base_optimizer_name = config["base_optimizer"]
        final_params = config["params"]

        model = current_model.to(device) # Ensure model is on the correct device for this run

        criterion = nn.CrossEntropyLoss()
        layer_names = get_layer_names(model)

        # Instantiate optimizer
        if base_optimizer_name.upper() == "MILO":
            optimizer = milo(model.parameters(), **final_params)
        elif base_optimizer_name.upper() == "MILO_LW":
            optimizer = milo(model.parameters(), **final_params)
        else:
            raise ValueError(f"Unknown base optimizer for ablation: {base_optimizer_name}")

        scheduler = None
        
        settings = {
            "model": model.__class__.__name__, # Get model name dynamically
            "model_architecture": {layer: str(list(param.shape)) for layer, param in model.named_parameters()},
            "optimizer_params": final_params,
            "batch_size": current_loader.batch_size,
            "dataset": config.get("dataset_name", "Unknown"), # Add dataset name to config later
            "criterion": criterion.__class__.__name__,
            "device": str(device),
            "config_name": config_name
        }

        # Run training and get steps_per_epoch
        val_metrics_hist, norm_walltimes, gradient_norms, iter_costs, _, steps_per_epoch, train_metrics_hist = run_training(
            model, current_loader, current_loader, optimizer, criterion, device, EPOCHS,
            scheduler=scheduler, layer_names=layer_names
        )
        # Provide test_metrics placeholder and steps_per_epoch for experiment_runner
        test_metrics = {}
        result = (
            val_metrics_hist['val_loss'], val_metrics_hist['val_accuracy'], val_metrics_hist['val_f1_score'],
            val_metrics_hist['val_auc'],
            iter_costs, norm_walltimes, gradient_norms, layer_names,
            test_metrics, steps_per_epoch, train_metrics_hist
        )

        if return_settings:
            return result + (settings,)
        return result

    return train_experiment_ablation


# --- Main Execution Logic ---

if __name__ == "__main__":
    """Main execution block for the MILO ablation study."""

    # Loop through each experiment configuration (model, dataset)
    for model_name, dataset_name, model_args, dataset_transform in EXPERIMENT_CONFIGS:
        print(f"\n{'='*20} Starting Ablation for {model_name} on {dataset_name} {'='*20}")

        # --- Setup for current model/dataset ---
        model_instance = get_model(model_name, model_args)
        train_loader_instance = get_dataloader(dataset_name, dataset_transform, BATCH_SIZE, train=True)

        # Reset global store for this model/dataset pair
        ABLATION_CONFIGS_STORE = {}
        grouped_configs_for_runs = {}

        # Define output directories for this specific experiment config
        exp_config_name = f"{model_name}_{dataset_name}"
        current_results_dir = os.path.join(results_dir_root, exp_config_name)
        current_visuals_dir = os.path.join(visuals_dir_root, exp_config_name)
        os.makedirs(current_results_dir, exist_ok=True)
        os.makedirs(current_visuals_dir, exist_ok=True)

        # --- Generate Ablation Configurations ---
        base_configs = { # Get fresh base configs for this iteration
            "MILO": OPTIMIZER_PARAMS.get("MILO", {}).copy(),
            "MILO_LW": OPTIMIZER_PARAMS.get("MILO_LW", {}).copy(),
        }

        for opt_type in ["MILO", "MILO_LW"]:
            base_params = base_configs[opt_type]
            if not base_params: continue # Skip if optimizer type not defined

            for param, values in ABLATION_PARAMS.get(opt_type, {}).items():
                group_name = f"{opt_type}_ablate_{param}"
                grouped_configs_for_runs[group_name] = []

                # 1. Add Baseline Config (using actual value in name)
                if param not in base_params:
                    print(f"Warning: Baseline parameter '{param}' not found in base config for {opt_type}. Skipping ablation group.")
                    continue # Skip this parameter ablation if the base param doesn't exist

                baseline_value = base_params[param]
                baseline_name = f"{opt_type}_{param}_{baseline_value}" # Use actual value

                # Ensure baseline value is included in the values to test if not already present
                # This handles cases where the default isn't explicitly listed in ABLATION_PARAMS
                if baseline_value not in values:
                    values = [baseline_value] + list(values) # Prepend baseline value

                ABLATION_CONFIGS_STORE[baseline_name] = {
                    "base_optimizer": opt_type,
                    "params": copy.deepcopy(base_params),
                    "dataset_name": dataset_name                }
                grouped_configs_for_runs[group_name].append(baseline_name)

                # 2. Add Parameter Variations
                for value in values:
                    # Skip creating a variation if it's the same as the baseline (already added)
                    if value == baseline_value: continue
                    # Skip scale_factor variation if scale_aware is not enabled in the base config
                    if param == "scale_factor" and not base_params.get("scale_aware", False): continue

                    config_name = f"{opt_type}_{param}_{value}"
                    # Avoid duplicate config names if the same value appears multiple times (unlikely but safe)
                    if config_name in ABLATION_CONFIGS_STORE: continue

                    current_params_variation = copy.deepcopy(base_params)
                    current_params_variation[param] = value

                    # Ensure consistency for scale_aware/layer_wise flags based on variation
                    if param == "scale_factor": current_params_variation["scale_aware"] = True
                    if opt_type == "MILO_LW": current_params_variation["layer_wise"] = True
                    elif opt_type == "MILO": current_params_variation["layer_wise"] = False

                    ABLATION_CONFIGS_STORE[config_name] = {
                        "base_optimizer": opt_type,
                        "params": current_params_variation,
                        "dataset_name": dataset_name # Store dataset context
                    }
                    grouped_configs_for_runs[group_name].append(config_name)

        # --- Run Experiments for Each Ablation Group ---
        for group_name, config_names_list in grouped_configs_for_runs.items():
            if len(config_names_list) <= 1:
                print(f"Skipping ablation group '{group_name}' for {exp_config_name} - no variations found.")
                continue

            # Define directories specific to this ablation group
            group_results_dir = os.path.join(current_results_dir, group_name)
            group_visuals_dir = os.path.join(current_visuals_dir, group_name)
            os.makedirs(group_results_dir, exist_ok=True)
            os.makedirs(group_visuals_dir, exist_ok=True)

            # --- Check if results already exist (conditional) ---
            csv_filename = f"{group_name}_metrics.csv"
            results_csv_path = os.path.join(group_results_dir, csv_filename)
            if SKIP_EXISTING_ABLATION_GROUPS and os.path.exists(results_csv_path): 
                print(f"Skipping ablation group '{group_name}' for {exp_config_name} - results file already exists: {results_csv_path}")
                continue
            # --- End Check ---

            print(f"\n--- Running Ablation Group: {group_name} for {exp_config_name} ---")
            print(f"Comparing Configurations: {config_names_list}")

            # Create the specific training function for this model/dataset
            train_fn = create_train_experiment_fn(model_instance, train_loader_instance)

            # Run the experiment group
            run_experiments(
                train_fn, # Pass the dynamically created training function
                group_results_dir,
                group_visuals_dir,
                EPOCHS,
                optimizer_names=config_names_list, # These are the config names
                loss_title=f"{exp_config_name} Ablation {group_name}: Loss",
                acc_title=f"{exp_config_name} Ablation {group_name}: Accuracy",
                plot_filename=f"{group_name}_curves",
                csv_filename=csv_filename, 
                experiment_title=f"{exp_config_name} Ablation: {group_name}",
                num_runs=RUNS_PER_OPTIMIZER
            )

        print(f"\n{'='*20} Finished Ablation for {model_name} on {dataset_name} {'='*20}")

    print("\n--- Full Ablation Study Complete ---")

