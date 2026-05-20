#!/usr/bin/env python3
"""
Example usage script for custom plotting.

This script demonstrates various ways to use the custom plotting functionality
to create publication-quality plots from training results.
"""

import os
import sys
from pathlib import Path

# Add experiments directory to path
sys.path.append(str(Path(__file__).parent))
from custom_plotting import CustomPlotter, create_default_config

def example_basic_usage():
    """Example 1: Basic usage with default configuration."""
    print("=== Example 1: Basic Usage ===")
    
    # Use default configuration
    config = create_default_config()
    
    # Focus on specific optimizers
    config['include_optimizers'] = ['MILO', 'MILO_TUNED', 'SGD']
    
    # Set axis limits for better comparison
    config['global']['xlimit'] = 10
    config['metrics']['train_accuracy']['ylimit'] = [91, 94]
    
    # Create plotter
    plotter = CustomPlotter(config)
    
    # Load and plot data (example path - adjust as needed)
    results_file = "supervised_learning/logistic/results/logistic_training_metrics.csv"
    if os.path.exists(results_file):
        df = plotter.load_results(results_file)
        plotter.plot_multiple_metrics(df, "output/basic_plots")
        print("✓ Basic plots created in output/basic_plots/")
    else:
        print(f"Results file not found: {results_file}")


def example_publication_quality():
    """Example 2: Publication-quality plots with custom styling."""
    print("\\n=== Example 2: Publication Quality ===")
    
    config = {
        "style": {
            "figure_size": [10, 6],
            "dpi": 300,
            "font_size": 16,
            "line_width": 3
        },
        "global": {
            "xlimit": 10
        },
        "plot_metrics": ["train_accuracy", "train_loss"],
        "metrics": {
            "train_accuracy": {
                "title": "MILO Optimization: Training Accuracy Comparison",
                "ylabel": "Training Accuracy (%)",
                "ylimit": [91.5, 93.5],
                "xlimit": 10
            },
            "train_loss": {
                "title": "MILO Optimization: Loss Convergence",
                "ylabel": "Cross-Entropy Loss",
                "ylimit": [0.24, 0.33],
                "xlimit": 10
            }
        },
        "include_optimizers": ["MILO", "MILO_LW", "MILO_TUNED", "MILO_LW_TUNED", "SGD"],
        "exclude_optimizers": None
    }
    
    plotter = CustomPlotter(config)
    
    results_file = "supervised_learning/logistic/results/logistic_training_metrics.csv"
    if os.path.exists(results_file):
        df = plotter.load_results(results_file)
        plotter.plot_multiple_metrics(df, "output/publication_plots")
        print("✓ Publication plots created in output/publication_plots/")
    else:
        print(f"Results file not found: {results_file}")


def example_comparison_subplot():
    """Example 3: Create comparison subplots."""
    print("\\n=== Example 3: Comparison Subplots ===")
    
    config = {
        "style": {
            "figure_size": [16, 10],
            "dpi": 300,
            "font_size": 14
        },
        "global": {
            "xlimit": 10
        },
        "metrics": {
            "train_accuracy": {
                "ylabel": "Accuracy (%)",
                "ylimit": [91, 94]
            },
            "train_loss": {
                "ylabel": "Loss",
                "ylimit": [0.24, 0.33]
            },
            "train_f1_score": {
                "ylabel": "F1 Score",
                "ylimit": [0.91, 0.95]
            }
        },
        "include_optimizers": ["MILO", "MILO_TUNED", "SGD"]
    }
    
    plotter = CustomPlotter(config)
    
    results_file = "supervised_learning/logistic/results/logistic_training_metrics.csv"
    if os.path.exists(results_file):
        df = plotter.load_results(results_file)
        
        # Create subplot comparison
        metrics = ["train_accuracy", "train_loss", "train_f1_score"]
        subplot_titles = ["Training Accuracy", "Training Loss", "F1 Score"]
        
        os.makedirs("output/subplot_comparison", exist_ok=True)
        plotter.create_subplot_comparison(
            df, 
            "output/subplot_comparison/milo_comparison.png", 
            metrics, 
            subplot_titles
        )
        print("✓ Subplot comparison created in output/subplot_comparison/")
    else:
        print(f"Results file not found: {results_file}")


def example_milo_variants_focus():
    """Example 4: Focus specifically on MILO variants."""
    print("\\n=== Example 4: MILO Variants Focus ===")
    
    config = {
        "style": {
            "figure_size": [12, 8],
            "dpi": 300,
            "font_size": 15
        },
        "global": {
            "xlimit": 10
        },
        "plot_metrics": ["train_accuracy", "train_loss"],
        "metrics": {
            "train_accuracy": {
                "title": "MILO Variant Comparison: Training Accuracy",
                "ylabel": "Training Accuracy (%)",
                "ylimit": [91.5, 93.5]
            },
            "train_loss": {
                "title": "MILO Variant Comparison: Loss Convergence", 
                "ylabel": "Cross-Entropy Loss",
                "ylimit": [0.24, 0.30]
            }
        },
        "include_optimizers": ["MILO", "MILO_LW", "MILO_TUNED", "MILO_LW_TUNED"],
        "exclude_optimizers": None
    }
    
    plotter = CustomPlotter(config)
    
    results_file = "supervised_learning/logistic/results/logistic_training_metrics.csv"
    if os.path.exists(results_file):
        df = plotter.load_results(results_file)
        plotter.plot_multiple_metrics(df, "output/milo_variants")
        print("✓ MILO variants plots created in output/milo_variants/")
    else:
        print(f"Results file not found: {results_file}")


def main():
    """Run all examples."""
    print("Custom Plotting Examples")
    print("=" * 50)
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Run examples
    example_basic_usage()
    example_publication_quality()
    example_comparison_subplot()
    example_milo_variants_focus()
    
    print("\\n" + "=" * 50)
    print("All examples completed!")
    print("\\nCommand-line usage examples:")
    print("1. Basic plot:")
    print("   python custom_plotting.py -r logistic_training_metrics.csv -o plots/")
    print("\\n2. With specific optimizers:")
    print("   python custom_plotting.py -r logistic_training_metrics.csv --optimizers MILO SGD -o plots/")
    print("\\n3. With configuration file:")
    print("   python custom_plotting.py -r logistic_training_metrics.csv -c example_plot_config.json -o plots/")
    print("\\n4. Create subplot comparison:")
    print("   python custom_plotting.py -r logistic_training_metrics.csv --subplot -o plots/")
    print("\\n5. Create default config file:")
    print("   python custom_plotting.py --create-config -c my_config.json")


if __name__ == "__main__":
    main()
