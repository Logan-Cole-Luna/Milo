#!/usr/bin/env python3
"""
Custom plotting script for training results with flexible configuration.

This script allows you to generate publication-quality plots from saved training results
with full control over:
- Which optimizers to include/exclude
- Plot bounds and axis limits
- Color schemes and styling
- Figure size and DPI
- Custom titles and labels
- Multiple metrics on subplots

Usage:
    python custom_plotting.py --config plot_config.json
    python custom_plotting.py --results-dir path/to/results --optimizers MILO SGD ADAMW
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from plotting import plot_seaborn_style_with_error_bars, setup_plot_style

class CustomPlotter:
    """Flexible plotter for training experiment results."""
    
    def __init__(self, config: Dict):
        """
        Initialize plotter with configuration.
        
        Args:
            config: Dictionary containing plotting configuration
        """
        self.config = config
        self.setup_style()
        
    def setup_style(self):
        """Setup plot styling based on configuration."""
        # Apply base style
        setup_plot_style()
        
        # Override with custom settings if provided
        style_config = self.config.get('style', {})
        
        if 'figure_size' in style_config:
            plt.rcParams['figure.figsize'] = style_config['figure_size']
        if 'dpi' in style_config:
            plt.rcParams['figure.dpi'] = style_config['dpi']
        if 'font_size' in style_config:
            plt.rcParams['font.size'] = style_config['font_size']
        if 'line_width' in style_config:
            plt.rcParams['lines.linewidth'] = style_config['line_width']
            
    def load_results(self, results_path: str) -> pd.DataFrame:
        """
        Load training results from CSV file.
        
        Args:
            results_path: Path to CSV file containing training metrics
            
        Returns:
            DataFrame with training results
        """
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Results file not found: {results_path}")
            
        df = pd.read_csv(results_path)
        
        # Filter optimizers if specified and values are not None/empty
        include_opts = self.config.get('include_optimizers')
        if include_opts:
            df = df[df['optimizer'].isin(include_opts)]

        exclude_opts = self.config.get('exclude_optimizers')
        if exclude_opts:
            df = df[~df['optimizer'].isin(exclude_opts)]
            
        return df
        
    def prepare_data_for_plotting(self, df: pd.DataFrame, metric: str) -> Tuple[Dict, Dict]:
        """
        Prepare data for plotting by computing means and standard errors.
        
        Args:
            df: DataFrame with training results
            metric: Metric name to plot (e.g., 'train_accuracy', 'train_loss')
            
        Returns:
            Tuple of (means_dict, std_err_dict) for plotting
        """
        data = {}
        data_std_err = {}
        
        # Get unique optimizers
        optimizers = df['optimizer'].unique()
        
        for optimizer in optimizers:
            opt_data = df[df['optimizer'] == optimizer]
            
            # Group by epoch and compute statistics
            grouped = opt_data.groupby('epoch')[metric].agg(['mean', 'std', 'count'])
            
            # Compute standard error
            grouped['std_err'] = grouped['std'] / np.sqrt(grouped['count'])
            
            data[optimizer] = grouped['mean'].tolist()
            data_std_err[optimizer] = grouped['std_err'].tolist()
            
        return data, data_std_err
        
    def get_x_values(self, df: pd.DataFrame) -> List:
        """Get x-axis values (epochs) from DataFrame."""
        return sorted(df['epoch'].unique())
        
    def plot_metric(self, df: pd.DataFrame, metric: str, output_path: str, 
                   title: Optional[str] = None, ylabel: Optional[str] = None):
        """
        Plot a specific metric with customizable options.
        
        Args:
            df: DataFrame with training results
            metric: Metric to plot
            output_path: Path to save the plot
            title: Custom title (optional)
            ylabel: Custom y-axis label (optional)
        """
        # Prepare data
        data, data_std_err = self.prepare_data_for_plotting(df, metric)
        x_values = self.get_x_values(df)
        
        # Get configuration for this metric
        metric_config = self.config.get('metrics', {}).get(metric, {})
        
        # Set default title and ylabel if not provided
        if title is None:
            title = metric_config.get('title', f'{metric.replace("_", " ").title()} vs. Epoch')
        if ylabel is None:
            ylabel = metric_config.get('ylabel', metric.replace('_', ' ').title())
            
        # Get axis limits
        xlimit = metric_config.get('xlimit', self.config.get('global', {}).get('xlimit'))
        ylimit = metric_config.get('ylimit')
        yscale = metric_config.get('yscale')
        
        # Create the plot
        plot_seaborn_style_with_error_bars(
            data=data,
            data_std_err=data_std_err,
            x_values=x_values,
            title=title,
            filename=os.path.splitext(os.path.basename(output_path))[0],
            y_label=ylabel,
            visuals_dir=os.path.dirname(output_path),
            xlabel="Epoch",
            xlimit=xlimit,
            yscale=yscale
        )
        
        # Apply y-axis limits if specified
        if ylimit:
            plt.ylim(ylimit)
            
        print(f"Saved plot: {output_path}")
        
    def plot_multiple_metrics(self, df: pd.DataFrame, output_dir: str):
        """
        Plot multiple metrics based on configuration.
        
        Args:
            df: DataFrame with training results
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get metrics to plot
        metrics_to_plot = self.config.get('plot_metrics', [
            'train_loss', 'train_accuracy', 'train_f1_score', 'train_auc'
        ])
        
        # Plot each metric
        for metric in metrics_to_plot:
            if metric in df.columns:
                output_path = os.path.join(output_dir, f"{metric}_plot.png")
                self.plot_metric(df, metric, output_path)
            else:
                print(f"Warning: Metric '{metric}' not found in data. Skipping.")
                
    def create_subplot_comparison(self, df: pd.DataFrame, output_path: str, 
                                 metrics: List[str], subplot_titles: Optional[List[str]] = None):
        """
        Create a subplot comparison of multiple metrics.
        
        Args:
            df: DataFrame with training results
            output_path: Path to save the plot
            metrics: List of metrics to plot
            subplot_titles: Optional list of titles for subplots
        """
        n_metrics = len(metrics)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), dpi=300)
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
            
        for i, metric in enumerate(metrics):
            if metric not in df.columns:
                print(f"Warning: Metric '{metric}' not found in data. Skipping subplot.")
                continue
                
            ax = axes[i]
            plt.sca(ax)
            
            # Prepare data for this metric
            data, data_std_err = self.prepare_data_for_plotting(df, metric)
            x_values = self.get_x_values(df)
            
            # Get configuration for this metric
            metric_config = self.config.get('metrics', {}).get(metric, {})
            
            # Plot with current axis
            for j, (optimizer, values) in enumerate(data.items()):
                color = plt.cm.tab10(j)
                std_err = data_std_err.get(optimizer, [0] * len(values))
                
                ax.plot(x_values[:len(values)], values, label=optimizer, 
                       color=color, linewidth=2, marker='o', markersize=4)
                ax.fill_between(x_values[:len(values)], 
                               np.array(values) - np.array(std_err),
                               np.array(values) + np.array(std_err),
                               color=color, alpha=0.2)
                               
            # Customize subplot
            title = subplot_titles[i] if subplot_titles and i < len(subplot_titles) else metric.replace('_', ' ').title()
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_config.get('ylabel', metric.replace('_', ' ').title()))
            ax.grid(True, alpha=0.3)
            
            # Apply limits if specified
            xlimit = metric_config.get('xlimit', self.config.get('global', {}).get('xlimit'))
            ylimit = metric_config.get('ylimit')
            if xlimit:
                ax.set_xlim(right=xlimit)
            if ylimit:
                ax.set_ylim(ylimit)
                
            # Add legend only to first subplot
            if i == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
        # Remove extra subplots
        for i in range(n_metrics, len(axes)):
            fig.delaxes(axes[i])
            
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved subplot comparison: {output_path}")


def create_default_config() -> Dict:
    """Create a default configuration dictionary."""
    return {
        "style": {
            "figure_size": [10, 6],
            "dpi": 300,
            "font_size": 12,
            "line_width": 2
        },
        "global": {
            "xlimit": None,
        },
        "plot_metrics": [
            "train_loss",
            "train_accuracy", 
            "train_f1_score",
            "train_auc"
        ],
        "metrics": {
            "train_loss": {
                "title": "Training Loss vs. Epoch",
                "ylabel": "Loss",
                "yscale": None,
                "ylimit": None,
                "xlimit": None
            },
            "train_accuracy": {
                "title": "Training Accuracy vs. Epoch", 
                "ylabel": "Accuracy (%)",
                "yscale": None,
                "ylimit": [85, 95],
                "xlimit": None
            },
            "train_f1_score": {
                "title": "Training F1 Score vs. Epoch",
                "ylabel": "F1 Score",
                "yscale": None,
                "ylimit": [0.85, 0.95],
                "xlimit": None
            },
            "train_auc": {
                "title": "Training AUC vs. Epoch",
                "ylabel": "AUC",
                "yscale": None,
                "ylimit": [0.99, 1.0],
                "xlimit": None
            }
        },
        "include_optimizers": None,  # None means include all
        "exclude_optimizers": None   # None means exclude none
    }


def save_default_config(config_path: str):
    """Save default configuration to file."""
    config = create_default_config()
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved default configuration to: {config_path}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Custom plotting for training results")
    parser.add_argument('--results-file', '-r', required=True,
                       help='Path to CSV file with training results')
    parser.add_argument('--config', '-c', 
                       help='Path to JSON configuration file')
    parser.add_argument('--output-dir', '-o', default='./custom_plots',
                       help='Output directory for plots')
    parser.add_argument('--create-config', action='store_true',
                       help='Create a default configuration file')
    parser.add_argument('--optimizers', nargs='+',
                       help='List of optimizers to include (overrides config)')
    parser.add_argument('--exclude-optimizers', nargs='+',
                       help='List of optimizers to exclude (overrides config)')
    parser.add_argument('--xlimit', type=int,
                       help='X-axis limit for all plots (overrides config)')
    parser.add_argument('--subplot', action='store_true',
                       help='Create subplot comparison instead of individual plots')
    
    args = parser.parse_args()
    
    # Create default config if requested
    if args.create_config:
        config_path = args.config or 'plot_config.json'
        save_default_config(config_path)
        return
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
        if args.config:
            print(f"Config file {args.config} not found. Using defaults.")
    
    # Override config with command-line arguments
    if args.optimizers:
        config['include_optimizers'] = args.optimizers
    if args.exclude_optimizers:
        config['exclude_optimizers'] = args.exclude_optimizers
    if args.xlimit:
        config['global']['xlimit'] = args.xlimit
    
    # Create plotter and load data
    plotter = CustomPlotter(config)
    df = plotter.load_results(args.results_file)
    
    print(f"Loaded {len(df)} records for {len(df['optimizer'].unique())} optimizers")
    print(f"Available optimizers: {list(df['optimizer'].unique())}")
    print(f"Available metrics: {[col for col in df.columns if col.startswith('train_')]}")
    
    # Create plots
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.subplot:
        # Create subplot comparison
        metrics = config.get('plot_metrics', ['train_loss', 'train_accuracy', 'train_f1_score'])
        output_path = os.path.join(args.output_dir, 'metrics_comparison.png')
        plotter.create_subplot_comparison(df, output_path, metrics)
    else:
        # Create individual plots
        plotter.plot_multiple_metrics(df, args.output_dir)


if __name__ == '__main__':
    main()
