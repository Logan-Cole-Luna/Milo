"""
Utilities for tracking and saving compute resource usage during experiments.

This module provides a `ResourceTracker` class to monitor CPU, memory, and disk usage,
as well as system information. It also includes a function to save this information
to a CSV file.
"""

import os
import sys
import time
import platform
import psutil
import torch
import datetime
import json
import pandas as pd
from typing import Dict, Any, Optional

class ResourceTracker:
    """Track compute resources used during experiments."""
    
    def __init__(self):
        """
        Initialize the ResourceTracker.
        
        Sets the start time and collects initial system information.
        """
        self.start_time = time.time()
        self.end_time = None
        self.system_info = self._collect_system_info()
        
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect information about the system.

        Returns:
            Dict[str, Any]: A dictionary containing system platform, Python version,
                            processor, CPU counts, memory, disk usage, hostname,
                            CUDA availability, and GPU details if available.
        """
        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "disk_usage_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
            "hostname": platform.node(),
            "cuda_available": torch.cuda.is_available(),
            "compute_type": "GPU" if torch.cuda.is_available() else "CPU",
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        # Add GPU information if available
        if torch.cuda.is_available():
            try:
                info["gpu_count"] = torch.cuda.device_count()
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_memory_total_gb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
                )
            except Exception as e:
                info["gpu_error"] = str(e)
        
        return info
    
    def start(self):
        """Start tracking resources.
        
        Records the start time and initial process-specific memory and CPU usage.

        Returns:
            ResourceTracker: The instance of the tracker.
        """
        self.start_time = time.time()
        self.start_process_info = {
            "memory_used_gb": round(psutil.Process(os.getpid()).memory_info().rss / (1024**3), 2),
            "cpu_percent": psutil.Process(os.getpid()).cpu_percent(),
        }
        return self
        
    def stop(self):
        """Stop tracking resources.
        
        Records the end time, calculates the duration, and captures
        final process-specific memory and CPU usage.

        Returns:
            ResourceTracker: The instance of the tracker.
        """
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.end_process_info = {
            "memory_used_gb": round(psutil.Process(os.getpid()).memory_info().rss / (1024**3), 2),
            "cpu_percent": psutil.Process(os.getpid()).cpu_percent(),
        }
        return self
    
    def get_info(self) -> Dict[str, Any]:
        """Get all tracked information.

        Returns:
            Dict[str, Any]: A dictionary containing the duration of tracking,
                            start time, all system information, and process-specific
                            metrics at the start and end of tracking, including
                            memory increase.
        """
        info = {
            "duration_seconds": round(self.duration, 2) if hasattr(self, "duration") else None,
            "start_time": datetime.datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S"),
            **self.system_info,
        }
        
        if hasattr(self, "start_process_info"):
            info.update({f"start_{k}": v for k, v in self.start_process_info.items()})
            
        if hasattr(self, "end_process_info"):
            info.update({f"end_{k}": v for k, v in self.end_process_info.items()})
            info["memory_increase_gb"] = round(
                info.get("end_memory_used_gb", 0) - info.get("start_memory_used_gb", 0), 2
            )
            
        return info

def save_resource_info(info_list: list, output_path: str) -> pd.DataFrame:
    """Save resource information to a CSV file and return the DataFrame.

    Args:
        info_list (list): A list of dictionaries, where each dictionary
                          is the output of `ResourceTracker.get_info()`.
        output_path (str): The path to save the CSV file.

    Returns:
        pd.DataFrame: The DataFrame containing the saved resource information.
    """
    df = pd.DataFrame(info_list)
    df.to_csv(output_path, index=False)
    print(f"Resource information saved to {output_path}")
    return df
