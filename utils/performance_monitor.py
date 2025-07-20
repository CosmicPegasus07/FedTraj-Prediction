"""
Performance monitoring utilities for PyG data loading optimization
"""

import time
import psutil
import torch
import gc
from contextlib import contextmanager
from typing import Dict, Any
import matplotlib.pyplot as plt
import json
from pathlib import Path

class PerformanceMonitor:
    """Monitor system performance during data loading"""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = None
        self.process = psutil.Process()
        
    def start_monitoring(self, task_name: str):
        """Start monitoring a task"""
        self.start_time = time.time()
        self.metrics[task_name] = {
            'start_time': self.start_time,
            'start_memory': self.process.memory_info().rss / 1024 / 1024,  # MB
            'start_cpu_percent': self.process.cpu_percent(),
            'gpu_memory_start': self._get_gpu_memory() if torch.cuda.is_available() else 0
        }
        
    def end_monitoring(self, task_name: str, additional_info: Dict[str, Any] = None):
        """End monitoring a task"""
        if task_name not in self.metrics:
            return
            
        end_time = time.time()
        self.metrics[task_name].update({
            'end_time': end_time,
            'duration': end_time - self.metrics[task_name]['start_time'],
            'end_memory': self.process.memory_info().rss / 1024 / 1024,  # MB
            'end_cpu_percent': self.process.cpu_percent(),
            'gpu_memory_end': self._get_gpu_memory() if torch.cuda.is_available() else 0,
            'peak_memory': self.process.memory_info().peak_wss / 1024 / 1024 if hasattr(self.process.memory_info(), 'peak_wss') else 0
        })
        
        if additional_info:
            self.metrics[task_name].update(additional_info)
            
        # Calculate derived metrics
        self.metrics[task_name]['memory_increase'] = (
            self.metrics[task_name]['end_memory'] - self.metrics[task_name]['start_memory']
        )
        self.metrics[task_name]['gpu_memory_increase'] = (
            self.metrics[task_name]['gpu_memory_end'] - self.metrics[task_name]['gpu_memory_start']
        )
        
    def _get_gpu_memory(self):
        """Get current GPU memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0
        
    def print_summary(self, task_name: str = None):
        """Print performance summary"""
        if task_name:
            tasks = [task_name] if task_name in self.metrics else []
        else:
            tasks = list(self.metrics.keys())
            
        for task in tasks:
            metrics = self.metrics[task]
            print(f"\n=== Performance Summary: {task} ===")
            print(f"Duration: {metrics['duration']:.2f} seconds")
            print(f"Memory usage: {metrics['start_memory']:.1f} MB -> {metrics['end_memory']:.1f} MB")
            print(f"Memory increase: {metrics['memory_increase']:.1f} MB")
            if torch.cuda.is_available():
                print(f"GPU memory: {metrics['gpu_memory_start']:.1f} MB -> {metrics['gpu_memory_end']:.1f} MB")
                print(f"GPU memory increase: {metrics['gpu_memory_increase']:.1f} MB")
            
            # Additional metrics if available
            if 'num_scenarios' in metrics:
                scenarios_per_sec = metrics['num_scenarios'] / metrics['duration']
                print(f"Processing rate: {scenarios_per_sec:.2f} scenarios/second")
                print(f"Time per scenario: {metrics['duration']/metrics['num_scenarios']*1000:.2f} ms")
                
    def save_metrics(self, filepath: str):
        """Save metrics to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
    def load_metrics(self, filepath: str):
        """Load metrics from JSON file"""
        with open(filepath, 'r') as f:
            self.metrics = json.load(f)

@contextmanager
def monitor_performance(task_name: str, monitor: PerformanceMonitor = None, **kwargs):
    """Context manager for performance monitoring"""
    if monitor is None:
        monitor = PerformanceMonitor()
    
    monitor.start_monitoring(task_name)
    try:
        yield monitor
    finally:
        monitor.end_monitoring(task_name, kwargs)

def compare_performance(old_metrics_file: str, new_metrics_file: str):
    """Compare performance between old and new implementations"""
    old_monitor = PerformanceMonitor()
    new_monitor = PerformanceMonitor()
    
    old_monitor.load_metrics(old_metrics_file)
    new_monitor.load_metrics(new_metrics_file)
    
    print("\n=== Performance Comparison ===")
    
    for task_name in old_monitor.metrics:
        if task_name in new_monitor.metrics:
            old_metrics = old_monitor.metrics[task_name]
            new_metrics = new_monitor.metrics[task_name]
            
            duration_improvement = (old_metrics['duration'] - new_metrics['duration']) / old_metrics['duration'] * 100
            memory_change = new_metrics['memory_increase'] - old_metrics['memory_increase']
            
            print(f"\nTask: {task_name}")
            print(f"Duration: {old_metrics['duration']:.2f}s -> {new_metrics['duration']:.2f}s ({duration_improvement:+.1f}%)")
            print(f"Memory increase: {old_metrics['memory_increase']:.1f}MB -> {new_metrics['memory_increase']:.1f}MB ({memory_change:+.1f}MB)")
            
            if 'num_scenarios' in old_metrics and 'num_scenarios' in new_metrics:
                old_rate = old_metrics['num_scenarios'] / old_metrics['duration']
                new_rate = new_metrics['num_scenarios'] / new_metrics['duration']
                rate_improvement = (new_rate - old_rate) / old_rate * 100
                print(f"Processing rate: {old_rate:.2f} -> {new_rate:.2f} scenarios/sec ({rate_improvement:+.1f}%)")

def benchmark_data_loading(scenario_dir: str, num_scenarios: int = 100, seq_len: int = 30):
    """Benchmark data loading performance"""
    from utils.data_utils import get_pyg_data_loader, get_cache_info, clear_cache
    
    monitor = PerformanceMonitor()
    
    print(f"Benchmarking data loading with {num_scenarios} scenarios...")
    
    # Clear cache for fair comparison
    clear_cache()
    
    # Test without cache
    monitor.start_monitoring("data_loading_no_cache")
    loader = get_pyg_data_loader(
        scenario_dir=scenario_dir,
        batch_size=1,
        num_scenarios=num_scenarios,
        shuffle=False,
        mode='train',
        seq_len=seq_len
    )
    monitor.end_monitoring("data_loading_no_cache", {"num_scenarios": num_scenarios})
    
    # Test with cache
    monitor.start_monitoring("data_loading_with_cache")
    loader2 = get_pyg_data_loader(
        scenario_dir=scenario_dir,
        batch_size=1,
        num_scenarios=num_scenarios,
        shuffle=False,
        mode='train',
        seq_len=seq_len
    )
    monitor.end_monitoring("data_loading_with_cache", {"num_scenarios": num_scenarios})
    
    # Print results
    monitor.print_summary()
    
    # Cache info
    cache_info = get_cache_info()
    print(f"\nCache info: {cache_info['cached_files']} files, {cache_info['cache_size_mb']:.1f} MB")
    
    return monitor

def system_info():
    """Print system information"""
    print("=== System Information ===")
    print(f"CPU cores: {psutil.cpu_count()}")
    print(f"Available memory: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.1f} GB")
    print(f"Total memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.1f} GB")
    else:
        print("GPU: Not available")
