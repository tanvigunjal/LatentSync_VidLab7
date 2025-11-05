"""
Performance monitoring utilities for tracking CPU, GPU, memory usage,
and timing for each stage of a pipeline across multiple runs.
This is my additional context for the performance monitor. 
"""

import time
import torch
import psutil
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from contextlib import contextmanager

class PerformanceMonitor:
    """
    A comprehensive monitor for tracking CPU, GPU, memory usage, and timing 
    for each stage of a pipeline across multiple runs. It supports context 
    management for easy profiling.
    """
    
    def __init__(self, device, enabled=True):
        self.device = device
        self.enabled = enabled
        self.is_cuda = torch.cuda.is_available() and 'cuda' in str(device)
        self.runs = []
        self.current_run_metrics = None
        self.current_stage = None

    def start_run(self):
        """Start a new profiling run."""
        if not self.enabled:
            return
        self.current_run_metrics = {
            'stage_times': defaultdict(list),
            'stage_memory': defaultdict(lambda: defaultdict(list)),
            'cpu_usage': [],
            'gpu_usage': []
        }

    def end_run(self):
        """End the current profiling run and store its metrics."""
        if not self.enabled or self.current_run_metrics is None:
            return
        self.runs.append(self.current_run_metrics)
        self.current_run_metrics = None

    @contextmanager
    def profile(self, stage_name):
        """A context manager to profile a block of code."""
        if not self.enabled or self.current_run_metrics is None:
            yield
            return
        
        self.start_stage(stage_name)
        try:
            yield
        finally:
            self.end_stage(stage_name)

    def start_stage(self, stage_name):
        """Start monitoring a pipeline stage"""
        if not self.enabled or self.current_run_metrics is None:
            return
            
        self.current_stage = stage_name
        self.stage_start_time = time.time()
        
        if self.is_cuda:
            torch.cuda.synchronize()
            self.stage_start_memory = torch.cuda.memory_allocated(self.device) / 1024**2
        else:
            self.stage_start_memory = psutil.Process().memory_info().rss / 1024**2
    
    def end_stage(self, stage_name):
        """End monitoring a pipeline stage"""
        if not self.enabled or self.current_stage != stage_name or self.current_run_metrics is None:
            return

        if self.is_cuda:
            torch.cuda.synchronize()
        
        elapsed_time = time.time() - self.stage_start_time
        
        if self.is_cuda:
            current_memory = torch.cuda.memory_allocated(self.device) / 1024**2
            peak_memory = torch.cuda.max_memory_allocated(self.device) / 1024**2
        else:
            current_memory = psutil.Process().memory_info().rss / 1024**2
            peak_memory = current_memory
        
        memory_used = current_memory - self.stage_start_memory
        
        self.current_run_metrics['stage_times'][stage_name].append(elapsed_time)
        self.current_run_metrics['stage_memory'][stage_name]['used'].append(memory_used)
        self.current_run_metrics['stage_memory'][stage_name]['peak'].append(peak_memory)
        
        self.current_run_metrics['cpu_usage'].append(psutil.cpu_percent(interval=0.1))
        if self.is_cuda:
            gpu_util = torch.cuda.utilization(self.device) if hasattr(torch.cuda, 'utilization') else 0
            self.current_run_metrics['gpu_usage'].append(gpu_util)
        
        self.current_stage = None

    def get_summary(self):
        """Get summary statistics aggregated across all runs."""
        if not self.runs:
            return {}

        aggregated_times = defaultdict(list)
        aggregated_memory_peak = defaultdict(list)
        aggregated_cpu = []
        aggregated_gpu = []
        
        for run in self.runs:
            for stage, times in run['stage_times'].items():
                aggregated_times[stage].extend(times)
            for stage, mem_data in run['stage_memory'].items():
                aggregated_memory_peak[stage].extend(mem_data['peak'])
            aggregated_cpu.extend(run['cpu_usage'])
            aggregated_gpu.extend(run['gpu_usage'])

        summary = {
            'num_runs': len(self.runs),
            'stage_breakdown': {},
            'total_time_stats': {},
            'total_memory_peak_stats': {},
            'avg_cpu_usage_stats': self._calculate_stats(aggregated_cpu),
            'avg_gpu_usage_stats': self._calculate_stats(aggregated_gpu),
        }

        all_total_times = []
        all_peak_memories = []

        for run in self.runs:
            run_total_time = sum(sum(run['stage_times'].values(), []))
            if run_total_time > 0:
                all_total_times.append(run_total_time)
            
            run_peak_memories = [max(mem['peak']) for mem in run['stage_memory'].values() if mem['peak']]
            if run_peak_memories:
                all_peak_memories.append(max(run_peak_memories))

        summary['total_time_stats'] = self._calculate_stats(all_total_times)
        summary['total_memory_peak_stats'] = self._calculate_stats(all_peak_memories)

        for stage in aggregated_times:
            if not aggregated_times[stage]: continue # Skip stages that were never run
            summary['stage_breakdown'][stage] = {
                'time_stats': self._calculate_stats(aggregated_times[stage]),
                'memory_peak_stats': self._calculate_stats(aggregated_memory_peak[stage])
            }
        return summary
    
    def _calculate_stats(self, data):
        if not data:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data)
        }

    def save_metrics(self, output_dir):
        """Save metrics to JSON and create visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        summary = self.get_summary()
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(summary, f, indent=4)
        
        self.create_visualizations(output_dir, summary)
    
    def create_visualizations(self, output_dir, summary):
        """Create performance visualization graphs from aggregated data"""
        if not summary:
            return
        
        plt.style.use('seaborn-v0_8-darkgrid')
        
        stages = list(summary['stage_breakdown'].keys())
        avg_times = [s['time_stats']['mean'] for s in summary['stage_breakdown'].values()]
        time_std = [s['time_stats']['std'] for s in summary['stage_breakdown'].values()]
        
        # Time distribution pie chart (of averages)
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(stages)))
        ax.pie(avg_times, labels=stages, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Average Pipeline Stage Time Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'time_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Time comparison bar chart with error bars
        fig, ax = plt.subplots(figsize=(12, 6))
        x_pos = np.arange(len(stages))
        ax.bar(x_pos, avg_times, yerr=time_std, capsize=5, color='skyblue', alpha=0.8, edgecolor='black')
        ax.set_xlabel('Pipeline Stage', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Average Execution Time by Stage (with Std Dev)', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stages, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'time_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Memory usage chart
        avg_mem_peak = [s['memory_peak_stats']['mean'] for s in summary['stage_breakdown'].values()]
        mem_std = [s['memory_peak_stats']['std'] for s in summary['stage_breakdown'].values()]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x_pos, avg_mem_peak, yerr=mem_std, capsize=5, label='Peak Memory', color='coral', alpha=0.8, edgecolor='black')
        ax.set_xlabel('Pipeline Stage', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Peak Memory (MB)', fontsize=12, fontweight='bold')
        ax.set_title('Average Peak Memory by Stage (with Std Dev)', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stages, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'memory_usage.png'), dpi=300, bbox_inches='tight')
        plt.close()