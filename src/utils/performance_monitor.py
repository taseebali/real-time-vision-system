"""
Performance monitoring and profiling utilities
"""

import time
import psutil
import torch
from collections import deque
from typing import Dict, List, Optional
import threading

class PerformanceMonitor:
    """Monitor and log performance metrics"""
    
    def __init__(self, window_size: int = 30):
        """
        Initialize performance monitor.
        
        Args:
            window_size: Number of samples to keep for rolling averages
        """
        self.window_size = window_size
        self.metrics = {
            'object_detection': deque(maxlen=window_size),
            'scene_captioning': deque(maxlen=window_size),
            'text_detection': deque(maxlen=window_size),
            'narration_generation': deque(maxlen=window_size),
            'total_processing': deque(maxlen=window_size),
            'fps': deque(maxlen=window_size)
        }
        self.counters = {
            'frames_processed': 0,
            'frames_skipped': 0,
            'captions_generated': 0,
            'narrations_generated': 0
        }
        self.start_time = time.time()
        self.last_report_time = time.time()
        self.lock = threading.Lock()
    
    def record_metric(self, metric_name: str, duration: float):
        """Record a performance metric"""
        with self.lock:
            if metric_name in self.metrics:
                self.metrics[metric_name].append(duration)
    
    def increment_counter(self, counter_name: str, value: int = 1):
        """Increment a counter"""
        with self.lock:
            if counter_name in self.counters:
                self.counters[counter_name] += value
    
    def get_average(self, metric_name: str) -> float:
        """Get average value for a metric"""
        with self.lock:
            if metric_name in self.metrics and len(self.metrics[metric_name]) > 0:
                return sum(self.metrics[metric_name]) / len(self.metrics[metric_name])
            return 0.0
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics"""
        with self.lock:
            stats = {
                'averages': {},
                'counters': self.counters.copy(),
                'uptime': time.time() - self.start_time
            }
            
            for metric_name, values in self.metrics.items():
                if len(values) > 0:
                    stats['averages'][metric_name] = {
                        'mean': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values),
                        'recent': values[-1] if values else 0
                    }
            
            # GPU stats
            if torch.cuda.is_available():
                stats['gpu'] = {
                    'memory_allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                    'memory_reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
                }
                
                # Try to get GPU utilization (requires pynvml and NVIDIA drivers)
                try:
                    if hasattr(torch.cuda, 'utilization'):
                        stats['gpu']['utilization'] = torch.cuda.utilization()
                    else:
                        stats['gpu']['utilization'] = None
                except Exception:
                    # Catch all exceptions (ModuleNotFoundError, RuntimeError, NVMLError, etc.)
                    stats['gpu']['utilization'] = None
            
            # CPU/RAM stats
            stats['system'] = {
                'cpu_percent': psutil.cpu_percent(),
                'ram_percent': psutil.virtual_memory().percent,
                'ram_used_gb': psutil.virtual_memory().used / 1024**3
            }
            
            return stats
    
    def print_report(self, force: bool = False):
        """Print performance report"""
        current_time = time.time()
        
        # Print every 10 seconds or when forced
        if not force and (current_time - self.last_report_time) < 10:
            return
        
        self.last_report_time = current_time
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("PERFORMANCE REPORT")
        print("="*60)
        
        # Timing stats
        print("\nTiming (ms):")
        for metric, values in stats['averages'].items():
            print(f"  {metric:25s}: {values['mean']*1000:6.1f} "
                  f"(min: {values['min']*1000:5.1f}, max: {values['max']*1000:5.1f})")
        
        # Counters
        print("\nCounters:")
        for counter, value in stats['counters'].items():
            print(f"  {counter:25s}: {value}")
        
        # Calculate rates
        uptime = stats['uptime']
        if uptime > 0:
            print("\nRates:")
            print(f"  Processing FPS:           {stats['counters']['frames_processed'] / uptime:.2f}")
            print(f"  Skip ratio:               {stats['counters']['frames_skipped'] / max(1, stats['counters']['frames_processed']) * 100:.1f}%")
        
        # System stats
        if 'gpu' in stats:
            print("\nGPU:")
            print(f"  Memory allocated:         {stats['gpu']['memory_allocated']:.2f} GB")
            print(f"  Memory reserved:          {stats['gpu']['memory_reserved']:.2f} GB")
            if stats['gpu'].get('utilization') is not None:
                print(f"  Utilization:              {stats['gpu']['utilization']}%")
        
        print("\nSystem:")
        print(f"  CPU usage:                {stats['system']['cpu_percent']:.1f}%")
        print(f"  RAM usage:                {stats['system']['ram_percent']:.1f}% "
              f"({stats['system']['ram_used_gb']:.2f} GB)")
        
        print("="*60 + "\n")


class Timer:
    """Context manager for timing code blocks"""
    
    def __init__(self, monitor: PerformanceMonitor, metric_name: str):
        self.monitor = monitor
        self.metric_name = metric_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.monitor.record_metric(self.metric_name, duration)
        return False
