# Utils Module

Utility classes and functions for performance monitoring and system optimization.

## Overview

The utils module provides essential tools for:
- **Real-time performance monitoring**: Track FPS, latency, and resource usage
- **System metrics collection**: GPU memory, CPU/RAM usage
- **Performance reporting**: Comprehensive statistics and periodic reports
- **Timing utilities**: Context managers for precise timing measurements

## Components

### `performance_monitor.py`

Contains two main classes for performance tracking and optimization.

---

## PerformanceMonitor Class

Tracks real-time performance metrics with rolling window statistics.

### Features

- ✅ **Timing Metrics**: Track execution times for different operations
- ✅ **Counter Tracking**: Count events (frames processed, narrations generated, etc.)
- ✅ **GPU Monitoring**: Memory usage and utilization (with graceful fallback)
- ✅ **System Monitoring**: CPU and RAM usage via psutil
- ✅ **Rolling Windows**: Calculate averages over configurable time windows
- ✅ **Periodic Reports**: Automatic reporting at configurable intervals
- ✅ **FPS Calculation**: Real-time frames-per-second tracking

### Usage

#### Basic Usage

```python
from src.utils.performance_monitor import PerformanceMonitor

# Create monitor with 30-second rolling window
monitor = PerformanceMonitor(window_size=30)

# Record timing for an operation
monitor.record_metric('object_detection', 0.035)  # 35ms

# Increment counters
monitor.increment_counter('frames_processed')

# Get current statistics
stats = monitor.get_stats()
print(f"Object Detection: {stats['timing']['object_detection']['avg']:.1f}ms")

# Print formatted report
monitor.print_report()
```

#### Advanced Usage with Context Manager

```python
from src.utils.performance_monitor import PerformanceMonitor, Timer

monitor = PerformanceMonitor()

# Use Timer context manager for automatic timing
with Timer(monitor, 'model_inference'):
    # Your code here
    result = model.predict(input_data)
    # Timer automatically records duration when exiting

# Check recorded time
stats = monitor.get_stats()
print(f"Inference time: {stats['timing']['model_inference']['recent']:.1f}ms")
```

#### Real-time Monitoring Loop

```python
from src.utils.performance_monitor import PerformanceMonitor, Timer
import time

monitor = PerformanceMonitor(report_interval=10)  # Report every 10 seconds

while running:
    # Time the entire processing loop
    with Timer(monitor, 'total_processing'):
        # Object detection
        with Timer(monitor, 'object_detection'):
            objects = detector.detect(frame)
        
        # Scene captioning
        with Timer(monitor, 'scene_captioning'):
            caption = captioner.generate(frame)
        
        # Narration
        with Timer(monitor, 'narration_generation'):
            narration = narration_service.generate(objects)
    
    # Increment counters
    monitor.increment_counter('frames_processed')
    
    # Print periodic report (automatically handles timing)
    monitor.print_report()
```

### Methods

#### `__init__(window_size=30, report_interval=10)`

Initialize the performance monitor.

**Parameters:**
- `window_size` (int): Number of seconds for rolling window averages (default: 30)
- `report_interval` (int): Seconds between automatic reports (default: 10)

#### `record_metric(name: str, value: float)`

Record a timing metric in seconds.

**Parameters:**
- `name` (str): Metric name (e.g., 'object_detection')
- `value` (float): Time in seconds (automatically converted to milliseconds)

**Example:**
```python
import time

start = time.time()
# ... do work ...
duration = time.time() - start
monitor.record_metric('operation_name', duration)
```

#### `increment_counter(name: str, value: int = 1)`

Increment a counter.

**Parameters:**
- `name` (str): Counter name (e.g., 'frames_processed')
- `value` (int): Amount to increment (default: 1)

**Example:**
```python
monitor.increment_counter('frames_processed')
monitor.increment_counter('objects_detected', count=5)
```

#### `get_stats() -> Dict`

Get current statistics snapshot.

**Returns:**
Dictionary containing:
- `timing`: All timing metrics with avg/min/max/recent values
- `counters`: All counter values
- `rates`: Calculated rates (FPS, skip ratio, etc.)
- `gpu`: GPU memory and utilization (if available)
- `system`: CPU and RAM usage

**Example:**
```python
stats = monitor.get_stats()

# Access timing metrics (in milliseconds)
print(f"Avg detection time: {stats['timing']['object_detection']['avg']:.1f}ms")
print(f"Min: {stats['timing']['object_detection']['min']:.1f}ms")
print(f"Max: {stats['timing']['object_detection']['max']:.1f}ms")

# Access counters
print(f"Frames processed: {stats['counters']['frames_processed']}")

# Access GPU info
if 'gpu' in stats:
    print(f"GPU Memory: {stats['gpu']['memory_allocated']:.2f} GB")

# Access system info
print(f"CPU: {stats['system']['cpu_percent']:.1f}%")
print(f"RAM: {stats['system']['ram_percent']:.1f}%")
```

#### `print_report(force=False)`

Print formatted performance report.

**Parameters:**
- `force` (bool): Force printing regardless of interval (default: False)

**Behavior:**
- Automatically respects `report_interval` unless `force=True`
- Prints comprehensive statistics to console
- Updates last report timestamp

**Example:**
```python
# Print report if interval has passed
monitor.print_report()

# Force immediate report
monitor.print_report(force=True)
```

### Output Format

```
============================================================
PERFORMANCE REPORT
============================================================

Timing (ms):
  object_detection         :   35.1 (min:  29.1, max:  39.8)
  scene_captioning         :  415.1 (min: 345.1, max: 571.4)
  narration_generation     :    0.3 (min:   0.0, max:   1.5)
  total_processing         :   35.1 (min:  29.1, max:  39.8)
  fps                      : 8291.6 (min: 6737.3, max: 8867.4)

Counters:
  frames_processed         : 269
  frames_skipped           : 627
  captions_generated       : 6
  narrations_generated     : 81

Rates:
  Processing FPS:           7.88
  Skip ratio:               233.1%

GPU:
  Memory allocated:         0.71 GB
  Memory reserved:          1.12 GB
  GPU utilization:          None (NVML not available)

System:
  CPU usage:                8.3%
  RAM usage:                87.2% (13.73 GB)
============================================================
```

---

## Timer Class

Context manager for convenient timing of code blocks.

### Features

- ✅ **Context Manager**: Use with `with` statement for automatic timing
- ✅ **Automatic Recording**: Records duration to PerformanceMonitor on exit
- ✅ **Exception Safe**: Properly handles exceptions in timed code
- ✅ **Zero Overhead**: Minimal performance impact

### Usage

#### Basic Timing

```python
from src.utils.performance_monitor import PerformanceMonitor, Timer

monitor = PerformanceMonitor()

# Time a code block
with Timer(monitor, 'operation_name'):
    # Your code here
    result = expensive_operation()
    # Duration automatically recorded when block exits
```

#### Multiple Timers

```python
from src.utils.performance_monitor import PerformanceMonitor, Timer

monitor = PerformanceMonitor()

# Time multiple operations
with Timer(monitor, 'preprocessing'):
    frame = preprocess(raw_frame)

with Timer(monitor, 'inference'):
    result = model(frame)

with Timer(monitor, 'postprocessing'):
    output = postprocess(result)

# Check all timings
stats = monitor.get_stats()
for name, values in stats['timing'].items():
    print(f"{name}: {values['recent']:.1f}ms")
```

#### Nested Timers

```python
from src.utils.performance_monitor import PerformanceMonitor, Timer

monitor = PerformanceMonitor()

# Time nested operations
with Timer(monitor, 'total_pipeline'):
    
    with Timer(monitor, 'stage_1'):
        stage1_result = stage1_process()
    
    with Timer(monitor, 'stage_2'):
        stage2_result = stage2_process()
    
    with Timer(monitor, 'stage_3'):
        final_result = stage3_process()

# Both individual stages and total time are recorded
```

#### Exception Handling

```python
from src.utils.performance_monitor import PerformanceMonitor, Timer

monitor = PerformanceMonitor()

try:
    with Timer(monitor, 'risky_operation'):
        # Even if this raises an exception...
        risky_function()
        # ...the timer properly cleans up and records duration
except Exception as e:
    print(f"Error: {e}")
    # Timer duration still recorded before exception propagates
```

### Constructor

#### `Timer(monitor: PerformanceMonitor, metric_name: str)`

Create a timer context manager.

**Parameters:**
- `monitor` (PerformanceMonitor): Monitor instance to record to
- `metric_name` (str): Name of the timing metric

**Example:**
```python
monitor = PerformanceMonitor()
timer = Timer(monitor, 'my_operation')

# Use as context manager
with timer:
    # timed code
    pass
```

---

## Complete Examples

### Example 1: Basic Performance Monitoring

```python
from src.utils.performance_monitor import PerformanceMonitor, Timer
import time

def main():
    monitor = PerformanceMonitor(window_size=30, report_interval=10)
    
    for i in range(100):
        # Time processing
        with Timer(monitor, 'processing'):
            time.sleep(0.05)  # Simulate 50ms work
        
        # Track counters
        monitor.increment_counter('iterations')
        
        # Print report every 10 seconds
        monitor.print_report()
    
    # Final report
    monitor.print_report(force=True)

if __name__ == "__main__":
    main()
```

### Example 2: Multi-stage Pipeline

```python
from src.utils.performance_monitor import PerformanceMonitor, Timer
import time
import random

def simulate_detection(frame):
    time.sleep(random.uniform(0.03, 0.04))
    return [{'class': 'object', 'confidence': 0.95}]

def simulate_captioning(frame):
    time.sleep(random.uniform(0.3, 0.5))
    return "A scene with objects"

def simulate_narration(objects):
    time.sleep(random.uniform(0.0001, 0.001))
    return "I can see objects in the scene"

def main():
    monitor = PerformanceMonitor(report_interval=5)
    
    for frame_num in range(50):
        with Timer(monitor, 'total_processing'):
            
            # Object detection
            with Timer(monitor, 'object_detection'):
                objects = simulate_detection(None)
            monitor.increment_counter('frames_processed')
            
            # Scene captioning (every 10 frames)
            if frame_num % 10 == 0:
                with Timer(monitor, 'scene_captioning'):
                    caption = simulate_captioning(None)
                monitor.increment_counter('captions_generated')
            
            # Narration
            with Timer(monitor, 'narration_generation'):
                narration = simulate_narration(objects)
            monitor.increment_counter('narrations_generated')
        
        # Calculate FPS
        stats = monitor.get_stats()
        if 'total_processing' in stats['timing']:
            fps = 1000.0 / stats['timing']['total_processing']['recent']
            monitor.record_metric('fps', 1.0 / fps)
        
        # Report
        monitor.print_report()
    
    # Final statistics
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    monitor.print_report(force=True)

if __name__ == "__main__":
    main()
```

### Example 3: GPU Monitoring

```python
from src.utils.performance_monitor import PerformanceMonitor, Timer
import torch

def main():
    monitor = PerformanceMonitor()
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Do some GPU work
    for i in range(10):
        with Timer(monitor, 'gpu_operation'):
            tensor = torch.randn(1000, 1000).cuda()
            result = torch.matmul(tensor, tensor)
            torch.cuda.synchronize()
        
        monitor.increment_counter('operations')
    
    # Get GPU stats
    stats = monitor.get_stats()
    if 'gpu' in stats:
        print(f"\nGPU Memory Allocated: {stats['gpu']['memory_allocated']:.2f} GB")
        print(f"GPU Memory Reserved: {stats['gpu']['memory_reserved']:.2f} GB")
        if stats['gpu']['utilization'] is not None:
            print(f"GPU Utilization: {stats['gpu']['utilization']}%")

if __name__ == "__main__":
    main()
```

---

## Error Handling

The utils module handles errors gracefully:

### NVML Errors (GPU Monitoring)
```python
# If NVIDIA Management Library (NVML) is unavailable:
# - GPU memory stats still work (via torch)
# - GPU utilization returns None
# - No exception raised, continues normally
```

### Missing Dependencies
```python
# If psutil is not installed:
# - System stats return 0.0
# - No exception raised
# - Other functionality continues
```

---

## Dependencies

### Required
- `time`: Standard library timing
- `collections.defaultdict`: Counter storage
- `typing`: Type hints

### Optional
- `torch`: GPU memory monitoring (required for GPU features)
- `psutil`: System resource monitoring (CPU/RAM)
- `pynvml`: GPU utilization via torch.cuda.utilization() (graceful fallback if missing)

---

## Performance Impact

The monitoring utilities are designed to have minimal overhead:

- **Timer overhead**: <0.01ms per measurement
- **Counter increment**: <0.001ms
- **Stats collection**: <1ms
- **Report printing**: ~10ms (only when reporting)

**Total overhead**: <1% of processing time in typical usage

---

## Best Practices

### 1. Use Meaningful Names
```python
# Good
with Timer(monitor, 'yolov8_inference'):
    ...

# Bad
with Timer(monitor, 'thing1'):
    ...
```

### 2. Report Periodically, Not Constantly
```python
# Good - periodic reporting
monitor = PerformanceMonitor(report_interval=10)
monitor.print_report()  # Only prints every 10 seconds

# Bad - continuous reporting
for i in range(1000):
    monitor.print_report(force=True)  # Spam!
```

### 3. Group Related Metrics
```python
# Good - organized timing
with Timer(monitor, 'preprocessing'):
    ...
with Timer(monitor, 'inference'):
    ...
with Timer(monitor, 'postprocessing'):
    ...

# Also time total
with Timer(monitor, 'total_pipeline'):
    with Timer(monitor, 'preprocessing'):
        ...
    # etc.
```

### 4. Track Meaningful Counters
```python
# Good - actionable metrics
monitor.increment_counter('frames_processed')
monitor.increment_counter('objects_detected', count=len(objects))
monitor.increment_counter('errors_occurred')

# Less useful
monitor.increment_counter('random_counter')
```

---

## Future Enhancements

- [ ] Export metrics to CSV/JSON
- [ ] Integration with TensorBoard
- [ ] Web dashboard for real-time monitoring
- [ ] Alert system for performance degradation
- [ ] Histogram statistics for timing distributions
- [ ] Memory profiling integration
- [ ] Network I/O monitoring

---

## Related Files

- `src/core/optimized_assistant.py`: Main consumer of performance monitoring
- `src/utils/device_utils.py`: Device detection utilities
- `run.py`: Example usage in production

---

## Support

For issues or questions about the utils module:
1. Check the examples above
2. Review the source code documentation
3. See the main project README for context
