# Core Module

This module contains the main orchestration logic for the Real Time Vision System application.

## Overview

The core module is responsible for:
- Coordinating all AI services with optimized performance
- Managing multi-threaded processing pipeline
- Handling inter-service communication
- Real-time performance monitoring
- Selective scene captioning for efficiency

## Components

### `optimized_assistant.py`

The main `OptimizedAssistant` class that orchestrates the entire application with performance optimizations.

#### Key Features

- **🚀 High Performance**: Achieves 8-9 FPS processing (10x improvement over original)
- **🎯 Smart Analysis**: Intelligent throttling (every 5 seconds, requires 2+ object changes)
- **🖼️ Selective Captioning**: Scene captioning only when needed (every 10 seconds or 30% scene change)
- **📊 Performance Monitoring**: Real-time metrics tracking (FPS, latency, GPU usage)
- **⚡ Adaptive Processing**: Frame skipping and smart resource management
- **Multi-threaded Architecture**: Separate threads for camera capture, frame processing, and audio output
- **Queue-based Communication**: Thread-safe data exchange
- **GPU Acceleration**: CUDA mixed precision for all AI models

#### Architecture

```python
OptimizedAssistant
├── Camera Thread (_capture_frames)
│   └── Continuously captures frames from camera source
├── Processing Thread (_process_frames)
│   ├── Adaptive frame skipping (every 3rd frame)
│   ├── Object detection with YOLOv8 (35-40ms)
│   ├── Selective scene captioning with BLIP (~400ms, triggered smartly)
│   ├── Smart analysis generation (<1ms)
│   └── Performance metrics tracking
├── Audio Thread (_handle_audio)
│   └── Converts narrations to speech with queue management
└── Performance Monitor
    ├── Real-time FPS tracking
    ├── Per-model timing (detection, captioning, narration)
    ├── GPU memory monitoring
    ├── System resource tracking (CPU, RAM)
    └── Periodic performance reports (every 10 seconds)
```

## Performance Metrics

### Processing Speed
- **Overall FPS**: 8-9 FPS (with captioning enabled)
- **Object Detection**: 35-40ms per frame
- **Scene Captioning**: 350-450ms (only when triggered)
- **Narration Generation**: <1ms
- **Frame Skip Ratio**: ~240% (processes 1 in 3 frames efficiently)

### Resource Usage (RTX 3050 Laptop)
- **GPU Memory**: 0.7GB allocated, 1.1GB reserved
- **CPU Usage**: 8-12% average
- **RAM Usage**: Depends on system

### Narration Strategy
- **Interval**: Every 5 seconds minimum
- **Trigger**: Requires 2+ object changes (additions/removals/moves)
- **Logic**: Scene change AND time passed (prevents spam)
- **Result**: ~7-8 meaningful narrations per 38 seconds

## Usage

### Basic Usage

```python
from src.core.optimized_assistant import OptimizedAssistant

# Initialize with display and scene captioning
assistant = OptimizedAssistant(
    show_display=True,
    camera_ip="192.168.1.100",
    enable_captioning=True  # Enable AI scene descriptions
)

# Start the assistant
assistant.start()
```

### Advanced Configuration

```python
from src.core.optimized_assistant import OptimizedAssistant

# Headless mode without captioning (fastest)
assistant = OptimizedAssistant(
    show_display=False,
    camera_ip="192.168.1.100",
    enable_captioning=False  # Disable for maximum speed
)

# Custom performance settings
assistant.frame_skip = 5  # Process every 5th frame (faster but less responsive)
assistant.narration_interval = 3  # Narrate every 3 seconds
assistant.caption_interval = 15  # Caption every 15 seconds

# Start
assistant.start()
```

### Running from Command Line

```bash
# Interactive mode with prompts
python run.py

# Example interaction:
# Enter phone IP: 192.168.1.100
# Enable scene captioning? (y/n): y
# Show visual display? (y/n): y
```

## Class Methods

### `__init__(show_display=True, camera_ip=None, enable_captioning=True)`

Initialize the Real Time Vision System.

**Parameters:**
- `show_display` (bool): Whether to show the visual display window with performance overlay
- `camera_ip` (str): IP address of the camera (IP Webcam app or other source)
- `enable_captioning` (bool): Enable scene captioning with BLIP model (adds ~400ms but provides context)

**Example:**
```python
# Full features
vision_system = OptimizedAssistant(
    show_display=True,
    camera_ip="192.168.1.100",
    enable_captioning=True
)

# Maximum speed (no captioning)
vision_system = OptimizedAssistant(
    show_display=False,
    camera_ip="192.168.1.100",
    enable_captioning=False
)
```

### `start()`

Start the system and begin processing frames with performance monitoring.

**Behavior:**
- Starts all worker threads
- Begins camera capture
- Opens display window with performance overlay (if enabled)
- Prints performance reports every 10 seconds
- Blocks until Ctrl+C or 'Q' key pressed

**Example:**
```python
try:
    vision_system.start()
except KeyboardInterrupt:
    print("Stopping...")
```

### `cleanup()`

Clean up resources and stop all threads with final performance report.

**Behavior:**
- Stops all worker threads gracefully
- Releases camera resources
- Closes display windows
- Cleans up audio files
- Prints final performance summary

**Example:**
```python
try:
    vision_system.start()
finally:
    vision_system.cleanup()  # Always cleanup, even on error
```

## Threading Model

### Frame Queue
- **Purpose**: Buffer frames between camera and processing threads
- **Size**: 2 frames (configurable)
- **Behavior**: Drops old frames when full to maintain real-time processing

### Narration Queue
- **Purpose**: Buffer narrations for text-to-speech
- **Behavior**: Clears old narrations to always speak the latest

### Display Queue
- **Purpose**: Buffer annotated frames for display thread
- **Size**: 1 frame (most recent)
- **Behavior**: Non-blocking updates for smooth display

## Performance Monitoring

The `PerformanceMonitor` class tracks real-time metrics:

### Metrics Tracked
- **Timing**: Per-model execution times (detection, captioning, narration)
- **Counters**: Frames processed/skipped, captions/narrations generated
- **Rates**: Processing FPS, skip ratio
- **GPU**: Memory allocated/reserved, utilization (optional)
- **System**: CPU usage, RAM usage

### Performance Reports
- **Interval**: Every 10 seconds during operation
- **Final Report**: Comprehensive summary on cleanup
- **Display Overlay**: Real-time FPS on video feed (if display enabled)

Example output:
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

System:
  CPU usage:                8.3%
  RAM usage:                87.2% (13.73 GB)
============================================================
```

## Optimization Strategies

### 1. Adaptive Frame Processing
- **Frame Skipping**: Processes every 3rd frame by default
- **Smart Queuing**: Drops old frames to maintain real-time performance
- **Result**: Reduces processing load by 67% while maintaining responsiveness

### 2. Selective Scene Captioning
- **Trigger Conditions**:
  - Time-based: Every 10 seconds maximum
  - Scene change: 30% of objects changed position/added/removed
- **Benefits**: Reduces captioning overhead from 100% to ~20%
- **Result**: Saves ~400ms per frame on average

### 3. Intelligent Narration Throttling
- **Requirements**:
  - Minimum 5 seconds between narrations (prevents spam)
  - Requires 2+ object changes (additions/removals/significant moves)
  - Both conditions must be met (AND logic)
- **Position Sensitivity**: 5x5 grid (coarser than before for stability)
- **Result**: ~7-8 meaningful narrations vs 96 spam narrations in 38 seconds

### 4. GPU Optimization
- **Mixed Precision**: `torch.amp.autocast` for faster inference
- **Memory Management**: Efficient tensor handling
- **Model Loading**: All models preloaded and cached
- **Result**: 35-40ms object detection (down from 100-150ms)

### 5. Performance Monitoring Overhead
- **Minimal Impact**: <1ms per frame for metrics collection
- **Smart Reporting**: Periodic reports avoid continuous logging
- **Optional GPU Stats**: Graceful fallback if NVML unavailable

## Configuration

### Timing Parameters

```python
# Configurable instance variables
assistant.frame_skip = 3              # Process every Nth frame (default: 3)
assistant.narration_interval = 5      # Seconds between narrations (default: 5)
assistant.caption_interval = 10       # Seconds between captions (default: 10)
assistant.caption_threshold = 0.3     # Scene change threshold for captioning (default: 0.3)
assistant.min_scene_change = 2        # Minimum object changes for narration (default: 2)
```

### Performance Monitor Configuration

```python
# PerformanceMonitor settings
assistant.perf_monitor.window_size = 30  # Rolling window for averages
assistant.perf_monitor.report_interval = 10  # Seconds between reports
```

### Queue Sizes

```python
self.frame_queue = Queue(maxsize=2)      # Frame buffer (keeps latest)
self.narration_queue = Queue()           # Text to speak (FIFO)
self.display_queue = Queue(maxsize=1)    # Display frame (latest only)
```

## Error Handling

The assistant handles various error conditions:

- **Camera disconnection**: Automatic reconnection
- **Model inference errors**: Logs and continues
- **Audio playback errors**: Skips failed narrations
- **Thread exceptions**: Graceful degradation

## Dependencies

- `cv2` (OpenCV): Video capture and display
- `torch`: GPU acceleration
- `threading`: Multi-threaded processing
- `queue`: Thread-safe communication
- All service modules from `src.services`

## Code Snippets

### Complete Working Example

```python
from src.core.optimized_assistant import OptimizedAssistant

def main():
    # Get camera IP from user
    camera_ip = input("Enter camera IP address (e.g., 192.168.1.100): ")
    
    # Initialize system with all features
    vision_system = OptimizedAssistant(
        show_display=True,         # Show video feed with performance overlay
        camera_ip=camera_ip,
        enable_captioning=True     # Enable AI scene descriptions
    )
    
    print("Starting Real Time Vision System...")
    print("Features: Object Detection + Scene Captioning + Performance Monitoring")
    print("Press 'Q' or Ctrl+C to stop")
    
    try:
        vision_system.start()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        vision_system.cleanup()
        print("Cleanup complete - Final performance report shown above")

if __name__ == "__main__":
    main()
```

### Headless Mode (Maximum Performance)

```python
from src.core.optimized_assistant import OptimizedAssistant

# Perfect for deployment on devices without screens
# Achieves maximum FPS by disabling display and captioning
vision_system = OptimizedAssistant(
    show_display=False,          # No visual display
    camera_ip="192.168.1.100",
    enable_captioning=False      # Disable captioning for speed
)

try:
    print("Running in headless mode (max performance)...")
    vision_system.start()
except KeyboardInterrupt:
    vision_system.cleanup()
```

### Custom Performance Tuning

```python
from src.core.optimized_assistant import OptimizedAssistant

# Initialize with custom settings
vision_system = OptimizedAssistant(
    show_display=True,
    camera_ip="192.168.1.100",
    enable_captioning=True
)

# Fine-tune performance parameters
vision_system.frame_skip = 5              # Process every 5th frame (faster but less responsive)
vision_system.narration_interval = 3      # More frequent narrations (every 3 seconds)
vision_system.caption_interval = 15       # Less frequent captions (every 15 seconds)
vision_system.min_scene_change = 3        # Require more changes before narrating

# Start with custom settings
vision_system.start()
```

### Monitoring Performance Programmatically

```python
from src.core.optimized_assistant import OptimizedAssistant
import threading
import time

vision_system = OptimizedAssistant(show_display=True, camera_ip="192.168.1.100")

# Custom monitoring thread
def monitor_performance():
    while vision_system.running:
        stats = vision_system.perf_monitor.get_stats()
        
        # Check FPS
        if 'fps' in stats['timing']:
            current_fps = stats['timing']['fps']['recent']
            print(f"Current FPS: {current_fps:.2f}")
        
        # Check GPU memory
        if 'gpu' in stats:
            gpu_mem = stats['gpu']['memory_allocated']
            print(f"GPU Memory: {gpu_mem:.2f} GB")
        
        time.sleep(5)

# Start monitoring
monitor_thread = threading.Thread(target=monitor_performance, daemon=True)
monitor_thread.start()

# Start system
try:
    vision_system.start()
finally:
    vision_system.cleanup()
```

## Error Handling

The assistant handles various error conditions gracefully:

- **Camera disconnection**: Automatic reconnection attempts
- **Model inference errors**: Logs and continues processing
- **Audio playback errors**: Skips failed narrations, continues operation
- **Thread exceptions**: Graceful degradation, other threads continue
- **NVML errors**: Falls back gracefully if GPU monitoring unavailable
- **Caption generation errors**: Continues with object detection only

All errors are logged without crashing the application.

## Dependencies

Core dependencies:
- `torch` + `torchvision`: Deep learning framework with CUDA support
- `cv2` (OpenCV): Video capture and display
- `numpy`: Numerical computing
- `threading` + `queue`: Multi-threaded processing
- `psutil`: System resource monitoring

Service dependencies:
- `ultralytics`: YOLOv8 object detection
- `transformers` + `PIL`: BLIP image captioning
- `gtts` + `pygame`: Text-to-speech and audio playback

## Future Improvements

- [x] ~~Performance monitoring and optimization~~ ✅ Complete
- [x] ~~Selective scene captioning~~ ✅ Complete  
- [x] ~~Intelligent narration throttling~~ ✅ Complete
- [ ] Voice command integration with Whisper
- [ ] TensorRT optimization for 10+ FPS
- [ ] INT8 quantization for mobile deployment
- [ ] Configuration file support (YAML/JSON)
- [ ] Recording/playback mode for testing
- [ ] Web interface for remote monitoring
- [ ] Mobile app development

## Related Files

- `src/utils/performance_monitor.py`: Performance tracking implementation
- `src/services/`: All AI service modules
- `run.py`: Main entry point with interactive configuration
- `OPTIMIZATION_ROADMAP.md`: Detailed 3-week optimization plan
