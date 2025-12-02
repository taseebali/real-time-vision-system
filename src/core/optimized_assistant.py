"""
Optimized assistant with selective scene captioning and performance monitoring
"""

import cv2
import time
import threading
import torch
import numpy as np
from queue import Queue, Empty
from typing import Optional, Dict, List

from src.services.camera_service import CameraService
from src.services.detection.object_detector import ObjectDetector
from src.services.detection.text_detector import TextDetector
from src.services.detection.caption_generator import CaptionGenerator
from src.services.audio.text_to_speech import TextToSpeech
from src.services.narration_service import NarrationService
from src.utils.performance_monitor import PerformanceMonitor, Timer

class OptimizedAssistant:
    """
    Optimized Real Time Vision System with:
    - Selective scene captioning (only when needed)
    - Performance monitoring
    - Adaptive frame processing
    - Smart analysis strategies
    """
    
    def __init__(self, show_display=True, camera_ip=None, enable_captioning=True):
        """
        Initialize the Real Time Vision System.
        
        Args:
            show_display (bool): Whether to show the visual display
            camera_ip (str): IP address of the camera (IP Webcam or other source)
            enable_captioning (bool): Enable scene captioning feature
        """
        self.show_display = show_display
        self.enable_captioning = enable_captioning
        
        # Initialize performance monitor
        self.perf_monitor = PerformanceMonitor(window_size=30)
        
        # Initialize queues
        self.frame_queue = Queue(maxsize=2)
        self.narration_queue = Queue()
        self.display_queue = Queue(maxsize=1)
        
        # Initialize services
        print("Initializing services...")
        self.camera = CameraService(source="phone", ip_address=camera_ip)
        self.object_detector = ObjectDetector()
        
        # Only initialize caption generator if enabled
        if self.enable_captioning:
            print("Loading scene captioning model...")
            self.caption_generator = CaptionGenerator()
        else:
            self.caption_generator = None
            
        self.tts = TextToSpeech()
        self.narration_service = NarrationService()
        
        # Thread control
        self.running = False
        self.threads = []
        
        # Processing control
        self.frame_skip = 3  # Process every 3rd frame
        self.last_process_time = 0
        
        # Captioning control
        self.caption_interval = 10  # Caption every 10 seconds max
        self.last_caption_time = 0
        self.caption_threshold = 0.3  # Scene change threshold for captioning
        self.last_caption_objects = set()
        self.last_caption = None
        
        # Narration control
        self.narration_interval = 4  # Narrate every 4 seconds max
        self.last_narration_time = 0
        self.previous_objects = set()
        self.min_scene_change = 2  # Require at least 2 object changes for narration
        
        print("Initialization complete!")
    
    def start(self):
        """Start the system and begin processing frames."""
        print("\n" + "="*60)
        print("STARTING REAL TIME VISION SYSTEM")
        print("="*60)
        print(f"Captioning: {'ENABLED' if self.enable_captioning else 'DISABLED'}")
        print(f"Display: {'ENABLED' if self.show_display else 'DISABLED'}")
        print(f"Frame skip: Every {self.frame_skip} frames")
        print("="*60 + "\n")
        
        self.running = True
        
        # Start worker threads
        self.threads = [
            threading.Thread(target=self._capture_frames, name="CameraThread"),
            threading.Thread(target=self._process_frames, name="ProcessingThread"),
            threading.Thread(target=self._handle_audio, name="AudioThread"),
            threading.Thread(target=self._performance_reporter, name="MonitorThread")
        ]
        
        for thread in self.threads:
            thread.daemon = True
            thread.start()
        
        # Main display loop or wait
        try:
            if self.show_display:
                self._display_loop()
            else:
                while self.running:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping Real Time Vision System...")
        finally:
            self.cleanup()
    
    def _capture_frames(self):
        """Continuously capture frames from camera."""
        self.camera.start()
        frame_count = 0
        
        while self.running:
            frame = self.camera.capture_frame()
            if frame is not None:
                frame_count += 1
                
                # Clear queue if full (drop old frames)
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                        self.perf_monitor.increment_counter('frames_skipped')
                    except Empty:
                        pass
                
                self.frame_queue.put((frame, time.time(), frame_count))
    
    def _should_generate_caption(self, objects: List[Dict], current_time: float) -> bool:
        """
        Determine if we should generate a caption based on:
        - Time since last caption
        - Scene change magnitude
        - Presence of new objects
        """
        if not self.enable_captioning or self.caption_generator is None:
            return False
        
        # Always caption if enough time has passed
        if (current_time - self.last_caption_time) >= self.caption_interval:
            return True
        
        # Check for significant scene change
        current_object_set = set(obj['class'] for obj in objects)
        
        # Calculate change ratio
        if len(self.last_caption_objects) > 0:
            added = current_object_set - self.last_caption_objects
            removed = self.last_caption_objects - current_object_set
            change_ratio = (len(added) + len(removed)) / len(self.last_caption_objects)
            
            # Caption if scene changed significantly
            if change_ratio > self.caption_threshold:
                return True
        
        return False
    
    def _process_frames(self):
        """Process frames with optimized AI models."""
        frame_count = 0
        start_time = time.time()
        
        while self.running:
            try:
                # Get frame from queue
                frame, timestamp, frame_number = self.frame_queue.get(timeout=1.0)
                if frame is None:
                    continue
                
                # Skip frames for performance
                if frame_number % self.frame_skip != 0:
                    self.perf_monitor.increment_counter('frames_skipped')
                    continue
                
                frame_count += 1
                current_time = time.time()
                
                # Start total processing timer
                process_start = time.time()
                
                # Object Detection
                with Timer(self.perf_monitor, 'object_detection'):
                    with torch.amp.autocast('cuda'):
                        try:
                            objects, annotated_frame = self.object_detector.detect(frame.copy())
                        except Exception as e:
                            print(f"Object detection error: {e}")
                            objects = []
                            annotated_frame = frame
                
                # Scene Captioning (Selective)
                caption = None
                if self._should_generate_caption(objects, current_time):
                    with Timer(self.perf_monitor, 'scene_captioning'):
                        with torch.amp.autocast('cuda'):
                            try:
                                caption, confidence = self.caption_generator.generate(frame)
                                self.last_caption = caption
                                self.last_caption_time = current_time
                                self.last_caption_objects = set(obj['class'] for obj in objects)
                                self.perf_monitor.increment_counter('captions_generated')
                                print(f"[CAPTION] {caption} (confidence: {confidence:.2f})")
                            except Exception as e:
                                print(f"Captioning error: {e}")
                                caption = None
                else:
                    # Reuse last caption if available
                    caption = self.last_caption
                
                # Check for scene change (object type or significant position change)
                # Less sensitive position tracking - divide by 5 instead of 10 for coarser grid
                current_objects = set((obj['class'], 
                                      int(obj['position']['x'] * 5), 
                                      int(obj['position']['y'] * 5)) 
                                     for obj in objects)
                
                # Calculate how many objects changed (added/removed/moved)
                changes = len(current_objects.symmetric_difference(self.previous_objects))
                scene_changed = changes >= self.min_scene_change
                time_for_narration = (current_time - self.last_narration_time) >= self.narration_interval
                
                # Generate Narration (require BOTH scene change AND time interval)
                if objects and scene_changed and time_for_narration:
                    with Timer(self.perf_monitor, 'narration_generation'):
                        # Include caption in narration context if available
                        narration_objects = objects.copy()
                        if caption:
                            # Add caption as context (narration service could use this)
                            narration_objects = objects  # For now, just use objects
                        
                        narration = self.narration_service.generate(narration_objects, [])
                        
                        # Enhance narration with caption if available and different
                        if caption and narration and caption.lower() not in narration.lower():
                            narration = f"{narration}. {caption}"
                        elif caption and not narration:
                            narration = caption
                        
                        if narration:
                            self.narration_queue.put(narration)
                            self.last_narration_time = current_time
                            self.previous_objects = current_objects
                            self.perf_monitor.increment_counter('narrations_generated')
                            print(f"[{current_time - start_time:.1f}s] {narration}")
                
                # Record total processing time
                total_time = time.time() - process_start
                self.perf_monitor.record_metric('total_processing', total_time)
                self.perf_monitor.increment_counter('frames_processed')
                
                # Update display
                if self.show_display:
                    # Add performance info to frame
                    self._add_performance_overlay(annotated_frame)
                    
                    if not self.display_queue.full():
                        self.display_queue.put(annotated_frame)
                
                # Print FPS periodically
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    self.perf_monitor.record_metric('fps', fps)
                    print(f"\nProcessing: {fps:.2f} FPS | "
                          f"Avg: {self.perf_monitor.get_average('total_processing')*1000:.1f}ms")
                
            except Empty:
                continue
            except Exception as e:
                print(f"Error in processing thread: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    def _add_performance_overlay(self, frame: np.ndarray):
        """Add performance metrics overlay to frame"""
        stats = self.perf_monitor.get_stats()
        
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (0, 255, 0)
        
        # FPS
        if 'fps' in stats['averages']:
            fps = stats['averages']['fps']['recent']
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, y_offset), 
                       font, font_scale, color, thickness)
            y_offset += 25
        
        # Processing time
        if 'total_processing' in stats['averages']:
            proc_time = stats['averages']['total_processing']['mean'] * 1000
            cv2.putText(frame, f"Processing: {proc_time:.0f}ms", (10, y_offset),
                       font, font_scale, color, thickness)
            y_offset += 25
        
        # Captioning status
        if self.enable_captioning:
            caption_time = stats['averages'].get('scene_captioning', {}).get('mean', 0) * 1000
            captions = stats['counters']['captions_generated']
            cv2.putText(frame, f"Captions: {captions} ({caption_time:.0f}ms)", 
                       (10, y_offset), font, font_scale, color, thickness)
    
    def _handle_audio(self):
        """Handle text-to-speech output."""
        while self.running:
            try:
                # Clear old narrations (always speak latest)
                while self.narration_queue.qsize() > 1:
                    _ = self.narration_queue.get_nowait()
                
                text = self.narration_queue.get(timeout=1.0)
                if text:
                    self.tts.speak(text)
            except Empty:
                continue
            except Exception as e:
                print(f"Error in audio thread: {e}")
                continue
    
    def _performance_reporter(self):
        """Periodically print performance reports"""
        while self.running:
            time.sleep(10)  # Report every 10 seconds
            self.perf_monitor.print_report()
    
    def _display_loop(self):
        """Display annotated frames."""
        cv2.namedWindow('Real Time Vision System', cv2.WINDOW_NORMAL)
        
        while self.running:
            try:
                frame = self.display_queue.get(timeout=0.1)
                cv2.imshow('Real Time Vision System', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
            except Empty:
                continue
        
        cv2.destroyAllWindows()
    
    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")
        self.running = False
        
        # Final performance report
        print("\n" + "="*60)
        print("FINAL PERFORMANCE REPORT")
        self.perf_monitor.print_report(force=True)
        
        # Stop services
        try:
            self.camera.stop()
        except:
            pass
        
        try:
            self.tts.cleanup()
        except:
            pass
        
        # Wait for threads
        for thread in self.threads:
            thread.join(timeout=2.0)
        
        print("Cleanup complete!")
