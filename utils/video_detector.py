"""
Video Target Detection - Detect objects in videos using the detection pipeline
Extracts evenly spaced frames, detects targets, draws bounding boxes, and creates output video
"""

import os
import cv2
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import numpy as np

from models.detector import DetectionEngine
from models.detection_config import DetectionConfig
from utils.visualizer import DetectionVisualizer


@dataclass
class VideoDetectionResult:
    """Result of video detection processing"""
    output_video_path: str
    total_frames: int
    extracted_frames: int
    frames_with_detections: int
    total_detections: int
    processing_time: float
    fps: float
    frame_detections: List[Dict] = field(default_factory=list)


class VideoDetector:
    """Detect objects in videos and create annotated output video"""
    
    def __init__(self, config: DetectionConfig = None):
        """
        Initialize video detector with detection engine
        
        Args:
            config: DetectionConfig for detection optimization (optional)
        """
        self.detector = DetectionEngine(config=config)
        self.config = config or DetectionConfig.balanced()
        self.temp_frames_dir = Path("outputs/temp_video_frames")
        self.output_dir = Path("outputs/video_detection")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def detect_video(self, video_path: str, output_video_name: str = "detected_video.mp4",
                    max_frames: int = 60) -> VideoDetectionResult:
        """
        Detect objects in video by processing evenly spaced frames
        
        Args:
            video_path: Path to input video
            output_video_name: Name of output video file
            max_frames: Maximum number of frames to extract (default 60)
        
        Returns:
            VideoDetectionResult with detection summary
        """
        print(f"\n{'='*70}")
        print("VIDEO TARGET DETECTION")
        print(f"{'='*70}")
        print(f"📁 Input video: {video_path}")
        print(f"🎬 Extracting up to {max_frames} evenly spaced frames...")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📊 Video info: {width}x{height} @ {fps} FPS, {total_frames} total frames")
        
        if total_frames <= 0:
            raise ValueError("Video has 0 frames")
        
        # Calculate frame indices to extract (evenly spaced)
        frame_indices = self._get_evenly_spaced_frames(total_frames, max_frames)
        extracted_frames = len(frame_indices)
        print(f"✓ Will process {extracted_frames} frames (evenly spaced)")
        
        # Create temp directory for frames
        self.temp_frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract and process frames
        result = self._process_extracted_frames(
            cap, video_path, frame_indices, width, height, fps, output_video_name
        )
        
        cap.release()
        
        # Clean up temp frames
        self._cleanup_temp_frames()
        
        print(f"\n{'='*70}")
        print("✅ VIDEO DETECTION COMPLETE!")
        print(f"{'='*70}")
        print(f"📁 Output video: {result.output_video_path}")
        print(f"📊 Total frames processed: {result.extracted_frames}")
        print(f"📊 Frames with detections: {result.frames_with_detections}")
        print(f"📊 Total objects detected: {result.total_detections}")
        print(f"⏱️  Processing time: {result.processing_time:.2f}s")
        print(f"🎬 Output FPS: {result.fps:.2f}")
        
        return result
    
    def _get_evenly_spaced_frames(self, total_frames: int, max_frames: int) -> List[int]:
        """
        Calculate indices of evenly spaced frames
        
        Args:
            total_frames: Total number of frames in video
            max_frames: Maximum number of frames to extract
        
        Returns:
            List of frame indices to extract
        """
        if total_frames <= max_frames:
            return list(range(total_frames))
        
        # Create evenly spaced indices
        indices = []
        step = total_frames / max_frames
        for i in range(max_frames):
            indices.append(int(i * step))
        
        return indices
    
    def _process_extracted_frames(self, cap: cv2.VideoCapture, video_path: str,
                                 frame_indices: List[int], width: int, height: int,
                                 fps: float, output_video_name: str) -> VideoDetectionResult:
        """
        Extract, detect, and process frames
        
        Args:
            cap: OpenCV VideoCapture object
            video_path: Path to video (for detection)
            frame_indices: Indices of frames to extract
            width: Frame width
            height: Frame height
            fps: Video FPS
            output_video_name: Name for output video
        
        Returns:
            VideoDetectionResult with processing results
        """
        import time
        start_time = time.perf_counter()
        
        # Setup video writer
        output_path = str(self.output_dir / output_video_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create video writer: {output_path}")
        
        result = VideoDetectionResult(
            output_video_path=output_path,
            total_frames=len(frame_indices),
            extracted_frames=0,
            frames_with_detections=0,
            total_detections=0,
            processing_time=0,
            fps=fps,
        )
        
        # Process each frame
        current_frame_idx = 0
        target_frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check if this is a frame we want to process
            if target_frame_idx >= len(frame_indices):
                break
            
            if current_frame_idx == frame_indices[target_frame_idx]:
                # This is a frame to process
                result.extracted_frames += 1
                print(f"\n[{result.extracted_frames}/{len(frame_indices)}] Processing frame {current_frame_idx}...")
                
                # Save frame temporarily
                temp_frame_path = str(self.temp_frames_dir / f"frame_{target_frame_idx:04d}.jpg")
                cv2.imwrite(temp_frame_path, frame)
                
                try:
                    # Detect objects in frame (all classes)
                    det_result = self.detector.detect(temp_frame_path, "any object", config=self.config)
                    detections = det_result.raw_detections
                    
                    # Apply NMS to raw detections
                    if detections:
                        from models.enhanced_postprocessor import EnhancedPostProcessor
                        from PIL import Image
                        img = Image.open(temp_frame_path)
                        img_width, img_height = img.size
                        
                        detections = EnhancedPostProcessor.apply_nms(
                            detections,
                            config=self.config,
                            image_width=img_width,
                            image_height=img_height,
                        )
                    
                    # Draw bounding boxes on frame
                    annotated_frame = self._draw_detections_on_frame(frame, detections)
                    
                    if detections:
                        result.frames_with_detections += 1
                        result.total_detections += len(detections)
                        
                        # Log detections
                        frame_det = {
                            "frame_index": current_frame_idx,
                            "detections": []
                        }
                        for det in detections:
                            frame_det["detections"].append({
                                "class": det.label,
                                "confidence": det.confidence
                            })
                        result.frame_detections.append(frame_det)
                        
                        print(f"  ✓ Found {len(detections)} objects")
                    else:
                        print(f"  No detections")
                    
                    # Write annotated frame to video
                    writer.write(annotated_frame)
                    
                except Exception as e:
                    print(f"  ⚠️  Error processing frame: {e}")
                    writer.write(frame)
                
                target_frame_idx += 1
            else:
                # Skip frames we don't need
                pass
            
            current_frame_idx += 1
        
        writer.release()
        result.processing_time = time.perf_counter() - start_time
        
        return result
    
    @staticmethod
    def _draw_detections_on_frame(frame: np.ndarray, detections: list) -> np.ndarray:
        """
        Draw bounding boxes with class labels on frame
        
        Args:
            frame: Video frame (numpy array)
            detections: List of Detection objects
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.bbox)
            confidence = detection.confidence
            label = detection.label
            
            # Draw bounding box (green)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw class label with confidence
            text = f"{label} {confidence:.2f}"
            text_size, baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw label background
            cv2.rectangle(
                annotated,
                (x1, y1 - text_size[1] - 10),
                (x1 + text_size[0], y1),
                (0, 255, 0),
                -1
            )
            
            # Draw label text (white)
            cv2.putText(
                annotated,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return annotated
    
    def _cleanup_temp_frames(self):
        """Remove temporary frame directory"""
        if self.temp_frames_dir.exists():
            import shutil
            shutil.rmtree(self.temp_frames_dir, ignore_errors=True)


def main():
    """Interactive video detector"""
    print("\n" + "="*70)
    print("VIDEO TARGET DETECTION")
    print("="*70)
    
    # Get video path
    video_path = input("\nEnter video path: ").strip()
    
    if not os.path.exists(video_path):
        print(f"❌ Error: File not found: {video_path}")
        return
    
    # Get output video name (optional)
    output_name = input("Output video name (default: detected_video.mp4): ").strip()
    if not output_name:
        output_name = "detected_video.mp4"
    
    # Get max frames (optional)
    max_frames_input = input("Max frames to process (default: 60): ").strip()
    try:
        max_frames = int(max_frames_input) if max_frames_input else 60
    except ValueError:
        max_frames = 60
    
    # Choose configuration
    print("\nSelect detection config:")
    print("  1) Speed Optimized")
    print("  2) Balanced (default)")
    print("  3) Accuracy Optimized")
    config_choice = input("> ").strip() or "2"
    
    if config_choice == "1":
        config = DetectionConfig.speed_optimized()
    elif config_choice == "3":
        config = DetectionConfig.accuracy_optimized()
    else:
        config = DetectionConfig.balanced()
    
    # Process video
    try:
        detector = VideoDetector(config=config)
        result = detector.detect_video(video_path, output_name, max_frames)
        
        # Print summary
        print("\nDetection Summary by Frame:")
        for frame_det in result.frame_detections:
            print(f"  Frame {frame_det['frame_index']}: {len(frame_det['detections'])} detections")
            for det in frame_det['detections'][:5]:  # Show first 5
                print(f"    - {det['class']}: {det['confidence']:.2f}")
            if len(frame_det['detections']) > 5:
                print(f"    ... and {len(frame_det['detections']) - 5} more")
    
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
