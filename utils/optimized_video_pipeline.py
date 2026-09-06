"""
Optimized Video Target Detection Pipeline
- Automatically selects evenly spaced key frames based on video length
- Runs image detection pipeline on each key frame
- Detects and draws bounding boxes for ALL matching target objects
- Keeps all valid detections after NMS
- Outputs only the processed key frames to result video
- Supports Fast/Balanced/Accurate detection modes
"""

import os
import cv2
import time
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

from models.detector import DetectionEngine
from models.detection_config import DetectionConfig


@dataclass
class VideoDetectionPipelineResult:
    """Result of video detection pipeline processing"""
    output_video_path: str
    target_object: str
    total_frames: int
    key_frames: int
    processed_key_frames: int
    total_detections: int
    processing_time: float
    fps: float
    frame_detections: List[Dict] = field(default_factory=list)


class OptimizedVideoDetectionPipeline:
    """Video detection pipeline - detect ALL target objects in selected key frames"""
    
    def __init__(self, config: DetectionConfig = None):
        """
        Initialize pipeline with detection engine
        
        Args:
            config: DetectionConfig for detection optimization
        """
        self.detector = DetectionEngine(config=config)
        self.config = config or DetectionConfig.balanced()
        self.output_dir = Path("sample_images/video_test/annotated_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def calculate_optimal_key_frames(total_frames: int, fps: float) -> int:
        """
        Calculate optimal number of key frames based on video length
        
        Args:
            total_frames: Total frames in video
            fps: Frames per second
        
        Returns:
            Optimal number of key frames to extract
        """
        video_duration = total_frames / fps if fps > 0 else total_frames / 30
        
        # Heuristic: 1 key frame per 2-3 seconds
        if video_duration < 10:
            return min(total_frames, 5)
        elif video_duration < 30:
            return max(5, int(video_duration / 3))
        elif video_duration < 60:
            return max(10, int(video_duration / 2.5))
        else:
            return max(15, int(video_duration / 2))
    
    def process_video(self, video_path: str, target_object: str, 
                     speed_mode: str = "balanced") -> VideoDetectionPipelineResult:
        """
        Process video - detect ALL target objects in evenly spaced key frames
        
        Args:
            video_path: Path to input video
            target_object: Target object to detect (e.g., "car", "person", "truck")
            speed_mode: Detection speed mode ("fast", "balanced", "accurate")
        
        Returns:
            VideoDetectionPipelineResult with processing results
        """
        print(f"\n{'='*70}")
        print("VIDEO TARGET DETECTION PIPELINE")
        print(f"{'='*70}")
        print(f"📁 Input video: {video_path}")
        print(f"🎯 Target object: {target_object}")
        print(f"⚡ Speed mode: {speed_mode}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📊 Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
        
        if total_frames <= 0:
            raise ValueError("Video has 0 frames")
        
        # Calculate optimal key frames
        num_key_frames = self.calculate_optimal_key_frames(total_frames, fps)
        print(f"🔑 Key frames to process: {num_key_frames}")
        
        # Get key frame indices (evenly spaced)
        key_frame_indices = self._get_key_frame_indices(total_frames, num_key_frames)
        
        # Setup video writer (output will have FPS based on key frames)
        output_path = str(self.output_dir / "tracked_result.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create video writer: {output_path}")
        
        print(f"\n{'='*70}")
        print("Processing key frames...")
        print(f"{'='*70}")
        
        start_time = time.perf_counter()
        
        # Read all frames
        frames = []
        print("📥 Loading video frames...")
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            frame_idx += 1
        
        cap.release()
        print(f"✓ Loaded {len(frames)} frames")
        
        # Process key frames
        result = VideoDetectionPipelineResult(
            output_video_path=output_path,
            target_object=target_object,
            total_frames=total_frames,
            key_frames=num_key_frames,
            processed_key_frames=0,
            total_detections=0,
            processing_time=0,
            fps=fps,
        )
        
        temp_dir = Path("outputs/temp_pipeline_frames")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n🔍 Detecting on key frames...")
        
        for key_idx, frame_num in enumerate(key_frame_indices, 1):
            if frame_num >= len(frames):
                break
            
            frame = frames[frame_num].copy()
            
            # Save frame temporarily
            temp_path = str(temp_dir / f"frame_{frame_num}.jpg")
            cv2.imwrite(temp_path, frame)
            
            try:
                # Detect ALL objects of target class
                result_det = self.detector.detect(temp_path, target_object, config=self.config)
                detections = result_det.filtered_detections
                
                print(f"  [{key_idx}/{num_key_frames}] Frame {frame_num}: Found {len(detections)} {target_object}(s)")
                
                # Draw ALL detections on frame
                annotated_frame = self._draw_all_detections(frame, detections, target_object)
                
                # Write annotated frame to output
                writer.write(annotated_frame)
                result.processed_key_frames += 1
                result.total_detections += len(detections)
                
                # Log detections
                if detections:
                    frame_det = {
                        "frame_index": frame_num,
                        "num_detections": len(detections),
                        "detections": []
                    }
                    for det in detections:
                        frame_det["detections"].append({
                            "class": det.label,
                            "confidence": f"{det.confidence:.2f}"
                        })
                    result.frame_detections.append(frame_det)
            
            except Exception as e:
                print(f"  [{key_idx}/{num_key_frames}] Frame {frame_num}: Error - {e}")
            
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        # Clean up temp directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        writer.release()
        result.processing_time = time.perf_counter() - start_time
        
        print(f"\n{'='*70}")
        print("✅ PIPELINE COMPLETE!")
        print(f"{'='*70}")
        print(f"📁 Output: {result.output_video_path}")
        print(f"🔑 Key frames processed: {result.processed_key_frames}")
        print(f"📊 Total objects detected: {result.total_detections}")
        print(f"⏱️  Processing time: {result.processing_time:.2f}s")
        
        return result
    
    def _get_key_frame_indices(self, total_frames: int, num_key_frames: int) -> List[int]:
        """Get evenly spaced key frame indices"""
        if num_key_frames >= total_frames:
            return list(range(total_frames))
        
        indices = []
        step = total_frames / num_key_frames
        for i in range(num_key_frames):
            indices.append(int(i * step))
        
        return indices
    
    @staticmethod
    def _draw_all_detections(frame, detections: list, target_object: str):
        """
        Draw ALL detections on frame
        
        Args:
            frame: Video frame
            detections: List of Detection objects
            target_object: Target object name for label
        
        Returns:
            Annotated frame with all detections
        """
        annotated = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.bbox)
            confidence = detection.confidence
            
            # Draw bounding box (green)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw class label with confidence
            label = f"{target_object} {confidence:.2f}"
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
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
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return annotated


def main():
    """Interactive video detection pipeline"""
    print("\n" + "="*70)
    print("VIDEO TARGET DETECTION PIPELINE")
    print("="*70)
    
    # Get video path
    video_path = input("\nEnter video path: ").strip()
    
    if not os.path.exists(video_path):
        print(f"❌ Error: File not found: {video_path}")
        return
    
    # Get target object
    target_object = input("Enter target object (e.g., car, person, truck): ").strip()
    if not target_object:
        print("❌ Error: Target object is required")
        return
    
    # Get speed mode
    print("\nSelect speed mode:")
    print("  1) Fast (fewer key frames, faster)")
    print("  2) Balanced (default)")
    print("  3) Accurate (more key frames, slower but more accurate)")
    
    speed_choice = input("> ").strip() or "2"
    
    speed_modes = {"1": "fast", "2": "balanced", "3": "accurate"}
    speed_mode = speed_modes.get(speed_choice, "balanced")
    
    # Select config based on speed mode
    if speed_mode == "fast":
        config = DetectionConfig.speed_optimized()
    elif speed_mode == "accurate":
        config = DetectionConfig.accuracy_optimized()
    else:
        config = DetectionConfig.balanced()
    
    # Process video
    try:
        pipeline = OptimizedVideoDetectionPipeline(config=config)
        result = pipeline.process_video(video_path, target_object, speed_mode)
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Target: {result.target_object}")
        print(f"Key frames processed: {result.processed_key_frames}/{result.key_frames}")
        print(f"Total objects detected: {result.total_detections}")
        print(f"Processing time: {result.processing_time:.2f}s")
        
        if result.frame_detections:
            print(f"\nDetections by frame:")
            for frame_det in result.frame_detections:
                print(f"  Frame {frame_det['frame_index']}: {frame_det['num_detections']} objects")
                for det in frame_det['detections'][:3]:
                    print(f"    - {det['class']}: {det['confidence']}")
                if len(frame_det['detections']) > 3:
                    print(f"    ... and {len(frame_det['detections']) - 3} more")
        
        print(f"\n✅ Output saved to: {result.output_video_path}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
