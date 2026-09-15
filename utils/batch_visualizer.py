"""
Batch Visualizer - Draw bounding boxes with class labels on all images/videos
Processes all files and displays/saves annotated results with detection classes
"""

import cv2
import os
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from models.detector import DetectionEngine
from models.detection_config import DetectionConfig
from utils.visualizer import DetectionVisualizer


class BatchVisualizer:
    """Process multiple images/videos and draw bounding boxes with class labels"""
    
    def __init__(self, config: DetectionConfig = None):
        """
        Initialize batch visualizer with detection engine
        
        Args:
            config: DetectionConfig for detection optimization (optional)
        """
        self.detector = DetectionEngine(config=config)
        self.config = config or DetectionConfig.balanced()
        self.output_dir = Path("outputs/annotated")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_image(self, image_path: str, target_class: str, save_output: bool = True) -> Tuple[str, int]:
        """
        Detect objects in image and draw bounding boxes with class labels
        
        Args:
            image_path: Path to image file
            target_class: Object class to detect (e.g., "person", "car", "aircraft")
            save_output: Whether to save annotated image
        
        Returns:
            Tuple of (output_path, num_detections)
        """
        print(f"\n{'='*70}")
        print(f"Processing: {image_path}")
        print(f"Target: {target_class}")
        print(f"{'='*70}")
        
        # Run detection
        result = self.detector.detect(image_path, target_class, config=self.config)
        detections = result.filtered_detections
        
        print(f"Found {len(detections)} {target_class}(s)")
        
        # Draw bounding boxes
        if save_output:
            filename = Path(image_path).stem
            output_path = str(self.output_dir / f"{filename}_annotated.jpg")
            DetectionVisualizer.draw(image_path, detections, output_path)
            return output_path, len(detections)
        
        return "", len(detections)
    
    def visualize_batch_images(self, image_folder: str, target_class: str, pattern: str = "*.jpg") -> Dict:
        """
        Process all images in a folder
        
        Args:
            image_folder: Path to folder containing images
            target_class: Object class to detect
            pattern: File pattern (default: *.jpg)
        
        Returns:
            Dictionary with processing results
        """
        image_folder = Path(image_folder)
        if not image_folder.exists():
            raise FileNotFoundError(f"Folder not found: {image_folder}")
        
        images = list(image_folder.glob(pattern))
        print(f"\nFound {len(images)} images matching '{pattern}'")
        
        results = {
            "total_files": len(images),
            "total_detections": 0,
            "processed": [],
            "failed": []
        }
        
        for idx, image_path in enumerate(images, 1):
            try:
                output_path, count = self.visualize_image(str(image_path), target_class)
                results["processed"].append({
                    "file": str(image_path),
                    "output": output_path,
                    "detections": count
                })
                results["total_detections"] += count
                print(f"[{idx}/{len(images)}] ✓ Saved: {output_path}")
            except Exception as e:
                results["failed"].append({"file": str(image_path), "error": str(e)})
                print(f"[{idx}/{len(images)}] ✗ Error: {e}")
        
        return results
    
    def visualize_video(self, video_path: str, target_class: str, 
                       save_output: bool = True, max_frames: int = None) -> Dict:
        """
        Detect objects in video and draw bounding boxes frame by frame
        
        Args:
            video_path: Path to video file
            target_class: Object class to detect
            save_output: Whether to save annotated video
            max_frames: Limit number of frames to process (None = all)
        
        Returns:
            Dictionary with video processing results
        """
        print(f"\n{'='*70}")
        print(f"Processing video: {video_path}")
        print(f"Target: {target_class}")
        print(f"{'='*70}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        results = {
            "total_frames": total_frames,
            "processed_frames": 0,
            "total_detections": 0,
            "video_output": "",
            "frames_with_detections": 0
        }
        
        # Setup video writer if saving output
        video_writer = None
        if save_output:
            video_filename = Path(video_path).stem
            output_path = str(self.output_dir / f"{video_filename}_annotated.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            results["video_output"] = output_path
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if max_frames and frame_count > max_frames:
                break
            
            # Save frame temporarily
            temp_frame_path = str(self.output_dir / f"temp_frame_{frame_count}.jpg")
            cv2.imwrite(temp_frame_path, frame)
            
            # Detect objects in frame
            try:
                result = self.detector.detect(temp_frame_path, target_class, config=self.config)
                detections = result.filtered_detections
                
                if detections:
                    results["frames_with_detections"] += 1
                    results["total_detections"] += len(detections)
                
                # Draw bounding boxes on frame
                annotated_frame = self._draw_boxes_on_frame(
                    frame, detections, target_class
                )
                
                if video_writer:
                    video_writer.write(annotated_frame)
                
                results["processed_frames"] = frame_count
                
                if frame_count % 10 == 0:
                    print(f"  Frame {frame_count}/{total_frames} - "
                          f"Found {len(detections)} {target_class}(s)")
                
            except Exception as e:
                print(f"  Frame {frame_count} error: {e}")
                if video_writer:
                    video_writer.write(frame)
            
            finally:
                # Clean up temp file
                if os.path.exists(temp_frame_path):
                    os.remove(temp_frame_path)
        
        cap.release()
        if video_writer:
            video_writer.release()
        
        print(f"\n✓ Video processing complete!")
        print(f"  Total frames: {results['processed_frames']}")
        print(f"  Total detections: {results['total_detections']}")
        print(f"  Frames with detections: {results['frames_with_detections']}")
        if save_output:
            print(f"  Output saved: {results['video_output']}")
        
        return results
    
    @staticmethod
    def _draw_boxes_on_frame(frame: np.ndarray, detections: list, 
                            target_class: str) -> np.ndarray:
        """
        Draw bounding boxes with class labels on frame
        
        Args:
            frame: Video frame (numpy array)
            detections: List of Detection objects
            target_class: Target object class name
        
        Returns:
            Annotated frame with boxes and labels
        """
        annotated = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.bbox)
            confidence = detection.confidence
            
            # Draw bounding box (green)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw class label with confidence
            label = f"{target_class} {confidence:.2f}"
            label_size, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw label background
            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
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
    """Simple batch visualizer - Only asks for image path"""
    print("\n" + "="*70)
    print("OBJECT DETECTION & VISUALIZATION")
    print("="*70)
    
    # Get image path from user (only input)
    image_path = input("\nEnter image path: ").strip()
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found: {image_path}")
        return
    
    # Use default accuracy optimized config to detect all objects
    config = DetectionConfig.accuracy_optimized()
    visualizer = BatchVisualizer(config=config)
    
    print("\n" + "="*70)
    print("Detecting all objects in image...")
    print("="*70)
    
    try:
        # Get raw detections without class filtering
        # Use empty target to capture all classes, then use raw detections
        result = visualizer.detector.detect(image_path, "any object", config=config)
        # Use raw_detections which includes ALL detected objects (before class filtering)
        detections = result.raw_detections
        
        # Apply NMS to raw detections to remove near-duplicates
        if detections:
            from models.enhanced_postprocessor import EnhancedPostProcessor
            from PIL import Image
            img = Image.open(image_path)
            image_width, image_height = img.size
            
            # Apply NMS to all raw detections (no class filtering)
            detections = EnhancedPostProcessor.apply_nms(
                detections,
                config=config,
                image_width=image_width,
                image_height=image_height,
            )
        # Save to outputs/detection_result.jpg
        output_path = "outputs/detection_result.jpg"
        os.makedirs("outputs", exist_ok=True)
        DetectionVisualizer.draw(image_path, detections, output_path)
        
        print("\n" + "="*70)
        print("✅ SUCCESS - DETECTION COMPLETE!")
        print("="*70)
        print(f"📁 Output saved: {output_path}")
        print(f"📊 Total objects detected: {len(detections)}")
        print()
        
        if detections:
            # Group by class
            classes_dict = {}
            for det in detections:
                if det.label not in classes_dict:
                    classes_dict[det.label] = []
                classes_dict[det.label].append(det.confidence)
            
            print("Detected Objects:")
            count = 1
            for class_name, confidences in sorted(classes_dict.items()):
                for conf in confidences:
                    print(f"  {count}. {class_name} - Confidence: {conf:.2f}")
                    count += 1
        else:
            print("No objects detected in this image.")
    
    except Exception as e:
        print(f"❌ Error processing image: {e}")


if __name__ == "__main__":
    main()
