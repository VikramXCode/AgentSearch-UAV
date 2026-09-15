from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image

import importlib

try:
    imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
except Exception:
    imageio_ffmpeg = None

from models.detection_config import DEFAULT_CONFIG, DetectionConfig
from models.enhanced_postprocessor import EnhancedPostProcessor
from models.schemas import Detection
from models.super_resolution import SuperResolutionEngine
from models.yolo_world import YOLOWorldDetector
from utils.model_paths import is_visdrone_checkpoint, VISDRONE_CLASS_NAMES
from utils.search_utils import (
    canonicalize_target,
    extract_query_components,
    get_visdrone_classes_for_target,
    normalize_label,
)


@dataclass
class VideoTrackingResult:
    output_video_path: str
    total_processing_time: float
    detection_time: float
    tracking_time: float
    frames_processed: int
    skipped_frames: int
    tracked_targets: int
    fps: float
    detections: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackState:
    track_id: int
    label: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2]
    velocity: tuple[float, float] = (0.0, 0.0)  # (vx, vy) center velocity
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    color: tuple[int, int, int] = (0, 255, 0)
    history: list[tuple[float, float]] = field(default_factory=list)
    total_distance: float = 0.0


class VideoTracker:
    """
    VideoTracker:
    Autonomous Multi-Agent Video Tracking Engine for UAV reconnaissance streams.
    - Reuses fine-tuned YOLO-World and VisDrone target mapping.
    - Multi-Object Tracking (MOT) with IoU & distance matching.
    - Anti-flicker Exponential Moving Average (EMA) coordinate smoothing.
    - Track persistence across brief dropouts and occlusions.
    - Universal H.264 browser MP4 encoding with exact FPS, resolution, and duration preservation.
    - Real-time progress reporting.
    """

    _SHARED_DETECTOR: YOLOWorldDetector | None = None
    _SHARED_SR_ENGINE: SuperResolutionEngine | None = None

    COLOR_PALETTE = [
        (0, 220, 255),    # Cyan
        (0, 255, 136),    # Emerald green
        (255, 191, 0),    # Amber
        (255, 105, 180),  # Hot pink
        (138, 43, 226),   # Blue violet
        (0, 191, 255),    # Deep sky blue
        (50, 205, 50),    # Lime green
        (255, 140, 0),    # Dark orange
        (255, 215, 0),    # Gold
        (147, 112, 219),  # Medium purple
    ]

    def __init__(self, model_path: str | None = None):
        if VideoTracker._SHARED_DETECTOR is None:
            VideoTracker._SHARED_DETECTOR = YOLOWorldDetector(model_path=model_path)
        self.yolo_world = VideoTracker._SHARED_DETECTOR

        if VideoTracker._SHARED_SR_ENGINE is None:
            VideoTracker._SHARED_SR_ENGINE = SuperResolutionEngine()
        self.sr_engine = VideoTracker._SHARED_SR_ENGINE

    @classmethod
    def get_detector(cls, model_path: str | None = None) -> YOLOWorldDetector:
        if cls._SHARED_DETECTOR is None:
            cls._SHARED_DETECTOR = YOLOWorldDetector(model_path=model_path)
        return cls._SHARED_DETECTOR

    @staticmethod
    def _box_center(box: list[float]) -> tuple[float, float]:
        x1, y1, x2, y2 = [float(v) for v in box]
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _box_area(box: list[float]) -> float:
        x1, y1, x2, y2 = [float(v) for v in box]
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        return w * h

    @staticmethod
    def _box_iou(box_a: list[float], box_b: list[float]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter_area = iw * ih

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union_area = area_a + area_b - inter_area

        if union_area <= 0.0:
            return 0.0
        return inter_area / union_area

    @staticmethod
    def _box_distance(box_a: list[float], box_b: list[float]) -> float:
        ca = VideoTracker._box_center(box_a)
        cb = VideoTracker._box_center(box_b)
        return ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5

    def _resolve_target_classes(self, query: str) -> list[dict[str, Any]]:
        """
        Extract canonical query components and map to target detector classes.
        Supports single queries ("car", "red car") and compound queries ("cars and pedestrians", "truck, bus").
        Returns a list of target specifications:
        [{"target": str, "display_name": str, "classes": list[str], "attributes": dict[str, str]}, ...]
        """
        parts = re.split(r'\s+(?:and|&|\+|or)\s+|,', query, flags=re.IGNORECASE)
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            parts = [query.strip()]

        specs: list[dict[str, Any]] = []
        for part in parts:
            components = extract_query_components(part)
            target = components.target or canonicalize_target(part)
            if not target:
                target = part.lower().strip()

            attributes: dict[str, str] = {}
            if components.color:
                attributes["color"] = components.color
            if components.size:
                attributes["size"] = components.size

            if self.yolo_world.is_visdrone:
                matched_classes = get_visdrone_classes_for_target(target)
                if not matched_classes:
                    matched_classes = [target]
            else:
                matched_classes = [target]

            display_name = f"{attributes.get('color', '')} {target}".strip() if "color" in attributes else target
            specs.append({
                "target": target,
                "display_name": display_name,
                "classes": matched_classes,
                "attributes": attributes,
            })

        return specs

    def _verify_crop_color(self, frame: np.ndarray, bbox: list[float], requested_color: str) -> bool:
        """
        Fast HSV verification for color attributes on detection crop.
        """
        if not requested_color:
            return True

        h, w = frame.shape[:2]
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(h, int(bbox[3]))

        if x2 <= x1 or y2 <= y1:
            return False

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            return True

        try:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            requested_lower = requested_color.lower()

            if requested_lower == "red":
                m1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
                m2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
                mask = cv2.bitwise_or(m1, m2)
            elif requested_lower in {"white", "silver"}:
                mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))
            elif requested_lower in {"black", "dark"}:
                mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 65]))
            elif requested_lower == "blue":
                mask = cv2.inRange(hsv, np.array([100, 70, 50]), np.array([135, 255, 255]))
            elif requested_lower == "green":
                mask = cv2.inRange(hsv, np.array([35, 70, 50]), np.array([85, 255, 255]))
            elif requested_lower == "yellow":
                mask = cv2.inRange(hsv, np.array([20, 70, 50]), np.array([35, 255, 255]))
            else:
                return True

            ratio = np.count_nonzero(mask) / float(crop.shape[0] * crop.shape[1])
            return ratio >= 0.08
        except Exception:
            return True

    def _detect_frame(
        self,
        frame: np.ndarray,
        target_specs: list[dict[str, Any]],
        confidence_threshold: float = 0.25,
        config: DetectionConfig | None = None,
    ) -> list[Detection]:
        """
        Run YOLO detection on a single frame array with NMS and compound query filtering.
        """
        cfg = config or DEFAULT_CONFIG
        frame_h, frame_w = frame.shape[:2]

        # Use 640/960/1280 inference resolution for speed and accuracy balance
        max_dim = max(frame_h, frame_w)
        imgsz = 640 if max_dim <= 1280 else 960

        try:
            results = self.yolo_world.model.predict(
                source=frame,
                conf=confidence_threshold,
                verbose=False,
                imgsz=imgsz,
            )
        except Exception as err:
            print(f"Prediction error on frame: {err}")
            return []

        raw_detections: list[Detection] = []

        for res in results:
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                cls_id = int(box.cls)
                raw_label = str(res.names.get(cls_id, cls_id)).lower()
                conf = float(box.conf)
                xyxy = box.xyxy[0].tolist()

                # Coordinate validation
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                if x2 <= x1 or y2 <= y1:
                    continue
                if (x2 - x1) * (y2 - y1) < 50:
                    continue

                # Match against target specifications
                matched_spec = None
                for spec in target_specs:
                    spec_classes = {c.lower() for c in spec["classes"]}
                    spec_target = spec["target"].lower()
                    if (
                        raw_label in spec_classes
                        or canonicalize_target(raw_label) == spec_target
                        or spec_target in raw_label
                    ):
                        matched_spec = spec
                        break

                if matched_spec is None:
                    continue

                # Color attribute verification if specified in query
                if "color" in matched_spec["attributes"]:
                    if not self._verify_crop_color(frame, [x1, y1, x2, y2], matched_spec["attributes"]["color"]):
                        continue

                label_to_show = matched_spec["display_name"]

                raw_detections.append(
                    Detection(
                        label=label_to_show,
                        confidence=conf,
                        bbox=[x1, y1, x2, y2],
                    )
                )

        if not raw_detections:
            return []

        # Intra-frame Non-Maximum Suppression (NMS) to eliminate duplicate boxes
        filtered = EnhancedPostProcessor.apply_nms(
            raw_detections,
            config=cfg,
            image_width=frame_w,
            image_height=frame_h,
        )

        return filtered

    def _draw_annotations(
        self,
        frame: np.ndarray,
        active_tracks: list[TrackState],
        query: str,
        frame_idx: int,
        total_frames: int,
        fps: float,
    ) -> np.ndarray:
        """
        Render crisp, anti-aliased tactical bounding boxes, labels, trajectory trails, and telemetry.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Draw active tracks
        for track in active_tracks:
            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            color = track.color

            # 1. Motion Trajectory Trail (Historical Path)
            if len(track.history) > 1:
                hist_len = len(track.history)
                for idx in range(1, hist_len):
                    pt_prev = (int(track.history[idx - 1][0]), int(track.history[idx - 1][1]))
                    pt_curr = (int(track.history[idx][0]), int(track.history[idx][1]))
                    thickness = max(1, int(1 + (idx / hist_len) * 2))
                    cv2.line(annotated, pt_prev, pt_curr, color, thickness, cv2.LINE_AA)
                cx_cur, cy_cur = int(track.history[-1][0]), int(track.history[-1][1])
                cv2.circle(annotated, (cx_cur, cy_cur), 3, (255, 255, 255), -1, cv2.LINE_AA)

            # 2. Tactical Bounding Box with Corner Brackets
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            corner_len = max(6, min(16, int(min(x2 - x1, y2 - y1) * 0.2)))
            # Top-left
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), 2, cv2.LINE_AA)
            # Top-right
            cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), (255, 255, 255), 2, cv2.LINE_AA)
            # Bottom-left
            cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), (255, 255, 255), 2, cv2.LINE_AA)
            # Bottom-right
            cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), (255, 255, 255), 2, cv2.LINE_AA)

            # 3. Dynamic Velocity / Direction Heading Vector
            vx, vy = track.velocity
            speed = (vx ** 2 + vy ** 2) ** 0.5
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            if speed >= 1.2:
                arrow_end = (
                    int(max(0, min(w - 1, cx + vx * 4.5))),
                    int(max(0, min(h - 1, cy + vy * 4.5))),
                )
                cv2.arrowedLine(annotated, (cx, cy), arrow_end, (0, 255, 255), 2, tipLength=0.35, line_type=cv2.LINE_AA)

            # 4. Label Badge with Speed & Confidence
            speed_str = f" {int(speed * fps)}px/s" if speed >= 1.2 else ""
            label_text = f"{track.label} #{track.track_id} {track.confidence:.2f}{speed_str}"
            font_scale = 0.45
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            badge_y1 = max(0, y1 - text_h - 8)
            badge_y2 = y1
            badge_x1 = x1
            badge_x2 = x1 + text_w + 10

            cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), (15, 20, 25), -1)
            cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), color, 1)

            cv2.putText(
                annotated,
                label_text,
                (badge_x1 + 5, badge_y2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        # Top-left telemetry HUD
        hud_h = 44
        hud_w = min(440, w - 20)
        cv2.rectangle(annotated, (10, 10), (10 + hud_w, 10 + hud_h), (10, 15, 22), -1)
        cv2.rectangle(annotated, (10, 10), (10 + hud_w, 10 + hud_h), (0, 200, 255), 1)

        cv2.putText(
            annotated,
            f"TARGET: {query.upper()} | TRACKS: {len(active_tracks)}",
            (20, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        pct = int((frame_idx + 1) / max(1, total_frames) * 100)
        cv2.putText(
            annotated,
            f"FRAME: {frame_idx + 1}/{total_frames} ({pct}%) | SPEED: {fps:.1f} FPS",
            (20, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        return annotated

    def process_video(
        self,
        video_path: str,
        query: str,
        output_dir: str | None = None,
        output_filename: str = "detected_video.mp4",
        confidence_threshold: float = 0.25,
        detect_interval: int = 1,
        progress_callback: Callable[[int, int, float, str], None] | None = None,
    ) -> VideoTrackingResult:
        """
        Process video frame-by-frame:
        - Detects matching query targets.
        - Applies MOT tracking with EMA coordinate smoothing.
        - Preserves exact resolution, FPS, duration, and frame order.
        - Encodes browser-playable H.264 MP4.
        """
        query = query.strip()
        if not query:
            raise ValueError("Query string cannot be empty.")

        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            raise FileNotFoundError(f"Input video not found: {video_path}")

        # Target resolution (supports compound and single queries)
        target_specs = self._resolve_target_classes(query)

        # Video capture
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if orig_width <= 0 or orig_height <= 0 or total_frames <= 0:
            cap.release()
            raise ValueError(f"Video file has invalid dimensions ({orig_width}x{orig_height}) or zero frames.")

        out_dir = Path(output_dir) if output_dir else video_path_obj.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_video_path = out_dir / output_filename
        raw_output_path = out_dir / f"raw_{output_filename}"

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(raw_output_path),
            fourcc,
            fps,
            (orig_width, orig_height),
        )

        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open VideoWriter for {raw_output_path}")

        tracks: dict[int, TrackState] = {}
        next_track_id = 1
        max_age = 8          # Frames to keep track without detection before dropping
        ema_alpha = 0.65     # Coordinate smoothing factor (0.0 to 1.0)

        all_detections_summary: list[dict[str, Any]] = []
        frames_processed = 0
        total_detection_time = 0.0
        total_tracking_time = 0.0
        start_wall_time = time.perf_counter()

        frame_idx = 0
        while True:
            t_frame_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Detection step
            new_detections: list[Detection] = []
            should_detect = (frame_idx % detect_interval == 0) or (len(tracks) == 0)

            if should_detect:
                t_det = time.perf_counter()
                new_detections = self._detect_frame(
                    frame=frame,
                    target_specs=target_specs,
                    confidence_threshold=confidence_threshold,
                )
                total_detection_time += time.perf_counter() - t_det

            t_track = time.perf_counter()

            # Predict current position of active tracks using estimated velocity
            for trk in tracks.values():
                vx, vy = trk.velocity
                px1 = trk.bbox[0] + vx
                py1 = trk.bbox[1] + vy
                px2 = trk.bbox[2] + vx
                py2 = trk.bbox[3] + vy
                trk.bbox = [px1, py1, px2, py2]
                trk.age += 1
                trk.time_since_update += 1
                pcx, pcy = self._box_center(trk.bbox)
                trk.history.append((pcx, pcy))
                if len(trk.history) > 35:
                    trk.history.pop(0)
                trk.total_distance += (vx ** 2 + vy ** 2) ** 0.5

            matched_tracks: set[int] = set()
            matched_dets: set[int] = set()

            if new_detections and tracks:
                # Match detections to existing tracks using IoU + center distance
                track_ids = list(tracks.keys())
                iou_matrix = np.zeros((len(track_ids), len(new_detections)), dtype=np.float32)

                for r_idx, tid in enumerate(track_ids):
                    for c_idx, det in enumerate(new_detections):
                        iou = self._box_iou(tracks[tid].bbox, det.bbox)
                        dist = self._box_distance(tracks[tid].bbox, det.bbox)
                        # High IoU or close center distance counts as a match candidate
                        dist_score = max(0.0, 1.0 - (dist / 150.0))
                        iou_matrix[r_idx, c_idx] = (0.7 * iou) + (0.3 * dist_score)

                # Greedy bipartite matching
                while True:
                    max_val = np.max(iou_matrix) if iou_matrix.size > 0 else 0.0
                    if max_val < 0.25:
                        break
                    r_idx, c_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                    tid = track_ids[r_idx]
                    det = new_detections[c_idx]

                    # EMA Coordinate smoothing to prevent box flickering/jitter
                    old_box = tracks[tid].bbox
                    det_box = det.bbox
                    smooth_box = [
                        (ema_alpha * det_box[0]) + ((1.0 - ema_alpha) * old_box[0]),
                        (ema_alpha * det_box[1]) + ((1.0 - ema_alpha) * old_box[1]),
                        (ema_alpha * det_box[2]) + ((1.0 - ema_alpha) * old_box[2]),
                        (ema_alpha * det_box[3]) + ((1.0 - ema_alpha) * old_box[3]),
                    ]

                    # Update velocity
                    old_cx, old_cy = self._box_center(old_box)
                    new_cx, new_cy = self._box_center(smooth_box)
                    vx = new_cx - old_cx
                    vy = new_cy - old_cy
                    old_vx, old_vy = tracks[tid].velocity
                    tracks[tid].velocity = (0.7 * vx + 0.3 * old_vx, 0.7 * vy + 0.3 * old_vy)

                    # Update track state & trajectory history
                    tracks[tid].bbox = smooth_box
                    tracks[tid].confidence = (0.7 * det.confidence) + (0.3 * tracks[tid].confidence)
                    tracks[tid].hits += 1
                    tracks[tid].time_since_update = 0
                    tracks[tid].history.append((new_cx, new_cy))
                    if len(tracks[tid].history) > 35:
                        tracks[tid].history.pop(0)
                    tracks[tid].total_distance += ((new_cx - old_cx) ** 2 + (new_cy - old_cy) ** 2) ** 0.5

                    matched_tracks.add(tid)
                    matched_dets.add(c_idx)

                    # Invalidate row and column
                    iou_matrix[r_idx, :] = -1.0
                    iou_matrix[:, c_idx] = -1.0

            # Create new tracks for unmatched detections
            if should_detect:
                for idx, det in enumerate(new_detections):
                    if idx in matched_dets:
                        continue
                    if det.confidence >= confidence_threshold:
                        color = self.COLOR_PALETTE[(next_track_id - 1) % len(self.COLOR_PALETTE)]
                        cx, cy = self._box_center(det.bbox)
                        tracks[next_track_id] = TrackState(
                            track_id=next_track_id,
                            label=det.label,
                            confidence=det.confidence,
                            bbox=det.bbox,
                            velocity=(0.0, 0.0),
                            age=1,
                            hits=1,
                            time_since_update=0,
                            color=color,
                            history=[(cx, cy)],
                            total_distance=0.0,
                        )
                        next_track_id += 1

            # Prune dead tracks that exceeded max_age or drifted out of frame bounds
            alive_tracks: dict[int, TrackState] = {}
            for tid, trk in tracks.items():
                if trk.time_since_update > max_age:
                    continue
                # Keep if box is reasonably within frame bounds
                bx1, by1, bx2, by2 = trk.bbox
                if bx2 < -20 or by2 < -20 or bx1 > orig_width + 20 or by1 > orig_height + 20:
                    continue
                alive_tracks[tid] = trk
            tracks = alive_tracks

            # Only display tracks that have been confirmed (hits >= 1) and not stale
            display_tracks = [trk for trk in tracks.values() if trk.time_since_update <= 3]

            total_tracking_time += time.perf_counter() - t_track

            # Record detections for telemetry
            if display_tracks and (frame_idx % 10 == 0 or frame_idx == 0):
                for trk in display_tracks:
                    vx, vy = trk.velocity
                    speed_px_s = ((vx ** 2 + vy ** 2) ** 0.5) * fps
                    all_detections_summary.append({
                        "track_id": trk.track_id,
                        "label": trk.label,
                        "confidence": round(float(trk.confidence), 3),
                        "bbox": [round(float(v), 1) for v in trk.bbox],
                        "frame": frame_idx,
                        "distance_px": round(float(trk.total_distance), 1),
                        "speed_px_s": round(float(speed_px_s), 1),
                        "trajectory_points": len(trk.history),
                    })


            # Calculate current running FPS
            elapsed_so_far = time.perf_counter() - start_wall_time
            running_fps = (frame_idx + 1) / elapsed_so_far if elapsed_so_far > 0 else 0.0

            # Draw annotations and write frame
            annotated_frame = self._draw_annotations(
                frame=frame,
                active_tracks=display_tracks,
                query=query,
                frame_idx=frame_idx,
                total_frames=total_frames,
                fps=running_fps,
            )

            # Strictly write every frame in order
            writer.write(annotated_frame)
            frames_processed += 1
            frame_idx += 1

            # Progress callback
            if progress_callback is not None and (frame_idx % 5 == 0 or frame_idx == total_frames):
                stage_desc = f"Analyzing frame {frame_idx}/{total_frames} ({len(display_tracks)} active targets)"
                progress_callback(frame_idx, total_frames, running_fps, stage_desc)

        cap.release()
        writer.release()

        total_wall_time = time.perf_counter() - start_wall_time
        final_fps = frames_processed / total_wall_time if total_wall_time > 0 else 0.0

        # Encode / Transcode to Browser-Friendly H.264 MP4 with yuv420p
        if progress_callback is not None:
            progress_callback(total_frames, total_frames, final_fps, "Finalizing H.264 video encoding...")

        final_video_path = self._transcode_to_h264(
            raw_mp4_path=str(raw_output_path),
            final_mp4_path=str(out_video_path),
            fps=fps,
        )

        unique_target_count = next_track_id - 1

        return VideoTrackingResult(
            output_video_path=final_video_path,
            total_processing_time=total_wall_time,
            detection_time=total_detection_time,
            tracking_time=total_tracking_time,
            frames_processed=frames_processed,
            skipped_frames=0,
            tracked_targets=unique_target_count,
            fps=final_fps,
            detections=all_detections_summary,
            metrics={
                "fps": round(final_fps, 2),
                "total_processing_time": round(total_wall_time, 2),
                "detection_time": round(total_detection_time, 2),
                "tracking_time": round(total_tracking_time, 2),
                "frames_processed": frames_processed,
                "skipped_frames": 0,
                "tracked_targets": unique_target_count,
                "input_resolution": f"{orig_width}x{orig_height}",
                "input_fps": round(fps, 2),
            },
        )

    def _transcode_to_h264(self, raw_mp4_path: str, final_mp4_path: str, fps: float) -> str:
        """
        Encode raw OpenCV video output to standard H.264 (yuv420p) for seamless
        native playback in HTML5 <video> elements on all modern web browsers.
        """
        raw_p = Path(raw_mp4_path)
        final_p = Path(final_mp4_path)

        # Resolve ffmpeg executable from imageio_ffmpeg or system PATH
        ffmpeg_exe = None
        if imageio_ffmpeg is not None:
            try:
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                ffmpeg_exe = None

        if not ffmpeg_exe or not Path(ffmpeg_exe).exists():
            ffmpeg_exe = shutil.which("ffmpeg")

        if ffmpeg_exe and Path(ffmpeg_exe).exists():
            try:
                # Transcode with veryfast preset, yuv420p for instant universal browser rendering
                cmd = [
                    str(ffmpeg_exe),
                    "-y",
                    "-i", str(raw_p),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-preset", "veryfast",
                    "-crf", "22",
                    "-r", str(fps),
                    str(final_p),
                ]
                kwargs: dict[str, Any] = {
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.PIPE,
                    "timeout": 120,
                }
                if os.name == "nt":
                    kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)

                result = subprocess.run(cmd, **kwargs)
                if result.returncode == 0 and final_p.exists() and final_p.stat().st_size > 0:
                    raw_p.unlink(missing_ok=True)
                    return str(final_p)
                else:
                    print(f"ffmpeg transcode warning: {result.stderr.decode('utf-8', errors='ignore')[:300]}")
            except Exception as e:
                print(f"ffmpeg transcode failed: {e}")

        # Fallback: Use the raw OpenCV mp4 directly
        if raw_p.exists():
            if raw_p != final_p:
                try:
                    final_p.unlink(missing_ok=True)
                    raw_p.rename(final_p)
                    return str(final_p)
                except Exception:
                    return str(raw_p)
            return str(raw_p)

        return str(final_p)

    @staticmethod
    def run_from_terminal():
        video_path = input("Video path: ").strip().strip('"')
        query = input("Target query: ").strip()

        if not video_path:
            raise ValueError("Video path is required.")
        if not query:
            raise ValueError("Query is required.")

        output_dir = str(Path(video_path).resolve().parent / "annotated_output")

        def console_progress(cur, total, fps, stage):
            pct = int(cur / max(1, total) * 100)
            print(f"\r[{pct}%] {cur}/{total} frames @ {fps:.1f} FPS - {stage}", end="", flush=True)

        print(f"\nStarting video detection for '{query}'...")
        result = VideoTracker().process_video(
            video_path=video_path,
            query=query,
            output_dir=output_dir,
            progress_callback=console_progress,
        )

        print("\n\n==============================")
        print("VIDEO TRACKING SUMMARY")
        print("==============================")
        print(f"Input video: {video_path}")
        print(f"Query: {query}")
        print(f"Output video: {result.output_video_path}")
        print(f"Frames processed: {result.frames_processed}")
        print(f"Average FPS: {result.fps:.2f}")
        print(f"Total processing time: {result.total_processing_time:.2f}s")
        print(f"Tracked targets: {result.tracked_targets}")
