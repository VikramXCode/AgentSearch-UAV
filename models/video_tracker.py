from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

from models.yolo_world import YOLOWorldDetector


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


class VideoTracker:
    def __init__(self, model_path: str | None = None):
        self.yolo_world = YOLOWorldDetector(model_path=model_path)

    @staticmethod
    def _box_center(box):
        x1, y1, x2, y2 = [float(v) for v in box]
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _box_area(box):
        x1, y1, x2, y2 = [float(v) for v in box]
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        return w * h

    @staticmethod
    def _box_distance(box_a, box_b):
        a = VideoTracker._box_center(box_a)
        b = VideoTracker._box_center(box_b)
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    @staticmethod
    def _filter_detections(frame_shape, detections, query: str):
        height, width = frame_shape[:2]
        filtered = []
        for det in detections:
            bbox = det.get("bbox")
            if bbox is None:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            if x2 <= x1 or y2 <= y1:
                continue
            w = x2 - x1
            h = y2 - y1
            area = w * h
            if area < 1200:
                continue
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                continue
            margin = min(x1, y1, width - x2, height - y2)
            if margin < 8:
                continue
            label = str(det.get("label", "")).lower()
            if query.lower() not in label and label not in {"car", "vehicle", "truck", "bus"}:
                continue
            filtered.append(det)
        return filtered

    @staticmethod
    def _score_detection(frame_shape, det):
        bbox = det["bbox"]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        h, w = frame_shape[:2]
        center_score = -abs(cx - (w / 2.0)) - abs(cy - (h / 2.0))
        area_score = VideoTracker._box_area(bbox)
        conf_score = float(det.get("confidence", 0.0))
        return (area_score * 0.0001) + conf_score + center_score * 0.001

    @staticmethod
    def _select_best_detection(frame_shape, detections, query: str):
        valid = VideoTracker._filter_detections(frame_shape, detections, query)
        if not valid:
            return None
        return max(valid, key=lambda det: VideoTracker._score_detection(frame_shape, det))

    @staticmethod
    def _select_best_detections(frame_shape, detections, query: str, max_targets: int | None = None):
        valid = VideoTracker._filter_detections(frame_shape, detections, query)
        if not valid:
            return []
        ranked = sorted(valid, key=lambda det: VideoTracker._score_detection(frame_shape, det), reverse=True)
        if max_targets is None:
            return ranked
        return ranked[:max_targets]

    @staticmethod
    def _match_tracks(tracks: dict[int, dict[str, Any]], detections: list[dict[str, Any]]):
        matched_detections: set[int] = set()
        updated: dict[int, dict[str, Any]] = {}

        for track_id, track in sorted(tracks.items()):
            best_det_idx = None
            best_score = None
            for idx, det in enumerate(detections):
                if idx in matched_detections:
                    continue
                dist = VideoTracker._box_distance(track["bbox"], det["bbox"])
                if best_score is None or dist < best_score:
                    best_score = dist
                    best_det_idx = idx

            if best_det_idx is not None and (best_score is None or best_score < 120.0):
                det = detections[best_det_idx]
                matched_detections.add(best_det_idx)
                updated[track_id] = {
                    "label": det["label"],
                    "confidence": det["confidence"],
                    "bbox": det["bbox"],
                    "last_seen": track.get("last_seen", 0) + 1,
                }
            else:
                updated[track_id] = track

        for idx, det in enumerate(detections):
            if idx in matched_detections:
                continue
            new_id = max(tracks.keys(), default=0) + 1
            updated[new_id] = {
                "label": det["label"],
                "confidence": det["confidence"],
                "bbox": det["bbox"],
                "last_seen": 1,
            }

        return updated

    @staticmethod
    def _motion_follow_box(prev_frame, curr_frame, prev_bbox):
        if prev_frame is None or curr_frame is None:
            return None, 0.0

        x1, y1, x2, y2 = [float(v) for v in prev_bbox]
        width = max(10.0, x2 - x1)
        height = max(10.0, y2 - y1)

        pad_x = max(20.0, width * 0.25)
        pad_y = max(20.0, height * 0.25)

        search_x1 = max(0, int(x1 - pad_x))
        search_y1 = max(0, int(y1 - pad_y))
        search_x2 = min(curr_frame.shape[1], int(x2 + pad_x))
        search_y2 = min(curr_frame.shape[0], int(y2 + pad_y))

        prev_crop = prev_frame[int(y1):int(y2), int(x1):int(x2)]
        if prev_crop.size == 0:
            return None, 0.0

        search_region = curr_frame[search_y1:search_y2, search_x1:search_x2]
        if search_region.size == 0:
            return None, 0.0

        prev_gray = cv2.cvtColor(prev_crop, cv2.COLOR_BGR2GRAY)
        search_gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
        if prev_gray.shape[0] < 10 or prev_gray.shape[1] < 10:
            return None, 0.0

        result = cv2.matchTemplate(search_gray, prev_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < 0.7:
            return None, max_val

        dx = max_loc[0]
        dy = max_loc[1]
        new_x1 = search_x1 + dx
        new_y1 = search_y1 + dy
        new_x2 = new_x1 + width
        new_y2 = new_y1 + height

        new_bbox = [
            max(0.0, new_x1),
            max(0.0, new_y1),
            min(float(curr_frame.shape[1]), new_x2),
            min(float(curr_frame.shape[0]), new_y2),
        ]
        return new_bbox, max_val

    @staticmethod
    def run_from_terminal():
        video_path = input("Video path: ").strip().strip('"')
        query = input("Target query: ").strip()

        if not video_path:
            raise ValueError("Video path is required.")
        if not query:
            raise ValueError("Query is required.")

        output_dir = str(Path(video_path).resolve().parent / "annotated_output")
        result = VideoTracker().process_video(
            video_path=video_path,
            query=query,
            output_dir=output_dir,
            max_skip_frames=0,
            recheck_every=25,
        )

        print("\n==============================")
        print("VIDEO TRACKING SUMMARY")
        print("==============================")
        print(f"Input video: {video_path}")
        print(f"Query: {query}")
        print(f"Output video: {result.output_video_path}")
        print(f"Frames processed: {result.frames_processed}")
        print(f"Skipped frames: {result.skipped_frames}")
        print(f"Average FPS: {result.fps:.2f}")
        print(f"Total processing time: {result.total_processing_time:.2f}s")
        print(f"Tracked targets: {result.tracked_targets}")

    def process_video(
        self,
        video_path: str,
        query: str,
        output_dir: str | None = None,
        max_skip_frames: int = 0,
        recheck_every: int = 12,
        output_stride: int = 2,
        seed_bbox: tuple[float, float, float, float] | None = None,
    ) -> VideoTrackingResult:
        video_path = str(video_path)
        output_dir = output_dir or str(Path(video_path).resolve().parent)
        output_path = str(Path(output_dir) / "tracked_result.mp4")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Video could not be opened: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        if width <= 0 or height <= 0:
            raise ValueError(f"Video has invalid frame dimensions: {video_path}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, original_fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for: {output_path}")

        tracks: dict[int, dict[str, Any]] = {}
        previous_frame = None
        previous_bbox = None
        frame_index = 0
        frames_processed = 0
        skipped_frames = 0
        detection_time = 0.0
        tracking_time = 0.0
        total_start = time.perf_counter()
        detections: list[dict[str, Any]] = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if max_skip_frames > 0 and frame_index > 0 and frame_index % (max_skip_frames + 1) != 0:
                skipped_frames += 1
                frame_index += 1
                continue

            frame_start = time.perf_counter()
            frames_processed += 1
            annotated = frame.copy()

            needs_detection = not tracks or frame_index % recheck_every == 0
            active_bbox = previous_bbox

            if needs_detection:
                detection_start = time.perf_counter()
                candidate_boxes = self._detect_initial_boxes(frame, query)
                detection_time += time.perf_counter() - detection_start
                filtered = self._filter_detections(frame.shape, candidate_boxes, query)
                if filtered:
                    ranked = self._select_best_detections(frame.shape, filtered, query, max_targets=None)
                    if ranked:
                        active_bbox = ranked[0]["bbox"]
                        tracks = {
                            idx + 1: {
                                "label": det["label"],
                                "confidence": det["confidence"],
                                "bbox": det["bbox"],
                                "last_seen": 1,
                            }
                            for idx, det in enumerate(ranked)
                        }
                    else:
                        tracks = {}
                elif not tracks:
                    cv2.putText(annotated, f"No match for: {query}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    writer.write(annotated)
                    previous_frame = frame.copy()
                    previous_bbox = None
                    frame_index += 1
                    continue
            elif previous_bbox is not None:
                motion_bbox, motion_score = self._motion_follow_box(previous_frame, frame, previous_bbox)
                if motion_bbox is not None and motion_score > 0.35:
                    active_bbox = motion_bbox
                    tracks = {1: {"label": query, "confidence": 0.8, "bbox": active_bbox, "last_seen": 1}}
                else:
                    active_bbox = previous_bbox

            if tracks:
                for track_id, track in sorted(tracks.items()):
                    bbox = track["bbox"] if active_bbox is None else track["bbox"]
                    x1, y1, x2, y2 = [float(v) for v in bbox]
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(
                        annotated,
                        f"{query}",
                        (int(x1), max(20, int(y1) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                    detections.append({
                        "label": track.get("label", query),
                        "confidence": float(track.get("confidence", 0.9)),
                        "bbox": [float(v) for v in bbox],
                        "track_id": track_id,
                    })
                    previous_bbox = bbox

            previous_frame = frame.copy()
            tracking_time += time.perf_counter() - frame_start
            if output_stride <= 1 or frame_index % output_stride == 0:
                writer.write(annotated)
            frame_index += 1

        cap.release()
        writer.release()

        total_processing_time = time.perf_counter() - total_start
        avg_fps = frames_processed / total_processing_time if total_processing_time > 0 else 0.0

        tracked_count = len(detections) if detections else 0
        return VideoTrackingResult(
            output_video_path=output_path,
            total_processing_time=total_processing_time,
            detection_time=detection_time,
            tracking_time=tracking_time,
            frames_processed=frames_processed,
            skipped_frames=skipped_frames,
            tracked_targets=tracked_count,
            fps=avg_fps,
            detections=detections,
            metrics={
                "fps": avg_fps,
                "total_processing_time": total_processing_time,
                "detection_time": detection_time,
                "tracking_time": tracking_time,
                "frames_processed": frames_processed,
                "skipped_frames": skipped_frames,
                "tracked_targets": tracked_count,
            },
        )

    def _detect_initial_boxes(self, frame, query: str):
        frame_h, frame_w = frame.shape[:2]
        scale = 1.0
        max_side = 1280
        if max(frame_h, frame_w) > max_side:
            scale = max_side / float(max(frame_h, frame_w))
            frame = cv2.resize(frame, (max(1, int(frame_w * scale)), max(1, int(frame_h * scale))), interpolation=cv2.INTER_LINEAR)

        temp_path = Path("outputs") / "video_tracker_probe.jpg"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(temp_path), frame)
        try:
            detections, _ = self.yolo_world.detect(
                image_path=str(temp_path),
                classes=[query],
                confidence=0.25,
            )
        finally:
            temp_path.unlink(missing_ok=True)

        results = []
        for det in detections:
            bbox = [float(v) for v in det.bbox]
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            if scale != 1.0:
                bbox = [bbox[0] / scale, bbox[1] / scale, bbox[2] / scale, bbox[3] / scale]
            results.append({
                "label": det.label,
                "confidence": float(det.confidence),
                "bbox": bbox,
            })

        if not results:
            return []

        results.sort(key=lambda item: float(item["confidence"]), reverse=True)
        return results
