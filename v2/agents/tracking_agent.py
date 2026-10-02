from typing import List, Dict, Optional, Tuple
from v2.schemas.state import AgentStateV2, Candidate
import uuid

def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

class TrackState:
    def __init__(self, track_id: int, bbox: List[float], label: str):
        self.track_id = track_id
        self.last_bbox = bbox
        self.label = label
        self.missed_frames = 0
        self.active = True

class TrackingAgentV2:
    """
    Temporal association component for V2 architecture.
    """
    def __init__(self):
        self.next_track_id = 1
        self.tracks: Dict[int, TrackState] = {}
        self.max_missed_frames = 5
        
    def run(self, state: AgentStateV2, new_candidates: Optional[List[Candidate]] = None) -> AgentStateV2:
        """
        Main entry point to update tracking state.
        If `new_candidates` is provided (e.g., from a REDETECT or initial DETECT),
        we associate them with existing tracks or spawn new ones.
        If no candidates, we assume normal frame progression and simulate track decay.
        """
        if new_candidates is not None:
            self.associate_candidates(new_candidates)
        else:
            self._simulate_missing_frame()
            
        # Update AgentStateV2 tracking state
        active_count = sum(1 for t in self.tracks.values() if t.active)
        state.tracking_state.active_tracks = active_count
        state.tracking_state.frames_processed += 1
        
        # Compute a simple overall degradation score based on missed frames of active tracks
        if active_count > 0:
            avg_missed = sum(t.missed_frames for t in self.tracks.values() if t.active) / active_count
            degradation = min(1.0, avg_missed / self.max_missed_frames)
        else:
            # If we lost all tracks that were active, degradation is high
            degradation = 1.0 if len(self.tracks) > 0 else 0.0
            
        state.tracking_state.track_degradation_score = degradation
        
        # Populate candidates with current track info
        tracked_candidates = []
        for t in self.tracks.values():
            if t.active:
                tracked_candidates.append(Candidate(
                    id=str(uuid.uuid4()),
                    bbox=t.last_bbox,
                    class_label=t.label,
                    confidence=1.0 - (t.missed_frames * 0.1), # decay conf
                    source="TRACKER",
                    track_id=t.track_id,
                    frame_id=state.tracking_state.frames_processed
                ))
        state.candidates = tracked_candidates
        return state

    def associate_candidates(self, candidates: List[Candidate]):
        """
        Associates fresh detections with existing tracks using greedy IoU matching.
        Unmatched detections spawn new tracks.
        """
        unmatched_candidates = list(candidates)
        
        # Simple Greedy IoU matching
        for track_id, track in self.tracks.items():
            if not track.active and track.missed_frames > self.max_missed_frames:
                continue # completely dead track
                
            best_iou = 0.0
            best_candidate = None
            
            for cand in unmatched_candidates:
                if cand.class_label == track.label:
                    iou = compute_iou(track.last_bbox, cand.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_candidate = cand
                        
            # Threshold for association
            if best_iou > 0.3 and best_candidate:
                # Associated!
                track.last_bbox = best_candidate.bbox
                track.missed_frames = 0
                track.active = True
                best_candidate.track_id = track.track_id
                unmatched_candidates.remove(best_candidate)
            else:
                # Missed
                track.missed_frames += 1
                if track.missed_frames > self.max_missed_frames:
                    track.active = False
                    
        # Spawn new tracks for unmatched candidates
        for cand in unmatched_candidates:
            new_id = self.next_track_id
            self.next_track_id += 1
            self.tracks[new_id] = TrackState(new_id, cand.bbox, cand.class_label)
            cand.track_id = new_id

    def _simulate_missing_frame(self):
        """Simulates frame advancement without new detections (tracker coasting)."""
        for track in self.tracks.values():
            if track.active:
                track.missed_frames += 1
                if track.missed_frames > self.max_missed_frames:
                    track.active = False
