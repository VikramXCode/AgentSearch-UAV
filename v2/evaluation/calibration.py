import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict
from enum import Enum
import numpy as np

class GroundTruthLabel(str, Enum):
    MATCH = "MATCH"
    NON_MATCH = "NON_MATCH"

@dataclass
class CalibrationRecord:
    query_type: str  # "text" or "image"
    query_text: Optional[str]
    reference_image_path: Optional[str]
    candidate_crop_info: str # identifier or coordinates
    ground_truth: GroundTruthLabel
    similarity_score: float

class ThresholdEvaluator:
    def __init__(self, records: List[CalibrationRecord]):
        self.records = records

    def calculate_metrics_at_threshold(self, threshold: float) -> Dict[str, float]:
        tp, fp, tn, fn = 0, 0, 0, 0
        
        for r in self.records:
            predicted_match = r.similarity_score >= threshold
            actual_match = (r.ground_truth == GroundTruthLabel.MATCH)
            
            if predicted_match and actual_match:
                tp += 1
            elif predicted_match and not actual_match:
                fp += 1
            elif not predicted_match and not actual_match:
                tn += 1
            elif not predicted_match and actual_match:
                fn += 1
                
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "threshold": threshold,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def sweep_thresholds(self, steps: int = 100) -> List[Dict[str, float]]:
        if not self.records:
            return []
            
        scores = [r.similarity_score for r in self.records]
        min_score, max_score = min(scores), max(scores)
        
        # Add a tiny margin to endpoints
        thresholds = np.linspace(max(0.0, min_score - 0.05), min(1.0, max_score + 0.05), steps)
        
        results = []
        for t in thresholds:
            results.append(self.calculate_metrics_at_threshold(float(t)))
            
        return results
        
    def find_operating_points(self) -> Dict[str, Dict[str, float]]:
        """Finds candidate operating points (balanced, high-precision, high-recall)."""
        sweep = self.sweep_thresholds(200)
        if not sweep:
            return {}
            
        # Balanced F1 (max F1)
        best_f1 = max(sweep, key=lambda x: x["f1"])
        
        # High Precision (Target precision >= 0.9, then max recall)
        high_prec_candidates = [x for x in sweep if x["precision"] >= 0.9]
        best_high_prec = max(high_prec_candidates, key=lambda x: x["recall"]) if high_prec_candidates else None
        
        # High Recall (Target recall >= 0.9, then max precision)
        high_rec_candidates = [x for x in sweep if x["recall"] >= 0.9]
        best_high_rec = max(high_rec_candidates, key=lambda x: x["precision"]) if high_rec_candidates else None
        
        points = {"balanced_f1": best_f1}
        if best_high_prec: points["high_precision"] = best_high_prec
        if best_high_rec: points["high_recall"] = best_high_rec
        
        return points

    def get_score_distribution(self) -> Dict[str, List[float]]:
        return {
            "MATCH": [r.similarity_score for r in self.records if r.ground_truth == GroundTruthLabel.MATCH],
            "NON_MATCH": [r.similarity_score for r in self.records if r.ground_truth == GroundTruthLabel.NON_MATCH]
        }
