from typing import List, Optional
from v2.schemas.state import QuerySpec, Candidate, VerificationResult, ConstraintResult, MediaMetadata
from PIL import Image
from v2.models.semantic_adapter import SemanticEmbeddingAdapter, MockSemanticAdapter

class VerificationAgentV2:
    """
    Generic verification agent. Evaluates candidates against all constraints
    in the QuerySpec.
    """
    def __init__(self, semantic_adapter: Optional[SemanticEmbeddingAdapter] = None):
        # Default to a mock if none provided (e.g. for unit tests)
        self.semantic_adapter = semantic_adapter or MockSemanticAdapter()
        
    def run(self, query_spec: QuerySpec, candidates: List[Candidate], media_metadata: MediaMetadata = None, image: Image.Image = None) -> List[VerificationResult]:
        results = []
        for candidate in candidates:
            verification = self._verify_candidate(query_spec, candidate, media_metadata, image)
            results.append(verification)
        return results

    def _get_crop(self, image: Image.Image, candidate: Candidate) -> Optional[Image.Image]:
        if image is None:
            return None
        x1, y1, x2, y2 = candidate.bbox
        # Clamp to image boundaries
        w, h = image.size
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return image.crop((x1, y1, x2, y2))

    def _verify_candidate(self, query: QuerySpec, candidate: Candidate, media_meta: MediaMetadata = None, image: Image.Image = None) -> VerificationResult:
        from v2.schemas.state import ConstraintStatus
        
        constraint_results = []
        overall_status = ConstraintStatus.SATISFIED
        rejection_reason = None
        overall_conf = 1.0
        
        candidate_crop = self._get_crop(image, candidate)
        
        for constraint in query.constraints:
            res = self._evaluate_constraint(constraint, candidate, media_meta, candidate_crop)
            constraint_results.append(res)
            
        if query.reference_image_path:
            # We don't fast-fail evaluating constraints here to keep logic clean, 
            # but if it's already violated we could skip. We'll evaluate it anyway or skip if VIOLATED.
            # Actually, to save compute, we skip if already violated.
            has_violated = any(c.status == ConstraintStatus.VIOLATED for c in constraint_results)
            if not has_violated:
                ref_res = self._evaluate_reference_image(query.reference_image_path, candidate, candidate_crop)
                constraint_results.append(ref_res)
                
        # Aggregation Logic
        statuses = [c.status for c in constraint_results]
        
        if ConstraintStatus.VIOLATED in statuses:
            overall_status = ConstraintStatus.VIOLATED
            violated_constraints = [c.constraint_id for c in constraint_results if c.status == ConstraintStatus.VIOLATED]
            rejection_reason = f"Violated constraints: {', '.join(violated_constraints)}"
            overall_conf = 0.0
        elif ConstraintStatus.UNCERTAIN in statuses:
            overall_status = ConstraintStatus.UNCERTAIN
            rejection_reason = "Uncertain verification due to low confidence evidence."
            overall_conf = 0.0
        elif ConstraintStatus.UNSUPPORTED in statuses:
            overall_status = ConstraintStatus.UNSUPPORTED
            rejection_reason = "Unsupported constraints in query."
            overall_conf = 0.0
        else:
            overall_status = ConstraintStatus.SATISFIED
            if constraint_results:
                overall_conf = min(c.confidence for c in constraint_results)
            else:
                overall_conf = candidate.confidence
            
        return VerificationResult(
            candidate_id=candidate.id,
            status=overall_status,
            satisfies_query=(overall_status == ConstraintStatus.SATISFIED),
            constraint_results=constraint_results,
            overall_confidence=overall_conf,
            ambiguity=1.0 if overall_status in (ConstraintStatus.UNCERTAIN, ConstraintStatus.UNSUPPORTED) else 0.0,
            rejection_reason=rejection_reason
        )

    def _evaluate_constraint(self, constraint, candidate: Candidate, media_meta: MediaMetadata = None, candidate_crop: Image.Image = None) -> ConstraintResult:
        from v2.schemas.state import ConstraintStatus
        
        if constraint.constraint_type == "attribute":
            if candidate.source == "OPEN_WORLD":
                return ConstraintResult(
                    constraint_id=constraint.value,
                    status=ConstraintStatus.SATISFIED,
                    satisfied=True,
                    confidence=candidate.confidence,
                    evidence="Pre-verified by YOLO-World open-vocabulary detector"
                )
                
            if candidate_crop is None:
                return ConstraintResult(
                    constraint_id=constraint.value,
                    status=ConstraintStatus.UNSUPPORTED,
                    satisfied=False,
                    confidence=0.0,
                    evidence="Missing candidate crop, cannot evaluate semantic attribute"
                )
            score = self.semantic_adapter.score_image_against_text(candidate_crop, constraint.value)
            
            # Semantic similarity thresholds
            accept_threshold = 0.25
            reject_threshold = 0.20
            
            if score >= accept_threshold:
                status = ConstraintStatus.SATISFIED
            elif score < reject_threshold:
                status = ConstraintStatus.VIOLATED
            else:
                status = ConstraintStatus.UNCERTAIN
            
            return ConstraintResult(
                constraint_id=constraint.value,
                status=status,
                satisfied=(status == ConstraintStatus.SATISFIED),
                confidence=1.0 if status == ConstraintStatus.SATISFIED else score,
                evidence=f"Semantic similarity: {score:.3f} (Thresholds: Accept>={accept_threshold}, Reject<{reject_threshold})"
            )
        elif constraint.constraint_type == "spatial":
            val = constraint.value.lower()
            if media_meta and media_meta.resolution:
                w, h = media_meta.resolution
                x1, y1, x2, y2 = candidate.bbox
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                
                satisfied = False
                if val == "left" and cx < w / 2: satisfied = True
                elif val == "right" and cx > w / 2: satisfied = True
                elif val == "top" and cy < h / 2: satisfied = True
                elif val == "bottom" and cy > h / 2: satisfied = True
                
                status = ConstraintStatus.SATISFIED if satisfied else ConstraintStatus.VIOLATED
                return ConstraintResult(
                    constraint_id=constraint.value,
                    status=status,
                    satisfied=satisfied,
                    confidence=1.0,
                    evidence=f"Candidate is {'on' if satisfied else 'not on'} the {val}"
                )
                    
            return ConstraintResult(
                constraint_id=constraint.value,
                status=ConstraintStatus.UNSUPPORTED,
                satisfied=False,
                confidence=0.0,
                evidence="Spatial reasoning requires media resolution, which is missing."
            )
        elif constraint.constraint_type == "relation":
            return ConstraintResult(
                constraint_id=constraint.value,
                status=ConstraintStatus.UNSUPPORTED,
                satisfied=False,
                confidence=0.0,
                evidence=f"Relation '{constraint.value}' not fully supported in deterministic backend."
            )
        else:
            return ConstraintResult(
                constraint_id=constraint.value,
                status=ConstraintStatus.UNSUPPORTED,
                satisfied=False,
                confidence=0.0,
                evidence=f"Unsupported constraint type: {constraint.constraint_type}"
            )
            
    def _evaluate_reference_image(self, ref_path: str, candidate: Candidate, candidate_crop: Image.Image = None) -> ConstraintResult:
        """
        Evaluates a candidate crop against a reference image.
        """
        from v2.schemas.state import ConstraintStatus
        if candidate_crop is None:
            return ConstraintResult(
                constraint_id="reference_image_similarity",
                status=ConstraintStatus.UNSUPPORTED,
                satisfied=False,
                confidence=0.0,
                evidence="Missing candidate crop, cannot evaluate reference image"
            )
            
        try:
            ref_image = Image.open(ref_path).convert("RGB")
        except Exception as e:
            return ConstraintResult(
                constraint_id="reference_image_similarity",
                status=ConstraintStatus.UNSUPPORTED,
                satisfied=False,
                confidence=0.0,
                evidence=f"Failed to load reference image: {e}"
            )
            
        score = self.semantic_adapter.score_image_against_image(candidate_crop, ref_image)
        
        accept_threshold = 0.70
        reject_threshold = 0.60
        
        if score >= accept_threshold:
            status = ConstraintStatus.SATISFIED
        elif score < reject_threshold:
            status = ConstraintStatus.VIOLATED
        else:
            status = ConstraintStatus.UNCERTAIN
            
        return ConstraintResult(
            constraint_id="reference_image_similarity",
            status=status,
            satisfied=(status == ConstraintStatus.SATISFIED),
            confidence=1.0 if status == ConstraintStatus.SATISFIED else score,
            evidence=f"Reference image similarity: {score:.3f} (Thresholds: Accept>={accept_threshold}, Reject<{reject_threshold})"
        )
