from v2.schemas.state import AgentStateV2, ActionType, MediaType
import statistics

class PlanningAgentV2:
    """
    Deterministic state-machine planner with adaptive enhancement validation.
    """
    
    KNOWN_AERIAL_TARGETS = {"person", "car", "motorcycle", "bicycle", "truck", "pedestrian", "bus", "van", "awning-tricycle", "tricycle"}

    def __init__(self):
        # Configurable policy thresholds (NOT calibrated values, purely for logic routing)
        self.config_sahi_min_image_dimension = 2000
        self.config_sr_max_candidate_area = 1024  # 32x32
        self.config_sr_max_confidence = 0.4
        self.config_max_retries = 2

    def _candidate_statistics(self, state: AgentStateV2):
        count = len(state.candidates)
        if count == 0:
            return 0, 0.0, 0.0
            
        confidences = [c.confidence for c in state.candidates]
        max_conf = max(confidences)
        
        areas = []
        for c in state.candidates:
            x1, y1, x2, y2 = c.bbox
            areas.append((x2 - x1) * (y2 - y1))
        
        median_area = statistics.median(areas) if areas else 0.0
        return count, max_conf, median_area

    def _should_try_sahi(self, state: AgentStateV2, count: int) -> bool:
        # SAHI is disabled by default for the V2 P2 pipeline.
        return False

    def _should_try_sr(self, state: AgentStateV2, count: int, max_conf: float, median_area: float) -> bool:
        # SR is disabled by default for the V2 P2 pipeline.
        return False

    def run(self, state: AgentStateV2) -> ActionType:
        if not state.plan.history:
            return self._decide_initial_detection(state)
            
        last_action = state.plan.history[-1]
        
        if last_action in (ActionType.DETECT_SPECIALIST, ActionType.DETECT_OPEN_WORLD, ActionType.ENHANCE_SAHI, ActionType.REDETECT):
            return self._evaluate_detections(state)
            
        if last_action == ActionType.ENHANCE_SR:
            # SR modifies crops, we should verify them now.
            state.plan.decision_rationale = "SR applied to candidate crops. Proceeding to verification."
            return ActionType.VERIFY_CANDIDATES
            
        if last_action == ActionType.VERIFY_CANDIDATES:
            return self._evaluate_verification(state)
            
        if last_action == ActionType.TRACK:
            return self._evaluate_tracking(state)
            
        state.plan.decision_rationale = "No valid next action determined. Terminating."
        return ActionType.TERMINATE

    def _decide_initial_detection(self, state: AgentStateV2) -> ActionType:
        target = state.query_spec.target.lower()
        if target in self.KNOWN_AERIAL_TARGETS and not state.query_spec.constraints:
            state.plan.decision_rationale = f"Target '{target}' is known and simple. Routing to specialist."
            return ActionType.DETECT_SPECIALIST
        else:
            state.plan.decision_rationale = f"Target '{target}' is unknown or query has complex constraints. Routing to open-world detector."
            return ActionType.DETECT_OPEN_WORLD

    def _evaluate_detections(self, state: AgentStateV2) -> ActionType:
        count, max_conf, median_area = self._candidate_statistics(state)
        
        if self._should_try_sahi(state, count):
            state.plan.decision_rationale = "Zero candidates found in high-res image (budget remains). Using SAHI."
            state.tool_usage.sahi_used = True
            state.tool_usage.retry_count += 1
            return ActionType.ENHANCE_SAHI
            
        if count == 0:
            state.plan.decision_rationale = "Zero candidates and SAHI not eligible. Terminating."
            return ActionType.TERMINATE
            
        if self._should_try_sr(state, count, max_conf, median_area):
            state.plan.decision_rationale = "Candidates exist but are weak/tiny. Using SR."
            state.tool_usage.sr_used = True
            state.tool_usage.retry_count += 1
            return ActionType.ENHANCE_SR
            
        state.plan.decision_rationale = f"Found {count} confident candidates. Proceeding to verification."
        return ActionType.VERIFY_CANDIDATES

    def _evaluate_verification(self, state: AgentStateV2) -> ActionType:
        from v2.schemas.state import ConstraintStatus
        
        if any(v.status == ConstraintStatus.SATISFIED for v in state.verification_results):
            if state.media_metadata.type == MediaType.VIDEO:
                state.plan.decision_rationale = "Verified targets found in video. Proceeding to tracking."
                return ActionType.TRACK
            else:
                state.plan.decision_rationale = "Verified targets found in image. Mission success."
                return ActionType.TERMINATE
        
        has_uncertain = any(v.status == ConstraintStatus.UNCERTAIN for v in state.verification_results)
        has_violated = any(v.status == ConstraintStatus.VIOLATED for v in state.verification_results)
        has_unsupported = any(v.status == ConstraintStatus.UNSUPPORTED for v in state.verification_results)
        
        if has_unsupported:
             state.plan.decision_rationale = "Verification failed due to unsupported constraints. Terminating without retry."
             return ActionType.TERMINATE

        if has_uncertain or has_violated:
            if state.tool_usage.retry_count < self.config_max_retries:
                state.plan.decision_rationale = "Verification UNCERTAIN/VIOLATED. Retrying with REDETECT."
                state.tool_usage.retry_count += 1
                return ActionType.REDETECT
            else:
                state.plan.decision_rationale = "Verification UNCERTAIN/VIOLATED and retries exhausted. Terminating."
                return ActionType.TERMINATE
                
        state.plan.decision_rationale = "Verification failed or unsupported. Terminating."
        return ActionType.TERMINATE

    def _evaluate_tracking(self, state: AgentStateV2) -> ActionType:
        if state.tracking_state.track_degradation_score > 0.7:
            state.plan.decision_rationale = f"Track degradation is high ({state.tracking_state.track_degradation_score}). Requesting redetection."
            return ActionType.REDETECT
        
        state.plan.decision_rationale = "Tracking stable. Terminating tracker (conceptually for video end)."
        return ActionType.TERMINATE
