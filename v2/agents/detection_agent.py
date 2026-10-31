from v2.schemas.state import AgentStateV2, ActionType
from v2.models.model_registry import ModelRegistry
from typing import Any, Optional

class DetectionAgentV2:
    """
    Executes detection based on the planner's routing decision.
    Uses the model registry to obtain the correct adapter.
    """
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        
    def run(self, state: AgentStateV2, image: Any, mock_model_instance: Optional[Any] = None) -> AgentStateV2:
        action = state.plan.current_action
        
        logical_name = None
        if action in (ActionType.DETECT_SPECIALIST, ActionType.REDETECT):
            # REDETECT should ideally check detector_routing, but we default to specialist or previous for now
            logical_name = "specialist_p2"
        elif action == ActionType.DETECT_OPEN_WORLD:
            logical_name = "open_world_yolo_world"
        else:
            state.failure_reason = f"DetectionAgent called with invalid action: {action}"
            return state
            
        try:
            adapter = self.registry.get_adapter(logical_name, mock_instance=mock_model_instance)
        except ValueError as e:
            state.failure_reason = str(e)
            return state
            
        candidates = adapter.detect(image, state.query_spec, state.media_metadata)
        
        # Override candidates
        state.candidates = candidates
        state.detector_routing.selected_detector = logical_name
        
        return state
