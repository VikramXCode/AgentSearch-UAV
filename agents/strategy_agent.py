from workflows.state import AgentState


class StrategyAgent:

    def __init__(self):
        self._detector_name = "YOLO-World"

    def run(self, state: AgentState) -> AgentState:
        print("\n==============================")
        print("     STRATEGY AGENT")
        print("==============================")

        target = (state.query.target or "").lower().strip()
        reasoning = []
        execution = []

        detector = self._detector_name

        small_objects = {
            "person",
            "pedestrian",
            "bicycle",
            "motorcycle",
            "dog",
            "cat",
            "bird",
            "bottle",
            "backpack",
            "traffic light",
            "bench",
        }

        if target in small_objects:
            state.strategy.enable_super_resolution = True
            state.strategy.enable_sahi = True
            reasoning.append(
                f"{target} is likely to appear as a small object in UAV imagery."
            )
            execution.extend(["Super Resolution", "SAHI"])

        attribute_filters = {
            key: value for key, value in state.query.attributes.items() if key != "quantity"
        }

        if len(attribute_filters) > 0 or state.query.target:
            state.strategy.enable_clip_verification = True
            reasoning.append("Semantic attribute search detected. Enable CLIP verification.")

        state.strategy.detector = detector
        state.strategy.execution_priority = self._build_execution_priority(state, detector, execution)
        state.strategy.reasoning = reasoning

        print("\nStrategy")
        print(state.strategy.model_dump())

        return state

    def refine_with_image(self, state: AgentState, image_path: str) -> AgentState:
        from PIL import Image

        width, height = Image.open(image_path).size
        total_pixels = width * height
        max_side = max(width, height)
        query_text = state.query.raw_query.lower()

        if not state.strategy.reasoning:
            state.strategy.reasoning = []

        if any(keyword in query_text for keyword in ("uav", "drone", "aerial", "satellite")):
            state.strategy.enable_sahi = True
            state.strategy.reasoning.append("UAV/satellite-style query detected. Enable SAHI.")

        if total_pixels >= 1_500_000 or max_side >= 1600:
            state.strategy.enable_sahi = True
            state.strategy.reasoning.append("Large image detected. Enable SAHI for tiled inference.")

        if total_pixels <= 800_000 or max_side <= 900:
            state.strategy.enable_super_resolution = True
            state.strategy.reasoning.append("Low-resolution image detected. Enable Super Resolution.")

        if any(keyword in query_text for keyword in ("low resolution", "blurry", "blurred", "tiny", "small", "distant", "far", "hard to see")):
            state.strategy.enable_super_resolution = True
            state.strategy.reasoning.append("Low-quality or small-object search detected. Enable Super Resolution.")

        # Keep confidence threshold >= 0.30 by default for user-facing results
        state.strategy.confidence_threshold = max(0.30, state.strategy.confidence_threshold)

        state.strategy.execution_priority = self._build_execution_priority(state, self._detector_name, state.strategy.execution_priority)
        return state

    def _build_execution_priority(self, state: AgentState, detector: str, execution: list[str]) -> list[str]:
        ordered = []

        if state.strategy.enable_super_resolution:
            ordered.append("Super Resolution")
        if state.strategy.enable_sahi:
            ordered.append("SAHI")
        ordered.append(detector)
        if state.strategy.enable_clip_verification:
            ordered.append("CLIP Verification")

        unique_ordered = []
        seen = set()
        for step in ordered + execution:
            if step not in seen:
                seen.add(step)
                unique_ordered.append(step)
        return unique_ordered