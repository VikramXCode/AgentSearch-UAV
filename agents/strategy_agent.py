from workflows.state import AgentState


class StrategyAgent:

    def run(self, state: AgentState) -> AgentState:

        print("\n==============================")
        print("     STRATEGY AGENT")
        print("==============================")

        target = state.query.target.lower()

        reasoning = []
        execution = []

        # Default detector
        detector = "YOLO-World"

        # Small objects generally benefit from SR + SAHI
        small_objects = [
            "person",
            "pedestrian",
            "bicycle",
            "motorcycle",
            "dog",
            "cat",
            "bird",
            "bottle",
            "backpack"
        ]

        if target in small_objects:
            state.strategy.enable_super_resolution = True
            state.strategy.enable_sahi = True

            reasoning.append(
                f"{target} is likely to appear as a small object in UAV imagery."
            )

            execution.append("Super Resolution")
            execution.append("SAHI")

        # Attribute-based search
        if len(state.query.attributes) > 0:

            state.strategy.enable_clip_verification = True

            reasoning.append(
                "Attribute-based search detected. Enable CLIP verification."
            )

        execution.append(detector)

        if state.strategy.enable_clip_verification:
            execution.append("CLIP Verification")

        state.strategy.detector = detector
        state.strategy.execution_priority = execution
        state.strategy.reasoning = reasoning

        print("\nStrategy")

        print(state.strategy.model_dump())

        return state