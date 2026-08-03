import os

from models.super_resolution import SuperResolutionEngine
from workflows.state import AgentState


class ToolAgent:

    def __init__(self):
        self.super_resolution = SuperResolutionEngine()

    def run(
        self,
        state: AgentState,
        image_path: str,
    ) -> tuple[AgentState, str]:

        print("\n==============================")
        print("       TOOL AGENT")
        print("==============================")

        processed_image_path = image_path

        if state.strategy.enable_super_resolution:

            processed_image_path = self.super_resolution.upscale(
                image_path=image_path,
                output_dir=os.path.join(
                    "outputs",
                    "preprocessed",
                ),
            )

            print(
                f"Super Resolution applied: "
                f"{processed_image_path}"
            )

        else:
            print("No preprocessing tools required.")

        state.detection.image_path = image_path
        state.detection.processed_image_path = processed_image_path

        return state, processed_image_path