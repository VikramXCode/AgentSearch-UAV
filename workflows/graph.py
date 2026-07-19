import os
from datetime import datetime, timezone

from agents.detection_agent import DetectionAgent
from agents.explanation_agent import ExplanationAgent
from agents.knowledge_agent import KnowledgeAgent
from agents.query_agent import QueryAgent
from agents.strategy_agent import StrategyAgent
from agents.verification_agent import VerificationAgent
from workflows.state import AgentState
from utils.paths import DETECTION_OUTPUT_PATH

def main():

    state = AgentState()

    query_agent = QueryAgent()
    knowledge_agent = KnowledgeAgent()
    strategy_agent = StrategyAgent()
    detection_agent = DetectionAgent()
    verification_agent = VerificationAgent()
    explanation_agent = ExplanationAgent()

    state.mission.mission_id = f"mission-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    state.mission.steps = [
        "query",
        "knowledge",
        "strategy",
        "detection",
        "verification",
        "explanation",
    ]
    state.mission.current_step = 0
    state.mission.status = "Running"

    try:
        state = query_agent.run(state)
        state.mission.current_step = 1

        state = knowledge_agent.run(state)
        state.mission.current_step = 2

        state = strategy_agent.run(state)
        state.mission.current_step = 3

        image_path = input("Enter image path: ").strip()

        if not image_path:
            raise ValueError("Image path is required.")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        state = strategy_agent.refine_with_image(state, image_path)
        state.mission.current_step = 3

        state = detection_agent.run(state, image_path=image_path)
        state.mission.current_step = 4

        if state.strategy.enable_clip_verification:
            state = verification_agent.run(state)
        else:
            state.verification.verified_objects = list(state.detection.objects_found)
            state.verification.confidence_score = _average_confidence(state.detection.objects_found)

        final_objects = state.verification.verified_objects if state.strategy.enable_clip_verification else state.detection.objects_found

        if len(final_objects) > 0:
            detection_agent.save_annotated_image(
                image_path=state.detection.processed_image_path or state.detection.image_path,
                detections=final_objects,
                output_path=DETECTION_OUTPUT_PATH,
            )

        state.mission.current_step = 5

        state = explanation_agent.run(state)

        final_objects = state.verification.verified_objects if state.strategy.enable_clip_verification else state.detection.objects_found

        state.mission.status = "Completed" if final_objects else "No matching objects detected"
        knowledge_agent.record_search(state)

    except Exception as exc:
        state.mission.status = f"Failed: {exc}"
        print(f"\nPipeline error: {exc}")

    print("\n==============================")
    print("MISSION SUMMARY")
    print("==============================")

    if state.mission.status.startswith("Failed"):
        print("--------------------------------------------------")
        print("Mission Failed")
        print()
        print(f"Reason: {state.mission.status}")
        print(f"Output Image: {state.detection.output_image_path or DETECTION_OUTPUT_PATH}")
        print("--------------------------------------------------")
        print("\n==============================")
        print(" CURRENT AGENT STATE")
        print("==============================")
        print(state.model_dump_json(indent=4))
        return

    final_objects = state.verification.verified_objects if state.strategy.enable_clip_verification else state.detection.objects_found
    final_confidence = state.verification.confidence_score if state.strategy.enable_clip_verification else _average_confidence(state.detection.objects_found)

    print("--------------------------------------------------")
    print("Mission Completed")
    print()
    print(f"Target: {state.query.target}")
    print()
    if state.query.attributes:
        print(f"Attributes: {', '.join(f'{key}={value}' for key, value in state.query.attributes.items())}")
        print()
    if len(final_objects) == 0:
        print("No matching objects detected.")
    print(f"Objects Found: {len(final_objects)}")
    print(f"Confidence: {final_confidence:.2f}")
    print(f"Detector: {state.strategy.detector}")
    print(f"Verification: {'CLIP' if state.strategy.enable_clip_verification else 'None'}")
    print(f"Output Image: {state.detection.output_image_path or DETECTION_OUTPUT_PATH}")
    print("--------------------------------------------------")

    print("\n==============================")
    print(" CURRENT AGENT STATE")
    print("==============================")

    print(state.model_dump_json(indent=4))


def _average_confidence(detections: list[dict]) -> float:

    if len(detections) == 0:
        return 0.0

    return sum(float(detection["confidence"]) for detection in detections) / len(detections)


if __name__ == "__main__":
    main()