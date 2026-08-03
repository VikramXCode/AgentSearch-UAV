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


def run_pipeline(query: str, image_path: str) -> AgentState:
    """
    Run the complete UAV object retrieval pipeline.

    This function can be called from:
    - CLI
    - API
    - Frontend backend
    - Tests
    """

    if not query.strip():
        raise ValueError("Query is required.")

    if not image_path.strip():
        raise ValueError("Image path is required.")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    state = AgentState()

    query_agent = QueryAgent()
    knowledge_agent = KnowledgeAgent()
    strategy_agent = StrategyAgent()
    detection_agent = DetectionAgent()
    verification_agent = VerificationAgent()
    explanation_agent = ExplanationAgent()

    # -----------------------------
    # Mission setup
    # -----------------------------

    state.mission.mission_id = (
        f"mission-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    )

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

        # -----------------------------
        # Query Agent
        # -----------------------------

        state = query_agent.run(
            state,
            query=query,
        )

        state.mission.current_step = 1

        # -----------------------------
        # Knowledge Agent
        # -----------------------------

        state = knowledge_agent.run(state)

        state.mission.current_step = 2

        # -----------------------------
        # Strategy Agent
        # -----------------------------

        state = strategy_agent.run(state)

        state = strategy_agent.refine_with_image(
            state,
            image_path,
        )

        state.mission.current_step = 3

        # -----------------------------
        # Detection Agent
        # -----------------------------

        state = detection_agent.run(
            state,
            image_path=image_path,
        )

        state.mission.current_step = 4

        # -----------------------------
        # Verification Agent
        # -----------------------------

        if state.strategy.enable_clip_verification:

            state = verification_agent.run(state)

        else:

            state.verification.verified_objects = list(
                state.detection.objects_found
            )

            state.verification.confidence_score = _average_confidence(
                state.detection.objects_found
            )

        # -----------------------------
        # Final objects
        # -----------------------------

        final_objects = (
            state.verification.verified_objects
            if state.strategy.enable_clip_verification
            else state.detection.objects_found
        )

        # -----------------------------
        # Final annotated image
        # -----------------------------

        if final_objects:

            detection_agent.save_annotated_image(
                image_path=(
                    state.detection.processed_image_path
                    or state.detection.image_path
                ),
                detections=final_objects,
                output_path=DETECTION_OUTPUT_PATH,
            )

        state.mission.current_step = 5

        # -----------------------------
        # Explanation Agent
        # -----------------------------

        state = explanation_agent.run(state)

        # -----------------------------
        # Mission result
        # -----------------------------

        state.mission.status = (
            "Completed"
            if final_objects
            else "No matching objects detected"
        )

        knowledge_agent.record_search(state)

    except Exception as exc:

        state.mission.status = f"Failed: {exc}"

        raise

    return state


def _average_confidence(detections: list[dict]) -> float:

    if not detections:
        return 0.0

    return sum(
        float(detection["confidence"])
        for detection in detections
    ) / len(detections)


def main():

    print("\n==============================")
    print("      UAV SEARCH SYSTEM")
    print("==============================")

    query = input("Enter search query: ").strip()
    image_path = input("Enter image path: ").strip()

    try:

        state = run_pipeline(
            query=query,
            image_path=image_path,
        )

    except Exception as exc:

        print(f"\nPipeline error: {exc}")
        return

    final_objects = (
        state.verification.verified_objects
        if state.strategy.enable_clip_verification
        else state.detection.objects_found
    )

    final_confidence = (
        state.verification.confidence_score
        if state.strategy.enable_clip_verification
        else _average_confidence(state.detection.objects_found)
    )

    print("\n==============================")
    print("MISSION SUMMARY")
    print("==============================")

    print("--------------------------------------------------")
    print(f"Status: {state.mission.status}")
    print()
    print(f"Target: {state.query.target}")

    if state.query.attributes:

        print(
            "Attributes: "
            + ", ".join(
                f"{key}={value}"
                for key, value in state.query.attributes.items()
            )
        )

    print()
    print(f"Objects Found: {len(final_objects)}")
    print(f"Confidence: {final_confidence:.2f}")
    print(f"Detector: {state.strategy.detector}")

    print(
        "Verification: "
        + (
            "CLIP"
            if state.strategy.enable_clip_verification
            else "None"
        )
    )

    print(
        f"Output Image: "
        f"{state.detection.output_image_path or DETECTION_OUTPUT_PATH}"
    )

    print("--------------------------------------------------")

    print("\n==============================")
    print(" CURRENT AGENT STATE")
    print("==============================")

    print(state.model_dump_json(indent=4))


if __name__ == "__main__":
    main()