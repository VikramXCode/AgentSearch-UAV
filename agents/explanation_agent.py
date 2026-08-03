from workflows.state import AgentState
from utils.search_utils import canonicalize_target


class ExplanationAgent:

    def run(self, state: AgentState) -> AgentState:

        print("\n==============================")
        print("   EXPLANATION AGENT")
        print("==============================")

        if state.strategy.enable_clip_verification:
            detections = state.verification.verified_objects
        else:
            detections = state.detection.objects_found

        target = canonicalize_target(state.query.target)
        color = state.query.attributes.get("color", "").strip().lower()
        size = state.query.attributes.get("size", "").strip().lower()

        if len(detections) == 0:
            state.explanation.summary = self._build_negative_summary(target, color, size)
            state.explanation.reasoning = [
                self._build_negative_reason(target, color, size),
            ]
            return state

        counts = {}
        confidences = []

        for detection in detections:
            label = detection["label"]
            confidence = float(detection["confidence"])

            counts[label] = counts.get(label, 0) + 1
            confidences.append(confidence)

        parts = []
        for label, count in sorted(counts.items()):
            noun = label if count == 1 else f"{label}s"
            parts.append(f"{count} {noun}")

        average_confidence = sum(confidences) / len(confidences)

        state.explanation.summary = self._build_positive_summary(counts, target, color, size)
        state.explanation.reasoning = [
            f"Target searched: {state.query.target}",
            f"Attributes: {state.query.attributes or {}}",
            f"Average confidence: {average_confidence:.3f}",
        ]

        return state

    @staticmethod
    def _build_positive_summary(counts: dict, target: str, color: str, size: str) -> str:

        label = target
        count = sum(counts.values())
        prefix = " ".join(part for part in [color, size, label] if part).strip()

        if count == 1:
            return f"Detected 1 {prefix}." if prefix else f"Detected 1 {label}."

        plural_label = label if label.endswith("s") else f"{label}s"

        if color or size:
            prefix_plural = " ".join(part for part in [color, size, plural_label] if part).strip()
            return f"Detected {count} {prefix_plural}."

        return f"Detected {count} {plural_label}."

    @staticmethod
    def _build_negative_reason(target: str, color: str, size: str) -> str:

        phrase = " ".join(part for part in [color, size, target] if part).strip()

        if phrase:
            return f"No {phrase} found in the final filtered detections."

        return "The requested target was not present in the final filtered detections."

    @staticmethod
    def _build_negative_summary(target: str, color: str, size: str) -> str:

        phrase = " ".join(part for part in [color, size, target] if part).strip()

        if phrase:
            return f"No {phrase} found."

        return "No matching objects detected."
