import json
import os
from datetime import datetime, timezone

from workflows.state import AgentState


class KnowledgeAgent:

    HISTORY_PATH = os.path.join("memory", "search_history.json")

    def run(self, state: AgentState) -> AgentState:

        print("\n==============================")
        print("    KNOWLEDGE AGENT")
        print("==============================")

        history = self._load_history()

        state.knowledge.previous_searches = history

        print(f"Loaded {len(history)} previous searches.")

        return state

    def record_search(self, state: AgentState) -> None:

        history = self._load_history()

        new_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "query": state.query.raw_query,
            "target": state.query.target,
            "attributes": state.query.attributes,
            "quantity": state.query.quantity,
            "search_mode": state.query.search_mode,
            "detector": state.strategy.detector,
            "use_sahi": state.strategy.enable_sahi,
            "use_super_resolution": state.strategy.enable_super_resolution,
            "verification_enabled": state.strategy.enable_clip_verification,
            "objects_found": len(state.detection.objects_found),
            "confidence_score": state.verification.confidence_score,
            "image_path": state.detection.image_path,
            "processed_image_path": state.detection.processed_image_path,
            "output_image_path": state.detection.output_image_path,
            "status": state.mission.status,
        }

        if not self._is_duplicate_entry(history, new_entry):
            history.append(new_entry)

        os.makedirs(os.path.dirname(self.HISTORY_PATH), exist_ok=True)

        with open(self.HISTORY_PATH, "w", encoding="utf-8") as file_handle:
            json.dump(history, file_handle, indent=2)

        state.knowledge.previous_searches = history

    def _load_history(self) -> list:

        try:
            with open(self.HISTORY_PATH, "r", encoding="utf-8") as file_handle:
                return json.load(file_handle)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    @staticmethod
    def _is_duplicate_entry(history: list, new_entry: dict) -> bool:

        if len(history) == 0:
            return False

        latest = history[-1].copy()
        latest.pop("timestamp", None)

        comparable_new = new_entry.copy()
        comparable_new.pop("timestamp", None)

        return latest == comparable_new