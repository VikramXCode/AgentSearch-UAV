import json

from workflows.state import AgentState


class KnowledgeAgent:

    def run(self, state: AgentState) -> AgentState:

        print("\n==============================")
        print("    KNOWLEDGE AGENT")
        print("==============================")

        try:

            with open("memory/search_history.json", "r") as f:
                history = json.load(f)

        except:

            history = []

        state.knowledge.previous_searches = history

        print(f"Loaded {len(history)} previous searches.")

        return state