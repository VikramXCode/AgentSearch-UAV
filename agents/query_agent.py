from workflows.state import AgentState
from utils.search_utils import extract_query_components


class QueryAgent:

    def run(
        self,
        state: AgentState,
        query: str | None = None,
    ) -> AgentState:

        print("\n==============================")
        print("      QUERY AGENT")
        print("==============================")

        # CLI fallback
        if query is None:
            query = input("Enter your search query: ").strip()

        if not query:
            raise ValueError("Search query is required.")

        state.query.raw_query = query

        components = extract_query_components(query)

        state.query.attributes = {}

        if components.color:
            state.query.attributes["color"] = components.color

        if components.size:
            state.query.attributes["size"] = components.size

        state.query.quantity = components.quantity
        state.query.quantity_value = components.quantity_value
        state.query.target = components.target
        state.query.original_target = components.original_target
        state.query.canonical_query = components.canonical_query
        state.query.search_mode = "text"

        print("\nParsed Query")
        print(state.query.model_dump())

        return state