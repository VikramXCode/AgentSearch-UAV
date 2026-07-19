from workflows.state import AgentState
from utils.search_utils import extract_query_components


class QueryAgent:

    def run(self, state: AgentState) -> AgentState:

        print("\n==============================")
        print("      QUERY AGENT")
        print("==============================")

        user_query = input("Enter your search query: ")

        state.query.raw_query = user_query

        components = extract_query_components(user_query)

        state.query.attributes = {}

        if components.color:
            state.query.attributes["color"] = components.color

        if components.size:
            state.query.attributes["size"] = components.size

        state.query.quantity = components.quantity
        state.query.target = components.target

        state.query.search_mode = "text"

        print("\nParsed Query")
        print(state.query.model_dump())

        return state