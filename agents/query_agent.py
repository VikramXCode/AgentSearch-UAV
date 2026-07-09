from workflows.state import AgentState


class QueryAgent:

    def run(self, state: AgentState) -> AgentState:

        print("\n==============================")
        print("      QUERY AGENT")
        print("==============================")

        user_query = input("Enter your search query: ")

        state.query.raw_query = user_query

        # Simple parser (we'll replace this with an LLM later)
        words = user_query.lower().split()

        colors = [
            "red",
            "blue",
            "green",
            "white",
            "black",
            "yellow",
            "orange",
            "gray",
            "grey",
            "brown"
        ]

        detected_color = None

        for word in words:
            if word in colors:
                detected_color = word
                break

        state.query.attributes = {}

        if detected_color:
            state.query.attributes["color"] = detected_color

        if len(words) > 0:
            state.query.target = words[-1]

        state.query.search_mode = "text"

        print("\nParsed Query")
        print(state.query.model_dump())

        return state