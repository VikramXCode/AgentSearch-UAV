from workflows.state import AgentState

from agents.query_agent import QueryAgent
from agents.knowledge_agent import KnowledgeAgent
from agents.strategy_agent import StrategyAgent

def main():

    state = AgentState()

    query_agent = QueryAgent()
    knowledge_agent = KnowledgeAgent()
    strategy_agent = StrategyAgent()

    state = query_agent.run(state)
    state = knowledge_agent.run(state)
    state = strategy_agent.run(state)

    print("\n==============================")
    print(" CURRENT AGENT STATE")
    print("==============================")

    print(state.model_dump_json(indent=4))


if __name__ == "__main__":
    main()