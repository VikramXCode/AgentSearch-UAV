import pytest
from v2.schemas.state import AgentStateV2, QuerySpec, Plan, ActionType, DetectorRouting, ToolUsage
from v2.agents.planning_agent import PlanningAgentV2
from v2.agents.query_agent import QueryAgentV2

def test_query_parsing():
    agent = QueryAgentV2()
    
    # Known category
    q1 = agent.run("car")
    assert q1.target == "car"
    
    # Open-world query with constraints
    q2 = agent.run("find a red drone")
    # QueryAgentV2 fallback to last word "drone" if not in hardcoded person/car/motorcycle list
    assert q2.target == "drone"
    assert len(q2.constraints) == 1
    assert q2.constraints[0].value == "red"

def test_planner_routing_known_vs_open_world():
    planner = PlanningAgentV2()
    
    # Case 1: Known Target
    state_known = AgentStateV2(
        query_spec=QuerySpec(target="car"),
        plan=Plan(),
        detector_routing=DetectorRouting(),
        tool_usage=ToolUsage()
    )
    action_known = planner.run(state_known)
    assert action_known == ActionType.DETECT_SPECIALIST
    assert "known. Routing to specialist" in state_known.plan.decision_rationale
    
    # Case 2: Open-world Target
    state_open = AgentStateV2(
        query_spec=QuerySpec(target="drone"),
        plan=Plan(),
        detector_routing=DetectorRouting(),
        tool_usage=ToolUsage()
    )
    action_open = planner.run(state_open)
    assert action_open == ActionType.DETECT_OPEN_WORLD
    assert "unknown" in state_open.plan.decision_rationale
