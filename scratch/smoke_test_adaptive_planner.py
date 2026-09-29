from v2.schemas.state import AgentStateV2, ActionType, MediaType, Candidate
from v2.agents.planning_agent import PlanningAgentV2

def run_smoke_test():
    print("--- ADAPTIVE PLANNER SMOKE TEST ---")
    planner = PlanningAgentV2()
    
    # 1. Known target -> Specialist
    state = AgentStateV2()
    state.query_spec.target = "person"
    action = planner.run(state)
    print(f"Initial known target action: {action}")
    
    # 2. 0 candidates, large image -> SAHI
    state.plan.history.append(action)
    state.media_metadata.resolution = [3840, 2160]
    action = planner.run(state)
    print(f"0 candidates on 4K image action: {action}")
    
    # 3. tiny candidate -> SR
    state = AgentStateV2()
    state.query_spec.target = "person"
    action = planner.run(state)
    state.plan.history.append(action)
    state.candidates.append(Candidate(id="1", bbox=[0, 0, 10, 10], confidence=0.25))
    action = planner.run(state)
    print(f"Weak/tiny candidate action: {action}")
    
    print("Smoke test completed.")

if __name__ == "__main__":
    run_smoke_test()
