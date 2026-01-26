"""
LangGraph Conditional Edges - Decision Logic
=============================================
"""
from app.graph.state import AgentState
from app.config import config


def decide_to_generate(state: AgentState) -> str:
    """
    Conditional edge: decide whether to generate or retry.
    
    Returns:
        "generate" - if context is relevant OR max retries reached
        "transform_query" - if context needs improvement and retries remain
    """
    if state["grade"] == "yes":
        print("---DECISION: CONTEXT RELEVANT, GENERATING---")
        return "generate"
    
    if state["retry_count"] >= config.MAX_RETRY:
        print(f"---DECISION: MAX RETRIES ({config.MAX_RETRY}) REACHED, GENERATING ANYWAY---")
        return "generate"
    
    print("---DECISION: CONTEXT NOT RELEVANT, REWRITING QUERY---")
    return "transform_query"
