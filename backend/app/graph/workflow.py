"""
LangGraph Workflow - Self-Correcting RAG
========================================

retrieve → grade → (rewrite if poor) → generate
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from langgraph.graph import StateGraph, END

from app.graph.state import AgentState
from app.graph.edges import decide_to_generate
from app.config import config

if TYPE_CHECKING:
    from app.core import RAGStore


def create_workflow(rag: "RAGStore", max_retries: int | None = None):
    """
    Create self-correcting RAG workflow from a RAGStore.
    
    Usage:
        rag = RAGStore(namespace="my_kb")
        workflow = create_workflow(rag)
        result = workflow.invoke({"question": "..."})
    """
    from app.graph.nodes import create_nodes
    
    nodes = create_nodes(rag)
    _max = max_retries or config.MAX_RETRY
    
    def should_generate(state):
        if state["grade"] == "yes" or state["retry_count"] >= _max:
            return "generate"
        return "rewrite"
    
    wf = StateGraph(AgentState)
    wf.add_node("retrieve", nodes["retrieve"])
    wf.add_node("grade", nodes["grade"])
    wf.add_node("rewrite", nodes["rewrite"])
    wf.add_node("generate", nodes["generate"])
    
    wf.set_entry_point("retrieve")
    wf.add_edge("retrieve", "grade")
    wf.add_conditional_edges("grade", should_generate, {"rewrite": "rewrite", "generate": "generate"})
    wf.add_edge("rewrite", "retrieve")
    wf.add_edge("generate", END)
    
    return wf.compile()


# Default workflow using singleton config
try:
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg_pool import ConnectionPool
    
    pool = ConnectionPool(conninfo=config.DATABASE_URL, min_size=1, max_size=10)
    checkpointer = PostgresSaver(pool)
except Exception:
    checkpointer = None

from app.graph.nodes import retrieve_node, grade_documents_node, transform_query_node, generate_node

workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_documents_node)
workflow.add_node("transform_query", transform_query_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges("grade", decide_to_generate, {"transform_query": "transform_query", "generate": "generate"})
workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("generate", END)

app = workflow.compile(checkpointer=checkpointer)
