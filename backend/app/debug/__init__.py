"""
RAG Debug Toolkit
=================

Tools for visualizing, evaluating, and debugging your RAG system.

Usage:
    from app.debug import RAGVisualizer, RAGEvaluator, RetrievalDebugger, RAGRobustnessTests
    
    rag = RAGStore(namespace="my_knowledge")
    
    # Visualize the knowledge graph
    viz = RAGVisualizer(rag)
    viz.visualize("graph.html")
    
    # Debug a specific query
    debugger = RetrievalDebugger(rag)
    debugger.debug_search("my query")
    
    # Run evaluation
    evaluator = RAGEvaluator(rag)
    evaluator.evaluate(test_cases)
    
    # Run robustness tests
    tests = RAGRobustnessTests(rag)
    tests.run_all()
"""

from app.debug.visualizer import RAGVisualizer
from app.debug.evaluator import RAGEvaluator
from app.debug.retrieval_debugger import RetrievalDebugger
from app.debug.robustness import RAGRobustnessTests

__all__ = [
    "RAGVisualizer",
    "RAGEvaluator", 
    "RetrievalDebugger",
    "RAGRobustnessTests",
]
