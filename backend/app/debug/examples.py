#!/usr/bin/env python3
"""
RAG Debug Toolkit Examples
==========================

Run these examples to learn how to debug your RAG system.

Usage:
    python -m app.debug.examples
"""

import os
import sys

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.core import RAGStore
from app.debug import (
    RAGVisualizer,
    RAGEvaluator,
    RetrievalDebugger,
    RAGRobustnessTests,
)


def example_visualize():
    """Example: Visualize your knowledge graph."""
    print("\n" + "=" * 60)
    print("📊 Example: Graph Visualization")
    print("=" * 60)
    
    rag = RAGStore(namespace="default")
    viz = RAGVisualizer(rag)
    
    # Full graph
    result = viz.visualize("full_graph.html")
    print(f"Generated: {result['output']}")
    print(f"  Nodes: {result['nodes']}")
    print(f"  Edges: {result['edges']}")
    
    # Search path
    query = "fast editing"
    result = viz.visualize_search_path(query, "search_path.html")
    print(f"\nSearch path for '{query}':")
    print(f"  Results: {len(result['results'])}")
    print(f"  Output: {result['output']}")


def example_debug_search():
    """Example: Debug a search query."""
    print("\n" + "=" * 60)
    print("🔍 Example: Debug Search")
    print("=" * 60)
    
    rag = RAGStore(namespace="default")
    debugger = RetrievalDebugger(rag)
    
    # Debug a query
    query = "fast energetic style"
    result = debugger.debug_search(query)
    
    # Print detailed output
    debugger.print_debug(result)
    
    # Access programmatically
    print("\nProgrammatic access:")
    print(f"  BM25 found: {result.bm25.count} docs")
    print(f"  Vector found: {result.vector.count} docs")
    print(f"  Overlap: {len(result.overlap_bm25_vector)} docs")
    print(f"  Graph additions: {len(result.graph_additions)} docs")


def example_evaluate():
    """Example: Evaluate RAG quality."""
    print("\n" + "=" * 60)
    print("📝 Example: RAG Evaluation")
    print("=" * 60)
    
    rag = RAGStore(namespace="default")
    evaluator = RAGEvaluator(rag)
    
    # Define test cases
    test_cases = [
        {
            "question": "What is fast editing?",
            "expected_answer": "Quick cuts and energetic pacing",
        },
        {
            "question": "How to create a hook?",
            "expected_answer": "Start with attention-grabbing content",
        },
    ]
    
    # Run evaluation (without RAGAS for simplicity)
    results = evaluator.evaluate(test_cases, use_ragas=False)
    
    # Print summary
    evaluator.print_summary(results)
    
    # Save results
    evaluator.save_results(results, "evaluation_results.json")
    print("Results saved to: evaluation_results.json")


def example_robustness():
    """Example: Run robustness tests."""
    print("\n" + "=" * 60)
    print("🧪 Example: Robustness Tests")
    print("=" * 60)
    
    rag = RAGStore(namespace="default")
    tests = RAGRobustnessTests(rag)
    
    # Configure domain-specific tests (optional)
    tests.configure_tests(
        synonym_pairs=[
            ("fast", "quick"),
            ("editing", "cutting"),
        ],
        typo_pairs=[
            ("video", "vdieo"),
            ("style", "stlye"),
        ],
    )
    
    # Run all tests
    suite = tests.run_all()
    
    # Print results
    tests.print_results(suite)
    
    # Access programmatically
    print("\nProgrammatic access:")
    print(f"  Pass rate: {suite.pass_rate:.1%}")
    print(f"  Passed: {suite.passed}/{suite.total}")


def example_content_analysis():
    """Example: Analyze content coverage."""
    print("\n" + "=" * 60)
    print("📈 Example: Content Analysis")
    print("=" * 60)
    
    rag = RAGStore(namespace="default")
    debugger = RetrievalDebugger(rag)
    
    analysis = debugger.analyze_content_coverage(sample_size=50)
    
    if "error" in analysis:
        print(f"⚠️ {analysis['error']}")
        return
    
    print(f"Documents analyzed: {analysis['document_count']}")
    print(f"\nContent stats:")
    print(f"  Avg length: {analysis['content_length']['avg']:.0f} chars")
    print(f"  Avg words: {analysis['word_count']['avg']:.0f} words")
    
    print(f"\nTop 5 words:")
    for word, count in analysis['top_words'][:5]:
        print(f"  {word}: {count}")
    
    print(f"\nRecommendations:")
    for rec in analysis['recommendations']:
        print(f"  {rec}")


def example_compare_queries():
    """Example: Compare multiple queries."""
    print("\n" + "=" * 60)
    print("⚖️ Example: Compare Queries")
    print("=" * 60)
    
    rag = RAGStore(namespace="default")
    debugger = RetrievalDebugger(rag)
    
    queries = [
        "fast editing",
        "quick cuts",
        "energetic pacing",
    ]
    
    result = debugger.compare_queries(queries)
    
    print(f"Comparing {len(queries)} queries:\n")
    for r in result['results']:
        print(f"  '{r['query']}'")
        print(f"    BM25: {r['bm25_count']} | Vector: {r['vector_count']} | Overlap: {r['overlap']}")
    
    print(f"\nSummary:")
    for key, value in result['summary'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")


def main():
    """Run all examples."""
    print("\n🚀 RAG Debug Toolkit Examples")
    print("=" * 60)
    print("These examples demonstrate the debug toolkit capabilities.")
    print("Make sure you have documents in your RAG store first.")
    print("=" * 60)
    
    try:
        # Uncomment the examples you want to run
        
        # example_visualize()
        example_debug_search()
        # example_evaluate()
        example_robustness()
        # example_content_analysis()
        # example_compare_queries()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("  1. Your .env file has SUPABASE_URL and SUPABASE_KEY")
        print("  2. You've run the SQL migrations")
        print("  3. You have documents in your RAG store")


if __name__ == "__main__":
    main()
