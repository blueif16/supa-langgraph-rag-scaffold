#!/usr/bin/env python3
"""
RAG Debug CLI
=============

Command-line interface for debugging and visualizing your RAG system.

Usage:
    # Visualize the knowledge graph
    python -m app.debug.cli visualize --namespace my_knowledge --output graph.html
    
    # Debug a search query
    python -m app.debug.cli debug "energetic fast cuts" --namespace my_knowledge
    
    # Run robustness tests
    python -m app.debug.cli test --namespace my_knowledge
    
    # Analyze content coverage
    python -m app.debug.cli analyze --namespace my_knowledge
    
    # Evaluate with test cases
    python -m app.debug.cli evaluate --namespace my_knowledge --cases test_cases.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import click
except ImportError:
    print("CLI requires click. Install with: pip install click")
    sys.exit(1)

from app.core import RAGStore
from app.debug import (
    RAGVisualizer,
    RAGEvaluator,
    RetrievalDebugger,
    RAGRobustnessTests,
)


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """RAG Debug Toolkit - Visualize, Debug, and Test your RAG system."""
    pass


@cli.command()
@click.option("--namespace", "-n", default="default", help="RAG namespace")
@click.option("--output", "-o", default="rag_graph.html", help="Output HTML file")
@click.option("--limit", "-l", default=200, help="Max documents to visualize")
def visualize(namespace: str, output: str, limit: int):
    """Generate interactive graph visualization."""
    click.echo(f"🔄 Loading RAG store for namespace: {namespace}")
    
    try:
        rag = RAGStore(namespace=namespace)
        viz = RAGVisualizer(rag)
        
        click.echo(f"🎨 Generating visualization...")
        result = viz.visualize(output, limit=limit)
        
        click.echo(f"✅ Generated {output}")
        click.echo(f"   Nodes: {result['nodes']}")
        click.echo(f"   Edges: {result['edges']}")
        click.echo(f"\n   Open in browser: file://{Path(output).absolute()}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("query")
@click.option("--namespace", "-n", default="default", help="RAG namespace")
@click.option("--output", "-o", default=None, help="Save search path visualization")
def debug(query: str, namespace: str, output: str | None):
    """Debug a search query - compare BM25, Vector, and Hybrid."""
    click.echo(f"🔍 Debugging query: {query}")
    click.echo(f"   Namespace: {namespace}\n")
    
    try:
        rag = RAGStore(namespace=namespace)
        debugger = RetrievalDebugger(rag)
        
        result = debugger.debug_search(query)
        debugger.print_debug(result)
        
        # Optionally save visualization
        if output:
            viz = RAGVisualizer(rag)
            viz.visualize_search_path(query, output)
            click.echo(f"📊 Search path saved to: {output}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--namespace", "-n", default="default", help="RAG namespace")
@click.option("--output", "-o", default=None, help="Save results to JSON file")
def test(namespace: str, output: str | None):
    """Run robustness tests on the RAG system."""
    click.echo(f"🧪 Running robustness tests for namespace: {namespace}\n")
    
    try:
        rag = RAGStore(namespace=namespace)
        tests = RAGRobustnessTests(rag)
        
        suite = tests.run_all()
        tests.print_results(suite)
        
        if output:
            with open(output, "w") as f:
                json.dump(suite.to_dict(), f, indent=2)
            click.echo(f"📄 Results saved to: {output}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--namespace", "-n", default="default", help="RAG namespace")
@click.option("--sample", "-s", default=100, help="Sample size for analysis")
def analyze(namespace: str, sample: int):
    """Analyze content coverage and retrieval characteristics."""
    click.echo(f"📊 Analyzing namespace: {namespace}\n")
    
    try:
        rag = RAGStore(namespace=namespace)
        debugger = RetrievalDebugger(rag)
        
        analysis = debugger.analyze_content_coverage(sample_size=sample)
        
        if "error" in analysis:
            click.echo(f"⚠️ {analysis['error']}")
            return
        
        click.echo(f"Documents analyzed: {analysis['document_count']}")
        click.echo(f"\nContent Length:")
        click.echo(f"  Min: {analysis['content_length']['min']} chars")
        click.echo(f"  Max: {analysis['content_length']['max']} chars")
        click.echo(f"  Avg: {analysis['content_length']['avg']:.0f} chars")
        
        click.echo(f"\nWord Count:")
        click.echo(f"  Min: {analysis['word_count']['min']} words")
        click.echo(f"  Max: {analysis['word_count']['max']} words")
        click.echo(f"  Avg: {analysis['word_count']['avg']:.0f} words")
        
        click.echo(f"\nTop 10 Words:")
        for word, count in analysis['top_words'][:10]:
            click.echo(f"  {word}: {count}")
        
        click.echo(f"\n📋 Recommendations:")
        for rec in analysis['recommendations']:
            click.echo(f"  {rec}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--namespace", "-n", default="default", help="RAG namespace")
@click.option("--cases", "-c", default=None, help="JSON file with test cases")
@click.option("--generate", "-g", default=0, help="Generate N synthetic test cases")
@click.option("--output", "-o", default=None, help="Save results to JSON file")
def evaluate(namespace: str, cases: str | None, generate: int, output: str | None):
    """Evaluate RAG quality with test cases."""
    click.echo(f"📝 Evaluating namespace: {namespace}\n")
    
    try:
        rag = RAGStore(namespace=namespace)
        evaluator = RAGEvaluator(rag)
        
        # Load or generate test cases
        if cases:
            with open(cases) as f:
                test_cases = json.load(f)
            click.echo(f"Loaded {len(test_cases)} test cases from {cases}")
        elif generate > 0:
            test_cases = evaluator.generate_test_cases(num_cases=generate)
            click.echo(f"Generated {len(test_cases)} synthetic test cases")
        else:
            click.echo("⚠️ Provide --cases file or --generate N")
            return
        
        if not test_cases:
            click.echo("⚠️ No test cases to evaluate")
            return
        
        # Run evaluation
        results = evaluator.evaluate(test_cases)
        evaluator.print_summary(results)
        
        if output:
            evaluator.save_results(results, output)
            click.echo(f"📄 Results saved to: {output}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--namespace", "-n", default="default", help="RAG namespace")
@click.argument("queries", nargs=-1)
def compare(namespace: str, queries: tuple[str]):
    """Compare results from multiple queries."""
    if not queries:
        click.echo("⚠️ Provide at least one query")
        return
    
    click.echo(f"🔄 Comparing {len(queries)} queries in namespace: {namespace}\n")
    
    try:
        rag = RAGStore(namespace=namespace)
        debugger = RetrievalDebugger(rag)
        
        result = debugger.compare_queries(list(queries))
        
        click.echo("Results:")
        click.echo("-" * 60)
        for r in result['results']:
            click.echo(f"\n📌 Query: {r['query']}")
            click.echo(f"   BM25: {r['bm25_count']} | Vector: {r['vector_count']} | Overlap: {r['overlap']}")
        
        click.echo("\n" + "-" * 60)
        click.echo("Summary:")
        for key, value in result['summary'].items():
            click.echo(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--namespace", "-n", default="default", help="RAG namespace")
def stats(namespace: str):
    """Show RAG store statistics."""
    try:
        rag = RAGStore(namespace=namespace)
        stats = rag.stats()

        click.echo(f"\n📊 RAG Store Stats - {namespace}")
        click.echo("-" * 40)
        click.echo(f"  Documents: {stats['documents']}")
        click.echo(f"  Relations: {stats['relations']}")
        click.echo("-" * 40 + "\n")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def namespaces():
    """List all available namespaces with statistics."""
    try:
        click.echo(f"\n{'='*60}")
        click.echo(f"Available Namespaces")
        click.echo(f"{'='*60}\n")

        # Get all unique namespaces from documents table
        rag = RAGStore()
        result = rag.client.table("documents").select("namespace").execute()

        if not result.data:
            click.echo("No namespaces found.")
            return

        # Count documents and relations per namespace
        namespaces = {}
        for row in result.data:
            ns = row["namespace"]
            if ns not in namespaces:
                namespaces[ns] = {"namespace": ns, "documents": 0, "relations": 0}
            namespaces[ns]["documents"] += 1

        # Get relations count
        rel_result = rag.client.table("doc_relations").select("namespace").execute()
        for row in rel_result.data:
            ns = row["namespace"]
            if ns in namespaces:
                namespaces[ns]["relations"] += 1

        # Display
        namespace_list = sorted(namespaces.values(), key=lambda x: x["namespace"])
        for ns_info in namespace_list:
            click.echo(f"  {ns_info['namespace']}")
            click.echo(f"    Documents: {ns_info['documents']}")
            click.echo(f"    Relations: {ns_info['relations']}")
            click.echo()

        click.echo(f"Total: {len(namespace_list)} namespace(s)")
        click.echo(f"{'='*60}\n")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
