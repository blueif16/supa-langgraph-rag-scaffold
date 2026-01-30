"""
Retrieval Debugger
==================

Debug retrieval issues by comparing BM25, Vector, and Hybrid search.

Helps answer:
- Is BM25 finding keyword matches?
- Is Vector search finding semantic matches?
- Is RRF fusion working correctly?
- Is graph expansion adding value or noise?

Usage:
    from app.debug import RetrievalDebugger
    from app.core import RAGStore
    
    rag = RAGStore(namespace="my_knowledge")
    debugger = RetrievalDebugger(rag)
    
    # Debug a specific query
    result = debugger.debug_search("energetic fast cuts")
    print(result["diagnosis"])
    
    # Compare multiple queries
    debugger.compare_queries(["fast cuts", "energetic opening", "quick editing"])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from collections import Counter

if TYPE_CHECKING:
    from app.core.rag_store import RAGStore

logger = logging.getLogger(__name__)


@dataclass
class SearchMethodResult:
    """Results from a single search method."""
    
    method: str
    results: list[dict]
    latency_ms: float = 0.0
    
    @property
    def ids(self) -> set[int]:
        return {r["id"] for r in self.results}
    
    @property
    def count(self) -> int:
        return len(self.results)
    
    def preview(self, max_items: int = 5, max_len: int = 80) -> list[dict]:
        """Get preview of results for display."""
        return [
            {
                "id": r["id"],
                "preview": r["content"][:max_len] + "..." if len(r["content"]) > max_len else r["content"],
                "score": r.get("score") or r.get("similarity") or r.get("rank"),
            }
            for r in self.results[:max_items]
        ]


@dataclass
class DebugResult:
    """Complete debug result comparing all search methods."""
    
    query: str
    bm25: SearchMethodResult
    vector: SearchMethodResult
    hybrid_rrf: SearchMethodResult
    hybrid_graph: SearchMethodResult
    diagnosis: list[str] = field(default_factory=list)
    
    @property
    def overlap_bm25_vector(self) -> set[int]:
        """Documents found by both BM25 and Vector."""
        return self.bm25.ids & self.vector.ids
    
    @property
    def only_bm25(self) -> set[int]:
        """Documents found only by BM25."""
        return self.bm25.ids - self.vector.ids
    
    @property
    def only_vector(self) -> set[int]:
        """Documents found only by Vector."""
        return self.vector.ids - self.bm25.ids
    
    @property
    def graph_additions(self) -> set[int]:
        """Documents added by graph expansion."""
        return self.hybrid_graph.ids - self.hybrid_rrf.ids
    
    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "bm25": self.bm25.preview(),
            "vector": self.vector.preview(),
            "hybrid_rrf": self.hybrid_rrf.preview(),
            "hybrid_graph": self.hybrid_graph.preview(),
            "analysis": {
                "bm25_count": self.bm25.count,
                "vector_count": self.vector.count,
                "overlap_count": len(self.overlap_bm25_vector),
                "graph_additions": len(self.graph_additions),
            },
            "diagnosis": self.diagnosis,
        }


class RetrievalDebugger:
    """
    Debug retrieval by comparing different search methods.
    
    Isolates issues in:
    - BM25 (keyword matching)
    - Vector search (semantic matching)
    - RRF fusion (combining both)
    - Graph expansion (relationship traversal)
    """
    
    def __init__(self, rag: "RAGStore"):
        self.rag = rag
        self.client = rag.client
        self.namespace = rag.namespace
    
    def debug_search(self, query: str, top_k: int = 5) -> DebugResult:
        """
        Run a query through all search methods and compare results.
        
        Args:
            query: Search query
            top_k: Number of results per method
            
        Returns:
            DebugResult with comparison and diagnosis
        """
        import time
        
        embedding = self.rag._embeddings.embed_query(query)
        
        # 1. BM25 only
        start = time.time()
        bm25_results = self._search_bm25(query, top_k)
        bm25_latency = (time.time() - start) * 1000
        
        # 2. Vector only
        start = time.time()
        vector_results = self.rag.search_vector(query, top_k=top_k)
        vector_latency = (time.time() - start) * 1000
        
        # 3. Hybrid RRF (no graph)
        start = time.time()
        hybrid_results = self.rag.search(query, top_k=top_k, graph_depth=0)
        hybrid_latency = (time.time() - start) * 1000
        
        # 4. Hybrid + Graph
        start = time.time()
        full_results = self.rag.search(query, top_k=top_k, graph_depth=2)
        full_latency = (time.time() - start) * 1000
        
        result = DebugResult(
            query=query,
            bm25=SearchMethodResult("bm25", bm25_results, bm25_latency),
            vector=SearchMethodResult("vector", vector_results, vector_latency),
            hybrid_rrf=SearchMethodResult("hybrid_rrf", hybrid_results, hybrid_latency),
            hybrid_graph=SearchMethodResult("hybrid_graph", full_results, full_latency),
        )
        
        # Generate diagnosis
        result.diagnosis = self._diagnose(result)
        
        return result
    
    def _search_bm25(self, query: str, top_k: int) -> list[dict]:
        """BM25-only search using Supabase function."""
        try:
            result = self.client.rpc("debug_bm25_search", {
                "query_text": query,
                "match_count": top_k,
                "filter_namespace": self.namespace,
            }).execute()
            return result.data or []
        except Exception as e:
            # Fallback to direct FTS query
            logger.warning(f"debug_bm25_search function not found, using fallback: {e}")
            return self._bm25_fallback(query, top_k)
    
    def _bm25_fallback(self, query: str, top_k: int) -> list[dict]:
        """Fallback BM25 search using raw SQL."""
        try:
            # Use textSearch filter
            result = self.client.table("documents").select(
                "id, content, metadata"
            ).text_search(
                "content", query
            ).eq("namespace", self.namespace).limit(top_k).execute()
            return result.data or []
        except Exception as e:
            logger.error(f"BM25 fallback failed: {e}")
            return []
    
    def _diagnose(self, result: DebugResult) -> list[str]:
        """Generate diagnostic messages based on results."""
        diagnosis = []
        
        # Check BM25
        if result.bm25.count == 0:
            diagnosis.append(
                "⚠️ BM25 returned nothing - query terms may not exist in content. "
                "Try checking if your content contains the exact keywords."
            )
        elif result.bm25.count < 2:
            diagnosis.append(
                "🟡 BM25 found very few results - content may be sparse on these keywords."
            )
        
        # Check Vector
        if result.vector.count == 0:
            diagnosis.append(
                "⚠️ Vector search returned nothing - this is unusual. "
                "Check if embeddings are properly stored."
            )
        
        # Check overlap
        overlap = result.overlap_bm25_vector
        if len(overlap) == 0 and result.bm25.count > 0 and result.vector.count > 0:
            diagnosis.append(
                "🔴 Zero overlap between BM25 and Vector - content might be too sparse "
                "or not self-describing enough. Consider enriching content with keywords."
            )
        elif len(overlap) < 2 and result.bm25.count >= 2 and result.vector.count >= 2:
            diagnosis.append(
                "🟡 Low overlap between BM25 and Vector - RRF fusion may not be optimal. "
                "Check if content describes concepts in natural language."
            )
        elif len(overlap) >= 2:
            diagnosis.append(
                "✅ Good overlap between keyword and semantic search - RRF fusion is working well."
            )
        
        # Check graph expansion
        graph_adds = result.graph_additions
        if len(graph_adds) > 0:
            diagnosis.append(
                f"📊 Graph expansion added {len(graph_adds)} documents via relations."
            )
        else:
            diagnosis.append(
                "ℹ️ No additional documents from graph expansion - consider adding relations."
            )
        
        # Check only_bm25 vs only_vector
        if len(result.only_bm25) > 2:
            diagnosis.append(
                f"🔍 BM25 found {len(result.only_bm25)} docs that Vector missed - "
                "these likely contain exact query terms."
            )
        if len(result.only_vector) > 2:
            diagnosis.append(
                f"🔍 Vector found {len(result.only_vector)} docs that BM25 missed - "
                "these are semantically related but use different words."
            )
        
        return diagnosis
    
    def compare_queries(self, queries: list[str], top_k: int = 5) -> dict:
        """
        Compare retrieval behavior across multiple queries.
        
        Useful for understanding which queries work well and which don't.
        """
        results = []
        for query in queries:
            result = self.debug_search(query, top_k)
            results.append({
                "query": query,
                "bm25_count": result.bm25.count,
                "vector_count": result.vector.count,
                "overlap": len(result.overlap_bm25_vector),
                "graph_additions": len(result.graph_additions),
                "diagnosis_summary": result.diagnosis[0] if result.diagnosis else "No issues",
            })
        
        return {
            "queries": queries,
            "results": results,
            "summary": self._summarize_comparison(results),
        }
    
    def _summarize_comparison(self, results: list[dict]) -> dict:
        """Summarize comparison across queries."""
        return {
            "avg_bm25_count": sum(r["bm25_count"] for r in results) / len(results),
            "avg_vector_count": sum(r["vector_count"] for r in results) / len(results),
            "avg_overlap": sum(r["overlap"] for r in results) / len(results),
            "queries_with_zero_overlap": sum(1 for r in results if r["overlap"] == 0),
            "queries_with_graph_additions": sum(1 for r in results if r["graph_additions"] > 0),
        }
    
    def analyze_content_coverage(self, sample_size: int = 100) -> dict:
        """
        Analyze content to understand retrieval characteristics.
        
        Helps identify:
        - Content too short/sparse
        - Missing keywords
        - Embedding distribution
        """
        docs = self.client.table("documents").select(
            "id, content, metadata"
        ).eq("namespace", self.namespace).limit(sample_size).execute()
        
        if not docs.data:
            return {"error": "No documents found"}
        
        # Analyze content
        lengths = [len(d["content"]) for d in docs.data]
        word_counts = [len(d["content"].split()) for d in docs.data]
        
        # Common words (for BM25 insight)
        all_words = []
        for d in docs.data:
            all_words.extend(d["content"].lower().split())
        word_freq = Counter(all_words)
        
        return {
            "document_count": len(docs.data),
            "content_length": {
                "min": min(lengths),
                "max": max(lengths),
                "avg": sum(lengths) / len(lengths),
            },
            "word_count": {
                "min": min(word_counts),
                "max": max(word_counts),
                "avg": sum(word_counts) / len(word_counts),
            },
            "top_words": word_freq.most_common(20),
            "recommendations": self._content_recommendations(lengths, word_counts),
        }
    
    def _content_recommendations(self, lengths: list[int], word_counts: list[int]) -> list[str]:
        """Generate recommendations based on content analysis."""
        recommendations = []
        
        avg_len = sum(lengths) / len(lengths)
        avg_words = sum(word_counts) / len(word_counts)
        
        if avg_len < 200:
            recommendations.append(
                "⚠️ Average content length is very short (<200 chars). "
                "Consider adding more context to each document."
            )
        elif avg_len < 500:
            recommendations.append(
                "🟡 Content is relatively short. More context could improve retrieval."
            )
        else:
            recommendations.append("✅ Content length looks good.")
        
        if avg_words < 30:
            recommendations.append(
                "⚠️ Low word count - BM25 may struggle to find matches."
            )
        
        short_docs = sum(1 for l in lengths if l < 100)
        if short_docs > len(lengths) * 0.2:
            recommendations.append(
                f"⚠️ {short_docs} documents are very short (<100 chars). "
                "These may not be self-describing enough."
            )
        
        return recommendations
    
    def print_debug(self, result: DebugResult) -> None:
        """Print debug result in human-readable format."""
        print(f"\n{'='*60}")
        print(f"🔍 Query: {result.query}")
        print('='*60)
        
        for method_name in ["bm25", "vector", "hybrid_rrf", "hybrid_graph"]:
            method = getattr(result, method_name)
            print(f"\n--- {method.method.upper()} ({method.count} results, {method.latency_ms:.1f}ms) ---")
            for r in method.preview(3):
                score_str = f" [score: {r['score']:.3f}]" if r['score'] else ""
                print(f"  [{r['id']}]{score_str} {r['preview']}")
        
        print(f"\n{'='*60}")
        print("📊 Analysis:")
        print(f"  BM25 ∩ Vector overlap: {len(result.overlap_bm25_vector)}")
        print(f"  Only BM25: {len(result.only_bm25)}")
        print(f"  Only Vector: {len(result.only_vector)}")
        print(f"  Graph additions: {len(result.graph_additions)}")
        
        print(f"\n{'='*60}")
        print("🩺 Diagnosis:")
        for d in result.diagnosis:
            print(f"  {d}")
        print('='*60 + "\n")
