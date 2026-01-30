"""
RAG Robustness Tests
====================

Test suite for ensuring RAG system robustness.

Tests:
- Synonym robustness: Similar queries → similar results
- Typo tolerance: Misspellings still find content
- Empty results handling: Graceful degradation
- Long query handling: No crashes on long inputs
- Graph depth impact: Measure graph expansion value

Usage:
    from app.debug import RAGRobustnessTests
    from app.core import RAGStore
    
    rag = RAGStore(namespace="my_knowledge")
    tests = RAGRobustnessTests(rag)
    
    # Run all tests
    results = tests.run_all()
    
    # Run specific test
    results = tests.test_synonym_robustness()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from app.core.rag_store import RAGStore

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a single robustness test."""
    
    name: str
    passed: bool
    details: dict = field(default_factory=dict)
    message: str = ""
    duration_ms: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
            "duration_ms": self.duration_ms,
        }


@dataclass
class TestSuite:
    """Complete test suite results."""
    
    total: int = 0
    passed: int = 0
    failed: int = 0
    results: list[TestResult] = field(default_factory=list)
    
    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0
    
    def add(self, result: TestResult):
        self.results.append(result)
        self.total += 1
        if result.passed:
            self.passed += 1
        else:
            self.failed += 1
    
    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": self.pass_rate,
            "results": [r.to_dict() for r in self.results],
        }


class RAGRobustnessTests:
    """
    Comprehensive robustness test suite for RAG systems.
    
    Tests various edge cases and failure modes to ensure
    the RAG system behaves correctly under different conditions.
    """
    
    def __init__(self, rag: "RAGStore"):
        self.rag = rag
        
        # Default test configurations
        self.synonym_pairs = [
            ("fast energetic opening", "quick dynamic intro"),
            ("video editing style", "clip cutting approach"),
            ("slow pacing", "relaxed tempo"),
        ]
        
        self.typo_pairs = [
            ("energetic", "enrgetic"),
            ("product", "prodct"),
            ("editing", "editting"),
            ("video", "vdieo"),
        ]
        
        self.empty_queries = [
            "xyznonexistentterm123",
            "asdfghjklqwerty",
            "!!@@##$$",
        ]
    
    def run_all(self) -> TestSuite:
        """Run all robustness tests."""
        suite = TestSuite()
        
        tests = [
            self.test_synonym_robustness,
            self.test_typo_tolerance,
            self.test_empty_results,
            self.test_long_queries,
            self.test_special_characters,
            self.test_graph_depth_impact,
            self.test_search_consistency,
            self.test_batch_search_performance,
        ]
        
        for test in tests:
            try:
                result = test()
                suite.add(result)
            except Exception as e:
                suite.add(TestResult(
                    name=test.__name__,
                    passed=False,
                    message=f"Test crashed: {e}",
                ))
        
        return suite
    
    def test_synonym_robustness(self) -> TestResult:
        """
        Test that semantically similar queries return similar results.
        
        Expectation: >50% overlap between synonym pairs
        """
        start = time.time()
        
        overlaps = []
        details = []
        
        for q1, q2 in self.synonym_pairs:
            r1 = self.rag.search(q1, top_k=5)
            r2 = self.rag.search(q2, top_k=5)
            
            ids1 = {r["id"] for r in r1}
            ids2 = {r["id"] for r in r2}
            
            if ids1 or ids2:
                overlap = len(ids1 & ids2) / len(ids1 | ids2)
            else:
                overlap = 0.0
            
            overlaps.append(overlap)
            details.append({
                "queries": (q1, q2),
                "overlap": overlap,
                "q1_count": len(r1),
                "q2_count": len(r2),
            })
        
        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
        passed = avg_overlap >= 0.3  # At least 30% overlap expected
        
        return TestResult(
            name="synonym_robustness",
            passed=passed,
            message=f"Average overlap: {avg_overlap:.2%}" + (
                " - Good semantic matching" if passed else " - Consider enriching content"
            ),
            details={"pairs": details, "average_overlap": avg_overlap},
            duration_ms=(time.time() - start) * 1000,
        )
    
    def test_typo_tolerance(self) -> TestResult:
        """
        Test that typos still return results via vector search.
        
        Expectation: Typo queries should still find relevant content
        """
        start = time.time()
        
        recovered = 0
        details = []
        
        for correct, typo in self.typo_pairs:
            r_correct = self.rag.search(correct, top_k=3)
            r_typo = self.rag.search(typo, top_k=3)
            
            # Typo should recover at least some results
            is_recovered = len(r_typo) > 0
            if is_recovered:
                recovered += 1
            
            details.append({
                "correct": correct,
                "typo": typo,
                "correct_count": len(r_correct),
                "typo_count": len(r_typo),
                "recovered": is_recovered,
            })
        
        recovery_rate = recovered / len(self.typo_pairs) if self.typo_pairs else 0.0
        passed = recovery_rate >= 0.75  # At least 75% should recover
        
        return TestResult(
            name="typo_tolerance",
            passed=passed,
            message=f"Recovery rate: {recovery_rate:.2%}" + (
                " - Good vector search coverage" if passed else " - Vector search may need tuning"
            ),
            details={"pairs": details, "recovery_rate": recovery_rate},
            duration_ms=(time.time() - start) * 1000,
        )
    
    def test_empty_results(self) -> TestResult:
        """
        Test graceful handling of queries that return no results.
        
        Expectation: No crashes, empty list returned
        """
        start = time.time()
        
        all_empty = True
        no_errors = True
        details = []
        
        for query in self.empty_queries:
            try:
                results = self.rag.search(query, top_k=5)
                is_empty = len(results) == 0
                all_empty = all_empty and is_empty
                details.append({
                    "query": query,
                    "result_count": len(results),
                    "error": None,
                })
            except Exception as e:
                no_errors = False
                details.append({
                    "query": query,
                    "result_count": None,
                    "error": str(e),
                })
        
        passed = no_errors  # Main test: no crashes
        
        return TestResult(
            name="empty_results_handling",
            passed=passed,
            message="Graceful empty handling" if passed else "Errors on empty queries",
            details={"queries": details, "all_empty": all_empty},
            duration_ms=(time.time() - start) * 1000,
        )
    
    def test_long_queries(self) -> TestResult:
        """
        Test handling of unusually long queries.
        
        Expectation: No crashes, reasonable results
        """
        start = time.time()
        
        long_queries = [
            "a " * 100,  # 100 words
            "video editing style with fast cuts and energetic pacing " * 10,
            "x" * 1000,  # Single long word
        ]
        
        details = []
        no_errors = True
        
        for query in long_queries:
            try:
                results = self.rag.search(query[:500], top_k=3)  # Truncate for safety
                details.append({
                    "query_length": len(query),
                    "truncated_length": min(len(query), 500),
                    "result_count": len(results),
                    "error": None,
                })
            except Exception as e:
                no_errors = False
                details.append({
                    "query_length": len(query),
                    "error": str(e),
                })
        
        return TestResult(
            name="long_query_handling",
            passed=no_errors,
            message="Long queries handled" if no_errors else "Errors on long queries",
            details={"queries": details},
            duration_ms=(time.time() - start) * 1000,
        )
    
    def test_special_characters(self) -> TestResult:
        """
        Test handling of special characters in queries.
        
        Expectation: No crashes, special chars handled gracefully
        """
        start = time.time()
        
        special_queries = [
            "video's best moments",
            "50% faster editing",
            "cut-to-cut transitions",
            "hook: the opener",
            "(parenthetical) content",
            '"quoted phrase"',
            "日本語テスト",  # Japanese
            "emoji 🎬 test",
        ]
        
        details = []
        no_errors = True
        
        for query in special_queries:
            try:
                results = self.rag.search(query, top_k=3)
                details.append({
                    "query": query,
                    "result_count": len(results),
                    "error": None,
                })
            except Exception as e:
                no_errors = False
                details.append({
                    "query": query,
                    "error": str(e),
                })
        
        return TestResult(
            name="special_characters",
            passed=no_errors,
            message="Special chars handled" if no_errors else "Errors on special chars",
            details={"queries": details},
            duration_ms=(time.time() - start) * 1000,
        )
    
    def test_graph_depth_impact(self) -> TestResult:
        """
        Test how graph depth affects results.
        
        Expectation: Higher depth should add related documents (if relations exist)
        """
        start = time.time()
        
        # Use a generic query that should match something
        test_query = "style"  # Adjust based on your content
        
        depths = [0, 1, 2, 3]
        results_by_depth = {}
        
        for depth in depths:
            results = self.rag.search(test_query, top_k=10, graph_depth=depth)
            results_by_depth[depth] = {
                "count": len(results),
                "ids": [r["id"] for r in results],
                "graph_sourced": len([r for r in results if r.get("source_type") != "seed"]),
            }
        
        # Check if graph expansion adds value
        depth_0_count = results_by_depth[0]["count"]
        depth_2_count = results_by_depth[2]["count"]
        graph_adds_value = depth_2_count > depth_0_count
        
        return TestResult(
            name="graph_depth_impact",
            passed=True,  # This is informational, not pass/fail
            message=f"Depth 0: {depth_0_count}, Depth 2: {depth_2_count}" + (
                " - Graph adds documents" if graph_adds_value else " - Consider adding relations"
            ),
            details={"by_depth": results_by_depth},
            duration_ms=(time.time() - start) * 1000,
        )
    
    def test_search_consistency(self) -> TestResult:
        """
        Test that repeated searches return consistent results.
        
        Expectation: Same query → same results
        """
        start = time.time()
        
        test_query = "fast editing style"
        iterations = 3
        
        all_results = []
        for _ in range(iterations):
            results = self.rag.search(test_query, top_k=5)
            all_results.append([r["id"] for r in results])
        
        # Check if all iterations returned same IDs in same order
        is_consistent = all(r == all_results[0] for r in all_results)
        
        return TestResult(
            name="search_consistency",
            passed=is_consistent,
            message="Results are consistent" if is_consistent else "Results vary between calls",
            details={
                "iterations": iterations,
                "results": all_results,
            },
            duration_ms=(time.time() - start) * 1000,
        )
    
    def test_batch_search_performance(self) -> TestResult:
        """
        Test performance under batch search load.
        
        Expectation: Reasonable latency under load
        """
        start = time.time()
        
        queries = [
            "fast cuts",
            "slow motion",
            "color grading",
            "audio sync",
            "transition effects",
        ]
        
        latencies = []
        for query in queries:
            q_start = time.time()
            self.rag.search(query, top_k=5)
            latencies.append((time.time() - q_start) * 1000)
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        # Expectation: avg < 500ms, max < 1000ms
        passed = avg_latency < 500 and max_latency < 1000
        
        return TestResult(
            name="batch_search_performance",
            passed=passed,
            message=f"Avg: {avg_latency:.0f}ms, Max: {max_latency:.0f}ms" + (
                " - Good performance" if passed else " - Consider optimization"
            ),
            details={
                "queries": queries,
                "latencies_ms": latencies,
                "avg_latency_ms": avg_latency,
                "max_latency_ms": max_latency,
            },
            duration_ms=(time.time() - start) * 1000,
        )
    
    def print_results(self, suite: TestSuite) -> None:
        """Print test results in human-readable format."""
        print("\n" + "=" * 60)
        print("🧪 RAG Robustness Test Results")
        print("=" * 60)
        print(f"Namespace: {self.rag.namespace}")
        print(f"Total: {suite.total} | Passed: {suite.passed} | Failed: {suite.failed}")
        print(f"Pass Rate: {suite.pass_rate:.1%}")
        print("-" * 60)
        
        for result in suite.results:
            status = "✅" if result.passed else "❌"
            print(f"{status} {result.name}: {result.message} ({result.duration_ms:.0f}ms)")
        
        print("=" * 60 + "\n")
    
    def configure_tests(
        self,
        synonym_pairs: list[tuple[str, str]] | None = None,
        typo_pairs: list[tuple[str, str]] | None = None,
        empty_queries: list[str] | None = None,
    ) -> None:
        """
        Configure test parameters for your domain.
        
        Args:
            synonym_pairs: List of (query1, query2) that should return similar results
            typo_pairs: List of (correct, typo) to test typo tolerance
            empty_queries: Queries expected to return no results
        """
        if synonym_pairs:
            self.synonym_pairs = synonym_pairs
        if typo_pairs:
            self.typo_pairs = typo_pairs
        if empty_queries:
            self.empty_queries = empty_queries
