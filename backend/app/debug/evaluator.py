"""
RAG Evaluator
=============

Systematic evaluation of RAG quality using RAGAS metrics.

Metrics:
- Context Precision: Are retrieved docs relevant to the question?
- Context Recall: Did we retrieve all relevant docs?
- Faithfulness: Is the answer grounded in the context?
- Answer Relevancy: Does the answer address the question?

Usage:
    from app.debug import RAGEvaluator
    from app.core import RAGStore
    
    rag = RAGStore(namespace="my_knowledge")
    evaluator = RAGEvaluator(rag)
    
    test_cases = [
        {
            "question": "What's the hook pacing for Product Hunt?",
            "expected_answer": "Fast 1.5-2s cuts in first 10 seconds",
        }
    ]
    
    results = evaluator.evaluate(test_cases)
    print(results)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable
import json

if TYPE_CHECKING:
    from app.core.rag_store import RAGStore

logger = logging.getLogger(__name__)


@dataclass
class EvalTestCase:
    """A single test case for RAG evaluation."""
    
    question: str
    expected_answer: str | None = None
    ground_truth_context: list[str] | None = None
    tags: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "expected_answer": self.expected_answer,
            "ground_truth_context": self.ground_truth_context,
            "tags": self.tags,
        }


@dataclass 
class EvalResult:
    """Result of evaluating a single test case."""
    
    question: str
    retrieved_contexts: list[str]
    expected_answer: str | None
    generated_answer: str | None
    
    # Scores (0-1)
    context_precision: float | None = None
    context_recall: float | None = None
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    
    # Debug info
    retrieval_ids: list[int] = field(default_factory=list)
    retrieval_scores: list[float] = field(default_factory=list)
    
    @property
    def overall_score(self) -> float:
        """Weighted average of all metrics."""
        scores = [
            s for s in [
                self.context_precision,
                self.context_recall, 
                self.faithfulness,
                self.answer_relevancy,
            ] if s is not None
        ]
        return sum(scores) / len(scores) if scores else 0.0
    
    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "expected_answer": self.expected_answer,
            "generated_answer": self.generated_answer,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "overall_score": self.overall_score,
            "retrieval_ids": self.retrieval_ids,
        }


class RAGEvaluator:
    """
    Evaluate RAG system quality with RAGAS metrics.
    
    Supports both:
    - Retrieval-only evaluation (no LLM)
    - Full RAG evaluation (with LLM generation)
    """
    
    def __init__(
        self,
        rag: "RAGStore",
        answer_generator: Callable[[str, list[str]], str] | None = None,
    ):
        """
        Args:
            rag: RAGStore instance to evaluate
            answer_generator: Optional function that takes (question, contexts) 
                            and returns generated answer. If not provided,
                            uses first context as answer (retrieval-only mode).
        """
        self.rag = rag
        self.answer_generator = answer_generator
        self._ragas_available = None
    
    def _check_ragas(self) -> bool:
        """Check if RAGAS is available."""
        if self._ragas_available is None:
            try:
                import ragas
                self._ragas_available = True
            except ImportError:
                self._ragas_available = False
                logger.warning(
                    "RAGAS not installed. Install with: pip install ragas\n"
                    "Using simplified metrics instead."
                )
        return self._ragas_available
    
    def evaluate(
        self,
        test_cases: list[dict | EvalTestCase],
        top_k: int = 5,
        use_ragas: bool = True,
    ) -> dict:
        """
        Evaluate RAG system on test cases.
        
        Args:
            test_cases: List of test cases (dicts or EvalTestCase objects)
            top_k: Number of documents to retrieve per query
            use_ragas: Whether to use RAGAS metrics (if available)
            
        Returns:
            Evaluation results with per-case and aggregate metrics
        """
        # Normalize test cases
        cases = []
        for tc in test_cases:
            if isinstance(tc, EvalTestCase):
                cases.append(tc)
            else:
                cases.append(EvalTestCase(**tc))
        
        results = []
        for case in cases:
            result = self._evaluate_single(case, top_k, use_ragas)
            results.append(result)
        
        # Aggregate metrics
        aggregate = self._compute_aggregate(results)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "namespace": self.rag.namespace,
            "num_cases": len(cases),
            "aggregate": aggregate,
            "results": [r.to_dict() for r in results],
        }
    
    def _evaluate_single(
        self,
        case: EvalTestCase,
        top_k: int,
        use_ragas: bool,
    ) -> EvalResult:
        """Evaluate a single test case."""
        # Retrieve contexts
        search_results = self.rag.search(case.question, top_k=top_k)
        
        contexts = [r["content"] for r in search_results]
        retrieval_ids = [r["id"] for r in search_results]
        retrieval_scores = [r.get("score", 0.0) for r in search_results]
        
        # Generate answer
        if self.answer_generator:
            generated_answer = self.answer_generator(case.question, contexts)
        else:
            # Retrieval-only mode: use first context
            generated_answer = contexts[0] if contexts else ""
        
        # Compute metrics
        if use_ragas and self._check_ragas():
            metrics = self._compute_ragas_metrics(
                question=case.question,
                contexts=contexts,
                answer=generated_answer,
                ground_truth=case.expected_answer,
            )
        else:
            metrics = self._compute_simple_metrics(
                question=case.question,
                contexts=contexts,
                answer=generated_answer,
                expected=case.expected_answer,
            )
        
        return EvalResult(
            question=case.question,
            retrieved_contexts=contexts,
            expected_answer=case.expected_answer,
            generated_answer=generated_answer,
            retrieval_ids=retrieval_ids,
            retrieval_scores=retrieval_scores,
            **metrics,
        )
    
    def _compute_ragas_metrics(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: str | None,
    ) -> dict:
        """Compute metrics using RAGAS library."""
        try:
            from ragas import evaluate
            from ragas.metrics import (
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            )
            from datasets import Dataset
            
            # Build dataset
            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }
            if ground_truth:
                data["ground_truth"] = [ground_truth]
            
            dataset = Dataset.from_dict(data)
            
            # Select metrics
            metrics = [context_precision, faithfulness, answer_relevancy]
            if ground_truth:
                metrics.append(context_recall)
            
            result = evaluate(dataset, metrics=metrics)
            
            return {
                "context_precision": result.get("context_precision"),
                "context_recall": result.get("context_recall"),
                "faithfulness": result.get("faithfulness"),
                "answer_relevancy": result.get("answer_relevancy"),
            }
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return self._compute_simple_metrics(question, contexts, answer, ground_truth)
    
    def _compute_simple_metrics(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        expected: str | None,
    ) -> dict:
        """
        Compute simplified metrics without RAGAS.
        
        These are approximations based on keyword overlap.
        """
        # Context precision: how much of retrieved context is relevant
        # (approximated by keyword overlap with question)
        question_words = set(question.lower().split())
        context_text = " ".join(contexts).lower()
        context_words = set(context_text.split())
        
        if context_words:
            precision = len(question_words & context_words) / len(question_words)
        else:
            precision = 0.0
        
        # Context recall: approximated by expected answer overlap
        recall = None
        if expected:
            expected_words = set(expected.lower().split())
            if expected_words:
                recall = len(expected_words & context_words) / len(expected_words)
        
        # Answer relevancy: keyword overlap between answer and question
        answer_words = set(answer.lower().split()) if answer else set()
        if answer_words:
            relevancy = len(question_words & answer_words) / len(question_words)
        else:
            relevancy = 0.0
        
        # Faithfulness: answer words should come from context
        if answer_words:
            faithfulness = len(answer_words & context_words) / len(answer_words)
        else:
            faithfulness = 0.0
        
        return {
            "context_precision": min(precision, 1.0),
            "context_recall": min(recall, 1.0) if recall else None,
            "faithfulness": min(faithfulness, 1.0),
            "answer_relevancy": min(relevancy, 1.0),
        }
    
    def _compute_aggregate(self, results: list[EvalResult]) -> dict:
        """Compute aggregate metrics across all results."""
        def safe_avg(values: list) -> float | None:
            valid = [v for v in values if v is not None]
            return sum(valid) / len(valid) if valid else None
        
        return {
            "context_precision": safe_avg([r.context_precision for r in results]),
            "context_recall": safe_avg([r.context_recall for r in results]),
            "faithfulness": safe_avg([r.faithfulness for r in results]),
            "answer_relevancy": safe_avg([r.answer_relevancy for r in results]),
            "overall_score": safe_avg([r.overall_score for r in results]),
        }
    
    def generate_test_cases(
        self,
        num_cases: int = 10,
        sample_docs: int = 50,
    ) -> list[dict]:
        """
        Generate synthetic test cases from existing documents.
        
        Uses document content to create question-answer pairs.
        Useful for bootstrapping evaluation before you have real test data.
        """
        # Fetch sample documents
        docs = self.rag.client.table("documents").select(
            "id, content"
        ).eq("namespace", self.rag.namespace).limit(sample_docs).execute()
        
        if not docs.data:
            logger.warning("No documents found to generate test cases")
            return []
        
        test_cases = []
        for doc in docs.data[:num_cases]:
            content = doc["content"]
            
            # Simple heuristic: first sentence often describes the topic
            sentences = content.split(". ")
            if sentences:
                # Question from first sentence
                question = f"What is {sentences[0].split(':')[0].lower().strip()}?"
                expected = sentences[0]
                
                test_cases.append({
                    "question": question,
                    "expected_answer": expected,
                    "tags": ["synthetic", f"doc_{doc['id']}"],
                })
        
        return test_cases
    
    def save_results(self, results: dict, output_path: str) -> None:
        """Save evaluation results to JSON file."""
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")
    
    def print_summary(self, results: dict) -> None:
        """Print a human-readable summary of evaluation results."""
        agg = results["aggregate"]
        
        print("\n" + "=" * 50)
        print(f"RAG Evaluation Summary - {results['namespace']}")
        print("=" * 50)
        print(f"Test cases: {results['num_cases']}")
        print(f"Timestamp: {results['timestamp']}")
        print("-" * 50)
        print("Aggregate Metrics:")
        print(f"  Context Precision: {agg['context_precision']:.3f}" if agg['context_precision'] else "  Context Precision: N/A")
        print(f"  Context Recall:    {agg['context_recall']:.3f}" if agg['context_recall'] else "  Context Recall:    N/A")
        print(f"  Faithfulness:      {agg['faithfulness']:.3f}" if agg['faithfulness'] else "  Faithfulness:      N/A")
        print(f"  Answer Relevancy:  {agg['answer_relevancy']:.3f}" if agg['answer_relevancy'] else "  Answer Relevancy:  N/A")
        print("-" * 50)
        print(f"  Overall Score:     {agg['overall_score']:.3f}" if agg['overall_score'] else "  Overall Score:     N/A")
        print("=" * 50 + "\n")
