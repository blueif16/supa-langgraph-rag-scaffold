# RAG Debug Toolkit

A comprehensive debugging and visualization toolkit for your RAG system.

## Installation

```bash
# Install with debug dependencies
pip install -e ".[debug]"

# Or install all dependencies
pip install -e ".[all]"
```

Then run the SQL migration:
```sql
-- In Supabase SQL Editor
-- Run: supabase/migrations/20260126_debug_functions.sql
```

## Environment Configuration

The toolkit automatically searches for `.env` file from the current working directory upwards to the project root. This allows you to:

1. Install this scaffold in any project subdirectory
2. Run `rag-debug` commands from anywhere in your project
3. The tool will find your `.env` file in the project root automatically

**Required environment variables:**
```bash
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SECRET_KEY=sb_secret_your_secret_key_here
GEMINI_API_KEY=your-gemini-api-key
```

See `.env.example` for all available options.

## Quick Start

### CLI Usage

```bash
# List all namespaces with statistics
rag-debug namespaces

# Visualize your knowledge graph
rag-debug visualize --namespace video_styles --output graph.html

# Debug a search query
rag-debug debug "energetic fast cuts" --namespace video_styles

# Run robustness tests
rag-debug test --namespace video_styles

# Analyze content coverage
rag-debug analyze --namespace video_styles

# Compare multiple queries
rag-debug compare --namespace video_styles "fast cuts" "quick editing" "energetic style"

# Evaluate with test cases
rag-debug evaluate --namespace video_styles --cases test_cases.json

# Or generate synthetic test cases
rag-debug evaluate --namespace video_styles --generate 10
```

### Python Usage

```python
from app.core import RAGStore
from app.debug import (
    RAGVisualizer,
    RAGEvaluator,
    RetrievalDebugger,
    RAGRobustnessTests,
)

rag = RAGStore(namespace="video_styles")

# 1. Visualize the knowledge graph
viz = RAGVisualizer(rag)
viz.visualize("graph.html")  # Interactive HTML
viz.visualize_search_path("energetic style", "search.html")  # See search results

# 2. Debug a specific query
debugger = RetrievalDebugger(rag)
result = debugger.debug_search("energetic fast cuts")
debugger.print_debug(result)
# Shows: BM25 results, Vector results, Hybrid results, Diagnosis

# 3. Run evaluation
evaluator = RAGEvaluator(rag)
results = evaluator.evaluate([
    {"question": "What's the hook pacing?", "expected_answer": "1.5-2s cuts"},
])
evaluator.print_summary(results)

# 4. Run robustness tests
tests = RAGRobustnessTests(rag)
suite = tests.run_all()
tests.print_results(suite)
```

## Tools Overview

### 1. Graph Visualizer (`RAGVisualizer`)

Interactive visualization of your knowledge graph using PyVis.

**Use when:**
- You want to see the overall structure of your knowledge graph
- Debugging why certain documents aren't being found
- Understanding how documents are connected via relations

**Features:**
- Full graph visualization
- Search path visualization (see exactly what `search_context_mesh` returns)
- Query comparison (compare results from multiple queries)
- Export to interactive HTML

### 2. Retrieval Debugger (`RetrievalDebugger`)

Compare different search methods side-by-side.

**Use when:**
- Search results seem wrong or incomplete
- You want to understand why certain documents rank higher
- Debugging BM25 vs Vector vs Hybrid differences

**Shows:**
- BM25 results (keyword matching)
- Vector results (semantic matching)
- Hybrid RRF results (combined)
- Hybrid + Graph results (with expansion)
- Automatic diagnosis of issues

**Example Diagnosis Output:**
```
✅ Good overlap between keyword and semantic search - RRF fusion is working well.
📊 Graph expansion added 3 documents via relations.
🔍 Vector found 2 docs that BM25 missed - these are semantically related but use different words.
```

### 3. RAG Evaluator (`RAGEvaluator`)

Systematic evaluation using RAGAS metrics.

**Metrics:**
- **Context Precision**: Are retrieved docs relevant to the question?
- **Context Recall**: Did we retrieve all relevant docs?
- **Faithfulness**: Is the answer grounded in the context?
- **Answer Relevancy**: Does the answer address the question?

**Use when:**
- Before/after making changes (regression testing)
- Setting up CI/CD quality gates
- Comparing different configurations

### 4. Robustness Tests (`RAGRobustnessTests`)

Automated test suite for edge cases.

**Tests:**
- `test_synonym_robustness`: Similar queries → similar results
- `test_typo_tolerance`: Misspellings still find content
- `test_empty_results`: Graceful handling of no results
- `test_long_queries`: No crashes on long inputs
- `test_special_characters`: Handle quotes, emoji, unicode
- `test_graph_depth_impact`: Measure graph expansion value
- `test_search_consistency`: Repeated queries → same results
- `test_batch_search_performance`: Latency under load

## SQL Debug Functions

The migration adds these functions to Supabase:

```sql
-- BM25-only search (isolate keyword matching)
SELECT * FROM debug_bm25_search('your query', 5, 'namespace');

-- Detailed RRF fusion breakdown
SELECT * FROM debug_rrf_fusion('query', embedding, 10, 60, 'namespace');
-- Returns: id, content, bm25_rank, vector_rank, bm25_score, vector_score, rrf_score

-- Graph statistics
SELECT * FROM debug_graph_stats('namespace');
-- Returns: total_documents, total_relations, avg_relations_per_doc, relation_types, orphan_documents

-- Find similar documents (embedding quality check)
SELECT * FROM debug_find_similar(doc_id, 5);

-- Content analysis
SELECT * FROM debug_content_stats('namespace');
-- Returns: avg_content_length, min/max, docs_under_100_chars, docs_without_embedding

-- Document neighborhood (for graph viz)
SELECT * FROM debug_get_neighborhood(doc_id, 2);
```

## Common Issues & Diagnosis

### "BM25 returned nothing"
- Query terms don't exist in your content
- Try adding more descriptive keywords to your content
- Check if content is self-describing

### "Zero overlap between BM25 and Vector"
- Content might be too sparse
- Consider enriching content with synonyms and descriptions
- Check if embeddings are properly generated

### "Graph expansion added 0 documents"
- No relations exist in your namespace
- Add relations with `rag.add_relation()`

### "Low Context Precision"
- Retrieved docs aren't relevant to the question
- Improve content descriptions
- Consider adjusting `rrf_k` parameter

### "Low Faithfulness"
- Answers aren't grounded in retrieved context
- Retrieved context might be insufficient
- Consider increasing `top_k`

## Best Practices

1. **Run robustness tests in CI/CD**
   ```bash
   rag-debug test --namespace prod --output results.json
   # Check pass rate in CI
   ```

2. **Visualize before and after changes**
   ```bash
   rag-debug visualize -n my_namespace -o before.html
   # Make changes
   rag-debug visualize -n my_namespace -o after.html
   ```

3. **Create domain-specific test cases**
   ```python
   tests = RAGRobustnessTests(rag)
   tests.configure_tests(
       synonym_pairs=[
           ("your domain term", "synonym"),
       ],
       typo_pairs=[
           ("correct", "typo"),
       ],
   )
   ```

4. **Monitor evaluation metrics over time**
   ```python
   results = evaluator.evaluate(test_cases)
   evaluator.save_results(results, f"eval_{datetime.now().isoformat()}.json")
   ```
