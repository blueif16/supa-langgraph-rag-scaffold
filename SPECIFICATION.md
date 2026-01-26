# RAG Scaffold Specification
## For LLM Consumption - Read This First

> **Purpose**: Guide LLMs to correctly set up RAG using this scaffold's 2026 SOTA patterns. Do NOT invent new approaches - use what's here.

---

## Core Philosophy

1. **Content is king** - Make it rich, self-describing. Embeddings capture everything.
2. **Metadata is minimal** - Only for isolation/reference, not filtering.
3. **Power from indexing** - RRF (BM25 + Vector) + Graph traversal does the work.

---

## Schema

One flexible schema for everything:

```sql
content TEXT NOT NULL,        -- Rich, self-describing knowledge
metadata JSONB DEFAULT '{}',  -- Minimal: source, type (optional)
namespace TEXT,               -- Project/feature isolation
embedding vector(1536),       -- Semantic search
-- + graph edges for relationships
```

**Metadata keeps it simple:**
```json
{
  "source": "where this came from",
  "type": "pattern | guide | note | reference"  // optional
}
```

**Content does the heavy lifting:**
```
"Product Hunt hook pacing: Fast, energetic openings with 1.5-2 second cuts 
in the first 10 seconds. High energy, immediate value proposition, no slow 
builds. Cold open straight into the product. Works for SaaS launches, tool 
demos, app reveals. Common music: upbeat electronic, 120-130 BPM. 
Text animations: quick fade-up, 0.3s duration. Reference: top 50 PH 2025."
```

The embedding + BM25 indexes all of this. Query "energetic fast cuts" or "SaaS launch video style" - it finds it.

---

## Decision Flow

```
1. WHAT DATA? → Define namespace
2. HOW TO GET IT? → Research prompt + format for user
3. HOW TO SEARCH? → Always hybrid + graph (default SOTA)
```

---

## Ingestion Patterns

### Pattern A: Research-to-Ingest (Most Common)

LLM outputs for user:
1. **Research prompt** - What to investigate
2. **Content format** - How to structure findings

User runs research (with advanced LLM/deep research), reviews, then ingests.

### Pattern B: Direct Ingest

For existing content: docs, notes, files. Use `DataAdapter` to transform and ingest.

### Pattern C: Incremental Learning

Agent learns from conversations/feedback. Ingest with `source: "learned"`.

---

## Research Prompt Template

When user needs knowledge for a domain, output this format:

```markdown
## Research Task: [Domain]

### Prompt for Deep Research
[Specific research prompt - what to analyze, what to extract, what to compare]

### Structure Your Findings As
Each piece of knowledge should be a self-contained paragraph/section that includes:
- What it is (name/title inline)
- The actual insight/pattern/practice
- When it applies / context
- Any relevant specifics (numbers, examples)
- Source/reference if applicable

### Example Output
[One concrete example of properly formatted content]

### Namespace
`[suggested_namespace]`
```

---

## Examples by Use Case

### Example 1: Video Editing Style Guide

**User says**: "I want to give my video editing agent taste for Product Hunt style videos"

**LLM outputs**:

---

## Research Task: Product Hunt Video Style

### Prompt for Deep Research
Analyze the top 50 most successful Product Hunt launch videos from 2024-2025. For each, extract:
- Opening hook structure and timing
- Cut rhythm and pacing patterns by section (intro, demo, CTA)
- Text animation styles and timing
- Music/audio patterns (energy, BPM ranges, style)
- Transition types and frequency
- Color grading tendencies
- Thumbnail patterns that drive clicks
- Common structures that get high engagement

Synthesize into actionable patterns, not just observations.

### Structure Your Findings As
Each pattern as a self-contained block:
```
[Pattern Name]: [Core insight in first sentence]. [Details, specifics, numbers]. 
[When to use / context]. [What to avoid if relevant]. [Reference/source].
```

### Example Output
```
Product Hunt Hook Pacing: Open cold with product in action within first 2 seconds, 
no logos or intros. Cut rhythm of 1.5-2s in first 10 seconds creates energy and 
holds attention. Slow to 3-4s cuts during feature demos for comprehension. Works 
for SaaS, dev tools, productivity apps. Avoid: slow fades, long title cards, 
"welcome to" openings. Based on top 50 PH launches 2024-2025.
```

### Namespace
`video_style_product_hunt`

---

### Example 2: Agent Personal Memory

**User says**: "I want my agent to remember user preferences and context"

**LLM outputs**:

---

## Research Task: Not Applicable - Incremental Ingestion

This is learned over time, not researched upfront.

### Ingestion Pattern
When agent learns something about user/project, ingest:

```python
rag.ingest({
    "content": "User prefers concise responses without bullet points. They work in fintech and understand technical jargon. Timezone is PST, usually active mornings.",
    "metadata": {"source": "learned_from_conversation"}
}, namespace="user_alice_prefs")
```

### Content Format
Natural sentences that capture the full context:
```
[What was learned]. [Relevant context]. [When it applies].
```

### Namespace
`user_{user_id}_prefs` or `project_{project_id}_context`

---

### Example 3: Technical Documentation

**User says**: "I need to ingest our API docs for a support bot"

**LLM outputs**:

---

## Research Task: Not Applicable - Direct Ingestion

### Ingestion Pattern
Use `DataAdapter` to chunk and ingest existing docs:

```python
from app.core import RAGStore, DataAdapter

rag = RAGStore(namespace="api_docs_v2")

# From markdown files
chunks = DataAdapter.from_text_chunks(
    markdown_content, 
    chunk_size=1500,  # Larger chunks preserve context
    chunk_overlap=200
)
rag.ingest_batch(chunks)
```

### Content Already Exists
No research needed. Ensure chunks are self-contained (include endpoint name, method, description in each chunk).

### Namespace
`{product}_docs_v{version}`

---

## Retrieval

**Default: Always use `search_context_mesh`** (hybrid RRF + graph)

Only use `search_vector` for speed-critical simple lookups.

```python
# SOTA search - handles everything
results = rag.search_context_mesh("energetic hook style for saas demo")

# Fast fallback (rare)
results = rag.search_vector("password reset", match_count=3)
```

---

## Namespace Strategy

Namespaces isolate data. Same table, different contexts.

| Pattern | Example |
|---------|---------|
| By feature | `video_styles`, `code_patterns`, `support_kb` |
| By project | `project_acme`, `project_beta` |
| By user | `user_alice`, `user_bob` |
| By version | `docs_v1`, `docs_v2` |

---

## Graph Edges

Use when relationships matter:

```python
# Style A relates to Style B
rag.add_relation(style_a_id, style_b_id, "complements")

# Pattern X contradicts Pattern Y  
rag.add_relation(pattern_x_id, pattern_y_id, "contradicts")

# Guide contains multiple practices
rag.add_relation(guide_id, practice_id, "contains")
```

Graph traversal automatically finds related content during search.

---

## Anti-Patterns (Do NOT Do)

❌ **Heavy metadata filtering** - Use rich content + semantic search instead
❌ **Inventing new search methods** - Use the provided SOTA functions
❌ **Tiny chunks** - Lose context. 1000-2000 chars minimum.
❌ **Structured data in content** - No JSON in content field. Natural text only.
❌ **Separate fields for attributes** - Put it in the content text.

---

## Quick Start Commands

```python
from app.core import RAGStore, DataAdapter

# 1. Initialize
rag = RAGStore(namespace="my_knowledge")

# 2. Setup (once)
print(rag.get_setup_sql())  # Run in Supabase

# 3. Ingest
rag.ingest("Your rich, self-describing content here...")

# 4. Search
results = rag.search_context_mesh("your query")
```
