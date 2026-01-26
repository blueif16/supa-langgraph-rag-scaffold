# Supabase RAG Scaffold

**The source of truth for LLM-assisted RAG setup.**

When you or an LLM needs to add RAG to a project, reference this scaffold. Don't invent new patterns - use what's here.

## Philosophy

- **Content is rich** - Self-describing natural text. Embeddings capture everything.
- **Metadata is minimal** - Just `source` and `type` for reference, not filtering.
- **Power from indexing** - RRF (BM25 + Vector) + Graph does the work.

## Quick Start

```python
from app.core import RAGStore

# Initialize
rag = RAGStore(namespace="my_knowledge")

# Ingest rich, self-describing content
rag.ingest("""
Product Hunt hook pacing: Fast energetic openings with 1.5-2 second cuts 
in the first 10 seconds. High energy, immediate value prop, no slow builds. 
Cold open straight into product. Works for SaaS launches, tool demos.
""", source="ph_analysis_2025")

# Search by feeling/semantics
results = rag.search("energetic fast opening style")
```

## For LLMs: Read SPECIFICATION.md First

See [SPECIFICATION.md](./SPECIFICATION.md) for:
- Decision flow for setting up RAG
- Research prompt templates
- Expected content formats
- Examples by use case

## Setup

1. **Get SQL**
   ```python
   print(rag.get_setup_sql())
   ```

2. **Run in Supabase SQL Editor**

3. **Start ingesting**

## Core API

```python
from app.core import RAGStore

rag = RAGStore(namespace="my_project")

# Ingest
rag.ingest("Content here...", source="optional", type="optional")
rag.ingest_batch(["content 1", "content 2", ...])

# Search (SOTA: hybrid RRF + graph)
results = rag.search("query")

# Graph edges
rag.add_relation(id1, id2, "relates_to")

# Utils
rag.stats()
rag.delete_all()
```

## Data Adapters

```python
from app.core import DataAdapter

# From various sources → list of strings
contents = DataAdapter.from_json_file("data.json", content_field="text")
contents = DataAdapter.from_csv("data.csv", content_column="body")
contents = DataAdapter.from_api_response(response, content_field="description")
contents = DataAdapter.from_text_chunks(long_text, chunk_size=1500)

# Then ingest
rag.ingest_batch(contents, source="my_source")
```

## LangGraph Integration

```python
from app.core import RAGStore, create_search_tool, create_search_fn

rag = RAGStore(namespace="agent_kb")

# As tool (for ReAct agents)
tool = create_search_tool(rag, name="search_kb")
agent = create_react_agent(llm, tools=[tool])

# As function (for custom nodes)
search = create_search_fn(rag)
def retrieve_node(state):
    return {"context": search(state["question"])}
```

## Project Structure

```
├── SPECIFICATION.md      # LLM instruction manual
├── backend/app/core/     # The portable module
│   ├── rag_store.py      # RAGStore class
│   ├── tool_factory.py   # LangGraph tools
│   └── adapters.py       # Data extraction
└── supabase/migrations/  # SQL schema
```

## Clone Into Your Project

```bash
cp -r backend/app/core your_project/rag
```

Then `from rag import RAGStore`.
