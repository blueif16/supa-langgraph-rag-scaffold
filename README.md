# Supabase RAG Scaffold

**The source of truth for LLM-assisted RAG setup.**

When you or an LLM needs to add RAG to a project, reference this scaffold. Don't invent new patterns - use what's here.

## Philosophy

- **Content is rich** - Self-describing natural text. Embeddings capture everything.
- **Metadata is minimal** - Just `source` and `type` for reference, not filtering.
- **Power from indexing** - RRF (BM25 + Vector) + Graph does the work.
- **Gemini embeddings** - 使用 Gemini API 的 768 维 embeddings，性能与质量平衡。

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

## Environment Setup

配置 `.env` 文件：

```bash
# Supabase
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=your-service-role-key
DATABASE_URL=postgresql://postgres:password@db.xxx.supabase.co:5432/postgres

# Gemini API
GEMINI_API_KEY=your-gemini-api-key

# Optional
EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_DIM=768
CHAT_MODEL=gemini-2.0-flash-exp
MATCH_COUNT=5
RRF_K=60
GRAPH_DEPTH=2
MAX_RETRY=2
```

## For LLMs: Read SPECIFICATION.md First

See [SPECIFICATION.md](./SPECIFICATION.md) for:
- Decision flow for setting up RAG
- Research prompt templates
- Expected content formats
- Examples by use case

## Setup

1. **Run initial migration**
   ```bash
   # 在 Supabase SQL Editor 中运行
   supabase/migrations/20260125_init_graph_schema.sql
   ```

2. **Run embedding dimension update**
   ```bash
   # 在 Supabase SQL Editor 中运行
   supabase/migrations/20260126_update_embedding_dim.sql
   ```

3. **Install dependencies**
   ```bash
   cd backend
   pip install -e .
   ```

4. **Start ingesting**

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
│   ├── gemini_embeddings.py  # Gemini embeddings 封装
│   ├── tool_factory.py   # LangGraph tools
│   └── adapters.py       # Data extraction
└── supabase/migrations/  # SQL schema
```

## Clone Into Your Project

```bash
cp -r backend/app/core your_project/rag
```

Then `from rag import RAGStore`.

## Technical Details

### Embeddings
- 模型: `gemini-embedding-001`
- 维度: 768 (推荐值，平衡性能和质量)
- 任务类型:
  - 查询: `RETRIEVAL_QUERY`
  - 文档: `RETRIEVAL_DOCUMENT`
- 归一化: 自动对非 3072 维度的 embeddings 进行归一化

### Vector Search
- 索引类型: HNSW (Hierarchical Navigable Small World)
- 距离度量: Cosine similarity
- 混合检索: RRF (Reciprocal Rank Fusion) 结合 BM25 和向量搜索
- 图遍历: 支持多层关系扩展
