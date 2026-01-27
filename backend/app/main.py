"""
FastAPI App - RAG API
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from app.config import config


class ChatRequest(BaseModel):
    query: str
    conversation_id: str = "default"


class IngestRequest(BaseModel):
    content: str
    source: str | None = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


app = FastAPI(title="RAG API")

# Default workflow
from app.graph.workflow import app as agent_app
from app.services.supabase_ops import supabase_ops
from app.core.gemini_embeddings import GeminiEmbeddings

_emb = GeminiEmbeddings(
    model=config.EMBEDDING_MODEL,
    output_dimensionality=config.EMBEDDING_DIM
)


@app.post("/chat")
async def chat(req: ChatRequest):
    """Self-correcting RAG agent."""
    try:
        result = agent_app.invoke(
            {"question": req.query},
            config={"configurable": {"thread_id": req.conversation_id}}
        )
        return {
            "response": result["messages"][-1].content if result.get("messages") else "",
            "retries": result.get("retry_count", 0)
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/search")
async def search(req: SearchRequest):
    """Direct search."""
    vec = _emb.embed_query(req.query)
    results = supabase_ops.retrieve_context_mesh(req.query, vec)
    return {"results": results[:req.top_k]}


@app.post("/ingest")
async def ingest(req: IngestRequest, bg: BackgroundTasks):
    """Ingest content."""
    from app.ingestion.extractor import ingest_document
    bg.add_task(ingest_document, req.content, {"source": req.source} if req.source else {})
    return {"status": "queued"}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)
