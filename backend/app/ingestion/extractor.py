"""
Graph Extractor - Extract relations from content
================================================

Uses LLM to find relationships between content pieces.
"""
from __future__ import annotations
from typing import List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

from app.config import config
from app.services.supabase_ops import supabase_ops


class Edge(BaseModel):
    target_snippet: str = Field(description="Quote from text that relates")
    relation: str = Field(description="relates_to, contradicts, contains, requires")


class Extraction(BaseModel):
    edges: List[Edge]


_emb = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
_llm = ChatOpenAI(model=config.CHAT_MODEL, temperature=0)


def ingest_document(content: str, metadata: dict | None = None) -> dict:
    """Ingest with optional graph extraction."""
    # Embed and insert
    vec = _emb.embed_query(content)
    node = supabase_ops.insert_document(content, vec, metadata or {})
    source_id = node["id"]
    
    # Try to extract relations
    edges_created = 0
    try:
        extractor = _llm.with_structured_output(Extraction)
        result = extractor.invoke(f"Extract key relationships from:\n{content[:3000]}")
        
        for edge in result.edges:
            target = supabase_ops.find_document_by_content(edge.target_snippet[:100])
            if target and target["id"] != source_id:
                try:
                    supabase_ops.insert_relation(source_id, target["id"], edge.relation, {})
                    edges_created += 1
                except Exception:
                    pass
    except Exception:
        pass
    
    return {"id": source_id, "edges": edges_created}
