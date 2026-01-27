-- ============================================================================
-- 更新 Embedding 维度: 从 OpenAI 1536 维 -> Gemini 768 维
-- ============================================================================

-- 1. 删除旧的向量索引
DROP INDEX IF EXISTS idx_docs_embedding;

-- 2. 修改 embedding 列的维度
ALTER TABLE documents
  ALTER COLUMN embedding TYPE vector(768);

-- 3. 重建向量索引
CREATE INDEX idx_docs_embedding
  ON documents USING hnsw (embedding vector_cosine_ops);

-- 4. 更新搜索函数 - 支持 768 维向量
CREATE OR REPLACE FUNCTION search_context_mesh(
  query_text TEXT,
  query_embedding vector(768),
  match_count INT DEFAULT 5,
  rrf_k INT DEFAULT 60,
  graph_depth INT DEFAULT 2,
  filter_namespace TEXT DEFAULT NULL
)
RETURNS TABLE (
  id BIGINT,
  content TEXT,
  metadata JSONB,
  source_type TEXT,
  score FLOAT,
  depth INT
)
LANGUAGE plpgsql STABLE
AS $$
BEGIN
  RETURN QUERY
  WITH RECURSIVE

  -- A. Full-Text Search (keyword matching)
  fts AS (
    SELECT
      d.id,
      ts_rank_cd(to_tsvector('english', d.content), plainto_tsquery('english', query_text)) AS rank
    FROM documents d
    WHERE to_tsvector('english', d.content) @@ plainto_tsquery('english', query_text)
      AND (filter_namespace IS NULL OR d.namespace = filter_namespace)
    ORDER BY rank DESC
    LIMIT match_count * 3
  ),
  fts_ranked AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY rank DESC) AS rank_pos FROM fts
  ),

  -- B. Vector Search (semantic matching)
  vec AS (
    SELECT
      d.id,
      d.embedding <=> query_embedding AS dist
    FROM documents d
    WHERE filter_namespace IS NULL OR d.namespace = filter_namespace
    ORDER BY dist
    LIMIT match_count * 3
  ),
  vec_ranked AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY dist) AS rank_pos FROM vec
  ),

  -- C. Reciprocal Rank Fusion
  rrf AS (
    SELECT
      COALESCE(f.id, v.id) AS id,
      COALESCE(1.0 / (rrf_k + f.rank_pos), 0.0) +
      COALESCE(1.0 / (rrf_k + v.rank_pos), 0.0) AS rrf_score
    FROM fts_ranked f
    FULL OUTER JOIN vec_ranked v ON f.id = v.id
  ),
  seeds AS (
    SELECT id, rrf_score FROM rrf ORDER BY rrf_score DESC LIMIT match_count
  ),

  -- D. Graph Expansion
  graph AS (
    -- Seeds
    SELECT
      d.id, d.content, d.metadata,
      'seed'::TEXT AS source_type,
      s.rrf_score AS score,
      0 AS depth,
      ARRAY[d.id] AS path
    FROM seeds s
    JOIN documents d ON s.id = d.id

    UNION ALL

    -- Traverse
    SELECT
      d.id, d.content, d.metadata,
      r.type AS source_type,
      g.score * 0.8,
      g.depth + 1,
      g.path || d.id
    FROM graph g
    JOIN doc_relations r ON r.source_id = g.id
    JOIN documents d ON d.id = r.target_id
    WHERE g.depth < graph_depth
      AND NOT d.id = ANY(g.path)
      AND (filter_namespace IS NULL OR d.namespace = filter_namespace)
  )

  SELECT DISTINCT ON (graph.id)
    graph.id,
    graph.content,
    graph.metadata,
    graph.source_type,
    graph.score::FLOAT,
    graph.depth
  FROM graph
  ORDER BY graph.id, graph.score DESC;
END;
$$;

-- 5. 更新简单向量搜索函数
CREATE OR REPLACE FUNCTION search_vector(
  query_embedding vector(768),
  match_count INT DEFAULT 5,
  filter_namespace TEXT DEFAULT NULL
)
RETURNS TABLE (id BIGINT, content TEXT, metadata JSONB, similarity FLOAT)
LANGUAGE sql STABLE
AS $$
  SELECT
    id, content, metadata,
    (1 - (embedding <=> query_embedding))::FLOAT AS similarity
  FROM documents
  WHERE filter_namespace IS NULL OR namespace = filter_namespace
  ORDER BY embedding <=> query_embedding
  LIMIT match_count;
$$;
