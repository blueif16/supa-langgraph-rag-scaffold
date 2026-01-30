-- ============================================================================
-- RAG Debug Functions
-- Run this after the main schema migration
-- ============================================================================

-- BM25-only search for debugging (isolate keyword matching)
CREATE OR REPLACE FUNCTION debug_bm25_search(
  query_text TEXT,
  match_count INT DEFAULT 5,
  filter_namespace TEXT DEFAULT NULL
)
RETURNS TABLE (
  id BIGINT,
  content TEXT,
  metadata JSONB,
  rank FLOAT
)
LANGUAGE sql STABLE
AS $$
  SELECT
    d.id,
    d.content,
    d.metadata,
    ts_rank_cd(to_tsvector('english', d.content), plainto_tsquery('english', query_text))::FLOAT AS rank
  FROM documents d
  WHERE to_tsvector('english', d.content) @@ plainto_tsquery('english', query_text)
    AND (filter_namespace IS NULL OR d.namespace = filter_namespace)
  ORDER BY rank DESC
  LIMIT match_count;
$$;

-- Debug RRF fusion (show scores from both methods)
CREATE OR REPLACE FUNCTION debug_rrf_fusion(
  query_text TEXT,
  query_embedding vector(768),
  match_count INT DEFAULT 10,
  rrf_k INT DEFAULT 60,
  filter_namespace TEXT DEFAULT NULL
)
RETURNS TABLE (
  id BIGINT,
  content TEXT,
  bm25_rank INT,
  vector_rank INT,
  bm25_score FLOAT,
  vector_score FLOAT,
  rrf_score FLOAT
)
LANGUAGE plpgsql STABLE
AS $$
BEGIN
  RETURN QUERY
  WITH
  -- BM25 ranking
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
    SELECT id, rank, ROW_NUMBER() OVER (ORDER BY rank DESC)::INT AS rank_pos
    FROM fts
  ),
  
  -- Vector ranking
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
    SELECT id, 1 - dist AS similarity, ROW_NUMBER() OVER (ORDER BY dist)::INT AS rank_pos
    FROM vec
  ),
  
  -- Combine with RRF
  combined AS (
    SELECT
      COALESCE(f.id, v.id) AS id,
      f.rank_pos AS bm25_rank,
      v.rank_pos AS vector_rank,
      COALESCE(f.rank, 0)::FLOAT AS bm25_score,
      COALESCE(v.similarity, 0)::FLOAT AS vector_score,
      (COALESCE(1.0 / (rrf_k + f.rank_pos), 0.0) +
       COALESCE(1.0 / (rrf_k + v.rank_pos), 0.0))::FLOAT AS rrf_score
    FROM fts_ranked f
    FULL OUTER JOIN vec_ranked v ON f.id = v.id
  )
  
  SELECT
    c.id,
    d.content,
    c.bm25_rank,
    c.vector_rank,
    c.bm25_score,
    c.vector_score,
    c.rrf_score
  FROM combined c
  JOIN documents d ON c.id = d.id
  ORDER BY c.rrf_score DESC
  LIMIT match_count;
END;
$$;

-- Graph statistics for debugging
CREATE OR REPLACE FUNCTION debug_graph_stats(
  filter_namespace TEXT DEFAULT NULL
)
RETURNS TABLE (
  total_documents BIGINT,
  total_relations BIGINT,
  avg_relations_per_doc FLOAT,
  relation_types JSONB,
  orphan_documents BIGINT
)
LANGUAGE plpgsql STABLE
AS $$
DECLARE
  doc_count BIGINT;
  rel_count BIGINT;
  orphan_count BIGINT;
  rel_types JSONB;
BEGIN
  -- Total documents
  SELECT COUNT(*) INTO doc_count
  FROM documents
  WHERE filter_namespace IS NULL OR namespace = filter_namespace;
  
  -- Total relations
  SELECT COUNT(*) INTO rel_count
  FROM doc_relations
  WHERE filter_namespace IS NULL OR namespace = filter_namespace;
  
  -- Relation types breakdown
  SELECT jsonb_object_agg(type, cnt) INTO rel_types
  FROM (
    SELECT type, COUNT(*) as cnt
    FROM doc_relations
    WHERE filter_namespace IS NULL OR namespace = filter_namespace
    GROUP BY type
  ) t;
  
  -- Orphan documents (no relations)
  SELECT COUNT(*) INTO orphan_count
  FROM documents d
  WHERE (filter_namespace IS NULL OR d.namespace = filter_namespace)
    AND NOT EXISTS (
      SELECT 1 FROM doc_relations r
      WHERE r.source_id = d.id OR r.target_id = d.id
    );
  
  RETURN QUERY SELECT
    doc_count,
    rel_count,
    CASE WHEN doc_count > 0 THEN (rel_count::FLOAT / doc_count) ELSE 0 END,
    COALESCE(rel_types, '{}'::JSONB),
    orphan_count;
END;
$$;

-- Find similar documents (for debugging embedding quality)
CREATE OR REPLACE FUNCTION debug_find_similar(
  doc_id BIGINT,
  match_count INT DEFAULT 5
)
RETURNS TABLE (
  id BIGINT,
  content TEXT,
  similarity FLOAT
)
LANGUAGE sql STABLE
AS $$
  SELECT
    d2.id,
    d2.content,
    (1 - (d1.embedding <=> d2.embedding))::FLOAT AS similarity
  FROM documents d1
  CROSS JOIN documents d2
  WHERE d1.id = doc_id
    AND d2.id != doc_id
    AND d1.namespace = d2.namespace
  ORDER BY d1.embedding <=> d2.embedding
  LIMIT match_count;
$$;

-- Debug content analysis
CREATE OR REPLACE FUNCTION debug_content_stats(
  filter_namespace TEXT DEFAULT NULL
)
RETURNS TABLE (
  total_documents BIGINT,
  avg_content_length FLOAT,
  min_content_length INT,
  max_content_length INT,
  avg_word_count FLOAT,
  docs_under_100_chars BIGINT,
  docs_without_embedding BIGINT
)
LANGUAGE sql STABLE
AS $$
  SELECT
    COUNT(*)::BIGINT,
    AVG(LENGTH(content))::FLOAT,
    MIN(LENGTH(content))::INT,
    MAX(LENGTH(content))::INT,
    AVG(array_length(regexp_split_to_array(content, '\s+'), 1))::FLOAT,
    COUNT(*) FILTER (WHERE LENGTH(content) < 100)::BIGINT,
    COUNT(*) FILTER (WHERE embedding IS NULL)::BIGINT
  FROM documents
  WHERE filter_namespace IS NULL OR namespace = filter_namespace;
$$;

-- Get document neighborhood (for graph visualization)
CREATE OR REPLACE FUNCTION debug_get_neighborhood(
  doc_id BIGINT,
  depth INT DEFAULT 2
)
RETURNS TABLE (
  id BIGINT,
  content TEXT,
  relation_type TEXT,
  hop_distance INT
)
LANGUAGE plpgsql STABLE
AS $$
BEGIN
  RETURN QUERY
  WITH RECURSIVE neighborhood AS (
    -- Starting document
    SELECT
      d.id,
      d.content,
      'self'::TEXT AS relation_type,
      0 AS hop_distance,
      ARRAY[d.id] AS path
    FROM documents d
    WHERE d.id = doc_id
    
    UNION ALL
    
    -- Connected documents
    SELECT
      d.id,
      d.content,
      r.type,
      n.hop_distance + 1,
      n.path || d.id
    FROM neighborhood n
    JOIN doc_relations r ON r.source_id = n.id OR r.target_id = n.id
    JOIN documents d ON d.id = CASE
      WHEN r.source_id = n.id THEN r.target_id
      ELSE r.source_id
    END
    WHERE n.hop_distance < depth
      AND NOT d.id = ANY(n.path)
  )
  SELECT DISTINCT ON (neighborhood.id)
    neighborhood.id,
    neighborhood.content,
    neighborhood.relation_type,
    neighborhood.hop_distance
  FROM neighborhood
  ORDER BY neighborhood.id, neighborhood.hop_distance;
END;
$$;

-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON FUNCTION debug_bm25_search IS 'BM25-only search for isolating keyword matching issues';
COMMENT ON FUNCTION debug_rrf_fusion IS 'Show detailed RRF fusion scores from both BM25 and Vector';
COMMENT ON FUNCTION debug_graph_stats IS 'Get statistics about the knowledge graph structure';
COMMENT ON FUNCTION debug_find_similar IS 'Find documents similar to a given document (embedding similarity)';
COMMENT ON FUNCTION debug_content_stats IS 'Analyze content characteristics in the namespace';
COMMENT ON FUNCTION debug_get_neighborhood IS 'Get all documents connected to a document within N hops';
