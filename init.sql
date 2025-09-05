CREATE EXTENSION IF NOT EXISTS vector;

-- 384 dimensions for 'all-MiniLM-L6-v2' sentence-transformers
CREATE TABLE IF NOT EXISTS chunks (
  id BIGSERIAL PRIMARY KEY,
  doc_id TEXT NOT NULL,
  chunk_id INTEGER NOT NULL,
  content TEXT NOT NULL,
  embedding VECTOR(384) NOT NULL,
  meta JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
