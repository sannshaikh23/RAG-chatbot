import os
import io
import re
import uuid
import psycopg2
from psycopg2.extras import Json
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
import pdfplumber  
import pytesseract  
from pdf2image import convert_from_bytes  
from PIL import Image  


_EMB_MODEL = None
def get_embedder():
    global _EMB_MODEL
    if _EMB_MODEL is None:
        _EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMB_MODEL

def connect_db():
    conn = psycopg2.connect(
        host=os.getenv("PGHOST"),
        port=int(os.getenv("PGPORT")),
        dbname=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
    )
    conn.autocommit = True
    return conn

def _to_vector_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

def ensure_tables(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
          id BIGSERIAL PRIMARY KEY,
          doc_id TEXT NOT NULL,
          chunk_id INTEGER NOT NULL,
          content TEXT NOT NULL,
          embedding VECTOR(384) NOT NULL,
          meta JSONB DEFAULT '{}'::jsonb
        );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks (doc_id);")
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_embedding
        ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_meta_filename ON chunks ((meta->>'filename'));")

def extract_text_from_pdf(file_bytes: bytes, min_text_chars: int = 100) -> str:
    """
    Extracts text & tables from PDFs using pdfplumber,
    and uses OCR when very little text is found.
    min_text_chars: threshold below which OCR is triggered.
    """
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            if len(page_text.strip()) < min_text_chars:
                # Use OCR for this page
                img_page = convert_from_bytes(file_bytes, first_page=i+1, last_page=i+1)[0]
                ocr_text = pytesseract.image_to_string(img_page)
                text_parts.append(ocr_text)
            else:
                text_parts.append(page_text)

            # Extract tables 
            tables = page.extract_tables()
            for table in tables:
                rows = [" | ".join(cell or "" for cell in row) for row in table if any(cell for cell in row)]
                if rows:
                    text_parts.append("\n".join(rows))

    return "\n\n".join(text_parts) 

def _clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, buf = [], ""
    for p in paragraphs:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p) if buf else p
        else:
            if buf:
                chunks.append(_clean_text(buf))
            carry = buf[-overlap:] if overlap > 0 and buf else ""
            buf = (carry + " " + p).strip()
            while len(buf) > max_chars:
                chunks.append(_clean_text(buf[:max_chars]))
                buf = _clean_text(buf[max_chars - overlap:])
    if buf:
        chunks.append(_clean_text(buf))
    return [c for c in chunks if c]

def embed_texts(texts: List[str]) -> List[List[float]]:
    return get_embedder().encode(texts, normalize_embeddings=True).tolist()

def upsert_chunks(conn, doc_id: str, chunks: List[str], meta: Dict | None = None):
    meta = meta or {}
    embs = embed_texts(chunks)
    with conn.cursor() as cur:
        for i, (c, e) in enumerate(zip(chunks, embs)):
            vec_lit = _to_vector_literal(e)
            cur.execute(
                """
                INSERT INTO chunks (doc_id, chunk_id, content, embedding, meta)
                VALUES (%s, %s, %s, %s::vector, %s)
                ON CONFLICT DO NOTHING
                """,
                (doc_id, i, c, vec_lit, Json(meta))
            )

def search_chunks(conn, query: str, k: int = 5) -> List[Tuple[str, float]]:
    q_vec_lit = _to_vector_literal(embed_texts([query])[0])
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT content, (embedding <#> %s::vector) AS distance
            FROM chunks
            ORDER BY embedding <#> %s::vector
            LIMIT %s
            """,
            (q_vec_lit, q_vec_lit, k)
        )
        res = cur.fetchall()
    return [(r[0], float(r[1])) for r in res]

def clear_all(conn):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks;")

def new_doc_id() -> str:
    return str(uuid.uuid4())

def ingest_folder(conn, folder_path: str):
    with conn.cursor() as cur:
        for root, dirs, files in os.walk(folder_path):
            for f in files:
                file_path = os.path.join(root, f)
                if not (f.lower().endswith(".pdf") or f.lower().endswith(".txt")):
                    continue

                cur.execute("SELECT 1 FROM chunks WHERE meta->>'filename' = %s LIMIT 1;", (f,))
                if cur.fetchone():
                    print(f"Skipping already ingested file: {f}")
                    continue

                text = None
                if f.lower().endswith(".pdf"):
                    with open(file_path, "rb") as fp:
                        text = extract_text_from_pdf(fp.read())
                elif f.lower().endswith(".txt"):
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as fp:
                        text = fp.read()

                if text:
                    chunks = chunk_text(text)
                    if chunks:
                        upsert_chunks(conn, new_doc_id(), chunks, meta={"filename": f})
                        print(f"Ingested new file: {f}")