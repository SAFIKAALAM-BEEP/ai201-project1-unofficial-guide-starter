"""
config.py — Central configuration for the Hunter College RAG pipeline.
All path and chunking settings live here so every other script imports
from a single source of truth.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_PATH = os.path.join(BASE_DIR, "documents")

# ── Chunking ───────────────────────────────────────────────────────────────
# 400 tokens ≈ 1600 characters (at ~4 chars/token) — matches ingest_and_chunk.py
# 80-token overlap ≈ 320 characters
CHUNK_SIZE = 1600    # characters
CHUNK_OVERLAP = 320  # characters
MIN_CHUNK_LENGTH = 80

# ── Embedding ──────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Retrieval ──────────────────────────────────────────────────────────────
TOP_K = 5

# ── Vector store ───────────────────────────────────────────────────────────
# Must match the collection name used in embed_and_retrieve.py and query.py
CHROMA_COLLECTION = "hunter_guide"
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")