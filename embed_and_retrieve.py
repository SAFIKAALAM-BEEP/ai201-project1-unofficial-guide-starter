"""
Milestone 3 — Embedding + Vector Store + Retrieval
Hunter College Unofficial Guide RAG Pipeline

Architecture (from planning.md):
  [2] CHUNKING  →  [3] EMBEDDING + VECTOR STORE  →  [4] RETRIEVAL

Embedding model : sentence-transformers/all-MiniLM-L6-v2  (local, no API key)
Vector store    : ChromaDB  (local persistent collection)
Top-k           : 5

NOTE on chunk size vs. model context:
  all-MiniLM-L6-v2 has a 256-token context window. Chunks that exceed 256 tokens
  are silently truncated by the model before embedding. Our average chunk is ~325
  tokens, so longer chunks lose their tail. This is acceptable for a prototype — the
  most semantically dense part of a passage is usually near the beginning — but a
  production system should either (a) reduce chunk size to ≤200 tokens or (b) switch
  to all-mpnet-base-v2 (512-token window). Noted in planning.md.
"""

import os
import sys

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR  = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "uploads"))
CHROMA_DIR   = os.path.join(SCRIPT_DIR, "chroma_db")   # persisted vector store
COLLECTION_NAME = "hunter_guide"

# ---------------------------------------------------------------------------
# Imports — fail fast with helpful messages
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit("Missing dependency: pip install sentence-transformers")

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    sys.exit("Missing dependency: pip install chromadb")

# Re-use ingestion logic from milestone 2
sys.path.insert(0, SCRIPT_DIR)
from ingest_and_chunk import SOURCES, load_documents, chunk_document, Chunk


# ---------------------------------------------------------------------------
# Step 1 — Load and chunk all documents
# ---------------------------------------------------------------------------

def build_chunks():
    """Run the full ingestion + chunking pipeline and return all Chunk objects."""
    # Find the documents directory (uploads or documents/)
    candidates = [
        UPLOADS_DIR,
        os.path.join(SCRIPT_DIR, "documents"),
        os.path.join(SCRIPT_DIR, "uploads"),
    ]
    docs_dir = next((c for c in candidates if os.path.isdir(c)), None)
    if docs_dir is None:
        raise FileNotFoundError("Cannot find documents directory.")

    print(f"Loading documents from: {docs_dir}")
    loaded = load_documents(docs_dir)
    all_chunks = []
    for item in loaded:
        all_chunks.extend(chunk_document(item["source"], item["raw_text"]))
    print(f"  → {len(all_chunks)} chunks ready for embedding\n")
    return all_chunks


# ---------------------------------------------------------------------------
# Step 2 — Embed + upsert into ChromaDB
# ---------------------------------------------------------------------------

def build_vector_store(chunks: list[Chunk], force_rebuild: bool = False):
    """
    Embed every chunk with all-MiniLM-L6-v2 and upsert into a persistent
    ChromaDB collection.

    ChromaDB API notes:
      - chromadb.PersistentClient(path)  opens (or creates) a local SQLite-backed
        store at `path`. Data survives across Python sessions.
      - collection.upsert(ids, embeddings, documents, metadatas)
          ids        : unique string IDs (we use "<source_id>_chunk_<index>")
          embeddings : list of float vectors (one per chunk)
          documents  : the raw chunk text (stored for retrieval)
          metadatas  : list of dicts — arbitrary key/value pairs stored alongside
                       each vector; retrieved with the chunk so we can cite sources
      - collection.query(query_embeddings, n_results)
          Returns a dict with keys: ids, documents, metadatas, distances
          distances  : L2 (Euclidean) distances — lower = more similar
    """
    print("=== Setting up ChromaDB ===")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # If collection already exists and we're not forcing a rebuild, reuse it
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing and not force_rebuild:
        collection = client.get_collection(COLLECTION_NAME)
        print(f"  Reusing existing collection '{COLLECTION_NAME}' "
              f"({collection.count()} vectors)\n")
        return collection

    # Delete stale collection before rebuilding
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # use cosine similarity (returns distance 0–2; lower = more similar)
    )
    print(f"  Created new collection '{COLLECTION_NAME}'")

    # Load embedding model
    print("\n=== Loading embedding model: all-MiniLM-L6-v2 ===")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("  Model loaded.\n")

    # Embed all chunk texts in one batched call (fast)
    print(f"=== Embedding {len(chunks)} chunks ===")
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    print(f"  Embedding shape: {embeddings.shape}\n")  # (n_chunks, 384)

    # Build parallel lists required by ChromaDB upsert
    ids        = [f"{c.source_id}_chunk_{c.chunk_index}" for c in chunks]
    documents  = texts
    metadatas  = [
        {
            "source_id":    c.source_id,
            "source_url":   c.source_url,
            "source_date":  c.source_date,
            "chunk_index":  c.chunk_index,
        }
        for c in chunks
    ]

    # Upsert: insert or update — safe to run multiple times
    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
    )
    print(f"=== Upserted {collection.count()} vectors into ChromaDB ===\n")
    return collection


# ---------------------------------------------------------------------------
# Step 3 — Retrieval function
# ---------------------------------------------------------------------------

def retrieve(query: str, collection, model: SentenceTransformer,
             top_k: int = 5) -> list[dict]:
    """
    Embed `query` and return the top_k most similar chunks from ChromaDB.

    Returns a list of dicts, each with:
      text        : chunk text
      source_id   : document filename stem
      source_url  : original URL
      source_date : publication year/date
      chunk_index : position in source document
      distance    : cosine distance (0 = identical, 2 = opposite; good results < 0.5)
    """
    query_embedding = model.encode([query])

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # ChromaDB returns nested lists (one per query); we sent one query so index [0]
    chunks_out = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks_out.append({
            "text":        doc,
            "source_id":   meta["source_id"],
            "source_url":  meta["source_url"],
            "source_date": meta["source_date"],
            "chunk_index": meta["chunk_index"],
            "distance":    round(dist, 4),
        })
    return chunks_out


# ---------------------------------------------------------------------------
# Step 4 — Test retrieval with 3 eval-plan queries
# ---------------------------------------------------------------------------

def test_retrieval(collection, model):
    """
    Run 3 of the 5 evaluation-plan queries and print retrieved chunks + distances.

    Evaluation plan queries (from planning.md):
      Q1: "What discount does Two Wheels Vietnamese restaurant offer to Hunter students?"
          Expected: 15% discount with Hunter Student ID when you mention Hunter VSA
      Q3: "What financial benefits does the Macaulay Honors College scholarship include?"
          Expected: Full tuition, study grants (soph–senior), free laptop
      Q5: "What is the deposit amount required after being accepted to a Hunter study abroad program?"
          Expected: $350 money order or certified check made out to Hunter College
    """
    test_queries = [
        {
            "id": "Q1",
            "query": "What discount does Two Wheels Vietnamese restaurant offer to Hunter students?",
            "expected": "15% discount with Hunter Student ID when you mention Hunter VSA",
        },
        {
            "id": "Q3",
            "query": "What financial benefits does the Macaulay Honors College scholarship include?",
            "expected": "Full tuition, study grants from sophomore to senior year, free laptop",
        },
        {
            "id": "Q5",
            "query": "What is the deposit amount required after being accepted to a Hunter study abroad program?",
            "expected": "$350 money order or certified check made out to Hunter College",
        },
    ]

    print("=" * 70)
    print("RETRIEVAL TEST — 3 Evaluation Plan Queries")
    print("=" * 70)

    for q in test_queries:
        print(f"\n{'─' * 70}")
        print(f"[{q['id']}] {q['query']}")
        print(f"Expected: {q['expected']}")
        print(f"{'─' * 70}")

        results = retrieve(q["query"], collection, model, top_k=5)

        for rank, r in enumerate(results, 1):
            relevant_flag = "✓" if r["distance"] < 0.5 else "△"
            print(f"\n  Rank {rank} | dist={r['distance']} {relevant_flag} | "
                  f"{r['source_id']} (chunk {r['chunk_index']})")
            # Print up to 400 chars of chunk text
            preview = r["text"].replace("\n", " ")[:400]
            print(f"  \"{preview}{'...' if len(r['text']) > 400 else ''}\"")

    print(f"\n{'=' * 70}")
    print("Distance interpretation (cosine, hnsw:space=cosine):")
    print("  < 0.3  → highly relevant")
    print("  0.3–0.5 → relevant")
    print("  > 0.5  → likely off-topic")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Build chunks
    chunks = build_chunks()

    # Build / load vector store
    collection = build_vector_store(chunks, force_rebuild=True)

    # Load model for query-time embedding
    print("=== Loading model for retrieval ===")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("  Ready.\n")

    # Run retrieval tests
    test_retrieval(collection, model)


if __name__ == "__main__":
    main()
