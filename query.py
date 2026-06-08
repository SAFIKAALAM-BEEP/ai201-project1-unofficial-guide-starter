"""
Milestone 4 — Grounded Generation + Source Attribution
Hunter College Unofficial Guide RAG Pipeline

Pipeline stage covered here (from planning.md architecture):
  [4] RETRIEVAL  →  [5] GENERATION

LLM     : Groq  llama-3.3-70b-versatile  (free-tier, OpenAI-compatible)
Grounding: System prompt PROHIBITS answers outside retrieved context.
           Sources are appended PROGRAMMATICALLY after generation — the
           LLM cannot omit or fabricate citations.
"""

import os
import sys
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env (expects GROQ_API_KEY=...)
# ---------------------------------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    sys.exit(
        "Missing GROQ_API_KEY. Create a free account at https://console.groq.com, "
        "generate a key, and add it to a .env file:\n  GROQ_API_KEY=gsk_..."
    )

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
try:
    from groq import Groq
except ImportError:
    sys.exit("Missing dependency: pip install groq")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit("Missing dependency: pip install sentence-transformers")

try:
    import chromadb
except ImportError:
    sys.exit("Missing dependency: pip install chromadb")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR      = os.path.join(SCRIPT_DIR, "chroma_db")
COLLECTION_NAME = "hunter_guide"
TOP_K           = 5
MODEL           = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# Lazy singletons — loaded once, reused across calls
# ---------------------------------------------------------------------------
_model      = None
_collection = None
_client_llm = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        if not os.path.isdir(CHROMA_DIR):
            raise FileNotFoundError(
                f"ChromaDB not found at {CHROMA_DIR}. "
                "Run embed_and_retrieve.py first to build the vector store."
            )
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def _get_llm():
    global _client_llm
    if _client_llm is None:
        _client_llm = Groq(api_key=GROQ_API_KEY)
    return _client_llm


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Return top_k chunks from ChromaDB most similar to query."""
    model      = _get_model()
    collection = _get_collection()

    query_vec = model.encode([query])
    results   = collection.query(
        query_embeddings=query_vec.tolist(),
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":        doc,
            "source_id":   meta["source_id"],
            "source_url":  meta["source_url"],
            "source_date": meta["source_date"],
            "chunk_index": meta["chunk_index"],
            "distance":    round(dist, 4),
        })
    return chunks


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

# Human-readable labels for source IDs shown in citations
SOURCE_LABELS = {
    "01_niche_reviews":       "Niche.com — Hunter Student Reviews",
    "02_athenian_dining":     "The Athenian — Dining Near Campus (2023)",
    "03_gradreports_reviews": "GradReports — Hunter Graduate Reviews",
    "04_cuny_commons":        "CUNY Commons — Student Resources Portal",
    "05_student_living_guide":"Hunter Official — Student Living Guide",
    "06_reddit_diversity":    "Reddit r/HunterCollege — Diversity Thread",
    "07_student_clubs":       "Hunter Official — Student Clubs & Organizations",
    "08_honors_programs":     "Hunter Official — Honors & Scholars Programs",
    "09_scholarships":        "Hunter Official — Scholarships & Financial Aid",
    "10_study_abroad":        "Hunter Official — Study Abroad Application Guide",
}


SYSTEM_PROMPT = """\
You are a helpful guide for CUNY Hunter College students.

STRICT RULES — you must follow these without exception:
1. Answer ONLY using information explicitly stated in the CONTEXT DOCUMENTS below.
2. Do NOT draw on your training data, general knowledge, or information not present in the documents.
3. If the context documents do not contain enough information to answer the question, respond with exactly: "I don't have enough information on that in the documents I was given."
4. Do NOT speculate, infer, or fill gaps with plausible-sounding facts.
5. Keep your answer concise and direct. Quote or paraphrase the documents; do not embellish.
6. Do NOT include a source list in your response — sources are added separately by the system.
"""


def build_user_message(query: str, chunks: list[dict]) -> str:
    """
    Construct the user turn: numbered context documents + the question.
    Numbering the documents lets the model reference them naturally in its answer
    without us relying on it to reproduce URLs correctly.
    """
    context_block = "\n\n".join(
        f"[Document {i+1}] (Source: {SOURCE_LABELS.get(c['source_id'], c['source_id'])}, "
        f"Date: {c['source_date']})\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    return (
        f"CONTEXT DOCUMENTS:\n\n{context_block}\n\n"
        f"---\n\nQUESTION: {query}"
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(query: str, chunks: list[dict]) -> str:
    """Call the LLM with the grounding system prompt and retrieved context."""
    llm          = _get_llm()
    user_message = build_user_message(query, chunks)

    response = llm.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": user_message},
        ],
        temperature=0.0,   # deterministic — no creative hallucination
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Source attribution — programmatic, not LLM-generated
# ---------------------------------------------------------------------------

def format_sources(chunks: list[dict]) -> list[str]:
    """
    Build a deduplicated, human-readable source list from the retrieved chunks.
    This is constructed from chunk metadata BEFORE the LLM is called — the LLM
    cannot omit or fabricate entries.  The source list is appended to the answer
    by the caller, not generated by the model.
    """
    seen   = set()
    labels = []
    for c in chunks:
        sid = c["source_id"]
        if sid not in seen:
            seen.add(sid)
            label = SOURCE_LABELS.get(sid, sid)
            labels.append(f"{label} — {c['source_url']}")
    return labels


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def ask(question: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG call.

    Returns:
      {
        "answer":  str,          # grounded LLM answer
        "sources": list[str],    # programmatically derived source labels + URLs
        "chunks":  list[dict],   # raw retrieved chunks (for debugging / eval)
      }
    """
    chunks  = retrieve(question, top_k=top_k)
    answer  = generate(question, chunks)
    sources = format_sources(chunks)
    return {"answer": answer, "sources": sources, "chunks": chunks}


# ---------------------------------------------------------------------------
# Quick CLI test — run with: python query.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TEST_QUESTIONS = [
        # Q1 — should hit 02_athenian_dining
        "What discount does Two Wheels Vietnamese restaurant offer to Hunter students?",
        # Q3 — should hit 08_honors_programs / 09_scholarships
        "What financial benefits does the Macaulay Honors College scholarship include?",
        # Q5 — should hit 10_study_abroad
        "What is the deposit amount required after being accepted to a Hunter study abroad program?",
        # OUT-OF-SCOPE — documents say nothing about this; system must admit it
        "What is the cost of parking near Hunter College's campus?",
    ]

    for q in TEST_QUESTIONS:
        print("\n" + "=" * 70)
        print(f"Q: {q}")
        print("=" * 70)
        result = ask(q)
        print(f"\nAnswer:\n{result['answer']}")
        print("\nSources (programmatic):")
        for s in result["sources"]:
            print(f"  • {s}")
        print(f"\n[Top retrieved chunk distances: "
              f"{[c['distance'] for c in result['chunks'][:3]]}]")
