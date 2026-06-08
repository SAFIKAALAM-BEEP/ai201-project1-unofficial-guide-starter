"""
Milestone 2 — Document Ingestion and Chunking
Hunter College Unofficial Guide RAG Pipeline

Spec (from planning.md):
  - Chunk size: 400 tokens (approximated as 400 * 4 = 1600 chars for RecursiveCharacterTextSplitter)
  - Overlap:    80 tokens  (approximated as  80 * 4 =  320 chars)
  - Splitter:   LangChain RecursiveCharacterTextSplitter
  - Metadata per chunk: source_id, source_url, source_date, chunk_index
  - For official/procedural docs, prefer splitting at section headers before falling
    back to paragraph and then character boundaries.
"""

import re
import os
from dataclasses import dataclass, asdict
from typing import List

# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

SOURCES = [
    {
        "id": "01_niche_reviews",
        "url": "https://www.niche.com/colleges/cuny-hunter-college/reviews/",
        "date": "2024",
        "doc_type": "reviews",          # short, self-contained paragraphs
    },
    {
        "id": "02_athenian_dining",
        "url": "https://brie.hunter.cuny.edu/hunterathenian/2023/05/food-for-thought-places-where-hunter-college-students-should-eat-near-campus/",
        "date": "2023-05",
        "doc_type": "article",
    },
    {
        "id": "03_gradreports_reviews",
        "url": "https://www.gradreports.com/colleges/cuny-hunter-college",
        "date": "2024",
        "doc_type": "reviews",
    },
    {
        "id": "04_cuny_commons",
        "url": "https://forstudents.commons.gc.cuny.edu/",
        "date": "2024",
        "doc_type": "procedural",
    },
    {
        "id": "05_student_living_guide",
        "url": "https://www.hunter.cuny.edu/students/campus-life/student-living-guide/",
        "date": "2024",
        "doc_type": "procedural",
    },
    {
        "id": "06_reddit_diversity",
        "url": "https://www.reddit.com/r/HunterCollege/comments/n1klvq/looking_for_pluralism_diversity_recommendations/",
        "date": "2021",
        "doc_type": "reviews",
    },
    {
        "id": "07_student_clubs",
        "url": "https://www.hunter.cuny.edu/students/campus-life/student-clubs/",
        "date": "2024",
        "doc_type": "procedural",
    },
    {
        "id": "08_honors_programs",
        "url": "https://www.hunter.cuny.edu/honors-scholars-programs/",
        "date": "2024",
        "doc_type": "procedural",
    },
    {
        "id": "09_scholarships",
        "url": "https://www.hunter.cuny.edu/students/financial-aid/financial-aid-types/scholarships/",
        "date": "2024",
        "doc_type": "procedural",
    },
    {
        "id": "10_study_abroad",
        "url": "https://www.hunter.cuny.edu/students/opportunities/study-abroad/apply/#cuny",
        "date": "2024",
        "doc_type": "procedural",
    },
]

# ---------------------------------------------------------------------------
# Chunk size constants
# Approximation: 1 token ≈ 4 characters (conservative for mixed English/jargon)
# 400 tokens → 1600 chars | 80 tokens overlap → 320 chars
# ---------------------------------------------------------------------------
CHUNK_SIZE_CHARS = 1600
CHUNK_OVERLAP_CHARS = 320


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    text: str
    source_id: str
    source_url: str
    source_date: str
    chunk_index: int

    def token_estimate(self) -> int:
        """Rough token count: len(text) / 4."""
        return len(self.text) // 4


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

# Section-header pattern used in the official .txt files: "--- HEADING ---"
SECTION_HEADER_RE = re.compile(r"^---\s+.+\s+---$", re.MULTILINE)


def clean_text(raw: str) -> str:
    """
    Light cleaning:
    - Collapse 3+ consecutive blank lines to 2 (preserve paragraph boundaries)
    - Strip trailing whitespace per line
    - Normalize Windows line endings
    """
    text = raw.replace("\r\n", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    # collapse runs of blank lines
    cleaned_lines = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run <= 2:
                cleaned_lines.append(line)
        else:
            blank_run = 0
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def split_by_tokens(text: str, chunk_size: int = CHUNK_SIZE_CHARS,
                    overlap: int = CHUNK_OVERLAP_CHARS) -> List[str]:
    """
    RecursiveCharacterTextSplitter-style splitting without requiring LangChain.
    Tries to split on section headers, then double newlines (paragraphs),
    then single newlines, then spaces, then characters — in that order.

    This mirrors LangChain's RecursiveCharacterTextSplitter with separators:
      ["--- ", "\n\n", "\n", " ", ""]
    """
    separators = ["\n\n", "\n", " ", ""]

    def _split(text: str, seps: List[str]) -> List[str]:
        if not seps or len(text) <= chunk_size:
            return [text] if text.strip() else []

        sep = seps[0]
        rest = seps[1:]

        parts = text.split(sep) if sep else list(text)

        chunks = []
        current = ""

        for part in parts:
            candidate = (current + sep + part) if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current)
                if len(part) > chunk_size:
                    # recursively split the oversized part
                    sub = _split(part, rest)
                    chunks.extend(sub[:-1])
                    current = sub[-1] if sub else ""
                else:
                    current = part

        if current.strip():
            chunks.append(current)

        return chunks

    raw_chunks = _split(text, separators)

    # Apply overlap: each chunk reuses the tail of the previous chunk
    if overlap == 0 or len(raw_chunks) <= 1:
        return raw_chunks

    overlapped = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        prev_tail = overlapped[-1][-overlap:]
        overlapped.append(prev_tail + raw_chunks[i])

    return overlapped


def chunk_document(source: dict, text: str) -> List[Chunk]:
    """
    Chunk a single document into Chunk objects.

    For procedural documents the .txt files use "--- SECTION ---" headers as
    natural split points. We split on those first, then apply token-based
    splitting within each section.
    """
    cleaned = clean_text(text)
    doc_type = source.get("doc_type", "article")

    if doc_type == "procedural":
        # Split on the "--- HEADING ---" markers that appear in official docs
        sections = SECTION_HEADER_RE.split(cleaned)
        headers = SECTION_HEADER_RE.findall(cleaned)

        # Reattach headers to each section body
        labelled = []
        for idx, section in enumerate(sections):
            header = headers[idx - 1] if idx > 0 and idx - 1 < len(headers) else ""
            body = (header + "\n" + section).strip() if header else section.strip()
            if body:
                labelled.append(body)

        raw_texts = []
        for section_body in labelled:
            raw_texts.extend(split_by_tokens(section_body))
    else:
        # Review / article docs: straight recursive splitting
        raw_texts = split_by_tokens(cleaned)

    chunks = []
    for idx, t in enumerate(raw_texts):
        if t.strip():
            chunks.append(Chunk(
                text=t.strip(),
                source_id=source["id"],
                source_url=source["url"],
                source_date=source["date"],
                chunk_index=idx,
            ))
    return chunks


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def load_documents(documents_dir: str) -> List[dict]:
    """Load all .txt files from the given directory, keyed by source id."""
    loaded = []
    for source in SOURCES:
        fname = source["id"] + ".txt"
        fpath = os.path.join(documents_dir, fname)
        if not os.path.exists(fpath):
            print(f"  [WARN] Missing file: {fpath}")
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            raw = f.read()
        loaded.append({"source": source, "raw_text": raw})
        print(f"  Loaded {fname}  ({len(raw):,} chars)")
    return loaded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Resolve documents directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # The user said documents are in a folder called "documents" alongside the script
    # but the uploads are at ../uploads/ — support both
    candidates = [
        os.path.join(script_dir, "documents"),
        os.path.join(script_dir, "..", "uploads"),
        os.path.join(script_dir, "uploads"),
    ]
    docs_dir = None
    for c in candidates:
        if os.path.isdir(c):
            docs_dir = os.path.abspath(c)
            break
    if docs_dir is None:
        raise FileNotFoundError(
            "Could not find documents directory. "
            "Expected a 'documents/' or 'uploads/' folder next to this script."
        )
    print(f"\n=== Loading documents from: {docs_dir} ===\n")

    loaded = load_documents(docs_dir)
    if not loaded:
        raise RuntimeError("No documents loaded — check file names match source ids.")

    print(f"\n=== Chunking {len(loaded)} documents ===")
    print(f"    chunk_size ≈ {CHUNK_SIZE_CHARS} chars (~{CHUNK_SIZE_CHARS//4} tokens)")
    print(f"    overlap    ≈ {CHUNK_OVERLAP_CHARS} chars (~{CHUNK_OVERLAP_CHARS//4} tokens)\n")

    all_chunks: List[Chunk] = []
    per_doc_counts = {}

    for item in loaded:
        chunks = chunk_document(item["source"], item["raw_text"])
        per_doc_counts[item["source"]["id"]] = len(chunks)
        all_chunks.extend(chunks)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("=== Chunks per document ===")
    for doc_id, count in per_doc_counts.items():
        print(f"  {doc_id}: {count} chunks")

    print(f"\n  TOTAL CHUNKS: {len(all_chunks)}")
    avg_tokens = sum(c.token_estimate() for c in all_chunks) / len(all_chunks)
    print(f"  Average chunk length: ~{avg_tokens:.0f} tokens\n")

    # -----------------------------------------------------------------------
    # Print 5 representative chunks (spread across different sources)
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("REPRESENTATIVE CHUNKS (5 selected across different sources)")
    print("=" * 70)

    # Pick: chunk 0 from niche, mid-chunk from study_abroad, chunk from
    # gradreports, reddit, and scholarships
    targets = [
        ("01_niche_reviews",     0),
        ("03_gradreports_reviews", 2),
        ("06_reddit_diversity",  0),
        ("09_scholarships",      1),
        ("10_study_abroad",      2),
    ]

    shown = 0
    for target_id, target_idx in targets:
        candidates_for_target = [
            c for c in all_chunks
            if c.source_id == target_id and c.chunk_index == target_idx
        ]
        if not candidates_for_target:
            # fall back to first chunk from that source
            candidates_for_target = [c for c in all_chunks if c.source_id == target_id]
        if not candidates_for_target:
            continue
        chunk = candidates_for_target[0]
        shown += 1
        print(f"\n--- Chunk {shown} | source: {chunk.source_id} | "
              f"index: {chunk.chunk_index} | ~{chunk.token_estimate()} tokens ---")
        print(chunk.text[:800] + ("..." if len(chunk.text) > 800 else ""))

    # -----------------------------------------------------------------------
    # Range check
    # -----------------------------------------------------------------------
    total = len(all_chunks)
    print("\n" + "=" * 70)
    if total < 50:
        print(f"WARNING: Only {total} chunks — chunks may be too large. "
              "Consider reducing chunk_size.")
    elif total > 2000:
        print(f"WARNING: {total} chunks — chunks may be too small. "
              "Consider increasing chunk_size.")
    else:
        print(f"OK: {total} chunks is in the recommended range (50–2000).")
    print("=" * 70)

    return all_chunks


if __name__ == "__main__":
    chunks = main()
