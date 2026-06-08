# The Unofficial Guide — Project 1

> **How to use this template:**
> Complete each section *after* you've built and tested the corresponding part of your system.
> Do not write placeholder text — if a section isn't done yet, leave it blank and come back.
> Every section below is required for submission. One-liners will not receive full credit.

---

## Domain

<!-- What topic or category of knowledge does your system cover?
     Why is this knowledge valuable, and why is it hard to find through official channels?
     Example: "Student reviews of CS professors at [university] — useful because official
     course descriptions don't reflect teaching style, exam difficulty, or workload." -->

     This system covers student life at CUNY Hunter College — spanning dining near campus, scholarships and financial aid, study abroad, honors programs, student clubs, and peer reviews of the overall college experience. This knowledge is valuable because it answers the practical, experience-based questions that prospective and current students actually need: which restaurants offer discounts, what the Macaulay scholarship actually covers, what students think of the advising office. Official Hunter channels (the website, course catalogs, advising PDFs) provide policy text but strip out student perspective entirely, and student reviews are scattered across Niche, GradReports, Reddit, and a student newspaper — no single place aggregates them with the official procedural detail.

---

## Document Sources

<!-- List every source you collected documents from.
     Be specific: include URLs, subreddit names, forum thread titles, or file names.
     Aim for variety — sources that together cover different subtopics or perspectives. -->

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | Niche | 2,500+ student reviews covering academics, social life, admin, professors, and overall experience | https://www.niche.com/colleges/cuny-hunter-college/reviews/ |
| 2 | The Athenian | Article by Hunter students listing 5 affordable food spots near campus with specific menu recommendations and student discounts | https://brie.hunter.cuny.edu/hunterathenian/2023/05/food-for-thought-places-where-hunter-college-students-should-eat-near-campus/ |
| 3 | GradReports | 217 degree-specific reviews from graduates rating career impact, professor quality, and program strengths/weaknesses by major | https://www.gradreports.com/colleges/cuny-hunter-college |
| 4 | CUNY Commons | CUNY-wide student resource portal covering advising, clubs, academic support, and community programs | https://forstudents.commons.gc.cuny.edu/ |
| 5 | Hunter Official — Student Living Guide | Official guide to housing, neighborhood resources, commuter tips, and campus facilities | https://www.hunter.cuny.edu/students/campus-life/student-living-guide/ |
| 6 | Reddit r/HunterCollege — Diversity | Student discussion thread about pluralism, diversity courses, and campus culture recommendations | https://www.reddit.com/r/HunterCollege/comments/n1klvq/looking_for_pluralism_diversity_recommendations/|
| 7 | Hunter Official — Student Clubs & Organizations | Full listing of 100+ clubs by category (academic, cultural, advocacy, recreational, Greek life) | https://www.hunter.cuny.edu/students/campus-life/student-clubs/ |
| 8 | Hunter Official — Honors & Scholars Programs | Overview of Macaulay Honors College, Thomas Hunter Scholars, and other competitive scholar tracks with eligibility and benefits | https://www.hunter.cuny.edu/honors-scholars-programs/ |
| 9 | Hunter Official — Scholarships & Financial Aid | Details on Macaulay full-tuition scholarship, Guttman Transfer Scholarships, TAP, FAFSA, and other funding sources | https://www.hunter.cuny.edu/students/financial-aid/financial-aid-types/scholarships/ |
| 10 | Hunter Official — Study Abroad| Step-by-step application process, program types (exchange, CUNY, external), eligibility requirements, deadlines, and deposit policies | https://www.hunter.cuny.edu/students/opportunities/study-abroad/apply/#cuny |

---

## Chunking Strategy

<!-- Describe your chunking approach with enough specificity that someone else could reproduce it.
     Include:
     - Chunk size (characters or tokens) and why that size fits your documents
     - Overlap size and why (or why not) you used overlap
     - Any preprocessing you did before chunking (e.g., stripping HTML, removing headers)
     - What your final chunk count was across all documents -->

**Chunk size:** 400 tokens, approximated as 1,600 characters (using the common 1 token ≈ 4 characters conversion)

**Overlap:** 80 tokens ≈ 320 characters (20% overlap)

**Why these choices fit your documents:** The corpus mixes two structurally different document types. Review documents (Niche, GradReports, Reddit) consist of short, self-contained paragraphs — individual student opinions averaging 100–200 words. A 400-token ceiling is large enough to hold a complete review without cutting it mid-sentence, but small enough that a retrieved chunk stays focused on one opinion rather than mixing multiple conflicting reviews. Official procedural documents (study abroad, scholarships, student living) contain longer sequential text where a single policy spans several paragraphs; the 80-token overlap ensures that facts split across paragraph boundaries (e.g., an eligibility requirement mentioned at the end of one paragraph and its exception noted at the start of the next) still appear together in at least one chunk.

**Final chunk count:** 62

---

## Embedding Model

<!-- Name the embedding model you used and explain your choice.
     Then answer: if you were deploying this system for real users and cost wasn't a constraint,
     what tradeoffs would you weigh in choosing a different model?
     Consider: context length limits, multilingual support, accuracy on domain-specific text,
     latency, and local vs. API-hosted. -->

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers` (local, no API key required). Chosen because it is fast, free to run locally, and widely benchmarked on semantic similarity tasks. It produces 384-dimensional vectors and runs on CPU without noticeable latency on a corpus of this size.

**Production tradeoff reflection:** The main limitation of `all-MiniLM-L6-v2` in production would be its 256-token context window. Our chunks average ~325 tokens, meaning the tail of each chunk is silently truncated before embedding — the model never sees the last ~70 tokens of a chunk. For a prototype this is acceptable because the most semantically dense part of a passage usually appears early, but it would cause retrieval failures on chunks where the key fact is near the end

---

## Grounded Generation

<!-- Explain how your system enforces grounding — how does it prevent the LLM from answering
     beyond the retrieved documents?
     Describe both your system prompt (what instruction you gave the model) and any structural
     choices (e.g., how you formatted the context, whether you filtered low-relevance chunks).
     Do not just say "I told it to use the documents" — show the actual instruction or explain
     the mechanism. -->

**System prompt grounding instruction:** The system prompt explicitly prohibits the model from drawing on training knowledge:

> "Answer ONLY using information explicitly stated in the CONTEXT DOCUMENTS below. Do NOT draw on your training data, general knowledge, or information not present in the documents. If the context documents do not contain enough information to answer the question, respond with exactly: 'I don't have enough information on that in the documents I was given.' Do NOT speculate, infer, or fill gaps with plausible-sounding facts."
 
Temperature is set to `0.0` to prevent creative variation. Each retrieved chunk is passed to the model as a numbered `[Document N]` block with its source label and date, so the model can reference specific documents without needing to reproduce URLs.

**How source attribution is surfaced in the response:** Source attribution is constructed **programmatically from chunk metadata before the LLM is called** — it is not generated by the model. After retrieval, `format_sources()` in `query.py` reads the `source_id` field from each returned chunk, maps it to a human-readable label via a hard-coded dictionary, deduplicates by source, and appends the source list to the answer in the UI. This guarantees that sources are always shown and always correspond to the actual retrieved documents, regardless of whether the model mentions them.

---

## Evaluation Report

<!-- Run your 5 test questions from planning.md through your system and record the results.
     Be honest — a partially accurate or inaccurate result that you explain well is more
     valuable than a suspiciously perfect result. -->

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | What discount does Two Wheels Vietnamese restaurant offer to Hunter students? | 15% discount with Hunter Student ID when mentioning Hunter VSA | "Two Wheels offers a 15% discount with a Hunter Student ID when mentioning Hunter VSA." — correct and specific | Partially relevant — correct chunk from The Athenian retrieved at rank 1; Student Living Guide also retrieved but irrelevant to this query | Accurate |
| 2 | What financial benefits does the Macaulay Honors College scholarship include? | Full tuition each year, study grants from sophomore through senior year, free laptop computer | Correct on all three: full tuition, study grants (soph–senior) for study abroad/internships/research, free laptop | Relevant — Scholarships and Honors Programs documents both retrieved and both contain the relevant information | Accurate |
| 3 | What do students most commonly criticize about Hunter's administration? | Poor communication, difficulty getting answers from advising and financial aid offices, inconsistent support | Retrieved specific named quotes from GradReports (Madeline Platt, James Ramsawmy) and a Niche senior review; accurately surfaced the advising and financial aid themes | Relevant — GradReports and Niche chunks correctly retrieved | Accurate |
| 4 | What is the deposit amount required after being accepted to a Hunter study abroad program? | $350 money order or certified check made out to Hunter College | "$350, submitted as a Money Order or Certified Check made out to HUNTER COLLEGE" — exact match | Relevant — Study Abroad Application Guide retrieved as the top source | Accurate |
| 5 | What GPA do I need to apply for a semester-long exchange program? | Minimum 3.0 GPA and at least 60 completed credits at time of application | Returned only the 3.0 GPA requirement; the 60-credit requirement was not included in the response | Partially relevant — Study Abroad guide retrieved correctly, but Honors Programs and Scholarships also retrieved and are irrelevant to this query | Partially accurate — GPA is correct but the 60-credit prerequisite was omitted |

**Retrieval quality:** Partially relevant
**Response accuracy:** Partially accurate

---

## Failure Case Analysis

<!-- Identify at least one question where retrieval or generation did not work as expected.
     Write a specific explanation of *why* it failed, tied to a part of the pipeline.

     "The answer was wrong" is not an explanation.

     "The relevant information was split across a chunk boundary, so retrieval returned
     only half the context — the model didn't have enough to answer correctly" is an explanation.

     "The embedding model treated the professor's nickname as out-of-vocabulary and returned
     results from an unrelated review" is an explanation. -->

**Question that failed:** "What GPA do I need to apply for a semester-long exchange program?" (Q5)
 
**What the system returned:** The minimum 3.0 GPA requirement — correct — but it omitted the second eligibility condition: at least 60 completed credits at the time of application. The retrieved source list also included Hunter Official — Honors & Scholars Programs and Hunter Official — Scholarships & Financial Aid, neither of which is relevant to exchange program eligibility.
 
**Root cause (tied to a specific pipeline stage):** Two pipeline stages contributed. First, at the **chunking stage**, the study abroad document's eligibility section lists the GPA requirement and the credit requirement in close proximity — but if the 3.0 GPA sentence fell near the end of one chunk and the 60-credit requirement began the next, the overlap window (320 characters ≈ 80 tokens) may not have been wide enough to bridge them. The retrieved chunk then contained only the GPA half of the eligibility criteria. Second, at the **retrieval stage**, `all-MiniLM-L6-v2` found strong semantic similarity between "GPA to apply for a semester-long program" and chunks from the Honors and Scholarships documents, which also discuss GPA thresholds for academic programs. The model cannot distinguish between "GPA for an exchange program" and "GPA for an honors scholarship" because both surface as "GPA + program + eligibility" in the embedding space. With top-k=5, two of the five retrieved slots were consumed by these off-topic chunks, leaving less context about the actual exchange program requirements for the LLM to draw from.
 
**What you would change to fix it:** Two targeted fixes. For the chunking issue: when processing official procedural documents, split at numbered list items as a separator level above paragraph breaks, so eligibility criteria that appear as a numbered list are never separated across chunks. For the retrieval issue: increase top-k to 7 for queries containing "GPA," "requirement," or "eligibility" (or apply a post-retrieval filter that drops chunks whose source distance exceeds 0.4) so that the relevant study abroad chunk is not crowded out by semantically adjacent but topically wrong results.

---

## Spec Reflection

<!-- Reflect on how planning.md shaped your implementation.
     Answer both questions with at least 2–3 sentences each. -->

**One way the spec helped during implementation:** The distinction between `doc_type: "reviews"` and `doc_type: "procedural"` in `planning.md` directly shaped the chunking logic in `ingest_and_chunk.py`. Because the spec identified upfront that official documents have logical section headers while review documents are flat paragraphs, the implementation could apply two different split strategies in the same function — splitting procedural documents on `--- SECTION ---` markers first, then falling back to paragraph boundaries, while applying straight recursive splitting to review corpora. Without the doc_type distinction in the spec, this would have been discovered only after inspecting bad chunks in the output, costing debugging time.
 
**One way the implementation diverged from the spec, and why:** The spec called for using LangChain's `RecursiveCharacterTextSplitter` as the chunking tool. The implementation instead reimplements the same recursive logic from scratch in pure Python, without importing LangChain. This change was made because LangChain's character-based splitter approximates token counts using character length, and the custom implementation could apply the same approximation more transparently — making it easier to tune and inspect without digging through LangChain internals. The behavior is functionally identical (same separator priority order, same overlap logic), but the dependency footprint is smaller and the chunk boundaries are easier to trace during debugging.

---

## AI Usage

<!-- Describe at least 2 specific instances where you used an AI tool during this project.
     For each: what did you give the AI as input, what did it produce, and what did you
     change, override, or direct differently?

     "I used Claude to help me code" is not sufficient.
     "I gave Claude my Chunking Strategy section from planning.md and asked it to implement
     chunk_text(). It returned a function using a fixed character split. I overrode the
     chunk size from 500 to 200 because my documents are short reviews, not long guides." -->

**Instance 1**

- *What I gave the AI:* The Documents table and Chunking Strategy section from `planning.md`, specifying 400-token chunks (~1,600 chars), 80-token overlap (~320 chars), the preference for header-based splits on procedural documents, and the desired output schema: `{text, source_id, source_url, source_date, chunk_index}`.
- *What it produced:* A complete `ingest_and_chunk.py` with a `Chunk` dataclass, a `clean_text()` function, a recursive character splitter, and a `chunk_document()` function that branches on `doc_type`. It also produced a `main()` that loaded all 10 documents, printed per-source chunk counts, and displayed 5 representative chunks.
- *What I changed or overrode:* The AI initially used LangChain's `RecursiveCharacterTextSplitter` as specified in the planning doc. I directed it to replace that with a dependency-free recursive implementation using the same separator hierarchy, because LangChain's pydantic version requirements conflicted with ChromaDB on the development machine. I also added the range check (50–2,000 chunks) and the average token-length printout, which the AI had not included.

**Instance 2 — Grounded generation and Gradio interface (Milestone 4)**
 
- *What I gave the AI:* The Architecture diagram from `planning.md` (five pipeline stages with tool labels), the grounding requirement (answers from retrieved context only, no training data), the output format (answer + programmatic source list), and the 5 evaluation questions as test cases.
- *What it produced:* `query.py` with the `SYSTEM_PROMPT`, `build_user_message()`, `generate()`, `format_sources()`, and `ask()` functions, plus `app.py` with the full Gradio `gr.Blocks` layout, example queries, and the `handle_query` handler wired to both button click and Enter.
- *What I changed or overrode:* The AI's initial system prompt said "try to use only the provided documents" — a suggestion, not a prohibition. I rewrote it to say "Answer ONLY using information explicitly stated in the CONTEXT DOCUMENTS" with an explicit fallback phrase ("I don't have enough information on that") and the instruction not to speculate or fill gaps. I also moved source attribution out of the LLM's response entirely and into `format_sources()`, which the AI had left as an instruction to the model rather than a programmatic guarantee.
