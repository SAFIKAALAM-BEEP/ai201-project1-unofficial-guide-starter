# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

<!-- What domain did you choose? Why is this knowledge valuable and hard to find through official channels? -->

---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

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

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size:** 400 tokens

**Overlap:** 80 tokens (20%)

**Reasoning:** The corpus mixes two very different document types. Review documents (Niche, GradReports, Reddit) consist of short, self-contained paragraphs — individual student opinions that average 100–200 words each. Official procedural documents (study abroad, scholarships, student living) contain longer flowing text where a single policy spans several paragraphs. A 400-token chunk is large enough to capture a complete review or a coherent policy section, but small enough that a retrieved chunk isn't diluted with off-topic content. The 80-token overlap ensures that no key fact is severed at a chunk boundary — for example, a review that mentions both professor quality and financial aid in the same breath won't have those observations split across non-overlapping chunks. For the review-heavy sources, most chunks will naturally fall well under 400 tokens and the chunker will just use paragraph breaks; for the procedural sources, the 400-token ceiling keeps retrieved context focused.

---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:** all-MiniLM-L6-v2 via sentence-transformers

**Top-k:** 5

**Production tradeoff reflection:** Context length since all-MiniLM-L6-v2 has a 256-token limit; longer chunks would get silently truncated. Domain specificity since general-purpose embeddings may underperform on CUNY-specific jargon (ePermit, CUNYfirst, TAP, Macaulay).

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | What discount does Two Wheels Vietnamese restaurant offer to Hunter students? | 15% discount with Hunter Student ID when you mention Hunter VSA |
| 2 | What GPA is required to apply for a semester-long exchange program at Hunter? | Minimum 3.0 GPA and at least 60 completed credits at time of application |
| 3 | What financial benefits does the Macaulay Honors College scholarship include? | Full tuition each year, study grants from sophomore through senior year for study abroad or unpaid internships, and a free laptop computer |
| 4 | What do students most commonly criticize about Hunter's administration? | Poor communication, difficulty getting answers from advising and financial aid offices, inconsistent support, and feeling unsupported navigating enrollment and bureaucratic processes |
| 5 | What is the deposit amount required after being accepted to a Hunter study abroad program? | $350 money order or certified check made out to Hunter College |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1. 

2. 

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

**Milestone 3 — Ingestion and chunking:** Claude 



**Milestone 4 — Embedding and retrieval:** Claude

**Milestone 5 — Generation and interface:**
