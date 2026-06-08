"""
Milestone 4 — Gradio Web Interface
Hunter College Unofficial Guide

Run with: python app.py
Then open:  http://localhost:7860
"""

import gradio as gr
from query import ask


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handle_query(question: str):
    """
    Called by Gradio on every button click or Enter keypress.
    Returns (answer_text, sources_text) — one string per output box.
    """
    question = question.strip()
    if not question:
        return "Please enter a question.", ""

    result  = ask(question)
    answer  = result["answer"]
    sources = "\n".join(f"• {s}" for s in result["sources"])

    # Append debug distances in a collapsible-friendly way (plain text footer)
    distances = [c["distance"] for c in result["chunks"]]
    debug_line = f"\n\n[Retrieval distances: {distances}]"

    return answer + debug_line, sources


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

DESCRIPTION = """
## 🗺️ Hunter College Unofficial Guide
Ask any question about Hunter College student life — dining, housing, scholarships,
study abroad, clubs, honors programs, and more.

Answers are grounded in real student reviews and official Hunter documents.
"""

EXAMPLES = [
    ["What discount does Two Wheels Vietnamese restaurant offer to Hunter students?"],
    ["What financial benefits does the Macaulay Honors College scholarship include?"],
    ["What is the deposit amount required after being accepted to a Hunter study abroad program?"],
    ["What do students say about Hunter's advising and financial aid offices?"],
    ["What GPA do I need to apply for a semester-long exchange program?"],
]

with gr.Blocks(title="Hunter College Unofficial Guide") as demo:

    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=3):
            inp = gr.Textbox(
                label="Your question",
                placeholder="e.g. What clubs are available at Hunter?",
                lines=2,
            )
        with gr.Column(scale=1):
            btn = gr.Button("Ask", variant="primary")

    with gr.Row():
        answer_box = gr.Textbox(
            label="Answer",
            lines=10,
        )

    with gr.Row():
        sources_box = gr.Textbox(
            label="Retrieved from",
            lines=4,
        )

    gr.Examples(
        examples=EXAMPLES,
        inputs=inp,
        label="Try one of these:",
    )

    # Wire up both click and Enter-to-submit
    btn.click(handle_query, inputs=inp, outputs=[answer_box, sources_box])
    inp.submit(handle_query, inputs=inp, outputs=[answer_box, sources_box])


if __name__ == "__main__":
    demo.launch()