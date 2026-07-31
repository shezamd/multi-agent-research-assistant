"""
Writer agent: synthesises all research summaries into a structured answer.

This node runs after all parallel Researchers have completed and their
results have been merged into state['research_results'] by the operator.add
reducer.  The Writer's job is purely editorial — no new research, just
combining and structuring what the Researchers found.

If the Critic later returns feedback, the Writer will be called again with
that feedback included so it can revise its draft.
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableConfig

from src.state import ResearchState


_llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

_SYSTEM_PROMPT = """You are a research writer producing clear, well-structured answers.

Given a research question, a set of research summaries, and optional
revision feedback, write a comprehensive answer that:
- Opens with a direct answer to the question
- Uses short paragraphs or bullet points for readability
- Cites only information present in the summaries
- Incorporates any revision feedback if provided

Keep the answer focused and under 400 words."""


def writer_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Combine research summaries into a structured draft answer.

    Parameters
    ----------
    state : ResearchState
        Reads ``question``, ``research_results``, and ``feedback``.
        ``feedback`` is an empty string on the first pass; the Critic
        populates it if a revision is needed.
    config : RunnableConfig
        Forwarded to the LLM for LangSmith tracing.

    Returns
    -------
    dict
        ``{"draft_answer": str}`` — the synthesised answer text.
    """
    summaries_text = "\n\n".join(
        f"Research finding {i+1}:\n{s}"
        for i, s in enumerate(state["research_results"])
    )

    # Include Critic feedback in the prompt if this is a revision pass
    feedback_section = ""
    if state.get("feedback"):
        feedback_section = f"\n\nRevision feedback from reviewer:\n{state['feedback']}\n\nPlease address all feedback points in your revised answer."

    messages = [
        ("system", _SYSTEM_PROMPT),
        (
            "human",
            f"Research question: {state['question']}\n\n"
            f"Research summaries:\n{summaries_text}"
            f"{feedback_section}",
        ),
    ]

    print(f"\n[Writer] Synthesising {len(state['research_results'])} research findings...")

    response = _llm.invoke(messages, config=config)
    draft = response.content

    print(f"[Writer] Draft complete ({len(draft)} chars)")

    return {"draft_answer": draft}
