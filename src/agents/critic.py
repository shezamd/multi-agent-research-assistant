"""
Critic agent: reviews the Writer's draft and either approves it or requests
a revision with specific feedback.

This node drives the only conditional edge in the graph:

    writer_node → critic_node → (approved?) → END
                                            ↘ (rejected?) → writer_node

The conditional routing logic lives in src/graph.py, not here.  The Critic's
only job is to populate state correctly:
  - Approved  → set final_answer = draft_answer, feedback = ""
  - Rejected  → set feedback = "...", leave final_answer = ""

A max_revisions guard is applied in the graph to prevent infinite loops.
"""

from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableConfig

from src.state import ResearchState


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class CriticOutput(BaseModel):
    """Structured verdict from the Critic.

    Parameters
    ----------
    approved : bool
        True if the answer is satisfactory and ready to return to the user.
    feedback : str
        Specific, actionable revision notes. Must be a non-empty string
        when approved=False, and an empty string when approved=True.
    """

    approved: bool = Field(description="True if the answer is satisfactory.")
    feedback: str = Field(
        description=(
            "Specific revision notes for the Writer if approved=False. "
            "Empty string if approved=True."
        )
    )


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

_llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
_critic_chain = _llm.with_structured_output(CriticOutput)

_SYSTEM_PROMPT = """You are a critical editor reviewing a research answer.

Approve the answer if it:
- Directly addresses the original question
- Is well-structured and readable
- Contains no obvious factual contradictions

Reject it (approved=false) only if there are MAJOR gaps — missing a key
aspect of the question, internally contradictory, or completely off-topic.
Be constructive: provide specific, actionable feedback the Writer can act on.

Do not reject for minor stylistic preferences."""


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def critic_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Review the draft answer and return an approval or revision request.

    Parameters
    ----------
    state : ResearchState
        Reads ``question`` and ``draft_answer``.
    config : RunnableConfig
        Forwarded to the LLM for LangSmith tracing.

    Returns
    -------
    dict
        If approved: ``{"final_answer": draft, "feedback": ""}``.
        If rejected: ``{"feedback": "<revision notes>", "final_answer": ""}``.
    """
    messages = [
        ("system", _SYSTEM_PROMPT),
        (
            "human",
            f"Original question: {state['question']}\n\n"
            f"Draft answer to review:\n{state['draft_answer']}",
        ),
    ]

    print("\n[Critic] Reviewing draft answer...")

    result: CriticOutput = _critic_chain.invoke(messages, config=config)

    if result.approved:
        print("[Critic] Approved.")
        return {"final_answer": state["draft_answer"], "feedback": ""}
    else:
        print(f"[Critic] Rejected. Feedback: {result.feedback[:80]}...")
        return {"feedback": result.feedback, "final_answer": ""}
