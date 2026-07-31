"""
Researcher agent: searches for information on a single task and summarises it.

This node is designed to run in parallel via LangGraph's Send API.
Each instance receives one task (not the full task list) and returns a
single summary string that gets appended to state['research_results']
via the operator.add reducer defined in ResearchState.

How parallel execution works
----------------------------
When the graph reaches the "fan-out" point after the Planner, a routing
function returns a list of Send() objects — one per task.  LangGraph
schedules all of them concurrently, then waits for all to finish before
merging their outputs and moving to the Writer node.

    Send("researcher_node", {"task": "...", "question": "..."})
              ↓  (runs in parallel for each task)
    researcher_node({"task": "...", "question": "..."})
              ↓
    {"research_results": ["one summary"]}
              ↓  (operator.add merges all parallel results)
    state["research_results"] == ["summary A", "summary B", ...]
"""

from typing import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableConfig

from src.tools import mock_search

# ---------------------------------------------------------------------------
# Input schema for this node
# ---------------------------------------------------------------------------
# When LangGraph's Send API dispatches this node, it passes a dict we
# define — not the full ResearchState.  Using a TypedDict makes the
# expected shape explicit and enables type checking.

class ResearcherInput(TypedDict):
    """Input passed to each parallel Researcher instance via Send().

    Parameters
    ----------
    task : str
        The single research task this instance should investigate.
    question : str
        The original user question — kept for context so the summary
        stays relevant to the overall research goal.
    """

    task: str
    question: str


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

_llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

_SYSTEM_PROMPT = """You are a research assistant summarising search results.

Given a research task and several search result snippets, write a concise
2-3 sentence summary of the key findings relevant to the task.
Be factual. Do not add information not present in the snippets."""


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def researcher_node(state: ResearcherInput, config: RunnableConfig) -> dict:
    """Search for information on one task and return a short summary.

    Parameters
    ----------
    state : ResearcherInput
        Contains ``task`` (the specific research task) and ``question``
        (the original user question for context).
    config : RunnableConfig
        Forwarded to the LLM so LangSmith can nest this call under the
        correct parallel branch in the trace.

    Returns
    -------
    dict
        ``{"research_results": [summary_string]}`` — a single-item list
        so operator.add can concatenate results from parallel instances.
    """
    task = state["task"]
    question = state["question"]

    # Step 1: call the search tool
    snippets = mock_search(task)
    snippets_text = "\n\n".join(f"- {s}" for s in snippets)

    print(f"\n[Researcher] Task: {task[:60]}...")

    # Step 2: ask Claude to summarise the snippets
    messages = [
        ("system", _SYSTEM_PROMPT),
        (
            "human",
            f"Original question: {question}\n\n"
            f"Research task: {task}\n\n"
            f"Search results:\n{snippets_text}",
        ),
    ]

    response = _llm.invoke(messages, config=config)
    summary = response.content

    print(f"[Researcher] Done: {summary[:80]}...")

    # Return a single-item list — operator.add will concatenate all
    # parallel researchers' lists into one list in the shared state.
    return {"research_results": [summary]}
