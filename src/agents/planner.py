"""
Planner agent: breaks a research question into 2-4 sub-tasks.

This is the first node in the graph. It receives the user's question
from state and populates state['tasks'] with a list of focused research
tasks for the parallel Researcher nodes to pick up.
"""

from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableConfig

from src.state import ResearchState


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------
# By defining a Pydantic model and using .with_structured_output(), we tell
# Claude to respond in a specific JSON shape.  No manual parsing needed.
# LangSmith will show you the raw JSON Claude produced in the trace.

class PlannerOutput(BaseModel):
    """Structured output from the Planner LLM call.

    Parameters
    ----------
    tasks : list[str]
        Between 2 and 4 focused research tasks derived from the question.
        Each task should be a single, searchable sentence.
    """

    tasks: list[str] = Field(
        description="2-4 focused research tasks derived from the question.",
        min_length=2,
        max_length=4,
    )


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------
# We instantiate the model once at module level so it is reused across calls.
# model_name: claude-sonnet-4-6 balances quality and speed well for planning.

_llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

# .with_structured_output() wraps the model so it automatically:
#   1. Injects a tool / JSON schema derived from PlannerOutput into the prompt.
#   2. Parses Claude's response back into a PlannerOutput instance.
_planner_chain = _llm.with_structured_output(PlannerOutput)

_SYSTEM_PROMPT = """You are a research planning assistant.

Given a research question, break it down into 2-4 focused sub-tasks.
Each task should be a single, specific, searchable question or topic.
Do not overlap tasks. Cover the question comprehensively but concisely."""


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------
# Every LangGraph node has the signature: (state, config?) -> dict
# The dict contains only the keys you want to update in the state.
# LangGraph merges the returned dict into the full state automatically.

def planner_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Plan the research by breaking the question into sub-tasks.

    Parameters
    ----------
    state : ResearchState
        Current graph state. Reads ``state['question']``.
    config : RunnableConfig
        Passed through to the LLM call. LangSmith uses this to attach
        the node's trace to the parent graph run automatically.

    Returns
    -------
    dict
        ``{"tasks": list[str]}`` — partial state update consumed by LangGraph.
    """
    messages = [
        ("system", _SYSTEM_PROMPT),
        ("human", f"Research question: {state['question']}"),
    ]

    # config is forwarded so LangSmith nests this LLM call under the
    # correct parent trace in the graph run.
    result: PlannerOutput = _planner_chain.invoke(messages, config=config)

    print(f"\n[Planner] Created {len(result.tasks)} tasks:")
    for i, task in enumerate(result.tasks, 1):
        print(f"  {i}. {task}")

    return {"tasks": result.tasks}
