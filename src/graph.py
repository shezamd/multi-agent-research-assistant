"""
Graph definition: wires all agents together into a LangGraph StateGraph.

This module is the only place that knows about the overall flow.  Individual
agent modules know nothing about each other — they just read/write state.

Graph topology
--------------
START
  └─► planner_node
          └─► [fan_out]  ← conditional edge using Send API
                  ├─► researcher_node (task 1)  ┐
                  ├─► researcher_node (task 2)  ├─ run in parallel
                  └─► researcher_node (task N)  ┘
                          └─► writer_node
                                  └─► critic_node
                                          ├─► END          (approved)
                                          └─► writer_node  (needs revision)

LangSmith
---------
Because LANGSMITH_TRACING=true is set in .env, the compiled graph
automatically emits a trace for every .invoke() call.  Each node appears
as a child span with its own prompt, response, token count, and latency.
No extra instrumentation code is needed here.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send

from src.state import ResearchState
from src.agents.planner import planner_node
from src.agents.researcher import researcher_node
from src.agents.writer import writer_node
from src.agents.critic import critic_node

MAX_REVISIONS = 2  # prevent runaway Critic→Writer loops


# ---------------------------------------------------------------------------
# Routing / edge functions
# ---------------------------------------------------------------------------

def fan_out_to_researchers(state: ResearchState) -> list[Send]:
    """Fan out one Researcher node per task using the Send API.

    Parameters
    ----------
    state : ResearchState
        Reads ``tasks`` and ``question``.

    Returns
    -------
    list[Send]
        One Send object per task.  LangGraph schedules all of them
        concurrently, waits for all to finish, then merges results via
        the operator.add reducer on research_results.

    Notes
    -----
    Send(node_name, state_dict) creates a new invocation of node_name
    with state_dict as its input.  This is how LangGraph implements
    dynamic fan-out — the number of parallel branches isn't known until
    the Planner runs.
    """
    return [
        Send("researcher_node", {"task": task, "question": state["question"]})
        for task in state["tasks"]
    ]


def route_after_critic(state: ResearchState) -> str:
    """Decide whether to end or loop back to the Writer.

    Parameters
    ----------
    state : ResearchState
        Reads ``final_answer`` and ``revision_count``.

    Returns
    -------
    str
        ``"__end__"`` if approved or max revisions reached,
        ``"writer_node"`` to request a revision.
    """
    if state.get("final_answer"):
        return END

    if state.get("revision_count", 0) >= MAX_REVISIONS:
        print(f"\n[Graph] Max revisions ({MAX_REVISIONS}) reached — accepting draft.")
        return END

    return "writer_node"


def increment_revision(state: ResearchState) -> dict:
    """Tiny node that bumps revision_count before each Writer retry.

    Parameters
    ----------
    state : ResearchState
        Reads ``revision_count``.

    Returns
    -------
    dict
        ``{"revision_count": incremented_value}``

    Notes
    -----
    We use a dedicated node instead of updating the counter inside the
    Writer or Critic so those agents stay focused on their core job.
    """
    return {"revision_count": state.get("revision_count", 0) + 1}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Construct and compile the research assistant graph.

    Returns
    -------
    StateGraph
        The compiled graph, ready to call with .invoke().
    """
    builder = StateGraph(ResearchState)

    # --- Register nodes ---
    # Each string name becomes the node's identifier in traces and edges.
    builder.add_node("planner_node", planner_node)
    builder.add_node("researcher_node", researcher_node)
    builder.add_node("writer_node", writer_node)
    builder.add_node("critic_node", critic_node)
    builder.add_node("increment_revision", increment_revision)

    # --- Static edges ---
    builder.add_edge(START, "planner_node")

    # After all parallel researchers finish, go to writer.
    # LangGraph automatically waits for every Send() to complete
    # before moving past the fan-out point.
    builder.add_edge("researcher_node", "writer_node")
    builder.add_edge("writer_node", "critic_node")

    # Bump the counter, then go back to the writer for another pass
    builder.add_edge("increment_revision", "writer_node")

    # --- Conditional edges ---
    # Fan-out: planner → N parallel researchers (dynamic, via Send)
    builder.add_conditional_edges(
        "planner_node",
        fan_out_to_researchers,
    )

    # Critic decision: approve → END, reject → increment → writer
    builder.add_conditional_edges(
        "critic_node",
        route_after_critic,
        {
            END: END,
            "writer_node": "increment_revision",
        },
    )

    return builder.compile()


# Compile once at import time so main.py can import `app` directly.
app = build_graph()
