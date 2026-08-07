# Multi-Agent Research Assistant

A command-line application where multiple AI agents collaborate to answer a research question. Built with **LangGraph** for orchestration and **LangSmith** for full observability.

---

## How It Works

```
User Question
      ↓
[Semantic Cache]    check if a similar question was answered before
      ↓ miss
  [Planner]         breaks the question into 2–4 research tasks
      ↓
  [Researchers]     one agent per task, all run in parallel
      ↓
  [Writer]          combines findings into a structured answer
      ↓
  [Critic]          approves the answer or sends it back for revision
      ↓
[Semantic Cache]    store result for future similar questions
      ↓
  Final Answer
```

### Agents

| Agent | Role |
|---|---|
| **Planner** | Uses structured output to produce a clean list of focused sub-tasks |
| **Researcher** | Calls a search tool and summarises the results for one task |
| **Writer** | Synthesises all research summaries into a readable answer |
| **Critic** | Reviews the draft; either approves it or returns specific revision notes |

### Parallelism via LangGraph's `Send` API

After the Planner runs, a routing function returns one `Send()` object per task. LangGraph schedules all Researcher nodes concurrently, waits for every branch to finish, then merges their outputs using an `operator.add` reducer before handing off to the Writer. The number of parallel branches is dynamic — it depends on how many tasks the Planner produced at runtime.

```python
def fan_out_to_researchers(state) -> list[Send]:
    return [
        Send("researcher_node", {"task": task, "question": state["question"]})
        for task in state["tasks"]
    ]
```

### Semantic Caching

Before invoking the graph, every question is checked against a persistent semantic cache backed by **ChromaDB**. The question is embedded with a local `sentence-transformers` model (`all-MiniLM-L6-v2`) and compared against stored question embeddings using cosine distance. If a sufficiently similar question has been answered before (distance ≤ 0.15, ≈92% similarity), the cached answer is returned immediately — skipping the entire multi-agent pipeline.

```
Question → embed → ChromaDB HNSW lookup → distance ≤ 0.15? → return cached answer
                                         → distance > 0.15? → run graph → store result
```

Key design choices:
- **Embeddings are computed on questions, not answers** — similarity is judged on what was asked, not what was said
- **ChromaDB persists to `.cache/chroma/`** — the cache survives across runs
- **`upsert` is used for writes** — re-running the same question updates the entry rather than duplicating it
- **Threshold is configurable** — pass `threshold=` to `SemanticCache()` to make matching stricter or looser

---

### Critic → Writer Revision Loop

The Critic drives the only conditional edge in the graph. If it finds major gaps, it writes feedback into the shared state and the graph routes back to the Writer for a revision pass. A `MAX_REVISIONS` guard prevents infinite loops.

```
critic_node → approved?  → END
           → rejected?  → increment_revision → writer_node → critic_node → ...
```

---

## LangSmith Observability

LangSmith provides full tracing with zero instrumentation code. Setting `LANGSMITH_TRACING=true` in `.env` is all that's needed — LangGraph hooks into LangChain's callback system automatically.

**What you can see in the LangSmith UI after each run:**

- The complete graph execution timeline, including which nodes ran in parallel
- Every agent's exact prompt and model response
- Token usage and latency per node
- The Critic's structured verdict and any revision loops

Each run is tagged with a `run_id` printed to the terminal so you can find it instantly in your project dashboard at [smith.langchain.com](https://smith.langchain.com).

---

## Project Structure

```
src/
  state.py              # Shared TypedDict that flows through every node
  tools.py              # Mock search tool (swap for Tavily/SerpAPI later)
  graph.py              # StateGraph wiring, routing functions, Send fan-out
  cache.py              # Semantic cache: ChromaDB + sentence-transformers
  agents/
    planner.py          # Breaks question into tasks (structured output)
    researcher.py       # Searches + summarises one task (parallel via Send)
    writer.py           # Combines summaries into a draft answer
    critic.py           # Reviews draft, approves or requests revision
main.py                 # CLI entry point
.cache/chroma/          # Persistent ChromaDB vector store (auto-created)
```

---

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management
- An [Anthropic API key](https://console.anthropic.com)
- A [LangSmith API key](https://smith.langchain.com) (free tier available)

### Install

```bash
git clone https://github.com/shehryarahmad/multi-agent-research-assistant.git
cd multi-agent-research-assistant
uv sync
```

### Configure

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

```env
ANTHROPIC_API_KEY=your_anthropic_key_here
LANGSMITH_API_KEY=your_langsmith_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=multi-agent-research-assistant
```

### Run

```bash
# Pass the question as an argument
uv run python main.py "What are the main causes and effects of ocean acidification?"

# Or run interactively
uv run python main.py
```

---

## Tech Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — multi-agent graph orchestration
- [LangSmith](https://smith.langchain.com) — tracing and observability
- [LangChain Anthropic](https://python.langchain.com/docs/integrations/chat/anthropic/) — Claude integration
- [ChromaDB](https://www.trychroma.com) — persistent vector store for semantic caching
- [sentence-transformers](https://www.sbert.net) — local question embeddings (`all-MiniLM-L6-v2`)
- [uv](https://docs.astral.sh/uv/) — dependency management
- [python-dotenv](https://github.com/theskumar/python-dotenv) — environment variable loading
