"""
Semantic cache for full research pipeline results.

Stores (question_embedding → answer) pairs in a persistent ChromaDB
collection so that semantically similar questions return cached answers
without re-running the graph.

How it works
------------
1. On each question, encode it with a local sentence-transformer model.
2. Query ChromaDB for the nearest stored question embedding (cosine distance).
3. If the distance is below ``threshold``, return the cached answer.
4. Otherwise, run the graph and store the result.

The embeddings are computed from *questions*, not answers — so similarity
is judged on what the user asked, not on the content of the answer.
"""

import hashlib

import chromadb
from sentence_transformers import SentenceTransformer

# Cosine distance: 0.0 = identical, 2.0 = opposite.
# 0.15 ≈ ~92% cosine similarity — tight enough to avoid false hits.
_DEFAULT_THRESHOLD = 0.15


class SemanticCache:
    """Persistent semantic cache backed by ChromaDB + sentence-transformers.

    Parameters
    ----------
    threshold : float
        Maximum cosine distance to consider a cache hit.  Lower = stricter.
    persist_dir : str
        Directory where ChromaDB persists its data across runs.
    model_name : str
        Sentence-transformer model used to embed questions.
        ``all-MiniLM-L6-v2`` is ~80 MB, fast, and works well for Q&A.
    """

    def __init__(
        self,
        threshold: float = _DEFAULT_THRESHOLD,
        persist_dir: str = ".cache/chroma",
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.threshold = threshold
        self._model = SentenceTransformer(model_name)
        self._client = chromadb.PersistentClient(path=persist_dir)
        # No embedding_function set — we provide embeddings manually so that
        # query similarity is computed on question embeddings, not answer text.
        self._collection = self._client.get_or_create_collection(
            name="research_cache",
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, question: str) -> str | None:
        """Return a cached answer if a semantically similar question exists.

        Parameters
        ----------
        question : str
            The incoming research question.

        Returns
        -------
        str or None
            The cached answer string, or ``None`` on a cache miss.
        """
        if self._collection.count() == 0:
            return None

        results = self._collection.query(
            query_embeddings=[self._embed(question)],
            n_results=1,
        )

        distance = results["distances"][0][0]
        if distance <= self.threshold:
            matched_q = results["metadatas"][0][0]["question"]
            print(f"\n[Cache] Hit  distance={distance:.3f}  matched: '{matched_q[:60]}...'")
            return results["documents"][0][0]

        print(f"\n[Cache] Miss distance={distance:.3f} — running graph.")
        return None

    def set(self, question: str, answer: str) -> None:
        """Store a question/answer pair in the cache.

        Parameters
        ----------
        question : str
            The research question (used to compute the embedding).
        answer : str
            The full research answer to store and later retrieve.
        """
        doc_id = hashlib.md5(question.encode()).hexdigest()
        self._collection.upsert(
            documents=[answer],
            embeddings=[self._embed(question)],
            ids=[doc_id],
            metadatas=[{"question": question}],
        )
        print(f"[Cache] Stored result for: '{question[:60]}...'")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        return self._model.encode(text).tolist()
