"""
G.I.D.E.O.N Semantic Vector Memory (ChromaDB + Ollama embeddings)

Two behaviours matter here beyond plain storage:

1. The embedder runs on CPU by default. It is only a 274MB model, but on a 4GB
   card it would otherwise evict the resident chat model on every write.
2. Not everything is worth remembering, and not every hit is worth recalling.
   Junk in ("ok", "thanks") crowds out real memories; unfiltered hits out mean
   three arbitrary snippets get injected into every single turn.

The store opens lazily, so constructing a VectorMemory costs nothing and
interfaces can be wired up before the engine has started.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import ollama
from chromadb import Documents, EmbeddingFunction, Embeddings

from gideon.core.config import (
    CHROMA_PATH,
    EMBED_KEEP_ALIVE,
    EMBED_OPTIONS,
    EMBEDDING_MODEL,
    MEMORY_MAX_DISTANCE,
    MEMORY_MIN_CHARS,
    MEMORY_RESULTS,
    MEMORY_SNIPPET_CHARS,
)
from gideon.core.logs import get_logger

log = get_logger("gideon.memory")

COLLECTION_NAME = "gideon_history"

# Content that adds noise to the vector store rather than signal.
_JUNK_PHRASES = {
    "ok", "okay", "k", "yes", "no", "yeah", "nope", "sure", "thanks",
    "thank you", "ty", "hi", "hey", "hello", "bye", "goodbye", "cool",
    "nice", "got it", "understood", "right", "yep", "hmm", "lol",
}

# Internal bookkeeping written by the agent; never meaningful to recall.
TOOL_BOOKKEEPING_PREFIXES = ("[Tool Call:", "[Tool Result]")


class OllamaEmbeddingFunction(EmbeddingFunction):
    """Embeds documents with Ollama, batched, pinned to CPU by default."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        if not input:
            return []
        try:
            # One batched call rather than a round-trip per document.
            response = ollama.embed(
                model=self.model_name,
                input=list(input),
                options=EMBED_OPTIONS,
                keep_alive=EMBED_KEEP_ALIVE,
            )
            return response["embeddings"]
        except Exception as exc:
            log.error("Embedding failed (%s): %s", self.model_name, exc)
            raise


def is_worth_remembering(content: str, role: str = "user") -> bool:
    """Decide whether a message carries retrievable meaning.

    Filler and tool bookkeeping degrade recall quality for everything else,
    so they never reach the vector store.
    """
    if role == "tool":
        return False

    text = (content or "").strip()
    if len(text) < MEMORY_MIN_CHARS:
        return False
    if text.lower().strip(".!?, ") in _JUNK_PHRASES:
        return False
    return not text.startswith(TOOL_BOOKKEEPING_PREFIXES)


class VectorMemory:
    """The ChromaDB collection plus the policy for what goes in and comes out."""

    def __init__(self, path: Path = CHROMA_PATH, collection: Any = None) -> None:
        self._path = Path(path)
        self._collection = collection
        self._open_lock = threading.Lock()
        # One writer thread: embedding is slow and must never block a reply.
        self._writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chroma_writer")

    @property
    def collection(self) -> Any:
        if self._collection is None:
            with self._open_lock:
                if self._collection is None:
                    self._collection = self._open()
        return self._collection

    def _open(self) -> Any:
        import chromadb

        client = chromadb.PersistentClient(path=str(self._path))
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=OllamaEmbeddingFunction(),
        )
        log.info(
            "Vector memory ready (%d documents, embedder on %s)",
            collection.count(),
            "CPU" if EMBED_OPTIONS.get("num_gpu") == 0 else "GPU",
        )
        return collection

    @property
    def is_open(self) -> bool:
        return self._collection is not None

    def count(self) -> int:
        return self._collection.count() if self._collection is not None else 0

    # ------------------------------------------------------------------
    #  Writes
    # ------------------------------------------------------------------
    def add(self, message_id: int, content: str, session_id: str, role: str, timestamp: str) -> bool:
        """Index a message if it is worth recalling later."""
        if not is_worth_remembering(content, role):
            log.debug("Skipped indexing message %s (low value)", message_id)
            return False

        try:
            self.collection.add(
                ids=[f"msg_{message_id}"],
                documents=[content],
                metadatas=[
                    {
                        "session_id": session_id,
                        "role": role,
                        "timestamp": timestamp,
                        "sqlite_id": message_id,
                    }
                ],
            )
            return True
        except Exception as exc:
            log.error("Vector write failed for message %s: %s", message_id, exc)
            return False

    def add_async(self, message_id: int, content: str, session_id: str, role: str, timestamp: str) -> None:
        """Queue a write on the background thread (non-blocking)."""
        self._writer.submit(self.add, message_id, content, session_id, role, timestamp)

    def upsert_fact(self, fact_id: str, text: str, kind: str, updated_at: str) -> bool:
        """Store or replace a fact (fact memory, PRD §6.2), recallable like a message.

        Facts are not conversation: they have an id of their own (`fact_id`,
        e.g. "project:c:/…/dumbpipe") and are rewritten when the fact changes.
        """
        try:
            self.collection.upsert(
                ids=[f"fact_{fact_id}"],
                documents=[text],
                metadatas=[{"kind": kind, "fact_id": fact_id, "timestamp": updated_at}],
            )
            return True
        except Exception as exc:
            log.error("Fact write failed for %s: %s", fact_id, exc)
            return False

    def delete_fact(self, fact_id: str) -> None:
        try:
            self.collection.delete(ids=[f"fact_{fact_id}"])
        except Exception as exc:
            log.warning("Fact delete failed for %s: %s", fact_id, exc)

    def delete_session(self, session_id: str) -> None:
        """Remove every vector belonging to a deleted session."""
        try:
            self.collection.delete(where={"session_id": session_id})
        except Exception as exc:
            log.warning("Vector cleanup for session %s failed: %s", session_id, exc)

    # ------------------------------------------------------------------
    #  Reads
    # ------------------------------------------------------------------
    def query(
        self,
        text: str,
        n_results: int = MEMORY_RESULTS,
        max_distance: float | None = MEMORY_MAX_DISTANCE,
    ) -> list[dict[str, Any]]:
        """Search past messages, discarding hits that are not actually relevant.

        ChromaDB always returns the n nearest neighbours, even when the nearest
        one is unrelated. Filtering on distance is what stops "hi" from pulling
        three random memories into the prompt.
        """
        if not text or not text.strip():
            return []

        try:
            results = self.collection.query(query_texts=[text], n_results=n_results)
        except Exception as exc:
            log.error("Vector query failed: %s", exc)
            return []

        documents = (results or {}).get("documents") or [[]]
        if not documents or not documents[0]:
            return []

        distances = (results.get("distances") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]

        memories = []
        for i, doc in enumerate(documents[0]):
            distance = distances[i] if i < len(distances) else None
            if max_distance is not None and distance is not None and distance > max_distance:
                continue
            snippet = doc if len(doc) <= MEMORY_SNIPPET_CHARS else doc[:MEMORY_SNIPPET_CHARS] + "…"
            memories.append(
                {
                    "content": snippet,
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "distance": distance,
                }
            )

        log.debug("Recall: %d/%d hits cleared threshold", len(memories), len(documents[0]))
        return memories

    def close(self) -> None:
        self._writer.shutdown(wait=False)
