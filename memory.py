"""
G.I.D.E.O.N Semantic Vector Memory (ChromaDB + Ollama embeddings)

Two behaviours matter here beyond plain storage:

1. The embedder runs on CPU by default. It is only a 274MB model, but on a 4GB
   card it would otherwise evict the resident chat model on every write.
2. Not everything is worth remembering, and not every hit is worth recalling.
   Junk in ("ok", "thanks") crowds out real memories; unfiltered hits out mean
   three arbitrary snippets get injected into every single turn.
"""

from __future__ import annotations

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import ollama

from config import (
    CHROMA_PATH,
    EMBED_KEEP_ALIVE,
    EMBED_OPTIONS,
    EMBEDDING_MODEL,
    MEMORY_MAX_DISTANCE,
    MEMORY_MIN_CHARS,
    MEMORY_RESULTS,
    MEMORY_SNIPPET_CHARS,
    get_logger,
)

log = get_logger("gideon.memory")

COLLECTION_NAME = "gideon_history"

# Content that adds noise to the vector store rather than signal.
_JUNK_PHRASES = {
    "ok", "okay", "k", "yes", "no", "yeah", "nope", "sure", "thanks",
    "thank you", "ty", "hi", "hey", "hello", "bye", "goodbye", "cool",
    "nice", "got it", "understood", "right", "yep", "hmm", "lol",
}


class OllamaEmbeddingFunction(EmbeddingFunction):
    """Embeds documents with Ollama, batched, pinned to CPU by default."""

    def __init__(self, modelName: str = EMBEDDING_MODEL):
        self.modelName = modelName

    def __call__(self, input: Documents) -> Embeddings:
        if not input:
            return []
        try:
            # One batched call rather than a round-trip per document.
            response = ollama.embed(
                model=self.modelName,
                input=list(input),
                options=EMBED_OPTIONS,
                keep_alive=EMBED_KEEP_ALIVE,
            )
            return response["embeddings"]
        except Exception as exc:
            log.error("Embedding failed (%s): %s", self.modelName, exc)
            raise


def initMemory(chromaPath=CHROMA_PATH):
    """Initialize the ChromaDB persistent client and return the collection."""
    client = chromadb.PersistentClient(path=str(chromaPath))
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


def isWorthRemembering(content: str, role: str = "user") -> bool:
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
    # Internal bookkeeping written by the orchestrator.
    if text.startswith("[Tool Call:") or text.startswith("[Tool Result]"):
        return False
    return True


def addMemory(collection, messageId, content, sessionId, role, timestamp):
    """Add a message to the vector store if it is worth recalling later."""
    if not isWorthRemembering(content, role):
        log.debug("Skipped indexing message %s (low value)", messageId)
        return False

    try:
        collection.add(
            ids=[f"msg_{messageId}"],
            documents=[content],
            metadatas=[
                {
                    "session_id": sessionId,
                    "role": role,
                    "timestamp": timestamp,
                    "sqlite_id": messageId,
                }
            ],
        )
        return True
    except Exception as exc:
        log.error("Vector write failed for message %s: %s", messageId, exc)
        return False


def queryMemory(
    collection,
    queryText: str,
    nResults: int = MEMORY_RESULTS,
    maxDistance: float | None = MEMORY_MAX_DISTANCE,
):
    """Search past messages, discarding hits that are not actually relevant.

    ChromaDB always returns the n nearest neighbours, even when the nearest one
    is unrelated. Filtering on distance is what stops "hi" from pulling three
    random memories into the prompt.
    """
    if not queryText or not queryText.strip():
        return []

    try:
        results = collection.query(query_texts=[queryText], n_results=nResults)
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
        if maxDistance is not None and distance is not None and distance > maxDistance:
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


def deleteSessionMemories(collection, sessionId: str) -> None:
    """Remove every vector belonging to a deleted session."""
    try:
        collection.delete(where={"session_id": sessionId})
    except Exception as exc:
        log.warning("Vector cleanup for session %s failed: %s", sessionId, exc)
