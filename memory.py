import chromadb
from chromadb import EmbeddingFunction, Embeddings, Documents
import ollama


EMBEDDING_MODEL = "nomic-embed-text"
COLLECTION_NAME = "gideon_history"
CHROMA_PATH = ".chromadb"


class OllamaEmbeddingFunction(EmbeddingFunction):
    """Custom embedding function that uses Ollama's nomic-embed-text model locally."""

    def __init__(self, modelName=EMBEDDING_MODEL):
        self.modelName = modelName

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for a list of text documents using Ollama."""
        embeddings = []
        for text in input:
            response = ollama.embed(model=self.modelName, input=text)
            embeddings.append(response['embeddings'][0])
        return embeddings


def initMemory(chromaPath=CHROMA_PATH):
    """Initialize ChromaDB persistent client and return the collection."""
    client = chromadb.PersistentClient(path=chromaPath)
    embeddingFn = OllamaEmbeddingFunction()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embeddingFn
    )
    return collection


def addMemory(collection, messageId, content, sessionId, role, timestamp):
    """Add a message to the ChromaDB vector store.
    
    Maps SQLite's autoincremented message_id to a string ID for ChromaDB.
    """
    stringId = f"msg_{messageId}"
    collection.add(
        ids=[stringId],
        documents=[content],
        metadatas=[{
            "session_id": sessionId,
            "role": role,
            "timestamp": timestamp,
            "sqlite_id": messageId
        }]
    )


def queryMemory(collection, queryText, nResults=5):
    """Search ChromaDB for the most relevant past messages matching the query text."""
    results = collection.query(
        query_texts=[queryText],
        n_results=nResults
    )
    
    # Flatten the results into a list of dicts for easy consumption
    memories = []
    if results and results['documents'] and results['documents'][0]:
        for i, doc in enumerate(results['documents'][0]):
            memory = {
                "content": doc,
                "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                "distance": results['distances'][0][i] if results['distances'] else None
            }
            memories.append(memory)
    return memories
