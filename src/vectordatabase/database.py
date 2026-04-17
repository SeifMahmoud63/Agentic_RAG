"""
Vector database adapter — thin wrapper delegating to qdrant_db.
Maintains backward-compatible function signatures where possible.
"""

from VectorDatabase import QdrantDb
from models import ResponseSignal


def search(query: str, top_k: int = None):
    """Search for relevant documents using Qdrant hybrid search."""
    return QdrantDb.hybrid_search(query=query, top_k=top_k)
