"""
Qdrant HTTP client singleton.

Connects to the Qdrant server at QDRANT_URL (default http://localhost:6333).
A single client instance is shared across all callers within the same process.
"""

import warnings

warnings.filterwarnings("ignore")

from typing import Optional

from qdrant_client import QdrantClient

from config import QDRANT_URL, QDRANT_API_KEY

_client: Optional[QdrantClient] = None


def get_client() -> QdrantClient:
    """Return the shared Qdrant client, creating it on first call."""
    global _client
    if _client is None:
        _client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,   # None for local, required for Qdrant Cloud
            check_compatibility=False,
        )
    return _client
