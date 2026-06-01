"""
Qdrant client singleton.

If QDRANT_LOCAL_PATH is set, uses a local embedded (file-based) store — handy for
local development without a server or cloud. Otherwise connects to QDRANT_URL
(default http://localhost:6333), with QDRANT_API_KEY for Qdrant Cloud.
A single client instance is shared across all callers within the same process.
"""

import warnings

warnings.filterwarnings("ignore")

from typing import Optional

from qdrant_client import QdrantClient

from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_LOCAL_PATH

_client: Optional[QdrantClient] = None


def _build_client() -> QdrantClient:
    """Construct a new QdrantClient based on the configured backend."""
    if QDRANT_LOCAL_PATH:
        return QdrantClient(path=QDRANT_LOCAL_PATH)
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,   # None for local, required for Qdrant Cloud
        check_compatibility=False,
    )


# When running under Streamlit, use @st.cache_resource so the same client
# survives reruns and threading. Embedded local Qdrant is single-process and
# non-reentrant on its file lock; without this, Streamlit's rerun-on-interaction
# can re-enter the constructor and trip "Storage folder ... already accessed".
def _streamlit_cached_builder():
    try:
        import streamlit as st
    except ImportError:
        return None
    try:
        # Verify we are actually inside a Streamlit script run; outside one
        # (e.g. plain `python -c`), get_script_run_ctx returns None.
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is None:
            return None
    except ImportError:
        return None
    return st.cache_resource(show_spinner=False)(_build_client)


def get_client() -> QdrantClient:
    """Return the shared Qdrant client, creating it on first call."""
    global _client
    if _client is None:
        cached = _streamlit_cached_builder()
        _client = cached() if cached is not None else _build_client()
    return _client
