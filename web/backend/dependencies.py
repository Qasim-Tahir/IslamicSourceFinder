"""
Dependency injection for the FastAPI application.
Adds the parent ISF directory to sys.path so we can import the existing
IslamicSourceFinderClient without copying or modifying it.
"""

import sys
import os
from pathlib import Path

# Add the ISF root directory to Python path so we can import client.py
ISF_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if ISF_ROOT not in sys.path:
    sys.path.insert(0, ISF_ROOT)

# Now we can import the existing client
from client import IslamicSourceFinderClient

# Module-level singleton reference
_client_instance: IslamicSourceFinderClient | None = None


def get_client() -> IslamicSourceFinderClient:
    """
    FastAPI dependency that provides the singleton ISF client.
    The client is initialized once on first call and reused thereafter.
    """
    global _client_instance
    if _client_instance is None:
        # Use the chroma_data directory relative to the ISF root
        chroma_path = os.path.join(ISF_ROOT, "chroma_data")
        _client_instance = IslamicSourceFinderClient(persist_directory=chroma_path)
    return _client_instance


def startup_warmup():
    """
    Called during FastAPI startup to pre-initialize the client
    and optionally warm up the embedding model.
    """
    client = get_client()
    # Pre-load the SentenceTransformer model so the first query isn't slow
    client.get_model()
    return client
