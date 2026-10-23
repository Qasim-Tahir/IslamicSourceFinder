"""
Status and health check routes.
"""

import logging
from collections import Counter

from fastapi import APIRouter, Depends
from dependencies import get_client
from schemas import StatusResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["status"])


_cached_counts: dict[str, int] = {}

@router.get("/status", response_model=StatusResponse)
async def get_status(client=Depends(get_client)):
    """
    Returns system health, collection stats, and indexed source names with
    passage counts. Used by the frontend Masthead and Library components.
    """
    global _cached_counts
    try:
        collection = client.collection
        count = collection.count()

        if not _cached_counts and count > 0:
            # Count passages per PDF source from all metadata
            counts = Counter(
                meta["pdf_name"]
                for meta in collection.get(include=["metadatas"]).get("metadatas", [])
                if meta and "pdf_name" in meta
            )
            _cached_counts = dict(sorted(counts.items()))

        return StatusResponse(
            status="healthy",
            total_documents=count,
            collection_name=client.collection_name,
            sources=list(_cached_counts),
            source_counts=_cached_counts,
        )

    except Exception as e:
        logger.error(f"Status check error: {e}")
        return StatusResponse(
            status="degraded",
            total_documents=0,
            collection_name="unknown",
            sources=[],
        )
