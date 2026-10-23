"""
Pydantic models for request validation and response serialization.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
import re


# ── Request Models ──────────────────────────────────────────────

class TextSearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=5000, description="Search query text")
    n_results: int = Field(default=5, ge=1, le=20, description="Number of results to return")

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        # Strip excessive whitespace
        return " ".join(v.split())


class VideoSearchRequest(BaseModel):
    url: str = Field(..., description="YouTube or video URL")
    n_results: int = Field(default=5, ge=1, le=20, description="Number of results to return")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        # Basic URL validation — allow YouTube and other common video URLs
        url_pattern = re.compile(
            r'^https?://'
            r'(?:(?:www\.)?youtube\.com/watch\?v=[\w-]+'
            r'|youtu\.be/[\w-]+'
            r'|(?:www\.)?youtube\.com/shorts/[\w-]+'
            r'|.+\.(mp4|webm|mkv|avi))'
        )
        if not url_pattern.match(v.strip()):
            raise ValueError("Invalid video URL. Please provide a valid YouTube URL.")
        return v.strip()


# ── Response Models ─────────────────────────────────────────────

class SearchResult(BaseModel):
    pdf_name: str = Field(description="Source PDF filename")
    book_number: str = Field(description="Book number in the collection")
    hadees_number: str = Field(description="Hadees number within the book")
    text: str = Field(description="Full hadees text with metadata")
    relevance_score: float = Field(description="Cosine similarity score (0-1)")


class TextSearchResponse(BaseModel):
    results: list[SearchResult]
    query: str
    total_results: int


class ImageSearchResponse(BaseModel):
    results: list[SearchResult]
    extracted_text: str
    total_results: int


class VideoSearchResponse(BaseModel):
    results: list[SearchResult]
    transcription: str
    total_results: int


class StatusResponse(BaseModel):
    status: str = "healthy"
    total_documents: int
    collection_name: str
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    sources: list[str] = Field(default_factory=list, description="List of indexed PDF names")
    source_counts: dict[str, int] = Field(default_factory=dict, description="Indexed passages per PDF")


class ErrorResponse(BaseModel):
    detail: str
    error_type: Optional[str] = None
