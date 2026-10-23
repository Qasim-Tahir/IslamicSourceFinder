"""
Islamic Source Finder — FastAPI Backend

Wraps the existing IslamicSourceFinderClient as a REST API with:
- CORS for the React dev server
- Rate limiting for public safety
- SSE streaming for long-running pipelines
- Structured error handling
"""

import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from the ISF root .env *before* importing other modules
# This ensures libraries like sentence-transformers pick up HF_HOME upon import
ISF_ROOT = str(Path(__file__).resolve().parent.parent.parent)
env_path = os.path.join(ISF_ROOT, ".env")
load_dotenv(env_path)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from dependencies import startup_warmup
from routes import search, status

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("isf-api")


# ── Rate Limiter ────────────────────────────────────────────────

class RateLimiter:
    """Simple in-memory rate limiter per IP address."""

    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, ip: str) -> bool:
        now = time.time()
        # Clean old entries
        self.requests[ip] = [
            t for t in self.requests[ip] if now - t < self.window
        ]
        if len(self.requests[ip]) >= self.max_requests:
            return False
        self.requests[ip].append(now)
        return True


rate_limiter = RateLimiter(max_requests=30, window_seconds=60)


# ── Application Lifecycle ───────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle manager."""
    logger.info("Starting ISF API server...")
    logger.info(f"Loading .env from: {env_path}")

    # Pre-initialize the client and warm up the model
    try:
        client = startup_warmup()
        doc_count = client.collection.count()
        logger.info(f"Client initialized. {doc_count} documents in collection.")
    except Exception as e:
        logger.error(f"Failed to initialize client: {e}")
        logger.warning("Server starting in degraded mode — queries will fail.")

    yield  # Server is running

    logger.info("Shutting down ISF API server.")


# ── FastAPI App ─────────────────────────────────────────────────

app = FastAPI(
    title="Islamic Source Finder API",
    description="Multi-modal semantic search over indexed Hadith collections.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the React dev server and common production origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",   # Fallback
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Middleware ──────────────────────────────────────────────────

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting to all API routes."""
    # Skip rate limiting for non-API routes (e.g., docs)
    if not request.url.path.startswith("/api"):
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests. Please wait a moment and try again."},
        )
    return await call_next(request)


# ── Register Routers ───────────────────────────────────────────

app.include_router(search.router)
app.include_router(status.router)


# ── Root ────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "Islamic Source Finder API",
        "version": "1.0.0",
        "docs": "/docs",
    }
