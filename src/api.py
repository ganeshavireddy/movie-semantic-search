import logging
from typing import List, Dict, Any, Optional, Literal
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, ConfigDict

from src.config import settings
from src.search import MovieSearchEngine
from src.hybrid import HybridSearchEngine
from src.rag import MovieRAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Movie Plot Semantic Search & RAG API",
    description="Production-grade API for semantic search, keyword search, hybrid RRF search, and grounded RAG Q&A over movie plots.",
    version="1.0.0"
)

# Global singleton instances
search_engine = MovieSearchEngine()
hybrid_engine = HybridSearchEngine(search_engine)
rag_pipeline = MovieRAGPipeline(search_engine=search_engine)


# --- Pydantic Models ---

class SearchRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "a lonely robot compacts trash on abandoned Earth"})
    mode: Literal["semantic", "keyword", "hybrid"] = Field(default="hybrid", description="Search execution mode")
    k: int = Field(default=5, ge=1, le=50, description="Top K results to return")
    genre: Optional[str] = Field(default=None, json_schema_extra={"example": "Sci-Fi"})
    year_from: Optional[int] = Field(default=None, json_schema_extra={"example": 1990})
    year_to: Optional[int] = Field(default=None, json_schema_extra={"example": 2020})
    origin: Optional[str] = Field(default=None, json_schema_extra={"example": "American"})


class MovieResult(BaseModel):
    title: str
    release_year: int
    genre: str
    director: str
    cast: str
    origin: str
    wiki_page: str
    score: float
    plot_snippet: str


class SearchResponse(BaseModel):
    query: str
    mode: str
    total_results: int
    results: List[MovieResult]


class ChatMessage(BaseModel):
    role: str = Field(..., json_schema_extra={"example": "user"})
    content: str = Field(..., json_schema_extra={"example": "Who is the main robot in WALL-E?"})


class AskRequest(BaseModel):
    question: str = Field(..., json_schema_extra={"example": "What happens in WALL-E?"})
    k: int = Field(default=5, ge=1, le=10)
    history: Optional[List[ChatMessage]] = Field(default=None, description="Previous conversation turns")


class AskResponse(BaseModel):
    question: str
    answer: str
    source_chunks: List[MovieResult]
    conversation_history: List[ChatMessage]


class MovieDetailResponse(BaseModel):
    title: str
    release_year: int
    genre: str
    director: str
    cast: str
    origin: str
    wiki_page: str
    total_chunks: int
    plot: str


class HealthResponse(BaseModel):
    status: str
    total_vectors: int
    total_metadata_chunks: int
    index_type: str
    embedding_model: str
    llm_provider: str


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Health check endpoint exposing system & index stats."""
    return HealthResponse(
        status="ok",
        total_vectors=search_engine.movie_index.index.ntotal if search_engine.movie_index.index else 0,
        total_metadata_chunks=len(search_engine.movie_index.metadata),
        index_type=settings.index_type,
        embedding_model=settings.embedding_model_name,
        llm_provider=settings.llm_provider
    )


@app.post("/search", response_model=SearchResponse, tags=["Search"])
def search_movies(req: SearchRequest):
    """Search movie plots using semantic, keyword, or hybrid RRF mode."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    if req.mode == "semantic":
        results = search_engine.search_semantic(
            req.query, k=req.k, genre=req.genre, year_from=req.year_from, year_to=req.year_to, origin=req.origin
        )
    elif req.mode == "keyword":
        results = search_engine.search_keyword(
            req.query, k=req.k, genre=req.genre, year_from=req.year_from, year_to=req.year_to, origin=req.origin
        )
    else:
        results = hybrid_engine.search(
            req.query, k=req.k, genre=req.genre, year_from=req.year_from, year_to=req.year_to, origin=req.origin
        )

    formatted_results = [MovieResult(**res) for res in results]
    return SearchResponse(
        query=req.query,
        mode=req.mode,
        total_results=len(formatted_results),
        results=formatted_results
    )


@app.post("/ask", response_model=AskResponse, tags=["RAG Q&A"])
def ask_question(req: AskRequest):
    """Answer movie questions using grounded RAG with citations."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    raw_history = [turn.model_dump() for turn in req.history] if req.history else None
    res = rag_pipeline.ask(req.question, k=req.k, history=raw_history)

    formatted_sources = [MovieResult(**s) for s in res["source_chunks"]]
    formatted_history = [ChatMessage(**h) for h in res["conversation_history"]]

    return AskResponse(
        question=res["question"],
        answer=res["answer"],
        source_chunks=formatted_sources,
        conversation_history=formatted_history
    )


@app.get("/movie/{title}", response_model=MovieDetailResponse, tags=["Movies"])
def get_movie_details(title: str):
    """Get complete metadata and full plot text for a specific movie title."""
    details = search_engine.get_movie_details(title)
    if not details:
        raise HTTPException(status_code=404, detail=f"Movie '{title}' not found in index.")

    return MovieDetailResponse(**details)
