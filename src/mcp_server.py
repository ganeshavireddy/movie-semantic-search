import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastmcp import FastMCP
from src.config import settings
from src.search import MovieSearchEngine
from src.hybrid import HybridSearchEngine
from src.rag import MovieRAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastMCP Server
mcp = FastMCP("Movie Semantic Search & RAG")

# Lazy-loaded core engine singletons
_search_engine: Optional[MovieSearchEngine] = None
_hybrid_engine: Optional[HybridSearchEngine] = None
_rag_pipeline: Optional[MovieRAGPipeline] = None


def get_search_engine() -> MovieSearchEngine:
    global _search_engine
    if _search_engine is None:
        _search_engine = MovieSearchEngine()
    return _search_engine


def get_hybrid_engine() -> HybridSearchEngine:
    global _hybrid_engine
    if _hybrid_engine is None:
        _hybrid_engine = HybridSearchEngine(get_search_engine())
    return _hybrid_engine


def get_rag_pipeline() -> MovieRAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = MovieRAGPipeline(search_engine=get_search_engine())
    return _rag_pipeline


# --- MCP TOOLS ---

@mcp.tool()
def semantic_search(
    query: str,
    k: int = 5,
    genre: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None
) -> str:
    """Search movie plots by conceptual meaning and themes rather than exact keywords.
    
    Args:
        query: Conceptual search query (e.g. 'lonely robot cleans up an abandoned Earth')
        k: Maximum number of movies to return (default: 5)
        genre: Optional genre filter (e.g. 'Sci-Fi', 'Animation')
        year_from: Optional minimum release year filter (e.g. 1990)
        year_to: Optional maximum release year filter (e.g. 2020)
    """
    engine = get_search_engine()
    results = engine.search_semantic(query=query, k=k, genre=genre, year_from=year_from, year_to=year_to)
    
    if not results:
        return f"No movies found matching semantic query: '{query}'"
        
    formatted = [f"### Semantic Search Results for '{query}':\n"]
    for i, r in enumerate(results, start=1):
        formatted.append(
            f"**{i}. {r['title']} ({r['release_year']})** - Similarity Score: {r['score']:.4f}\n"
            f"*Genre*: {r['genre']} | *Director*: {r['director']}\n"
            f"*Plot Snippet*: {r['plot_snippet']}\n"
        )
    return "\n".join(formatted)


@mcp.tool()
def keyword_search(query: str, k: int = 5) -> str:
    """Perform BM25 keyword search over movie plots for exact character names, places, or proper nouns.
    
    Args:
        query: Exact keyword query (e.g. 'Sollozzo mafia Corleone')
        k: Maximum number of movies to return (default: 5)
    """
    engine = get_search_engine()
    results = engine.search_keyword(query=query, k=k)
    
    if not results:
        return f"No movies found matching keyword query: '{query}'"
        
    formatted = [f"### Keyword Search Results for '{query}':\n"]
    for i, r in enumerate(results, start=1):
        formatted.append(
            f"**{i}. {r['title']} ({r['release_year']})** - BM25 Score: {r['score']:.4f}\n"
            f"*Genre*: {r['genre']} | *Director*: {r['director']}\n"
            f"*Plot Snippet*: {r['plot_snippet']}\n"
        )
    return "\n".join(formatted)


@mcp.tool()
def get_movie_details(title: str) -> str:
    """Retrieve full metadata, cast, director, wiki link, and complete plot text for a specific movie title.
    
    Args:
        title: Exact or approximate title of the movie (e.g. 'Inception')
    """
    engine = get_search_engine()
    details = engine.get_movie_details(title=title)
    
    if not details:
        return f"Movie '{title}' was not found in the dataset index."
        
    return (
        f"# {details['title']} ({details['release_year']})\n"
        f"- **Genre**: {details['genre']}\n"
        f"- **Director**: {details['director']}\n"
        f"- **Cast**: {details['cast']}\n"
        f"- **Origin**: {details['origin']}\n"
        f"- **Wiki**: [{details['wiki_page']}]({details['wiki_page']})\n\n"
        f"### Full Movie Plot\n{details['plot']}"
    )


@mcp.tool()
def find_similar_movies(title: str, k: int = 5) -> str:
    """Find movies with semantically similar plots to a specified target movie.
    
    Args:
        title: Target movie title to compare against (e.g. 'Heat')
        k: Number of similar movies to return (default: 5)
    """
    engine = get_search_engine()
    similar = engine.find_similar_movies(title=title, k=k)
    
    if not similar:
        return f"Could not find recommendations for movie '{title}'."
        
    formatted = [f"### Movies Similar to '{title}':\n"]
    for i, r in enumerate(similar, start=1):
        formatted.append(
            f"**{i}. {r['title']} ({r['release_year']})** - Similarity: {r['score']:.4f}\n"
            f"*Genre*: {r['genre']}\n"
            f"*Plot Snippet*: {r['plot_snippet']}\n"
        )
    return "\n".join(formatted)


@mcp.tool()
def ask_movies(question: str) -> str:
    """Answer complex movie questions using RAG grounded strictly in plot context with citations.
    
    Args:
        question: Question about movie plots (e.g. 'How does Morpheus rescue Neo in The Matrix?')
    """
    rag = get_rag_pipeline()
    res = rag.ask(question=question, k=5)
    
    answer_text = res["answer"]
    sources = res.get("source_chunks", [])
    
    formatted = [f"### Answer\n{answer_text}\n\n### Grounded Source Movies Used:"]
    for s in sources:
        formatted.append(f"- **{s['title']} ({s['release_year']})** [{s['genre']}]")
        
    return "\n".join(formatted)


# --- MCP RESOURCE ---

@mcp.resource("movies://stats")
def movie_stats() -> str:
    """Read-only resource providing vector index statistics and metadata."""
    engine = get_search_engine()
    total_vecs = engine.movie_index.index.ntotal if engine.movie_index.index else 0
    total_chunks = len(engine.movie_index.metadata)
    
    stats_data = {
        "status": "healthy",
        "total_vectors": total_vecs,
        "total_chunks": total_chunks,
        "index_type": settings.index_type,
        "embedding_model": settings.embedding_model_name,
        "llm_provider": settings.llm_provider,
        "raw_dataset_path": str(settings.raw_csv_path),
        "index_cache_path": str(settings.index_cache_path)
    }
    return json.dumps(stats_data, indent=2)


# --- MCP PROMPT ---

@mcp.prompt()
def recommend_movie(genre: str, mood: str) -> str:
    """Prompt template for recommending movies based on genre and user mood."""
    return f"Recommend 3 movies in the '{genre}' genre suitable for someone feeling '{mood}'. Use semantic_search tool to find movies matching '{mood} {genre}'."


if __name__ == "__main__":
    mcp.run(transport="stdio")
