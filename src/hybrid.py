import logging
from typing import List, Dict, Any, Optional
from src.search import MovieSearchEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    bm25_results: List[Dict[str, Any]],
    semantic_results: List[Dict[str, Any]],
    k: int = 5,
    rrf_k: int = 60,
    weight_bm25: float = 1.0,
    weight_semantic: float = 1.0
) -> List[Dict[str, Any]]:
    """Combine BM25 and Semantic search results using Reciprocal Rank Fusion (RRF)."""
    movie_scores: Dict[str, float] = {}
    movie_details: Dict[str, Dict[str, Any]] = {}

    # Process BM25 rankings
    for rank, item in enumerate(bm25_results, start=1):
        title = item["title"]
        rrf_score = weight_bm25 / (rrf_k + rank)
        movie_scores[title] = movie_scores.get(title, 0.0) + rrf_score
        if title not in movie_details:
            movie_details[title] = item.copy()

    # Process Semantic rankings
    for rank, item in enumerate(semantic_results, start=1):
        title = item["title"]
        rrf_score = weight_semantic / (rrf_k + rank)
        movie_scores[title] = movie_scores.get(title, 0.0) + rrf_score
        if title not in movie_details:
            movie_details[title] = item.copy()
        else:
            # Update score and keep best plot snippet
            movie_details[title]["plot_snippet"] = item.get("plot_snippet", movie_details[title].get("plot_snippet"))

    # Assign combined score and sort
    hybrid_list = []
    for title, score in movie_scores.items():
        entry = movie_details[title].copy()
        entry["score"] = float(round(score, 6))
        hybrid_list.append(entry)

    hybrid_list = sorted(hybrid_list, key=lambda x: x["score"], reverse=True)
    return hybrid_list[:k]


class HybridSearchEngine:
    """Hybrid search wrapper combining BM25 keyword and dense semantic vector search via RRF."""

    def __init__(self, search_engine: MovieSearchEngine):
        self.search_engine = search_engine

    def search(
        self,
        query: str,
        k: int = 5,
        genre: Optional[str] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        origin: Optional[str] = None,
        rrf_k: int = 60
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search."""
        fetch_k = max(k * 4, 20)
        bm25_res = self.search_engine.search_keyword(
            query, k=fetch_k, genre=genre, year_from=year_from, year_to=year_to, origin=origin
        )
        semantic_res = self.search_engine.search_semantic(
            query, k=fetch_k, genre=genre, year_from=year_from, year_to=year_to, origin=origin
        )

        return reciprocal_rank_fusion(bm25_res, semantic_res, k=k, rrf_k=rrf_k)
