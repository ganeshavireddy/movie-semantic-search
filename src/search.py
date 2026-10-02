import logging
import re
from typing import List, Dict, Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi

from src.config import settings
from src.embedder import MovieEmbedder
from src.index import MovieIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def tokenize_text(text: str) -> List[str]:
    """Basic lowercased word tokenization for BM25."""
    return re.findall(r"\w+", text.lower())


class MovieSearchEngine:
    """Unified search engine supporting dense semantic search, BM25 keyword search, and metadata filtering."""

    def __init__(self, movie_index: Optional[MovieIndex] = None, embedder: Optional[MovieEmbedder] = None):
        self.embedder = embedder or MovieEmbedder()

        if movie_index is not None:
            self.movie_index = movie_index
        else:
            self.movie_index = MovieIndex()
            if settings.index_cache_path.exists() and settings.metadata_cache_path.exists():
                self.movie_index.load(settings.index_cache_path, settings.metadata_cache_path)

        # Build BM25 index over loaded chunk metadata
        self.bm25: Optional[BM25Okapi] = None
        self._init_bm25()

    def _init_bm25(self) -> None:
        """Initialize BM25 index over chunk texts."""
        if not self.movie_index.metadata:
            logger.warning("No metadata available to initialize BM25.")
            return

        corpus = [tokenize_text(item.get("chunk_text", "")) for item in self.movie_index.metadata]
        logger.info(f"Initializing BM25 index over {len(corpus)} chunk documents...")
        self.bm25 = BM25Okapi(corpus)

    def _apply_filters(
        self,
        meta: Dict[str, Any],
        genre: Optional[str] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        origin: Optional[str] = None
    ) -> bool:
        """Check if metadata satisfies filtering conditions."""
        if genre and genre.strip():
            target_genre = genre.strip().lower()
            meta_genre = str(meta.get("genre", "")).lower()
            if target_genre not in meta_genre:
                return False

        if origin and origin.strip():
            target_origin = origin.strip().lower()
            meta_origin = str(meta.get("origin", "")).lower()
            if target_origin not in meta_origin:
                return False

        rel_year = meta.get("release_year", 0)
        if year_from is not None and rel_year < year_from:
            return False
        if year_to is not None and rel_year > year_to:
            return False

        return True

    def _deduplicate_chunks_by_movie(
        self,
        scored_items: List[tuple[float, Dict[str, Any]]],
        k: int,
        genre: Optional[str] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        origin: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Deduplicate chunk matches by movie title, keeping highest score per movie."""
        seen_movies: Dict[str, Dict[str, Any]] = {}

        for score, meta in scored_items:
            if not self._apply_filters(meta, genre=genre, year_from=year_from, year_to=year_to, origin=origin):
                continue

            title = meta.get("title", "")
            if not title:
                continue

            if title not in seen_movies or score > seen_movies[title]["score"]:
                seen_movies[title] = {
                    "title": title,
                    "release_year": meta.get("release_year", 0),
                    "genre": meta.get("genre", "Unknown"),
                    "director": meta.get("director", "Unknown"),
                    "cast": meta.get("cast", "Unknown"),
                    "origin": meta.get("origin", "Unknown"),
                    "wiki_page": meta.get("wiki_page", ""),
                    "score": float(round(score, 4)),
                    "plot_snippet": meta.get("chunk_text", ""),
                    "chunk_id": meta.get("chunk_id", "")
                }

        # Sort by score descending and take top k
        results = sorted(seen_movies.values(), key=lambda x: x["score"], reverse=True)
        return results[:k]

    def search_semantic(
        self,
        query: str,
        k: int = 5,
        genre: Optional[str] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        origin: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Perform dense vector semantic search."""
        if not query.strip():
            return []

        query_emb = self.embedder.encode(query, normalize_embeddings=True)
        # Fetch candidate top chunks (fetch more than k to account for deduplication & filtering)
        fetch_k = max(k * 10, 50)
        raw_results = self.movie_index.search(query_emb[0], top_k=fetch_k)

        return self._deduplicate_chunks_by_movie(
            raw_results, k=k, genre=genre, year_from=year_from, year_to=year_to, origin=origin
        )

    def search_keyword(
        self,
        query: str,
        k: int = 5,
        genre: Optional[str] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        origin: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Perform BM25 keyword search."""
        if not query.strip() or self.bm25 is None:
            return []

        tokens = tokenize_text(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        scored_items = [(float(score), meta) for score, meta in zip(scores, self.movie_index.metadata) if score > 0]
        scored_items = sorted(scored_items, key=lambda x: x[0], reverse=True)

        return self._deduplicate_chunks_by_movie(
            scored_items, k=k, genre=genre, year_from=year_from, year_to=year_to, origin=origin
        )

    def get_movie_details(self, title: str) -> Optional[Dict[str, Any]]:
        """Get full details and combined plot for a specific movie title."""
        target_title = title.strip().lower()
        matching_chunks = [
            meta for meta in self.movie_index.metadata
            if meta.get("title", "").strip().lower() == target_title
        ]

        if not matching_chunks:
            return None

        first = matching_chunks[0]
        # Combine chunks in order
        sorted_chunks = sorted(matching_chunks, key=lambda x: x.get("chunk_index", 0))
        combined_plot = " ".join([c.get("chunk_text", "") for c in sorted_chunks])

        return {
            "title": first.get("title"),
            "release_year": first.get("release_year"),
            "genre": first.get("genre"),
            "director": first.get("director"),
            "cast": first.get("cast"),
            "origin": first.get("origin"),
            "wiki_page": first.get("wiki_page"),
            "total_chunks": len(matching_chunks),
            "plot": combined_plot
        }

    def find_similar_movies(self, title: str, k: int = 5) -> List[Dict[str, Any]]:
        """Find movies semantically similar to a given movie title."""
        details = self.get_movie_details(title)
        if not details:
            return []

        # Encode movie plot text and search
        results = self.search_semantic(query=details["plot"], k=k + 1)
        # Exclude self title
        similar = [r for r in results if r["title"].strip().lower() != details["title"].strip().lower()]
        return similar[:k]
