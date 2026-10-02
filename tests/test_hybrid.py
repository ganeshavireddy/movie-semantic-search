from src.embedder import MovieEmbedder
from src.index import MovieIndex
from src.search import MovieSearchEngine
from src.hybrid import HybridSearchEngine, reciprocal_rank_fusion


def test_reciprocal_rank_fusion_logic():
    bm25 = [
        {"title": "Movie A", "score": 10.0, "plot_snippet": "A"},
        {"title": "Movie B", "score": 5.0, "plot_snippet": "B"}
    ]
    semantic = [
        {"title": "Movie B", "score": 0.9, "plot_snippet": "B_sem"},
        {"title": "Movie C", "score": 0.8, "plot_snippet": "C"}
    ]

    hybrid = reciprocal_rank_fusion(bm25, semantic, k=3, rrf_k=60)
    assert len(hybrid) == 3
    # Movie B appeared in both rank 2 (BM25) and rank 1 (Semantic) -> should get high RRF score
    titles = [h["title"] for h in hybrid]
    assert "Movie B" in titles


def test_hybrid_search_engine_integration():
    embedder = MovieEmbedder()
    index = MovieIndex()

    metadata = [
        {
            "chunk_id": "c1", "title": "WALL-E", "release_year": 2008, "genre": "Animation",
            "director": "Andrew Stanton", "cast": "Ben Burtt", "origin": "American", "wiki_page": "",
            "chunk_text": "A lonely robot compacts trash on a deserted Earth."
        },
        {
            "chunk_id": "c2", "title": "The Matrix", "release_year": 1999, "genre": "Sci-Fi",
            "director": "Wachowskis", "cast": "Keanu Reeves", "origin": "American", "wiki_page": "",
            "chunk_text": "A computer hacker discovers reality is a machine simulation."
        }
    ]

    embeddings = embedder.encode([m["chunk_text"] for m in metadata], normalize_embeddings=True)
    index.build(embeddings, metadata, index_type="Flat")

    search_engine = MovieSearchEngine(movie_index=index, embedder=embedder)
    hybrid_engine = HybridSearchEngine(search_engine)

    results = hybrid_engine.search("hacker simulation matrix", k=2)
    assert len(results) >= 1
    assert results[0]["title"] == "The Matrix"
