import numpy as np
import pytest
from pathlib import Path
from src.index import MovieIndex


@pytest.fixture
def dummy_embeddings_and_meta():
    np.random.seed(42)
    embeddings = np.random.randn(20, 384).astype(np.float32)
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    metadata = [
        {"chunk_id": f"c_{i}", "title": f"Movie {i}", "release_year": 2000 + i, "chunk_text": f"Plot of movie {i}"}
        for i in range(20)
    ]
    return embeddings, metadata


def test_index_flat_build_search_save_load(dummy_embeddings_and_meta, tmp_path: Path):
    embeddings, metadata = dummy_embeddings_and_meta
    idx = MovieIndex(dimension=384)
    idx.build(embeddings, metadata, index_type="Flat")

    assert idx.index.ntotal == 20
    assert len(idx.metadata) == 20

    # Search
    query = embeddings[0]
    results = idx.search(query, top_k=3)
    assert len(results) == 3
    # Top 1 should be Movie 0 with score ~ 1.0
    score, meta = results[0]
    assert meta["title"] == "Movie 0"
    assert np.isclose(score, 1.0, atol=1e-4)

    # Save & Load
    idx_file = tmp_path / "faiss.index"
    meta_file = tmp_path / "meta.json"
    idx.save(idx_file, meta_file)

    loaded_idx = MovieIndex()
    loaded_idx.load(idx_file, meta_file)
    assert loaded_idx.index.ntotal == 20
    results_loaded = loaded_idx.search(query, top_k=3)
    assert results_loaded[0][1]["title"] == "Movie 0"


def test_index_hnsw(dummy_embeddings_and_meta):
    embeddings, metadata = dummy_embeddings_and_meta
    idx = MovieIndex(dimension=384)
    idx.build(embeddings, metadata, index_type="HNSW")
    assert idx.index.ntotal == 20

    query = embeddings[5]
    results = idx.search(query, top_k=1)
    assert results[0][1]["title"] == "Movie 5"


def test_index_ivf(dummy_embeddings_and_meta):
    embeddings, metadata = dummy_embeddings_and_meta
    idx = MovieIndex(dimension=384)
    idx.build(embeddings, metadata, index_type="IVF")
    assert idx.index.ntotal == 20

    query = embeddings[3]
    results = idx.search(query, top_k=1)
    assert results[0][1]["title"] == "Movie 3"
