import numpy as np
from src.embedder import MovieEmbedder


def test_embedder_single_and_batch():
    embedder = MovieEmbedder(model_name="all-MiniLM-L6-v2")
    assert embedder.embedding_dim == 384

    # Test single sentence
    emb1 = embedder.encode("A robot cleans Earth.")
    assert isinstance(emb1, np.ndarray)
    assert emb1.shape == (1, 384)

    # Check normalization: L2 norm of normalized vector must be approx 1.0
    norm = np.linalg.norm(emb1[0])
    assert np.isclose(norm, 1.0, atol=1e-5)

    # Test batch
    texts = ["A robot cleans Earth.", "A heist crew in Los Angeles."]
    emb_batch = embedder.encode(texts)
    assert emb_batch.shape == (2, 384)

    # Similarity check: identical text similarity ~1.0
    emb_same = embedder.encode(["A robot cleans Earth."])
    sim = np.dot(emb1[0], emb_same[0])
    assert sim > 0.99
