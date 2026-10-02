import pandas as pd
from src.chunking import chunk_text, chunk_dataset


def test_chunk_text_short():
    text = "Short plot snippet."
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_long_with_overlap():
    words = [f"word_{i}" for i in range(100)]
    long_text = " ".join(words)
    chunks = chunk_text(long_text, chunk_size=30, chunk_overlap=10)
    
    assert len(chunks) > 1
    # Check overlap: end of first chunk and start of second chunk
    c1_words = chunks[0].split()
    c2_words = chunks[1].split()
    assert len(c1_words) == 30
    assert c1_words[-10:] == c2_words[:10]


def test_chunk_dataset_preserves_metadata():
    df_movies = pd.DataFrame([{
        "title": "WALL-E",
        "release_year": 2008,
        "origin": "American",
        "director": "Andrew Stanton",
        "cast": "Ben Burtt",
        "genre": "Animation",
        "wiki_page": "http://wiki/walle",
        "plot": " ".join([f"sentence_{i}" for i in range(100)])
    }])

    df_chunks = chunk_dataset(df_movies, chunk_size_words=20, chunk_overlap_words=5)
    assert len(df_chunks) > 1
    assert "chunk_id" in df_chunks.columns
    assert df_chunks["title"].iloc[0] == "WALL-E"
    assert df_chunks["release_year"].iloc[0] == 2008
    assert df_chunks["director"].iloc[0] == "Andrew Stanton"
    assert "chunk_text" in df_chunks.columns
