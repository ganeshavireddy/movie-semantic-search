import numpy as np
import pytest
from src.embedder import MovieEmbedder
from src.index import MovieIndex
from src.search import MovieSearchEngine


@pytest.fixture
def search_engine():
    # Setup dummy index
    embedder = MovieEmbedder()
    movie_index = MovieIndex()

    metadata = [
        {
            "chunk_id": "c1",
            "title": "WALL-E",
            "release_year": 2008,
            "origin": "American",
            "director": "Andrew Stanton",
            "cast": "Ben Burtt",
            "genre": "Animation / Sci-Fi",
            "wiki_page": "http://walle",
            "chunk_text": "A lonely robot cleans up garbage on an abandoned desolate Earth."
        },
        {
            "chunk_id": "c2",
            "title": "Heat",
            "release_year": 1995,
            "origin": "American",
            "director": "Michael Mann",
            "cast": "Robert De Niro, Al Pacino",
            "genre": "Crime / Drama",
            "wiki_page": "http://heat",
            "chunk_text": "A methodical heist crew in Los Angeles is betrayed by their leader during a LAPD pursuit."
        },
        {
            "chunk_id": "c3",
            "title": "The Godfather",
            "release_year": 1972,
            "origin": "American",
            "director": "Francis Ford Coppola",
            "cast": "Marlon Brando",
            "genre": "Crime / Drama",
            "wiki_page": "http://godfather",
            "chunk_text": "An Italian-American crime syndicate family purges rival mafia bosses in New York."
        }
    ]

    texts = [m["chunk_text"] for m in metadata]
    embeddings = embedder.encode(texts, normalize_embeddings=True)
    movie_index.build(embeddings, metadata, index_type="Flat")

    engine = MovieSearchEngine(movie_index=movie_index, embedder=embedder)
    return engine


def test_semantic_search_robot_query(search_engine):
    results = search_engine.search_semantic("a lonely robot cleans up an abandoned Earth", k=1)
    assert len(results) == 1
    assert results[0]["title"] == "WALL-E"
    assert results[0]["release_year"] == 2008


def test_semantic_search_heist_query(search_engine):
    results = search_engine.search_semantic("heist crew betrayed by their leader", k=1)
    assert len(results) == 1
    assert results[0]["title"] == "Heat"


def test_keyword_search(search_engine):
    results = search_engine.search_keyword("mafia bosses", k=1)
    assert len(results) == 1
    assert results[0]["title"] == "The Godfather"


def test_search_filters(search_engine):
    # Year filter: WALL-E (2008) should be excluded if year_to = 2000
    results = search_engine.search_semantic("robot", k=5, year_to=2000)
    titles = [r["title"] for r in results]
    assert "WALL-E" not in titles

    # Genre filter
    results_crime = search_engine.search_semantic("crew", k=5, genre="Crime")
    assert len(results_crime) > 0
    for r in results_crime:
        assert "Crime" in r["genre"]


def test_get_movie_details(search_engine):
    details = search_engine.get_movie_details("WALL-E")
    assert details is not None
    assert details["title"] == "WALL-E"
    assert "lonely robot" in details["plot"]


def test_find_similar_movies(search_engine):
    # godfather and heat are crime dramas
    similar = search_engine.find_similar_movies("Heat", k=1)
    assert len(similar) == 1
    assert similar[0]["title"] != "Heat"
