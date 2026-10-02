from src.mcp_server import (
    semantic_search,
    keyword_search,
    get_movie_details,
    find_similar_movies,
    ask_movies,
    movie_stats,
    recommend_movie,
    get_rag_pipeline
)
from src.llm import BaseLLMProvider


class MockLLM(BaseLLMProvider):
    def generate(self, prompt: str, system_instruction: str = "") -> str:
        return "WALL-E is a robot [WALL-E (2008)]."


def test_mcp_semantic_search():
    res = semantic_search(query="robot cleans Earth", k=1)
    assert "WALL-E" in res
    assert "Similarity Score" in res


def test_mcp_keyword_search():
    res = keyword_search(query="mafia", k=1)
    assert "The Godfather" in res or "Results" in res


def test_mcp_get_movie_details():
    res = get_movie_details(title="WALL-E")
    assert "# WALL-E (2008)" in res
    assert "Full Movie Plot" in res


def test_mcp_find_similar_movies():
    res = find_similar_movies(title="Heat", k=1)
    assert "Movies Similar to 'Heat'" in res


def test_mcp_ask_movies():
    # Inject mock LLM
    get_rag_pipeline().llm_provider = MockLLM()
    res = ask_movies(question="Who is WALL-E?")
    assert "Answer" in res
    assert "WALL-E" in res


def test_mcp_resource_stats():
    stats_json = movie_stats()
    assert "total_vectors" in stats_json
    assert "index_type" in stats_json


def test_mcp_prompt_recommend_movie():
    prompt = recommend_movie(genre="Sci-Fi", mood="adventurous")
    assert "Sci-Fi" in prompt
    assert "adventurous" in prompt
