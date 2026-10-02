import pytest
from fastapi.testclient import TestClient
from src.api import app, rag_pipeline
from src.llm import BaseLLMProvider


class MockLLM(BaseLLMProvider):
    def generate(self, prompt: str, system_instruction: str = "") -> str:
        return "WALL-E is a robot on Earth [WALL-E (2008)]."


@pytest.fixture
def client():
    # Replace LLM in global rag_pipeline with mock for tests
    rag_pipeline.llm_provider = MockLLM()
    with TestClient(app) as client:
        yield client


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "total_vectors" in data


def test_search_endpoint_hybrid(client):
    payload = {
        "query": "lonely robot compacts trash",
        "mode": "hybrid",
        "k": 2
    }
    res = client.post("/search", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["query"] == payload["query"]
    assert len(data["results"]) >= 1
    assert data["results"][0]["title"] == "WALL-E"


def test_search_endpoint_empty_query(client):
    payload = {"query": "   ", "mode": "semantic"}
    res = client.post("/search", json=payload)
    assert res.status_code == 400


def test_ask_endpoint(client):
    payload = {
        "question": "Who compacts trash in WALL-E?",
        "k": 2
    }
    res = client.post("/ask", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "WALL-E" in data["answer"]
    assert len(data["source_chunks"]) >= 1
    assert len(data["conversation_history"]) == 2


def test_get_movie_details(client):
    res = client.get("/movie/WALL-E")
    assert res.status_code == 200
    data = res.json()
    assert data["title"] == "WALL-E"
    assert data["release_year"] == 2008


def test_get_movie_details_not_found(client):
    res = client.get("/movie/NonExistentMovie999")
    assert res.status_code == 404
