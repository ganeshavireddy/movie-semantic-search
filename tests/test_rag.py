import pytest
from src.embedder import MovieEmbedder
from src.index import MovieIndex
from src.search import MovieSearchEngine
from src.llm import BaseLLMProvider
from src.rag import MovieRAGPipeline


class TestMockLLM(BaseLLMProvider):
    def generate(self, prompt: str, system_instruction: str = "") -> str:
        if "WALL-E" in prompt:
            return "WALL-E is a lonely robot who compacts trash on Earth [WALL-E (2008)]."
        return "I cannot answer this question based on the provided movie plot context."


@pytest.fixture
def rag_pipeline():
    embedder = MovieEmbedder()
    index = MovieIndex()

    metadata = [
        {
            "chunk_id": "c1", "title": "WALL-E", "release_year": 2008, "genre": "Animation",
            "director": "Andrew Stanton", "cast": "Ben Burtt", "origin": "American", "wiki_page": "",
            "chunk_text": "WALL-E is a lonely robot who spends his days compacting trash into cubes on a deserted Earth."
        }
    ]

    embeddings = embedder.encode([m["chunk_text"] for m in metadata], normalize_embeddings=True)
    index.build(embeddings, metadata, index_type="Flat")

    search_engine = MovieSearchEngine(movie_index=index, embedder=embedder)
    mock_llm = TestMockLLM()
    pipeline = MovieRAGPipeline(search_engine=search_engine, llm_provider=mock_llm)
    return pipeline


def test_rag_ask_success(rag_pipeline):
    res = rag_pipeline.ask("Who cleans trash on Earth?", k=1)
    assert res["answer"].startswith("WALL-E is a lonely robot")
    assert len(res["source_chunks"]) == 1
    assert res["source_chunks"][0]["title"] == "WALL-E"
    assert len(res["conversation_history"]) == 2


def test_rag_prompt_construction(rag_pipeline):
    chunks = [{
        "title": "Inception", "release_year": 2010, "genre": "Sci-Fi",
        "director": "Nolan", "plot_snippet": "Dream theft"
    }]
    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    prompt = rag_pipeline.build_prompt("How does inception work?", chunks, history=history)

    assert "Inception (2010)" in prompt
    assert "Dream theft" in prompt
    assert "PREVIOUS CONVERSATION" in prompt
    assert "Hi" in prompt
