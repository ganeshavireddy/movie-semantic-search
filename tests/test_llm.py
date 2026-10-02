import pytest
from src.llm import BaseLLMProvider, GeminiLLMProvider, OllamaLLMProvider, get_llm_provider


class MockLLMProvider(BaseLLMProvider):
    def generate(self, prompt: str, system_instruction: str = "") -> str:
        return "Mock Answer citing [WALL-E (2008)]."


def test_mock_llm_provider():
    provider = MockLLMProvider()
    res = provider.generate("What is WALL-E?")
    assert "WALL-E" in res


def test_gemini_provider_missing_key():
    provider = GeminiLLMProvider(api_key="")
    with pytest.raises(RuntimeError) as exc_info:
        provider.generate("Test prompt")
    assert "GEMINI_API_KEY is missing" in str(exc_info.value)
