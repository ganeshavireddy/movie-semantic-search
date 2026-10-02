import logging
import os
from typing import Optional
import httpx
from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BaseLLMProvider:
    """Base interface for LLM providers."""

    def generate(self, prompt: str, system_instruction: str = "") -> str:
        raise NotImplementedError


class GeminiLLMProvider(BaseLLMProvider):
    """Google Gemini API provider using official google-genai SDK."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = settings.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")

        model = model_name or settings.gemini_model or "gemini-3.5-flash-lite"
        if "1.5" in str(model):
            model = "gemini-3.5-flash-lite"
        self.model_name = model
        self.client = None

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                logger.info(f"Initialized Gemini LLM Provider (primary: {self.model_name}).")
            except Exception as e:
                logger.warning(f"Failed to initialize google-genai client: {e}")
        else:
            logger.warning("GEMINI_API_KEY is not set. GeminiLLMProvider will fall back if invoked.")

    def generate(self, prompt: str, system_instruction: str = "") -> str:
        if not self.api_key or not self.client:
            raise RuntimeError("GEMINI_API_KEY is missing. Please set GEMINI_API_KEY in environment or .env file.")

        full_prompt = prompt
        if system_instruction:
            full_prompt = f"System Instruction:\n{system_instruction}\n\nUser Question / Task:\n{prompt}"

        # Model candidates to try in order (only modern supported Gemini models)
        candidates = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-2.5-flash"]
        if self.model_name and self.model_name not in candidates and "1.5" not in self.model_name:
            candidates.append(self.model_name)

        unique_models = []
        for m in candidates:
            if m and m not in unique_models:
                unique_models.append(m)

        last_err = None
        for m in unique_models:
            try:
                response = self.client.models.generate_content(
                    model=m,
                    contents=full_prompt
                )
                return response.text.strip()
            except Exception as e:
                last_err = e
                logger.warning(f"Gemini API model '{m}' failed: {e}. Retrying fallback candidate...")

        logger.error(f"All Gemini API model candidates failed. Last error: {last_err}")
        raise last_err


class OllamaLLMProvider(BaseLLMProvider):
    """Local Ollama HTTP provider."""

    def __init__(self, base_url: str = settings.ollama_base_url, model: str = settings.ollama_model):
        self.base_url = base_url.rstrip("/")
        self.model = model
        logger.info(f"Initialized Ollama Provider ({self.model} at {self.base_url}).")

    def generate(self, prompt: str, system_instruction: str = "") -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_instruction,
            "stream": False
        }
        try:
            response = httpx.post(url, json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except Exception as e:
            logger.error(f"Ollama generation error at {url}: {e}")
            raise RuntimeError(f"Ollama server connection failed: {e}")


def get_llm_provider(provider_type: Optional[str] = None) -> BaseLLMProvider:
    """Factory function to get LLM provider based on config or argument."""
    selected = (provider_type or settings.llm_provider).lower()
    
    if selected == "gemini":
        api_key = settings.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        if api_key:
            return GeminiLLMProvider(api_key=api_key)
        else:
            logger.warning("Gemini API key missing. Attempting Ollama fallback...")
            try:
                return OllamaLLMProvider()
            except Exception:
                return GeminiLLMProvider()  # Will raise clear error on generate if key is missing
    elif selected == "ollama":
        return OllamaLLMProvider()

    return GeminiLLMProvider()
