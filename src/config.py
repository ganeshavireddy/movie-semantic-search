from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Base Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = base_dir / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    cache_dir: Path = data_dir / "cache"

    # File Paths
    raw_csv_path: Path = raw_data_dir / "wiki_movie_plots_deduped.csv"
    processed_chunks_path: Path = processed_data_dir / "chunks.parquet"
    embeddings_cache_path: Path = cache_dir / "embeddings.npy"
    index_cache_path: Path = cache_dir / "faiss.index"
    metadata_cache_path: Path = cache_dir / "chunk_metadata.json"
    bm25_cache_path: Path = cache_dir / "bm25.pkl"

    # Embedding Model Settings
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # Vector Index Settings
    index_type: Literal["Flat", "HNSW", "IVF"] = "Flat"

    # Chunking Settings
    chunk_size_words: int = 200
    chunk_overlap_words: int = 30

    # LLM Settings
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    llm_provider: Literal["gemini", "ollama"] = "gemini"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"


settings = Settings()
