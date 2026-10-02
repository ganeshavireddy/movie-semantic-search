import logging
from typing import List, Dict, Any
import pandas as pd
from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 200, chunk_overlap: int = 30) -> List[str]:
    """Split text into word chunks with a specified overlap."""
    words = text.split()
    if not words:
        return []
    
    if len(words) <= chunk_size:
        return [" ".join(words)]
    
    chunks = []
    step = chunk_size - chunk_overlap
    if step <= 0:
        step = max(1, chunk_size // 2)

    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
            
        # Stop if we reached the end of the text
        if i + chunk_size >= len(words):
            break
            
    return chunks


def chunk_dataset(
    df: pd.DataFrame,
    chunk_size_words: int = settings.chunk_size_words,
    chunk_overlap_words: int = settings.chunk_overlap_words
) -> pd.DataFrame:
    """Transform a movie DataFrame into a chunk DataFrame preserving all movie metadata."""
    logger.info(f"Chunking {len(df)} movies (chunk_size={chunk_size_words}, overlap={chunk_overlap_words})...")
    
    chunk_records: List[Dict[str, Any]] = []

    for movie_idx, row in df.iterrows():
        plot = str(row.get("plot", ""))
        chunks = chunk_text(plot, chunk_size=chunk_size_words, chunk_overlap=chunk_overlap_words)
        
        for chunk_idx, text in enumerate(chunks):
            record = {
                "chunk_id": f"movie_{movie_idx}_chunk_{chunk_idx}",
                "movie_index": int(movie_idx),
                "title": str(row.get("title", "")),
                "release_year": int(row.get("release_year", 0)),
                "origin": str(row.get("origin", "")),
                "director": str(row.get("director", "")),
                "cast": str(row.get("cast", "")),
                "genre": str(row.get("genre", "")),
                "wiki_page": str(row.get("wiki_page", "")),
                "chunk_index": chunk_idx,
                "total_chunks": len(chunks),
                "chunk_text": text
            }
            chunk_records.append(record)

    chunks_df = pd.DataFrame(chunk_records)
    logger.info(f"Created {len(chunks_df)} total chunks from {len(df)} movies.")
    return chunks_df
