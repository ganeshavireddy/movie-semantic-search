import argparse
import time
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.ingest import ingest_data
from src.chunking import chunk_dataset
from src.embedder import MovieEmbedder
from src.index import MovieIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_pipeline(
    raw_csv_path: Path = settings.raw_csv_path,
    index_type: str = settings.index_type,
    force_rebuild_embeddings: bool = False
):
    """Run end-to-end data ingestion, chunking, embedding generation, and Faiss index building."""
    start_total = time.time()
    
    # 1. Ingestion
    logger.info("--- Step 1: Ingesting Data ---")
    df_clean = ingest_data(raw_csv_path)
    
    # 2. Chunking
    logger.info("--- Step 2: Chunking Plots ---")
    df_chunks = chunk_dataset(df_clean)
    
    # Save processed chunks parquet
    settings.processed_chunks_path.parent.mkdir(parents=True, exist_ok=True)
    df_chunks.to_parquet(settings.processed_chunks_path, index=False)
    logger.info(f"Saved processed chunks to {settings.processed_chunks_path}")

    # Convert chunks metadata to dict list
    chunk_records = df_chunks.to_dict(orient="records")
    chunk_texts = df_chunks["chunk_text"].tolist()

    # 3. Embedding (with disk caching)
    logger.info("--- Step 3: Generating Embeddings ---")
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    
    if settings.embeddings_cache_path.exists() and not force_rebuild_embeddings:
        logger.info(f"Loading cached embeddings from {settings.embeddings_cache_path}...")
        embeddings = np.load(settings.embeddings_cache_path)
        if len(embeddings) != len(chunk_texts):
            logger.warning("Cached embeddings length mismatch. Rebuilding embeddings...")
            force_rebuild_embeddings = True

    if not settings.embeddings_cache_path.exists() or force_rebuild_embeddings:
        embedder = MovieEmbedder()
        logger.info("Encoding chunks using SentenceTransformer...")
        
        # Using tqdm for progress display
        batch_size = 64
        embeddings_list = []
        for i in tqdm(range(0, len(chunk_texts), batch_size), desc="Encoding text chunks"):
            batch = chunk_texts[i : i + batch_size]
            emb_batch = embedder.encode(batch, normalize_embeddings=True)
            embeddings_list.append(emb_batch)
            
        embeddings = np.vstack(embeddings_list)
        np.save(settings.embeddings_cache_path, embeddings)
        logger.info(f"Cached {len(embeddings)} embeddings to {settings.embeddings_cache_path}")

    # 4. Building Faiss Index
    logger.info(f"--- Step 4: Building Faiss Index ({index_type}) ---")
    movie_index = MovieIndex(dimension=embeddings.shape[1])
    movie_index.build(embeddings, chunk_records, index_type=index_type)

    # 5. Saving Faiss Index & Metadata
    movie_index.save(settings.index_cache_path, settings.metadata_cache_path)
    
    elapsed_total = time.time() - start_total
    index_size_mb = settings.index_cache_path.stat().st_size / (1024 * 1024)
    logger.info(f"=== Pipeline Complete in {elapsed_total:.2f}s ===")
    logger.info(f"Total Movies: {len(df_clean)}, Chunks: {len(df_chunks)}")
    logger.info(f"Faiss Index Type: {index_type}, Size: {index_size_mb:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Movie Semantic Search Faiss Index")
    parser.add_argument("--index-type", type=str, default=settings.index_type, choices=["Flat", "HNSW", "IVF"], help="Faiss index type")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild embeddings cache")
    args = parser.parse_args()

    build_pipeline(index_type=args.index_type, force_rebuild_embeddings=args.rebuild)
