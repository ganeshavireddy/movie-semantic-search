import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import faiss
import numpy as np
from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MovieIndex:
    """Faiss vector index manager supporting Flat, HNSW, and IVF index types with metadata mapping."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index: faiss.Index = None
        self.metadata: List[Dict[str, Any]] = []
        self.index_type: str = "Flat"

    def create_index(self, index_type: str = "Flat", num_vectors: int = 0) -> faiss.Index:
        """Create a Faiss index of the specified type."""
        index_type_clean = index_type.upper()

        if index_type_clean == "FLAT":
            logger.info(f"Creating Faiss IndexFlatIP (dim={self.dimension})...")
            return faiss.IndexFlatIP(self.dimension)

        elif index_type_clean == "HNSW":
            M = 32  # Number of connections per node
            logger.info(f"Creating Faiss IndexHNSWFlat (dim={self.dimension}, M={M})...")
            return faiss.IndexHNSWFlat(self.dimension, M, faiss.METRIC_INNER_PRODUCT)

        elif index_type_clean == "IVF":
            nlist = max(1, min(int(np.sqrt(max(num_vectors, 1))), 100))
            logger.info(f"Creating Faiss IndexIVFFlat (dim={self.dimension}, nlist={nlist})...")
            quantizer = faiss.IndexFlatIP(self.dimension)
            return faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)

        else:
            raise ValueError(f"Unsupported index type '{index_type}'. Expected 'Flat', 'HNSW', or 'IVF'.")

    def build(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]], index_type: str = "Flat") -> None:
        """Build Faiss index from embeddings and store associated chunk metadata."""
        if len(embeddings) != len(metadata):
            raise ValueError(f"Embeddings count ({len(embeddings)}) must match metadata count ({len(metadata)})")

        start_time = time.time()
        self.index_type = index_type
        num_vectors = len(embeddings)

        self.index = self.create_index(index_type, num_vectors=num_vectors)

        if not self.index.is_trained:
            logger.info("Training Faiss index...")
            self.index.train(embeddings)

        logger.info(f"Adding {num_vectors} vectors to Faiss index...")
        self.index.add(embeddings)
        self.metadata = metadata

        if index_type.upper() == "IVF":
            # Set nprobe for IVF search
            self.index.nprobe = max(1, min(10, self.index.nlist))

        elapsed = time.time() - start_time
        logger.info(f"Successfully built Faiss index ({index_type}) with {self.index.ntotal} vectors in {elapsed:.3f}s")

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        """Search top_k nearest neighbors for a single L2-normalized query vector."""
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("Index is not built or empty.")

        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        query_vector = query_vector.astype(np.float32)

        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append((float(score), self.metadata[idx]))

        return results

    def save(self, index_path: Path, metadata_path: Path) -> None:
        """Save Faiss index binary and metadata JSON to disk."""
        if self.index is None:
            raise RuntimeError("Cannot save an uninitialized index.")

        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving Faiss index to {index_path}...")
        faiss.write_index(self.index, str(index_path))

        logger.info(f"Saving metadata to {metadata_path}...")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        size_mb = index_path.stat().st_size / (1024 * 1024)
        logger.info(f"Index successfully saved to disk (Size: {size_mb:.2f} MB)")

    def load(self, index_path: Path, metadata_path: Path) -> None:
        """Load Faiss index binary and metadata JSON from disk."""
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Index or metadata file missing: {index_path}, {metadata_path}")

        logger.info(f"Loading Faiss index from {index_path}...")
        self.index = faiss.read_index(str(index_path))
        self.dimension = self.index.d

        logger.info(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        logger.info(f"Loaded Faiss index with {self.index.ntotal} vectors and {len(self.metadata)} metadata items.")
