import logging
from typing import List, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MovieEmbedder:
    """Batch-encodes text using sentence-transformers and normalizes vectors to unit length."""

    def __init__(self, model_name: str = settings.embedding_model_name):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading SentenceTransformer model '{self.model_name}' on device '{self.device}'...")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        if hasattr(self.model, "get_embedding_dimension"):
            self.embedding_dim = self.model.get_embedding_dimension()
        else:
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """Encode texts into embeddings matrix."""
        if isinstance(texts, str):
            texts = [texts]
            
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        logger.info(f"Encoding {len(texts)} texts in batches of {batch_size}...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True
        )

        return embeddings.astype(np.float32)
