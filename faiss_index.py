"""
faiss_index.py

FAISS index module for the Solar Still RAG System.
Builds, saves, loads, and queries a FAISS index using cosine similarity.

Author: Research Team
Journal: LLM-Powered Interactive Assistant for Solar Still Design Optimization
"""

import logging
import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Configure module logger
logger = logging.getLogger(__name__)

# Paths
EMBEDDINGS_DIR = Path(__file__).parent.parent / "embeddings"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "embeddings.npy"
METADATA_PATH = EMBEDDINGS_DIR / "metadata.json"
INDEX_DIR = Path(__file__).parent.parent / "index"
INDEX_PATH = INDEX_DIR / "faiss_index.bin"
NORMALIZED_INDEX_PATH = INDEX_DIR / "faiss_index_normalized.bin"


class FAISSIndexManager:
 """
 Manages FAISS index construction, persistence, and similarity search
 for solar still document embeddings.
 """

 def __init__(self, embedding_dim: int = 384):
 """
 Initialize the FAISS index manager.

 Args:
 embedding_dim: Dimension of the embedding vectors.
 """
 self.embedding_dim = embedding_dim
 self.index: Optional[faiss.Index] = None
 self.metadata: List[Dict[str, Any]] = []
 self.is_normalized = False

 def build_faiss_index(
 self,
 embeddings_path: Optional[Path] = None,
 metadata_path: Optional[Path] = None,
 use_cosine: bool = True
 ) -> faiss.Index:
 """
 Build a FAISS index from saved embeddings.

 Args:
 embeddings_path: Path to the .npy embeddings file.
 metadata_path: Path to the metadata JSON file.
 use_cosine: If True, L2-normalize vectors for cosine similarity.

 Returns:
 faiss.Index: Built FAISS index.
 """
 embeddings_path = embeddings_path or EMBEDDINGS_PATH
 metadata_path = metadata_path or METADATA_PATH

 if not embeddings_path.exists():
 raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
 if not metadata_path.exists():
 raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

 logger.info(f"Loading embeddings from: {embeddings_path}")
 embeddings = np.load(embeddings_path).astype(np.float32)

 logger.info(f"Loading metadata from: {metadata_path}")
 with open(metadata_path, "r", encoding="utf-8") as f:
 self.metadata = json.load(f)

 logger.info(f"Embeddings shape: {embeddings.shape}")
 logger.info(f"Metadata records: {len(self.metadata)}")

 if embeddings.shape[0] != len(self.metadata):
 raise ValueError(
 f"Embedding count ({embeddings.shape[0]}) does not match "
 f"metadata count ({len(self.metadata)})"
 )

 # Create FlatIP index (inner product = cosine sim for normalized vectors)
 self.index = faiss.IndexFlatIP(self.embedding_dim)

 if use_cosine:
 logger.info("Normalizing embeddings for cosine similarity...")
 # L2 normalize each vector
 faiss.normalize_L2(embeddings)
 self.is_normalized = True

 self.index.add(embeddings)
 logger.info(f"FAISS index built with {self.index.ntotal} vectors.")
 return self.index

 def save_index(self, index_path: Optional[Path] = None) -> Path:
 """
 Save the FAISS index to disk.

 Args:
 index_path: Path for the saved index file.

 Returns:
 Path: Path to the saved index file.
 """
 if self.index is None:
 raise ValueError("No index to save. Call build_faiss_index() first.")

 index_path = index_path or (
 NORMALIZED_INDEX_PATH if self.is_normalized else INDEX_PATH
 )
 index_path.parent.mkdir(parents=True, exist_ok=True)

 logger.info(f"Saving FAISS index to: {index_path}")
 faiss.write_index(self.index, str(index_path))
 logger.info("FAISS index saved successfully.")
 return index_path

 def load_index(self, index_path: Optional[Path] = None) -> faiss.Index:
 """
 Load a FAISS index from disk.

 Args:
 index_path: Path to the saved index file.

 Returns:
 faiss.Index: Loaded FAISS index.
 """
 # Try normalized index first, then fall back to standard
 if index_path is None:
 if NORMALIZED_INDEX_PATH.exists():
 index_path = NORMALIZED_INDEX_PATH
 elif INDEX_PATH.exists():
 index_path = INDEX_PATH
 else:
 raise FileNotFoundError("No FAISS index file found.")

 if not index_path.exists():
 raise FileNotFoundError(f"Index file not found: {index_path}")

 logger.info(f"Loading FAISS index from: {index_path}")
 self.index = faiss.read_index(str(index_path))
 self.is_normalized = "normalized" in index_path.name

 # Load metadata
 metadata_path = METADATA_PATH
 if metadata_path.exists():
 with open(metadata_path, "r", encoding="utf-8") as f:
 self.metadata = json.load(f)

 logger.info(f"FAISS index loaded. Vectors: {self.index.ntotal}")
 return self.index

 def retrieve_top_k(
 self,
 query_embedding: np.ndarray,
 k: int = 3
 ) -> Tuple[List[Dict[str, Any]], List[float]]:
 """
 Retrieve the top-k most similar documents for a query embedding.

 Args:
 query_embedding: Query vector of shape (embedding_dim,) or (1, embedding_dim).
 k: Number of results to retrieve.

 Returns:
 Tuple[List[Dict], List[float]]: Retrieved documents and similarity scores.
 """
 if self.index is None:
 raise ValueError("Index not loaded. Call load_index() or build_faiss_index() first.")

 # Ensure 2D input
 if query_embedding.ndim == 1:
 query_embedding = query_embedding.reshape(1, -1)
 query_embedding = query_embedding.astype(np.float32)

 # Normalize query vector if using cosine similarity
 if self.is_normalized:
 faiss.normalize_L2(query_embedding)

 # Perform search
 scores, indices = self.index.search(query_embedding, k)

 # Retrieve documents and scores
 retrieved_docs = []
 retrieved_scores = scores[0].tolist()

 for idx in indices[0]:
 if idx < len(self.metadata):
 retrieved_docs.append(self.metadata[idx])
 else:
 logger.warning(f"Index {idx} out of metadata bounds.")
 retrieved_docs.append({"text": "", "metadata": {}})

 logger.info(f"Retrieved {len(retrieved_docs)} documents (k={k}).")
 return retrieved_docs, retrieved_scores


def run_faiss_pipeline(
 embeddings_path: Optional[Path] = None,
 metadata_path: Optional[Path] = None,
 use_cosine: bool = True
) -> Path:
 """
 Run the full FAISS index build and save pipeline.

 Args:
 embeddings_path: Path to embeddings .npy file.
 metadata_path: Path to metadata JSON file.
 use_cosine: If True, normalize vectors for cosine similarity.

 Returns:
 Path: Path to the saved FAISS index file.
 """
 manager = FAISSIndexManager()
 manager.build_faiss_index(embeddings_path, metadata_path, use_cosine)
 return manager.save_index()


if __name__ == "__main__":
 logging.basicConfig(
 level=logging.INFO,
 format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
 )

 try:
 index_path = run_faiss_pipeline()
 print(f"FAISS pipeline complete. Index saved to: {index_path}")
 except Exception as e:
 logger.error(f"FAISS pipeline failed: {e}")
 raise
