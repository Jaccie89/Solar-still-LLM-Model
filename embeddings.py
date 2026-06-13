"""
embeddings.py

Embedding generation module for the Solar Still RAG System.
Uses Sentence Transformers (all-MiniLM-L6-v2) to generate dense vector representations.

Author: Research Team
Journal: LLM-Powered Interactive Assistant for Solar Still Design Optimization
"""

import logging
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from sentence_transformers import SentenceTransformer

# Configure module logger
logger = logging.getLogger(__name__)

# Paths
DOCUMENTS_DIR = Path(__file__).parent.parent / "documents"
DOCUMENTS_PATH = DOCUMENTS_DIR / "documents.jsonl"
EMBEDDINGS_DIR = Path(__file__).parent.parent / "embeddings"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "embeddings.npy"
METADATA_PATH = EMBEDDINGS_DIR / "metadata.json"

# Model configuration
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384 # Output dimension of all-MiniLM-L6-v2


class EmbeddingGenerator:
 """
 Generates and manages sentence embeddings for solar still document records.
 """

 def __init__(self, model_name: str = MODEL_NAME):
 """
 Initialize the embedding generator.

 Args:
 model_name: Name of the SentenceTransformer model to use.
 """
 self.model_name = model_name
 self.model: Optional[SentenceTransformer] = None
 self.documents: List[Dict[str, Any]] = []
 self.embeddings: Optional[np.ndarray] = None

 def load_documents(self, documents_path: Optional[Path] = None) -> List[Dict[str, Any]]:
 """
 Load documents from a JSONL file.

 Args:
 documents_path: Path to the JSONL file. Defaults to documents/documents.jsonl

 Returns:
 List[Dict[str, Any]]: List of document dictionaries.
 """
 documents_path = documents_path or DOCUMENTS_PATH

 if not documents_path.exists():
 raise FileNotFoundError(f"Documents file not found at: {documents_path}")

 logger.info(f"Loading documents from: {documents_path}")
 self.documents = []

 with open(documents_path, "r", encoding="utf-8") as f:
 for line_num, line in enumerate(f, 1):
 try:
 doc = json.loads(line.strip())
 self.documents.append(doc)
 except json.JSONDecodeError as e:
 logger.warning(f"Skipping malformed JSON at line {line_num}: {e}")

 logger.info(f"Loaded {len(self.documents)} documents.")
 return self.documents

 def load_embedding_model(self) -> SentenceTransformer:
 """
 Load the SentenceTransformer model.

 Returns:
 SentenceTransformer: Loaded model instance.
 """
 logger.info(f"Loading embedding model: {self.model_name}")
 self.model = SentenceTransformer(self.model_name)
 logger.info(f"Model loaded successfully. Device: {self.model.device}")
 return self.model

 def create_embeddings(self, texts: Optional[List[str]] = None) -> np.ndarray:
 """
 Generate embeddings for a list of text strings.

 Args:
 texts: List of text strings. If None, uses loaded documents.

 Returns:
 np.ndarray: Array of embeddings with shape (n_documents, embedding_dim).
 """
 if self.model is None:
 self.load_embedding_model()

 if texts is None:
 if not self.documents:
 raise ValueError("No documents loaded. Call load_documents() first.")
 texts = [doc["text"] for doc in self.documents]

 logger.info(f"Generating embeddings for {len(texts)} documents...")
 self.embeddings = self.model.encode(
 texts,
 batch_size=32,
 show_progress_bar=True,
 convert_to_numpy=True,
 normalize_embeddings=False # Normalization handled by FAISS index
 )

 logger.info(f"Embeddings generated. Shape: {self.embeddings.shape}")
 return self.embeddings

 def save_embeddings(
 self,
 embeddings_path: Optional[Path] = None,
 metadata_path: Optional[Path] = None
 ) -> Tuple[Path, Path]:
 """
 Save embeddings and metadata to disk.

 Args:
 embeddings_path: Path for the .npy embeddings file.
 metadata_path: Path for the metadata JSON file.

 Returns:
 Tuple[Path, Path]: Paths to saved embeddings and metadata files.
 """
 if self.embeddings is None:
 raise ValueError("No embeddings to save. Call create_embeddings() first.")

 embeddings_path = embeddings_path or EMBEDDINGS_PATH
 metadata_path = metadata_path or METADATA_PATH

 # Ensure directories exist
 embeddings_path.parent.mkdir(parents=True, exist_ok=True)

 logger.info(f"Saving embeddings to: {embeddings_path}")
 np.save(embeddings_path, self.embeddings)

 # Save metadata (document text + original metadata)
 metadata = []
 for doc in self.documents:
 metadata.append({
 "text": doc["text"],
 "metadata": doc["metadata"]
 })

 logger.info(f"Saving metadata to: {metadata_path}")
 with open(metadata_path, "w", encoding="utf-8") as f:
 json.dump(metadata, f, indent=2, ensure_ascii=False)

 logger.info("Embeddings and metadata saved successfully.")
 return embeddings_path, metadata_path


def run_embedding_pipeline(
 documents_path: Optional[Path] = None,
 model_name: str = MODEL_NAME
) -> Tuple[Path, Path]:
 """
 Run the full embedding generation pipeline.

 Args:
 documents_path: Path to the JSONL documents file.
 model_name: SentenceTransformer model name.

 Returns:
 Tuple[Path, Path]: Paths to saved embeddings and metadata files.
 """
 generator = EmbeddingGenerator(model_name)
 generator.load_documents(documents_path)
 generator.load_embedding_model()
 generator.create_embeddings()
 return generator.save_embeddings()


if __name__ == "__main__":
 logging.basicConfig(
 level=logging.INFO,
 format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
 )

 try:
 emb_path, meta_path = run_embedding_pipeline()
 print(f"Embedding pipeline complete.")
 print(f" Embeddings: {emb_path}")
 print(f" Metadata: {meta_path}")
 except Exception as e:
 logger.error(f"Embedding pipeline failed: {e}")
 raise
