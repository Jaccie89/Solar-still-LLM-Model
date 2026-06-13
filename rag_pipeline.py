"""
rag_pipeline.py

RAG Pipeline module for the Solar Still RAG System.
Orchestrates context retrieval, prompt construction, and GPT response generation.

Author: Research Team
Journal: LLM-Powered Interactive Assistant for Solar Still Design Optimization
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

from faiss_index import FAISSIndexManager

# Configure module logger
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.0
MAX_TOKENS = 1024
TOP_K = 3
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# System prompt for the RAG assistant
SYSTEM_PROMPT = """You are an expert assistant for solar still design optimization.
You answer questions based ONLY on the provided context from experimental records.
If the context does not contain enough information, state that clearly.
Always cite the experimental conditions (date, basin type, materials) when referencing records.
Be precise, technical, and concise."""


class RAGPipeline:
 """
 End-to-end RAG pipeline for solar still query answering.
 """

 def __init__(
 self,
 openai_model: str = MODEL_NAME,
 temperature: float = TEMPERATURE,
 top_k: int = TOP_K,
 embedding_model: str = EMBEDDING_MODEL
 ):
 """
 Initialize the RAG pipeline.

 Args:
 openai_model: OpenAI GPT model name.
 temperature: Sampling temperature (0 for deterministic).
 top_k: Number of context records to retrieve.
 embedding_model: SentenceTransformer model for query embeddings.
 """
 self.openai_model = openai_model
 self.temperature = temperature
 self.top_k = top_k
 self.embedding_model_name = embedding_model

 self.llm: Optional[ChatOpenAI] = None
 self.embedding_model: Optional[SentenceTransformer] = None
 self.faiss_manager: Optional[FAISSIndexManager] = None

 def load_openai_model(self) -> ChatOpenAI:
 """
 Initialize the OpenAI LLM via LangChain.

 Returns:
 ChatOpenAI: Configured LangChain ChatOpenAI instance.
 """
 api_key = os.getenv("OPENAI_API_KEY")
 if not api_key:
 raise ValueError("OPENAI_API_KEY not found. Set it in your .env file.")

 logger.info(f"Initializing OpenAI model: {self.openai_model}")
 self.llm = ChatOpenAI(
 model=self.openai_model,
 temperature=self.temperature,
 max_tokens=self.MAX_TOKENS,
 api_key=api_key
 )
 logger.info("OpenAI model initialized successfully.")
 return self.llm

 def load_components(self) -> None:
 """
 Load all pipeline components: embedding model and FAISS index.
 """
 logger.info("Loading embedding model...")
 self.embedding_model = SentenceTransformer(self.embedding_model_name)

 logger.info("Loading FAISS index...")
 self.faiss_manager = FAISSIndexManager()
 self.faiss_manager.load_index()

 def retrieve_context(self, query: str) -> Tuple[List[Dict[str, Any]], List[float]]:
 """
 Retrieve top-k relevant documents for a user query.

 Args:
 query: User's natural language query.

 Returns:
 Tuple[List[Dict], List[float]]: Retrieved documents and similarity scores.
 """
 if self.embedding_model is None or self.faiss_manager is None:
 self.load_components()

 logger.info(f"Embedding query: '{query[:50]}...'")
 query_embedding = self.embedding_model.encode(
 query,
 convert_to_numpy=True,
 normalize_embeddings=False
 )

 docs, scores = self.faiss_manager.retrieve_top_k(query_embedding, self.top_k)
 return docs, scores

 def create_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
 """
 Construct the RAG prompt from the user query and retrieved context.

 Args:
 query: User's question.
 context_docs: List of retrieved document dictionaries.

 Returns:
 str: Formatted prompt string.
 """
 context_text = ""
 for i, doc in enumerate(context_docs, 1):
 context_text += f"\n[Record {i}]\n{doc['text']}\n"

 prompt = f"""{SYSTEM_PROMPT}

--- CONTEXT (Experimental Records) ---
{context_text}
--- END CONTEXT ---

--- USER QUESTION ---
{query}
--- END QUESTION ---

Provide a helpful, evidence-based answer using the context above."""

 return prompt

 def generate_response(self, prompt: str) -> str:
 """
 Generate a response from the OpenAI model.

 Args:
 prompt: The constructed RAG prompt.

 Returns:
 str: Generated response text.
 """
 if self.llm is None:
 self.load_openai_model()

 logger.info("Generating GPT response...")
 messages = [
 SystemMessage(content=SYSTEM_PROMPT),
 HumanMessage(content=prompt)
 ]

 response = self.llm.invoke(messages)
 answer = response.content
 logger.info("Response generated successfully.")
 return answer

 def query(self, user_query: str) -> Dict[str, Any]:
 """
 Execute a full RAG query: retrieve context and generate answer.

 Args:
 user_query: User's natural language question.

 Returns:
 Dict[str, Any]: Dictionary containing answer, context, and scores.
 """
 logger.info(f"Processing query: '{user_query}'")

 # Retrieve context
 context_docs, scores = self.retrieve_context(user_query)

 # Build prompt
 prompt = self.create_prompt(user_query, context_docs)

 # Generate answer
 answer = self.generate_response(prompt)

 return {
 "query": user_query,
 "answer": answer,
 "context": context_docs,
 "scores": scores,
 "prompt": prompt
 }


def initialize_pipeline() -> RAGPipeline:
 """
 Initialize and return a ready-to-use RAG pipeline.

 Returns:
 RAGPipeline: Initialized pipeline instance.
 """
 pipeline = RAGPipeline()
 pipeline.load_openai_model()
 pipeline.load_components()
 return pipeline


if __name__ == "__main__":
 logging.basicConfig(
 level=logging.INFO,
 format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
 )

 # Example usage
 try:
 pipeline = initialize_pipeline()
 result = pipeline.query(
 "What basin liner material produces the highest yield?"
 )
 print("\n=== ANSWER ===")
 print(result["answer"])
 except Exception as e:
 logger.error(f"Pipeline execution failed: {e}")
 raise
