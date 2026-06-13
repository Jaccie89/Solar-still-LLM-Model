"""
app.py

Streamlit web application for the Solar Still RAG System.
Provides an interactive interface for querying the RAG pipeline.

Author: Research Team
Journal: LLM-Powered Interactive Assistant for Solar Still Design Optimization
"""

import logging
import os
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

# Import pipeline components
from rag_pipeline import RAGPipeline, initialize_pipeline

# Configure logging
logging.basicConfig(
 level=logging.INFO,
 format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
 page_title="Solar Still Design Optimizer",
 page_icon="