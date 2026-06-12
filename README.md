# Solar-still-LLM-Model
# Large Language Model-Powered Interactive Assistant for Solar Still Design Optimization

## Overview

This repository contains the dataset, source code, and reproducibility artifacts associated with the research paper:

**"Large Language Model-Powered Interactive Assistant for Solar Still Design Optimization"**

The proposed system integrates a Retrieval-Augmented Generation (RAG) framework with experimentally validated solar still performance data to provide explainable and evidence-based design recommendations through natural language interaction.

The framework combines:

- HuggingFace Sentence Transformers
- FAISS Vector Similarity Search
- LangChain Retrieval Pipeline
- OpenAI GPT-based Language Model
- Streamlit Web Application

---

## Repository Structure

```text
SolarStill-RAG-LLM/
│
├── data/
│   ├── solar_still_dataset_1012.csv
│   └── sample_dataset.csv
│
├── src/
│   ├── data_preprocessing.py
│   ├── embeddings.py
│   ├── faiss_index.py
│   ├── rag_pipeline.py
│   └── app.py
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── requirements.txt
├── Dockerfile
├── README.md
└── LICENSE
Dataset Description

The experimental dataset consists of 1012 records representing hourly observations collected during a three-month experimental period (March–May 2025).

Each record contains:

Variable	Description
Date	Observation date
Time	Observation time
Basin_Type	Single or stepped basin configuration
Basin_Liner	Basin liner configuration
Heat_Storage	Heat storage material
Wick_Material	Wick material used
Water_Depth_mm	Basin water depth
Glass_Thickness_mm	Glass cover thickness
Ambient_Temp_C	Ambient temperature
Solar_Irradiance_Wm2	Solar radiation intensity
Relative_Humidity_pct	Ambient relative humidity
Yield_mLm2day	Freshwater productivity
System Architecture

The proposed framework follows a Retrieval-Augmented Generation (RAG) workflow:

Experimental data collection
Data preprocessing and transformation
Conversion of records into descriptive text
Embedding generation using Sentence Transformers
FAISS vector indexing
Similarity-based retrieval
GPT-based response generation
User interaction through Streamlit
