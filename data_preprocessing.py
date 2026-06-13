"""
data_preprocessing.py

Data preprocessing module for the Solar Still RAG System.
Handles dataset loading, validation, text conversion, and document persistence.

Author: Research Team
Journal: LLM-Powered Interactive Assistant for Solar Still Design Optimization
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

# Configure module logger
logger = logging.getLogger(__name__)

# Dataset configuration
DATA_DIR = Path(__file__).parent.parent / "data"
DATASET_PATH = DATA_DIR / "solar_still_dataset_1012.csv"
DOCUMENTS_DIR = Path(__file__).parent.parent / "documents"
DOCUMENTS_PATH = DOCUMENTS_DIR / "documents.jsonl"

# Expected dataset columns
EXPECTED_COLUMNS = [
 "Date", "Time", "Basin_Type", "Basin_Liner", "Heat_Storage",
 "Wick_Material", "Water_Depth_mm", "Glass_Thickness_mm",
 "Ambient_Temp_C", "Solar_Irradiance_Wm2", "Relative_Humidity_pct",
 "Yield_mLm2day"
]


class DatasetPreprocessor:
 """
 Preprocessor class for solar still experimental datasets.
 Converts structured tabular data into natural-language document records.
 """

 def __init__(self, dataset_path: Optional[Path] = None):
 """
 Initialize the preprocessor with a dataset path.

 Args:
 dataset_path: Path to the CSV dataset. Defaults to DATA_DIR/solar_still_dataset_1012.csv
 """
 self.dataset_path = dataset_path or DATASET_PATH
 self.df: Optional[pd.DataFrame] = None
 self.documents: List[Dict[str, Any]] = []

 def load_dataset(self) -> pd.DataFrame:
 """
 Load the solar still dataset from CSV.

 Returns:
 pd.DataFrame: Loaded dataset.

 Raises:
 FileNotFoundError: If the dataset file does not exist.
 pd.errors.EmptyDataError: If the file is empty.
 """
 logger.info(f"Loading dataset from: {self.dataset_path}")

 if not self.dataset_path.exists():
 raise FileNotFoundError(f"Dataset not found at: {self.dataset_path}")

 try:
 self.df = pd.read_csv(self.dataset_path)
 logger.info(f"Dataset loaded successfully. Shape: {self.df.shape}")
 return self.df
 except pd.errors.EmptyDataError as e:
 logger.error("Dataset file is empty.")
 raise e
 except Exception as e:
 logger.error(f"Error loading dataset: {e}")
 raise

 def validate_dataset(self) -> bool:
 """
 Validate that the dataset contains all required columns.

 Returns:
 bool: True if validation passes.

 Raises:
 ValueError: If required columns are missing.
 """
 if self.df is None:
 raise ValueError("Dataset not loaded. Call load_dataset() first.")

 logger.info("Validating dataset schema...")
 missing_cols = set(EXPECTED_COLUMNS) - set(self.df.columns)

 if missing_cols:
 error_msg = f"Missing required columns: {missing_cols}"
 logger.error(error_msg)
 raise ValueError(error_msg)

 # Check for null values in critical columns
 critical_cols = ["Yield_mLm2day", "Solar_Irradiance_Wm2", "Ambient_Temp_C"]
 null_counts = self.df[critical_cols].isnull().sum()
 if null_counts.any():
 logger.warning(f"Null values detected in critical columns: {null_counts[null_counts > 0]}")

 logger.info("Dataset validation passed.")
 return True

 def convert_to_text(self) -> List[Dict[str, Any]]:
 """
 Convert each row of the dataset into a descriptive natural-language text record.

 Returns:
 List[Dict[str, Any]]: List of document dictionaries with 'text' and 'metadata' keys.
 """
 if self.df is None:
 raise ValueError("Dataset not loaded. Call load_dataset() first.")

 logger.info("Converting dataset rows to natural-language text...")
 self.documents = []

 for idx, row in self.df.iterrows():
 # Construct descriptive text
 text = (
 f"On {row['Date']} at {row['Time']}, an experiment was conducted using a "
 f"{row['Basin_Type']} basin type solar still with a {row['Basin_Liner']} basin liner "
 f"and {row['Heat_Storage']} heat storage material. The wick material used was "
 f"{row['Wick_Material']}. The water depth was set to {row['Water_Depth_mm']} mm "
 f"and the glass thickness was {row['Glass_Thickness_mm']} mm. During this experiment, "
 f"the ambient temperature was {row['Ambient_Temp_C']}°C, solar irradiance was "
 f"{row['Solar_Irradiance_Wm2']} W/m², and relative humidity was "
 f"{row['Relative_Humidity_pct']}%. The measured daily yield was "
 f"{row['Yield_mLm2day']} mL/m²/day."
 )

 # Store metadata for reference
 metadata = {
 "row_index": idx,
 "date": row["Date"],
 "time": row["Time"],
 "basin_type": row["Basin_Type"],
 "basin_liner": row["Basin_Liner"],
 "heat_storage": row["Heat_Storage"],
 "wick_material": row["Wick_Material"],
 "water_depth_mm": row["Water_Depth_mm"],
 "glass_thickness_mm": row["Glass_Thickness_mm"],
 "ambient_temp_c": row["Ambient_Temp_C"],
 "solar_irradiance_wm2": row["Solar_Irradiance_Wm2"],
 "relative_humidity_pct": row["Relative_Humidity_pct"],
 "yield_mlm2day": row["Yield_mLm2day"],
 }

 self.documents.append({"text": text, "metadata": metadata})

 logger.info(f"Converted {len(self.documents)} records to text.")
 return self.documents

 def save_documents(self, output_path: Optional[Path] = None) -> Path:
 """
 Save the converted documents to a JSONL file.

 Args:
 output_path: Output path for the JSONL file. Defaults to documents/documents.jsonl

 Returns:
 Path: Path to the saved file.
 """
 if not self.documents:
 raise ValueError("No documents to save. Call convert_to_text() first.")

 output_path = output_path or DOCUMENTS_PATH
 output_path.parent.mkdir(parents=True, exist_ok=True)

 logger.info(f"Saving documents to: {output_path}")

 with open(output_path, "w", encoding="utf-8") as f:
 for doc in self.documents:
 f.write(json.dumps(doc, ensure_ascii=False) + "\n")

 logger.info(f"Saved {len(self.documents)} documents to {output_path}")
 return output_path


def run_preprocessing_pipeline(dataset_path: Optional[Path] = None) -> Path:
 """
 Run the full preprocessing pipeline: load, validate, convert, and save.

 Args:
 dataset_path: Optional custom path to the dataset CSV.

 Returns:
 Path: Path to the saved JSONL documents file.
 """
 preprocessor = DatasetPreprocessor(dataset_path)
 preprocessor.load_dataset()
 preprocessor.validate_dataset()
 preprocessor.convert_to_text()
 return preprocessor.save_documents()


if __name__ == "__main__":
 # Configure logging
 logging.basicConfig(
 level=logging.INFO,
 format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
 )

 try:
 output_path = run_preprocessing_pipeline()
 print(f"Preprocessing complete. Documents saved to: {output_path}")
 except Exception as e:
 logger.error(f"Preprocessing failed: {e}")
 raise
