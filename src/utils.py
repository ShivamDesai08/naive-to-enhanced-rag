"""
Utility functions for the RAG system implementation.
"""

import logging
import os
import json
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    """Configuration class for the RAG system."""
    
    # Model configurations
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "microsoft/DialoGPT-medium"  # Fallback model
    
    # Vector database configurations
    MILVUS_DB_PATH = "rag_wikipedia_mini.db"
    COLLECTION_NAME = "rag_mini"
    
    # Retrieval configurations
    DEFAULT_TOP_K = 5
    EMBEDDING_DIM = 384  # For all-MiniLM-L6-v2
    
    # Evaluation configurations
    TEST_QUERY_LIMIT = 100
    EVALUATION_METRICS = ["f1", "exact_match", "faithfulness", "context_precision", "context_recall"]
    
    # File paths
    DATA_DIR = "data"
    RESULTS_DIR = "results"
    PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
    EVALUATION_DIR = os.path.join(DATA_DIR, "evaluation")
    PROCESSED_DATA_PATH = os.path.join(PROCESSED_DIR, "passages.parquet")
    EVALUATION_DATA_PATH = os.path.join(EVALUATION_DIR, "test_queries.csv")


def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        Config.DATA_DIR,
        Config.RESULTS_DIR,
        Config.PROCESSED_DIR,
        Config.EVALUATION_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def load_embedding_model(model_name: str = Config.EMBEDDING_MODEL) -> SentenceTransformer:
    """Load the sentence transformer model for embeddings."""
    try:
        model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Failed to load embedding model {model_name}: {e}")
        raise


def save_results(results: Dict[str, Any], filename: str):
    """Save results to JSON file."""
    filepath = os.path.join(Config.RESULTS_DIR, filename)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    converted_results = convert_numpy_types(results)
    
    with open(filepath, 'w') as f:
        json.dump(converted_results, f, indent=2)
    logger.info(f"Results saved to: {filepath}")


def load_results(filename: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    filepath = os.path.join(Config.RESULTS_DIR, filename)
    with open(filepath, 'r') as f:
        results = json.load(f)
    logger.info(f"Results loaded from: {filepath}")
    return results


def calculate_embedding_dimension(model_name: str) -> int:
    """Calculate the embedding dimension for a given model."""
    model = SentenceTransformer(model_name)
    # Get embedding dimension by encoding a sample text
    sample_embedding = model.encode("sample text")
    return len(sample_embedding)


def preprocess_text(text: str) -> str:
    """Basic text preprocessing."""
    if pd.isna(text) or text == "":
        return ""
    
    # Remove extra whitespace and normalize
    text = str(text).strip()
    return text


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate that DataFrame has required columns and no empty rows."""
    # Check required columns
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    # Check for empty rows
    empty_rows = df[required_columns].isna().any(axis=1).sum()
    if empty_rows > 0:
        logger.warning(f"Found {empty_rows} rows with missing values")
    
    return True


def create_prompt_template(system_prompt: str = None) -> str:
    """Create a standardized prompt template."""
    if system_prompt is None:
        system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
        Use only the information from the context to answer the question. If the context doesn't contain 
        enough information to answer the question, say so."""
    
    template = f"""{system_prompt}

Context: {{context}}

Question: {{question}}

Answer:"""
    
    return template


def format_context(passages: List[str], max_length: int = 2000) -> str:
    """Format retrieved passages into context string."""
    if not passages:
        return "No relevant context found."
    
    context_parts = []
    current_length = 0
    
    for i, passage in enumerate(passages):
        if current_length + len(passage) > max_length:
            break
        context_parts.append(f"[{i+1}] {passage}")
        current_length += len(passage)
    
    return "\n\n".join(context_parts)


def log_ai_usage(tool_name: str, purpose: str, input_query: str, output_usage: str, verification: str):
    """Log AI tool usage for academic integrity documentation."""
    log_entry = {
        "tool": tool_name,
        "purpose": purpose,
        "input": input_query,
        "output_usage": output_usage,
        "verification": verification,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    # Save to AI usage log file
    log_file = os.path.join(Config.RESULTS_DIR, "ai_usage_log.json")
    
    # Load existing logs or create new list
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = json.load(f)
    else:
        logs = []
    
    logs.append(log_entry)
    
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)
    
    logger.info(f"AI usage logged: {tool_name} - {purpose}")


def get_device():
    """Get the appropriate device for computation (GPU if available, else CPU)."""
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("Using GPU for computation")
    else:
        device = "cpu"
        logger.info("Using CPU for computation")
    
    return device


def print_system_info():
    """Print system information for reproducibility."""
    logger.info("=== System Information ===")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
    logger.info(f"Device: {get_device()}")
    logger.info("==========================")
