# Setup Instructions for Assignment 2 RAG System

## Prerequisites

- Python 3.8 or higher
- Git (for version control)
- At least 4GB of available RAM
- 2GB of free disk space

## Quick Setup

1. **Navigate to the project directory:**
   ```bash
   cd assignment2-rag
   ```

2. **Run the setup script:**
   ```bash
   python setup.py
   ```

3. **Activate the virtual environment:**
   ```bash
   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

4. **Verify installation:**
   ```bash
   python -c "import torch, transformers, sentence_transformers; print('All packages installed successfully!')"
   ```

## Manual Setup (Alternative)

If the automated setup fails, follow these manual steps:

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Create Directories
```bash
mkdir -p data/{processed,evaluation} results docs
```

## Running the System

### 1. Data Exploration
Start with the data exploration notebook:
```bash
jupyter notebook notebooks/data_exploration.ipynb
```

### 2. Naive RAG Implementation
Run the naive RAG system:
```python
from src.naive_rag import NaiveRAGSystem

# Initialize system
rag_system = NaiveRAGSystem()

# Load data and build index
passages_df = rag_system.load_data()
rag_system.setup_milvus_database(passages_df)
rag_system.create_search_index()

# Test a query
result = rag_system.query("What is machine learning?")
print(result)
```

### 3. Evaluation
Run evaluation on test queries:
```python
from src.evaluation import evaluate_naive_rag

# Load test queries
test_queries_df = rag_system.load_test_queries()

# Evaluate system
evaluation_results, detailed_results = evaluate_naive_rag(rag_system, test_queries_df, limit=100)
print(evaluation_results)
```

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues:**
   - The system will automatically fall back to CPU if GPU is not available
   - For GPU support, ensure you have compatible CUDA drivers

2. **Memory Issues:**
   - Reduce batch sizes in the configuration
   - Use smaller embedding models if needed
   - Process data in smaller chunks

3. **Milvus Connection Issues:**
   - Ensure the database file path is writable
   - Check that no other process is using the same database file

4. **Import Errors:**
   - Ensure virtual environment is activated
   - Reinstall requirements: `pip install -r requirements.txt --force-reinstall`

### Performance Optimization

1. **For faster embedding generation:**
   - Use GPU if available
   - Increase batch size in sentence-transformers
   - Use smaller embedding models for development

2. **For faster search:**
   - Create appropriate indexes in Milvus
   - Use smaller top-k values for initial testing
   - Optimize search parameters

## Dataset Information

The system uses the RAG Mini Wikipedia dataset:
- **Passages**: 3,200 Wikipedia passages
- **Test Queries**: Questions with ground truth answers
- **Format**: Parquet files from HuggingFace datasets

The dataset will be automatically downloaded when first accessed.

## Configuration

Key configuration parameters can be modified in `src/utils.py`:

```python
class Config:
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Embedding model
    DEFAULT_TOP_K = 5                      # Number of retrieved passages
    EMBEDDING_DIM = 384                    # Embedding dimension
    TEST_QUERY_LIMIT = 100                 # Limit for evaluation
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the error logs in the console output
3. Ensure all dependencies are correctly installed
4. Verify Python version compatibility

## Next Steps

After successful setup:
1. Run data exploration notebook
2. Implement naive RAG system
3. Evaluate performance
4. Implement enhancements
5. Run advanced evaluation with RAGAs
