"""
Naive RAG implementation following the assignment requirements.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os

from utils import (
    Config, setup_directories, load_embedding_model, 
    preprocess_text, validate_dataframe, create_prompt_template,
    format_context, get_device, print_system_info, log_ai_usage
)

logger = logging.getLogger(__name__)


class NaiveRAGSystem:
    """
    Naive RAG system implementation with Milvus vector database.
    """
    
    def __init__(self, embedding_model_name: str = Config.EMBEDDING_MODEL):
        """Initialize the naive RAG system."""
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        self.milvus_client = None
        self.llm_model = None
        self.llm_tokenizer = None
        self.device = get_device()
        
        # Setup directories
        setup_directories()
        print_system_info()
        
        logger.info("Initialized NaiveRAGSystem")
    
    def load_embedding_model(self):
        """Load the sentence transformer model."""
        self.embedding_model = load_embedding_model(self.embedding_model_name)
        logger.info(f"Embedding model loaded: {self.embedding_model_name}")
    
    def load_llm_model(self, model_name: str = "microsoft/DialoGPT-medium"):
        """Load the language model for generation."""
        try:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llm_model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Add padding token if not present
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            
            self.llm_model.to(self.device)
            logger.info(f"LLM model loaded: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load LLM model {model_name}: {e}")
            raise
    
    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """Load the RAG Mini Wikipedia dataset."""
        if data_path is None:
            # Use the HuggingFace dataset URL from the starter code
            data_path = "hf://datasets/rag-datasets/rag-mini-wikipedia/data/passages.parquet/part.0.parquet"
        
        try:
            passages_df = pd.read_parquet(data_path)
            logger.info(f"Loaded dataset with shape: {passages_df.shape}")
            
            # Basic data validation
            if not validate_dataframe(passages_df, ["passage"]):
                raise ValueError("Dataset validation failed")
            
            # Clean the data
            passages_df = passages_df.dropna(subset=["passage"])
            passages_df["passage"] = passages_df["passage"].apply(preprocess_text)
            passages_df = passages_df[passages_df["passage"] != ""]
            
            # Add ID column if not present
            if "id" not in passages_df.columns:
                passages_df["id"] = range(len(passages_df))
            
            logger.info(f"Cleaned dataset shape: {passages_df.shape}")
            return passages_df
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if self.embedding_model is None:
            self.load_embedding_model()
        
        try:
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def create_milvus_schema(self) -> CollectionSchema:
        """Create the Milvus collection schema."""
        # Define field schemas
        id_field = FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=False
        )
        
        passage_field = FieldSchema(
            name="passage",
            dtype=DataType.VARCHAR,
            max_length=10000  # Adjust based on your data
        )
        
        embedding_field = FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=Config.EMBEDDING_DIM
        )
        
        # Create collection schema
        schema = CollectionSchema(
            fields=[id_field, passage_field, embedding_field],
            description="RAG Mini Wikipedia collection"
        )
        
        return schema
    
    def setup_milvus_database(self, passages_df: pd.DataFrame):
        """Set up Milvus database and insert data."""
        try:
            # Create Milvus client
            self.milvus_client = MilvusClient(Config.MILVUS_DB_PATH)
            logger.info("Created Milvus client")
            
            # Create schema
            schema = self.create_milvus_schema()
            
            # Create collection
            self.milvus_client.create_collection(
                collection_name=Config.COLLECTION_NAME,
                schema=schema
            )
            logger.info(f"Created collection: {Config.COLLECTION_NAME}")
            
            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.generate_embeddings(passages_df["passage"].tolist())
            
            # Prepare data for insertion
            rag_data = []
            for i, (idx, row) in enumerate(passages_df.iterrows()):
                rag_data.append({
                    "id": int(row["id"]),
                    "passage": str(row["passage"]),
                    "embedding": embeddings[i].tolist()
                })
            
            # Insert data
            logger.info("Inserting data into Milvus...")
            result = self.milvus_client.insert(
                collection_name=Config.COLLECTION_NAME,
                data=rag_data
            )
            logger.info(f"Insert result: {result}")
            
            # Sanity check
            stats = self.milvus_client.get_collection_stats(Config.COLLECTION_NAME)
            schema_info = self.milvus_client.describe_collection(Config.COLLECTION_NAME)
            
            logger.info(f"Entity count: {stats['row_count']}")
            logger.info(f"Collection schema: {schema_info}")
            
            # Create search index
            self.create_search_index()
            
        except Exception as e:
            logger.error(f"Failed to setup Milvus database: {e}")
            raise
    
    def create_search_index(self):
        """Create index on the embedding field for efficient search."""
        try:
            index_params = self.milvus_client.prepare_index_params()
            
            # Add index on embedding field
            index_params.add_index(
                field_name="embedding",
                index_type="IVF_FLAT",
                metric_type="L2",
                params={"nlist": 1024}
            )
            
            # Create the index
            self.milvus_client.create_index(
                collection_name=Config.COLLECTION_NAME,
                index_params=index_params
            )
            logger.info("Created search index")
            
            # Load collection into memory (required for search)
            self.milvus_client.load_collection(Config.COLLECTION_NAME)
            logger.info("Collection loaded into memory")
            
        except Exception as e:
            logger.error(f"Failed to create search index: {e}")
            raise
    
    def search(self, query: str, top_k: int = Config.DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """Search for relevant passages given a query."""
        if self.embedding_model is None:
            self.load_embedding_model()
        
        if self.milvus_client is None:
            raise ValueError("Milvus client not initialized. Run setup_milvus_database first.")
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Search in Milvus
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            
            results = self.milvus_client.search(
                collection_name=Config.COLLECTION_NAME,
                data=query_embedding.tolist(),
                anns_field="embedding",
                search_params=search_params,
                limit=top_k,
                output_fields=["passage", "id"]
            )
            
            # Format results
            formatted_results = []
            for hit in results[0]:
                formatted_results.append({
                    "id": hit["entity"]["id"],
                    "passage": hit["entity"]["passage"],
                    "score": hit["distance"]
                })
            
            logger.info(f"Retrieved {len(formatted_results)} results for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def generate_answer(self, query: str, context: str, prompt_template: str = None) -> str:
        """Generate an answer using the LLM."""
        if self.llm_model is None:
            self.load_llm_model()
        
        try:
            # Create prompt
            if prompt_template is None:
                prompt_template = create_prompt_template()
            
            prompt = prompt_template.format(context=context, question=query)
            
            # Tokenize input
            inputs = self.llm_tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.llm_tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the answer part (after "Answer:")
            if "Answer:" in response:
                answer = response.split("Answer:")[-1].strip()
            else:
                answer = response[len(prompt):].strip()
            
            logger.info("Generated answer successfully")
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"Error generating answer: {str(e)}"
    
    def query(self, question: str, top_k: int = 1, use_multiple_contexts: bool = False) -> Dict[str, Any]:
        """Query the RAG system with a question."""
        try:
            # Search for relevant passages
            search_results = self.search(question, top_k=top_k)
            
            if not search_results:
                return {
                    "question": question,
                    "answer": "No relevant context found.",
                    "contexts": [],
                    "search_results": []
                }
            
            # Prepare context
            if use_multiple_contexts:
                contexts = [result["passage"] for result in search_results]
                context = format_context(contexts)
            else:
                # Use only the top result
                context = search_results[0]["passage"]
                contexts = [context]
            
            # Generate answer
            answer = self.generate_answer(question, context)
            
            return {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "retrieved_ids": [result["id"] for result in search_results],
                "search_results": search_results
            }
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "question": question,
                "answer": f"Error processing query: {str(e)}",
                "contexts": [],
                "retrieved_ids": [],
                "search_results": []
            }
    
    def load_test_queries(self, queries_path: str = None) -> pd.DataFrame:
        """Load test queries for evaluation."""
        if queries_path is None:
            queries_path = "hf://datasets/rag-datasets/rag-mini-wikipedia/data/test.parquet/part.0.parquet"
        
        try:
            # Handle both Parquet and CSV files
            if queries_path.endswith('.csv'):
                queries_df = pd.read_csv(queries_path)
            else:
                queries_df = pd.read_parquet(queries_path)
            
            logger.info(f"Loaded test queries with shape: {queries_df.shape}")
            
            # Clean the data
            queries_df = queries_df.dropna(subset=["question"])
            queries_df["question"] = queries_df["question"].apply(preprocess_text)
            queries_df = queries_df[queries_df["question"] != ""]
            
            logger.info(f"Cleaned test queries shape: {queries_df.shape}")
            return queries_df
            
        except Exception as e:
            logger.error(f"Failed to load test queries: {e}")
            raise
    
    def evaluate_on_test_set(self, test_queries_df: pd.DataFrame, limit: int = None) -> List[Dict[str, Any]]:
        """Evaluate the system on test queries."""
        if limit:
            test_queries_df = test_queries_df.head(limit)
        
        results = []
        logger.info(f"Evaluating on {len(test_queries_df)} test queries")
        
        for idx, row in test_queries_df.iterrows():
            try:
                result = self.query(row["question"])
                result["ground_truth"] = row.get("answer", "")
                results.append(result)
                
                if (idx + 1) % 10 == 0:
                    logger.info(f"Processed {idx + 1}/{len(test_queries_df)} queries")
                    
            except Exception as e:
                logger.error(f"Failed to process query {idx}: {e}")
                results.append({
                    "question": row["question"],
                    "answer": f"Error: {str(e)}",
                    "contexts": [],
                    "search_results": [],
                    "ground_truth": row.get("answer", "")
                })
        
        logger.info(f"Completed evaluation on {len(results)} queries")
        return results
    
    def save_results(self, results: List[Dict[str, Any]], filename: str = "naive_results.json"):
        """Save evaluation results."""
        from utils import save_results
        save_results(results, filename)
    
    def cleanup(self):
        """Clean up resources."""
        if self.milvus_client:
            try:
                self.milvus_client.drop_collection(Config.COLLECTION_NAME)
                logger.info("Dropped Milvus collection")
            except Exception as e:
                logger.warning(f"Failed to drop collection: {e}")
        
        # Clear GPU memory if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared GPU memory")
