"""
Parameter experimentation framework for RAG systems.
Tests different embedding sizes, retrieval strategies, and prompting approaches.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import json
import os
from sentence_transformers import SentenceTransformer
import time
from itertools import product

from naive_rag import NaiveRAGSystem
from evaluation import RAGEvaluator
from utils import Config, save_results, log_ai_usage

logger = logging.getLogger(__name__)


class RAGExperimenter:
    """
    Comprehensive experimentation framework for RAG systems.
    """
    
    def __init__(self):
        """Initialize the experimenter."""
        self.evaluator = RAGEvaluator()
        self.results = []
        logger.info("Initialized RAGExperimenter")
    
    def test_embedding_models(self, test_queries_df: pd.DataFrame, limit: int = 50) -> Dict[str, Any]:
        """Test different embedding models and sizes."""
        logger.info("Starting embedding model experimentation")
        
        # Different embedding models to test
        embedding_models = [
            ("all-MiniLM-L6-v2", 384),      # Current default
            ("all-mpnet-base-v2", 768),     # Larger, potentially better
            ("paraphrase-MiniLM-L6-v2", 384), # Alternative 384-dim model
        ]
        
        results = {}
        
        for model_name, expected_dim in embedding_models:
            logger.info(f"Testing embedding model: {model_name}")
            
            try:
                # Create new RAG system with different embedding model
                rag_system = NaiveRAGSystem(embedding_model_name=model_name)
                
                # Load and setup data
                passages_df = rag_system.load_data()
                rag_system.setup_milvus_database(passages_df)
                rag_system.create_search_index()
                
                # Test on subset of queries
                test_subset = test_queries_df.head(limit)
                evaluation_results, detailed_results = self.evaluator.evaluate_system(
                    rag_system.evaluate_on_test_set(test_subset)
                )
                
                results[model_name] = {
                    "embedding_dimension": expected_dim,
                    "evaluation_metrics": evaluation_results,
                    "sample_results": detailed_results[:5]  # Store first 5 for analysis
                }
                
                # Cleanup
                rag_system.cleanup()
                
                logger.info(f"Completed testing {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to test {model_name}: {e}")
                results[model_name] = {"error": str(e)}
        
        return results
    
    def test_retrieval_strategies(self, test_queries_df: pd.DataFrame, limit: int = 50) -> Dict[str, Any]:
        """Test different retrieval strategies (top-k values)."""
        logger.info("Starting retrieval strategy experimentation")
        
        # Different top-k values to test
        top_k_values = [1, 3, 5, 10]
        
        # Initialize base RAG system
        rag_system = NaiveRAGSystem()
        passages_df = rag_system.load_data()
        rag_system.setup_milvus_database(passages_df)
        rag_system.create_search_index()
        
        results = {}
        
        for top_k in top_k_values:
            logger.info(f"Testing top-{top_k} retrieval")
            
            try:
                # Test with different top-k values
                test_subset = test_queries_df.head(limit)
                detailed_results = []
                
                for idx, row in test_subset.iterrows():
                    result = rag_system.query(row['question'], top_k=top_k, use_multiple_contexts=True)
                    result['ground_truth'] = row.get('answer', '')
                    detailed_results.append(result)
                
                # Evaluate results
                evaluation_results = self.evaluator.evaluate_system(detailed_results)
                
                results[f"top_{top_k}"] = {
                    "top_k": top_k,
                    "evaluation_metrics": evaluation_results,
                    "sample_results": detailed_results[:3]  # Store first 3 for analysis
                }
                
                logger.info(f"Completed testing top-{top_k}")
                
            except Exception as e:
                logger.error(f"Failed to test top-{top_k}: {e}")
                results[f"top_{top_k}"] = {"error": str(e)}
        
        # Cleanup
        rag_system.cleanup()
        
        return results
    
    def test_prompting_strategies(self, test_queries_df: pd.DataFrame, limit: int = 30) -> Dict[str, Any]:
        """Test different prompting strategies."""
        logger.info("Starting prompting strategy experimentation")
        
        # Different prompting strategies
        prompting_strategies = {
            "instruction_prompt": """You are a helpful assistant that answers questions based on the provided context. 
Use only the information from the context to answer the question. If the context doesn't contain 
enough information to answer the question, say so.""",
            
            "cot_prompt": """You are an expert assistant. Think step by step to answer the question based on the provided context.
Step 1: Analyze the question
Step 2: Review the context
Step 3: Provide your answer based on the context
If the context doesn't contain enough information, explain what's missing.""",
            
            "persona_prompt": """You are a knowledgeable historian and researcher. Answer the question based on the provided context with accuracy and detail.
Use only the information from the context. If the context is insufficient, clearly state what information is missing.""",
            
            "simple_prompt": """Answer the question using the provided context. If you cannot answer based on the context, say so."""
        }
        
        # Initialize base RAG system
        rag_system = NaiveRAGSystem()
        passages_df = rag_system.load_data()
        rag_system.setup_milvus_database(passages_df)
        rag_system.create_search_index()
        
        results = {}
        
        for strategy_name, system_prompt in prompting_strategies.items():
            logger.info(f"Testing prompting strategy: {strategy_name}")
            
            try:
                test_subset = test_queries_df.head(limit)
                detailed_results = []
                
                for idx, row in test_subset.iterrows():
                    # Use custom prompt template
                    result = rag_system.query(row['question'], top_k=1)
                    
                    # Override the answer with custom prompt
                    if result['contexts']:
                        context = result['contexts'][0]
                        prompt = f"""{system_prompt}

Context: {context}

Question: {row['question']}

Answer:"""
                        
                        # Generate answer with custom prompt
                        answer = rag_system.generate_answer(row['question'], context, prompt)
                        result['answer'] = answer
                    
                    result['ground_truth'] = row.get('answer', '')
                    detailed_results.append(result)
                
                # Evaluate results
                evaluation_results = self.evaluator.evaluate_system(detailed_results)
                
                results[strategy_name] = {
                    "system_prompt": system_prompt,
                    "evaluation_metrics": evaluation_results,
                    "sample_results": detailed_results[:3]  # Store first 3 for analysis
                }
                
                logger.info(f"Completed testing {strategy_name}")
                
            except Exception as e:
                logger.error(f"Failed to test {strategy_name}: {e}")
                results[strategy_name] = {"error": str(e)}
        
        # Cleanup
        rag_system.cleanup()
        
        return results
    
    def run_comprehensive_experiment(self, test_queries_df: pd.DataFrame, 
                                   embedding_limit: int = 30,
                                   retrieval_limit: int = 30,
                                   prompting_limit: int = 20) -> Dict[str, Any]:
        """Run comprehensive parameter experimentation."""
        logger.info("Starting comprehensive RAG experimentation")
        
        start_time = time.time()
        
        # Run all experiments
        experiments = {
            "embedding_models": self.test_embedding_models(test_queries_df, embedding_limit),
            "retrieval_strategies": self.test_retrieval_strategies(test_queries_df, retrieval_limit),
            "prompting_strategies": self.test_prompting_strategies(test_queries_df, prompting_limit)
        }
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Compile results
        comprehensive_results = {
            "experiment_metadata": {
                "total_time_seconds": total_time,
                "embedding_limit": embedding_limit,
                "retrieval_limit": retrieval_limit,
                "prompting_limit": prompting_limit,
                "timestamp": pd.Timestamp.now().isoformat()
            },
            "experiments": experiments,
            "summary": self._generate_experiment_summary(experiments)
        }
        
        # Save results
        save_results(comprehensive_results, "parameter_experimentation_results.json")
        
        logger.info(f"Comprehensive experimentation completed in {total_time:.2f} seconds")
        
        return comprehensive_results
    
    def _generate_experiment_summary(self, experiments: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of experiment results."""
        summary = {}
        
        # Embedding models summary
        if "embedding_models" in experiments:
            embedding_results = experiments["embedding_models"]
            best_embedding = None
            best_f1 = 0
            
            for model_name, result in embedding_results.items():
                if "evaluation_metrics" in result:
                    f1_score = result["evaluation_metrics"].get("f1", 0)
                    if f1_score > best_f1:
                        best_f1 = f1_score
                        best_embedding = model_name
            
            summary["best_embedding_model"] = {
                "model": best_embedding,
                "f1_score": best_f1
            }
        
        # Retrieval strategies summary
        if "retrieval_strategies" in experiments:
            retrieval_results = experiments["retrieval_strategies"]
            best_retrieval = None
            best_f1 = 0
            
            for strategy_name, result in retrieval_results.items():
                if "evaluation_metrics" in result:
                    f1_score = result["evaluation_metrics"].get("f1", 0)
                    if f1_score > best_f1:
                        best_f1 = f1_score
                        best_retrieval = strategy_name
            
            summary["best_retrieval_strategy"] = {
                "strategy": best_retrieval,
                "f1_score": best_f1
            }
        
        # Prompting strategies summary
        if "prompting_strategies" in experiments:
            prompting_results = experiments["prompting_strategies"]
            best_prompting = None
            best_f1 = 0
            
            for strategy_name, result in prompting_results.items():
                if "evaluation_metrics" in result:
                    f1_score = result["evaluation_metrics"].get("f1", 0)
                    if f1_score > best_f1:
                        best_f1 = f1_score
                        best_prompting = strategy_name
            
            summary["best_prompting_strategy"] = {
                "strategy": best_prompting,
                "f1_score": best_f1
            }
        
        return summary
    
    def generate_experiment_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive experiment report."""
        report = f"""
# RAG Parameter Experimentation Report

## Experiment Overview
- **Total Time**: {results['experiment_metadata']['total_time_seconds']:.2f} seconds
- **Date**: {results['experiment_metadata']['timestamp']}
- **Test Limits**: Embedding({results['experiment_metadata']['embedding_limit']}), Retrieval({results['experiment_metadata']['retrieval_limit']}), Prompting({results['experiment_metadata']['prompting_limit']})

## Summary of Best Results
"""
        
        summary = results.get("summary", {})
        
        if "best_embedding_model" in summary:
            best_emb = summary["best_embedding_model"]
            report += f"- **Best Embedding Model**: {best_emb['model']} (F1: {best_emb['f1_score']:.4f})\n"
        
        if "best_retrieval_strategy" in summary:
            best_ret = summary["best_retrieval_strategy"]
            report += f"- **Best Retrieval Strategy**: {best_ret['strategy']} (F1: {best_ret['f1_score']:.4f})\n"
        
        if "best_prompting_strategy" in summary:
            best_prompt = summary["best_prompting_strategy"]
            report += f"- **Best Prompting Strategy**: {best_prompt['strategy']} (F1: {best_prompt['f1_score']:.4f})\n"
        
        report += "\n## Detailed Results\n"
        
        # Add detailed results for each experiment type
        for exp_type, exp_results in results["experiments"].items():
            report += f"\n### {exp_type.replace('_', ' ').title()}\n"
            
            for name, result in exp_results.items():
                if "evaluation_metrics" in result:
                    metrics = result["evaluation_metrics"]
                    report += f"- **{name}**: F1={metrics.get('f1', 0):.4f}, EM={metrics.get('exact_match', 0):.4f}\n"
                elif "error" in result:
                    report += f"- **{name}**: ERROR - {result['error']}\n"
        
        return report


def run_parameter_experiments():
    """Main function to run parameter experiments."""
    print("🔬 Starting RAG Parameter Experimentation")
    print("=" * 60)
    
    try:
        # Initialize experimenter
        experimenter = RAGExperimenter()
        
        # Load test queries
        rag_system = NaiveRAGSystem()
        test_queries_df = rag_system.load_test_queries()
        
        print(f"Loaded {len(test_queries_df)} test queries")
        
        # Run comprehensive experiments
        results = experimenter.run_comprehensive_experiment(
            test_queries_df,
            embedding_limit=20,    # Smaller limits for faster testing
            retrieval_limit=20,
            prompting_limit=15
        )
        
        # Generate and save report
        report = experimenter.generate_experiment_report(results)
        
        # Save report
        with open("results/experiment_report.md", "w") as f:
            f.write(report)
        
        print("\nExperiment Results Summary:")
        print("=" * 40)
        
        summary = results.get("summary", {})
        for key, value in summary.items():
            if isinstance(value, dict):
                print(f"{key}: {value}")
        
        print(f"\nExperiments completed successfully!")
        print(f"Detailed report saved to: results/experiment_report.md")
        print(f"Results saved to: results/parameter_experimentation_results.json")
        
        return results
        
    except Exception as e:
        print(f"Experimentation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    run_parameter_experiments()
