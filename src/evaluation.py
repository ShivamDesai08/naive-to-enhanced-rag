"""
Evaluation utilities for RAG systems using various metrics.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import json
import os
from datasets import Dataset
from evaluate import load
import re
import string
from collections import Counter

# RAGAs imports
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logging.warning("RAGAs not available. Install with: pip install ragas")

from utils import Config, save_results, log_ai_usage

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Comprehensive evaluator for RAG systems using multiple metrics.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.squad_metric = None
        self._load_metrics()
        logger.info("Initialized RAGEvaluator")
    
    def _load_metrics(self):
        """Load evaluation metrics."""
        try:
            # Load SQuAD metric for F1 and EM
            self.squad_metric = load("squad")
            logger.info("Loaded SQuAD metric")
        except Exception as e:
            logger.error(f"Failed to load SQuAD metric: {e}")
    
    def calculate_f1_score(self, prediction: str, ground_truth: str) -> float:
        """Calculate F1 score between prediction and ground truth."""
        def normalize_answer(s):
            """Lower text and remove punctuation, articles and extra whitespace."""
            def remove_articles(text):
                regex = re.compile(r'\b(a|an|the)\b', re.IGNORECASE)
                return re.sub(regex, ' ', text)
            
            def white_space_fix(text):
                return ' '.join(text.split())
            
            def remove_punc(text):
                exclude = set(string.punctuation)
                return ''.join(ch for ch in text if ch not in exclude)
            
            def lower(text):
                return text.lower()
            
            return white_space_fix(remove_articles(remove_punc(lower(s))))
        
        def f1_score(prediction, ground_truth):
            prediction_tokens = normalize_answer(prediction).split()
            ground_truth_tokens = normalize_answer(ground_truth).split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                return 0
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            return f1
        
        return f1_score(prediction, ground_truth)
    
    def calculate_exact_match(self, prediction: str, ground_truth: str) -> float:
        """Calculate exact match score."""
        def normalize_answer(s):
            """Lower text and remove punctuation, articles and extra whitespace."""
            def remove_articles(text):
                regex = re.compile(r'\b(a|an|the)\b', re.IGNORECASE)
                return re.sub(regex, ' ', text)
            
            def white_space_fix(text):
                return ' '.join(text.split())
            
            def remove_punc(text):
                exclude = set(string.punctuation)
                return ''.join(ch for ch in text if ch not in exclude)
            
            def lower(text):
                return text.lower()
            
            return white_space_fix(remove_articles(remove_punc(lower(s))))
        
        return float(normalize_answer(prediction) == normalize_answer(ground_truth))
    
    def evaluate_with_squad_metric(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Evaluate using SQuAD metric (F1 and EM)."""
        if self.squad_metric is None:
            logger.error("SQuAD metric not loaded")
            return {"f1": 0.0, "exact_match": 0.0}
        
        try:
            # Format data for SQuAD metric
            formatted_predictions = [{"id": str(i), "prediction_text": pred} for i, pred in enumerate(predictions)]
            formatted_references = [{"id": str(i), "answers": {"text": [ref], "answer_start": [0]}} for i, ref in enumerate(references)]
            
            # Calculate metrics
            results = self.squad_metric.compute(
                predictions=formatted_predictions,
                references=formatted_references
            )
            
            logger.info(f"SQuAD metrics: {results}")
            return results
            
        except Exception as e:
            logger.error(f"SQuAD evaluation failed: {e}")
            return {"f1": 0.0, "exact_match": 0.0}
    
    def evaluate_with_ragas(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate using RAGAs framework."""
        if not RAGAS_AVAILABLE:
            logger.error("RAGAs not available")
            return {}
        
        try:
            # Convert to RAGAs dataset format
            ragas_data = {
                "question": [item["question"] for item in data],
                "answer": [item["answer"] for item in data],
                "contexts": [item["contexts"] for item in data],
                "ground_truths": [[item["ground_truth"]] for item in data]
            }
            
            # Create dataset
            dataset = Dataset.from_dict(ragas_data)
            
            # Evaluate with available metrics only
            available_metrics = [faithfulness, answer_relevancy]
            
            # Try to add context metrics if possible
            try:
                # Check if we have enough data for context metrics
                if len(data) > 0 and data[0].get("contexts"):
                    available_metrics.extend([context_recall, context_precision])
            except Exception as e:
                logger.warning(f"Context metrics not available: {e}")
            
            # Evaluate
            result = evaluate(dataset, metrics=available_metrics)
            
            logger.info(f"RAGAs metrics: {result}")
            return result
            
        except Exception as e:
            logger.error(f"RAGAs evaluation failed: {e}")
            return {}
    
    def evaluate_system(self, results: List[Dict[str, Any]], use_ragas: bool = True) -> Dict[str, Any]:
        """Comprehensive evaluation of RAG system results."""
        logger.info(f"Evaluating {len(results)} results")
        
        # Extract predictions and references
        predictions = [result["answer"] for result in results]
        references = [result["ground_truth"] for result in results]
        
        # Basic metrics
        evaluation_results = {
            "total_queries": len(results),
            "successful_queries": len([r for r in results if not r["answer"].startswith("Error")]),
            "failed_queries": len([r for r in results if r["answer"].startswith("Error")])
        }
        
        # SQuAD metrics (F1 and EM)
        squad_metrics = self.evaluate_with_squad_metric(predictions, references)
        evaluation_results.update(squad_metrics)
        
        # RAGAs metrics if available
        if use_ragas and RAGAS_AVAILABLE:
            ragas_metrics = self.evaluate_with_ragas(results)
            evaluation_results.update(ragas_metrics)
        
        # Additional custom metrics
        evaluation_results.update(self._calculate_custom_metrics(results))
        
        logger.info(f"Evaluation completed: {evaluation_results}")
        return evaluation_results
    
    def _calculate_custom_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate custom metrics."""
        metrics = {}
        
        # Average context length
        context_lengths = []
        for result in results:
            if result["contexts"]:
                total_length = sum(len(ctx) for ctx in result["contexts"])
                context_lengths.append(total_length)
        
        if context_lengths:
            metrics["avg_context_length"] = np.mean(context_lengths)
            metrics["max_context_length"] = np.max(context_lengths)
            metrics["min_context_length"] = np.min(context_lengths)
        
        # Average number of contexts used
        num_contexts = [len(result["contexts"]) for result in results]
        metrics["avg_num_contexts"] = np.mean(num_contexts)
        
        # Response length statistics
        response_lengths = [len(result["answer"]) for result in results]
        metrics["avg_response_length"] = np.mean(response_lengths)
        metrics["max_response_length"] = np.max(response_lengths)
        metrics["min_response_length"] = np.min(response_lengths)
        
        return metrics
    
    def compare_systems(self, naive_results: Dict[str, Any], enhanced_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two RAG systems."""
        comparison = {
            "naive_system": naive_results,
            "enhanced_system": enhanced_results,
            "improvements": {}
        }
        
        # Calculate improvements for each metric
        for metric in ["f1", "exact_match", "faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
            if metric in naive_results and metric in enhanced_results:
                naive_val = naive_results[metric]
                enhanced_val = enhanced_results[metric]
                
                if naive_val > 0:
                    improvement = ((enhanced_val - naive_val) / naive_val) * 100
                else:
                    improvement = 0 if enhanced_val == 0 else float('inf')
                
                comparison["improvements"][metric] = {
                    "naive": naive_val,
                    "enhanced": enhanced_val,
                    "improvement_percent": improvement,
                    "absolute_improvement": enhanced_val - naive_val
                }
        
        return comparison
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any], system_name: str = "RAG System") -> str:
        """Generate a comprehensive evaluation report."""
        report = f"""
# Evaluation Report: {system_name}

## Summary
- **Total Queries**: {evaluation_results.get('total_queries', 0)}
- **Successful Queries**: {evaluation_results.get('successful_queries', 0)}
- **Failed Queries**: {evaluation_results.get('failed_queries', 0)}
- **Success Rate**: {(evaluation_results.get('successful_queries', 0) / max(evaluation_results.get('total_queries', 1), 1)) * 100:.2f}%

## Performance Metrics

### Basic Metrics
- **F1 Score**: {evaluation_results.get('f1', 0):.4f}
- **Exact Match**: {evaluation_results.get('exact_match', 0):.4f}

### RAGAs Metrics
- **Faithfulness**: {evaluation_results.get('faithfulness', 0):.4f}
- **Answer Relevancy**: {evaluation_results.get('answer_relevancy', 0):.4f}
- **Context Recall**: {evaluation_results.get('context_recall', 0):.4f}
- **Context Precision**: {evaluation_results.get('context_precision', 0):.4f}

### System Statistics
- **Average Context Length**: {evaluation_results.get('avg_context_length', 0):.2f} characters
- **Average Number of Contexts**: {evaluation_results.get('avg_num_contexts', 0):.2f}
- **Average Response Length**: {evaluation_results.get('avg_response_length', 0):.2f} characters

## Analysis
The system achieved a {evaluation_results.get('f1', 0):.4f} F1 score and {evaluation_results.get('exact_match', 0):.4f} exact match rate.
The faithfulness score of {evaluation_results.get('faithfulness', 0):.4f} indicates the quality of answer generation based on retrieved context.
"""
        return report
    
    def save_evaluation_results(self, results: Dict[str, Any], filename: str):
        """Save evaluation results to file."""
        save_results(results, filename)
        logger.info(f"Evaluation results saved to {filename}")
    
    def log_evaluation_metrics(self, results: Dict[str, Any], system_name: str):
        """Log evaluation metrics for analysis."""
        logger.info(f"=== {system_name} Evaluation Results ===")
        for metric, value in results.items():
            if isinstance(value, (int, float)):
                logger.info(f"{metric}: {value:.4f}")
            else:
                logger.info(f"{metric}: {value}")
        logger.info("=" * 50)


def evaluate_naive_rag(rag_system, test_queries_df: pd.DataFrame, limit: int = None) -> Dict[str, Any]:
    """Evaluate a naive RAG system."""
    evaluator = RAGEvaluator()
    
    # Run evaluation
    results = rag_system.evaluate_on_test_set(test_queries_df, limit=limit)
    
    # Calculate metrics
    evaluation_results = evaluator.evaluate_system(results)
    
    # Log results
    evaluator.log_evaluation_metrics(evaluation_results, "Naive RAG")
    
    return evaluation_results, results


def evaluate_enhanced_rag(rag_system, test_queries_df: pd.DataFrame, limit: int = None) -> Dict[str, Any]:
    """Evaluate an enhanced RAG system."""
    evaluator = RAGEvaluator()
    
    # Run evaluation
    results = rag_system.evaluate_on_test_set(test_queries_df, limit=limit)
    
    # Calculate metrics
    evaluation_results = evaluator.evaluate_system(results)
    
    # Log results
    evaluator.log_evaluation_metrics(evaluation_results, "Enhanced RAG")
    
    return evaluation_results, results


def compare_rag_systems(naive_results: Dict[str, Any], enhanced_results: Dict[str, Any]) -> Dict[str, Any]:
    """Compare naive and enhanced RAG systems."""
    evaluator = RAGEvaluator()
    comparison = evaluator.compare_systems(naive_results, enhanced_results)
    
    logger.info("=== System Comparison ===")
    for metric, data in comparison["improvements"].items():
        logger.info(f"{metric}: {data['naive']:.4f} -> {data['enhanced']:.4f} ({data['improvement_percent']:.2f}% improvement)")
    logger.info("=" * 30)
    
    return comparison
