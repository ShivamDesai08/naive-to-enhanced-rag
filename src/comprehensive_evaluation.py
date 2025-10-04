"""
Comprehensive evaluation script for comparing naive and enhanced RAG systems.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import json
import os
import time

from naive_rag import NaiveRAGSystem
from enhanced_rag import EnhancedRAGSystem
from evaluation import RAGEvaluator, compare_rag_systems
from utils import Config, setup_directories, save_results

logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    """
    Comprehensive evaluator for comparing naive and enhanced RAG systems.
    """
    
    def __init__(self):
        """Initialize the comprehensive evaluator."""
        self.evaluator = RAGEvaluator()
        logger.info("Initialized ComprehensiveEvaluator")
    
    def evaluate_naive_system(self, test_queries_df: pd.DataFrame, limit: int = 50) -> Dict[str, Any]:
        """Evaluate the naive RAG system."""
        logger.info("Starting naive RAG system evaluation")
        
        try:
            # Initialize naive RAG system
            naive_rag = NaiveRAGSystem()
            
            # Load data and setup
            passages_df = naive_rag.load_data()
            naive_rag.setup_milvus_database(passages_df)
            naive_rag.create_search_index()
            
            # Run evaluation
            test_subset = test_queries_df.head(limit)
            results = naive_rag.evaluate_on_test_set(test_subset)
            
            # Calculate metrics
            evaluation_results = self.evaluator.evaluate_system(results)
            
            # Cleanup
            naive_rag.cleanup()
            
            logger.info("Completed naive RAG system evaluation")
            return {
                "system_type": "naive",
                "evaluation_metrics": evaluation_results,
                "detailed_results": results[:10]  # Store first 10 for analysis
            }
            
        except Exception as e:
            logger.error(f"Naive RAG evaluation failed: {e}")
            return {"system_type": "naive", "error": str(e)}
    
    def evaluate_enhanced_system(self, test_queries_df: pd.DataFrame, limit: int = 50) -> Dict[str, Any]:
        """Evaluate the enhanced RAG system."""
        logger.info("Starting enhanced RAG system evaluation")
        
        try:
            # Initialize enhanced RAG system
            enhanced_rag = EnhancedRAGSystem()
            
            # Load data and setup
            passages_df = enhanced_rag.load_data()
            enhanced_rag.setup_milvus_database(passages_df)
            enhanced_rag.create_search_index()
            
            # Run evaluation with enhanced features
            test_subset = test_queries_df.head(limit)
            results = []
            
            for idx, row in test_subset.iterrows():
                try:
                    result = enhanced_rag.query(
                        row["question"], 
                        top_k=3, 
                        use_query_rewriting=True, 
                        use_reranking=True
                    )
                    result["ground_truth"] = row.get("answer", "")
                    results.append(result)
                    
                    if (idx + 1) % 10 == 0:
                        logger.info(f"Processed {idx + 1}/{len(test_subset)} queries")
                        
                except Exception as e:
                    logger.error(f"Failed to process query {idx}: {e}")
                    results.append({
                        "question": row["question"],
                        "answer": f"Error: {str(e)}",
                        "contexts": [],
                        "search_results": [],
                        "ground_truth": row.get("answer", "")
                    })
            
            # Calculate metrics
            evaluation_results = self.evaluator.evaluate_system(results)
            
            # Cleanup
            enhanced_rag.cleanup()
            
            logger.info("Completed enhanced RAG system evaluation")
            return {
                "system_type": "enhanced",
                "evaluation_metrics": evaluation_results,
                "detailed_results": results[:10]  # Store first 10 for analysis
            }
            
        except Exception as e:
            logger.error(f"Enhanced RAG evaluation failed: {e}")
            return {"system_type": "enhanced", "error": str(e)}
    
    def run_comprehensive_evaluation(self, test_queries_df: pd.DataFrame, limit: int = 50) -> Dict[str, Any]:
        """Run comprehensive evaluation of both systems."""
        logger.info("Starting comprehensive RAG evaluation")
        
        start_time = time.time()
        
        # Evaluate both systems
        naive_results = self.evaluate_naive_system(test_queries_df, limit)
        enhanced_results = self.evaluate_enhanced_system(test_queries_df, limit)
        
        # Compare systems
        comparison = None
        if "evaluation_metrics" in naive_results and "evaluation_metrics" in enhanced_results:
            comparison = compare_rag_systems(
                naive_results["evaluation_metrics"],
                enhanced_results["evaluation_metrics"]
            )
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Compile comprehensive results
        comprehensive_results = {
            "evaluation_metadata": {
                "total_time_seconds": total_time,
                "test_queries_limit": limit,
                "timestamp": pd.Timestamp.now().isoformat()
            },
            "naive_system": naive_results,
            "enhanced_system": enhanced_results,
            "comparison": comparison,
            "summary": self._generate_summary(naive_results, enhanced_results, comparison)
        }
        
        logger.info(f"Comprehensive evaluation completed in {total_time:.2f} seconds")
        return comprehensive_results
    
    def _generate_summary(self, naive_results: Dict[str, Any], enhanced_results: Dict[str, Any], 
                         comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of evaluation results."""
        summary = {
            "evaluation_successful": True,
            "systems_evaluated": 0,
            "best_system": None,
            "key_improvements": []
        }
        
        # Check if evaluations were successful
        if "error" in naive_results:
            summary["evaluation_successful"] = False
            summary["naive_error"] = naive_results["error"]
        
        if "error" in enhanced_results:
            summary["evaluation_successful"] = False
            summary["enhanced_error"] = enhanced_results["error"]
        
        if not summary["evaluation_successful"]:
            return summary
        
        # Count successful evaluations
        if "evaluation_metrics" in naive_results:
            summary["systems_evaluated"] += 1
        if "evaluation_metrics" in enhanced_results:
            summary["systems_evaluated"] += 1
        
        # Determine best system
        if comparison and "improvements" in comparison:
            improvements = comparison["improvements"]
            
            # Check F1 score improvement
            if "f1" in improvements:
                f1_improvement = improvements["f1"]["improvement_percent"]
                if f1_improvement > 0:
                    summary["best_system"] = "enhanced"
                    summary["key_improvements"].append(f"F1 score improved by {f1_improvement:.2f}%")
                else:
                    summary["best_system"] = "naive"
            
            # Check other metrics
            for metric, data in improvements.items():
                if metric != "f1" and data["improvement_percent"] > 0:
                    summary["key_improvements"].append(
                        f"{metric} improved by {data['improvement_percent']:.2f}%"
                    )
        
        return summary
    
    def generate_evaluation_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive evaluation report."""
        report = f"""
# Comprehensive RAG System Evaluation Report

## Evaluation Overview
- **Total Time**: {results['evaluation_metadata']['total_time_seconds']:.2f} seconds
- **Date**: {results['evaluation_metadata']['timestamp']}
- **Test Queries**: {results['evaluation_metadata']['test_queries_limit']}

## System Performance

### Naive RAG System
"""
        
        if "error" in results["naive_system"]:
            report += f"- **Status**: ERROR - {results['naive_system']['error']}\n"
        else:
            metrics = results["naive_system"]["evaluation_metrics"]
            report += f"- **F1 Score**: {metrics.get('f1', 0):.4f}\n"
            report += f"- **Exact Match**: {metrics.get('exact_match', 0):.4f}\n"
            report += f"- **Success Rate**: {(metrics.get('successful_queries', 0) / max(metrics.get('total_queries', 1), 1)) * 100:.2f}%\n"
        
        report += "\n### Enhanced RAG System\n"
        
        if "error" in results["enhanced_system"]:
            report += f"- **Status**: ERROR - {results['enhanced_system']['error']}\n"
        else:
            metrics = results["enhanced_system"]["evaluation_metrics"]
            report += f"- **F1 Score**: {metrics.get('f1', 0):.4f}\n"
            report += f"- **Exact Match**: {metrics.get('exact_match', 0):.4f}\n"
            report += f"- **Success Rate**: {(metrics.get('successful_queries', 0) / max(metrics.get('total_queries', 1), 1)) * 100:.2f}%\n"
        
        # Add comparison section
        if results.get("comparison"):
            report += "\n## System Comparison\n"
            improvements = results["comparison"]["improvements"]
            
            for metric, data in improvements.items():
                report += f"- **{metric}**: {data['naive']:.4f} → {data['enhanced']:.4f} ({data['improvement_percent']:.2f}% improvement)\n"
        
        # Add summary
        if results.get("summary"):
            summary = results["summary"]
            report += f"\n## Summary\n"
            report += f"- **Best System**: {summary.get('best_system', 'N/A')}\n"
            report += f"- **Key Improvements**: {', '.join(summary.get('key_improvements', ['None']))}\n"
        
        return report


def run_comprehensive_evaluation():
    """Main function to run comprehensive evaluation."""
    print("🔬 Starting Comprehensive RAG Evaluation")
    print("=" * 60)
    
    try:
        # Setup directories
        setup_directories()
        
        # Initialize evaluator
        evaluator = ComprehensiveEvaluator()
        
        # Load test queries
        naive_rag = NaiveRAGSystem()
        test_queries_df = naive_rag.load_test_queries()
        
        print(f"Loaded {len(test_queries_df)} test queries")
        
        # Run comprehensive evaluation
        results = evaluator.run_comprehensive_evaluation(test_queries_df, limit=30)
        
        # Generate and save report
        report = evaluator.generate_evaluation_report(results)
        
        # Save results
        save_results(results, "comprehensive_evaluation_results.json")
        
        # Save report
        with open("results/comprehensive_evaluation_report.md", "w") as f:
            f.write(report)
        
        print("\n📊 Comprehensive Evaluation Results:")
        print("=" * 40)
        
        # Print summary
        if results.get("summary"):
            summary = results["summary"]
            print(f"Evaluation Successful: {summary.get('evaluation_successful', False)}")
            print(f"Systems Evaluated: {summary.get('systems_evaluated', 0)}")
            print(f"Best System: {summary.get('best_system', 'N/A')}")
            print(f"Key Improvements: {', '.join(summary.get('key_improvements', ['None']))}")
        
        print(f"\nComprehensive evaluation completed successfully!")
        print(f"Detailed report saved to: results/comprehensive_evaluation_report.md")
        print(f"Results saved to: results/comprehensive_evaluation_results.json")
        
        return results
        
    except Exception as e:
        print(f"Comprehensive evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    run_comprehensive_evaluation()
