#!/usr/bin/env python3
"""
Test script for TA verification of RAG system functionality.
This script demonstrates that both naive and enhanced RAG systems work correctly.
"""

import sys
import os
sys.path.append('src')

def test_naive_rag():
    """Test the naive RAG system."""
    print("=" * 60)
    print("TESTING NAIVE RAG SYSTEM")
    print("=" * 60)
    
    try:
        from naive_rag import NaiveRAGSystem
        
        # Initialize system
        rag = NaiveRAGSystem()
        print("✓ Naive RAG system initialized")
        
        # Load data
        passages_df = rag.load_data()
        print(f"✓ Loaded {len(passages_df)} passages")
        
        # Setup database
        rag.setup_milvus_database(passages_df)
        print("✓ Milvus database setup complete")
        
        # Test query
        result = rag.query("What is artificial intelligence?", top_k=3)
        print(f"✓ Query successful: {len(result['contexts'])} contexts retrieved")
        print(f"  Answer: {result['answer'][:100]}...")
        print(f"  Retrieved IDs: {result['retrieved_ids']}")
        
        # Cleanup
        rag.cleanup()
        print("✓ Cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"✗ Naive RAG test failed: {e}")
        return False

def test_enhanced_rag():
    """Test the enhanced RAG system."""
    print("\n" + "=" * 60)
    print("TESTING ENHANCED RAG SYSTEM")
    print("=" * 60)
    
    try:
        from enhanced_rag import EnhancedRAGSystem
        
        # Initialize system
        enhanced_rag = EnhancedRAGSystem()
        print("✓ Enhanced RAG system initialized")
        
        # Load data
        passages_df = enhanced_rag.load_data()
        print(f"✓ Loaded {len(passages_df)} passages")
        
        # Setup database
        enhanced_rag.setup_milvus_database(passages_df)
        print("✓ Milvus database setup complete")
        
        # Test enhanced query
        result = enhanced_rag.query(
            "What is machine learning?", 
            top_k=3, 
            use_query_rewriting=True, 
            use_reranking=True
        )
        print(f"✓ Enhanced query successful: {len(result['contexts'])} contexts retrieved")
        print(f"  Answer: {result['answer'][:100]}...")
        print(f"  Query variations: {len(result['query_variations'])} generated")
        print(f"  Rerank scores: {[f'{score:.3f}' for score in result['rerank_scores'][:3]]}")
        
        # Cleanup
        enhanced_rag.cleanup()
        print("✓ Cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"✗ Enhanced RAG test failed: {e}")
        return False

def test_evaluation():
    """Test the evaluation system."""
    print("\n" + "=" * 60)
    print("TESTING EVALUATION SYSTEM")
    print("=" * 60)
    
    try:
        from evaluation import RAGEvaluator
        
        # Initialize evaluator
        evaluator = RAGEvaluator()
        print("✓ Evaluation system initialized")
        
        # Test with sample data
        sample_predictions = [
            {
                "query": "What is AI?",
                "ground_truth": "Artificial intelligence is intelligence demonstrated by machines",
                "answer": "AI is intelligence shown by machines",
                "contexts": ["AI is intelligence demonstrated by machines"],
                "retrieved_ids": [1]
            }
        ]
        
        results = evaluator.evaluate_system(sample_predictions, use_ragas=False)
        print(f"✓ Evaluation successful")
        print(f"  F1 Score: {results.get('f1', 'N/A')}")
        print(f"  Exact Match: {results.get('exact_match', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Evaluation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("RAG SYSTEM VERIFICATION FOR TA")
    print("This script tests that all components work correctly.")
    print("\nNote: This may take a few minutes to download models and data on first run.")
    
    # Run tests
    tests = [
        ("Naive RAG", test_naive_rag),
        ("Enhanced RAG", test_enhanced_rag),
        ("Evaluation", test_evaluation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n✓ All systems working correctly!")
        print("The RAG implementation is ready for evaluation.")
    else:
        print("\n✗ Some tests failed. Please check the error messages above.")
        print("The system may need debugging before evaluation.")

if __name__ == "__main__":
    main()
