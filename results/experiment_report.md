
# RAG Parameter Experimentation Report

## Experiment Overview
- **Total Time**: 165.03 seconds
- **Date**: 2025-10-03T16:48:18.945287
- **Test Limits**: Embedding(20), Retrieval(20), Prompting(15)

## Summary of Best Results
- **Best Embedding Model**: None (F1: 0.0000)
- **Best Retrieval Strategy**: None (F1: 0.0000)
- **Best Prompting Strategy**: cot_prompt (F1: 8.8889)

## Detailed Results

### Embedding Models
- **all-MiniLM-L6-v2**: ERROR - too many values to unpack (expected 2)
- **all-mpnet-base-v2**: ERROR - too many values to unpack (expected 2)
- **paraphrase-MiniLM-L6-v2**: ERROR - too many values to unpack (expected 2)

### Retrieval Strategies
- **top_1**: F1=0.0000, EM=0.0000
- **top_3**: F1=0.0000, EM=0.0000
- **top_5**: F1=0.0000, EM=0.0000
- **top_10**: F1=0.0000, EM=0.0000

### Prompting Strategies
- **instruction_prompt**: F1=0.0000, EM=0.0000
- **cot_prompt**: F1=8.8889, EM=6.6667
- **persona_prompt**: F1=6.6667, EM=6.6667
- **simple_prompt**: F1=0.0000, EM=0.0000
