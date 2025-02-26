# Reducing Large Language Model Safety Risks in Women's Health using Semantic Entropy

## Understanding the codebase
A description of the key files and data types.

### generate.py
This file performs question inference.
#### Args
- M: number of responses to generate
- temp: temperature of the generated responses
- model: LLM to use
- reasoning: whether to force reasoning prompt

## semantic_similarity.py
Compute the semantic sets among the generated and true answers.
#### Args
- temp: temperature of the generated responses used when running generate.py
- model: LLM used when running generate.py
- reasoning: whether forced reasoning was used when running generate.py
- oneshot: whether to use a oneshot entailment approach
- entailment: which model to use for entailment, gpt or deberta

## correctness.py
Compute whether the response(s) to a question is correct post clustering of responses by meaning.
#### Args
- temp: temperature of the generated responses used when running generate.py
- model: LLM used when running generate.py
- reasoning: whether forced reasoning was used when running generate.py
- oneshot: whether a oneshot entailment approach was used running semantic_similarity.py
- entailment: which model was used for entailment running semantic_similarity.py
- checker: which model to use for assessing correctness, gpt vs deberta


## confidence.py
Script for computing confidence metrics based on the defined semantic sets.
Current metrics include:
- Semantic entropy
- Discrete semantic entropy
- Perplexity
#### Args
- temp: temperature of the generated responses used when running generate.py
- model: LLM used when running generate.py
- reasoning: whether forced reasoning was used when running generate.py
- oneshot: whether a oneshot entailment approach was used running semantic_similarity.py
- entailment: which model was used for entailment running semantic_similarity.py
- agg: Whether to use a sum normalized aggregator during semantic entropy calculation

### prompt_utils.py
Convenience functions for generating answers

### entailment.py
Convenience functions for computing entailment.

### prep.py
Script for creating human validation dataset

### results_utils.py and roc.py
Functions to help calculate results

