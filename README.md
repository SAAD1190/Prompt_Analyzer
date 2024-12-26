<p align="center">
  <img src="./Assets/promptAnalyzer.jpg" width="200px" alt="Prompt Analyzer">
</p>

## Table of Contents

> [Overview](#overview)

> [Features](#features)

- [Prompt Pre-processing](#prompt-preprocessing)
- [Semantic Analysis](#semantic-analysis)
- [Similarity Reduction](#similarity-reduction)
- [Relevance and Complexity Analysis](#relevance-and-complexity-analysis)
---

## Overview

The **Prompt Analyzer** is a tool designed to analyze text prompts, often generated from image descriptions. It processes prompts to evaluate their quality based on similarity, semantic diversity, complexity, and relevance. Additionally, it provides a simple interface for users to rank prompts and remove redundant ones.

---

## Features

### Prompt Pre-processing
Handles basic cleaning and tokenization:
- Removes punctuation and stop words.
- Tokenizes text into words and identifies unique words.
- Prepares prompts for similarity and complexity analyses.

### Semantic Analysis
Analyzes prompts based on:
- **Semantic Diversity Score (SDS):** Measures variability in semantic content.
- **Semantic Repetition Penalty (SRP):** Penalizes excessive repetition of concepts.

### Similarity Reduction
Ensures diversity in prompts:
- Calculates cosine similarity between prompts.
- Removes prompts with similarity exceeding a user-defined threshold.

### Relevance and Complexity Analysis
- **Relevance Analysis:** Compares test prompts to user-provided reference prompts to compute similarity and alignment.
- **Complexity Analysis:** Evaluates prompts based on vocabulary richness, length, and linguistic complexity.

---