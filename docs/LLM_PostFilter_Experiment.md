# LLM Post-Filter Experiment for GLITCH Detections

## Overview

This experiment evaluates a hybrid approach that combines static analysis tools with Large Language Models (LLMs) to improve security smell detection in Infrastructure as Code (IaC) scripts. The approach uses GLITCH as a primary detector and LLMs as intelligent post-filters.

## Problem Statement

Static analysis tools like GLITCH demonstrate:

- **High Recall**: Successfully detect most true security smells (0.69-1.00)
- **Low Precision**: Generate many false positives (precision ≤0.50)

This leads to alert fatigue and reduced tool adoption in practice.

## Hypothesis

**LLMs can significantly improve precision while maintaining high recall by:**

- Understanding code context and semantics
- Distinguishing between actual security issues and false alarms
- Reasoning about intent and usage patterns

## Experimental Design

### Two-Stage Pipeline

1. **Stage 1 - Static Detection**: GLITCH identifies potential security smells
2. **Stage 2 - LLM Filtering**: LLM validates detections based on context

### Target Security Smells

1. **Hard-coded Secret**

   - Static tools may flag any constant-looking string
   - LLM can assess if it's truly a secret vs placeholder/example

2. **Suspicious Comment**

   - Static tools flag all TODO/FIXME comments
   - LLM can evaluate security relevance and criticality

3. **Use of Weak Cryptography**
   - Static tools detect keyword mentions ("MD5", "SHA1")
   - LLM can determine actual usage vs documentation/comments

### Evaluation Datasets

- **Chef Cookbooks**: 80 files, 58 GLITCH detections for target smells
- **Puppet Manifests**: 80 files, 96 GLITCH detections for target smells

## Baseline Performance

### GLITCH-Only Results

| IaC Tool           | Security Smell     | Precision | Recall    | F1-Score  |
| ------------------ | ------------------ | --------- | --------- | --------- |
| Chef               | Hard-coded secret  | 0.196     | 0.692     | 0.305     |
| Chef               | Suspicious comment | 0.400     | 1.000     | 0.571     |
| Chef               | Weak cryptography  | 0.500     | 1.000     | 0.667     |
| **Chef Overall**   |                    | **0.241** | **0.778** | **0.364** |
| Puppet             | Hard-coded secret  | 0.136     | 0.818     | 0.234     |
| Puppet             | Suspicious comment | 0.391     | 1.000     | 0.562     |
| Puppet             | Weak cryptography  | 0.571     | 1.000     | 0.727     |
| **Puppet Overall** |                    | **0.229** | **0.917** | **0.367** |

### Key Observations

- **High False Positive Rate**: 76% (Chef), 77% (Puppet)
- **Perfect Target**: High recall with significant precision improvement potential
- **Category Variations**: Different smell types show varying detection challenges

## Implementation Architecture

### Core Components

```
src/hybrid/
├── data_extractor.py     # Extract GLITCH detections with context
├── llm_filter.py         # LLM-based post-filtering
└── evaluator.py          # Performance evaluation framework
```

### Research Components

```
experiments/llm-postfilter/
├── notebooks/            # Analysis and experimentation
├── data/                # Extracted detections for LLM evaluation
└── results/             # Experimental outcomes
```

## Methodology

### Data Preparation

1. Extract all GLITCH True Positive and False Positive detections
2. Gather code context (±3 lines around detection)
3. Create structured dataset for LLM evaluation

### LLM Prompting Strategy

- **Smell-specific prompts** tailored to each security smell type
- **Context inclusion** with surrounding code for semantic understanding
- **Binary classification** (YES/NO) with reasoning explanation
- **Few-shot examples** to improve accuracy and consistency

### Evaluation Metrics

- **Precision Improvement**: (Precision_after - Precision_before)
- **Recall Retention**: Recall_after / Recall_before
- **F1 Enhancement**: F1_after - F1_before
- **False Positive Reduction**: (FP_before - FP_after) / FP_before
