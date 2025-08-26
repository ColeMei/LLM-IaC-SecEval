# LLM-IaC-SecEval: Reducing False Positives in IaC Security Analysis

## Overview

This research project investigates how Large Language Models (LLMs) can reduce false positives from static analysis tools for Infrastructure-as-Code (IaC) security analysis. We demonstrate that LLMs can serve as intelligent post-filters to improve the precision of existing security tools while maintaining high recall.

## Research Question

**Can LLMs reduce false positives from static analysis tools without sacrificing detection of real security issues?**

## Key Contribution

We leverage **LLM semantic understanding** to distinguish between:

- **True security vulnerabilities** vs **False alarms**
- **Real secrets** vs **Placeholder examples**
- **Security-critical comments** vs **General TODOs**
- **Actual weak crypto usage** vs **Documentation mentions**

This addresses the critical **alert fatigue** problem that prevents widespread adoption of security analysis tools.

## Target Security Smells

We focus on **4 security smell categories** where semantic understanding provides value:

1. **Hard-coded secrets** - Distinguish real secrets from placeholders
2. **Suspicious comments** - Identify security-relevant vs general comments
3. **Use of weak cryptography algorithms** - Detect actual usage vs documentation
4. **Use of HTTP without SSL/TLS** - Identify insecure communication patterns

_These smells benefit most from contextual analysis rather than pattern matching._

## Approach

### Two-Stage Detection Pipeline

1. **Static Analysis**: Use GLITCH for comprehensive detection (high recall, low precision)
2. **LLM Post-Filter**: Apply LLM to reduce false positives while preserving true positives

### Methodology Evaluation

We evaluated two approaches:

**Pure LLM**: Direct security smell detection

- Limited recall compared to static analysis tools
- Inconsistent performance across security categories

**Post-Filter LLM**: LLM as intelligent filter for static analysis results

- Significant precision improvement over baseline
- Maintains strong recall performance
- Selected as optimal approach

**Result**: Post-filter approach demonstrates substantial precision improvements while maintaining detection capability.

## Project Structure

```
LLM-IaC-SecEval/
├── data/                          # Original datasets
├── docs/                          # Documentation
├── experiments/                   # Execution interface
│   ├── llm_pure/                 # Pure LLM method execution (scripts)
│   ├── llm_postfilter/           # Hybrid method execution (notebooks + data)
│   ├── data-analysis/            # Analysis notebooks
│   └── zero-shot/                # Zero-shot experiments
├── results/                       # All results from both methodologies
│   ├── llm_pure/                 # Results from pure LLM method
│   └── llm_postfilter/           # Results from hybrid method
└── src/                           # Code logic only
    ├── llm_pure/                 # Pure LLM implementation
    ├── llm_postfilter/           # Hybrid implementation
    └── prompts/                  # Prompt templates
```

## Research Goals

### Primary Objective

Train open-source models to perform false positive filtering as effectively as closed-source APIs at significantly lower cost.

### Approach

1. **Dataset Expansion**: Create large-scale training datasets using pseudo-labeling
2. **Model Training**: Fine-tune both generative and encoder models
3. **Evaluation**: Compare open-source models against closed-source baselines
4. **Production**: Develop practical tools for DevOps integration

### Target Outcomes

- Cost-effective alternative to expensive API calls
- Practical tool for real-world security workflows
- Open-source models matching closed-source performance
- Integration with existing CI/CD pipelines

---

_This research contributes to AI-assisted cybersecurity and practical security tool development._
