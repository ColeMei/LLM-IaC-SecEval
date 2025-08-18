# LLM Post-Filter Pipeline

Technical documentation for the hybrid LLM post-filtering pipeline for security smell detection.

## Overview

The **LLM Post-Filter** methodology implements a **hybrid approach** combining static analysis (GLITCH) with Large Language Models (LLMs) to improve security smell detection in Infrastructure as Code (IaC) scripts.

**Core Idea**: Use GLITCH for high recall detection, then apply LLM as an intelligent post-filter to reduce false positives while maintaining true positives.

## Problem & Motivation

Current static analysis tools have a fundamental trade-off:

- **High Recall** (~70-90%): Catch most real security issues
- **Low Precision** (~20-30%): Generate many false alarms

This creates **alert fatigue** and reduces tool adoption in practice.

## Hypothesis

**LLMs can act as semantic filters** to:

- Understand code context and intent
- Distinguish real security issues from false alarms
- **Boost precision significantly** while retaining high recall

## Approach

### Two-Stage Detection Pipeline

1. **Stage 1 - GLITCH Detection**: Static analysis identifies potential security smells (high recall)
2. **Stage 2 - LLM Post-Filter**: An LLM evaluates each detection with context (improved precision)

### Target Security Smells

We focus on **4 security smell categories** where semantic understanding helps:

1. **Hard-coded Secrets**: Distinguish real secrets from placeholders/examples
2. **Suspicious Comments**: Identify security-relevant vs general TODO comments
3. **Weak Cryptography**: Detect actual usage vs documentation mentions
4. **Use of HTTP without SSL/TLS**: Identify insecure communication patterns vs documentation

### Evaluation Data

- **Chef & Puppet IaC scripts** with ground truth labels
- **Static analysis detections** (GLITCH) to filter and improve

## Architecture

```
src/llm_postfilter/
├── data_extractor.py      # Extract GLITCH detections
├── context_extractor.py   # Code context extraction (configurable lines)
├── llm_client.py          # Multi-provider LLM integration
├── llm_filter.py          # Main filtering pipeline
├── evaluator.py           # Performance evaluation
├── prompt_templates.py    # Backward-compatible prompt interface
└── prompt_loader.py       # External prompt system

src/prompts/llm_postfilter/
├── definition_based_conservative.txt    # Conservative prompt template
├── static_analysis_rules.txt           # Rule-based prompt template
├── smell_definitions_conservative.yaml # Security smell definitions
└── static_analysis_functions.yaml     # Pattern matching functions

experiments/llm_postfilter/
├── notebooks/
│   ├── 01_data_extraction.ipynb    # Data preparation
│   └── 02_llm_experiment.ipynb     # LLM filtering experiment
├── archive_results.py              # Archive experiment results
└── clean_working_dir.py            # Clean working directory
```

## Execution

The pipeline is executed through Jupyter notebooks that provide interactive analysis and visualization. The LLM provider and model are configurable.

### Data Preparation

```bash
# Run data extraction notebook
jupyter notebook experiments/llm_postfilter/notebooks/01_data_extraction.ipynb
```

**Process:**

1. Extract GLITCH detections with ground truth labels
2. Generate context-enhanced files with ±3 lines of code
3. Prepare datasets for LLM evaluation

### LLM Filtering

```bash
# Run LLM experiment notebook
jupyter notebook experiments/llm_postfilter/notebooks/02_llm_experiment.ipynb
```

**Configuration Options:**

- **LLM Provider**: OpenAI, Anthropic, Ollama, or OpenAI-compatible endpoints
- **Prompt Templates**: Multiple styles (definition_based_conservative, static_analysis_rules)
- **Context Lines**: Configurable number of lines around target detection (0 = target only)
- **Model Selection**: Provider-specific models (GPT-4o, Claude-3.5-Sonnet, CodeLlama, etc.)

**Provider/Model Selection:**

- Manual override in notebook (recommended): set `provider`, `model`, and optional `base_url`
- Or via environment variables:
  - `LLM_PROVIDER`: openai, anthropic, ollama, openai_compatible
  - `LLM_MODEL`: model name (e.g., gpt-4o-mini, claude-3-5-sonnet-latest, codellama:7b)
  - `LLM_BASE_URL`: for Ollama or OpenAI-compatible endpoints
  - API keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_COMPATIBLE_API_KEY`

**Process:**

1. Initialize LLM client (selected provider/model)
2. Apply LLM filtering to each detection with context
3. Evaluate GLITCH vs GLITCH+LLM performance
4. Generate comprehensive reports and analysis

## Key Features

### Multi-Provider LLM Support

- **OpenAI**: GPT-4o, GPT-4o-mini with API key authentication
- **Anthropic**: Claude-3.5-Sonnet and other models with API key authentication
- **Ollama**: Local models (CodeLlama, Llama, etc.) via HTTP API
- **OpenAI-Compatible**: Custom endpoints supporting OpenAI API format

### Configurable Prompt System

- **External Templates**: Prompt templates stored in separate files for easy modification
- **Multiple Styles**: Definition-based and static-analysis-rule-based prompts
- **Versioning**: Support for different prompt versions (conservative, standard, etc.)
- **IaC Technology Context**: Automatic detection and inclusion of IaC tool context

### Flexible Context Extraction

- **Configurable Context Lines**: Set number of lines around target detection (0 = target only)
- **Smart Context**: Automatically adjusts for file boundaries and code structure
- **Multiple Technologies**: Support for Chef, Puppet, and Ansible syntax

```
experiments/llm_postfilter/data/llm_results/    # Latest/working results
results/llm_postfilter/                         # Archived results
├── 20250806_143022/                           # Timestamped experiment
├── 20250806_151205_baseline/                  # Custom named experiment
└── evaluation/                                # Other evaluation results
```
