# LLM-IaC-SecEval Pipeline Development Documentation

This document outlines the development of an automated evaluation pipeline for Large Language Models (LLMs) on Infrastructure as Code (IaC) security smell detection, evolving from manual copy-paste workflows to a fully automated system.

## Development Evolution

### Phase 1: Manual Zero-Shot Approach

**Initial State**: Manual copy-paste workflow

- Copy prompt template content manually from `src/zero-shot/prompt/Template.txt` and `Template_detailed.txt`
- Manually insert file names and content into placeholders `{NAME_OF_FILE}` and `{CONTENT_OF_FILE}`
- Feed complete prompts into LLM interfaces (Ollama terminal, web UIs)
- Manually collect and format responses
- No systematic evaluation or comparison

**Problems Identified**:

- Time-intensive and error-prone process
- Inconsistent prompt formatting across experiments
- No automated evaluation metrics against ground truth
- Difficult to scale across large datasets (382 total files: 115 Ansible + 149 Chef + 118 Puppet)
- No systematic model comparison capability

### Phase 2: Automated Pipeline Architecture

**Solution**: Modular prompting pipeline with automated context separation

- **Modular Approach**: Background context separation (security smell definitions) from main prompts
- **Automated Processing**: File processing, prompt generation, and response collection
- **Batch Processing**: Handle entire datasets with single command execution
- **Structured Evaluation**: Automated precision/recall calculation against ground truth annotations
- **Extensible Architecture**: Abstract model client interface for easy LLM integration

## Prerequisites

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama server
ollama serve

# Pull Code Llama 7B model
ollama pull codellama:7b

# Install Python dependencies
pip install -r requirements.txt
```

## Pipeline Architecture

```
src/pipeline/
├── config.py          # Configuration management and paths
├── prompt_builder.py   # Modular prompt construction with context separation
├── model_client.py     # Abstract model interface
├── ollama_client.py    # Ollama API implementation
├── file_processor.py   # IaC file and ground truth handling
├── evaluator.py        # Metrics calculation and comparison
└── pipeline.py         # Main orchestration logic

scripts/
├── run_evaluation.py   # Main execution script
└── validate_pipeline.py # Setup validation and testing

data/
├── oracle-dataset-ansible/    # Ansible IaC script files
├── oracle-dataset-chef/       # Chef IaC script files
├── oracle-dataset-puppet/     # Puppet IaC script files
├── oracle-dataset-ansible.csv # Ansible ground truth annotations
├── oracle-dataset-chef.csv    # Chef ground truth annotations
├── oracle-dataset-puppet.csv  # Puppet ground truth annotations
└── smells-description.txt     # Security smell definitions

src/prompts/
├── Template.txt                # Original concise prompt template
├── Template_detailed.txt       # Complete prompt with embedded definitions
└── Template_instructions_only.txt # Instructions only (for modular approach)

results/
├── full_evaluation_*.json    # Complete evaluation reports
├── batch_*.json             # Per-technology results
├── raw_responses/           # Individual file responses
└── prompts/                 # Constructed prompts (when --save-prompts used)
```

## Script Specifications

### `scripts/run_evaluation.py`

Main execution script for running automated LLM evaluations.

#### Command Line Arguments

**Model Configuration**:

- `--model MODEL_NAME`

  - **Type**: String
  - **Default**: `codellama:7b`
  - **Description**: Specify the Ollama model to use for evaluation
  - **Examples**: `codellama:7b`, `mistral:7b`, `llama2:7b`

- `--ollama-url URL`
  - **Type**: String
  - **Default**: `http://localhost:11434`
  - **Description**: Ollama server endpoint URL
  - **Examples**: `http://localhost:11434`, `http://10.0.0.1:11434`

**Execution Scope**:

- `--iac-tech TECHNOLOGY`

  - **Type**: Choice from `["ansible", "chef", "puppet"]`
  - **Default**: All technologies
  - **Description**: Limit evaluation to specific IaC technology
  - **Usage**: `--iac-tech ansible` (evaluates only Ansible files)

- `--limit NUMBER`
  - **Type**: Integer
  - **Default**: No limit (all files)
  - **Description**: Maximum number of files to process per technology
  - **Usage**: `--limit 50` (processes first 50 files per technology)

**Prompting Strategy**:

- `--no-modular`
  - **Type**: Boolean flag
  - **Default**: False (modular prompting enabled)
  - **Description**: Disable modular prompting, use full context in prompts
  - **Usage**: Include flag to use traditional full-context prompting

**Generation Parameters**:

- `--temperature FLOAT`

  - **Type**: Float
  - **Default**: `0.1`
  - **Range**: `0.0` to `2.0`
  - **Description**: Controls randomness in model generation (lower = more deterministic)
  - **Usage**: `--temperature 0.2` for slightly more creative responses

- `--max-tokens INTEGER`
  - **Type**: Integer
  - **Default**: `512`
  - **Description**: Maximum number of tokens to generate in response
  - **Usage**: `--max-tokens 1024` for longer responses

**Execution Modes**:

- `--validate-only`

  - **Type**: Boolean flag
  - **Default**: False
  - **Description**: Only validate setup without running evaluation
  - **Usage**: Check if Ollama is running and models are available

- `--small-batch`
  - **Type**: Boolean flag
  - **Default**: False
  - **Description**: Run evaluation on 5 files per technology (quick test)
  - **Usage**: For testing pipeline functionality before full runs

**Debug and Visibility Options**:

- `--show-prompts`

  - **Type**: Boolean flag
  - **Default**: False
  - **Description**: Display constructed prompts in console output
  - **Usage**: Include flag to see exactly what prompts are sent to the LLM

- `--save-prompts`
  - **Type**: Boolean flag
  - **Default**: False
  - **Description**: Save constructed prompts to files in `results/prompts/`
  - **Usage**: Include flag to save prompts for inspection and debugging

#### Usage Examples

```bash
# Basic usage - all files, all technologies
python scripts/run_evaluation.py

# Validate setup only
python scripts/run_evaluation.py --validate-only

# Small batch test
python scripts/run_evaluation.py --small-batch

# Specific technology with limits
python scripts/run_evaluation.py --iac-tech ansible --limit 20

# Different model with custom parameters
python scripts/run_evaluation.py --model mistral:7b --temperature 0.2 --max-tokens 1024

# Remote Ollama server
python scripts/run_evaluation.py --ollama-url http://192.168.1.100:11434

# Full context prompting (no modular separation)
python scripts/run_evaluation.py --no-modular

# Combined parameters
python scripts/run_evaluation.py --iac-tech chef --limit 10 --temperature 0.15 --small-batch

# Debug prompts - see what's sent to LLM
python scripts/run_evaluation.py --show-prompts --limit 1

# Save prompts to files for inspection
python scripts/run_evaluation.py --save-prompts --small-batch

# Compare modular vs full context prompts
python scripts/run_evaluation.py --show-prompts --limit 1  # Modular (default)
python scripts/run_evaluation.py --show-prompts --limit 1 --no-modular  # Full context
```

### `scripts/validate_pipeline.py`

Validation script to test pipeline components and setup.

#### Functionality

- **File Processing Test**: Validates data loading and ground truth parsing
- **Ollama Connection Test**: Checks server availability and model access
- **End-to-End Test**: Processes single file to verify complete workflow

#### Usage

```bash
python scripts/validate_pipeline.py
```

## Modular Prompting Implementation Details

### Full Context Approach (--no-modular)

Complete prompt with all context embedded:

```
You are an expert in Infrastructure-as-Code (IaC) security analysis.

# Task
[Complete instructions and security smell definitions]

# Definitions of Security Smells
[All 9 security smell definitions]

# Script for Analysis
File name: example.yml
File content: [Complete file content]
```

### Modular Approach (default)

Separated background context with automated assembly:

```
# Background Context (loaded from smells-description.txt)
[Security smell definitions from separate file]

# Instructions (loaded from Template_instructions_only.txt)
You are an expert in Infrastructure-as-Code (IaC) security analysis.
# Task
Given an IaC script and the file name, identify **every line**...

# Script for Analysis
File name: example.yml
File content: [Complete file content]
```

**Key Differences**:

- **Modular**: 1,310 characters (context separated)
- **Full Context**: 4,423 characters (context embedded with duplication)
- **Token Efficiency**: Modular approach uses ~70% fewer tokens
- **Clarity**: Modular approach separates concerns for easier maintenance

### Prompt Visibility Features

The pipeline includes debugging features to see exactly what prompts are sent to the LLM:

**Console Display (`--show-prompts`)**:

```bash
python scripts/run_evaluation.py --show-prompts --limit 1

# Output shows:
# ================================================================================
# CONSTRUCTED PROMPT FOR: example.yml
# Mode: Modular
# Length: 1310 characters
# ================================================================================
# [Full prompt content displayed]
# ================================================================================
```

**File Saving (`--save-prompts`)**:

```bash
python scripts/run_evaluation.py --save-prompts --small-batch

# Creates files in results/prompts/:
# prompt_modular_example.yml_142259.txt
# prompt_full_example.yml_142318.txt
```

**Saved Prompt Format**:

```
# CONSTRUCTED PROMPT
# File: example.yml
# Mode: modular
# Timestamp: 2025-08-03T14:22:59.210264
# Length: 1310 characters
# ============================================================

[Full prompt content]
```

## Data Processing Specifications

### Input Files

- **IaC Scripts**: Located in `data/oracle-dataset-{technology}/`
- **Ground Truth**: CSV files with columns: `PATH,LINE,CATEGORY,AGREEMENT`
- **Security Definitions**: `data/smells-description.txt`
- **Instruction Templates**:
  - `src/prompts/Template_detailed.txt` (full context)
  - `src/prompts/Template_instructions_only.txt` (modular approach)

### Output Format

```json
{
  "experiment_info": {
    "experiment_id": "20241220_143022",
    "model_name": "codellama:7b",
    "timestamp": "2024-12-20T14:30:22",
    "technologies_evaluated": ["ansible", "chef", "puppet"]
  },
  "results_by_technology": {
    "ansible": {
      "overall_metrics": {
        "overall": {
          "precision": 0.75,
          "recall": 0.68,
          "f1": 0.71
        },
        "by_category": { ... }
      },
      "processing_metadata": {
        "use_modular": true,
        "generation_params": { ... }
      }
    }
  }
}
```

## Evaluation Metrics

- **Precision**: Correct predictions / Total predictions
- **Recall**: Correct predictions / Total actual smells
- **F1**: Harmonic mean of precision and recall
- **Support**: Number of actual instances per category

## Technical Implementation Notes

### Why "Modular Prompting" Not "RAG"?

Our implementation is **automated context separation**, not true Retrieval-Augmented Generation:

| Aspect              | **True RAG**                          | **Our Modular Approach**                         |
| ------------------- | ------------------------------------- | ------------------------------------------------ |
| **Context Storage** | Vector database/embeddings            | Static text files                                |
| **Retrieval**       | Dynamic based on query relevance      | Static assembly of all definitions               |
| **Context Size**    | Only relevant information retrieved   | All 9 security smell definitions always included |
| **LLM Input**       | Query + dynamically retrieved context | Complete assembled prompt                        |
| **Implementation**  | Semantic search + retrieval           | File loading + concatenation                     |

### What We Actually Do:

1. **Load** security smell definitions from `smells-description.txt`
2. **Load** instructions from template files
3. **Assemble** complete prompt by concatenation
4. **Send** full prompt to LLM (same as manual approach, but automated)

This is **automated prompt assembly**, not retrieval-based generation.

## Extending the Pipeline

### Adding New Model Clients

1. Inherit from `ModelClient` abstract class
2. Implement `generate()` and `is_available()` methods
3. Return `ModelResponse` objects with standardized format

```python
from pipeline.model_client import ModelClient, ModelResponse

class OpenAIClient(ModelClient):
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        # Implement OpenAI API call
        response = openai.ChatCompletion.create(...)
        return ModelResponse(
            content=response.choices[0].message.content,
            model_name=self.model_name,
            # ... other fields
        )

    def is_available(self) -> bool:
        # Check API key and connectivity
        pass
```

### Configuration Management

All settings managed through `src/pipeline/config.py`:

- File paths and directories
- Model default parameters
- Dataset configurations
- Output formatting options

### Adding New IaC Technologies

1. Add dataset directory: `data/oracle-dataset-{technology}/`
2. Add ground truth CSV: `data/oracle-dataset-{technology}.csv`
3. Update `config.py` to include new technology
4. Pipeline automatically processes new technology

## Makefile Commands

```bash
# Basic operations
make install         # Install dependencies
make validate        # Validate pipeline setup
make small-batch     # Run small test
make full           # Run complete evaluation

# Technology-specific
make ansible        # Ansible files only
make chef          # Chef files only
make puppet        # Puppet files only

# Comparisons
make modular-comparison  # Compare modular vs full context
```

## Development Benefits Achieved

### **Before**: Manual Copy-Paste Workflow

- ⏰ Manual prompt generation for each file
- 🐛 Error-prone file handling
- 📝 Manual result collection
- 🔄 No systematic comparison
- 📊 No automated metrics

### **After**: Automated Modular Pipeline

- ⚡ Batch process entire datasets
- 🎯 Consistent prompt generation
- 📊 Automatic metrics calculation
- 🔄 Systematic model comparison
- 💾 Structured result storage
- 🧩 Clean separation of concerns (context vs instructions)

The pipeline successfully transforms the research workflow from manual experimentation to systematic, reproducible evaluation at scale while maintaining accurate terminology for the technical approach used.
