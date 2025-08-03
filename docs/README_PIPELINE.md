# Pipeline Documentation

Technical documentation for the automated LLM evaluation pipeline.

## Architecture

```
src/automated/
├── config.py          # Configuration and paths
├── pipeline.py         # Main orchestration
├── prompt_builder.py   # Modular prompt construction
├── model_client.py     # Abstract model interface
├── ollama_client.py    # Ollama implementation
├── file_processor.py   # File and ground truth handling
└── evaluator.py        # Metrics calculation

src/prompts/
├── Template_detailed.txt           # Full context template
└── Template_instructions_only.txt  # Instructions only (modular)
```

## Script Usage

### `scripts/run_evaluation.py`

**Key Arguments:**

| Argument         | Type    | Default        | Description                             |
| ---------------- | ------- | -------------- | --------------------------------------- |
| `--model`        | String  | `codellama:7b` | Ollama model name                       |
| `--iac-tech`     | Choice  | All            | Technology: `ansible`, `chef`, `puppet` |
| `--limit`        | Integer | No limit       | Max files per technology                |
| `--no-modular`   | Flag    | False          | Use full context (disable modular)      |
| `--small-batch`  | Flag    | False          | Quick test (5 files per tech)           |
| `--show-prompts` | Flag    | False          | Display prompts in console              |
| `--save-prompts` | Flag    | False          | Save prompts to files                   |
| `--temperature`  | Float   | `0.1`          | Generation randomness (0.0-2.0)         |
| `--max-tokens`   | Integer | `512`          | Max response tokens                     |

**Examples:**

```bash
# Basic usage
python scripts/run_evaluation.py

# Small test with prompt debugging
python scripts/run_evaluation.py --small-batch --show-prompts

# Compare modular vs full context
python scripts/run_evaluation.py --limit 1 --show-prompts
python scripts/run_evaluation.py --limit 1 --show-prompts --no-modular

# Technology-specific evaluation
python scripts/run_evaluation.py --iac-tech ansible --limit 20
```

### `scripts/validate_pipeline.py`

Validates setup and tests pipeline components:

```bash
python scripts/validate_pipeline.py
```

## Prompting Approaches

### Modular Prompting (Default)

Separates background context from instructions:

```
# Background Context
[Security smell definitions from smells-description.txt]

# Instructions
[Task instructions from Template_instructions_only.txt]

# Script for Analysis
File name: example.yml
File content: [IaC script content]
```

### Full Context (`--no-modular`)

Complete prompt with embedded definitions:

```
[Complete Template_detailed.txt with all definitions and instructions]

# Script for Analysis
File name: example.yml
File content: [IaC script content]
```

**Comparison:**

- **Modular**: ~1,300 characters, cleaner separation
- **Full Context**: ~4,400 characters, traditional approach

## Prompt Debugging

**Console Display:**

```bash
python scripts/run_evaluation.py --show-prompts --limit 1
```

**Save to Files:**

```bash
python scripts/run_evaluation.py --save-prompts --small-batch
# Creates: results/prompts/prompt_modular_filename_timestamp.txt
```

## Output Format

```json
{
  "experiment_info": {
    "experiment_id": "20250803_143022",
    "model_name": "codellama:7b",
    "approach": "modular"
  },
  "results_by_technology": {
    "ansible": {
      "overall_metrics": {
        "precision": 0.234,
        "recall": 0.301,
        "f1": 0.267
      },
      "files_processed": 81,
      "predictions_made": 312
    }
  }
}
```

## Evaluation Metrics

- **Precision**: Correct predictions / Total predictions
- **Recall**: Correct predictions / Total actual smells
- **F1**: Harmonic mean of precision and recall
- **Support**: Number of actual instances per category

Results saved to:

- `results/full_evaluation_*.json` - Complete reports
- `results/batch_*.json` - Per-technology results
- `results/raw_responses/` - Individual LLM responses
- `results/prompts/` - Constructed prompts (when `--save-prompts` used)
