# LLM Pure Evaluation Pipeline

Technical documentation for the pure LLM-based security smell detection pipeline.

## Overview

The **LLM Pure** methodology evaluates Infrastructure as Code (IaC) files directly using Large Language Models without any pre-filtering by static analysis tools. This approach tests the standalone capability of LLMs to detect security smells in IaC scripts.

## Architecture

```
src/llm_pure/
├── config.py           # Configuration and paths
├── pipeline.py         # Main orchestration
├── prompt_builder.py   # Modular prompt construction
├── model_client.py     # Abstract model interface
├── ollama_client.py    # Ollama implementation
├── openai_client.py    # OpenAI implementation
├── file_processor.py   # File and ground truth handling
└── evaluator.py        # Metrics calculation

src/prompts/
├── Template_detailed.txt           # Full context template
└── Template_instructions_only.txt  # Instructions only (modular)

experiments/llm_pure/
├── run_evaluation.py    # Main execution script
├── validate_pipeline.py # Pipeline validation
└── cleanup_results.py   # Results management
```

## Execution

### `experiments/llm_pure/run_evaluation.py`

**Key Arguments:**

| Argument           | Type    | Default           | Description                             |
| ------------------ | ------- | ----------------- | --------------------------------------- |
| `--client`         | Choice  | `ollama`          | LLM client: `ollama`, `openai`          |
| `--model`          | String  | (varies)          | Model name (default varies by client)   |
| `--ollama-url`     | String  | `localhost:11434` | Ollama server URL                       |
| `--openai-api-key` | String  | (env var)         | OpenAI API key                          |
| `--iac-tech`       | Choice  | All               | Technology: `ansible`, `chef`, `puppet` |
| `--limit`          | Integer | No limit          | Max files per technology                |
| `--no-modular`     | Flag    | False             | Use full context (disable modular)      |
| `--small-batch`    | Flag    | False             | Quick test (5 files per tech)           |
| `--show-prompts`   | Flag    | False             | Display prompts in console              |
| `--save-prompts`   | Flag    | False             | Save prompts to files                   |
| `--temperature`    | Float   | `0.1`             | Generation randomness (0.0-2.0)         |
| `--max-tokens`     | Integer | `512`             | Max response tokens                     |

**Usage Examples:**

```bash
# Basic usage (Ollama - default)
python experiments/llm_pure/run_evaluation.py

# OpenAI GPT-4o-mini
export OPENAI_API_KEY="your-key"
python experiments/llm_pure/run_evaluation.py --client openai

# Custom model and parameters
python experiments/llm_pure/run_evaluation.py --client openai --model gpt-4 --temperature 0.05

# Small test with debugging
python experiments/llm_pure/run_evaluation.py --client openai --small-batch --show-prompts

# Model comparison
python experiments/llm_pure/run_evaluation.py --client ollama --model codellama:7b --limit 5
python experiments/llm_pure/run_evaluation.py --client openai --model gpt-4o-mini --limit 5

# Technology-specific evaluation
python experiments/llm_pure/run_evaluation.py --client openai --iac-tech ansible --limit 20
```

## Multi-Client Setup

| Client     | Type  | Setup Commands                                 | Default Model  |
| ---------- | ----- | ---------------------------------------------- | -------------- |
| **Ollama** | Local | `ollama serve` <br> `ollama pull codellama:7b` | `codellama:7b` |
| **OpenAI** | Cloud | `export OPENAI_API_KEY="your-key"`             | `gpt-4o-mini`  |

## Validation & Testing

### Complete Validation

```bash
# Test all components + both clients
python experiments/llm_pure/validate_pipeline.py
```

**Sample Output:**

```
VALIDATION SUMMARY
  File processing: ✓
  OLLAMA connection: ✓
  OLLAMA end-to-end: ✓
  OPENAI connection: ✗
  OPENAI end-to-end: ✗

🎉 Pipeline validated with OLLAMA client(s)!
```

### Individual Client Testing

```bash
# Test specific clients only
python experiments/llm_pure/run_evaluation.py --client ollama --validate-only
python experiments/llm_pure/run_evaluation.py --client openai --validate-only
```

## Prompt Debugging

```bash
# Console display
python experiments/llm_pure/run_evaluation.py --show-prompts --limit 1

# Save to files (creates: results/llm_pure/prompts/prompt_modular_filename_timestamp.txt)
python experiments/llm_pure/run_evaluation.py --save-prompts --small-batch
```

## Client Comparison

| Feature      | Ollama                      | OpenAI               |
| ------------ | --------------------------- | -------------------- |
| **Cost**     | Free (local compute)        | ~$0.002/1K tokens    |
| **Speed**    | Medium                      | Fast                 |
| **Privacy**  | Complete (local)            | Cloud-based          |
| **Setup**    | Requires local installation | API key only         |
| **Use Case** | Development, privacy        | Production, accuracy |

## Results & Metrics

**Evaluation Metrics:**

- **Precision**: Correct predictions / Total predictions
- **Recall**: Correct predictions / Total actual smells
- **F1**: Harmonic mean of precision and recall

**Output Files:**

- `results/llm_pure/full_evaluation_*.json` - Complete reports
- `results/llm_pure/batch_*.json` - Per-technology results
- `results/llm_pure/raw_responses/` - Individual LLM responses
- `results/llm_pure/prompts/` - Constructed prompts (when `--save-prompts` used)

**Sample Output:**

```json
{
  "experiment_info": {
    "experiment_id": "20250803_143022",
    "model_name": "gpt-4o-mini",
    "approach": "modular"
  },
  "results_by_technology": {
    "ansible": {
      "overall_metrics": { "precision": 0.234, "recall": 0.301, "f1": 0.267 },
      "files_processed": 81
    }
  }
}
```

## Troubleshooting

| Issue                 | Solution                                             |
| --------------------- | ---------------------------------------------------- |
| **OpenAI Auth Error** | Check `echo $OPENAI_API_KEY` and verify key validity |
| **Ollama Connection** | Run `ollama serve` and `ollama pull codellama:7b`    |
| **Model Not Found**   | Use specific model names (e.g., `--model gpt-4`)     |
