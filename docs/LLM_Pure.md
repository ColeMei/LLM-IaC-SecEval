# LLM Pure Evaluation Pipeline

Technical documentation for the pure LLM-based security smell detection pipeline.

## Overview

The **LLM Pure** methodology evaluates Infrastructure as Code (IaC) files directly using Large Language Models without any pre-filtering by static analysis tools. This approach tests the standalone capability of LLMs to detect all 9 security smell categories in IaC scripts with configurable prompt styles and multi-provider support.

## Prompt Styles

Two distinct prompting approaches are supported:

- **definition_based**: Human-readable security smell definitions with examples
- **static_analysis_rules**: Formal logical rules inspired by GLITCH paper with keyword heuristics

Both styles use identical structure and line numbering for consistent evaluation.

## Architecture

```
src/llm_pure/
├── config.py           # Configuration and paths
├── pipeline.py         # Main orchestration
├── prompt_builder.py   # Prompt construction with style support
├── model_client.py     # Abstract model interface
├── ollama_client.py    # Ollama implementation
├── openai_client.py    # OpenAI implementation
├── claude_client.py    # Claude implementation
├── file_processor.py   # File and ground truth handling
└── evaluator.py        # Metrics calculation

src/prompts/
├── Template_definition_based.txt      # Definition-based style template
└── Template_static_analysis_rules.txt # Static analysis rules style template

experiments/llm_pure/
├── run_evaluation.py    # Main execution script
├── validate_pipeline.py # Pipeline validation
├── generate_report.py   # Excel report generator
└── cleanup_results.py   # Results management
```

## Execution

### `experiments/llm_pure/run_evaluation.py`

**Key Arguments:**

| Argument           | Type    | Default            | Description                                               |
| ------------------ | ------- | ------------------ | --------------------------------------------------------- |
| `--client`         | Choice  | `ollama`           | LLM client: `ollama`, `openai`, `claude`                  |
| `--model`          | String  | (varies)           | Model name (default varies by client)                     |
| `--ollama-url`     | String  | `localhost:11434`  | Ollama server URL                                         |
| `--openai-api-key` | String  | (env var)          | OpenAI API key                                            |
| `--claude-api-key` | String  | (env var)          | Claude API key                                            |
| `--iac-tech`       | Choice  | All                | Technology: `ansible`, `chef`, `puppet`                   |
| `--limit`          | Integer | No limit           | Max files per technology                                  |
| `--prompt-style`   | Choice  | `definition_based` | Prompt style: `definition_based`, `static_analysis_rules` |
| `--small-batch`    | Flag    | False              | Quick test (5 files per tech)                             |
| `--show-prompts`   | Flag    | False              | Display prompts in console                                |
| `--save-prompts`   | Flag    | False              | Save prompts to files                                     |
| `--temperature`    | Float   | `0.1`              | Generation randomness (0.0-2.0)                           |
| `--max-tokens`     | Integer | `512`              | Max response tokens                                       |

**Usage Examples:**

```bash
# Basic usage (Ollama - default)
python experiments/llm_pure/run_evaluation.py

# OpenAI GPT-4o-mini
export OPENAI_API_KEY="your-key"
python experiments/llm_pure/run_evaluation.py --client openai

# Claude Sonnet
export ANTHROPIC_API_KEY="your-key"
python experiments/llm_pure/run_evaluation.py --client claude

# Custom model and parameters
python experiments/llm_pure/run_evaluation.py --client openai --model gpt-4 --temperature 0.05

# Compare prompt styles
python experiments/llm_pure/run_evaluation.py --prompt-style definition_based --small-batch
python experiments/llm_pure/run_evaluation.py --prompt-style static_analysis_rules --small-batch

# Small test with debugging
python experiments/llm_pure/run_evaluation.py --client openai --small-batch --show-prompts

# Model comparison
python experiments/llm_pure/run_evaluation.py --client ollama --model codellama:7b --limit 5
python experiments/llm_pure/run_evaluation.py --client openai --model gpt-4o-mini --limit 5
python experiments/llm_pure/run_evaluation.py --client claude --model claude-3-5-haiku-20241022 --limit 5

# Technology-specific evaluation
python experiments/llm_pure/run_evaluation.py --client openai --iac-tech ansible --limit 20
```

## Multi-Client Setup

| Client     | Type  | Setup Commands                                 | Default Model                |
| ---------- | ----- | ---------------------------------------------- | ---------------------------- |
| **Ollama** | Local | `ollama serve` <br> `ollama pull codellama:7b` | `codellama:7b`               |
| **OpenAI** | Cloud | `export OPENAI_API_KEY="your-key"`             | `gpt-4o-mini`                |
| **Claude** | Cloud | `export ANTHROPIC_API_KEY="your-key"`          | `claude-3-5-sonnet-20241022` |

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
  CLAUDE connection: ✓
  CLAUDE end-to-end: ✓

🎉 Pipeline validated with OLLAMA, CLAUDE client(s)!
```

### Individual Client Testing

```bash
# Test specific clients only
python experiments/llm_pure/run_evaluation.py --client ollama --validate-only
python experiments/llm_pure/run_evaluation.py --client openai --validate-only
python experiments/llm_pure/run_evaluation.py --client claude --validate-only
```

## Prompt Debugging

```bash
# Console display
python experiments/llm_pure/run_evaluation.py --show-prompts --limit 1

# Save to files (creates: results/llm_pure/prompts/prompt_STYLE_filename_EXPERIMENT_ID.txt)
python experiments/llm_pure/run_evaluation.py --save-prompts --small-batch

# Compare prompt styles
python experiments/llm_pure/run_evaluation.py --prompt-style definition_based --save-prompts --limit 1
python experiments/llm_pure/run_evaluation.py --prompt-style static_analysis_rules --save-prompts --limit 1
```

## Report Generation

### `experiments/llm_pure/generate_report.py`

Automatically generates Excel comparison reports after experiments are complete. Creates comprehensive analysis with GLITCH baseline comparisons.

**Key Features:**

- Multi-sheet Excel reports with formatted tables
- GLITCH baseline integration for academic comparison
- Per-smell metrics breakdown and overall statistics
- Automatic experiment discovery and comparison

**Usage Examples:**

```bash
# Compare latest 2 experiments for puppet
python experiments/llm_pure/generate_report.py --iac-tech puppet --latest 2

# Compare specific experiments
python experiments/llm_pure/generate_report.py --iac-tech ansible \
  --experiment-ids 20250103_143022 20250103_151045

# Generate report with custom output location
python experiments/llm_pure/generate_report.py --iac-tech chef \
  --latest 3 --output reports/chef_comparison.xlsx
```

**Generated Sheets:**

- **Summary**: Experiment overview and overall metrics
- **GLITCH_Comparison**: Academic format with baseline comparison
- **Detailed_Comparison**: Per-smell precision/recall/F1 breakdown
- **Full_Metrics**: Complete statistics with TP/FP/FN counts

## Client Comparison

| Feature      | Ollama                      | OpenAI               | Claude                |
| ------------ | --------------------------- | -------------------- | --------------------- |
| **Cost**     | Free (local compute)        | ~$0.002/1K tokens    | ~$0.003/1K tokens     |
| **Speed**    | Medium                      | Fast                 | Fast                  |
| **Privacy**  | Complete (local)            | Cloud-based          | Cloud-based           |
| **Setup**    | Requires local installation | API key only         | API key only          |
| **Use Case** | Development, privacy        | Production, accuracy | Production, reasoning |

## Results & Metrics

**Evaluation Metrics:**

- **Precision**: Correct predictions / Total predictions
- **Recall**: Correct predictions / Total actual smells
- **F1**: Harmonic mean of precision and recall

**GLITCH Baseline Integration:**

The pipeline uses accurate GLITCH baseline data for comparison:

- **Ansible**: `GLITCH-ansible-oracle_fixed_improved.csv` (manually verified line numbers)
- **Chef**: `GLITCH-chef-oracle.csv` (original GLITCH output)
- **Puppet**: `GLITCH-puppet-oracle.csv` (original GLITCH output)

**Output Files:**

All files from the same experiment use unified timestamps for easy grouping:

- `results/llm_pure/full_evaluation_EXPERIMENT_ID.json` - Complete reports
- `results/llm_pure/batch_TECH_EXPERIMENT_ID.json` - Per-technology results
- `results/llm_pure/raw_responses/FILENAME_EXPERIMENT_ID.json` - Individual LLM responses
- `results/llm_pure/prompts/prompt_STYLE_FILENAME_EXPERIMENT_ID.txt` - Constructed prompts

**Sample Output:**

```json
{
  "experiment_info": {
    "experiment_id": "20250803_143022",
    "model_name": "gpt-4o-mini",
    "prompt_style": "definition_based"
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

| Issue                 | Solution                                                |
| --------------------- | ------------------------------------------------------- |
| **OpenAI Auth Error** | Check `echo $OPENAI_API_KEY` and verify key validity    |
| **Claude Auth Error** | Check `echo $ANTHROPIC_API_KEY` and verify key validity |
| **Ollama Connection** | Run `ollama serve` and `ollama pull codellama:7b`       |
| **Model Not Found**   | Use specific model names (e.g., `--model gpt-4`)        |
