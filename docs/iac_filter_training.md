### Stage 1 — Dataset Expansion (Multi-IaC, Pseudo-Labels)

Objective: Build train/validation/test datasets for Chef, Ansible, and Puppet using GLITCH detections + LLM post-filtering (pseudo-labels).

#### Pipeline Steps

1. Collect unlabeled IaC scripts

- Target: ~2000+ scripts per technology (stored under `data/iac_filter_training/oracle-dataset-{iac_tech}`).

2. Run GLITCH → collect raw detections

- Input: `data/iac_filter_training/GLITCH-{iac_tech}-oracle.csv`
- Smells covered: Hard-coded secret, Suspicious comment, Use of weak cryptography algorithms, Use of HTTP without SSL/TLS

3. Sample detections upfront

- Targets based on test set sizes: Chef(675), Puppet(963), Ansible(432) = 2,070 total
- Preserves GLITCH's natural smell distribution within each technology
- Output: `GLITCH-{iac_tech}-oracle_sampled.csv`

4. Run LLM post-filter → pseudo-label TP/FP decisions

- Default: `gpt-4o-mini` for smoke tests; switch to Claude 4.0 for production
- Prompt style: `static_analysis_rules`
- Output: per-smell CSVs with decisions, plus prompt/response logs

5. Format dataset into JSONL (HF-ready) and split train/val/test

6. Spot-check random samples manually to estimate error rate

---

### Files and Usage

#### 0. Sample detections upfront

`src/iac_filter_training/detection_sampler.py`

- Samples GLITCH detections to meet exact target counts while preserving natural smell proportions
- Targets: Chef(675), Puppet(963), Ansible(432) = 2,070 total train/val samples
- Maintains GLITCH's natural distribution within each technology

Run:

```bash
# Sample all technologies
python src/iac_filter_training/detection_sampler.py

# Sample single technology
python src/iac_filter_training/detection_sampler.py --iac-tech chef
```

Outputs: `data/iac_filter_training/GLITCH-{iac_tech}-oracle_sampled.csv`

#### 1. Extract detections + context (±5 lines)

`src/iac_filter_training/extractor.py`

- Loads GLITCH detections
- Filters four smells, adds unique IDs
- Finds IaC files and wraps target line with ±5 lines context
- Supports Chef, Ansible, and Puppet with `--iac-tech` parameter

Run:

```bash
# For Chef
python src/iac_filter_training/extractor.py --iac-tech chef

# For Ansible
python src/iac_filter_training/extractor.py --iac-tech ansible

# For Puppet
python src/iac_filter_training/extractor.py --iac-tech puppet
```

Outputs (dir): `experiments/iac_filter_training/data/{iac_tech}/`

- `{iac_tech}_*_detections.csv`
- `{iac_tech}_*_detections_with_context.csv`
- `{iac_tech}_pseudo_label_summary.csv`

#### 2. Post-filter with LLM (pseudo-label TP/FP)

`src/iac_filter_training/post_filter.py`

Key flags:

- `--iac-tech` chef|ansible|puppet (default: chef)
- `--provider` openai|anthropic|grok|ollama|openai_compatible (default: openai)
- `--model` model name (default: gpt-4o-mini)
- `--base-url` for openai_compatible/ollama
- `--prompt-template` default: static_analysis_rules
- `--max-samples` limit for quick runs
- `--data-dir` input dir (auto-detected)
- `--results-dir` output dir (auto-detected)
- `--input` run on single context CSV file

Examples:

```bash
# OpenAI GPT-4o mini
python src/iac_filter_training/post_filter.py --iac-tech chef --provider openai --model gpt-4o-mini

# Anthropic Claude
python src/iac_filter_training/post_filter.py --iac-tech chef --provider anthropic --model claude-3-5-sonnet-latest

# xAI Grok
python src/iac_filter_training/post_filter.py --iac-tech chef --provider grok --model grok-code-fast-1

# OpenRouter (via openai_compatible)
python src/iac_filter_training/post_filter.py --iac-tech chef --provider openai_compatible --model openai/gpt-4o-mini --base-url https://openrouter.ai/api/v1

# Test run with limited samples
python src/iac_filter_training/post_filter.py --iac-tech chef --max-samples 10
```

Outputs (dir): `experiments/iac_filter_training/data/{iac_tech}/llm_results/`

- `*_llm_filtered.csv` (adds: llm_decision, keep_detection)
- `*_llm_summary.json`
- `*_prompts_and_responses.json` (debug log)

#### 3. Format JSONL (HF-ready) and split train/val/test

`src/iac_filter_training/dataset_formatter.py`

Behavior:

- Loads per-smell `*_llm_filtered.csv` + matching `*_prompts_and_responses.json`
- Emits JSONL with fields: smell, file, content, line, detection_span, with_context, with_prompt, label, source
- Prevents data leakage by excluding test files from train/val sets
- Uses **stratified sampling** to maintain GLITCH's natural smell proportions within each technology
- Auto-calculates train/val sizes (8:1 ratio) while preserving proportional distributions
- Creates `{iac_tech}_train.jsonl` and `{iac_tech}_val.jsonl` with maintained proportions
- Creates separate test summaries for each IaC technology

Run:

```bash
# For single technology
python src/iac_filter_training/dataset_formatter.py --iac-tech chef

# For combined dataset from all technologies
python src/iac_filter_training/dataset_formatter.py --combined
```

Outputs for single technology (dir): `experiments/iac_filter_training/data/{iac_tech}/formatted_dataset/`

- `{iac_tech}_train.jsonl`
- `{iac_tech}_val.jsonl`
- `{iac_tech}_test.jsonl` (if test data exists)
- `dataset_summary.json` (train/val statistics)
- `test_summary.json` (test statistics, if applicable)

Outputs for combined dataset (dir): `experiments/iac_filter_training/data/formatted_dataset/`

- `combined_train.jsonl` (1840 samples with maintained GLITCH proportions)
- `combined_val.jsonl` (230 samples with maintained GLITCH proportions)
- `dataset_summary.json` (combined statistics with technology and smell distributions)

---

### Directory Structure

```
experiments/iac_filter_training/data/
├── chef/                    # Chef-specific workspace
│   ├── *.csv               # Raw detections & context files
│   ├── formatted_dataset/
│   │   ├── chef_train.jsonl
│   │   ├── chef_val.jsonl
│   │   ├── chef_test.jsonl
│   │   ├── dataset_summary.json
│   │   └── test_summary.json
│   └── llm_results/        # LLM filtering results
├── ansible/                # Ansible workspace
├── puppet/                 # Puppet workspace
└── formatted_dataset/      # Combined dataset (when using --combined)
    ├── combined_train.jsonl
    ├── combined_val.jsonl
    └── dataset_summary.json
```
