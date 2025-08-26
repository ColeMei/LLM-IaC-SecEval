### Stage 1 — Dataset Expansion (Chef, Pseudo-Labels)

Objective: Build train/validation datasets for Chef using GLITCH detections + LLM post-filtering (pseudo-labels).

#### Pipeline Steps

1. Collect unlabeled Chef IaC scripts (from GitHub)

- Target: ~1000+ scripts (stored under `data/iac_filter_training/oracle-dataset_1000`).

2. Run GLITCH → collect raw detections

- Input: `data/iac_filter_training/GLITCH-chef-oracle.csv`
- Smells covered: Hard-coded secret, Suspicious comment, Use of weak cryptography algorithms, Use of HTTP without SSL/TLS

3. Run LLM post-filter → pseudo-label TP/FP with confidence

- Default: `gpt-4o-mini` for smoke tests; switch to Claude 3.7 for production
- Prompt style: `static_analysis_rules`
- Output: per-smell CSVs with decisions and confidence, plus prompt/response logs

4. Format dataset into JSONL (HF-ready) and split by confidence

5. Spot-check random samples manually to estimate error rate

---

### Files and Usage

#### 1. Extract detections + context (±5 lines)

`src/iac_filter_training/extractor.py`

- Loads GLITCH detections, filters four smells, adds unique IDs
- Finds Chef files (handles `epos-` prefixes) and wraps target line with ±5 lines
- Writes per-smell CSVs: `chef_*_detections.csv` and `chef_*_detections_with_context.csv`

Run:

```bash
python src/iac_filter_training/extractor.py
```

Outputs (dir): `experiments/iac_filter_training/data/`

- `chef_*_detections.csv`
- `chef_*_detections_with_context.csv`
- `chef_pseudo_label_summary.csv`

#### 2. Post-filter with LLM (pseudo-label TP/FP)

`src/iac_filter_training/post_filter.py`

Key flags:

- `--provider` openai|anthropic|ollama|openai_compatible (default: openai)
- `--model` model name (default: gpt-4o-mini)
- `--base-url` for openai_compatible/ollama
- `--prompt-template` default: static_analysis_rules
- `--max-samples` limit for quick runs
- `--data-dir` input dir (default: experiments/iac_filter_training/data)
- `--results-dir` output dir (default: experiments/iac_filter_training/data/llm_results)
- `--input` run on one `*_detections_with_context.csv`
- `--json-confidence` enforce JSON response with {decision, confidence}

Examples:

```bash
# Quick sanity run on all smells (OpenAI cheap model)
python src/iac_filter_training/post_filter.py --provider openai --model gpt-4o-mini --max-samples 30 --json-confidence

# Run Anthropic Claude 3.7 on all data
python src/iac_filter_training/post_filter.py --provider anthropic --model claude-3-7-sonnet-20250219 --json-confidence

# Single-file run
python src/iac_filter_training/post_filter.py --input experiments/iac_filter_training/data/chef_hardcoded_secret_detections_with_context.csv --json-confidence
```

Outputs (dir): `experiments/iac_filter_training/data/llm_results/`

- `*_llm_filtered.csv` (adds: llm_decision, llm_confidence, keep_detection)
- `*_llm_summary.json`
- `*_prompts_and_responses.json` (debug log: prompts + raw responses)

#### 3. Format JSONL (HF-ready) and split by confidence

`src/iac_filter_training/dataset_formatter.py`

Behavior:

- Loads per-smell `*_llm_filtered.csv` + matching `*_prompts_and_responses.json`
- Keeps only TP (YES)
- Emits JSONL with fields: smell, file, content, line, detection_span, with_context, with_prompt, label, confidence, source
- Sorts by confidence and writes `chef_train.jsonl` (top-N) and `chef_val.jsonl`

Run:

```bash
# Create train/val JSONL (adjust sizes as needed)
python src/iac_filter_training/dataset_formatter.py --train-size 640 --val-size 80
```

Outputs (dir): `experiments/iac_filter_training/data/formatted_dataset/`

- `chef_train.jsonl`
- `chef_val.jsonl`
- `dataset_summary.json`

---

### Notes

- Do not modify `src/llm_postfilter/*`; these are reused by import only
- Use `--json-confidence` to ensure confidence is parsed regardless of provider quirks
- File matching handles `epos-` filename prefixes automatically
- Context window is ±5 lines by default and already baked into the CSV
