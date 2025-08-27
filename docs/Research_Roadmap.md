# Roadmap (Chef Focus)

#### 🎯 Goal

Reduce false positives (FPs) from GLITCH by training a post-filter that classifies each GLITCH detection as TP or FP. Train open models to match/beat Claude 3.7 at much lower cost.

**Problem**: Static analysis tools (even SOTA like GLITCH) generate high false positives (FP).

**Core Contribution**: Use LLM power to filter/reduce FP → making static analysis more usable.

**Method**:

- Use GLITCH detections. DONE
- Pass them to LLMs (post-filter). DONE
- Select best-performing LLM (Claude worked best). DONE
- Use LLM-filtered outputs as pseudo-labels → expand dataset. DONE
- Train our own open models (so we don’t depend on expensive closed APIs).
- Compare approaches: generative LLM vs classification LLM.

#### 🔎 Problem Setup

- Static analysis tools (GLITCH) produce many FPs → harms usability.
- We need a post-filtering model to decide:
  - Detection (Predictable Positive From GLITCH) = True Positive (TP) vs False Positive (FP)
  - A sample = one GLITCH positive detection (with its minimal context), not a script.
  - The model never detects; it only relabels GLITCH detections as TP/FP.
- Closed models (Claude 3.7) work well but are expensive. (Proofed through experiments)
- Core contribution = show open models can match/exceed Claude’s FP reduction ability.

---

#### 📌 3-Stage Pipeline (CHEF Focus)

> #### Stage 1 — Dataset Expansion DONE

**Objective**: Build train/val sets of **detections** using GLITCH + Claude 3.7 as pseudo-labeling.

Steps:

0. Oracle test set (fixed)
   - Test size = sum of detections = 75
   - Get the size of Train / Val backwardly
1. Collect unlabeled Chef IaC scripts (from repos).
   - Target size: ≥ 5000 scripts (assumption)
   - Actual needed (based on oracle 75 detections)
1. Run GLITCH on collected scripts → get detections.
1. Run **Claude 3.7 post-filter** on detections → pseudo-label TP/FP.
   - Store **confidence scores** for each prediction.
   - Sort by confidence, select top-N
1. Format into JSONL/HF style dataset:

```json
{
  "smell": "hard_coded_secret",
  "file": "cookbooks/foo/recipes/bar.rb",
  "content": "password = 'hunter2'",
  "line": 123,
  "detection_span": [120, 128],
  "with_context": "<5-line window with detection>",
  "with_prompt": "<context inserted into template>",
  "label": "TP",
  "confidence": 0.93,
  "source": "claude-3.7"
}
```

1. Spot-check random samples manually to estimate error rate.

#### Stage 2 — Model Training Approaches

We will train **two types of models** on the pseudo-labeled dataset:

#### A. Generative LLM (Decoder/Encoder–Decoder, e.g., CodeLLaMA)

- Input: Prompt + detection lines (wrapped in static-analysis prompt template).
- Output: `"TP"` or `"FP"`.
- Fine-tuning: **LoRA adapters**.
- Scaling strategy: start from 32B → reduce (13B, 7B) until performance drops.

#### B. Encoder-Only Models (e.g., CodeBERT, CodeT5-base)

- Input: Raw detection lines only (no prompt).
- Output: Binary classification.
- Training: **Full fine-tuning**.
- Baseline computationally cheaper.

**Key Comparison Contribution**:

- Which paradigm works better for FP reduction?
- Does context-based reasoning (generative) beat lightweight classifiers?

### Stage 3 — Evaluation

**Test set**: All Predictable Positive from Oracle dataset for Chef

**Baselines**:

- GLITCH raw (no filtering).
- GLITCH + Claude 3.7 post-filter.

**Metrics**:

- Precision, Recall, F1.
- False Positive Reduction Rate (FPRR).
- Cost–benefit:
  - Fine-tuned open model (one-time training).
  - API calls to Claude (recurring).

---

#### 🛠️ Implementation TODOs

### Data Pipeline

- [x] Scrape/download Chef IaC repos.
- [x] Run GLITCH → store detections.
- [x] Run Claude 3.7 filter → save predictions + confidence.
- [x] Select top-N pseudo-labeled samples (720).
- [x] Format into JSONL.
- [ ] Manual spot-check.

### Model Training

- **Generative Path**:
  - [ ] Wrap GLITCH detections in static-analysis prompt template.
  - [ ] Prepare LoRA fine-tuning pipeline (CodeLLaMA 32B, then scale).
- **Encoder Path**:
  - [ ] Convert detection lines → inputs.
  - [ ] Fine-tune CodeBERT/CodeT5.

### Evaluation

- [ ] Test on oracle Chef dataset.
- [ ] Compare vs GLITCH baseline.
- [ ] Compare vs Claude 3.7.
- [ ] Report Precision/Recall/F1/FPRR.
- [ ] Analyze cost trade-off.

---

## 📂 Expected Outputs

- **Datasets**:
  - `chef_train.jsonl`
  - `chef_val.jsonl`
  - `chef_test_oracle.jsonl`
- **Models**:
  - Fine-tuned CodeLLaMA (LoRA adapters).
  - Fine-tuned CodeBERT/CodeT5 classifier.
- **Results**:
  - Table: model vs baseline (metrics + cost).
  - Analysis: trade-offs.
