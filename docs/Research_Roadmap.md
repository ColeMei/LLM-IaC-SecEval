# Research Roadmap

## Goal

**Reduce false positives** from static analysis tools (GLITCH) using Large Language Models as intelligent post-filters. Train **open-source models** that match closed-source API performance at significantly lower cost.

## Problem Definition

### Challenge

- Static analysis tools produce many false positives → reduces usability
- LLM post-filtering works well but closed models are expensive for production
- Need cost-effective solution for real-world DevOps integration

### Research Question

**Can we train open-source models to perform false positive filtering as effectively as Claude 3.7 Sonnet?**

### Success Criteria

- Match Claude's precision improvement (+40-60%)
- Maintain high true positive retention (>80%)
- Achieve significant cost reduction vs API calls
- Demonstrate practical deployment feasibility

## Current Work: Post-Filter Validation

### Methodology Validation

- **Post-filter LLM >> Pure LLM** for security smell detection
- **Claude 3.7 Sonnet** selected as best performing model
- **+29% to +60% precision improvement** demonstrated
- **Oracle dataset validated** (80 scripts each for Ansible, Chef, Puppet)

### Target Security Smells

1. **Hard-coded secrets** - Distinguish real vs placeholders
2. **Suspicious comments** - Security-relevant vs general TODOs
3. **Weak cryptography** - Actual usage vs documentation
4. **HTTP without SSL/TLS** - Insecure patterns vs examples

### Architecture

```
GLITCH Detection → LLM Post-Filter → Reduced False Positives
(High Recall)      (High Precision)   (Practical Usage)
```

## Future Work: Open-Source Model Development

### Objective

Create training datasets and develop open-source alternatives to Claude 3.7 Sonnet for IaC security analysis.

### Chef-First Strategy

**Rationale**: Validate approach on single technology before expanding to others

### Dataset Expansion

#### Target Dataset Sizes

- **Train**: 640 Chef scripts (8x oracle size)
- **Validation**: 80 Chef scripts (1x oracle size)
- **Test**: 80 Chef scripts (existing oracle dataset)

#### Data Collection Process

1. **GitHub Mining**: Collect 2000+ Chef IaC scripts from repositories
2. **GLITCH Analysis**: Run static analysis on collected scripts
3. **Pseudo-labeling**: Apply Claude 3.7 post-filter for TP/FP classification
4. **Quality Control**: Store confidence scores, select high-confidence samples, manual validation

#### Data Format

```json
{
  "input": "<detection_line_with_context>",
  "label": "TP" | "FP",
  "confidence": 0.93,
  "smell_type": "hard_coded_secret"
}
```

### Model Training Approaches

#### Generative LLM Fine-tuning

- **Models**: CodeLLaMA (32B → 13B → 7B scaling)
- **Method**: LoRA fine-tuning
- **Input**: Prompt template + detection context
- **Output**: "TP" or "FP" classification
- **Advantages**: Contextual reasoning, interpretable
- **Challenges**: Higher computational requirements

#### Encoder-Only Classification

- **Models**: CodeBERT, CodeT5-base
- **Method**: Full fine-tuning on classification head
- **Input**: Raw detection lines
- **Output**: Binary classification probability
- **Advantages**: Computationally efficient, fast inference
- **Challenges**: Less contextual understanding

### Training Infrastructure

#### Infrastructure Requirements

- **GPU Resources**: Multi-GPU setup for large model training
- **Model Management**: Hugging Face integration for versioning
- **Experiment Tracking**: MLflow/Weights & Biases for metrics
- **Evaluation Framework**: Automated testing against oracle dataset

#### Training Configuration

```yaml
generative_model:
  base_model: "codellama/CodeLlama-32b-hf"
  lora_config:
    r: 16
    alpha: 32
  training:
    epochs: 3
    batch_size: 4
    learning_rate: 2e-4

encoder_model:
  base_model: "microsoft/codebert-base"
  training:
    epochs: 5
    batch_size: 16
    learning_rate: 5e-5
```

### Evaluation & Production

#### Comprehensive Evaluation

- **Test Dataset**: Oracle Chef Dataset (80 scripts)
- **Baselines**: GLITCH Raw, GLITCH + Claude 3.7
- **Metrics**: Precision, Recall, F1, FPRR, TP Retention, Inference Speed, Cost Analysis

#### Production Pipeline

```
GitHub Repo → GLITCH → Trained Model → Security Report
```

#### Integration Points

- **CI/CD Integration**: GitHub Actions/GitLab CI plugins
- **CLI Tool**: Standalone command-line interface
- **REST API**: HTTP service for existing tool integration
- **IDE Extensions**: VS Code/IntelliJ plugins

#### Performance Requirements

- **Latency**: <2 seconds per file
- **Throughput**: 100+ files/minute
- **Memory**: <8GB for inference
- **Storage**: <10GB for model weights

## Implementation Plan

### Directory Structure for Future Work

```
src/
  llm_postfilter/              # Current post-filter work
  iac_filter_training/         # New: Model training pipeline
    data_pipeline.py           # Data collection and processing
    model_trainer.py           # Training infrastructure
    evaluator.py               # Model evaluation

experiments/
  llm_postfilter/              # Current experiments
  iac_filter_training/         # New: Training experiments
    01_data_collection.ipynb
    02_pseudo_labeling.ipynb
    03_model_training.ipynb
    04_evaluation.ipynb

results/
  llm_postfilter/              # Current results
  iac_filter_training/         # New: Training results
    datasets/
    models/
    evaluations/
```

### Implementation Tasks

#### Data Pipeline

- Set up GitHub scraping infrastructure
- Implement Chef repository collection (2000+ scripts)
- Deploy Claude 3.7 pseudo-labeling at scale
- Create train/validation dataset splits
- Manual quality assurance

#### Model Training

- Set up multi-GPU training environment
- Implement LoRA fine-tuning pipeline for CodeLLaMA
- Implement full fine-tuning for encoder models
- Experiment tracking and model versioning
- Hyperparameter optimization

#### Evaluation

- Oracle dataset evaluation framework
- Baseline comparison implementation
- Performance benchmarking
- Cost analysis calculations

#### Production Deployment

- Model optimization for inference
- CLI tool development
- REST API implementation
- CI/CD integration plugins

## Expected Outcomes

### Research Contributions

1. **Methodology**: Demonstrate open models can match closed API performance
2. **Cost-Effectiveness**: Achieve >80% cost reduction vs Claude API
3. **Practical Tool**: Production-ready false positive filter
4. **Comparative Analysis**: Generative vs encoder model trade-offs

### Academic Deliverables

- **Paper**: "Open-Source LLM Post-Filtering for IaC Security Analysis"
- **Dataset**: Pseudo-labeled Chef security smell dataset
- **Models**: Fine-tuned models for security analysis
- **Benchmark**: Evaluation framework for IaC security filtering

### Industry Impact

- **DevOps Integration**: Practical tool for security workflow
- **Cost Reduction**: Affordable alternative to expensive API calls
- **Open Source**: Community-driven security tool development

## Future Expansion

### Multi-Technology Support

1. **Chef** → **Puppet** → **Ansible**
2. Cross-technology transfer learning
3. Unified multi-IaC model development

### Advanced Features

- **Multi-class Classification**: Distinguish between different smell types
- **Severity Scoring**: Risk-based prioritization
- **Remediation Suggestions**: AI-powered fix recommendations
- **Custom Rule Integration**: User-defined security patterns

## Success Metrics

### Technical Metrics

- **Precision Improvement**: >40% vs GLITCH baseline
- **TP Retention**: >80% of true positives preserved
- **Inference Speed**: <2 seconds per file
- **Model Size**: <10GB deployable model

### Business Metrics

- **Cost Reduction**: >80% vs Claude API for equivalent performance
- **Adoption**: >100 GitHub stars, >10 production deployments
- **Community**: >5 external contributors to open-source project

### Research Metrics

- **Publication**: Top-tier security/ML conference acceptance
- **Citation Impact**: Referenced by follow-up research
- **Reproducibility**: Fully reproducible results and code release

---

_This roadmap represents our transition from research validation (Stage 1) to practical implementation (Stages 2-3) with Chef as the primary focus before expanding to other IaC technologies._
