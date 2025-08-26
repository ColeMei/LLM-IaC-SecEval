# Project Structure Guide

## Directory Organization

- **data/**: Datasets and ground truth labels for all IaC technologies
- **docs/**: Documentation, technical guides, and project specifications
- **src/**: Core implementation logic for all methodologies
- **experiments/**: Interactive notebooks and execution interfaces
- **results/**: Experiment outputs and evaluation results

## Methodology Structure

### Post-Filter Pipeline

- **src/llm_postfilter/**: Core post-filtering logic and evaluation
- **experiments/llm_postfilter/**: Notebooks for running post-filter experiments
- **results/llm_postfilter/**: Post-filter experiment results

### Model Training Pipeline

- **src/iac_filter_training/**: Training infrastructure for open-source models
- **experiments/iac_filter_training/**: Training notebooks and data pipeline
- **results/iac_filter_training/**: Trained models and evaluation results

## Naming Conventions

- **Directories**: Use `snake_case` for all directory names
- **Files**: Clear, descriptive names reflecting functionality
- **Methods**: Consistent naming across src/, experiments/, and results/
- **Technology**: Separate concerns by IaC technology when needed

---

## Adding New Components

1. **Documentation**: Add specifications in `docs/[component].md`
2. **Implementation**: Create modular code in `src/[component]/`
3. **Experiments**: Develop notebooks in `experiments/[component]/`
4. **Results**: Store outputs in `results/[component]/[timestamp]/`

## Best Practices

- **Modular Design**: Keep components independent and reusable
- **Clear Separation**: Maintain distinct boundaries between logic, interface, and results
- **Consistent Structure**: Mirror organization across src/, experiments/, and results/
- **Reproducibility**: Document all parameters, data sources, and configurations
- **Version Control**: Tag significant milestones and maintain clean history

---

## Project Structure Example

```
data/
  oracle-dataset-chef/          # Ground truth datasets
  oracle-dataset-puppet/
  oracle-dataset-ansible/
  training-datasets/            # Expanded datasets for training
    chef/
    puppet/
    ansible/

src/
  llm_postfilter/              # Post-filter implementation
    llm_filter.py              # Core filtering logic
    llm_client.py              # Multi-provider LLM integration
    evaluator.py               # Metrics and evaluation
  iac_filter_training/              # Training pipeline
    data_pipeline.py           # Data processing and preparation
    trainer.py                 # Model training infrastructure
    evaluator.py               # Training evaluation

experiments/
  llm_postfilter/              # Post-filter experiments
    01_data_extraction.ipynb
    02_llm_experiment.ipynb
  iac_filter_training/              # Training experiments
    01_data_preparation.ipynb
    02_iac_filter_training.ipynb
    03_evaluation.ipynb

results/
  llm_postfilter/              # Post-filter results
    [timestamp]/
  iac_filter_training/              # Training results
    models/
    evaluations/

docs/
  Research_Roadmap.md          # Project roadmap
  LLM_PostFilter.md            # Post-filter documentation
  iac_filter_training.md            # Training documentation
  Project_Structure_Guide.md   # This guide
```

---

**Summary:**

- **Clear separation**: Distinct boundaries between implementation, experimentation, and results
- **Modular design**: Independent components that can be developed and tested separately
- **Scalable structure**: Organization supports both current work and future expansion
- **Reproducible research**: Comprehensive documentation and version control

This structure supports both research development and practical implementation while maintaining clarity and organization.
