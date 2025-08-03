# LLM-IaC-SecEval 🔍

**Large Language Models for Infrastructure as Code Security Evaluation**

An automated evaluation framework for benchmarking Large Language Models (LLMs) on Infrastructure as Code (IaC) security smell detection across Ansible, Chef, and Puppet scripts.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 **Project Overview**

This project automates the evaluation of Large Language Models on Infrastructure as Code (IaC) security smell detection, providing quantitative benchmarking across multiple models and methodologies.

### **Research Goals**

- 📊 **Quantitative Benchmarking**: Compare LLM performance on security smell detection
- 🔍 **Multi-Model Evaluation**: Test Code Llama 7B, GPT-4o, Grok-4, Claude, etc.
- 🧩 **Methodology Comparison**: Zero-shot vs Modular prompting vs RAG (future)
- 📈 **Scalable Evaluation**: Process 241 IaC files with automated metrics

### **Key Features**

- ✅ **Automated Pipeline**: From manual copy-paste to full automation
- ✅ **Prompt Transparency**: See exactly what's sent to LLMs (`--show-prompts`)
- ✅ **Multi-Technology Support**: Ansible, Chef, Puppet
- ✅ **Comprehensive Metrics**: Precision, Recall, F1-score, detailed error analysis
- ✅ **Flexible Model Integration**: Local (Ollama) and cloud APIs
- ✅ **Reproducible Results**: Versioned experiments with structured output

## 🏗️ **Project Structure**

```
LLM-IaC-SecEval/
├── 📁 src/                           # Source code organized by approach
│   ├── 🤖 automated/                 # Automated evaluation pipeline
│   │   ├── __init__.py               # Package initialization
│   │   ├── config.py                 # Configuration management
│   │   ├── pipeline.py               # Main evaluation orchestrator
│   │   ├── prompt_builder.py         # Modular prompt construction
│   │   ├── model_client.py           # Abstract LLM client interface
│   │   ├── ollama_client.py          # Local Ollama implementation
│   │   ├── file_processor.py         # IaC file and ground truth loading
│   │   └── evaluator.py              # Metrics calculation and analysis
│   ├── 📝 prompts/                   # Shared prompt templates
│   │   ├── Template.txt              # Concise prompt template
│   │   ├── Template_detailed.txt     # Full context with embedded definitions
│   │   └── Template_instructions_only.txt # Instructions only (modular)
│   ├── 🔧 zero-shot/                 # Manual zero-shot experiments
│   │   └── prompt/                   # Original manual templates
│   └── 🚀 RAG/                       # Future: Retrieval-Augmented Generation
├── 📊 data/                          # Datasets and ground truth (gitignored)
│   ├── oracle-dataset-ansible/       # 81 Ansible IaC scripts
│   ├── oracle-dataset-chef/          # 80 Chef IaC scripts
│   ├── oracle-dataset-puppet/        # 80 Puppet IaC scripts
│   ├── oracle-dataset-ansible.csv    # Ansible ground truth annotations
│   ├── oracle-dataset-chef.csv       # Chef ground truth annotations
│   ├── oracle-dataset-puppet.csv     # Puppet ground truth annotations
│   └── smells-description.txt        # Security smell definitions
├── 🧪 scripts/                       # Executable scripts
│   ├── run_evaluation.py             # Main evaluation script
│   ├── validate_pipeline.py          # Setup validation and testing
│   └── cleanup_results.py            # Results directory management
├── 📋 docs/                          # Documentation
│   ├── README_PIPELINE.md            # Detailed pipeline documentation
│   └── MAKEFILE_USAGE.md             # Makefile commands guide
├── 🗂️ results/                       # Evaluation outputs (gitignored)
│   ├── full_evaluation_*.json        # Complete evaluation reports
│   ├── batch_*.json                  # Per-technology results
│   ├── raw_responses/                # Individual LLM responses
│   ├── prompts/                      # Saved constructed prompts
│   └── evaluations/                  # Detailed analysis reports
├── 🧪 experiments/                   # Research experiments (gitignored)
│   └── zero-shot/                    # Manual experiment records
├── ⚙️ Makefile                       # Development workflow automation
├── 📦 requirements.txt               # Python dependencies
├── 🚫 .gitignore                     # Git ignore patterns
└── 📖 README.md                      # This file
```

## 🚀 **Quick Start**

### **Prerequisites**

- Python 3.8+
- [Ollama](https://ollama.ai/) for local LLM inference
- 8GB+ RAM (recommended for Code Llama 7B)

### **Installation**

```bash
# 1. Clone the repository
git clone <repository-url>
cd LLM-IaC-SecEval

# 2. Install dependencies
make install
# or: pip install -r requirements.txt

# 3. Setup Ollama and download model
make ollama-pull
# or: ollama pull codellama:7b

# 4. Validate setup
make validate
```

### **Basic Usage**

```bash
# Quick test (1 file)
make test

# Small evaluation (15 files)
make small

# Full evaluation (241 files)
make run

# Clean up results
make clean
```

## 🔬 **Security Smells Detected**

The framework evaluates LLMs on detecting **9 categories** of IaC security smells:

1. **🔑 Admin by default**: Default administrative user privileges
2. **🔒 Empty password**: Zero-length password strings
3. **🗝️ Hard-coded secret**: Exposed passwords, keys, or tokens
4. **⚠️ Missing default in case statement**: Incomplete conditional logic
5. **🔍 No integrity check**: Downloads without checksum validation
6. **💬 Suspicious comment**: TODO, FIXME, HACK comments indicating issues
7. **🌐 Unrestricted IP address**: 0.0.0.0 bindings allowing all connections
8. **🔓 Use of HTTP without SSL/TLS**: Unencrypted communication
9. **🔐 Use of weak cryptography**: MD5, SHA-1 algorithms

### **Example Detection**

```yaml
# Hard-coded secret smell
- name: Configure database
  mysql_user:
    name: admin
    password: "admin123" # ← Security smell detected
    state: present
```

## 📊 **Evaluation Approaches**

### **1. Manual Zero-Shot** (Legacy)

- Manual copy-paste of prompts and file contents
- Time-intensive and error-prone
- Located in: `src/zero-shot/`

### **2. Automated Modular Prompting** (Current)

- Programmatic prompt assembly with context separation
- Consistent formatting and reproducible results
- Located in: `src/automated/`
- **Usage**: `make run` (default)

### **3. Retrieval-Augmented Generation** (Future)

- Dynamic context retrieval based on query relevance
- Planned for: `src/RAG/`

## 🧪 **Experiment Workflow**

### **Development Cycle**

```bash
# 1. Validate environment
make validate

# 2. Quick iteration
make test                    # Single file test
make small                   # 15 files

# 3. Debug prompts
python scripts/run_evaluation.py --show-prompts --limit 1

# 4. Technology-specific evaluation
make ansible                 # 81 files
make chef                    # 80 files
make puppet                  # 80 files

# 5. Full evaluation
make run                     # 241 files

# 6. Clean up
make clean-keep-3           # Keep latest 3 experiments
```

### **Model Comparison**

```bash
# Compare models
python scripts/run_evaluation.py --model codellama:7b --small-batch
python scripts/run_evaluation.py --model mistral:7b --small-batch

# Compare approaches
python scripts/run_evaluation.py --small-batch              # Modular
python scripts/run_evaluation.py --small-batch --no-modular # Full context
```

## 📈 **Results and Metrics**

### **Evaluation Metrics**

- **Precision**: Accuracy of positive predictions
- **Recall**: Coverage of actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **True Positives/Negatives**: Detailed classification results
- **Agreement Score**: Match with expert annotations

### **Output Format**

```json
{
  "experiment_id": "20250803_145459",
  "model": "codellama:7b",
  "approach": "modular",
  "summary": {
    "ansible": { "f1": 0.234, "precision": 0.189, "recall": 0.301 },
    "chef": { "f1": 0.156, "precision": 0.134, "recall": 0.187 },
    "puppet": { "f1": 0.198, "precision": 0.165, "recall": 0.245 },
    "average_f1": 0.196
  },
  "files_processed": 241,
  "total_predictions": 1247,
  "processing_time": "45.2 minutes"
}
```

## 🔧 **Advanced Usage**

### **Prompt Debugging**

```bash
# See constructed prompts
python scripts/run_evaluation.py --show-prompts --limit 1

# Save prompts to files
python scripts/run_evaluation.py --save-prompts --small-batch

# Compare prompt modes
python scripts/run_evaluation.py --show-prompts --limit 1 --no-modular
```

### **Custom Parameters**

```bash
# Adjust generation parameters
python scripts/run_evaluation.py \
  --temperature 0.2 \
  --max-tokens 1024 \
  --limit 50

# Connect to remote Ollama
python scripts/run_evaluation.py --ollama-url http://remote-server:11434

# Filter by technology
python scripts/run_evaluation.py --iac-tech ansible --limit 20
```

### **Results Management**

```bash
# Analyze results directory
make clean-analyze

# Archive specific experiments
python scripts/cleanup_results.py --archive 20250803_145459 20250803_142106

# Keep only recent experiments
make clean-keep-3

# Delete old experiments
python scripts/cleanup_results.py --delete-older-than 7
```

## 🛠️ **Development**

### **Adding New Models**

1. Implement `ModelClient` interface in `src/automated/model_client.py`
2. Create client class (e.g., `openai_client.py`)
3. Update `scripts/run_evaluation.py` to support new client

### **Adding New Metrics**

1. Extend `Evaluator` class in `src/automated/evaluator.py`
2. Implement custom metric calculation
3. Update result aggregation in `pipeline.py`

### **Testing Changes**

```bash
# Quick validation
make validate

# Single file test
make test

# Small batch before full evaluation
make small
```

## 📚 **Documentation**

- **[Pipeline Documentation](docs/README_PIPELINE.md)**: Detailed technical specifications
- **[Makefile Usage](docs/MAKEFILE_USAGE.md)**: Complete command reference
- **Code Documentation**: Inline docstrings throughout codebase

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- Infrastructure as Code security smell definitions based on research by Rahman et al. (2021)
- [Ollama](https://ollama.ai/) for local LLM inference capabilities
- Dataset contributors for expert-annotated ground truth

---

**🔬 Research Note**: This project is designed for academic research and evaluation purposes. Results and methodologies are continually evolving as we explore the intersection of Large Language Models and Infrastructure as Code security.
