# LLM-IaC-SecEval Pipeline Makefile

.PHONY: install validate test small-batch full help clean

# Default target
help:
	@echo "LLM-IaC-SecEval Pipeline Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  install     - Install Python dependencies"
	@echo "  validate    - Validate pipeline setup"
	@echo ""
	@echo "Execution:"
	@echo "  test        - Run single file test"
	@echo "  small-batch - Run evaluation on 5 files per technology"
	@echo "  full        - Run full evaluation on all datasets"
	@echo ""
	@echo "Specific technologies:"
	@echo "  ansible     - Run evaluation on Ansible files only"
	@echo "  chef        - Run evaluation on Chef files only"
	@echo "  puppet      - Run evaluation on Puppet files only"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean       - Interactive results cleanup"
	@echo "  clean-analyze - Analyze results directory"
	@echo "  clean-keep-3  - Keep only latest 3 experiments"
	@echo ""
	@echo "Utilities:"
	@echo "  ollama-pull - Pull Code Llama 7B model"
	@echo ""

# Install dependencies
install:
	pip install -r requirements.txt

# Validate pipeline setup
validate:
	python scripts/validate_pipeline.py

# Run single file test
test:
	python scripts/run_evaluation.py --validate-only

# Run small batch evaluation
small-batch:
	python scripts/run_evaluation.py --small-batch

# Run full evaluation
full:
	python scripts/run_evaluation.py

# Technology-specific evaluations
ansible:
	python scripts/run_evaluation.py --iac-tech ansible

chef:
	python scripts/run_evaluation.py --iac-tech chef

puppet:
	python scripts/run_evaluation.py --iac-tech puppet

# Cleanup commands
clean:
	python scripts/cleanup_results.py --interactive

clean-analyze:
	python scripts/cleanup_results.py --analyze

clean-keep-3:
	python scripts/cleanup_results.py --keep-latest 3

clean-dry-run:
	python scripts/cleanup_results.py --keep-latest 3 --dry-run

# Legacy clean command (use clean-analyze, clean-keep-3 instead)
clean-legacy:
	rm -rf results/*
	mkdir -p results/raw_responses results/evaluations

ollama-pull:
	ollama pull codellama:7b

# Development targets
lint:
	python -m flake8 src/ scripts/ --max-line-length=120 --ignore=E501,W503

format:
	python -m black src/ scripts/ --line-length=120

# Advanced usage examples
rag-comparison:
	@echo "Running RAG vs Full context comparison..."
	python scripts/run_evaluation.py --small-batch --limit 3
	python scripts/run_evaluation.py --small-batch --limit 3 --no-rag

model-comparison:
	@echo "Running model comparison (requires multiple models)..."
	python scripts/run_evaluation.py --small-batch --model codellama:7b
	python scripts/run_evaluation.py --small-batch --model mistral:7b

# Setup ollama (macOS/Linux)
setup-ollama:
	@echo "Setting up Ollama..."
	curl -fsSL https://ollama.ai/install.sh | sh
	@echo "Pulling Code Llama 7B..."
	ollama pull codellama:7b
	@echo "✓ Ollama setup complete!"