# Makefile Usage Guide

This document describes all available Makefile commands for the LLM-IaC-SecEval project.

## 📋 **Quick Reference**

```bash
make help          # Show all available commands
make install       # Install Python dependencies
make validate      # Validate pipeline setup
make run           # Run full evaluation
make clean         # Interactive results cleanup
```

## 🚀 **Setup Commands**

### `make install`

Installs all Python dependencies from `requirements.txt`.

```bash
make install
```

**What it does:**

- Runs `pip install -r requirements.txt`
- Installs: `pandas`, `requests`, `pathlib` and other dependencies

**When to use:**

- First time setting up the project
- After updating `requirements.txt`
- When dependencies are missing

### `make validate`

Validates that the automated evaluation pipeline is properly configured.

```bash
make validate
```

**What it tests:**

- ✅ Ollama server connectivity
- ✅ Required models availability (Code Llama 7B)
- ✅ Data directory structure
- ✅ Ground truth files
- ✅ Results directory permissions
- ✅ End-to-end single file processing

**Expected output:**

```
Testing file processing...
✓ FileProcessor initialized successfully
✓ Found 3 technologies: ['ansible', 'chef', 'puppet']
✓ ANSIBLE: 81 files, ground truth available
✓ CHEF: 80 files, ground truth available
✓ PUPPET: 80 files, ground truth available

Testing Ollama connectivity...
✓ Ollama server available at http://localhost:11434
✓ Model codellama:7b is available

Testing end-to-end processing...
✓ Successfully processed test file
✓ Generated response and extracted predictions
```

## 🧪 **Evaluation Commands**

### `make run` (alias: `make full`)

Runs complete evaluation across all IaC technologies.

```bash
make run
# Equivalent to:
python scripts/run_evaluation.py
```

**What it does:**

- Processes all files in `data/oracle-dataset-*/`
- Uses modular prompting approach (default)
- Evaluates against ground truth
- Saves results to timestamped files

**Typical runtime:** 30-60 minutes for full dataset (241 files)

### `make test`

Quick test on a single file to verify pipeline functionality.

```bash
make test
```

**What it does:**

- Processes 1 file from Ansible dataset
- Uses default settings (modular prompting, Code Llama 7B)
- Shows complete output for debugging

**Typical runtime:** 30-60 seconds

### `make small-batch` (alias: `make small`)

Runs evaluation on a small subset (5 files per technology).

```bash
make small
# Equivalent to:
python scripts/run_evaluation.py --small-batch
```

**What it does:**

- Processes 5 files from each technology (15 total)
- Perfect for testing changes
- Faster than full evaluation

**Typical runtime:** 3-5 minutes

## 🎯 **Technology-Specific Commands**

### `make ansible`

Evaluates only Ansible files (81 files).

```bash
make ansible
# Equivalent to:
python scripts/run_evaluation.py --iac-tech ansible
```

### `make chef`

Evaluates only Chef files (80 files).

```bash
make chef
# Equivalent to:
python scripts/run_evaluation.py --iac-tech chef
```

### `make puppet`

Evaluates only Puppet files (80 files).

```bash
make puppet
# Equivalent to:
python scripts/run_evaluation.py --iac-tech puppet
```

**Use cases:**

- Focusing on specific technology
- Comparing performance across technologies
- Debugging technology-specific issues

## 🧹 **Maintenance Commands**

### `make clean`

Interactive cleanup of the results directory.

```bash
make clean
```

**What it offers:**

- 📊 Analysis of current results directory
- 📦 Archive old experiments
- 🗑️ Delete experiments older than X days
- 🔍 Remove duplicate files
- 🛠️ Custom cleanup options

**Sample interaction:**

```
📊 Results Directory Analysis
==================================================
📁 Directory: /path/to/results
📄 Total files: 20
💾 Total size: 0.4 MB

🧹 Cleanup Options:
1. Archive old experiments (move to archive/ folder)
2. Delete experiments older than X days
3. Remove duplicate files
4. Custom cleanup
5. Exit

Select option (1-5):
```

### `make clean-analyze`

Shows analysis of results directory without making changes.

```bash
make clean-analyze
# Equivalent to:
python scripts/cleanup_results.py --analyze
```

**Output example:**

```
📂 Subdirectories:
   evaluations: 0 files, 0.0 MB
   raw_responses: 22 files, 0.1 MB
   prompts: 7 files, 0.0 MB

📋 Files by type:
   full_evaluations: 10 files
   batch_results: 10 files

🧪 Experiments by timestamp:
   20250803_145459 (2025-08-03 14:54:59): 2 files
   20250803_145436 (2025-08-03 14:54:36): 2 files
   ...
```

### `make clean-keep-3`

Keeps only the 3 most recent experiments, deletes older ones.

```bash
make clean-keep-3
# Equivalent to:
python scripts/cleanup_results.py --keep-latest 3
```

**Safe usage:** Always run `make clean-dry-run` first to preview changes.

### `make clean-dry-run`

Shows what `make clean-keep-3` would delete without actually deleting.

```bash
make clean-dry-run
# Output: Would delete 7 old experiments (14 files)
```

## 🛠️ **Utility Commands**

### `make ollama-pull`

Downloads the required Code Llama 7B model.

```bash
make ollama-pull
# Equivalent to:
ollama pull codellama:7b
```

**When to use:**

- First time setup
- Model is missing or corrupted
- After Ollama reinstallation

**Typical download:** ~4GB, takes 5-15 minutes depending on connection

## 🔧 **Advanced Usage Examples**

### Development Workflow

```bash
# 1. Setup (first time)
make install
make ollama-pull
make validate

# 2. Quick test after changes
make test

# 3. Small evaluation
make small

# 4. Clean up development artifacts
make clean-keep-3

# 5. Full evaluation when ready
make run
```

### Debugging Issues

```bash
# Check setup
make validate

# Test single technology
make ansible

# Check results organization
make clean-analyze

# Trace specific issues
python scripts/run_evaluation.py --show-prompts --limit 1 --iac-tech ansible
```

### Performance Testing

```bash
# Quick performance check
time make small

# Technology comparison
time make ansible
time make chef
time make puppet

# Full benchmark
time make run
```

## 📊 **Understanding Output**

### Successful Run Output

```
============================================================
LLM-IaC-SecEval Automated Pipeline
============================================================

1. Validating setup...
   Model available: ✓
   Data directory: ✓
   Results directory: ✓
   ANSIBLE: ✓ 81 files, True GT
   CHEF: ✓ 80 files, True GT
   PUPPET: ✓ 80 files, True GT

2. Running evaluation...
   Technologies: ['ansible', 'chef', 'puppet']
   Model: codellama:7b
   Modular prompting: True
   Limit per tech: all
   Generation params: {'temperature': 0.1, 'max_tokens': 512}

==================================================
Processing ANSIBLE
==================================================
Processing file 1: example-file.yml
Processing file 2: another-file.yml
...

3. Evaluation complete!
   Experiment ID: 20250803_145459
   ANSIBLE: F1=0.234, P=0.189, R=0.301
   CHEF: F1=0.156, P=0.134, R=0.187
   PUPPET: F1=0.198, P=0.165, R=0.245
   Average F1: 0.196

✓ Results saved in: /path/to/results
```

### Output Files Created

- `results/full_evaluation_YYYYMMDD_HHMMSS.json` - Complete evaluation report
- `results/batch_TECH_YYYYMMDD_HHMMSS.json` - Per-technology detailed results
- `results/raw_responses/FILE_HHMMSS.json` - Individual LLM responses (if enabled)
- `results/prompts/prompt_MODE_FILE_HHMMSS.txt` - Constructed prompts (if `--save-prompts`)

## ⚠️ **Troubleshooting**

### Common Issues

**🔴 "Ollama model not available"**

```bash
# Solution:
make ollama-pull
# or start Ollama service:
ollama serve
```

**🔴 "No module named 'automated'"**

```bash
# Solution: Install dependencies
make install
# and ensure you're in project root directory
```

**🔴 "Permission denied" on results directory**

```bash
# Solution: Create directory with proper permissions
mkdir -p results/raw_responses results/evaluations results/prompts
chmod 755 results results/*
```

**🔴 "Out of memory" during evaluation**

```bash
# Solution: Use smaller batches
python scripts/run_evaluation.py --limit 10
# or switch to smaller model variant
python scripts/run_evaluation.py --model codellama:7b-code
```

### Getting Help

- `make help` - Show all available commands
- `make validate` - Check system setup
- `python scripts/run_evaluation.py --help` - Detailed script options
- Check logs in `results/` directory for detailed error information

## 🎯 **Best Practices**

1. **Always validate before running:** `make validate`
2. **Test changes with small batches:** `make small`
3. **Clean up regularly:** `make clean-analyze` and `make clean-keep-3`
4. **Use dry-run for destructive operations:** `make clean-dry-run`
5. **Monitor disk space** - full evaluations can generate significant data
6. **Archive important results** before cleanup: Use `make clean` → Archive option

---

**📝 Note:** This Makefile is designed for development and research workflows. For production usage, consider using the Python scripts directly with specific parameters.
