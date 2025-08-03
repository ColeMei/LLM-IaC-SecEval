# Makefile Usage Guide

Quick reference for all available Makefile commands.

## Quick Reference

```bash
make help          # Show all commands
make install       # Install dependencies
make validate      # Validate setup
make run           # Full evaluation (241 files)
make small         # Quick test (15 files)
make test          # Single file test
make clean         # Interactive cleanup
```

## Setup Commands

**`make install`** - Install Python dependencies

```bash
make install
```

**`make validate`** - Validate pipeline setup

```bash
make validate
```

Tests Ollama connectivity, data files, and runs end-to-end test.

**`make ollama-pull`** - Download Code Llama 7B model

```bash
make ollama-pull
```

## Evaluation Commands

**`make run`** (alias: `make full`) - Complete evaluation

```bash
make run
# Processes all 241 files across ansible/chef/puppet
# Runtime: 30-60 minutes
```

**`make small`** (alias: `make small-batch`) - Quick test

```bash
make small
# Processes 5 files per technology (15 total)
# Runtime: 3-5 minutes
```

**`make test`** - Single file test

```bash
make test
# Tests 1 ansible file
# Runtime: 30-60 seconds
```

## Technology-Specific Commands

```bash
make ansible       # 81 ansible files only
make chef          # 80 chef files only
make puppet        # 80 puppet files only
```

## Maintenance Commands

**`make clean`** - Interactive cleanup

```bash
make clean
# Options: archive, delete old, remove duplicates
```

**`make clean-analyze`** - Show results directory analysis

```bash
make clean-analyze
# Shows file counts, sizes, experiment timestamps
```

**`make clean-keep-3`** - Keep latest 3 experiments

```bash
make clean-keep-3
# Deletes older experiment files
```

**`make clean-dry-run`** - Preview cleanup actions

```bash
make clean-dry-run
# Shows what would be deleted without doing it
```

## Development Workflow

```bash
# Setup (first time)
make install && make ollama-pull && make validate

# Development cycle
make test           # Quick validation
make small         # Test changes
make clean-keep-3  # Clean up
make run           # Full evaluation

# Debugging
make validate                                    # Check setup
python scripts/run_evaluation.py --show-prompts --limit 1
```

## Common Issues

**"Ollama model not available"**

```bash
make ollama-pull
# or: ollama serve
```

**"No module named 'automated'"**

```bash
make install
```

**Memory issues**

```bash
python scripts/run_evaluation.py --limit 10
```

## Output Files

- `results/full_evaluation_*.json` - Complete reports
- `results/batch_*.json` - Per-technology results
- `results/raw_responses/` - Individual responses
- `results/prompts/` - Constructed prompts (if saved)
