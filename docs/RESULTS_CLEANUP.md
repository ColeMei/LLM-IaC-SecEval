# Results Management Guide

Documentation for managing experiment results with simplified, workflow-focused cleanup.

## Overview

Simple experiment workflow management:

1. **Run experiments** → Files accumulate in `results/`
2. **Archive experiments** → Move completed experiments to `archive/`
3. **Clean archive** → Delete old archived experiments periodically

With unified timestamps, all files from the same experiment share the same ID, making cleanup reliable and straightforward.

## Quick Start

### Check Status

```bash
python experiments/llm_pure/cleanup_results.py --status
```

### Archive Completed Experiments

```bash
python experiments/llm_pure/cleanup_results.py --archive
```

### Clean Old Archives

```bash
python experiments/llm_pure/cleanup_results.py --clean-archive 30
```

### Interactive Mode

```bash
python experiments/llm_pure/cleanup_results.py
```

## Unified Timestamps

**Key Improvement:** All files from the same experiment now use the same `experiment_id` timestamp:

```
# Same experiment = same timestamp
batch_ansible_20250811_143022.json
prompt_definition_based_file1.pp_20250811_143022.txt
prompt_definition_based_file2.pp_20250811_143022.txt
file1.pp_20250811_143022.json
file2.pp_20250811_143022.json
```

This eliminates complex time proximity matching and makes file grouping 100% reliable.

## Command Options

| Option              | Description                       | Example              |
| ------------------- | --------------------------------- | -------------------- |
| `--status`          | Show current experiment status    | `--status`           |
| `--archive`         | Archive current experiments       | `--archive`          |
| `--clean-archive N` | Delete archives older than N days | `--clean-archive 30` |
| `--dry-run`         | Preview actions without executing | `--dry-run`          |

## Usage Examples

### Check Current Status

```bash
python experiments/llm_pure/cleanup_results.py --status
```

**Sample Output:**

```
📊 Experiment Status
🔬 Current Experiments: 3
   Files: 45, Size: 2.1 MB
   1. 20250811_143022 (2025-08-11 14:30) - 15 files
   2. 20250811_121045 (2025-08-11 12:10) - 18 files
   3. 20250811_093312 (2025-08-11 09:33) - 12 files

📦 Archived Experiments: 12
   Files: 156, Size: 8.3 MB
```

### Archive All Current Experiments

```bash
python experiments/llm_pure/cleanup_results.py --archive
```

### Archive Specific Experiments (Interactive)

```bash
python experiments/llm_pure/cleanup_results.py --archive
# Select specific experiments from the numbered list
```

### Clean Archive Directory

```bash
# Preview what would be deleted
python experiments/llm_pure/cleanup_results.py --clean-archive 30 --dry-run

# Delete archived experiments older than 30 days
python experiments/llm_pure/cleanup_results.py --clean-archive 30
```

## Interactive Mode

```bash
python experiments/llm_pure/cleanup_results.py
```

**Menu Options:**

1. **Archive current experiments** - Move to `archive/` directory
2. **Clean archive** - Delete old archived experiments
3. **Exit**

## File Structure

### Current Results

```
results/llm_pure/
├── full_evaluation_20250811_143022.json        # Main reports
├── batch_ansible_20250811_143022.json          # Per-technology results
├── raw_responses/                               # Individual responses
│   ├── file1.pp_20250811_143022.json
│   └── file2.pp_20250811_143022.json
├── prompts/                                     # Constructed prompts
│   ├── prompt_definition_based_file1.pp_20250811_143022.txt
│   └── prompt_static_analysis_rules_file2.pp_20250811_143022.txt
└── evaluations/                                 # Detailed evaluations
    └── eval_model_tech_20250811_143022.json
```

### Archived Results

```
results/llm_pure/archive/
└── 20250811_143022/                            # Complete experiment
    ├── full_evaluation_20250811_143022.json
    ├── batch_ansible_20250811_143022.json
    ├── raw_responses/
    │   ├── file1.pp_20250811_143022.json
    │   └── file2.pp_20250811_143022.json
    ├── prompts/
    │   ├── prompt_definition_based_file1.pp_20250811_143022.txt
    │   └── prompt_static_analysis_rules_file2.pp_20250811_143022.txt
    └── evaluations/
        └── eval_model_tech_20250811_143022.json
```

## Workflow Examples

### Development Workflow

```bash
# Run experiments
python experiments/llm_pure/run_evaluation.py --small-batch

# Check results
python experiments/llm_pure/cleanup_results.py --status

# Archive when satisfied with results
python experiments/llm_pure/cleanup_results.py --archive

# Continue with next experiment...
```

### Periodic Maintenance

```bash
# Clean old archives monthly
python experiments/llm_pure/cleanup_results.py --clean-archive 30

# Clean old archives weekly (more aggressive)
python experiments/llm_pure/cleanup_results.py --clean-archive 7
```

### Safe Cleanup Pattern

```bash
# Always preview first
python experiments/llm_pure/cleanup_results.py --clean-archive 30 --dry-run

# Review the list of experiments to be deleted
# Then execute if satisfied
python experiments/llm_pure/cleanup_results.py --clean-archive 30
```

## Best Practices

### Recommended Workflow

1. **Run experiments** and iterate until satisfied
2. **Archive completed experiments** to preserve them
3. **Periodically clean archive** to manage disk space
4. **Always use `--dry-run`** to preview deletions

### Archive Management

- **Archive frequently** - Keeps current results directory clean
- **Use descriptive experiment names** - Easy to identify in archive
- **Clean archive monthly** - Prevents excessive disk usage
- **Preview before deleting** - Always use `--dry-run` first

### Safety Tips

- Archived experiments preserve complete directory structure
- Can be manually moved back from `archive/` if needed
- Unified timestamps ensure no files are missed during operations
- Operations are atomic - either all files move or none do

## Troubleshooting

| Issue                         | Solution                                      |
| ----------------------------- | --------------------------------------------- |
| **No experiments to archive** | Run some experiments first                    |
| **Permission errors**         | Check file permissions: `chmod 755 results/`  |
| **Disk space warnings**       | Use `--clean-archive` to free space           |
| **Accidental deletion**       | Check `results/llm_pure/archive/` for backups |

## Migration from Old Cleanup

The new simplified cleanup is much more reliable because:

- ✅ **Unified timestamps** - No more time proximity guessing
- ✅ **Simpler logic** - 60% fewer lines of code
- ✅ **Focused workflow** - Only archive and clean operations
- ✅ **100% reliable** - No missed files or incorrect grouping

Old complex features (duplicates, custom cleanup, etc.) were removed to focus on the actual workflow.
