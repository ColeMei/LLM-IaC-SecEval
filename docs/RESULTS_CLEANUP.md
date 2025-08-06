# Results Management Guide

Documentation for managing experiment results for both LLM Pure and LLM Post-Filter methodologies.

## Overview

This guide covers result management for both evaluation methodologies:

1. **LLM Pure**: Terminal-based evaluation with automated cleanup scripts
2. **LLM Post-Filter**: Notebook-based evaluation with manual archiving scripts

Both approaches provide comprehensive result organization and archiving capabilities.

## Quick Start

### LLM Pure (Automated Cleanup)

```bash
# Analyze current state
python experiments/llm_pure/cleanup_results.py --analyze

# Interactive cleanup (recommended)
python experiments/llm_pure/cleanup_results.py
```

### LLM Post-Filter (Manual Archiving)

```bash
# Archive latest results
python experiments/llm_postfilter/archive_results.py

# Clean working directory
python experiments/llm_postfilter/clean_working_dir.py

# Preview actions
python experiments/llm_postfilter/archive_results.py --dry-run
```

## LLM Pure: Automated Cleanup

### Command Options

| Option                     | Description                          | Example                     |
| -------------------------- | ------------------------------------ | --------------------------- |
| `--analyze`                | Show directory analysis only         | `--analyze`                 |
| `--archive EXP_ID ...`     | Archive complete experiments         | `--archive 20250803_145459` |
| `--delete-older-than DAYS` | Delete experiments older than N days | `--delete-older-than 7`     |
| `--remove-duplicates`      | Remove duplicate files               | `--remove-duplicates`       |
| `--keep-latest N`          | Keep only latest N experiments       | `--keep-latest 5`           |
| `--dry-run`                | Preview actions without executing    | `--dry-run`                 |

### Interactive Mode

```bash
python experiments/llm_pure/cleanup_results.py
```

**Options:**

1. **Archive experiments** - Move complete experiments to `results/llm_pure/archive/` with preserved structure
2. **Delete old experiments** - Remove all files for experiments older than X days
3. **Remove duplicates** - Intelligent duplicate detection across all file types
4. **Custom cleanup** - Fine-grained control over specific experiments or file types

### Common Usage

**Archive complete experiments:**

```bash
python experiments/llm_pure/cleanup_results.py --archive 20250803_145459 20250803_142259
# Archives ALL files: main reports + raw responses + prompts + evaluations
```

**Delete old experiments:**

```bash
python experiments/llm_pure/cleanup_results.py --delete-older-than 7 --dry-run  # Preview
python experiments/llm_pure/cleanup_results.py --delete-older-than 7            # Execute
# Removes ALL files for experiments older than 7 days
```

**Keep recent experiments:**

```bash
python experiments/llm_pure/cleanup_results.py --keep-latest 5 --dry-run        # Preview
python experiments/llm_pure/cleanup_results.py --keep-latest 5                  # Execute
# Keeps 5 most recent COMPLETE experiments (all their files)
```

**Remove duplicates:**

```bash
python experiments/llm_pure/cleanup_results.py --remove-duplicates
# Intelligently removes duplicates across all experiment files
```

## LLM Post-Filter: Manual Archiving

### Command Options

| Option               | Description                       | Example                               |
| -------------------- | --------------------------------- | ------------------------------------- |
| `--dry-run`          | Preview actions without executing | `--dry-run`                           |
| `--custom-name NAME` | Custom name for archive           | `--custom-name "baseline_experiment"` |

### Archive Workflow

```bash
# 1. Archive latest results with timestamp
python experiments/llm_postfilter/archive_results.py

# 2. Archive with custom name
python experiments/llm_postfilter/archive_results.py --custom-name "baseline_experiment"

# 3. Clean working directory after archiving
python experiments/llm_postfilter/clean_working_dir.py

# 4. Preview actions before executing
python experiments/llm_postfilter/archive_results.py --dry-run
```

### Archive Features

Each archived experiment includes:

- **Timestamped directory**: `YYYYMMDD_HHMMSS[_custom_name]`
- **README.md**: Human-readable experiment summary
- **experiment_metadata.json**: Machine-readable metadata
- **All result files**: Complete copy of experiment results

## File Organization

### LLM Pure Structure

```
results/llm_pure/
├── full_evaluation_YYYYMMDD_HHMMSS.json     # - Complete reports
├── batch_TECH_YYYYMMDD_HHMMSS.json          # - Per-technology results
├── raw_responses/                           # - Individual LLM responses
│   └── FILENAME_HHMMSS.json                 # - Grouped by experiment timestamp
├── prompts/                                 # - Saved prompts
│   └── prompt_MODE_FILENAME_HHMMSS.txt      # - Grouped by experiment timestamp
├── evaluations/                             # - Detailed analysis
│   └── eval_MODEL_TECH_HHMMSS.json          # - Grouped by experiment timestamp
└── archive/                                 # - Archived experiments
    └── YYYYMMDD_HHMMSS/                     # - Complete experiment structure
        ├── full_evaluation_*.json
        ├── batch_*.json
        ├── raw_responses/
        ├── prompts/
        └── evaluations/
```

**Experiment Grouping:** Files are grouped by time proximity:

- **Main files**: `YYYYMMDD_HHMMSS` (e.g., `20250803_145459`)
- **Detail files**: `HHMMSS` (e.g., `145459`) - grouped within 15-minute window of main experiment

### LLM Post-Filter Structure

```
experiments/llm_postfilter/data/llm_results/  # - Latest/working results
├── chef_hard_coded_secret_llm_filtered.csv
├── puppet_suspicious_comment_llm_summary.json
└── evaluation/

results/llm_postfilter/                       # - Archived results
├── 20250806_143022/                         # - Timestamped experiment
│   ├── README.md                            # - Human summary
│   ├── experiment_metadata.json             # - Machine metadata
│   └── [experiment files...]
├── 20250806_151205_baseline/                # - Custom named experiment
└── evaluation/                              # - Other evaluation results
```

## Analysis Output

### LLM Pure Analysis

```bash
python experiments/llm_pure/cleanup_results.py --analyze
```

**Sample Output:**

```
📊 Results Directory Analysis
==================================================
📄 Total files: 34
💾 Total size: 0.4 MB

🧪 Experiments (complete with all files):
   20250803_204547 (2025-08-03 20:45:47): 34 files (4 main, 30 detailed) [204547-205208]

⚠️  Orphaned files (no matching experiment): 0 files
```

The time range `[204547-205208]` shows the span of timestamps for files in the experiment.

### LLM Post-Filter Analysis

```bash
python experiments/llm_postfilter/archive_results.py --dry-run
```

**Sample Output:**

```
🔍 DRY RUN - Would archive:
📁 Source: experiments/llm_postfilter/data/llm_results
📁 Target: results/llm_postfilter/20250806_143022
📊 Summary:
  - Files: 15
  - Size: 2.3 MB
  - Tools: chef, puppet
  - Smells: Hard Coded Secret, Suspicious Comment
  - Timestamp: 2025-08-06 14:30:22
```

## Best Practices

### LLM Pure Workflow

```bash
# Analyze current state including all files
make clean-analyze

# Keep recent COMPLETE experiments (all files)
make clean-keep-3

# Remove duplicates across ALL file types
python experiments/llm_pure/cleanup_results.py --remove-duplicates
```

**Safe workflow:**

1. Always use `--dry-run` first to preview complete changes
2. Archive before deleting important complete experiments
3. Understand that operations now affect ALL experiment files

**Development cycle:**

```bash
make clean-dry-run              # Preview what would be cleaned
make clean-keep-3               # Clean but preserve recent complete experiments
```

### LLM Post-Filter Workflow

```bash
# Archive results after each experiment
python experiments/llm_postfilter/archive_results.py --custom-name "experiment_description"

# Clean working directory
python experiments/llm_postfilter/clean_working_dir.py

# Always preview before executing
python experiments/llm_postfilter/archive_results.py --dry-run
```

**Recommended workflow:**

1. Run experiment in notebooks
2. Archive results with descriptive name
3. Clean working directory for next experiment
4. Use archived results for analysis and comparison

## Troubleshooting

**Permission errors:**

```bash
chmod 755 results results/*
```

**Orphaned files detected:**

- Rare with time-based grouping
- May indicate very old files or interrupted experiments
- Review and manually clean if needed

**Recovery:**

- Check `results/archive/` for complete archived experiments
- Archived experiments preserve full directory structure
- Can be moved back to `results/` manually
