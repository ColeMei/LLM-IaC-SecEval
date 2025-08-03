# Results Management Guide

Documentation for managing experiment results using `scripts/cleanup_results.py`.

## Overview

The cleanup script manages **complete experiments** including all associated files across subdirectories. Each experiment consists of main reports plus detailed files (raw responses, prompts, evaluations) that are grouped and managed together.

## Quick Start

```bash
# Analyze current state
python scripts/cleanup_results.py --analyze

# Interactive cleanup (recommended)
python scripts/cleanup_results.py

# Makefile shortcuts
make clean                # Interactive cleanup
make clean-analyze        # Analysis only
make clean-keep-3         # Keep latest 3 experiments
make clean-dry-run        # Preview actions
```

## Command Options

| Option                     | Description                          | Example                     |
| -------------------------- | ------------------------------------ | --------------------------- |
| `--analyze`                | Show directory analysis only         | `--analyze`                 |
| `--archive EXP_ID ...`     | Archive complete experiments         | `--archive 20250803_145459` |
| `--delete-older-than DAYS` | Delete experiments older than N days | `--delete-older-than 7`     |
| `--remove-duplicates`      | Remove duplicate files               | `--remove-duplicates`       |
| `--keep-latest N`          | Keep only latest N experiments       | `--keep-latest 5`           |
| `--dry-run`                | Preview actions without executing    | `--dry-run`                 |

## Interactive Mode

```bash
python scripts/cleanup_results.py
```

**Options:**

1. **Archive experiments** - Move complete experiments to `results/archive/` with preserved structure
2. **Delete old experiments** - Remove all files for experiments older than X days
3. **Remove duplicates** - Intelligent duplicate detection across all file types
4. **Custom cleanup** - Fine-grained control over specific experiments or file types

## Common Usage

**Archive complete experiments:**

```bash
python scripts/cleanup_results.py --archive 20250803_145459 20250803_142259
# Archives ALL files: main reports + raw responses + prompts + evaluations
```

**Delete old experiments:**

```bash
python scripts/cleanup_results.py --delete-older-than 7 --dry-run  # Preview
python scripts/cleanup_results.py --delete-older-than 7            # Execute
# Removes ALL files for experiments older than 7 days
```

**Keep recent experiments:**

```bash
python scripts/cleanup_results.py --keep-latest 5 --dry-run        # Preview
python scripts/cleanup_results.py --keep-latest 5                  # Execute
# Keeps 5 most recent COMPLETE experiments (all their files)
```

**Remove duplicates:**

```bash
python scripts/cleanup_results.py --remove-duplicates
# Intelligently removes duplicates across all experiment files
```

## File Organization

```
results/
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

**Experiment Grouping:** Files are grouped by timestamp patterns:

- **Main files**: `YYYYMMDD_HHMMSS` (e.g., `20250803_145459`)
- **Detail files**: `HHMMSS` (e.g., `145459`) - matched to main experiment

## Analysis Output

```bash
python scripts/cleanup_results.py --analyze
```

**Sample Output:**

```
📊 Results Directory Analysis
==================================================
📄 Total files: 45
💾 Total size: 2.1 MB

🧪 Experiments (complete with all files):
   20250803_145459 (2025-08-03 14:54:59): 8 files (2 main, 6 detailed)
   20250803_142318 (2025-08-03 14:23:18): 6 files (2 main, 4 detailed)

⚠️  Orphaned files (no matching experiment): 3 files
```

## Best Practices

**Complete experiment workflow:**

```bash
# Analyze current state including all files
make clean-analyze

# Keep recent COMPLETE experiments (all files)
make clean-keep-3

# Remove duplicates across ALL file types
python scripts/cleanup_results.py --remove-duplicates
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

## Troubleshooting

**Permission errors:**

```bash
chmod 755 results results/*
```

**Orphaned files detected:**

- Review orphaned files in analysis output
- Manually clean if they're truly unused
- May indicate interrupted experiments

**Recovery:**

- Check `results/archive/` for complete archived experiments
- Archived experiments preserve full directory structure
- Can be moved back to `results/` manually
