"""
IaC Dataset Formatter

Converts LLM-filtered results into JSONL format for training.
Supports stratified sampling to maintain GLITCH proportions.
"""

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import pandas as pd
import argparse
import logging

# Constants
TRAIN_VAL_RATIO = 8 / 9  # 8:1 train/val split
CONTEXT_LINES = 5
SUPPORTED_TECHS = ["chef", "ansible", "puppet"]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class IaCDatasetFormatter:
    """Formatter for IaC pseudo-labeled datasets with stratified sampling."""

    def __init__(self, data_dir: Path, results_dir: Path, output_dir: Path,
                 iac_tech: str = "chef", combined: bool = False):
        self.iac_tech = iac_tech.lower()
        self.combined = combined
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)

        # Setup directory structure
        if self.combined:
            self._setup_combined_dirs()
        else:
            self._setup_single_tech_dirs()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _setup_combined_dirs(self):
        """Setup directories for combined multi-technology mode."""
        base_path = Path("experiments/iac_filter_training/data")
        self.tech_dirs = {}
        self.tech_results_dirs = {}
        self.tech_output_dirs = {}

        for tech in SUPPORTED_TECHS:
            self.tech_dirs[tech] = self._resolve_path(self.data_dir, base_path, tech)
            self.tech_results_dirs[tech] = self._resolve_path(self.results_dir, base_path, tech, "llm_results")
            self.tech_output_dirs[tech] = self._resolve_path(self.output_dir, base_path, tech, "formatted_dataset")

    def _setup_single_tech_dirs(self):
        """Setup directories for single technology mode."""
        base_path = Path("experiments/iac_filter_training/data")
        self.data_dir = self._resolve_path(self.data_dir, base_path, self.iac_tech)
        self.results_dir = self._resolve_path(self.results_dir, base_path, self.iac_tech, "llm_results")
        self.output_dir = self._resolve_path(self.output_dir, base_path, self.iac_tech, "formatted_dataset")

    def _resolve_path(self, provided: Path, base: Path, tech: str, subdir: str = "") -> Path:
        """Resolve directory path with fallback to organized structure."""
        if str(provided) == str(base / (subdir or "")):
            return base / tech / (subdir or "")
        return provided / tech / (subdir or "") if subdir else provided / tech

    def _stratified_train_val_split(self, samples: List[Dict], train_size: int, val_size: int) -> Tuple[List[Dict], List[Dict]]:
        """Perform stratified split to maintain GLITCH proportions."""
        groups = self._group_samples_by_tech_smell(samples)
        logger.info(f"Stratified split across {len(groups)} groups:")

        train_samples, val_samples = [], []
        total_train, total_val = 0, 0

        for group_key, group_samples in groups.items():
            group_train, group_val = self._allocate_group_samples(
                group_samples, train_size, val_size, total_train, total_val
            )
            train_samples.extend(group_train)
            val_samples.extend(group_val)

            total_train += len(group_train)
            total_val += len(group_val)

            group_size = len(group_samples)
            logger.info(f"  {group_key}: {group_size} total → {len(group_train)} train, {len(group_val)} val")

        # Handle remaining samples
        train_samples, val_samples = self._handle_remaining_samples(
            samples, train_samples, val_samples, train_size, val_size
        )

        logger.info(f"Stratified split complete: {len(train_samples)} train, {len(val_samples)} val")
        return train_samples, val_samples

    def _group_samples_by_tech_smell(self, samples: List[Dict]) -> Dict[str, List[Dict]]:
        """Group samples by technology and smell type."""
        groups = defaultdict(list)
        for sample in samples:
            tech = self._extract_technology(sample.get('file', ''))
            smell = sample.get('smell', 'unknown')
            group_key = f"{tech}_{smell}"
            groups[group_key].append(sample)
        return groups

    def _extract_technology(self, file_path: str) -> str:
        """Extract technology from file path or use dataset context."""
        # For single technology mode, use the specified technology
        if not self.combined:
            return self.iac_tech

        # For combined mode, try to detect from file path
        file_lower = file_path.lower()
        for tech in SUPPORTED_TECHS:
            if tech in file_lower:
                return tech
        return 'unknown'

    def _allocate_group_samples(self, group_samples: List[Dict], train_size: int, val_size: int,
                               total_train: int, total_val: int) -> Tuple[List[Dict], List[Dict]]:
        """Allocate samples for a single group maintaining proportions."""
        group_size = len(group_samples)
        group_train_size = round(group_size * TRAIN_VAL_RATIO)
        group_val_size = group_size - group_train_size

        # Handle small groups
        if group_size <= 2:
            group_train_size, group_val_size = self._handle_small_group(
                group_size, total_train, total_val, train_size, val_size
            )

        # Final quota adjustments
        group_train_size = max(0, min(group_train_size, train_size - total_train))
        group_val_size = max(0, min(group_val_size, val_size - total_val))

        # Adjust to fit available samples
        if group_train_size + group_val_size > group_size:
            if group_train_size > group_val_size:
                group_train_size = group_size
                group_val_size = 0
            else:
                group_val_size = group_size
                group_train_size = 0

        random.shuffle(group_samples)
        return (group_samples[:group_train_size],
                group_samples[group_train_size:group_train_size + group_val_size])

    def _handle_small_group(self, group_size: int, total_train: int, total_val: int,
                           train_size: int, val_size: int) -> Tuple[int, int]:
        """Handle allocation for small groups (1-2 samples)."""
        if group_size == 1:
            return (1, 0) if total_val >= val_size else (0, 1)
        elif group_size == 2:
            if total_train < train_size and total_val < val_size:
                return (1, 1)
            elif total_train < train_size:
                return (2, 0)
            else:
                return (0, 2)

    def _handle_remaining_samples(self, all_samples: List[Dict], train_samples: List[Dict],
                                val_samples: List[Dict], train_size: int, val_size: int) -> Tuple[List[Dict], List[Dict]]:
        """Handle any samples not allocated during stratified split."""
        remaining = [s for s in all_samples if s not in train_samples and s not in val_samples]
        if not remaining:
            return train_samples, val_samples

        random.shuffle(remaining)
        train_needed = train_size - len(train_samples)
        val_needed = val_size - len(val_samples)

        if train_needed > 0:
            train_samples.extend(remaining[:train_needed])
            remaining = remaining[train_needed:]

        if val_needed > 0 and remaining:
            val_samples.extend(remaining[:val_needed])

        return train_samples, val_samples

    def _extract_detection_span(self, line_number: int) -> List[int]:
        """Get detection span around target line."""
        start = max(1, line_number - CONTEXT_LINES)
        return [start, line_number + CONTEXT_LINES]
    
    def _clean_context_snippet(self, context: str) -> str:
        """Clean context by removing headers and line numbers."""
        if not context:
            return ""
        
        lines = []
        for line in context.split('\n'):
            # Skip headers
            if line.startswith(('# Chef file:', '# Target line:')):
                continue
            # Extract code from target line or numbered lines
            if '>>>' in line:
                lines.append(line.split('>>>', 1)[1].strip())
            elif '     ' in line and ':' in line:
                parts = line.split(':', 1)
                if len(parts) > 1:
                    lines.append(parts[1].strip())
            else:
                lines.append(line.strip())
        
        return '\n'.join(lines)
    
    def _extract_target_content(self, context: str) -> str:
        """Extract target line content without line numbers."""
        if not context:
            return ""
        
        for line in context.split('\n'):
            if '>>>' in line:
                code_part = line.split('>>>', 1)[1].strip()
                return re.sub(r"^\d+:\s*", "", code_part)
        return ""
    
    def _normalize_smell_name(self, smell_category: str) -> str:
        """Normalize smell category names."""
        mapping = {
            'Hard-coded secret': 'hard_coded_secret',
            'Suspicious comment': 'suspicious_comment', 
            'Use of weak cryptography algorithms': 'weak_cryptography_algorithms',
            'Use of HTTP without SSL/TLS': 'use_of_http'
        }
        return mapping.get(smell_category, smell_category.lower().replace(' ', '_'))
    
    def _choose_content(self, row: pd.Series) -> str:
        """Choose content from target_content or extract from context."""
        raw = str(row.get('target_content') or '').strip()
        if raw:
            return re.sub(r"^\d+:\s*", "", raw)
        return self._extract_target_content(str(row.get('context_snippet') or ''))
    
    def format_single_file(self, csv_path: Path, prompts_path: Path, model_name: Optional[str] = None) -> List[Dict]:
        """Format LLM filtered CSV into JSONL samples."""
        df = pd.read_csv(csv_path)
        with open(prompts_path) as f:
            prompts_data = json.load(f)

        # Detect model from summary file if not provided
        if model_name is None:
            model_name = self._detect_model_from_summary(prompts_path)
        
        # Build prompts lookup
        prompts_lookup = {
            interaction['detection_id']: {
                'prompt': interaction['prompt'],
                'raw_response': interaction.get('response', {}).get('raw_response') or
                               interaction.get('response', {}).get('raw', '')
            }
            for interaction in prompts_data.get('interactions', [])
            }
        
        samples = []
        for _, row in df.iterrows():
            detection_id = row['detection_id']
            prompts_info = prompts_lookup.get(detection_id, {})
            
            # Clean file path
            file_path = row['file_path'][5:] if row['file_path'].startswith('epos-') else row['file_path']

            # Determine label
            llm_decision = row.get('llm_decision', '')
            if llm_decision not in ('YES', 'NO'):
                logger.warning(f"Unknown LLM decision '{llm_decision}' for {detection_id}, skipping")
                continue
            
            sample = {
                "smell": self._normalize_smell_name(row['smell_category']),
                "file": file_path,
                "content": self._choose_content(row),
                "line": int(row['line_number']),
                "detection_span": self._extract_detection_span(int(row['line_number'])),
                "with_context": self._clean_context_snippet(row.get('context_snippet', '')),
                "with_prompt": prompts_info.get('prompt', ''),
                "label": "TP" if llm_decision == 'YES' else "FP",
                "source": model_name
            }
            samples.append(sample)
            
        return samples
    
    def format_all_files(self) -> Dict[str, List[Dict]]:
        """Format all LLM filtered files into samples."""
        all_samples = {}
        
        if self.combined:
            for tech in SUPPORTED_TECHS:
                logger.info(f"\n=== Processing {tech.upper()} ===")
                self._process_tech_files(tech, all_samples)
        else:
            self._process_tech_files(self.iac_tech, all_samples)

        return all_samples

    def _process_tech_files(self, tech: str, all_samples: Dict[str, List[Dict]]):
        """Process files for a specific technology."""
        results_dir = self.tech_results_dirs[tech] if self.combined else self.results_dir
        pattern = f"{tech}_*_llm_filtered.csv"

        for csv_path in results_dir.glob(pattern):
            smell_name = csv_path.stem.replace('_llm_filtered', '')
            prompts_path = csv_path.parent / f"{smell_name}_prompts_and_responses.json"
            
            if not prompts_path.exists():
                logger.warning(f"No prompts file found for {csv_path.name}")
                continue
            
            logger.info(f"Formatting {csv_path.name}")
            samples = self.format_single_file(csv_path, prompts_path)

            # Use tech prefix in combined mode
            key = f"{tech}_{smell_name}" if self.combined else smell_name
            all_samples[key] = samples
            
            tp_count = sum(1 for s in samples if s['label'] == 'TP')
            fp_count = sum(1 for s in samples if s['label'] == 'FP')
            logger.info(f"Generated {len(samples)} samples for {key} (TP: {tp_count}, FP: {fp_count})")
    
    def save_jsonl(self, samples: List[Dict], output_path: Path):
        """Save samples to JSONL file"""
        with open(output_path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
        logger.info(f"Saved {len(samples)} samples to {output_path}")
    
    def save_train_val_split(self, all_samples: Dict[str, List[Dict]],
                           train_size: Optional[int] = None, val_size: Optional[int] = None):
        """Split samples into train/val sets with stratified sampling."""
        # Combine and filter samples
        all_combined = self._combine_and_filter_samples(all_samples)

        # Calculate sizes
        train_size, val_size = self._calculate_split_sizes(all_combined, train_size, val_size)

        # Stratified split
        train_samples, val_samples = self._stratified_train_val_split(all_combined, train_size, val_size)

        # Save files
        train_path, val_path = self._save_split_files(train_samples, val_samples)

        # Save summary
        summary_path = self._save_dataset_summary(all_combined, train_samples, val_samples)

        return train_path, val_path, summary_path

    def _combine_and_filter_samples(self, all_samples: Dict[str, List[Dict]]) -> List[Dict]:
        """Combine all samples and filter out test files."""
        all_combined = [sample for samples in all_samples.values() for sample in samples]

        test_files = self._load_test_files()
        if test_files:
            logger.info(f"Filtering out {len(test_files)} test files to prevent data leakage...")
            original_count = len(all_combined)
            all_combined = [s for s in all_combined if s['file'] not in test_files]
            logger.info(f"Removed {original_count - len(all_combined)} samples from test files")

        tp_count = sum(1 for s in all_combined if s['label'] == 'TP')
        fp_count = sum(1 for s in all_combined if s['label'] == 'FP')
        logger.info(f"Final samples: {len(all_combined)} (TP: {tp_count}, FP: {fp_count})")

        return all_combined

    def _calculate_split_sizes(self, samples: List[Dict], train_size: Optional[int],
                             val_size: Optional[int]) -> Tuple[int, int]:
        """Calculate train/val split sizes using 8:1 ratio."""
        total_samples = len(samples)

        if train_size is None or val_size is None:
            train_size = round(total_samples * TRAIN_VAL_RATIO)
            val_size = total_samples - train_size
            logger.info(f"Auto-calculated sizes: train={train_size}, val={val_size} (8:1 ratio)")
        
        if train_size + val_size > total_samples:
            logger.warning(f"Requested sizes exceed available samples, adjusting...")
            train_size = round(total_samples * TRAIN_VAL_RATIO)
            val_size = total_samples - train_size

        return train_size, val_size

    def _save_split_files(self, train_samples: List[Dict], val_samples: List[Dict]) -> Tuple[Path, Path]:
        """Save train and val samples to JSONL files."""
        if self.combined:
            train_filename = "combined_train.jsonl"
            val_filename = "combined_val.jsonl"
        else:
            train_filename = f"{self.iac_tech}_train.jsonl"
            val_filename = f"{self.iac_tech}_val.jsonl"

        train_path = self.output_dir / train_filename
        val_path = self.output_dir / val_filename

        self.save_jsonl(train_samples, train_path)
        self.save_jsonl(val_samples, val_path)
        
        return train_path, val_path

    def _save_dataset_summary(self, all_samples: List[Dict], train_samples: List[Dict],
                            val_samples: List[Dict]) -> Path:
        """Save dataset summary with statistics."""
        summary = {
            "iac_technology": "combined" if self.combined else self.iac_tech,
            "total_samples": len(all_samples),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "smell_distribution": {},
        }

        if self.combined:
            summary["technologies_included"] = SUPPORTED_TECHS
            summary["technology_distribution"] = {}

        # Count distributions
        for sample in all_samples:
            smell = sample['smell']
            summary["smell_distribution"][smell] = summary["smell_distribution"].get(smell, 0) + 1

            if self.combined:
                tech = self._extract_technology(sample.get('file', ''))
                summary["technology_distribution"][tech] = summary["technology_distribution"].get(tech, 0) + 1

        summary_path = self.output_dir / "dataset_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Create test summaries
        self._create_test_summaries()

        logger.info(f"Dataset summary saved to {summary_path}")
        return summary_path

    def _create_test_summaries(self):
        """Create test summaries for relevant technologies."""
        if self.combined:
            for tech in SUPPORTED_TECHS:
                test_file = self.tech_dirs[tech] / f"{tech}_test.jsonl"
                if test_file.exists():
                    self._create_iac_test_summary(test_file)
        else:
            test_file = self.data_dir / f"{self.iac_tech}_test.jsonl"
        if test_file.exists():
            self._create_iac_test_summary(test_file)

    def _create_iac_test_summary(self, test_file_path: Path):
        """Create test summary for a technology."""
        tp_count = fp_count = 0
        smell_counts = {}

        with open(test_file_path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    label = data.get('label', '')
                    if label == 'TP':
                        tp_count += 1
                    elif label == 'FP':
                        fp_count += 1

                    smell = data.get('smell', '')
                    if smell:
                        smell_counts[smell] = smell_counts.get(smell, 0) + 1

        total_samples = tp_count + fp_count
        precision = round(tp_count / total_samples, 3) if total_samples > 0 else 0

        test_summary = {
            "iac_technology": self.iac_tech,
            "total_test_samples": total_samples,
            "test_tp_samples": tp_count,
            "test_fp_samples": fp_count,
            "test_precision": precision,
            "smell_distribution": smell_counts,
            "dataset_info": f"{self.iac_tech.upper()} test dataset statistics"
        }

        test_summary_path = self.output_dir / "test_summary.json"
        with open(test_summary_path, 'w') as f:
            json.dump(test_summary, f, indent=2)

        logger.info(f"Test summary saved to {test_summary_path}")
        logger.info(f"  - {total_samples} samples (TP: {tp_count}, FP: {fp_count}, Precision: {precision:.1%})")

    def _load_test_files(self) -> Set[str]:
        """Load test file names to exclude from train/val."""
        test_files = set()

        if self.combined:
            for tech in SUPPORTED_TECHS:
                test_file = self.tech_dirs[tech] / f"{tech}_test.jsonl"
                if test_file.exists():
                    self._load_test_files_from_jsonl(test_file, test_files)
            if test_files:
                logger.info(f"Loaded {len(test_files)} test files to exclude")
        else:
            test_file = self.data_dir / f"{self.iac_tech}_test.jsonl"
            if test_file.exists():
                self._load_test_files_from_jsonl(test_file, test_files)
            else:
                logger.info("No test files found, proceeding without filtering")

        return test_files

    def _load_test_files_from_jsonl(self, file_path: Path, test_files: Set[str]):
        """Load test files from a JSONL file."""
        try:
            with open(file_path) as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line.strip())
                        file_path_data = data.get('file', '')
                        if file_path_data:
                            test_files.add(file_path_data)
        except Exception as e:
            logger.warning(f"Could not load test files from {file_path}: {e}")

    def _detect_model_from_summary(self, prompts_path: Path) -> str:
        """Detect model name from summary JSON file in the same directory."""
        # Try to find summary file with same base name
        base_name = prompts_path.stem.replace('_prompts_and_responses', '')
        summary_path = prompts_path.parent / f"{base_name}_llm_summary.json"

        if summary_path.exists():
            try:
                with open(summary_path) as f:
                    summary_data = json.load(f)
                    model = summary_data.get('model')
                    if model:
                        logger.info(f"Detected model '{model}' from {summary_path}")
                        return model
            except Exception as e:
                logger.warning(f"Could not read model from {summary_path}: {e}")

        # Fallback to default if summary file not found or invalid
        logger.warning(f"Could not detect model from summary file, using default 'gpt-4o-mini'")
        return "gpt-4o-mini"


def parse_args():
    parser = argparse.ArgumentParser(description="Format IaC pseudo-labeled data into JSONL")
    parser.add_argument("--combined", action="store_true", help="Create combined dataset from all IaC technologies")
    parser.add_argument("--iac-tech", type=str, default="chef", choices=SUPPORTED_TECHS,
                       help="IaC technology to process (ignored when --combined is used)")
    parser.add_argument("--data-dir", type=str, default="experiments/iac_filter_training/data")
    parser.add_argument("--results-dir", type=str, default="experiments/iac_filter_training/data/llm_results")
    parser.add_argument("--output-dir", type=str, default="experiments/iac_filter_training/data/formatted_dataset")
    parser.add_argument("--train-size", type=int, help="Number of training samples (auto-calculated using 8:1 ratio if not specified)")
    parser.add_argument("--val-size", type=int, help="Number of validation samples (auto-calculated using 8:1 ratio if not specified)")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_name = "Combined" if args.combined else args.iac_tech.title()
    
    formatter = IaCDatasetFormatter(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        iac_tech=args.iac_tech,
        combined=args.combined
    )
    
    all_samples = formatter.format_all_files()
    if not all_samples:
        logger.error("No samples found to format!")
        return
    
    train_path, val_path, summary_path = formatter.save_train_val_split(
        all_samples, args.train_size, args.val_size
    )

    print(f"\n{dataset_name} dataset formatting complete!")
    print(f"Train: {train_path}")
    print(f"Val: {val_path}")
    print(f"Dataset Summary: {summary_path}")

    # Print test summary paths
    if args.combined:
        test_summaries = [
            str(summary_path.parent / f"../{tech}/formatted_dataset/test_summary.json")
            for tech in SUPPORTED_TECHS
            if (summary_path.parent / f"../{tech}/formatted_dataset/test_summary.json").exists()
        ]
        if test_summaries:
            print(f"Test Summaries: {', '.join(test_summaries)}")
    elif (summary_path.parent / "test_summary.json").exists():
        print(f"Test Summary: {summary_path.parent / 'test_summary.json'}")


if __name__ == "__main__":
    main()
