"""
Dataset Formatter for Chef Pseudo-Labeled Data

Converts LLM post-filter results into JSONL/HuggingFace format for training.
Maps available data to the specified template structure.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import logging
import re

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ChefDatasetFormatter:
    def __init__(self, data_dir: Path, results_dir: Path, output_dir: Path):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir) 
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _extract_detection_span(self, line_number: int, context_lines: int = 5) -> List[int]:
        """Estimate detection span based on line number and context window"""
        start = max(1, line_number - context_lines)
        end = line_number + context_lines
        return [start, end]
    
    def _clean_context_snippet(self, context: str) -> str:
        """Clean the context snippet to remove file headers and line numbers"""
        if not context:
            return ""
        
        lines = context.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip file header lines
            if line.startswith('# Chef file:') or line.startswith('# Target line:'):
                continue
            # Remove line numbers and arrows, keep actual code
            if '>>>' in line:
                # Extract just the code part after the arrow
                code_part = line.split('>>>', 1)[1].strip()
                cleaned_lines.append(code_part)
            elif line.strip().startswith('     ') and ':' in line:
                # Extract code after line number
                parts = line.split(':', 1)
                if len(parts) > 1:
                    cleaned_lines.append(parts[1].strip())
            else:
                # Keep other lines as-is
                cleaned_lines.append(line.strip())
        
        return '\n'.join(cleaned_lines)
    
    def _extract_target_content(self, context: str) -> str:
        """Extract just the target line content from context, without leading line numbers"""
        if not context:
            return ""
        
        lines = context.split('\n')
        for line in lines:
            if '>>>' in line:
                # Extract the target line (marked with >>>)
                code_part = line.split('>>>', 1)[1].strip()
                # Strip a leading line-number pattern like `123:` if present
                code_part = re.sub(r"^\d+:\s*", "", code_part)
                return code_part
        
        return ""
    
    def _normalize_smell_name(self, smell_category: str) -> str:
        """Convert smell category to normalized format"""
        mapping = {
            'Hard-coded secret': 'hard_coded_secret',
            'Suspicious comment': 'suspicious_comment', 
            'Use of weak cryptography algorithms': 'weak_cryptography_algorithms',
            'Use of HTTP without SSL/TLS': 'use_of_http'
        }
        return mapping.get(smell_category, smell_category.lower().replace(' ', '_'))
    
    def _get_prompt_template_name(self, prompt_style: str) -> str:
        """Get the prompt template name"""
        return f"static-analysis-filtering-{prompt_style}-v1"
    
    def _choose_content(self, row: pd.Series) -> str:
        """Prefer CSV target_content; else extract from context, and strip leading line numbers."""
        raw = str(row.get('target_content') or '').strip()
        if raw:
            return re.sub(r"^\d+:\s*", "", raw)
        # Fallback to parse from context snippet
        extracted = self._extract_target_content(str(row.get('context_snippet') or ''))
        return re.sub(r"^\d+:\s*", "", extracted)
    
    def format_single_file(self, csv_path: Path, prompts_path: Path, model_name: str = "gpt-4o-mini") -> List[Dict]:
        """Format a single LLM filtered CSV file into JSONL format"""
        # Load data
        df = pd.read_csv(csv_path)
        with open(prompts_path, 'r') as f:
            prompts_data = json.load(f)
        
        # Create prompts lookup
        prompts_lookup = {}
        for interaction in prompts_data.get('interactions', []):
            detection_id = interaction['detection_id']
            prompts_lookup[detection_id] = {
                'prompt': interaction['prompt'],
                'raw_response': interaction['response']['raw']
            }
        
        formatted_samples = []
        
        for _, row in df.iterrows():
            detection_id = row['detection_id']
            prompts_info = prompts_lookup.get(detection_id, {})
            
            # Extract file path without epos- prefix
            file_path = row['file_path']
            if file_path.startswith('epos-'):
                file_path = file_path[5:]  # Remove epos- prefix
            
            # Determine label based on LLM decision
            llm_decision = row.get('llm_decision', '')
            if llm_decision == 'YES':
                label = "TP"  # True Positive
            elif llm_decision == 'NO':
                label = "FP"  # False Positive
            else:
                logger.warning(f"Unknown LLM decision '{llm_decision}' for detection {detection_id}, skipping")
                continue
            
            # Format the sample according to template
            sample = {
                "smell": self._normalize_smell_name(row['smell_category']),
                "file": file_path,
                "content": self._choose_content(row),
                "line": int(row['line_number']),
                "detection_span": self._extract_detection_span(int(row['line_number'])),
                "with_context": self._clean_context_snippet(row.get('context_snippet', '')),
                "with_prompt": prompts_info.get('prompt', ''),
                "label": label,
                "confidence": float(row.get('llm_confidence', 0.0)),
                "source": model_name
            }
            
            formatted_samples.append(sample)
        
        return formatted_samples
    
    def format_all_files(self) -> Dict[str, List[Dict]]:
        """Format all LLM filtered files into JSONL format"""
        all_samples = {}
        
        # Find all LLM filtered CSV files
        for csv_path in self.results_dir.glob("*_llm_filtered.csv"):
            smell_name = csv_path.stem.replace('_llm_filtered', '')
            
            # Find corresponding prompts file
            prompts_path = csv_path.parent / f"{smell_name}_prompts_and_responses.json"
            
            if not prompts_path.exists():
                logger.warning(f"No prompts file found for {csv_path.name}")
                continue
            
            logger.info(f"Formatting {csv_path.name}")
            samples = self.format_single_file(csv_path, prompts_path)
            all_samples[smell_name] = samples
            
            tp_count = sum(1 for s in samples if s['label'] == 'TP')
            fp_count = sum(1 for s in samples if s['label'] == 'FP')
            logger.info(f"Generated {len(samples)} samples for {smell_name} (TP: {tp_count}, FP: {fp_count})")
        
        return all_samples
    
    def save_jsonl(self, samples: List[Dict], output_path: Path):
        """Save samples to JSONL file"""
        with open(output_path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
        logger.info(f"Saved {len(samples)} samples to {output_path}")
    
    def save_by_confidence(self, all_samples: Dict[str, List[Dict]], 
                          train_size: Optional[int] = None, val_size: Optional[int] = None):
        """Sort by confidence and split into train/val sets using 8:1 ratio by default"""
        # Combine all samples
        all_combined = []
        for smell_name, samples in all_samples.items():
            for sample in samples:
                all_combined.append(sample)
        
        # Sort by confidence (highest first)
        all_combined.sort(key=lambda x: x['confidence'], reverse=True)
        
        total_samples = len(all_combined)
        tp_count = sum(1 for s in all_combined if s['label'] == 'TP')
        fp_count = sum(1 for s in all_combined if s['label'] == 'FP')
        logger.info(f"Total samples: {total_samples} (TP: {tp_count}, FP: {fp_count})")
        logger.info(f"Confidence range: {min(s['confidence'] for s in all_combined):.3f} - {max(s['confidence'] for s in all_combined):.3f}")
        
        # Auto-calculate sizes using 8:1 ratio if not specified
        if train_size is None or val_size is None:
            # Use 8:1 ratio (8/9 for train, 1/9 for val)
            auto_train_size = int(total_samples * 8 / 9)
            auto_val_size = total_samples - auto_train_size
            
            if train_size is None:
                train_size = auto_train_size
            if val_size is None:
                val_size = auto_val_size
            
            logger.info(f"Auto-calculated sizes: train={train_size}, val={val_size} (8:1 ratio)")
        
        # Ensure we don't exceed available samples
        if train_size + val_size > total_samples:
            logger.warning(f"Requested sizes ({train_size} + {val_size}) exceed available samples ({total_samples})")
            train_size = int(total_samples * 8 / 9)
            val_size = total_samples - train_size
            logger.info(f"Adjusted to: train={train_size}, val={val_size}")
        
        # Split into train/val
        train_samples = all_combined[:train_size]
        val_samples = all_combined[train_size:train_size + val_size]
        
        # Save train set
        train_path = self.output_dir / "chef_train.jsonl"
        self.save_jsonl(train_samples, train_path)
        
        # Save val set  
        val_path = self.output_dir / "chef_val.jsonl"
        self.save_jsonl(val_samples, val_path)
        
        # Save summary
        summary = {
            "total_samples": len(all_combined),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "confidence_stats": {
                "train_mean": sum(s['confidence'] for s in train_samples) / len(train_samples) if train_samples else 0,
                "val_mean": sum(s['confidence'] for s in val_samples) / len(val_samples) if val_samples else 0,
                "overall_mean": sum(s['confidence'] for s in all_combined) / len(all_combined) if all_combined else 0
            },
            "smell_distribution": {}
        }
        
        # Count samples per smell
        for sample in all_combined:
            smell = sample['smell']
            summary["smell_distribution"][smell] = summary["smell_distribution"].get(smell, 0) + 1
        
        summary_path = self.output_dir / "dataset_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Dataset summary saved to {summary_path}")
        return train_path, val_path, summary_path


def parse_args():
    parser = argparse.ArgumentParser(description="Format Chef pseudo-labeled data into JSONL")
    parser.add_argument("--data-dir", type=str, default="experiments/iac_filter_training/data", 
                       help="Input data directory")
    parser.add_argument("--results-dir", type=str, default="experiments/iac_filter_training/data/llm_results",
                       help="LLM results directory") 
    parser.add_argument("--output-dir", type=str, default="experiments/iac_filter_training/data/formatted_dataset",
                       help="Output directory for JSONL files")
    parser.add_argument("--train-size", type=int, default=None, 
                       help="Number of training samples (auto-calculated using 8:1 ratio if not specified)")
    parser.add_argument("--val-size", type=int, default=None, 
                       help="Number of validation samples (auto-calculated using 8:1 ratio if not specified)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    formatter = ChefDatasetFormatter(
        data_dir=args.data_dir,
        results_dir=args.results_dir, 
        output_dir=args.output_dir
    )
    
    # Format all files
    all_samples = formatter.format_all_files()
    
    if not all_samples:
        logger.error("No samples found to format!")
        return
    
    # Save by confidence
    train_path, val_path, summary_path = formatter.save_by_confidence(
        all_samples, args.train_size, args.val_size
    )
    
    print(f"\nDataset formatting complete!")
    print(f"Train: {train_path}")
    print(f"Val: {val_path}") 
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
