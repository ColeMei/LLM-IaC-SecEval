"""
IaC Detection Extractor

Extracts GLITCH detections with context for LLM pseudo-labeling.
Supports Chef, Ansible, and Puppet IaC technologies.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IaCDetectionExtractor:
    """Extracts GLITCH detections and wraps them with code context."""

    def __init__(self, project_root: Path, iac_tech: str = "chef"):
        self.project_root = project_root
        self.iac_tech = iac_tech.lower()

        # Directory paths
        self.data_dir = project_root / "data"
        self.iac_files_dir = self.data_dir / "iac_filter_training" / f"oracle-dataset-{self.iac_tech}"

        # Security smells to extract
        self.target_smells = [
            'hardcoded-secret',
            'suspicious comment',
            'weak cryptography algorithms',
            'use of http'
        ]

        # Display names for smells
        self.smell_display_names = {
            'hardcoded-secret': 'Hard-coded secret',
            'suspicious comment': 'Suspicious comment',
            'weak cryptography algorithms': 'Use of weak cryptography algorithms',
            'use of http': 'Use of HTTP without SSL/TLS'
        }

        # IaC technology configurations
        self.iac_config = {
            'chef': {
                'file_extensions': ['.rb'],
                'glitch_file': 'GLITCH-chef-oracle.csv'
            },
            'ansible': {
                'file_extensions': ['.yml', '.yaml'],
                'glitch_file': 'GLITCH-ansible-oracle.csv'
            },
            'puppet': {
                'file_extensions': ['.pp'],
                'glitch_file': 'GLITCH-puppet-oracle.csv'
            }
        }
        
    def load_glitch_detections(self) -> pd.DataFrame:
        """Load and filter GLITCH detections for target security smells."""
        glitch_filename = self.iac_config[self.iac_tech]['glitch_file']

        # Use sampled file if available, otherwise use original
        sampled_filename = glitch_filename.replace('.csv', '_sampled.csv')
        sampled_file = self.data_dir / "iac_filter_training" / sampled_filename

        if sampled_file.exists():
            glitch_file = sampled_file
            logger.info(f"Using sampled detections: {sampled_file.name}")
        else:
            glitch_file = self.data_dir / "iac_filter_training" / glitch_filename
            logger.info(f"Using original detections: {glitch_filename}")

        # Load and filter detections
        glitch_df = pd.read_csv(glitch_file)
        logger.info(f"Loaded {len(glitch_df)} detections")

        filtered_df = glitch_df[glitch_df['ERROR'].isin(self.target_smells)].copy()
        logger.info(f"Filtered to {len(filtered_df)} target smell detections")

        return self._process_detections(filtered_df)

    def _process_detections(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process detections DataFrame with IDs and metadata."""
        # Create unique detection IDs
        df['detection_id'] = (
            df['PATH'] + '_' +
            df['LINE'].astype(str) + '_' +
            df['ERROR']
        )

        # Rename and add metadata columns
        df = df.rename(columns={
            'PATH': 'file_path',
            'LINE': 'line_number',
            'ERROR': 'glitch_smell'
        })

        df['smell_category'] = df['glitch_smell'].map(self.smell_display_names)
        df['iac_tool'] = self.iac_tech
        df['glitch_detection'] = True
        df['detection_id_raw'] = df['file_path'] + '_' + df['line_number'].astype(str)

        # Select final columns
        cols = [
            'detection_id', 'iac_tool', 'smell_category', 'glitch_smell',
            'file_path', 'line_number', 'detection_id_raw', 'glitch_detection'
        ]

        return df[cols]
    
    def get_detection_statistics(self, df: pd.DataFrame) -> Dict:
        """Get statistics for each smell category."""
        stats = {}

        for smell in self.target_smells:
            smell_detections = df[df['glitch_smell'] == smell]
            stats[smell] = {
                'count': len(smell_detections),
                'display_name': self.smell_display_names[smell],
                'files_affected': smell_detections['file_path'].nunique()
            }

        return stats

    def find_iac_file(self, file_path: str) -> Optional[Path]:
        """Find IaC file in the dataset directory."""
        filename = Path(file_path).name

        # Try direct path first
        direct_path = self.iac_files_dir / file_path
        if direct_path.exists():
            return direct_path

        # Search by filename
        for found_file in self.iac_files_dir.rglob(filename):
            if found_file.is_file():
                return found_file

        logger.warning(f"File not found: {file_path}")
        return None

    def extract_context(self, file_path: Path, target_line: int, context_lines: int = 5) -> Dict:
        """Extract target line with surrounding context."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        total_lines = len(lines)
        target_idx = target_line - 1
        start_idx = max(0, target_idx - context_lines)
        end_idx = min(total_lines, target_idx + context_lines + 1)

        context_window = []
        for i in range(start_idx, end_idx):
            line_num = i + 1
            context_window.append({
                'line_number': line_num,
                'content': lines[i].rstrip('\n\r'),
                'is_target': (line_num == target_line)
            })

        target_content = lines[target_idx].strip() if 0 <= target_idx < total_lines else ""

        return {
            'file_path': str(file_path),
            'target_line': target_line,
            'target_content': target_content,
            'total_lines': total_lines,
            'context_window': context_window,
            'context_start': start_idx + 1,
            'context_end': end_idx,
            'success': True
        }

    def format_context_for_llm(self, context_data: Dict) -> str:
        """Format context as code snippet for LLM consumption."""
        lines = []
        file_name = Path(context_data['file_path']).name

        lines.append(f"# {self.iac_tech.title()} file: {file_name}")
        lines.append(f"# Target line: {context_data['target_line']}")
        lines.append("")

        for entry in context_data['context_window']:
            prefix = ">>> " if entry['is_target'] else "    "
            lines.append(f"{prefix}{entry['line_number']:3d}: {entry['content']}")

        return "\n".join(lines)
    
    def enhance_with_context(self, df: pd.DataFrame, context_lines: int = 5) -> pd.DataFrame:
        """Add context information to detections DataFrame."""
        enhanced = df.copy()
        enhanced['file_found'] = False
        enhanced['target_content'] = ""
        enhanced['context_snippet'] = ""
        enhanced['context_success'] = False

        for idx, row in enhanced.iterrows():
            file_path = self.find_iac_file(row['file_path'])
            if file_path is None:
                enhanced.at[idx, 'context_snippet'] = f"# File not found: {row['file_path']}\n"
                continue

            enhanced.at[idx, 'file_found'] = True
            context = self.extract_context(file_path, int(row['line_number']), context_lines)
            enhanced.at[idx, 'context_success'] = context['success']
            enhanced.at[idx, 'target_content'] = context['target_content']
            enhanced.at[idx, 'context_snippet'] = self.format_context_for_llm(context)

        successful = enhanced['context_success'].sum()
        logger.info(f"Context extracted: {successful}/{len(enhanced)} detections")
        return enhanced
    

    def save_detections(self, detections: pd.DataFrame, output_dir: Path) -> None:
        """Save detection CSVs per smell type."""
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_rows = []
        for smell in self.target_smells:
            smell_df = detections[detections['glitch_smell'] == smell]
            summary_rows.append({
                'iac_tool': self.iac_tech,
                'smell_category': self.smell_display_names[smell],
                'glitch_smell': smell,
                'total_detections': len(smell_df),
                'files_affected': smell_df['file_path'].nunique()
            })

            smell_filename = f"{self.iac_tech}_{smell.replace(' ', '_').replace('-', '_')}_detections.csv"
            smell_df.to_csv(output_dir / smell_filename, index=False)
            logger.info(f"Saved {len(smell_df)} {smell} detections")

        # Save summary
        pd.DataFrame(summary_rows).to_csv(output_dir / f"{self.iac_tech}_pseudo_label_summary.csv", index=False)

    def save_context_outputs(self, enhanced: pd.DataFrame, output_dir: Path) -> None:
        """Save context-enhanced CSVs per smell type."""
        output_dir.mkdir(parents=True, exist_ok=True)

        for smell in self.target_smells:
            smell_df = enhanced[enhanced['glitch_smell'] == smell]
            filename = f"{self.iac_tech}_{smell.replace(' ', '_').replace('-', '_')}_detections_with_context.csv"
            smell_df.to_csv(output_dir / filename, index=False)

        logger.info(f"Saved context-enhanced CSVs to {output_dir}")
    
    def run(self, output_dir: Optional[Path] = None, context_lines: int = 5) -> Dict[str, Path]:
        """Main extraction pipeline."""
        output_dir = Path(output_dir or self.project_root / "experiments" / "iac_filter_training" / "data" / self.iac_tech)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load and process detections
        detections = self.load_glitch_detections()
        stats = self.get_detection_statistics(detections)

        logger.info("Detection Statistics:")
        for smell, data in stats.items():
            logger.info(f"  {data['display_name']}: {data['count']} detections")

        # Save raw detections
        self.save_detections(detections, output_dir)

        # Add context and save
        enhanced = self.enhance_with_context(detections, context_lines)
        self.save_context_outputs(enhanced, output_dir)

        return {
            'summary_csv': output_dir / f"{self.iac_tech}_pseudo_label_summary.csv"
        }


def main():
    """Extract IaC detections with context."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract IaC detections with context")
    parser.add_argument("--iac-tech", type=str, default="chef", choices=["chef", "ansible", "puppet"],
                       help="IaC technology to process")
    parser.add_argument("--context-lines", type=int, default=5,
                       help="Number of context lines around each detection")

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    extractor = IaCDetectionExtractor(project_root, args.iac_tech)

    logger.info(f"Extracting {args.iac_tech} detections with context")

    outputs = extractor.run(context_lines=args.context_lines)
    logger.info(f"Extraction complete: {outputs}")


if __name__ == "__main__":
    main()
