"""
IaC Pseudo-Label Data Extractor for IaC Filter Training

This module extracts GLITCH detections from IaC datasets for pseudo-labeling.
It processes detections without ground truth labels and prepares them for LLM post-filtering.
Supports Chef, Ansible, and Puppet IaC technologies.
Also wraps each detection with code context from the respective dataset files.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IaCPseudoLabelExtractor:
    """Extracts GLITCH detections from IaC datasets for pseudo-labeling."""

    def __init__(self, project_root: Path, iac_tech: str = "chef"):
        self.project_root = Path(project_root)
        self.iac_tech = iac_tech.lower()
        self.data_dir = self.project_root / "data"
        # Updated to match the new consistent naming: oracle-dataset_1000_{iac_tech}
        self.iac_files_dir = self.data_dir / "iac_filter_training" / f"oracle-dataset_1000_{self.iac_tech}"
        # Use the new organized structure
        self.iac_data_dir = self.data_dir / "iac_filter_training" / self.iac_tech

        # Target security smells for pseudo-labeling (same for all IaC technologies)
        self.target_smells = [
            'hardcoded-secret',
            'suspicious comment',
            'weak cryptography algorithms',
            'use of http'
        ]

        # Mapping for display names
        self.smell_display_names = {
            'hardcoded-secret': 'Hard-coded secret',
            'suspicious comment': 'Suspicious comment',
            'weak cryptography algorithms': 'Use of weak cryptography algorithms',
            'use of http': 'Use of HTTP without SSL/TLS'
        }

        # IaC technology specific configurations
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
        """Load GLITCH detections from the IaC oracle dataset and filter 4 smells."""
        glitch_filename = self.iac_config[self.iac_tech]['glitch_file']
        # First try the organized structure, then fall back to the old location
        glitch_file = self.iac_data_dir / glitch_filename
        if not glitch_file.exists():
            # Fallback to old location for backward compatibility
            glitch_file = self.data_dir / "iac_filter_training" / glitch_filename
        
        if not glitch_file.exists():
            raise FileNotFoundError(f"GLITCH detection file not found: {glitch_file}")
            
        glitch_df = pd.read_csv(glitch_file)
        logger.info(f"Loaded {len(glitch_df)} GLITCH detections")
        
        # Filter for target smells only
        glitch_filtered = glitch_df[glitch_df['ERROR'].isin(self.target_smells)].copy()
        logger.info(f"Filtered to {len(glitch_filtered)} target smell detections")
        
        # Create unique detection IDs
        glitch_filtered['detection_id'] = (
            glitch_filtered['PATH'] + '_' + 
            glitch_filtered['LINE'].astype(str) + '_' + 
            glitch_filtered['ERROR']
        )
        
        # Normalize column names for downstream steps
        glitch_filtered.rename(columns={
            'PATH': 'file_path',
            'LINE': 'line_number',
            'ERROR': 'glitch_smell'
        }, inplace=True)
        glitch_filtered['smell_category'] = glitch_filtered['glitch_smell'].map(self.smell_display_names)
        glitch_filtered['iac_tool'] = 'chef'
        glitch_filtered['glitch_detection'] = True
        glitch_filtered['detection_id_raw'] = glitch_filtered['file_path'] + '_' + glitch_filtered['line_number'].astype(str)
        
        # Reorder columns (keep concise)
        cols = [
            'detection_id', 'iac_tool', 'smell_category', 'glitch_smell',
            'file_path', 'line_number', 'detection_id_raw', 'glitch_detection'
        ]
        glitch_filtered = glitch_filtered[cols]
        
        return glitch_filtered
    
    def get_detection_statistics(self, glitch_df: pd.DataFrame) -> Dict:
        """Get statistics for each smell category."""
        stats = {}
        
        for smell in self.target_smells:
            smell_detections = glitch_df[glitch_df['glitch_smell'] == smell]
            stats[smell] = {
                'count': len(smell_detections),
                'display_name': self.smell_display_names[smell],
                'files_affected': smell_detections['file_path'].nunique()
            }
            
        return stats
    
    # =========================
    # Context extraction helpers
    # =========================
    def find_iac_file(self, file_path: str) -> Optional[Path]:
        """Find the actual IaC file location for a given file path from detection data."""
        if not self.iac_files_dir.exists():
            logger.warning(f"{self.iac_tech.title()} files directory not found: {self.iac_files_dir}")
            return None

        filename = Path(file_path).name

        # Try direct match
        direct_path = self.iac_files_dir / file_path
        if direct_path.exists():
            return direct_path

        # Search by filename
        for found in self.iac_files_dir.rglob(filename):
            if found.is_file():
                return found

        logger.warning(f"{self.iac_tech.title()} file not found: {file_path}")
        return None
    
    def extract_lines_around(self, file_path: Path, target_line: int, context_lines: int = 5) -> Dict:
        """Extract target line and surrounding context from an IaC file (±5 lines by default)."""
        try:
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
        except Exception as e:
            logger.error(f"Error reading Chef file {file_path}: {e}")
            return {
                'file_path': str(file_path),
                'target_line': target_line,
                'target_content': "",
                'total_lines': 0,
                'context_window': [],
                'context_start': 0,
                'context_end': 0,
                'success': False,
                'error': str(e)
            }
    
    def format_context_for_llm(self, context_data: Dict) -> str:
        """Format extracted context as a readable Chef code snippet for LLM."""
        if not context_data.get('success'):
            return f"# Error reading file: {context_data.get('error', 'Unknown error')}\n"
        
        lines: List[str] = []
        file_name = Path(context_data['file_path']).name
        lines.append(f"# Chef file: {file_name}")
        lines.append(f"# Target line: {context_data['target_line']}")
        lines.append("")
        
        for entry in context_data['context_window']:
            prefix = ">>> " if entry['is_target'] else "    "
            lines.append(f"{prefix}{entry['line_number']:3d}: {entry['content']}")
        
        return "\n".join(lines)
    
    def enhance_with_context(self, df: pd.DataFrame, context_lines: int = 5) -> pd.DataFrame:
        """Given detections DataFrame, add context columns by locating files and extracting lines (±5)."""
        enhanced = df.copy()
        enhanced['file_found'] = False
        enhanced['target_content'] = ""
        enhanced['context_snippet'] = ""
        enhanced['context_success'] = False
        
        for idx, row in enhanced.iterrows():
            path = self.find_iac_file(row['file_path'])
            if path is None:
                enhanced.at[idx, 'context_snippet'] = f"# {self.iac_tech.title()} file not found: {row['file_path']}\n"
                continue
            enhanced.at[idx, 'file_found'] = True
            ctx = self.extract_lines_around(path, int(row['line_number']), context_lines)
            enhanced.at[idx, 'context_success'] = ctx.get('success', False)
            enhanced.at[idx, 'target_content'] = ctx.get('target_content', '')
            enhanced.at[idx, 'context_snippet'] = self.format_context_for_llm(ctx)
        
        ok = int(enhanced['context_success'].sum())
        logger.info(f"Successfully extracted context for {ok}/{len(enhanced)} detections")
        return enhanced
    
    # =========================
    # Save helpers
    # =========================
    def save_smell_csvs(self, detections: pd.DataFrame, output_dir: Path) -> None:
        """Save per-smell CSVs and summary (no combined CSV)."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Summary
        summary_rows = []
        for smell in self.target_smells:
            part = detections[detections['glitch_smell'] == smell]
            summary_rows.append({
                'iac_tool': self.iac_tech,
                'smell_category': self.smell_display_names[smell],
                'glitch_smell': smell,
                'total_detections': len(part),
                'files_affected': part['file_path'].nunique()
            })
            smell_safe = re.sub(r"[^a-z0-9]+", "_", smell.lower()).strip('_')
            (output_dir / f"{self.iac_tech}_{smell_safe}_detections.csv").write_text(part.to_csv(index=False))
            logger.info(f"Saved {len(part)} detections to {(output_dir / f'{self.iac_tech}_{smell_safe}_detections.csv')}\n       ")
        pd.DataFrame(summary_rows).to_csv(output_dir / f"{self.iac_tech}_pseudo_label_summary.csv", index=False)
    
    def save_context_outputs(self, enhanced: pd.DataFrame, output_dir: Path) -> None:
        """Save context-enhanced CSV per smell (no combined CSV/JSONL)."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Per-smell with context only
        for smell in self.target_smells:
            part = enhanced[enhanced['glitch_smell'] == smell]
            smell_safe = re.sub(r"[^a-z0-9]+", "_", smell.lower()).strip('_')
            part.to_csv(output_dir / f"{self.iac_tech}_{smell_safe}_detections_with_context.csv", index=False)

        logger.info(f"Saved per-smell context-enhanced CSVs to {output_dir}")
    
    # =========================
    # Orchestration
    # =========================
    def run(self, output_dir: Optional[Path] = None, context_lines: int = 5) -> Dict[str, Path]:
        if output_dir is None:
            # Use the new organized structure
            output_dir = self.project_root / "experiments" / "iac_filter_training" / "data" / self.iac_tech
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Filter detections
        detections = self.load_glitch_detections()
        stats = self.get_detection_statistics(detections)
        logger.info("Detection Statistics:")
        for smell, data in stats.items():
            logger.info(f"  {data['display_name']}: {data['count']} detections in {data['files_affected']} files")
        
        # Save raw detections (per-smell only) and summary
        self.save_smell_csvs(detections, output_dir)
        
        # Step 2: Enhance with context (±5), save per-smell with context
        enhanced = self.enhance_with_context(detections, context_lines=context_lines)
        self.save_context_outputs(enhanced, output_dir)
        
        return {
            'summary_csv': output_dir / f"{self.iac_tech}_pseudo_label_summary.csv"
        }


def main():
    """Main function to extract IaC detections for pseudo-labeling."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract IaC detections for pseudo-labeling")
    parser.add_argument("--iac-tech", type=str, default="chef", choices=["chef", "ansible", "puppet"],
                       help="IaC technology to process")
    parser.add_argument("--context-lines", type=int, default=5,
                       help="Number of context lines around each detection")

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    extractor = IaCPseudoLabelExtractor(project_root, args.iac_tech)

    logger.info(f"Starting {args.iac_tech.title()} pseudo-label detection extraction (with context)")

    try:
        outputs = extractor.run(context_lines=args.context_lines)
        logger.info(f"Outputs written: {outputs}")
    except Exception as e:
        logger.error(f"Failed to extract detections: {e}")
        raise

    logger.info("Detection extraction completed!")


if __name__ == "__main__":
    main()
