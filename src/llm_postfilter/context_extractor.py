"""
Code Context Extractor for LLM Post-Filtering

This module extracts code context around GLITCH detections for LLM evaluation.
It reads the actual IaC files and provides surrounding lines for semantic analysis.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CodeContextExtractor:
    """Extracts code context around security smell detections for LLM analysis."""
    
    def __init__(self, project_root: Path, data_dir: Optional[Path] = None):
        self.project_root = Path(project_root)
        if data_dir is None:
            # IaC files are typically in a subdirectory, we'll need to search for them
            self.search_paths = [
                self.project_root / "data",
                self.project_root / "iac_files", 
                self.project_root / "cookbooks",
                self.project_root / "manifests",
                self.project_root / "playbooks"
            ]
        else:
            self.search_paths = [Path(data_dir)]
    
    def find_file(self, file_path: str) -> Optional[Path]:
        """Find the actual file location for a given file path from detection data."""
        # The file_path from GLITCH might be just filename or relative path
        filename = Path(file_path).name
        
        # Search in all possible directories
        for search_dir in self.search_paths:
            if search_dir.exists():
                # Direct match
                direct_path = search_dir / file_path
                if direct_path.exists():
                    return direct_path
                
                # Search recursively for filename
                for found_file in search_dir.rglob(filename):
                    if found_file.is_file():
                        return found_file
        
        logger.warning(f"File not found: {file_path}")
        return None
    
    def extract_lines_around(self, file_path: Path, target_line: int, context_lines: int = 3) -> Dict:
        """Extract target line and surrounding context from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            total_lines = len(lines)
            target_idx = target_line - 1  # Convert to 0-based index
            
            # Calculate context window
            start_idx = max(0, target_idx - context_lines)
            end_idx = min(total_lines, target_idx + context_lines + 1)
            
            # Extract lines with line numbers
            context_lines_data = []
            for i in range(start_idx, end_idx):
                line_num = i + 1
                line_content = lines[i].rstrip('\n\r')
                is_target = (line_num == target_line)
                context_lines_data.append({
                    'line_number': line_num,
                    'content': line_content,
                    'is_target': is_target
                })
            
            # Find the target line content
            target_content = ""
            if 0 <= target_idx < total_lines:
                target_content = lines[target_idx].strip()
            
            return {
                'file_path': str(file_path),
                'target_line': target_line,
                'target_content': target_content,
                'total_lines': total_lines,
                'context_window': context_lines_data,
                'context_start': start_idx + 1,
                'context_end': end_idx,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
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
        """Format extracted context as a readable code snippet for LLM."""
        if not context_data['success']:
            return f"# Error reading file: {context_data.get('error', 'Unknown error')}\n"
        
        lines = []
        file_name = Path(context_data['file_path']).name
        lines.append(f"# File: {file_name}")
        lines.append(f"# Target line: {context_data['target_line']}")
        lines.append("")
        
        for line_data in context_data['context_window']:
            line_num = line_data['line_number']
            content = line_data['content']
            is_target = line_data['is_target']
            
            # Mark the target line with an arrow
            prefix = ">>> " if is_target else "    "
            lines.append(f"{prefix}{line_num:3d}: {content}")
        
        return "\n".join(lines)
    
    def extract_context_for_detections(self, detections_df: pd.DataFrame, context_lines: int = 3) -> pd.DataFrame:
        """Extract context for all detections in a DataFrame."""
        enhanced_detections = detections_df.copy()
        
        # Add new columns for context data
        enhanced_detections['file_found'] = False
        enhanced_detections['target_content'] = ""
        enhanced_detections['context_snippet'] = ""
        enhanced_detections['context_success'] = False
        
        for idx, detection in enhanced_detections.iterrows():
            file_path = self.find_file(detection['file_path'])
            
            if file_path:
                enhanced_detections.at[idx, 'file_found'] = True
                context_data = self.extract_lines_around(file_path, detection['line_number'], context_lines)
                
                enhanced_detections.at[idx, 'target_content'] = context_data['target_content']
                enhanced_detections.at[idx, 'context_snippet'] = self.format_context_for_llm(context_data)
                enhanced_detections.at[idx, 'context_success'] = context_data['success']
            else:
                enhanced_detections.at[idx, 'context_snippet'] = f"# File not found: {detection['file_path']}\n"
        
        success_count = enhanced_detections['context_success'].sum()
        total_count = len(enhanced_detections)
        logger.info(f"Successfully extracted context for {success_count}/{total_count} detections")
        
        return enhanced_detections
    
    def load_and_enhance_detections(self, detection_file: Path, context_lines: int = 3) -> pd.DataFrame:
        """Load detection CSV and enhance with code context."""
        detections_df = pd.read_csv(detection_file)
        logger.info(f"Loaded {len(detections_df)} detections from {detection_file.name}")
        
        enhanced_df = self.extract_context_for_detections(detections_df, context_lines)
        return enhanced_df
    
    def save_context_enhanced_detections(self, enhanced_df: pd.DataFrame, output_file: Path) -> None:
        """Save context-enhanced detections to CSV for transparency."""
        enhanced_df.to_csv(output_file, index=False)
        
        success_count = enhanced_df['context_success'].sum()
        total_count = len(enhanced_df)
        logger.info(f"Saved {total_count} detections with context to {output_file}")
        logger.info(f"Context extraction success rate: {success_count}/{total_count} ({success_count/total_count:.1%})")
    
    def process_and_save_detections(self, detection_file: Path, output_dir: Path, context_lines: int = 3) -> pd.DataFrame:
        """Complete pipeline: load, enhance with context, and save enhanced detections."""
        # Load and enhance detections
        enhanced_df = self.load_and_enhance_detections(detection_file, context_lines)
        
        # Create output filename
        base_name = detection_file.stem
        output_file = output_dir / f"{base_name}_with_context.csv"
        
        # Save enhanced detections
        self.save_context_enhanced_detections(enhanced_df, output_file)
        
        return enhanced_df


def main():
    """Test the context extractor."""
    import sys
    
    project_root = Path(__file__).parent.parent.parent
    extractor = CodeContextExtractor(project_root)
    

    data_dir = project_root / "experiments/llm-postfilter/data"
    chef_secrets_file = data_dir / "chef_hard_coded_secret_detections.csv"
    
    if chef_secrets_file.exists():
        print(f"Testing context extraction with {chef_secrets_file}")
        enhanced_df = extractor.load_and_enhance_detections(chef_secrets_file)
        
        # Show example
        if len(enhanced_df) > 0:
            example = enhanced_df.iloc[0]
            print(f"\n🔍 Example context extraction:")
            print(f"Detection: {example['detection_id']}")
            print(f"File found: {example['file_found']}")
            print(f"Success: {example['context_success']}")
            print(f"\nContext snippet:")
            print(example['context_snippet'])
    else:
        print(f"Detection file not found: {chef_secrets_file}")


if __name__ == "__main__":
    main()