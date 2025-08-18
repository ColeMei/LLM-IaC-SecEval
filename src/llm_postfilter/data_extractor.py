"""
Data Extractor for Hybrid LLM-Static Analysis Pipeline

This module extracts static analysis tool detections for LLM post-filtering.
It processes baseline experiment results to prepare data for hybrid evaluation.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GLITCHDetectionExtractor:
    """Extracts GLITCH detections with context for LLM post-filtering."""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data"
        self.target_smells = [
            'Hard-coded secret',
            'Suspicious comment', 
            'Use of weak cryptography algorithms',
            'Use of HTTP without SSL/TLS'
        ]
        self.glitch_smells = [
            'hardcoded-secret',
            'suspicious comment', 
            'weak cryptography algorithms',
            'use of http'
        ]
        self.smell_mapping = dict(zip(self.target_smells, self.glitch_smells))
        
    def load_baseline_data(self, iac_tool: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load oracle and GLITCH datasets for specified IaC tool."""
        oracle_file = self.data_dir / f"oracle-dataset-{iac_tool.lower()}.csv"
        # Handle special Ansible GLITCH filename
        if iac_tool.lower() == "ansible":
            glitch_file = self.data_dir / "GLITCH-ansible-oracle_fixed_improved.csv"
        else:
            glitch_file = self.data_dir / f"GLITCH-{iac_tool.lower()}-oracle.csv"
        
        if not oracle_file.exists() or not glitch_file.exists():
            raise FileNotFoundError(f"Dataset files not found for {iac_tool}")
            
        oracle_df = pd.read_csv(oracle_file)
        glitch_df = pd.read_csv(glitch_file)
        
        logger.info(f"Loaded {iac_tool} data: Oracle={len(oracle_df)}, GLITCH={len(glitch_df)}")
        return oracle_df, glitch_df
    
    def filter_target_smells(self, oracle_df: pd.DataFrame, glitch_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Filter datasets for target security smells only."""
        oracle_filtered = oracle_df[oracle_df['CATEGORY'].isin(self.target_smells)].copy()
        glitch_filtered = glitch_df[glitch_df['ERROR'].isin(self.glitch_smells)].copy()
        
        # Create unique IDs for comparison
        oracle_filtered['ID'] = oracle_filtered['PATH'] + '_' + oracle_filtered['LINE'].astype(str)
        glitch_filtered['ID'] = glitch_filtered['PATH'] + '_' + glitch_filtered['LINE'].astype(str)
        
        return oracle_filtered, glitch_filtered
    
    def classify_detections(self, oracle_df: pd.DataFrame, glitch_df: pd.DataFrame) -> Dict[str, Dict]:
        """Classify GLITCH detections as TP or FP for each smell category."""
        results = {}
        
        for oracle_smell, glitch_smell in self.smell_mapping.items():
            # Get oracle instances for this smell
            oracle_instances = set(oracle_df[oracle_df['CATEGORY'] == oracle_smell]['ID'])
            
            # Get GLITCH instances for this smell
            glitch_instances = glitch_df[glitch_df['ERROR'] == glitch_smell].copy()
            
            # Classify each GLITCH detection
            glitch_instances['is_true_positive'] = glitch_instances['ID'].isin(oracle_instances)
            
            # Separate TP and FP
            true_positives = glitch_instances[glitch_instances['is_true_positive']].copy()
            false_positives = glitch_instances[~glitch_instances['is_true_positive']].copy()
            
            results[oracle_smell] = {
                'glitch_smell': glitch_smell,
                'oracle_instances': oracle_instances,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'total_detections': len(glitch_instances),
                'tp_count': len(true_positives),
                'fp_count': len(false_positives)
            }
            
            logger.info(f"{oracle_smell}: {len(true_positives)} TP, {len(false_positives)} FP")
            
        return results
    
    def extract_detections_for_llm(self, iac_tool: str) -> Dict[str, List[Dict]]:
        """Extract all GLITCH detections with metadata for LLM evaluation."""
        oracle_df, glitch_df = self.load_baseline_data(iac_tool)
        oracle_filtered, glitch_filtered = self.filter_target_smells(oracle_df, glitch_df)
        detection_results = self.classify_detections(oracle_filtered, glitch_filtered)
        
        llm_input_data = {}
        
        for smell_category, data in detection_results.items():
            smell_detections = []
            
            # Process True Positives
            for _, detection in data['true_positives'].iterrows():
                smell_detections.append({
                    'detection_id': f"{iac_tool}_{smell_category}_{detection['ID']}",
                    'iac_tool': iac_tool,
                    'smell_category': smell_category,
                    'glitch_smell': data['glitch_smell'],
                    'file_path': detection['PATH'],
                    'line_number': detection['LINE'],
                    'detection_id_raw': detection['ID'],
                    'is_true_positive': True,
                    'glitch_detection': True
                })
            
            # Process False Positives  
            for _, detection in data['false_positives'].iterrows():
                smell_detections.append({
                    'detection_id': f"{iac_tool}_{smell_category}_{detection['ID']}",
                    'iac_tool': iac_tool,
                    'smell_category': smell_category,
                    'glitch_smell': data['glitch_smell'],
                    'file_path': detection['PATH'],
                    'line_number': detection['LINE'],
                    'detection_id_raw': detection['ID'],
                    'is_true_positive': False,
                    'glitch_detection': True
                })
                
            llm_input_data[smell_category] = smell_detections
            logger.info(f"Extracted {len(smell_detections)} detections for {smell_category}")
            
        return llm_input_data
    
    def save_detections(self, iac_tool: str, output_dir: Optional[Path] = None):
        """Extract and save detections for specified IaC tool."""
        if output_dir is None:
            output_dir = self.project_root / "experiments/llm-postfilter/data"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        detections = self.extract_detections_for_llm(iac_tool)
        
        # Save overall summary
        summary_data = []
        for smell, detection_list in detections.items():
            tp_count = sum(1 for d in detection_list if d['is_true_positive'])
            fp_count = sum(1 for d in detection_list if not d['is_true_positive'])
            summary_data.append({
                'iac_tool': iac_tool,
                'smell_category': smell,
                'total_detections': len(detection_list),
                'true_positives': tp_count,
                'false_positives': fp_count
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = output_dir / f"{iac_tool.lower()}_detection_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        # Save detailed detections for each smell
        for smell, detection_list in detections.items():
            if detection_list:  # Only save if we have detections
                detection_df = pd.DataFrame(detection_list)
                # Sanitize smell for filesystem-safe filenames
                smell_safe = re.sub(r"[^a-z0-9]+", "_", smell.lower()).strip('_')
                detail_file = output_dir / f"{iac_tool.lower()}_{smell_safe}_detections.csv"
                detection_df.to_csv(detail_file, index=False)
                logger.info(f"Saved {len(detection_list)} detections to {detail_file}")
        
        logger.info(f"Saved summary to {summary_file}")
        return detections


def main():
    """Main function to extract detections for Chef, Puppet, and Ansible."""
    # Get project root relative to this file location
    project_root = Path(__file__).parent.parent.parent
    extractor = GLITCHDetectionExtractor(project_root)
    
    logger.info("Starting GLITCH detection extraction for hybrid LLM pipeline")
    
    # Extract for supported IaC tools
    for iac_tool in ['chef', 'puppet', 'ansible']:
        logger.info(f"Processing {iac_tool.upper()} detections...")
        try:
            detections = extractor.save_detections(iac_tool)
            total_detections = sum(len(detection_list) for detection_list in detections.values())
            logger.info(f"Successfully extracted {total_detections} {iac_tool} detections")
        except Exception as e:
            logger.error(f"Failed to process {iac_tool}: {e}")
    
    logger.info("Detection extraction completed!")


if __name__ == "__main__":
    main()