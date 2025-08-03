"""
File processing utilities for IaC datasets
"""
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Generator
from .config import config

class FileProcessor:
    """Handles reading and processing IaC files and ground truth data"""
    
    def __init__(self):
        self.data_dir = config.data_dir
        
    def load_ground_truth(self, iac_tech: str) -> pd.DataFrame:
        """
        Load ground truth annotations for a specific IaC technology
        
        Args:
            iac_tech: IaC technology name (ansible, chef, puppet)
            
        Returns:
            DataFrame with columns: PATH, LINE, CATEGORY, AGREEMENT
        """
        csv_file = self.data_dir / config.oracle_datasets[iac_tech]
        
        if not csv_file.exists():
            raise FileNotFoundError(f"Ground truth file not found: {csv_file}")
            
        return pd.read_csv(csv_file)
    
    def get_iac_files(self, iac_tech: str) -> List[Path]:
        """
        Get list of IaC script files for a technology
        
        Args:
            iac_tech: IaC technology name
            
        Returns:
            List of file paths
        """
        iac_dir = self.data_dir / f"oracle-dataset-{iac_tech}"
        
        if not iac_dir.exists():
            raise FileNotFoundError(f"IaC directory not found: {iac_dir}")
        
        # Get all files in the directory
        files = []
        for file_path in iac_dir.iterdir():
            if file_path.is_file():
                files.append(file_path)
                
        return sorted(files)
    
    def read_iac_file(self, file_path: Path) -> Tuple[str, str]:
        """
        Read an IaC file and return filename and content
        
        Args:
            file_path: Path to the IaC file
            
        Returns:
            Tuple of (filename, file_content)
        """
        try:
            content = file_path.read_text(encoding='utf-8')
            return file_path.name, content
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                content = file_path.read_text(encoding='latin-1')
                return file_path.name, content
            except Exception as e:
                raise ValueError(f"Could not read file {file_path}: {e}")
    
    def get_file_ground_truth(self, filename: str, ground_truth_df: pd.DataFrame) -> List[Tuple[str, int, str]]:
        """
        Get ground truth annotations for a specific file
        
        Args:
            filename: Name of the file
            ground_truth_df: Ground truth DataFrame
            
        Returns:
            List of tuples: (filename, line_number, category)
        """
        file_gt = ground_truth_df[ground_truth_df['PATH'] == filename]
        
        results = []
        for _, row in file_gt.iterrows():
            # Filter by agreement threshold if needed
            if row['AGREEMENT'] >= config.agreement_threshold:
                results.append((filename, int(row['LINE']), row['CATEGORY']))
                
        # If no results, the file has no smells
        if not results:
            results.append((filename, 0, 'none'))
            
        return results
    
    def batch_process_files(self, iac_tech: str, limit: int = None) -> Generator[Dict, None, None]:
        """
        Generator that yields file processing batches
        
        Args:
            iac_tech: IaC technology name
            limit: Maximum number of files to process (None for all)
            
        Yields:
            Dictionary with file info and ground truth
        """
        # Load ground truth
        ground_truth_df = self.load_ground_truth(iac_tech)
        
        # Get file list
        files = self.get_iac_files(iac_tech)
        if limit:
            files = files[:limit]
        
        for file_path in files:
            try:
                filename, content = self.read_iac_file(file_path)
                ground_truth = self.get_file_ground_truth(filename, ground_truth_df)
                
                yield {
                    'iac_tech': iac_tech,
                    'filename': filename,
                    'file_path': str(file_path),
                    'content': content,
                    'ground_truth': ground_truth,
                    'file_size': len(content),
                    'line_count': len(content.split('\n'))
                }
                
            except Exception as e:
                print(f"Warning: Could not process file {file_path}: {e}")
                continue
    
    def get_dataset_stats(self) -> Dict[str, Dict]:
        """Get statistics about the datasets"""
        stats = {}
        
        for iac_tech in config.iac_technologies:
            try:
                files = self.get_iac_files(iac_tech)
                ground_truth_df = self.load_ground_truth(iac_tech)
                
                # Count smells by category
                smell_counts = ground_truth_df[
                    ground_truth_df['AGREEMENT'] >= config.agreement_threshold
                ]['CATEGORY'].value_counts().to_dict()
                
                stats[iac_tech] = {
                    'file_count': len(files),
                    'total_annotations': len(ground_truth_df),
                    'smell_counts': smell_counts,
                    'files_with_smells': len(ground_truth_df[ground_truth_df['CATEGORY'] != 'none']['PATH'].unique()),
                    'files_without_smells': len(ground_truth_df[ground_truth_df['CATEGORY'] == 'none']['PATH'].unique())
                }
                
            except Exception as e:
                print(f"Warning: Could not get stats for {iac_tech}: {e}")
                stats[iac_tech] = {'error': str(e)}
                
        return stats