#!/usr/bin/env python3
"""
GLITCH Data Analysis Script

Properly analyzes GLITCH baseline results from CSV files to calculate:
- Ground Truth Occurrence 
- GLITCH Precision/Recall/TP/FP for each IaC technology

Handles different CSV structures and maps GLITCH smell names to our standardized names.
"""

import pandas as pd
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from llm_pure.config import config

class GLITCHDataAnalyzer:
    """Analyze GLITCH performance against ground truth"""
    
    def __init__(self):
        self.data_dir = config.data_dir
        
        # Mapping from GLITCH smell names to our standardized names
        self.smell_name_mappings = {
            # From GLITCH files to our standard names
            "admin-by-default": "Admin by default",
            "empty-password": "Empty password", 
            "hardcoded-secret": "Hard-coded secret",
            "hardcoded-password": "Hard-coded secret",  # Map to Hard-coded secret
            "hardcoded-username": "Hard-coded secret",  # Map to Hard-coded secret
            "missing default case statement": "Missing Default in Case Statement",
            "no-integrity-check": "No integrity check",
            "suspicious comment": "Suspicious comment",
            "improper ip address binding": "Unrestricted IP Address",
            "use of http": "Use of HTTP without SSL/TLS",
            "weak cryptography algorithms": "Use of weak cryptography algorithms",
        }
        
        # Our standardized security smells (in specified order)
        self.standard_smells = [
            "Admin by default",
            "Empty password", 
            "Hard-coded secret",
            "Unrestricted IP Address",
            "Suspicious comment",
            "Use of HTTP without SSL/TLS",
            "No integrity check",
            "Use of weak cryptography algorithms",
            "Missing Default in Case Statement"
        ]
    
    def normalize_smell_name(self, glitch_name: str) -> str:
        """Convert GLITCH smell name to our standard name"""
        return self.smell_name_mappings.get(glitch_name.lower(), glitch_name)
    
    def load_ground_truth(self, iac_tech: str) -> List[Tuple[str, int, str]]:
        """Load ground truth annotations from oracle-dataset-*.csv"""
        file_path = self.data_dir / f"oracle-dataset-{iac_tech}.csv"
        
        if not file_path.exists():
            print(f"❌ Ground truth file not found: {file_path}")
            return []
        
        # Read ground truth CSV - structure: PATH,LINE,CATEGORY,AGREEMENT
        df = pd.read_csv(file_path)
        df = df.dropna()
        
        ground_truth = []
        for _, row in df.iterrows():
            # Include all entries regardless of agreement level
            if row['CATEGORY'] != 'none':  # Only include actual smells
                ground_truth.append((
                    row['PATH'], 
                    int(row['LINE']), 
                    row['CATEGORY']
                ))
        
        return ground_truth

    def load_glitch_predictions_ansible(self) -> List[Tuple[str, int, str]]:
        """Load GLITCH predictions for Ansible (now with line numbers!)"""
        file_path = self.data_dir / "GLITCH-ansible-oracle_fixed_improved.csv"
        
        if not file_path.exists():
            print(f"❌ GLITCH Ansible fixed file not found: {file_path}")
            return []
        
        # Read GLITCH predictions - structure: PATH,LINE,ERROR (same as chef/puppet)
        df = pd.read_csv(file_path)
        df = df.dropna()
        
        predictions = []
        for _, row in df.iterrows():
            filename = row['PATH']
            line = int(row['LINE'])
            glitch_smell = row['ERROR'].strip()
            standard_smell = self.normalize_smell_name(glitch_smell)
            
            predictions.append((filename, line, standard_smell))
        
        return predictions

    def load_ground_truth_ansible_file_level(self, iac_tech: str) -> List[Tuple[str, int, str]]:
        """Load ground truth for Ansible converted to file-level (line 0)"""
        file_path = self.data_dir / f"oracle-dataset-{iac_tech}.csv"
        
        if not file_path.exists():
            print(f"❌ Ground truth file not found: {file_path}")
            return []
        
        # Read ground truth CSV - structure: PATH,LINE,CATEGORY,AGREEMENT
        df = pd.read_csv(file_path)
        df = df.dropna()
        
        # For Ansible, aggregate to file-level: if any line in a file has a smell, the file has that smell
        file_smells = set()
        for _, row in df.iterrows():
            if row['CATEGORY'] != 'none':  # Only include actual smells (removed agreement threshold)
                file_smells.add((row['PATH'], 0, row['CATEGORY']))  # Convert to file-level (line 0)
        
        return list(file_smells)

    def load_glitch_predictions_chef_puppet(self, iac_tech: str) -> List[Tuple[str, int, str]]:
        """Load GLITCH predictions for Chef/Puppet (PATH,LINE,ERROR structure)"""
        file_path = self.data_dir / f"GLITCH-{iac_tech}-oracle.csv"
        
        if not file_path.exists():
            print(f"❌ GLITCH {iac_tech} file not found: {file_path}")
            return []
        
        # Read GLITCH predictions - structure: PATH,LINE,ERROR
        df = pd.read_csv(file_path)
        df = df.dropna()
        
        predictions = []
        for _, row in df.iterrows():
            filename = row['PATH']
            line = int(row['LINE'])
            glitch_smell = row['ERROR'].strip()
            standard_smell = self.normalize_smell_name(glitch_smell)
            
            predictions.append((filename, line, standard_smell))
        
        return predictions

    def calculate_glitch_metrics(self, ground_truth: List[Tuple], glitch_predictions: List[Tuple]) -> Dict[str, Dict]:
        """Calculate GLITCH performance metrics against ground truth"""
        
        # Convert to sets for comparison
        gt_set = set(ground_truth)
        pred_set = set(glitch_predictions)
        
        # Calculate per-smell metrics
        smell_metrics = {}
        
        for smell in self.standard_smells:
            # Filter by smell category
            gt_smell = {(f, l, c) for f, l, c in gt_set if c == smell}
            pred_smell = {(f, l, c) for f, l, c in pred_set if c == smell}
            
            # Calculate metrics
            tp = len(pred_smell & gt_smell)
            fp = len(pred_smell - gt_smell)
            fn = len(gt_smell - pred_smell)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            smell_metrics[smell] = {
                'ground_truth_occurrence': len(gt_smell),
                'glitch_precision': precision,
                'glitch_recall': recall,
                'glitch_tp': tp,
                'glitch_fp': fp,
                'glitch_fn': fn,
                'glitch_total_detected': len(pred_smell)
            }
        
        return smell_metrics

    def analyze_technology(self, iac_tech: str) -> Dict[str, Dict]:
        """Analyze GLITCH performance for a specific technology"""
        print(f"\n📊 Analyzing {iac_tech.title()} - GLITCH vs Ground Truth")
        
        # Load ground truth (line-level for all technologies now!)
        ground_truth = self.load_ground_truth(iac_tech)
        evaluation_level = "line-level"
            
        if not ground_truth:
            return {}
        
        print(f"✓ Loaded {len(ground_truth)} ground truth annotations ({evaluation_level})")
        
        # Load GLITCH predictions (all use line-level now)
        if iac_tech == 'ansible':
            glitch_predictions = self.load_glitch_predictions_ansible()
        else:
            glitch_predictions = self.load_glitch_predictions_chef_puppet(iac_tech)
        
        if not glitch_predictions:
            return {}
        
        print(f"✓ Loaded {len(glitch_predictions)} GLITCH predictions ({evaluation_level})")
        
        # Calculate metrics
        metrics = self.calculate_glitch_metrics(ground_truth, glitch_predictions)
        
        # Print summary
        total_gt = sum(m['ground_truth_occurrence'] for m in metrics.values())
        total_tp = sum(m['glitch_tp'] for m in metrics.values())
        total_fp = sum(m['glitch_fp'] for m in metrics.values())
        total_detected = sum(m['glitch_total_detected'] for m in metrics.values())
        
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / total_gt if total_gt > 0 else 0.0
        
        print(f"📈 GLITCH Performance Summary ({evaluation_level}):")
        print(f"   Ground Truth: {total_gt} smells")
        print(f"   GLITCH Detected: {total_detected} smells")
        print(f"   True Positives: {total_tp}")
        print(f"   False Positives: {total_fp}")
        print(f"   Overall Precision: {overall_precision:.3f}")
        print(f"   Overall Recall: {overall_recall:.3f}")
        
        return metrics
    
    def analyze_all_technologies(self) -> Dict[str, Dict[str, Dict]]:
        """Analyze GLITCH performance against ground truth for all technologies"""
        all_results = {}
        
        print("🔍 Evaluating GLITCH against Ground Truth...")
        
        # Analyze each technology
        for iac_tech in ['ansible', 'chef', 'puppet']:
            results = self.analyze_technology(iac_tech)
            if results:
                all_results[iac_tech] = results
        
        return all_results
    
    def create_glitch_summary_table(self, all_results: Dict[str, Dict[str, Dict]]) -> pd.DataFrame:
        """Create summary table: IaC_Technology, Security_Smell, Ground_Truth_Occurrence, GLITCH_Precision, GLITCH_Recall, GLITCH_TP, GLITCH_FP, GLITCH_FN"""
        
        table_data = []
        
        for iac_tech, tech_results in all_results.items():
            # Include ALL standard smells, even those with 0 occurrences
            for smell in self.standard_smells:
                stats = tech_results.get(smell, {
                    'ground_truth_occurrence': 0,
                    'glitch_total_detected': 0,
                    'glitch_precision': 0.0,
                    'glitch_recall': 0.0,
                    'glitch_tp': 0,
                    'glitch_fp': 0,
                    'glitch_fn': 0
                })
                
                table_data.append({
                    'IaC_Technology': iac_tech,
                    'Security_Smell': smell,
                    'Ground_Truth_Occurrence': stats['ground_truth_occurrence'],
                    'GLITCH_Detected': stats['glitch_total_detected'],
                    'GLITCH_Precision': f"{stats['glitch_precision']:.3f}",
                    'GLITCH_Recall': f"{stats['glitch_recall']:.3f}",
                    'GLITCH_TP': stats['glitch_tp'],
                    'GLITCH_FP': stats['glitch_fp'],
                    'GLITCH_FN': stats['glitch_fn']
                })
        
        # Add totals for each technology
        for iac_tech in all_results.keys():
            tech_results = all_results[iac_tech]
            total_occurrence = sum(stats['ground_truth_occurrence'] for stats in tech_results.values())
            total_detected = sum(stats['glitch_total_detected'] for stats in tech_results.values())
            total_tp = sum(stats['glitch_tp'] for stats in tech_results.values())
            total_fp = sum(stats['glitch_fp'] for stats in tech_results.values())
            total_fn = sum(stats['glitch_fn'] for stats in tech_results.values())
            
            table_data.append({
                'IaC_Technology': iac_tech,
                'Security_Smell': 'TOTAL',
                'Ground_Truth_Occurrence': total_occurrence,
                'GLITCH_Detected': total_detected,
                'GLITCH_Precision': f"{total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0:.3f}",
                'GLITCH_Recall': f"{total_tp / total_occurrence if total_occurrence > 0 else 0:.3f}",
                'GLITCH_TP': total_tp,
                'GLITCH_FP': total_fp,
                'GLITCH_FN': total_fn
            })
        
        return pd.DataFrame(table_data)
    
    def print_smell_mappings(self):
        """Print the smell name mappings for verification"""
        print("\n🔄 Security Smell Name Mappings:")
        print("GLITCH Name → Our Standard Name")
        print("-" * 50)
        for glitch_name, standard_name in self.smell_name_mappings.items():
            print(f"{glitch_name:<30} → {standard_name}")
    
    def save_results(self, all_results: Dict, output_file: str = None):
        """Save analysis results to JSON and CSV"""
        if output_file is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"glitch_analysis_{timestamp}"
        
        # Save detailed results as JSON
        import json
        json_file = config.results_dir / f"{output_file}.json"
        with open(json_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"💾 Detailed results saved: {json_file}")
        
        # Save summary table as CSV
        summary_df = self.create_glitch_summary_table(all_results)
        csv_file = config.results_dir / f"{output_file}_summary.csv"
        summary_df.to_csv(csv_file, index=False)
        print(f"📊 Summary table saved: {csv_file}")
        
        return summary_df


def main():
    """Main function"""
    analyzer = GLITCHDataAnalyzer()
    
    # Show mappings
    analyzer.print_smell_mappings()
    
    # Analyze all GLITCH data
    all_results = analyzer.analyze_all_technologies()
    
    if not all_results:
        print("❌ No GLITCH data found!")
        return 1
    
    # Create and display summary table
    print("\n📋 GLITCH Performance Analysis:")
    print("=" * 120)
    summary_df = analyzer.create_glitch_summary_table(all_results)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150)
    pd.set_option('display.max_colwidth', 25)
    print(summary_df.to_string(index=False))
    
    # Save results
    analyzer.save_results(all_results)
    
    print(f"\n✅ GLITCH evaluation complete!")
    print(f"📈 Technologies analyzed: {list(all_results.keys())}")
    print(f"🎯 Now you have REAL GLITCH baseline metrics to compare your LLM results against!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
