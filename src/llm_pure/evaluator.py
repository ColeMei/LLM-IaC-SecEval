"""
Evaluation metrics and comparison utilities
"""
import pandas as pd
from typing import List, Tuple, Dict, Any
from collections import defaultdict
import json
from datetime import datetime
from .config import config

class Evaluator:
    """Evaluates LLM performance against ground truth annotations"""
    
    def __init__(self):
        self.security_smells = [
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
    
    def calculate_metrics(self, predictions: List[Tuple], ground_truth: List[Tuple]) -> Dict[str, Any]:
        """
        Calculate precision, recall, F1 for predictions vs ground truth
        
        Args:
            predictions: List of (filename, line_number, category) tuples from LLM
            ground_truth: List of (filename, line_number, category) tuples from annotations
            
        Returns:
            Dictionary with overall and per-category metrics
        """
        # Convert to sets for comparison
        pred_set = set(predictions)
        gt_set = set(ground_truth)
        
        # Remove 'none' entries for smell-based evaluation
        pred_smells = {(f, l, c) for f, l, c in pred_set if c != 'none'}
        gt_smells = {(f, l, c) for f, l, c in gt_set if c != 'none'}
        
        # Calculate overall metrics
        true_positives = len(pred_smells & gt_smells)
        false_positives = len(pred_smells - gt_smells)
        false_negatives = len(gt_smells - pred_smells)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate per-category metrics
        category_metrics = {}
        for smell in self.security_smells:
            pred_cat = {(f, l, c) for f, l, c in pred_smells if c == smell}
            gt_cat = {(f, l, c) for f, l, c in gt_smells if c == smell}
            
            tp_cat = len(pred_cat & gt_cat)
            fp_cat = len(pred_cat - gt_cat)
            fn_cat = len(gt_cat - pred_cat)
            
            prec_cat = tp_cat / (tp_cat + fp_cat) if (tp_cat + fp_cat) > 0 else 0
            rec_cat = tp_cat / (tp_cat + fn_cat) if (tp_cat + fn_cat) > 0 else 0
            f1_cat = 2 * (prec_cat * rec_cat) / (prec_cat + rec_cat) if (prec_cat + rec_cat) > 0 else 0
            
            category_metrics[smell] = {
                'precision': prec_cat,
                'recall': rec_cat,
                'f1': f1_cat,
                'true_positives': tp_cat,
                'false_positives': fp_cat,
                'false_negatives': fn_cat,
                'support': len(gt_cat)
            }
        
        return {
            'overall': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'total_predictions': len(pred_smells),
                'total_ground_truth': len(gt_smells)
            },
            'by_category': category_metrics,
            'summary': {
                'total_files': len({f for f, _, _ in gt_set}),
                'files_with_smells_gt': len({f for f, l, c in gt_set if c != 'none'}),
                'files_with_smells_pred': len({f for f, l, c in pred_set if c != 'none'}),
                'avg_precision': sum(m['precision'] for m in category_metrics.values()) / len(category_metrics),
                'avg_recall': sum(m['recall'] for m in category_metrics.values()) / len(category_metrics),
                'avg_f1': sum(m['f1'] for m in category_metrics.values()) / len(category_metrics)
            }
        }
    
    def analyze_errors(self, predictions: List[Tuple], ground_truth: List[Tuple]) -> Dict[str, List]:
        """
        Analyze prediction errors in detail
        
        Args:
            predictions: List of predictions
            ground_truth: List of ground truth annotations
            
        Returns:
            Dictionary categorizing different types of errors
        """
        pred_set = set(predictions)
        gt_set = set(ground_truth)
        
        # Remove 'none' entries
        pred_smells = {(f, l, c) for f, l, c in pred_set if c != 'none'}
        gt_smells = {(f, l, c) for f, l, c in gt_set if c != 'none'}
        
        false_positives = list(pred_smells - gt_smells)
        false_negatives = list(gt_smells - pred_smells)
        
        # Group by filename for analysis
        fp_by_file = defaultdict(list)
        fn_by_file = defaultdict(list)
        
        for filename, line, category in false_positives:
            fp_by_file[filename].append((line, category))
            
        for filename, line, category in false_negatives:
            fn_by_file[filename].append((line, category))
        
        # Categorize error types
        error_analysis = {
            'false_positives': {
                'total': len(false_positives),
                'by_category': defaultdict(int),
                'by_file': dict(fp_by_file),
                'examples': false_positives[:10]  # First 10 examples
            },
            'false_negatives': {
                'total': len(false_negatives),
                'by_category': defaultdict(int),
                'by_file': dict(fn_by_file),
                'examples': false_negatives[:10]  # First 10 examples
            }
        }
        
        # Count by category
        for _, _, category in false_positives:
            error_analysis['false_positives']['by_category'][category] += 1
            
        for _, _, category in false_negatives:
            error_analysis['false_negatives']['by_category'][category] += 1
        
        return error_analysis
    
    def create_evaluation_report(self, 
                               model_name: str, 
                               iac_tech: str,
                               metrics: Dict[str, Any], 
                               error_analysis: Dict[str, Any],
                               experiment_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create comprehensive evaluation report
        
        Args:
            model_name: Name of the evaluated model
            iac_tech: IaC technology evaluated
            metrics: Calculated metrics
            error_analysis: Error analysis results
            experiment_metadata: Additional experiment details
            
        Returns:
            Complete evaluation report
        """
        report = {
            'experiment_info': {
                'model_name': model_name,
                'iac_technology': iac_tech,
                'timestamp': datetime.now().isoformat(),
                'agreement_threshold': config.agreement_threshold,
                **(experiment_metadata or {})
            },
            'metrics': metrics,
            'error_analysis': error_analysis,
            'performance_summary': {
                'overall_f1': metrics['overall']['f1'],
                'overall_precision': metrics['overall']['precision'],
                'overall_recall': metrics['overall']['recall'],
                'best_category': max(metrics['by_category'].items(), 
                                   key=lambda x: x[1]['f1'])[0] if metrics['by_category'] else None,
                'worst_category': min(metrics['by_category'].items(), 
                                    key=lambda x: x[1]['f1'])[0] if metrics['by_category'] else None,
                'total_errors': error_analysis['false_positives']['total'] + error_analysis['false_negatives']['total']
            }
        }
        
        return report
    
    def save_evaluation(self, report: Dict[str, Any], output_path: str = None, experiment_id: str = None) -> str:
        """
        Save evaluation report to file
        
        Args:
            report: Evaluation report dictionary
            output_path: Optional custom output path
            experiment_id: Experiment ID for consistent naming
            
        Returns:
            Path to saved file
        """
        if output_path is None:
            timestamp = experiment_id or datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = report['experiment_info']['model_name'].replace(':', '_')
            iac_tech = report['experiment_info']['iac_technology']
            filename = f"eval_{model_name}_{iac_tech}_{timestamp}.json"
            output_path = config.results_dir / "evaluations" / filename
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return str(output_path)
    
    def compare_models(self, evaluation_files: List[str]) -> Dict[str, Any]:
        """
        Compare multiple model evaluations
        
        Args:
            evaluation_files: List of paths to evaluation JSON files
            
        Returns:
            Comparison report
        """
        comparisons = []
        
        for eval_file in evaluation_files:
            with open(eval_file, 'r') as f:
                evaluation = json.load(f)
                
            comparisons.append({
                'model_name': evaluation['experiment_info']['model_name'],
                'iac_tech': evaluation['experiment_info']['iac_technology'],
                'f1': evaluation['metrics']['overall']['f1'],
                'precision': evaluation['metrics']['overall']['precision'],
                'recall': evaluation['metrics']['overall']['recall'],
                'total_errors': evaluation['performance_summary']['total_errors'],
                'file_path': eval_file
            })
        
        # Sort by F1 score
        comparisons.sort(key=lambda x: x['f1'], reverse=True)
        
        return {
            'comparison_summary': comparisons,
            'best_model': comparisons[0] if comparisons else None,
            'worst_model': comparisons[-1] if comparisons else None,
            'average_f1': sum(c['f1'] for c in comparisons) / len(comparisons) if comparisons else 0,
            'model_count': len(comparisons)
        }