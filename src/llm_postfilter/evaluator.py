"""
Hybrid Evaluation Pipeline

This module evaluates and compares the performance of:
1. GLITCH-only detection (baseline)  
2. GLITCH+LLM hybrid detection (post-filtered)

Implements the metrics comparison framework from the experimental roadmap.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class HybridEvaluator:
    """Evaluates GLITCH vs GLITCH+LLM performance across security smells."""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
    
    def calculate_metrics(self, tp: int, fp: int, fn: int) -> Dict[str, float]:
        """Calculate precision, recall, and F1-score from confusion matrix values."""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn
        }
    
    def evaluate_glitch_baseline(self, detections_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate baseline GLITCH performance metrics."""
        # Count TP and FP from GLITCH detections
        tp = detections_df['is_true_positive'].sum()
        fp = len(detections_df) - tp
        
        # For FN, we need to know how many true instances GLITCH missed
        # This should be calculated from the original oracle vs GLITCH comparison
        # For now, we'll mark it as unknown and calculate precision only
        fn = 0  # Will need to be provided from baseline analysis
        
        return self.calculate_metrics(tp, fp, fn)
    
    def evaluate_llm_filtered(self, filtered_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate LLM-filtered performance metrics."""
        # Only consider detections that LLM kept
        kept_detections = filtered_df[filtered_df['keep_detection']]
        
        tp = kept_detections['is_true_positive'].sum()
        fp = len(kept_detections) - tp
        
        # Calculate FN: original TP that LLM incorrectly filtered out
        original_tp = filtered_df['is_true_positive'].sum()
        fn = original_tp - tp
        
        return self.calculate_metrics(tp, fp, fn)
    
    def compare_performance(self, baseline_metrics: Dict, llm_metrics: Dict) -> Dict[str, float]:
        """Compare baseline vs LLM-filtered performance."""
        comparison = {}
        
        for metric in ["precision", "recall", "f1_score"]:
            baseline_val = baseline_metrics[metric]
            llm_val = llm_metrics[metric]
            
            # Absolute difference
            comparison[f"delta_{metric}"] = llm_val - baseline_val
            
            # Relative improvement (percentage)
            if baseline_val > 0:
                comparison[f"relative_improvement_{metric}"] = (llm_val - baseline_val) / baseline_val
            else:
                comparison[f"relative_improvement_{metric}"] = float('inf') if llm_val > 0 else 0.0
        
        # Special metrics
        baseline_fp = baseline_metrics["false_positives"]
        llm_fp = llm_metrics["false_positives"]
        
        if baseline_fp > 0:
            comparison["false_positive_reduction"] = (baseline_fp - llm_fp) / baseline_fp
        else:
            comparison["false_positive_reduction"] = 0.0
        
        baseline_tp = baseline_metrics["true_positives"]
        llm_tp = llm_metrics["true_positives"]
        
        if baseline_tp > 0:
            comparison["true_positive_retention"] = llm_tp / baseline_tp
        else:
            comparison["true_positive_retention"] = 1.0
        
        return comparison
    
    def evaluate_smell_category(self, filtered_df: pd.DataFrame, smell_category: str) -> Dict:
        """Evaluate performance for a specific security smell category."""
        # Filter for specific smell
        smell_detections = filtered_df[filtered_df['smell_category'] == smell_category]
        
        if len(smell_detections) == 0:
            logger.warning(f"No detections found for smell category: {smell_category}")
            return {}
        
        # Calculate baseline and LLM metrics
        baseline_metrics = self.evaluate_glitch_baseline(smell_detections)
        llm_metrics = self.evaluate_llm_filtered(smell_detections)
        comparison = self.compare_performance(baseline_metrics, llm_metrics)
        
        return {
            "smell_category": smell_category,
            "detection_count": len(smell_detections),
            "baseline_metrics": baseline_metrics,
            "llm_metrics": llm_metrics,
            "comparison": comparison
        }
    
    def evaluate_iac_tool(self, filtered_detections: List[pd.DataFrame], iac_tool: str) -> Dict:
        """Evaluate performance across all smell categories for an IaC tool."""
        results = {
            "iac_tool": iac_tool,
            "timestamp": datetime.now().isoformat(),
            "smell_results": {},
            "overall_results": {}
        }
        
        # Combine all filtered detections for this IaC tool
        all_detections = pd.concat(filtered_detections, ignore_index=True) if filtered_detections else pd.DataFrame()
        
        if len(all_detections) == 0:
            logger.warning(f"No detections found for {iac_tool}")
            return results
        
        # Evaluate each smell category
        smell_categories = all_detections['smell_category'].unique()
        
        for smell in smell_categories:
            smell_result = self.evaluate_smell_category(all_detections, smell)
            if smell_result:
                results["smell_results"][smell] = smell_result
        
        # Calculate overall metrics (macro-average across smells)
        if results["smell_results"]:
            overall_baseline = self.evaluate_glitch_baseline(all_detections)
            overall_llm = self.evaluate_llm_filtered(all_detections)
            overall_comparison = self.compare_performance(overall_baseline, overall_llm)
            
            results["overall_results"] = {
                "baseline_metrics": overall_baseline,
                "llm_metrics": overall_llm,
                "comparison": overall_comparison
            }
        
        return results
    
    def create_summary_table(self, evaluation_results: Dict) -> pd.DataFrame:
        """Create a summary table of evaluation results."""
        rows = []
        
        for iac_tool, tool_results in evaluation_results.items():
            if "smell_results" not in tool_results:
                continue
                
            for smell, smell_results in tool_results["smell_results"].items():
                baseline = smell_results["baseline_metrics"]
                llm = smell_results["llm_metrics"]
                comparison = smell_results["comparison"]
                
                row = {
                    "IaC_Tool": iac_tool,
                    "Security_Smell": smell,
                    "Detection_Count": smell_results["detection_count"],
                    
                    # Baseline metrics
                    "Baseline_Precision": baseline["precision"],
                    "Baseline_Recall": baseline["recall"],
                    "Baseline_F1": baseline["f1_score"],
                    "Baseline_TP": baseline["true_positives"],
                    "Baseline_FP": baseline["false_positives"],
                    
                    # LLM metrics
                    "LLM_Precision": llm["precision"],
                    "LLM_Recall": llm["recall"],
                    "LLM_F1": llm["f1_score"],
                    "LLM_TP": llm["true_positives"],
                    "LLM_FP": llm["false_positives"],
                    
                    # Improvements
                    "Delta_Precision": comparison["delta_precision"],
                    "Delta_Recall": comparison["delta_recall"],
                    "Delta_F1": comparison["delta_f1_score"],
                    "Precision_Improvement": comparison["relative_improvement_precision"],
                    "FP_Reduction": comparison["false_positive_reduction"],
                    "TP_Retention": comparison["true_positive_retention"]
                }
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def save_evaluation_results(self, evaluation_results: Dict, output_dir: Path):
        """Save evaluation results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed JSON results
        json_file = output_dir / "hybrid_evaluation_results.json"
        with open(json_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        logger.info(f"Saved detailed results to {json_file}")
        
        # Save summary table CSV
        summary_df = self.create_summary_table(evaluation_results)
        csv_file = output_dir / "hybrid_evaluation_summary.csv"
        summary_df.to_csv(csv_file, index=False)
        logger.info(f"Saved summary table to {csv_file}")
        
        # Create performance comparison table
        self._create_performance_comparison_table(summary_df, output_dir)
        
        return summary_df
    
    def _create_performance_comparison_table(self, summary_df: pd.DataFrame, output_dir: Path):
        """Create a formatted performance comparison table."""
        if len(summary_df) == 0:
            return
        
        # Create comparison table
        comparison_rows = []
        
        for _, row in summary_df.iterrows():
            comparison_rows.append({
                "IaC Tool": row["IaC_Tool"],
                "Security Smell": row["Security_Smell"],
                "Detections": row["Detection_Count"],
                "Baseline Precision": f"{row['Baseline_Precision']:.3f}",
                "LLM Precision": f"{row['LLM_Precision']:.3f}",
                "Precision Δ": f"{row['Delta_Precision']:+.3f}",
                "Baseline Recall": f"{row['Baseline_Recall']:.3f}",
                "LLM Recall": f"{row['LLM_Recall']:.3f}",
                "Recall Δ": f"{row['Delta_Recall']:+.3f}",
                "F1 Δ": f"{row['Delta_F1']:+.3f}",
                "FP Reduction": f"{row['FP_Reduction']:.1%}",
                "TP Retention": f"{row['TP_Retention']:.1%}"
            })
        
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_file = output_dir / "performance_comparison_table.csv"
        comparison_df.to_csv(comparison_file, index=False)
        logger.info(f"Saved performance comparison to {comparison_file}")


def main():
    """Test the evaluation pipeline."""
    project_root = Path(__file__).parent.parent.parent
    
    # Load example filtered data
    data_dir = project_root / "experiments/llm-postfilter/data"
    test_files = list(data_dir.glob("*_llm_filtered.csv"))
    
    if test_files:
        evaluator = HybridEvaluator(project_root)
        
        # Test with first filtered file
        test_file = test_files[0]
        filtered_df = pd.read_csv(test_file)
        
        print(f"Testing evaluation with {test_file.name}")
        print(f"Detections: {len(filtered_df)}")
        
        # Get smell category from filename or data
        if len(filtered_df) > 0:
            smell_category = filtered_df['smell_category'].iloc[0]
            result = evaluator.evaluate_smell_category(filtered_df, smell_category)
            
            if result:
                print(f"\nEvaluation Result for {smell_category}:")
                print(f"Baseline Precision: {result['baseline_metrics']['precision']:.3f}")
                print(f"LLM Precision: {result['llm_metrics']['precision']:.3f}")
                print(f"Precision Improvement: {result['comparison']['delta_precision']:+.3f}")
    else:
        print("No filtered detection files found. Run LLM filtering first.")


if __name__ == "__main__":
    main()