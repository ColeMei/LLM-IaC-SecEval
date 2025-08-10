"""
LLM Post-Filter for GLITCH Detections

This module implements the main LLM post-filtering pipeline that combines:
1. Context extraction from IaC files
2. Smell-specific prompt generation  
3. LLM evaluation and decision making
4. Results processing and analysis

Main entry point for the hybrid GLITCH+LLM approach.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import json
from datetime import datetime

try:
    # Relative imports (when used as package)
    from .context_extractor import CodeContextExtractor
    from .prompt_templates import SecuritySmellPrompts, SecuritySmell
    from .llm_client import (
        create_llm_client,
        Provider,
        GPT4OMiniClient,  # backward compatibility alias
        LLMDecision,
        LLMResponse,
    )
except ImportError:
    # Absolute imports (when run directly)
    from context_extractor import CodeContextExtractor
    from prompt_templates import SecuritySmellPrompts, SecuritySmell
    from llm_client import (
        create_llm_client,
        Provider,
        GPT4OMiniClient,  # backward compatibility alias
        LLMDecision,
        LLMResponse,
    )

logger = logging.getLogger(__name__)


class GLITCHLLMFilter:
    """Main LLM post-filter for GLITCH detections."""
    
    def __init__(
        self,
        project_root: Path,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        provider: str = Provider.OPENAI.value,
        base_url: Optional[str] = None,
    ):
        """Initialize the LLM filter with all required components."""
        self.project_root = Path(project_root)
        
        # Initialize components
        self.context_extractor = CodeContextExtractor(project_root)
        # Create a provider-specific client (OpenAI by default)
        self.llm_client = create_llm_client(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        
        # Configuration
        self.context_lines = 3  # ±3 lines around detection
        
        logger.info("Initialized GLITCH+LLM filter pipeline")
    
    def load_detections(self, detection_file: Path) -> pd.DataFrame:
        """Load GLITCH detections from CSV file."""
        if not detection_file.exists():
            raise FileNotFoundError(f"Detection file not found: {detection_file}")
        
        detections_df = pd.read_csv(detection_file)
        logger.info(f"Loaded {len(detections_df)} detections from {detection_file.name}")
        
        return detections_df
    
    def extract_context_for_detections(self, detections_df: pd.DataFrame) -> pd.DataFrame:
        """Extract code context for all detections."""
        logger.info("Extracting code context for detections...")
        enhanced_df = self.context_extractor.extract_context_for_detections(
            detections_df, self.context_lines
        )
        
        success_count = enhanced_df['context_success'].sum()
        total_count = len(enhanced_df)
        success_rate = success_count / total_count if total_count > 0 else 0
        
        logger.info(f"Context extraction: {success_count}/{total_count} successful ({success_rate:.1%})")
        
        return enhanced_df
    
    def generate_prompts(self, enhanced_detections: pd.DataFrame) -> List[Tuple[int, str, str]]:
        """Generate LLM prompts for all detections."""
        prompts = []
        
        for idx, detection in enhanced_detections.iterrows():
            # Skip if context extraction failed
            if not detection['context_success']:
                logger.warning(f"Skipping detection {idx} - context extraction failed")
                continue
            
            # Get security smell
            smell = SecuritySmellPrompts.get_smell_from_string(detection['smell_category'])
            if not smell:
                logger.warning(f"Unknown smell category: {detection['smell_category']}")
                continue
            
            # Generate prompt
            prompt = SecuritySmellPrompts.create_prompt(smell, detection['context_snippet'])
            prompts.append((idx, detection['detection_id'], prompt))
        
        logger.info(f"Generated {len(prompts)} prompts for LLM evaluation")
        return prompts
    
    def evaluate_with_llm(self, prompts: List[Tuple[int, str, str]]) -> Dict[int, LLMResponse]:
        """Evaluate detections using LLM."""
        logger.info(f"Starting LLM evaluation of {len(prompts)} detections...")
        
        # Extract just the prompt strings for batch evaluation
        prompt_strings = [prompt for _, _, prompt in prompts]
        
        # Progress callback
        def progress_callback(current, total, response):
            if current % 5 == 0 or current == total:
                logger.info(f"LLM progress: {current}/{total} ({current/total:.1%})")
        
        # Batch evaluate
        responses = self.llm_client.batch_evaluate(prompt_strings, progress_callback)
        
        # Map responses back to detection indices
        results = {}
        for i, (idx, detection_id, _) in enumerate(prompts):
            results[idx] = responses[i]
        
        # Log statistics
        stats = self.llm_client.get_statistics(responses)
        logger.info(f"LLM evaluation completed:")
        logger.info(f"  YES: {stats['yes_decisions']}, NO: {stats['no_decisions']}")
        logger.info(f"  ERROR: {stats['error_decisions']}, UNCLEAR: {stats['unclear_decisions']}")
        logger.info(f"  Success rate: {stats['success_rate']:.1%}")
        logger.info(f"  Total time: {stats['total_time_seconds']:.1f}s")
        logger.info(f"  Estimated cost: ${stats['estimated_cost_usd']:.4f}")
        
        return results
    
    def apply_llm_filter(self, enhanced_detections: pd.DataFrame, llm_results: Dict[int, LLMResponse]) -> pd.DataFrame:
        """Apply LLM filtering decisions to the detections."""
        filtered_df = enhanced_detections.copy()
        
        # Add LLM result columns
        filtered_df['llm_decision'] = ""
        filtered_df['llm_raw_response'] = ""
        filtered_df['llm_processing_time'] = np.nan
        filtered_df['llm_error'] = ""
        filtered_df['keep_detection'] = False
        
        for idx, llm_response in llm_results.items():
            if idx in filtered_df.index:
                filtered_df.at[idx, 'llm_decision'] = llm_response.decision.value
                filtered_df.at[idx, 'llm_raw_response'] = llm_response.raw_response
                filtered_df.at[idx, 'llm_processing_time'] = llm_response.processing_time or 0
                filtered_df.at[idx, 'llm_error'] = llm_response.error_message or ""
                
                # Keep detection if LLM says YES
                filtered_df.at[idx, 'keep_detection'] = (llm_response.decision == LLMDecision.YES)
        
        # For detections without LLM results (context extraction failed), keep original GLITCH decision
        no_llm_mask = filtered_df['llm_decision'] == ""
        filtered_df.loc[no_llm_mask, 'keep_detection'] = True  # Conservative approach
        filtered_df.loc[no_llm_mask, 'llm_decision'] = "SKIPPED"
        
        kept_count = filtered_df['keep_detection'].sum()
        total_count = len(filtered_df)
        logger.info(f"LLM filtering: keeping {kept_count}/{total_count} detections ({kept_count/total_count:.1%})")
        
        return filtered_df
    
    def filter_detections(self, detection_file: Path, output_dir: Optional[Path] = None) -> pd.DataFrame:
        """Complete pipeline: load, extract context, generate prompts, evaluate, filter."""
        logger.info(f"Starting LLM post-filtering pipeline for {detection_file.name}")
        
        # Step 1: Load detections
        detections_df = self.load_detections(detection_file)
        
        # Step 2: Extract context
        enhanced_detections = self.extract_context_for_detections(detections_df)
        
        # Step 3: Generate prompts
        prompts = self.generate_prompts(enhanced_detections)
        
        # Step 4: LLM evaluation
        llm_results = self.evaluate_with_llm(prompts)
        
        # Step 5: Apply filtering
        filtered_detections = self.apply_llm_filter(enhanced_detections, llm_results)
        
        # Step 6: Save results
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save filtered detections
            base_name = detection_file.stem
            output_file = output_dir / f"{base_name}_llm_filtered.csv"
            filtered_detections.to_csv(output_file, index=False)
            logger.info(f"Saved filtered detections to {output_file}")
            
            # Save LLM analysis summary
            summary_file = output_dir / f"{base_name}_llm_summary.json"
            self._save_analysis_summary(filtered_detections, llm_results, summary_file)
            
            # Save prompts and responses for reproducibility
            prompts_responses_file = output_dir / f"{base_name}_prompts_and_responses.json"
            self._save_prompts_and_responses(prompts, llm_results, enhanced_detections, prompts_responses_file)
        
        return filtered_detections
    
    def _save_analysis_summary(self, filtered_df: pd.DataFrame, llm_results: Dict[int, LLMResponse], summary_file: Path):
        """Save analysis summary as JSON."""
        # Calculate filtering statistics
        total_detections = len(filtered_df)
        kept_detections = filtered_df['keep_detection'].sum()
        
        original_tp = filtered_df['is_true_positive'].sum()
        original_fp = total_detections - original_tp
        
        # After LLM filtering
        kept_df = filtered_df[filtered_df['keep_detection']]
        kept_tp = kept_df['is_true_positive'].sum()
        kept_fp = len(kept_df) - kept_tp
        
        # LLM decision breakdown
        decision_counts = filtered_df['llm_decision'].value_counts().to_dict()
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "model": self.llm_client.model,
            "filtering_results": {
                "total_detections": int(total_detections),
                "kept_detections": int(kept_detections),
                "filtered_out": int(total_detections - kept_detections),
                "retention_rate": float(kept_detections / total_detections) if total_detections > 0 else 0
            },
            "ground_truth_comparison": {
                "original_tp": int(original_tp),
                "original_fp": int(original_fp),
                "kept_tp": int(kept_tp),
                "kept_fp": int(kept_fp),
                "tp_retention": float(kept_tp / original_tp) if original_tp > 0 else 0,
                "fp_reduction": float((original_fp - kept_fp) / original_fp) if original_fp > 0 else 0
            },
            "llm_decisions": decision_counts,
            "performance_metrics": {
                "original_precision": float(original_tp / total_detections) if total_detections > 0 else 0,
                "filtered_precision": float(kept_tp / kept_detections) if kept_detections > 0 else 0
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved analysis summary to {summary_file}")
    
    def _save_prompts_and_responses(
        self, 
        prompts: List[Tuple[int, str, str]], 
        llm_results: Dict[int, LLMResponse], 
        enhanced_detections: pd.DataFrame, 
        output_file: Path
    ):
        """Save prompts and LLM responses for full reproducibility."""
        logger.info(f"Saving prompts and responses to {output_file}")
        
        # Build comprehensive prompt/response log
        prompt_response_log = {
            "timestamp": datetime.now().isoformat(),
            "model": self.llm_client.model,
            "total_prompts": len(prompts),
            "successful_responses": len(llm_results),
            "interactions": []
        }
        
        # Process each prompt and its corresponding response
        for idx, detection_id, prompt in prompts:
            # Get detection metadata
            detection_row = enhanced_detections.loc[idx] if idx in enhanced_detections.index else None
            
            # Get LLM response
            llm_response = llm_results.get(idx)
            
            interaction = {
                "detection_index": int(idx),
                "detection_id": detection_id,
                "detection_metadata": {
                    "file_path": detection_row["file_path"] if detection_row is not None else "",
                    "line_number": int(detection_row["line_number"]) if detection_row is not None and pd.notna(detection_row["line_number"]) else None,
                    "smell_category": detection_row["smell_category"] if detection_row is not None else "",
                    "context_success": bool(detection_row["context_success"]) if detection_row is not None else False,
                    "is_true_positive": bool(detection_row["is_true_positive"]) if detection_row is not None else False
                },
                "prompt": {
                    "full_text": prompt,
                    "character_count": len(prompt),
                    "word_count": len(prompt.split())
                },
                "llm_response": None
            }
            
            # Add LLM response if available
            if llm_response:
                interaction["llm_response"] = {
                    "decision": llm_response.decision.value,
                    "raw_response": llm_response.raw_response,
                    "confidence_score": llm_response.confidence_score,
                    "processing_time_seconds": llm_response.processing_time,
                    "tokens_used": llm_response.tokens_used,
                    "error_message": llm_response.error_message,
                    "response_length": len(llm_response.raw_response) if llm_response.raw_response else 0
                }
            else:
                interaction["llm_response"] = {
                    "status": "no_response",
                    "reason": "context_extraction_failed_or_llm_error"
                }
            
            prompt_response_log["interactions"].append(interaction)
        
        # Save to JSON with pretty formatting for readability
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(prompt_response_log, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(prompt_response_log['interactions'])} prompt/response pairs to {output_file}")


def main():
    """Test the complete LLM filtering pipeline."""
    project_root = Path(__file__).parent.parent.parent
    
    # Test with a small detection file
    data_dir = project_root / "experiments/llm-postfilter/data"
    test_file = data_dir / "chef_suspicious_comment_detections.csv"
    
    if test_file.exists():
        try:
            # Initialize filter (requires OPENAI_API_KEY)
            filter_pipeline = GLITCHLLMFilter(project_root)
            
            # Run filtering pipeline
            results_dir = data_dir / "llm_results"
            filtered_df = filter_pipeline.filter_detections(test_file, results_dir)
            
            print(f"\n🎯 Filtering Results Summary:")
            print(f"Total detections: {len(filtered_df)}")
            print(f"Kept by LLM: {filtered_df['keep_detection'].sum()}")
            print(f"Original TP: {filtered_df['is_true_positive'].sum()}")
            print(f"Kept TP: {filtered_df[filtered_df['keep_detection']]['is_true_positive'].sum()}")
            
        except Exception as e:
            print(f"Test failed: {e}")
            print("Make sure OPENAI_API_KEY is set and files exist")
    else:
        print(f"Test file not found: {test_file}")


if __name__ == "__main__":
    main()