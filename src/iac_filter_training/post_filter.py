"""
IaC LLM Post-Filter for IaC Filter Training

Loads context-enhanced detection CSVs and runs LLM evaluation to filter TP/FP.
Supports all providers: OpenAI, Anthropic, Grok (xAI), Ollama, OpenAI-compatible (OpenRouter).
Outputs filtered results and debug logs to llm_results directory.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import json
import pandas as pd
import numpy as np
import argparse
import os
import sys

# Import from existing llm_postfilter package
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.append(str(_project_root))

from src.llm_postfilter.prompt_loader import ExternalPromptLoader, SecuritySmell
from src.llm_postfilter.llm_client import (
    create_llm_client,
    Provider,
    LLMDecision,
    LLMResponse,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class IaCLLMPostFilter:
    """LLM-based post-filter for IaC security detections."""

    def __init__(
        self,
        project_root: Path,
        iac_tech: str = "chef",
        provider: str = Provider.OPENAI.value,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        prompt_template: str = "static_analysis_rules",
        max_samples: Optional[int] = None,
        data_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
    ):
        self.project_root = project_root
        self.iac_tech = iac_tech.lower()
        self.data_dir = Path(data_dir) if data_dir else (project_root / "experiments" / "iac_filter_training" / "data" / self.iac_tech)
        self.results_dir = Path(results_dir) if results_dir else (self.data_dir / "llm_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.prompt_template = prompt_template
        self.prompt_loader = ExternalPromptLoader(prompt_template)
        self.client = create_llm_client(provider=provider, model=model, api_key=api_key, base_url=base_url)
        self.max_samples = max_samples

        logger.info(f"Initialized {self.iac_tech} post-filter: {provider}/{model} | Data: {self.data_dir.name}")

    def _load_detection_file(self, path: Path) -> pd.DataFrame:
        """Load detection CSV and optionally limit samples."""
        df = pd.read_csv(path)
        if self.max_samples and len(df) > self.max_samples:
            df = df.head(self.max_samples).copy()
            logger.info(f"Limited to {self.max_samples} samples for testing")
        logger.info(f"Loaded {len(df)} detections from {path.name}")
        return df

    def _build_prompts(self, df: pd.DataFrame) -> List[Tuple[int, str, str]]:
        """Build evaluation prompts for each detection."""
        prompts = []
        for idx, row in df.iterrows():
            # Skip failed context extractions
            if 'context_success' in df.columns and not row.get('context_success', True):
                continue

            smell = self._smell_from_string(str(row['smell_category']))
            if not smell:
                continue

            iac_tech = row.get('iac_tool', self.iac_tech)
            prompt = self.prompt_loader.create_prompt(
                smell,
                str(row.get('context_snippet', '')),
                str(iac_tech)
            )
            prompts.append((idx, str(row['detection_id']), prompt))

        logger.info(f"Built {len(prompts)} evaluation prompts")
        return prompts

    @staticmethod
    def _smell_from_string(smell_name: str) -> Optional[SecuritySmell]:
        """Convert smell display name to SecuritySmell enum."""
        return next((s for s in SecuritySmell if s.value == smell_name), None)

    def _evaluate_prompts(self, prompts: List[Tuple[int, str, str]]) -> Dict[int, LLMResponse]:
        """Send prompts to LLM and collect responses."""
        prompt_strings = [prompt for _, _, prompt in prompts]
        responses = self.client.batch_evaluate(prompt_strings)
        return dict(zip([idx for idx, _, _ in prompts], responses))

    def _merge_results(self, df: pd.DataFrame, results: Dict[int, LLMResponse]) -> pd.DataFrame:
        """Merge LLM responses back into the detections DataFrame."""
        merged = df.copy()

        # Add LLM result columns
        merged['llm_decision'] = ""
        merged['llm_raw_response'] = ""
        merged['llm_processing_time'] = np.nan
        merged['llm_error'] = ""
        merged['keep_detection'] = False

        # Populate results
        for idx, response in results.items():
            if idx not in merged.index:
                continue
            merged.at[idx, 'llm_decision'] = response.decision.value
            merged.at[idx, 'llm_raw_response'] = response.raw_response
            merged.at[idx, 'llm_processing_time'] = response.processing_time or 0
            merged.at[idx, 'llm_error'] = response.error_message or ""
            merged.at[idx, 'keep_detection'] = (response.decision == LLMDecision.YES)

        # Conservative: keep detections that weren't evaluated
        unevaluated = merged['llm_decision'] == ""
        merged.loc[unevaluated, 'keep_detection'] = True
        merged.loc[unevaluated, 'llm_decision'] = "SKIPPED"

        return merged

    def _save_outputs(self, input_csv: Path, filtered_df: pd.DataFrame,
                     prompts: List[Tuple[int, str, str]], results: Dict[int, LLMResponse]) -> Dict[str, Path]:
        """Save filtered results, summary, and debug logs."""
        base_name = input_csv.stem

        # Save filtered CSV
        filtered_path = self.results_dir / f"{base_name}_llm_filtered.csv"
        filtered_df.to_csv(filtered_path, index=False)

        # Create summary
        total = len(filtered_df)
        kept = filtered_df['keep_detection'].sum()
        summary = {
            "model": self.client.model,
            "prompt_template": self.prompt_template,
            "total_detections": total,
            "kept_detections": int(kept),
            "filtered_out": total - int(kept),
            "decisions": filtered_df['llm_decision'].value_counts().to_dict()
        }

        summary_path = self.results_dir / f"{base_name}_llm_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        # Save prompts and responses for debugging
        debug_log = {
            "model": self.client.model,
            "prompt_template": self.prompt_template,
            "input_file": str(input_csv),
            "total_prompts": len(prompts),
            "interactions": [
                {
                    "index": int(idx),
                    "detection_id": det_id,
                    "prompt": prompt,
                    "response": {
                        "decision": results.get(idx).decision.value if results.get(idx) else "",
                        "raw_response": results.get(idx).raw_response if results.get(idx) else "",
                        "error": results.get(idx).error_message if results.get(idx) else "",
                        "processing_time": results.get(idx).processing_time if results.get(idx) else None
                    }
                }
                for idx, det_id, prompt in prompts
            ]
        }

        debug_path = self.results_dir / f"{base_name}_prompts_and_responses.json"
        with open(debug_path, 'w', encoding='utf-8') as f:
            json.dump(debug_log, f, indent=2)

        return {
            "filtered_csv": filtered_path,
            "summary_json": summary_path,
            "debug_log": debug_path
        }

    def run_on_file(self, context_csv: Path) -> Dict[str, Path]:
        """Run LLM evaluation on a single detection file."""
        df = self._load_detection_file(context_csv)
        prompts = self._build_prompts(df)
        results = self._evaluate_prompts(prompts)
        merged = self._merge_results(df, results)
        return self._save_outputs(context_csv, merged, prompts, results)

    def run_on_all_smells(self) -> List[Dict[str, Path]]:
        """Run LLM evaluation on all detection files for this IaC technology."""
        pattern = f"{self.iac_tech}_*_detections_with_context.csv"
        results = []

        for csv_file in sorted(self.data_dir.glob(pattern)):
            logger.info(f"Processing {csv_file.name}")
            results.append(self.run_on_file(csv_file))

        logger.info(f"Completed post-filter for {self.iac_tech}")
        return results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="IaC LLM Post-Filter")
    parser.add_argument("--iac-tech", type=str, default="chef",
                       choices=["chef", "ansible", "puppet"], help="IaC technology")
    parser.add_argument("--provider", type=str, default=Provider.OPENAI.value,
                       choices=[p.value for p in Provider], help="LLM provider")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name")
    parser.add_argument("--api-key", type=str, default=None, help="API key")
    parser.add_argument("--base-url", type=str, default=None, help="Base URL for compatible providers")
    parser.add_argument("--prompt-template", type=str, default="static_analysis_rules", help="Prompt template")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for testing")
    parser.add_argument("--data-dir", type=str, default=None, help="Input data directory")
    parser.add_argument("--results-dir", type=str, default=None, help="Output results directory")
    parser.add_argument("--input", type=str, default=None, help="Single input CSV file")
    return parser.parse_args()


def _resolve_api_key(provider: str, explicit_key: Optional[str] = None) -> Optional[str]:
    """Resolve API key based on provider."""
    if explicit_key:
        return explicit_key

    # Provider-specific API key resolution
    provider_key_map = {
        "openai": ["OPENAI_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
        "grok": ["XAI_API_KEY", "GROK_API_KEY"],
        "xai": ["XAI_API_KEY", "GROK_API_KEY"],
        "ollama": [],  # Ollama doesn't need API key
        "openai_compatible": ["OPENAI_COMPATIBLE_API_KEY", "OPENAI_API_KEY"]
    }

    keys_to_try = provider_key_map.get(provider, [])
    for key_name in keys_to_try:
        key_value = os.getenv(key_name)
        if key_value:
            return key_value

    return None


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent.parent
    args = parse_args()

    # Resolve API key based on provider
    api_key = _resolve_api_key(args.provider, args.api_key)

    # Validate API key for providers that need it
    if args.provider not in ["ollama"] and not api_key:
        provider_key_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "grok": "XAI_API_KEY or GROK_API_KEY",
            "xai": "XAI_API_KEY or GROK_API_KEY",
            "openai_compatible": "OPENAI_COMPATIBLE_API_KEY"
        }
        required_key = provider_key_map.get(args.provider, f"API key for {args.provider}")
        print(f"Error: {required_key} environment variable not set")
        print(f"Set it with: export {required_key}='your-api-key-here'")
        return

    runner = IaCLLMPostFilter(
        project_root=project_root,
        iac_tech=args.iac_tech,
        provider=args.provider,
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
        prompt_template=args.prompt_template,
        max_samples=args.max_samples,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        results_dir=Path(args.results_dir) if args.results_dir else None,
    )

    try:
        if args.input:
            outputs = [runner.run_on_file(Path(args.input))]
        else:
            outputs = runner.run_on_all_smells()

        print("\nPost-filter completed:")
        for output in outputs:
            print(f"  {output}")

    except Exception as e:
        logger.error(f"Post-filter failed: {e}")
        print("Check API keys and ensure context CSV files exist")


if __name__ == "__main__":
    main()
