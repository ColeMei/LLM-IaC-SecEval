"""
Chef LLM Post-Filter (Lightweight) for IaC Filter Training

- Reuses existing llm_postfilter modules by importing them (no edits there)
- Loads context-enhanced detection CSVs from extractor outputs
- Uses static-analysis-rules prompts and ±5-line context (already extracted)
- Runs an LLM (default: OpenAI gpt-4o-mini; switchable to Anthropic Claude 3.7)
- Stores decision and confidence per detection; keeps YES, drops NO, conservatively keeps missing
- Writes concise outputs to experiments/iac_filter_training/data/llm_results
- Logs prompts sent to LLMs and raw responses for debugging
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import json
import pandas as pd
import numpy as np
import argparse
import os

# Import prompt system and client from existing postfilter package (read-only)
try:
    from src.llm_postfilter.prompt_loader import ExternalPromptLoader, SecuritySmell
    from src.llm_postfilter.llm_client import (
        create_llm_client,
        Provider,
        LLMDecision,
        LLMResponse,
    )
except ImportError:
    # Allow running as a script by adding project root to sys.path
    import sys as _sys
    _project_root = Path(__file__).parent.parent.parent
    if str(_project_root) not in _sys.path:
        _sys.path.append(str(_project_root))
    from src.llm_postfilter.prompt_loader import ExternalPromptLoader, SecuritySmell
    from src.llm_postfilter.llm_client import (
        create_llm_client,
        Provider,
        LLMDecision,
        LLMResponse,
    )

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _infer_iac_tech(file_path: str) -> str:
    path_lower = (file_path or "").lower()
    if path_lower.endswith('.rb') or '/cookbooks/' in path_lower or 'chef' in path_lower:
        return 'Chef'
    if path_lower.endswith('.pp') or '/manifests/' in path_lower or 'puppet' in path_lower:
        return 'Puppet'
    if path_lower.endswith('.yml') or path_lower.endswith('.yaml') or 'ansible' in path_lower or '/playbooks/' in path_lower:
        return 'Ansible'
    return 'IaC'




class ChefLLMPostFilter:
    def __init__(
        self,
        project_root: Path,
        provider: str = Provider.OPENAI.value,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        prompt_template: str = "static_analysis_rules",
        max_samples: Optional[int] = None,
        data_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
    ):
        self.project_root = Path(project_root)
        self.data_dir = Path(data_dir) if data_dir else (self.project_root / "experiments" / "iac_filter_training" / "data")
        self.results_dir = Path(results_dir) if results_dir else (self.data_dir / "llm_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Prompt system loader (modular/configurable)
        self.prompt_template = prompt_template
        self.prompt_loader = ExternalPromptLoader(prompt_template)
        # LLM client
        self.client = create_llm_client(provider=provider, model=model, api_key=api_key, base_url=base_url)
        # Optional cap for sanity checks
        self.max_samples = max_samples
        logger.info(f"Post-filter using provider={provider}, model={model}, template={prompt_template}, max_samples={max_samples}")
        logger.info(f"Data dir: {self.data_dir} | Results dir: {self.results_dir}")

    def _load_detection_file(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Detection file not found: {path}")
        df = pd.read_csv(path)
        # Optionally cap to first N samples for quick dry-run
        if self.max_samples is not None and len(df) > self.max_samples:
            df = df.head(self.max_samples).copy()
            logger.info(f"Capped to first {self.max_samples} rows for sanity run")
        logger.info(f"Loaded {len(df)} rows from {path.name}")
        return df

    def _build_prompts(self, df: pd.DataFrame) -> List[Tuple[int, str, str]]:
        prompts: List[Tuple[int, str, str]] = []
        for idx, row in df.iterrows():
            if 'context_success' in df.columns and not bool(row.get('context_success', True)):
                continue
            smell = self._smell_from_string(str(row['smell_category']))
            if not smell:
                continue
            iac_tech = str(row.get('iac_tool') or _infer_iac_tech(str(row.get('file_path', ''))))
            # Build prompt via ExternalPromptLoader
            prompt = self.prompt_loader.create_prompt(smell, str(row.get('context_snippet', '')), iac_tech)
            prompts.append((idx, str(row['detection_id']), prompt))
        logger.info(f"Prepared {len(prompts)} prompts")
        return prompts

    @staticmethod
    def _smell_from_string(smell_name: str) -> Optional[SecuritySmell]:
        for s in SecuritySmell:
            if s.value == smell_name:
                return s
        return None

    def _evaluate_prompts(self, prompts: List[Tuple[int, str, str]]) -> Dict[int, LLMResponse]:
        strings = [p for _, _, p in prompts]
        responses = self.client.batch_evaluate(strings)
        return {idx: resp for (idx, _, _), resp in zip(prompts, responses)}

    def _merge_results(self, df: pd.DataFrame, results: Dict[int, LLMResponse]) -> pd.DataFrame:
        out = df.copy()
        out['llm_decision'] = ""
        out['llm_raw_response'] = ""
        out['llm_processing_time'] = np.nan
        out['llm_error'] = ""
        out['keep_detection'] = False

        for idx, resp in results.items():
            if idx not in out.index:
                continue
            out.at[idx, 'llm_decision'] = resp.decision.value
            out.at[idx, 'llm_raw_response'] = resp.raw_response
            out.at[idx, 'llm_processing_time'] = resp.processing_time or 0
            out.at[idx, 'llm_error'] = resp.error_message or ""
            out.at[idx, 'keep_detection'] = (resp.decision == LLMDecision.YES)

        # Conservative policy: if no decision row present (e.g., context failed and was skipped), keep
        no_decision_mask = out['llm_decision'] == ""
        out.loc[no_decision_mask, 'keep_detection'] = True
        out.loc[no_decision_mask, 'llm_decision'] = "SKIPPED"
        return out

    def _save_outputs(self, input_csv: Path, filtered_df: pd.DataFrame, prompts: List[Tuple[int, str, str]], results: Dict[int, LLMResponse]) -> Dict[str, Path]:
        base = input_csv.stem
        filtered_path = self.results_dir / f"{base}_llm_filtered.csv"
        filtered_df.to_csv(filtered_path, index=False)

        # Summary
        decision_counts = filtered_df['llm_decision'].value_counts().to_dict()
        kept = int(filtered_df['keep_detection'].sum())
        total = int(len(filtered_df))
        summary = {
            "model": self.client.model,
            "prompt_style": self.prompt_template,
            "total": total,
            "kept": kept,
            "filtered_out": total - kept,
            "decisions": decision_counts,
        }
        summary_path = self.results_dir / f"{base}_llm_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        # Prompts & responses log for debugging
        pr_log = {
            "model": self.client.model,
            "prompt_style": self.prompt_template,
            "input_file": str(input_csv),
            "count": len(prompts),
            "interactions": []
        }
        for idx, det_id, prompt in prompts:
            resp = results.get(idx)
            pr_log["interactions"].append({
                "index": int(idx),
                "detection_id": det_id,
                "prompt": prompt,
                "response": {
                    "decision": resp.decision.value if resp else "",
                    "raw": (resp.raw_response if resp else ""),
                    "error": (resp.error_message if resp else ""),
                    "processing_time": (resp.processing_time if resp else None)
                }
            })
        pr_path = self.results_dir / f"{base}_prompts_and_responses.json"
        with open(pr_path, 'w', encoding='utf-8') as f:
            json.dump(pr_log, f, indent=2)

        return {"filtered_csv": filtered_path, "summary_json": summary_path, "prompts_and_responses": pr_path}

    def run_on_file(self, context_csv: Path) -> Dict[str, Path]:
        df = self._load_detection_file(context_csv)
        prompts = self._build_prompts(df)
        results = self._evaluate_prompts(prompts)
        merged = self._merge_results(df, results)
        return self._save_outputs(context_csv, merged, prompts, results)

    def run_on_all_smells(self) -> List[Dict[str, Path]]:
        outputs: List[Dict[str, Path]] = []
        for path in sorted(self.data_dir.glob("chef_*_detections_with_context.csv")):
            logger.info(f"Running post-filter on {path.name}")
            outputs.append(self.run_on_file(path))
        logger.info("Post-filter completed for all smells")
        return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chef LLM Post-Filter Runner")
    parser.add_argument("--provider", type=str, default=Provider.OPENAI.value, choices=[p.value for p in Provider], help="LLM provider")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--api-key", type=str, default=None, help="API key (falls back to env vars)")
    parser.add_argument("--base-url", type=str, default=None, help="Base URL for openai_compatible/ollama")
    parser.add_argument("--prompt-template", type=str, default="static_analysis_rules", help="Prompt template name")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples per file for sanity runs")
    parser.add_argument("--data-dir", type=str, default=None, help="Override input data directory")
    parser.add_argument("--results-dir", type=str, default=None, help="Override output results directory")
    parser.add_argument("--input", type=str, default=None, help="Run on a single context CSV file instead of all")
    return parser.parse_args()


def main():
    project_root = Path(__file__).parent.parent.parent
    args = parse_args()

    # Resolve API key fallback
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_COMPATIBLE_API_KEY")

    runner = ChefLLMPostFilter(
        project_root,
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
            context_csv = Path(args.input)
            outputs = [runner.run_on_file(context_csv)]
        else:
            outputs = runner.run_on_all_smells()
        print("\nLLM post-filter outputs:")
        for r in outputs:
            print(r)
    except Exception as e:
        print(f"Post-filter failed: {e}")
        print("Ensure API keys are set and context CSVs exist")


if __name__ == "__main__":
    main()
