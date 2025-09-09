"""
Top-up pipeline runner

Goal: Minimal, additive flow to fulfill smell-specific needs without duplicates
and without overwriting existing artifacts.

Workflow per technology requested in needs:
- Read needs from JSONL (tmp/new-train-val-test/smell_needs_only.jsonl)
- Read blacklist keys (tmp/new-train-val-test/blacklist_keys.json)
- From GLITCH-<tech>-oracle.csv, select unique detections by smell where
  normalized(content) + smell_key is not in blacklist
- Save subset to experiments/iac_filter_training/data/<tech>/top_up/<run_tag>/
  using existing extractor for context
- Run LLM post-filter in the same top_up/<run_tag>/ directory
- Format filtered results into samples
- Append samples into per-tech formatted_dataset/<tech>_{train,val}.jsonl
- Rebuild combined formatted_dataset/combined_{train,val}.jsonl from per-tech

Notes:
- No dynamic keys: blacklist is derived from the cleaned dataset
- No file-level constraints here (assumed already enforced upstream)
- Keeps all outputs isolated under top_up/<run_tag>/ to avoid overwrites
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Set

import pandas as pd

# Ensure project root is on sys.path for local imports
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.append(str(_project_root))

# Local imports
from src.iac_filter_training.extractor import IaCDetectionExtractor
from src.iac_filter_training.post_filter import IaCLLMPostFilter
from src.iac_filter_training.dataset_formatter import IaCDatasetFormatter


# -------------------------- Helpers and types --------------------------

SUPPORTED_TECHS: List[str] = ["chef", "ansible", "puppet"]


def now_tag() -> str:
    return datetime.now().strftime("top_up_%Y%m%d_%H%M%S")


def normalize_content(raw: str) -> str:
    """Apply the same normalization used in dedupe:
    - strip
    - collapse whitespace
    - lowercase
    - strip surrounding quotes
    """
    if raw is None:
        return ""
    s = str(raw).strip()
    # strip surrounding single/double quotes
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        s = s[1:-1]
    # collapse whitespace and lowercase
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def key_for_dedupe(smell_key: str, content: str) -> str:
    return f"{smell_key}||{normalize_content(content)}"


def map_need_smell_to_glitch(smell_key: str) -> str:
    """Map normalized smell keys from needs to GLITCH ERROR values."""
    mapping = {
        "hard_coded_secret": "hardcoded-secret",
        "suspicious_comment": "suspicious comment",
        "weak_cryptography_algorithms": "weak cryptography algorithms",
        "use_of_weak_cryptography_algorithms": "weak cryptography algorithms",
        "use_of_http": "use of http",
        "use_of_http_without_ssl_tls": "use of http",
    }
    return mapping.get(smell_key, smell_key)


def map_need_smell_to_samples(smell_key: str) -> str:
    """Map smell keys in needs to sample 'smell' emitted by formatter.

    Formatter emits:
    - hard_coded_secret
    - suspicious_comment
    - weak_cryptography_algorithms
    - use_of_http
    """
    mapping = {
        "hard_coded_secret": "hard_coded_secret",
        "suspicious_comment": "suspicious_comment",
        "weak_cryptography_algorithms": "weak_cryptography_algorithms",
        "use_of_weak_cryptography_algorithms": "weak_cryptography_algorithms",
        "use_of_http": "use_of_http",
        "use_of_http_without_ssl_tls": "use_of_http",
    }
    return mapping.get(smell_key, smell_key)


@dataclass
class NeedItem:
    tech: str
    split: str  # train | val
    smell: str  # needs smell key form
    count: int


def load_needs(needs_path: Path) -> List[NeedItem]:
    needs: List[NeedItem] = []
    with open(needs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("action") != "need":
                continue
            needs.append(
                NeedItem(
                    tech=obj["tech"].lower(),
                    split=obj["split"].lower(),
                    smell=obj["smell"],
                    count=int(obj["count"]),
                )
            )
    return needs


def load_blacklist(blacklist_path: Path) -> Set[str]:
    with open(blacklist_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return set(str(x) for x in data)


# -------------------------- Core runner --------------------------


class TopUpRunner:
    def __init__(
        self,
        project_root: Path,
        needs_path: Path,
        blacklist_path: Path,
        run_tag: Optional[str] = None,
        seed: int = 42,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4.0",
        context_lines: int = 5,
        target_dataset_dir: Optional[Path] = None,
    ):
        self.project_root = Path(project_root)
        self.needs_path = Path(needs_path)
        self.blacklist_path = Path(blacklist_path)
        self.run_tag = run_tag or now_tag()
        self.seed = seed
        self.provider = provider
        self.model = model
        self.context_lines = context_lines

        self.data_dir = self.project_root / "data" / "iac_filter_training"
        # Where to append and rebuild combined (cleaned dataset base)
        self.target_dataset_dir = Path(target_dataset_dir) if target_dataset_dir else (self.project_root / "tmp" / "new-train-val-test")
        # Session-level keys to avoid duplicates within the same top-up run across all techs/splits
        self.session_selected_keys: Set[str] = set()

        # Derived
        self.needs: List[NeedItem] = load_needs(self.needs_path)
        self.blacklist: Set[str] = load_blacklist(self.blacklist_path)

        # Group needs by tech and smell/split
        self.needs_by_tech: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
        for n in self.needs:
            self.needs_by_tech[n.tech][f"{n.split}:{n.smell}"] = n.count

    # -------------------------- Selection --------------------------

    def _read_glitch_csv(self, tech: str) -> pd.DataFrame:
        glitch_file = self.data_dir / f"GLITCH-{tech}-oracle.csv"
        if not glitch_file.exists():
            raise FileNotFoundError(f"GLITCH CSV not found: {glitch_file}")
        df = pd.read_csv(glitch_file)
        return df

    def _read_target_line(self, tech: str, file_rel: str, line_number: int) -> Optional[str]:
        extractor = IaCDetectionExtractor(self.project_root, tech)
        file_path = extractor.find_iac_file(file_rel)
        if file_path is None:
            return None
        ctx = extractor.extract_context(file_path, int(line_number), context_lines=self.context_lines)
        return ctx.get("target_content", "")

    def select_candidates_for_tech(self, tech: str) -> pd.DataFrame:
        """Return subset GLITCH detections for this tech matching needs and blacklist."""
        # Build smell->per-split quota
        needs_split_smell: Dict[str, Dict[str, int]] = defaultdict(dict)
        for key, count in self.needs_by_tech.get(tech, {}).items():
            split, smell_need = key.split(":", 1)
            needs_split_smell[split][smell_need] = int(count)

        if not needs_split_smell:
            return pd.DataFrame(columns=["PATH", "LINE", "ERROR"])  # nothing to do

        df = self._read_glitch_csv(tech)

        # Build list of candidate rows per need smell
        rng = random.Random(self.seed)
        selected_rows: List[Dict] = []
        selected_keys: Set[str] = set()

        # Start with blacklist + any keys already selected in this session (across techs/splits)
        global_used: Set[str] = set(self.blacklist) | set(self.session_selected_keys)
        # Also include existing cleaned dataset keys for this tech (train/val/test)
        tech_fmt_dir = self.target_dataset_dir / tech
        for split_name in ("train", "val", "test"):
            jsonl_path = tech_fmt_dir / f"{tech}_{split_name}.jsonl"
            if jsonl_path.exists():
                for row in self._read_jsonl(jsonl_path):
                    k = f"{row.get('smell','')}||{normalize_content(str(row.get('content','')))}"
                    global_used.add(k)

        for split, need_map in needs_split_smell.items():
            for smell_need, target_count in need_map.items():
                glitch_smell = map_need_smell_to_glitch(smell_need)
                subset = df[df["ERROR"] == glitch_smell].copy()

                # Shuffle deterministically
                idxs = list(subset.index)
                rng.shuffle(idxs)

                filled = 0
                for idx in idxs:
                    row = subset.loc[idx]
                    file_rel = str(row["PATH"])
                    line_no = int(row["LINE"])

                    # Read content and build key
                    content = self._read_target_line(tech, file_rel, line_no)
                    if content is None:
                        continue
                    # Use the canonical JSONL smell name for dedupe key (to match blacklist)
                    canonical_smell = map_need_smell_to_samples(smell_need)
                    key = key_for_dedupe(canonical_smell, content)
                    if key in global_used or key in selected_keys:
                        continue

                    selected_keys.add(key)
                    self.session_selected_keys.add(key)
                    global_used.add(key)
                    selected_rows.append({
                        "PATH": file_rel,
                        "LINE": line_no,
                        "ERROR": glitch_smell,
                    })
                    filled += 1
                    if filled >= target_count:
                        break

        # Build DataFrame
        if not selected_rows:
            return pd.DataFrame(columns=["PATH", "LINE", "ERROR"])

        result = pd.DataFrame(selected_rows)
        # Deduplicate rows if any accidental repeats
        result = result.drop_duplicates(subset=["PATH", "LINE", "ERROR"]).reset_index(drop=True)
        return result

    # -------------------------- Context & LLM --------------------------

    def _tech_topup_dir(self, tech: str) -> Path:
        return (self.project_root /
                "experiments" / "iac_filter_training" / "data" /
                tech / "top_up" / self.run_tag)

    def write_with_context(self, tech: str, subset_df: pd.DataFrame) -> Tuple[Path, List[Path]]:
        """Write detection CSVs and context-enhanced CSVs under top_up/<run_tag>"""
        out_dir = self._tech_topup_dir(tech)
        out_dir.mkdir(parents=True, exist_ok=True)

        extractor = IaCDetectionExtractor(self.project_root, tech)
        processed = extractor._process_detections(subset_df)
        extractor.save_detections(processed, out_dir)

        enhanced = extractor.enhance_with_context(processed, context_lines=self.context_lines)
        extractor.save_context_outputs(enhanced, out_dir)

        # Return paths of the context CSVs we produced
        context_files: List[Path] = []
        for smell in extractor.target_smells:
            filename = f"{tech}_{smell.replace(' ', '_').replace('-', '_')}_detections_with_context.csv"
            path = out_dir / filename
            if path.exists():
                context_files.append(path)

        return out_dir, context_files

    def run_llm_postfilter(self, tech: str, data_dir: Path) -> Path:
        results_dir = data_dir / "llm_results"
        runner = IaCLLMPostFilter(
            project_root=self.project_root,
            iac_tech=tech,
            provider=self.provider,
            model=self.model,
            data_dir=data_dir,
            results_dir=results_dir,
        )
        runner.run_on_all_smells()
        return results_dir

    # -------------------------- Formatting & merge --------------------------

    def _read_jsonl(self, path: Path) -> List[dict]:
        rows: List[dict] = []
        if not path.exists():
            return rows
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def _write_jsonl(self, path: Path, rows: List[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def format_and_merge(self, tech: str, results_dir: Path) -> None:
        # Format all llm_filtered files into samples
        formatter = IaCDatasetFormatter(
            data_dir=self.project_root / "experiments" / "iac_filter_training" / "data",
            results_dir=self.project_root / "experiments" / "iac_filter_training" / "data" / "llm_results",
            output_dir=self.project_root / "experiments" / "iac_filter_training" / "data" / "formatted_dataset",
            iac_tech=tech,
            combined=False,
        )

        all_samples: List[Dict] = []
        for csv_path in sorted(results_dir.glob(f"{tech}_*_llm_filtered.csv")):
            smell_name = csv_path.stem.replace("_llm_filtered", "")
            prompts_path = csv_path.parent / f"{smell_name}_prompts_and_responses.json"
            if not prompts_path.exists():
                continue
            samples = formatter.format_single_file(csv_path, prompts_path)
            all_samples.extend(samples)

        # Build per-smell pools
        samples_by_smell: Dict[str, List[Dict]] = defaultdict(list)
        for s in all_samples:
            samples_by_smell[s["smell"]].append(s)

        # Needs for this tech by split and smell (mapped to formatter smell names)
        needs_split_smell: Dict[str, Dict[str, int]] = defaultdict(dict)
        for key, count in self.needs_by_tech.get(tech, {}).items():
            split, smell_need = key.split(":", 1)
            mapped_smell = map_need_smell_to_samples(smell_need)
            needs_split_smell[split][mapped_smell] = int(count)

        # Helper to build dedupe key from a sample
        def sample_key(sample: Dict) -> str:
            return f"{sample.get('smell','')}||{normalize_content(str(sample.get('content','')))}"

        # Build initial used keys set from external blacklist AND existing cleaned dataset (train/val/test)
        used_keys: Set[str] = set(self.blacklist)
        tech_fmt_dir = self.target_dataset_dir / tech
        existing_train = self._read_jsonl(tech_fmt_dir / f"{tech}_train.jsonl")
        existing_val = self._read_jsonl(tech_fmt_dir / f"{tech}_val.jsonl")
        existing_test = self._read_jsonl(tech_fmt_dir / f"{tech}_test.jsonl")
        for row in existing_train + existing_val + existing_test:
            used_keys.add(f"{row.get('smell','')}||{normalize_content(str(row.get('content','')))}")

        # Allocate to splits ensuring no duplicates across splits or with existing
        rng = random.Random(self.seed)
        selected_for_split: Dict[str, List[Dict]] = {"train": [], "val": []}

        # Allocate VAL first, then TRAIN to avoid starving val due to dedupe
        for split in ("val", "train"):
            smell_map = needs_split_smell.get(split, {})
            for smell_key, target_count in smell_map.items():
                pool = list(samples_by_smell.get(smell_key, []))
                rng.shuffle(pool)
                taken: List[Dict] = []
                for sample in pool:
                    k = sample_key(sample)
                    if k in used_keys:
                        continue
                    used_keys.add(k)
                    # Add split marker for traceability (no schema change downstream)
                    sample = dict(sample)
                    sample["_topup_split"] = split
                    taken.append(sample)
                    if len(taken) >= target_count:
                        break
                # If pool exhausted and we didn't reach target_count, keep what we have
                if len(taken) < target_count:
                    # Log shortfall context via print (kept minimal per your style)
                    print(f"[TOP-UP] {tech} {split} {smell_key}: requested={target_count}, selected={len(taken)} (pool exhausted after dedupe)")
                selected_for_split[split].extend(taken)

        # Append to per-tech JSONLs in cleaned dataset base (no extra de-dupe beyond top-up batch)
        tech_fmt_dir = self.target_dataset_dir / tech
        for split in ("train", "val"):
            if not selected_for_split[split]:
                continue
            jsonl_path = tech_fmt_dir / f"{tech}_{split}.jsonl"
            existing = self._read_jsonl(jsonl_path)
            # Only append samples intended for this split (guard if any leakage)
            to_append = [s for s in selected_for_split[split] if s.get("_topup_split") == split]
            merged = existing + to_append
            # Atomic-ish write
            tmp_path = jsonl_path.with_suffix(".jsonl.new")
            self._write_jsonl(tmp_path, merged)
            tmp_path.replace(jsonl_path)

        # Rebuild combined from per-tech in cleaned dataset base
        combined_dir = self.target_dataset_dir / "formatted_dataset"
        train_all: List[dict] = []
        val_all: List[dict] = []
        for t in SUPPORTED_TECHS:
            t_dir = self.target_dataset_dir / t
            train_all.extend(self._read_jsonl(t_dir / f"{t}_train.jsonl"))
            val_all.extend(self._read_jsonl(t_dir / f"{t}_val.jsonl"))

        self._write_jsonl(combined_dir / "combined_train.jsonl", train_all)
        self._write_jsonl(combined_dir / "combined_val.jsonl", val_all)

    # -------------------------- Orchestration --------------------------

    def _latest_topup_dir_for_tech(self, tech: str) -> Optional[Path]:
        base = (self.project_root / "experiments" / "iac_filter_training" / "data" / tech / "top_up")
        if not base.exists():
            return None
        candidates = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
        return candidates[0] if candidates else None

    def run(self, merge_only: bool = False, only_tech: Optional[str] = None, topup_dir: Optional[Path] = None,
            merge_only_all: bool = False) -> None:
        if merge_only:
            if merge_only_all and not only_tech and not topup_dir:
                # Auto-detect latest top_up run per tech and merge all
                for tech in SUPPORTED_TECHS:
                    latest = self._latest_topup_dir_for_tech(tech)
                    if not latest:
                        continue
                    results_dir = latest / "llm_results"
                    if results_dir.exists():
                        self.format_and_merge(tech, results_dir)
                return

            if not only_tech or not topup_dir:
                raise ValueError("merge_only requires --tech and --topup-dir, or --merge-only-all with autodetect")
            results_dir = Path(topup_dir) / "llm_results"
            self.format_and_merge(only_tech, results_dir)
            return

        for tech in SUPPORTED_TECHS:
            if tech not in self.needs_by_tech:
                continue

            subset_df = self.select_candidates_for_tech(tech)
            if subset_df.empty:
                print(f"[TOP-UP] {tech}: no candidates selected (nothing to do)")
                continue

            data_dir, _ = self.write_with_context(tech, subset_df)
            results_dir = self.run_llm_postfilter(tech, data_dir)
            self.format_and_merge(tech, results_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Top-up sampler without duplicates")
    parser.add_argument("--needs-file", type=str, default="tmp/new-train-val-test/smell_needs_only.jsonl",
                        help="Path to JSONL with smell needs")
    parser.add_argument("--blacklist", type=str, default="tmp/new-train-val-test/blacklist_keys.json",
                        help="Path to blacklist keys JSON array")
    parser.add_argument("--run-tag", type=str, default=None, help="Run tag to isolate outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--provider", type=str, default="anthropic",
                        choices=["openai", "anthropic", "grok", "xai", "ollama", "openai_compatible"],
                        help="LLM provider for post-filter")
    parser.add_argument("--model", type=str, default="claude-sonnet-4.0",
                        help="Model name for post-filter")
    parser.add_argument("--context-lines", type=int, default=5)
    parser.add_argument("--target-dataset-dir", type=str, default="tmp/new-train-val-test",
                        help="Base directory of cleaned dataset to append to and rebuild combined")
    parser.add_argument("--merge-only", action="store_true", help="Only merge formatted top-up into cleaned dataset")
    parser.add_argument("--merge-only-all", action="store_true", help="Merge latest top_up run for all techs (autodetect)")
    parser.add_argument("--tech", type=str, choices=SUPPORTED_TECHS, help="Tech for merge-only mode")
    parser.add_argument("--topup-dir", type=str, help="Path to top_up/<run_tag> directory with llm_results for merge-only")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).parent.parent.parent
    runner = TopUpRunner(
        project_root=project_root,
        needs_path=Path(args.needs_file),
        blacklist_path=Path(args.blacklist),
        run_tag=args.run_tag,
        seed=args.seed,
        provider=args.provider,
        model=args.model,
        context_lines=args.context_lines,
        target_dataset_dir=Path(args.target_dataset_dir),
    )
    runner.run(
        merge_only=args.merge_only,
        only_tech=args.tech,
        topup_dir=Path(args.topup_dir) if args.topup_dir else None,
        merge_only_all=args.merge_only_all,
    )


if __name__ == "__main__":
    main()


