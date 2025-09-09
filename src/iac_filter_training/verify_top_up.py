"""
Verify top-up results: counts per tech/smell/split and duplicate-free status.

Dedup key: smell || normalized(content)
Normalization: strip, collapse whitespace, lowercase, strip surrounding quotes.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple


ROOT = Path(__file__).parent.parent.parent


def normalize_content(raw: str) -> str:
    if raw is None:
        return ""
    s = str(raw).strip()
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        s = s[1:-1]
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def stats_for_split(rows: List[dict]) -> Dict:
    total = len(rows)
    smell_counts: Dict[str, int] = defaultdict(int)
    keys: List[str] = []
    for r in rows:
        smell = str(r.get("smell", ""))
        content = str(r.get("content", ""))
        smell_counts[smell] += 1
        keys.append(f"{smell}||{normalize_content(content)}")

    unique = len(set(keys))
    dup_records = total - unique
    dup_keys = len([k for k, c in _counts(keys).items() if c > 1])
    return {
        "total": total,
        "unique": unique,
        "dup_records": dup_records,
        "dup_keys": dup_keys,
        "smells": dict(smell_counts),
    }


def _counts(items: List[str]) -> Dict[str, int]:
    d: Dict[str, int] = defaultdict(int)
    for it in items:
        d[it] += 1
    return d


def cross_overlap(a_rows: List[dict], b_rows: List[dict]) -> int:
    a_keys = {f"{r.get('smell','')}||{normalize_content(str(r.get('content','')))}" for r in a_rows}
    b_keys = {f"{r.get('smell','')}||{normalize_content(str(r.get('content','')))}" for r in b_rows}
    return len(a_keys & b_keys)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify top-up counts and duplicates")
    parser.add_argument("--base-dir", type=str, default="tmp/new-train-val-test",
                        help="Base directory containing per-tech JSONLs and combined formatted_dataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = ROOT / args.base_dir

    # Per-tech paths
    techs = ["chef", "ansible", "puppet"]
    per_split: Dict[str, Dict[str, List[dict]]] = {}
    for t in techs:
        # In cleaned dataset layout, per-tech JSONLs are at base/<tech>/{tech}_{split}.jsonl
        t_dir = base / t
        per_split[t] = {
            "train": load_jsonl(t_dir / f"{t}_train.jsonl"),
            "val": load_jsonl(t_dir / f"{t}_val.jsonl"),
            "test": load_jsonl(t_dir / f"{t}_test.jsonl"),
        }

    # Combined (train/val) in cleaned dataset layout
    combined_dir = base / "formatted_dataset"
    combined_train = load_jsonl(combined_dir / "combined_train.jsonl")
    combined_val = load_jsonl(combined_dir / "combined_val.jsonl")

    # Print counts and duplicates per tech
    print("\n=== Per-tech counts & duplicates ===")
    for t in techs:
        train_stats = stats_for_split(per_split[t]["train"]) 
        val_stats = stats_for_split(per_split[t]["val"]) 
        test_stats = stats_for_split(per_split[t]["test"]) if per_split[t]["test"] else {"total": 0, "unique": 0, "dup_records": 0, "dup_keys": 0, "smells": {}}
        print(f"\n{t.title()}:")
        print(f"  Train: total={train_stats['total']}, unique={train_stats['unique']}, dup_records={train_stats['dup_records']}, dup_keys={train_stats['dup_keys']}")
        print(f"    smells={train_stats['smells']}")
        print(f"  Val:   total={val_stats['total']}, unique={val_stats['unique']}, dup_records={val_stats['dup_records']}, dup_keys={val_stats['dup_keys']}")
        print(f"    smells={val_stats['smells']}")
        if per_split[t]["test"]:
            print(f"  Test:  total={test_stats['total']}, unique={test_stats['unique']}, dup_records={test_stats['dup_records']}, dup_keys={test_stats['dup_keys']}")

    # Cross-split overlaps (combined train/val vs tests)
    print("\n=== Cross-split overlaps (by smell||normalized(content)) ===")
    for t in techs:
        tr = per_split[t]["train"]
        va = per_split[t]["val"]
        te = per_split[t]["test"]
        if not te:
            continue
        print(f"{t.title()}:")
        print(f"  train∩test = {cross_overlap(tr, te)}")
        print(f"  val∩test   = {cross_overlap(va, te)}")
        print(f"  train∩val  = {cross_overlap(tr, va)}")

    # Combined duplicates (train/val)
    print("\n=== Combined (train/val) duplicates ===")
    cmb_train_stats = stats_for_split(combined_train)
    cmb_val_stats = stats_for_split(combined_val)
    print(f"Train: total={cmb_train_stats['total']}, unique={cmb_train_stats['unique']}, dup_records={cmb_train_stats['dup_records']}, dup_keys={cmb_train_stats['dup_keys']}")
    print(f"Val:   total={cmb_val_stats['total']}, unique={cmb_val_stats['unique']}, dup_records={cmb_val_stats['dup_records']}, dup_keys={cmb_val_stats['dup_keys']}")
    print(f"train∩val = {cross_overlap(combined_train, combined_val)}")

    # Emit duplicates detail to JSONL for inspection
    def key(r: dict) -> str:
        return f"{r.get('smell','')}||{normalize_content(str(r.get('content','')))}"

    def dump_dups(rows: List[dict], out_path: Path):
        counts = _counts([key(r) for r in rows])
        dup_keys = {k for k, c in counts.items() if c > 1}
        if not dup_keys:
            return
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                k = key(r)
                if k in dup_keys:
                    f.write(json.dumps({"key": k, **r}) + "\n")

    dump_dups(combined_train, base / "formatted_dataset" / "combined_train_duplicates.jsonl")
    dump_dups(combined_val, base / "formatted_dataset" / "combined_val_duplicates.jsonl")


if __name__ == "__main__":
    main()


