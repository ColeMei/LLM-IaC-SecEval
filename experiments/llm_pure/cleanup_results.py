#!/usr/bin/env python3
"""
Results Directory Cleanup Script - Simplified

Simple workflow:
1. Run experiments -> files accumulate in results/
2. Archive completed experiments -> move to archive/
3. Clean archive/ -> delete old archived experiments

With unified timestamps, cleanup is much simpler and more reliable.
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List

# Add src to path for pipeline imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from llm_pure.config import config


class ResultsCleanup:
    """Simplified cleanup for the typical experiment workflow"""

    def __init__(self, results_dir: Path = None):
        self.results_dir = results_dir or config.results_dir
        self.archive_dir = self.results_dir / "archive"

    def analyze_directory(self) -> Dict:
        """Analyze current and archived experiments"""
        analysis = {
            'current_experiments': defaultdict(list),
            'archived_experiments': defaultdict(list),
            'total_current_files': 0,
            'total_archived_files': 0,
            'current_size_mb': 0,
            'archived_size_mb': 0
        }

        # Analyze current results
        if self.results_dir.exists():
            for item in self.results_dir.iterdir():
                if item.is_file():
                    if item.name.startswith('comparison_report_'):
                        continue
                    experiment_id = self._extract_experiment_id(item.name)
                    if experiment_id:
                        analysis['current_experiments'][experiment_id].append(item)
                        analysis['total_current_files'] += 1
                        analysis['current_size_mb'] += item.stat().st_size / (1024 * 1024)

            for subdir in ['prompts', 'raw_responses', 'evaluations']:
                subdir_path = self.results_dir / subdir
                if subdir_path.exists():
                    for file_path in subdir_path.iterdir():
                        if file_path.is_file():
                            experiment_id = self._extract_experiment_id(file_path.name)
                            if experiment_id:
                                analysis['current_experiments'][experiment_id].append(file_path)
                                analysis['total_current_files'] += 1
                                analysis['current_size_mb'] += file_path.stat().st_size / (1024 * 1024)

        # Analyze archived experiments
        if self.archive_dir.exists():
            for exp_dir in self.archive_dir.iterdir():
                if exp_dir.is_dir():
                    experiment_id = exp_dir.name
                    files = [f for f in exp_dir.rglob('*') if f.is_file()]
                    analysis['archived_experiments'][experiment_id] = files
                    analysis['total_archived_files'] += len(files)
                    for f in files:
                        analysis['archived_size_mb'] += f.stat().st_size / (1024 * 1024)

        return analysis

    def _extract_experiment_id(self, filename: str) -> str:
        """Extract experiment ID (YYYYMMDD_HHMMSS) from filename"""
        parts = filename.split('_')
        for i, part in enumerate(parts):
            if len(part) == 8 and part.isdigit():
                if i + 1 < len(parts):
                    time_part = parts[i + 1].split('.')[0]
                    if len(time_part) == 6 and time_part.isdigit():
                        return f"{part}_{time_part}"
        return None

    def print_status(self):
        """Print current status of experiments"""
        analysis = self.analyze_directory()
        print(f"\n📊 Experiment Status")
        print(f"{'='*50}")
        print(f"📁 Results directory: {self.results_dir}")

        print(f"\n🔬 Current Experiments: {len(analysis['current_experiments'])}")
        print(f"   Files: {analysis['total_current_files']}, Size: {analysis['current_size_mb']:.1f} MB")

        if analysis['current_experiments']:
            sorted_current = sorted(analysis['current_experiments'].items(), reverse=True)
            for i, (exp_id, files) in enumerate(sorted_current[:10], 1):
                exp_date = self._format_timestamp(exp_id)
                print(f"   {i:2d}. {exp_id} ({exp_date}) - {len(files)} files")
            if len(sorted_current) > 10:
                print(f"       ... and {len(sorted_current) - 10} more")

        print(f"\n📦 Archived Experiments: {len(analysis['archived_experiments'])}")
        print(f"   Files: {analysis['total_archived_files']}, Size: {analysis['archived_size_mb']:.1f} MB")

        if analysis['archived_experiments']:
            sorted_archived = sorted(analysis['archived_experiments'].items(), reverse=True)
            for i, (exp_id, files) in enumerate(sorted_archived[:10], 1):
                exp_date = self._format_timestamp(exp_id)
                print(f"   {i:2d}. {exp_id} ({exp_date}) - {len(files)} files")
            if len(sorted_archived) > 10:
                print(f"       ... and {len(sorted_archived) - 10} more")

    def _format_timestamp(self, timestamp: str) -> str:
        """Format timestamp for display"""
        try:
            dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
            return dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            return timestamp

    def archive_experiments(self, experiment_ids: List[str] = None, interactive: bool = True) -> bool:
        """Archive experiments by moving them to archive/ directory"""
        analysis = self.analyze_directory()

        if not analysis['current_experiments']:
            print("✨ No current experiments to archive")
            return False

        if experiment_ids is None and interactive:
            print(f"\n📦 Available experiments to archive:")
            sorted_exp = sorted(analysis['current_experiments'].keys(), reverse=True)
            for i, exp_id in enumerate(sorted_exp, 1):
                exp_date = self._format_timestamp(exp_id)
                file_count = len(analysis['current_experiments'][exp_id])
                print(f"{i:2d}. {exp_id} ({exp_date}) - {file_count} files")
            selection = input(f"\nEnter experiment numbers (e.g., 1,2,5-8) or 'all': ").strip()
            if selection.lower() == 'all':
                experiment_ids = sorted_exp
            else:
                experiment_ids = self._parse_selection(selection, sorted_exp)
        elif experiment_ids is None:
            experiment_ids = list(analysis['current_experiments'].keys())

        if not experiment_ids:
            print("❌ No experiments selected")
            return False

        self.archive_dir.mkdir(parents=True, exist_ok=True)

        archived_count = 0
        for exp_id in experiment_ids:
            files = analysis['current_experiments'].get(exp_id, [])
            if files:
                exp_archive_dir = self.archive_dir / exp_id
                exp_archive_dir.mkdir(parents=True, exist_ok=True)
                (exp_archive_dir / "raw_responses").mkdir(parents=True, exist_ok=True)
                (exp_archive_dir / "prompts").mkdir(parents=True, exist_ok=True)
                (exp_archive_dir / "evaluations").mkdir(parents=True, exist_ok=True)
                for file_path in files:
                    try:
                        if 'raw_responses' in str(file_path):
                            target_dir = exp_archive_dir / "raw_responses"
                        elif 'prompts' in str(file_path):
                            target_dir = exp_archive_dir / "prompts"
                        elif 'evaluations' in str(file_path):
                            target_dir = exp_archive_dir / "evaluations"
                        else:
                            target_dir = exp_archive_dir
                        shutil.move(str(file_path), str(target_dir / file_path.name))
                    except Exception as e:
                        print(f"❌ Failed to archive {file_path.name}: {e}")
                print(f"📦 Archived {exp_id} ({len(files)} files)")
                archived_count += len(files)

        print(f"✅ Archived {len(experiment_ids)} experiments ({archived_count} files)")
        return True

    def clean_archive(self, days_to_keep: int = 30, dry_run: bool = True) -> List[str]:
        """Delete archived experiments older than specified days"""
        if not self.archive_dir.exists():
            print("📦 No archive directory found")
            return []

        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        experiments_to_delete = []
        files_to_delete = []

        for exp_dir in self.archive_dir.iterdir():
            if exp_dir.is_dir():
                exp_id = exp_dir.name
                try:
                    exp_date = datetime.strptime(exp_id, "%Y%m%d_%H%M%S")
                    if exp_date < cutoff_date:
                        experiments_to_delete.append(exp_id)
                        files = list(exp_dir.rglob('*'))
                        files_to_delete.extend([f for f in files if f.is_file()])
                except ValueError:
                    continue

        if not experiments_to_delete:
            print(f"✨ No archived experiments older than {days_to_keep} days found")
            return []

        print(f"\n🗑️  Found {len(experiments_to_delete)} archived experiments to delete:")
        for exp_id in experiments_to_delete[:10]:
            exp_date = self._format_timestamp(exp_id)
            print(f"   - {exp_id} ({exp_date})")
        if len(experiments_to_delete) > 10:
            print(f"   ... and {len(experiments_to_delete) - 10} more")
        print(f"   Total files to delete: {len(files_to_delete)}")

        if not dry_run:
            for exp_id in experiments_to_delete:
                exp_dir = self.archive_dir / exp_id
                try:
                    shutil.rmtree(exp_dir)
                    print(f"🗑️  Deleted archived experiment: {exp_id}")
                except Exception as e:
                    print(f"❌ Failed to delete {exp_id}: {e}")

        return experiments_to_delete

    def interactive_cleanup(self):
        """Simple interactive cleanup menu"""
        while True:
            self.print_status()
            print(f"\n🧹 Cleanup Options:")
            print(f"1. Archive current experiments")
            print(f"2. Clean archive (delete old archived experiments)")
            print(f"3. Exit")
            choice = input(f"\nSelect option (1-3): ").strip()
            if choice == "1":
                self.archive_experiments()
            elif choice == "2":
                days = input("Delete archived experiments older than how many days? (default: 30): ").strip()
                try:
                    days = int(days) if days else 30
                except ValueError:
                    print("❌ Invalid number of days")
                    continue
                to_delete = self.clean_archive(days, dry_run=True)
                if to_delete:
                    confirm = input(f"\nDelete these {len(to_delete)} archived experiments? (y/N): ").strip().lower()
                    if confirm == 'y':
                        self.clean_archive(days, dry_run=False)
            elif choice == "3":
                print("👋 Cleanup finished")
                break
            else:
                print("❌ Invalid choice")
            input("\nPress Enter to continue...")

    def _parse_selection(self, selection: str, items: List) -> List:
        """Parse user selection like '1,3,5-8' into list of items"""
        selected = []
        for part in selection.split(','):
            part = part.strip()
            if '-' in part:
                try:
                    start, end = map(int, part.split('-'))
                    for i in range(start, min(end + 1, len(items) + 1)):
                        if 1 <= i <= len(items):
                            selected.append(items[i - 1])
                except ValueError:
                    continue
            else:
                try:
                    i = int(part)
                    if 1 <= i <= len(items):
                        selected.append(items[i - 1])
                except ValueError:
                    continue
        return selected


def main():
    parser = argparse.ArgumentParser(description="Simple cleanup for experiment workflow")
    parser.add_argument("--status", action="store_true",
                        help="Show current status of experiments")
    parser.add_argument("--archive", nargs="*", metavar="EXP_ID",
                        help="Archive specific experiments (or all if none specified)")
    parser.add_argument("--clean-archive", type=int, metavar="DAYS", default=30,
                        help="Delete archived experiments older than N days (default: 30)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without actually doing it")

    args = parser.parse_args()
    cleanup = ResultsCleanup()

    if args.status:
        cleanup.print_status()
        return

    if args.archive is not None:
        if args.archive:
            cleanup.archive_experiments(args.archive, interactive=False)
        else:
            cleanup.archive_experiments(interactive=False)
        return

    if args.clean_archive:
        cleanup.clean_archive(args.clean_archive, dry_run=args.dry_run)
        return

    cleanup.interactive_cleanup()


if __name__ == "__main__":
    main()