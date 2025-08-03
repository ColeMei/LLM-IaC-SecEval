#!/usr/bin/env python3
"""
Results Directory Cleanup Script

This script helps organize and clean up the results/ directory after multiple
experiment runs, providing options to archive, delete, or reorganize files.
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple

# Add src to path for pipeline imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from automated.config import config


class ResultsCleanup:
    """Handles cleanup and organization of results directory"""
    
    def __init__(self, results_dir: Path = None):
        self.results_dir = results_dir or config.results_dir
        self.archive_dir = self.results_dir / "archive"
        
    def analyze_directory(self) -> Dict:
        """Analyze the current state of results directory"""
        analysis = {
            'total_files': 0,
            'total_size_mb': 0,
            'by_type': defaultdict(list),
            'by_date': defaultdict(list),
            'experiments': defaultdict(list),
            'directories': [],
            'orphaned_files': []
        }
        
        if not self.results_dir.exists():
            return analysis
            
        # Process root-level files
        for item in self.results_dir.iterdir():
            if item.is_dir():
                analysis['directories'].append({
                    'name': item.name,
                    'size_mb': self._get_dir_size(item) / (1024 * 1024),
                    'file_count': len(list(item.rglob('*'))) if item.exists() else 0
                })
                continue
                
            if item.is_file():
                analysis['total_files'] += 1
                size_mb = item.stat().st_size / (1024 * 1024)
                analysis['total_size_mb'] += size_mb
                
                # Categorize by file type
                if item.name.startswith('full_evaluation_'):
                    analysis['by_type']['full_evaluations'].append(item)
                elif item.name.startswith('batch_'):
                    analysis['by_type']['batch_results'].append(item)
                else:
                    analysis['by_type']['other'].append(item)
                
                # Extract experiment timestamp
                timestamp = self._extract_timestamp(item.name)
                if timestamp:
                    analysis['by_date'][timestamp].append(item)
                    analysis['experiments'][timestamp].append(item)
        
        # Process subdirectory files (raw_responses, prompts, evaluations)
        subdirs = ['raw_responses', 'prompts', 'evaluations']
        for subdir_name in subdirs:
            subdir = self.results_dir / subdir_name
            if subdir.exists() and subdir.is_dir():
                for file_path in subdir.iterdir():
                    if file_path.is_file():
                        analysis['total_files'] += 1
                        size_mb = file_path.stat().st_size / (1024 * 1024)
                        analysis['total_size_mb'] += size_mb
                        
                        # Categorize by subdirectory type
                        analysis['by_type'][f'{subdir_name}_files'].append(file_path)
                        
                        # Extract timestamp and group with experiment
                        timestamp = self._extract_timestamp(file_path.name)
                        if timestamp:
                            # Convert HHMMSS to full experiment_id by matching with existing experiments
                            experiment_id = self._find_matching_experiment(timestamp, analysis['experiments'])
                            if experiment_id:
                                analysis['experiments'][experiment_id].append(file_path)
                                analysis['by_date'][experiment_id].append(file_path)
                            else:
                                # No matching experiment found - orphaned file
                                analysis['orphaned_files'].append(file_path)
                        else:
                            # No timestamp - orphaned file
                            analysis['orphaned_files'].append(file_path)
        
        return analysis
    
    def _get_dir_size(self, path: Path) -> int:
        """Calculate total size of directory"""
        total = 0
        try:
            for item in path.rglob('*'):
                if item.is_file():
                    total += item.stat().st_size
        except PermissionError:
            pass
        return total
    
    def _extract_timestamp(self, filename: str) -> str:
        """Extract timestamp from filename (format: YYYYMMDD_HHMMSS or HHMMSS)"""
        parts = filename.split('_')
        
        # Check for full experiment ID (YYYYMMDD_HHMMSS)
        for i, part in enumerate(parts):
            if len(part) == 8 and part.isdigit():  # YYYYMMDD
                if i + 1 < len(parts):
                    time_part = parts[i + 1].split('.')[0]  # Remove extension
                    if len(time_part) == 6 and time_part.isdigit():  # HHMMSS
                        return f"{part}_{time_part}"
        
        # Check for time-only timestamp (HHMMSS) - for subdirectory files
        for part in reversed(parts):  # Check from end as timestamp is usually last
            clean_part = part.split('.')[0]  # Remove extension
            if len(clean_part) == 6 and clean_part.isdigit():  # HHMMSS
                return clean_part
                
        return None
    
    def _find_matching_experiment(self, timestamp: str, experiments: Dict) -> str:
        """Find experiment ID that matches a HHMMSS timestamp"""
        if len(timestamp) == 15:  # Already full experiment ID
            return timestamp
            
        if len(timestamp) == 6:  # HHMMSS format
            # Find experiment with matching time part
            for exp_id in experiments.keys():
                if exp_id.endswith(f"_{timestamp}"):
                    return exp_id
                    
        return None
    
    def print_analysis(self, analysis: Dict):
        """Print analysis of results directory"""
        print(f"\n📊 Results Directory Analysis")
        print(f"{'='*50}")
        print(f"📁 Directory: {self.results_dir}")
        print(f"📄 Total files: {analysis['total_files']}")
        print(f"💾 Total size: {analysis['total_size_mb']:.1f} MB")
        
        if analysis['directories']:
            print(f"\n📂 Subdirectories:")
            for dir_info in analysis['directories']:
                print(f"   {dir_info['name']}: {dir_info['file_count']} files, {dir_info['size_mb']:.1f} MB")
        
        if analysis['by_type']:
            print(f"\n📋 Files by type:")
            for file_type, files in analysis['by_type'].items():
                print(f"   {file_type}: {len(files)} files")
        
        if analysis['experiments']:
            print(f"\n🧪 Experiments (complete with all files):")
            sorted_experiments = sorted(analysis['experiments'].items(), reverse=True)
            for timestamp, files in sorted_experiments[:10]:  # Show last 10
                exp_date = self._format_timestamp(timestamp)
                # Count files by type for this experiment
                root_files = [f for f in files if str(f.parent) == str(self.results_dir)]
                subdir_files = len(files) - len(root_files)
                print(f"   {timestamp} ({exp_date}): {len(files)} files ({len(root_files)} main, {subdir_files} detailed)")
            
            if len(sorted_experiments) > 10:
                print(f"   ... and {len(sorted_experiments) - 10} older experiments")
        
        if analysis.get('orphaned_files'):
            print(f"\n⚠️  Orphaned files (no matching experiment): {len(analysis['orphaned_files'])} files")
            for orphan in analysis['orphaned_files'][:5]:  # Show first 5
                print(f"   - {orphan.name}")
            if len(analysis['orphaned_files']) > 5:
                print(f"   ... and {len(analysis['orphaned_files']) - 5} more")
    
    def _format_timestamp(self, timestamp: str) -> str:
        """Format timestamp for display"""
        try:
            dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return timestamp
    
    def clean_old_experiments(self, days_to_keep: int = 7, dry_run: bool = True) -> List[Path]:
        """Remove experiments older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        files_to_remove = []
        
        analysis = self.analyze_directory()
        
        for experiment_id, files in analysis['experiments'].items():
            if len(experiment_id) == 15:  # Full experiment ID format
                try:
                    exp_date = datetime.strptime(experiment_id, "%Y%m%d_%H%M%S")
                    if exp_date < cutoff_date:
                        files_to_remove.extend(files)
                except ValueError:
                    continue
        
        if not dry_run:
            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                    print(f"🗑️  Deleted: {file_path.name}")
                except Exception as e:
                    print(f"❌ Failed to delete {file_path.name}: {e}")
        
        return files_to_remove
    
    def archive_experiments(self, experiment_ids: List[str], dry_run: bool = True) -> bool:
        """Archive specific experiments to archive/ subdirectory"""
        if not experiment_ids:
            return False
            
        if not dry_run:
            self.archive_dir.mkdir(exist_ok=True)
        
        analysis = self.analyze_directory()
        archived_count = 0
        
        for exp_id in experiment_ids:
            matching_files = analysis['experiments'].get(exp_id, [])
            
            if matching_files:
                if not dry_run:
                    exp_archive_dir = self.archive_dir / exp_id
                    exp_archive_dir.mkdir(exist_ok=True)
                    
                    # Create subdirectories in archive to preserve structure
                    (exp_archive_dir / "raw_responses").mkdir(exist_ok=True)
                    (exp_archive_dir / "prompts").mkdir(exist_ok=True)
                    (exp_archive_dir / "evaluations").mkdir(exist_ok=True)
                    
                    for file_path in matching_files:
                        try:
                            # Determine target directory based on file location
                            if 'raw_responses' in str(file_path):
                                target_dir = exp_archive_dir / "raw_responses"
                            elif 'prompts' in str(file_path):
                                target_dir = exp_archive_dir / "prompts"
                            elif 'evaluations' in str(file_path):
                                target_dir = exp_archive_dir / "evaluations"
                            else:
                                target_dir = exp_archive_dir
                            
                            shutil.move(str(file_path), str(target_dir / file_path.name))
                            print(f"📦 Archived: {file_path.name} → archive/{exp_id}/{target_dir.name if target_dir != exp_archive_dir else ''}")
                        except Exception as e:
                            print(f"❌ Failed to archive {file_path.name}: {e}")
                
                archived_count += len(matching_files)
        
        return archived_count > 0
    
    def clean_duplicates(self, dry_run: bool = True) -> List[Path]:
        """Remove duplicate files (keep most recent)"""
        analysis = self.analyze_directory()
        duplicates_to_remove = []
        
        # Group by experiment and find duplicates within each experiment
        for timestamp, files in analysis['experiments'].items():
            # Group by file type within experiment
            file_groups = defaultdict(list)
            for file_path in files:
                if file_path.name.startswith('full_evaluation_'):
                    file_groups['full_evaluation'].append(file_path)
                elif file_path.name.startswith('batch_'):
                    file_groups['batch'].append(file_path)
                elif 'raw_responses' in str(file_path):
                    # Group raw responses by original filename
                    base_name = '_'.join(file_path.name.split('_')[:-1])  # Remove timestamp
                    file_groups[f'raw_response_{base_name}'].append(file_path)
                elif 'prompts' in str(file_path):
                    # Group prompts by original filename and mode
                    parts = file_path.name.split('_')
                    if len(parts) >= 3:
                        base_key = '_'.join(parts[:-1])  # Remove timestamp
                        file_groups[f'prompt_{base_key}'].append(file_path)
                elif 'evaluations' in str(file_path):
                    file_groups['evaluation'].append(file_path)
            
            # Find duplicates within each group
            for group_name, group_files in file_groups.items():
                if len(group_files) > 1:
                    # Sort by file size (keep largest) and modification time (keep newest)
                    group_files.sort(key=lambda x: (x.stat().st_size, x.stat().st_mtime), reverse=True)
                    duplicates_to_remove.extend(group_files[1:])  # Remove all but the first (best)
        
        if not dry_run:
            for file_path in duplicates_to_remove:
                try:
                    file_path.unlink()
                    print(f"🗑️  Removed duplicate: {file_path.name}")
                except Exception as e:
                    print(f"❌ Failed to remove {file_path.name}: {e}")
        
        return duplicates_to_remove
    
    def interactive_cleanup(self):
        """Interactive cleanup process"""
        analysis = self.analyze_directory()
        self.print_analysis(analysis)
        
        if analysis['total_files'] == 0:
            print("\n✨ Results directory is already clean!")
            return
        
        print(f"\n🧹 Cleanup Options:")
        print(f"1. Archive old experiments (move to archive/ folder)")
        print(f"2. Delete experiments older than X days")
        print(f"3. Remove duplicate files")
        print(f"4. Custom cleanup")
        print(f"5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            self._interactive_archive(analysis)
        elif choice == "2":
            self._interactive_delete_old()
        elif choice == "3":
            self._interactive_remove_duplicates()
        elif choice == "4":
            self._interactive_custom_cleanup(analysis)
        elif choice == "5":
            print("👋 Cleanup cancelled")
        else:
            print("❌ Invalid choice")
    
    def _interactive_archive(self, analysis: Dict):
        """Interactive archiving process"""
        experiments = list(analysis['experiments'].keys())
        if not experiments:
            print("No experiments found to archive")
            return
            
        print(f"\n📦 Available experiments to archive:")
        sorted_exp = sorted(experiments, reverse=True)
        
        for i, exp_id in enumerate(sorted_exp[:20], 1):  # Show last 20
            exp_date = self._format_timestamp(exp_id)
            file_count = len(analysis['experiments'][exp_id])
            print(f"{i:2d}. {exp_id} ({exp_date}) - {file_count} files")
        
        if len(sorted_exp) > 20:
            print(f"    ... and {len(sorted_exp) - 20} older experiments")
        
        selection = input("\nEnter experiment numbers to archive (e.g., 1,3,5-8) or 'all' for all: ").strip()
        
        if selection.lower() == 'all':
            selected_experiments = sorted_exp
        else:
            selected_experiments = self._parse_selection(selection, sorted_exp)
        
        if selected_experiments:
            print(f"\n📋 Will archive {len(selected_experiments)} experiments:")
            for exp_id in selected_experiments:
                print(f"   - {exp_id}")
            
            confirm = input(f"\nProceed with archiving? (y/N): ").strip().lower()
            if confirm == 'y':
                self.archive_experiments(selected_experiments, dry_run=False)
                print(f"✅ Archived {len(selected_experiments)} experiments")
    
    def _interactive_delete_old(self):
        """Interactive deletion of old experiments"""
        days = input("Delete experiments older than how many days? (default: 7): ").strip()
        try:
            days = int(days) if days else 7
        except ValueError:
            print("❌ Invalid number of days")
            return
        
        files_to_delete = self.clean_old_experiments(days, dry_run=True)
        
        if not files_to_delete:
            print(f"✨ No experiments older than {days} days found")
            return
        
        print(f"\n🗑️  Found {len(files_to_delete)} files to delete:")
        for file_path in files_to_delete[:10]:  # Show first 10
            print(f"   - {file_path.name}")
        
        if len(files_to_delete) > 10:
            print(f"   ... and {len(files_to_delete) - 10} more files")
        
        confirm = input(f"\nDelete these {len(files_to_delete)} files? (y/N): ").strip().lower()
        if confirm == 'y':
            self.clean_old_experiments(days, dry_run=False)
            print(f"✅ Deleted {len(files_to_delete)} old files")
    
    def _interactive_remove_duplicates(self):
        """Interactive duplicate removal"""
        duplicates = self.clean_duplicates(dry_run=True)
        
        if not duplicates:
            print("✨ No duplicate files found")
            return
        
        print(f"\n🔍 Found {len(duplicates)} duplicate files:")
        for file_path in duplicates:
            print(f"   - {file_path.name}")
        
        confirm = input(f"\nRemove these {len(duplicates)} duplicates? (y/N): ").strip().lower()
        if confirm == 'y':
            self.clean_duplicates(dry_run=False)
            print(f"✅ Removed {len(duplicates)} duplicate files")
    
    def _interactive_custom_cleanup(self, analysis: Dict):
        """Custom cleanup options"""
        print(f"\n🛠️  Custom Cleanup Options:")
        print(f"1. Clean specific file types")
        print(f"2. Clean specific experiments")
        print(f"3. Clean everything except latest N experiments")
        
        choice = input("Select option (1-3): ").strip()
        
        if choice == "1":
            self._cleanup_by_type(analysis)
        elif choice == "2":
            self._cleanup_specific_experiments(analysis)
        elif choice == "3":
            self._keep_latest_n(analysis)
    
    def _cleanup_by_type(self, analysis: Dict):
        """Clean up by file type"""
        print(f"\n📋 Available file types:")
        for i, (file_type, files) in enumerate(analysis['by_type'].items(), 1):
            print(f"{i}. {file_type}: {len(files)} files")
        
        selection = input("Select types to delete (e.g., 1,2): ").strip()
        selected_types = self._parse_selection(selection, list(analysis['by_type'].keys()))
        
        files_to_delete = []
        for file_type in selected_types:
            files_to_delete.extend(analysis['by_type'][file_type])
        
        if files_to_delete:
            confirm = input(f"Delete {len(files_to_delete)} files? (y/N): ").strip().lower()
            if confirm == 'y':
                for file_path in files_to_delete:
                    try:
                        file_path.unlink()
                        print(f"🗑️  Deleted: {file_path.name}")
                    except Exception as e:
                        print(f"❌ Failed to delete {file_path.name}: {e}")
    
    def _keep_latest_n(self, analysis: Dict):
        """Keep only the latest N experiments"""
        n = input("Keep how many latest experiments? (default: 5): ").strip()
        try:
            n = int(n) if n else 5
        except ValueError:
            print("❌ Invalid number")
            return
        
        # Only consider complete experiments (with full experiment ID)
        complete_experiments = {k: v for k, v in analysis['experiments'].items() if len(k) == 15}
        sorted_experiments = sorted(complete_experiments.items(), reverse=True)
        experiments_to_delete = sorted_experiments[n:]  # Skip first n (latest)
        
        files_to_delete = []
        for timestamp, files in experiments_to_delete:
            files_to_delete.extend(files)
        
        if files_to_delete:
            print(f"\n🗑️  Will delete {len(experiments_to_delete)} old experiments ({len(files_to_delete)} files)")
            print(f"Files include: main reports, raw responses, prompts, and evaluations")
            confirm = input("Proceed? (y/N): ").strip().lower()
            if confirm == 'y':
                for file_path in files_to_delete:
                    try:
                        file_path.unlink()
                        print(f"🗑️  Deleted: {file_path.name}")
                    except Exception as e:
                        print(f"❌ Failed to delete {file_path.name}: {e}")
                print(f"✅ Kept latest {n} experiments")
        else:
            print(f"✨ Already have {len(sorted_experiments)} or fewer experiments")
    
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
    parser = argparse.ArgumentParser(description="Clean up and organize results directory")
    parser.add_argument("--analyze", action="store_true", 
                       help="Only analyze directory without cleanup")
    parser.add_argument("--archive", nargs="*", metavar="EXP_ID",
                       help="Archive specific experiments")
    parser.add_argument("--delete-older-than", type=int, metavar="DAYS",
                       help="Delete experiments older than N days")
    parser.add_argument("--remove-duplicates", action="store_true",
                       help="Remove duplicate files")
    parser.add_argument("--keep-latest", type=int, metavar="N",
                       help="Keep only latest N experiments")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without actually doing it")
    parser.add_argument("--interactive", action="store_true", default=True,
                       help="Interactive cleanup mode (default)")
    
    args = parser.parse_args()
    
    cleanup = ResultsCleanup()
    
    # Non-interactive modes
    if args.analyze:
        analysis = cleanup.analyze_directory()
        cleanup.print_analysis(analysis)
        return
    
    if args.archive is not None:
        if args.archive:  # Specific experiments provided
            cleanup.archive_experiments(args.archive, dry_run=args.dry_run)
        else:
            print("❌ Please specify experiment IDs to archive")
        return
    
    if args.delete_older_than:
        files = cleanup.clean_old_experiments(args.delete_older_than, dry_run=args.dry_run)
        if args.dry_run:
            print(f"Would delete {len(files)} files older than {args.delete_older_than} days")
        return
    
    if args.remove_duplicates:
        duplicates = cleanup.clean_duplicates(dry_run=args.dry_run)
        if args.dry_run:
            print(f"Would remove {len(duplicates)} duplicate files")
        return
    
    if args.keep_latest:
        analysis = cleanup.analyze_directory()
        # Only consider complete experiments (with full experiment ID)
        complete_experiments = {k: v for k, v in analysis['experiments'].items() if len(k) == 15}
        sorted_experiments = sorted(complete_experiments.items(), reverse=True)
        experiments_to_delete = sorted_experiments[args.keep_latest:]
        
        files_to_delete = []
        for timestamp, files in experiments_to_delete:
            files_to_delete.extend(files)
        
        if args.dry_run:
            print(f"Would delete {len(experiments_to_delete)} old experiments ({len(files_to_delete)} files)")
            print(f"Files include: main reports, raw responses, prompts, and evaluations")
        else:
            for file_path in files_to_delete:
                try:
                    file_path.unlink()
                    print(f"🗑️  Deleted: {file_path.name}")
                except Exception as e:
                    print(f"❌ Failed to delete {file_path.name}: {e}")
            print(f"✅ Kept latest {args.keep_latest} experiments")
        return
    
    # Default: interactive mode
    cleanup.interactive_cleanup()


if __name__ == "__main__":
    main()