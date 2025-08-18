#!/usr/bin/env python3
"""
Archive LLM Post-Filter Experiment Results

This script automatically archives the latest experimental results from 
experiments/llm_postfilter/data/llm_results/ to results/llm_postfilter/
organized by experiment timestamp.

Usage:
    python archive_results.py [--dry-run] [--custom-name NAME]
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple

def get_experiment_timestamp(results_dir: Path) -> Optional[datetime]:
    """
    Get the experiment timestamp based on the newest file in the results directory.
    
    Args:
        results_dir: Path to the llm_results directory
        
    Returns:
        datetime of the newest file, or None if directory is empty
    """
    if not results_dir.exists() or not any(results_dir.iterdir()):
        return None
    
    newest_time = 0
    for file_path in results_dir.rglob('*'):
        if file_path.is_file():
            file_time = file_path.stat().st_mtime
            newest_time = max(newest_time, file_time)
    
    return datetime.fromtimestamp(newest_time) if newest_time > 0 else None

def get_archive_name(timestamp: datetime, custom_name: Optional[str] = None) -> str:
    """
    Generate archive directory name.
    
    Args:
        timestamp: Experiment timestamp
        custom_name: Optional custom name for the experiment
        
    Returns:
        Archive directory name
    """
    base_name = timestamp.strftime("%Y%m%d_%H%M%S")
    if custom_name:
        return f"{base_name}_{custom_name}"
    return base_name

def get_experiment_summary(results_dir: Path) -> dict:
    """
    Generate a summary of the experiment results.
    
    Args:
        results_dir: Path to the results directory
        
    Returns:
        Dictionary with experiment summary
    """
    summary = {
        'total_files': 0,
        'total_size_mb': 0,
        'file_types': {},
        'tools': set(),
        'smells': set()
    }
    
    for file_path in results_dir.rglob('*'):
        if file_path.is_file():
            summary['total_files'] += 1
            summary['total_size_mb'] += file_path.stat().st_size / (1024 * 1024)
            
            # Count file types
            suffix = file_path.suffix.lower()
            summary['file_types'][suffix] = summary['file_types'].get(suffix, 0) + 1
            
            # Extract tool and smell info from filename
            filename = file_path.stem
            if '_' in filename:
                parts = filename.split('_')
                if parts[0] in ['chef', 'puppet', 'ansible']:
                    summary['tools'].add(parts[0])
                    
                # Look for smell categories
                smell_keywords = ['hard_coded_secret', 'suspicious_comment', 'weak_cryptography', 'use_of_http_without_ssl_tls']
                for keyword in smell_keywords:
                    if keyword in filename:
                        summary['smells'].add(keyword.replace('_', ' ').title())
    
    summary['tools'] = list(summary['tools'])
    summary['smells'] = list(summary['smells'])
    summary['total_size_mb'] = round(summary['total_size_mb'], 2)
    
    return summary

def create_experiment_metadata(archive_dir: Path, summary: dict, timestamp: datetime):
    """
    Create metadata file for the archived experiment.
    
    Args:
        archive_dir: Path to the archive directory
        summary: Experiment summary
        timestamp: Experiment timestamp
    """
    metadata = {
        'experiment_timestamp': timestamp.isoformat(),
        'archive_created': datetime.now().isoformat(),
        'summary': summary
    }
    
    metadata_file = archive_dir / 'experiment_metadata.json'
    import json
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Also create a human-readable summary
    readme_content = f"""# Experiment Archive: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Timestamp**: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
- **Tools**: {', '.join(summary['tools']) if summary['tools'] else 'None'}
- **Security Smells**: {', '.join(summary['smells']) if summary['smells'] else 'None'}
- **Total Files**: {summary['total_files']}
- **Total Size**: {summary['total_size_mb']} MB

## File Types
"""
    
    for file_type, count in summary['file_types'].items():
        readme_content += f"- **{file_type or 'no extension'}**: {count} files\n"
    
    readme_content += f"""
## Archive Info
- **Archived on**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Source**: experiments/llm_postfilter/data/llm_results/
- **Archive location**: results/llm_postfilter/{archive_dir.name}/
"""
    
    readme_file = archive_dir / 'README.md'
    with open(readme_file, 'w') as f:
        f.write(readme_content)

def archive_experiment_results(
    project_root: Path,
    custom_name: Optional[str] = None,
    dry_run: bool = False
) -> Tuple[bool, str]:
    """
    Archive the latest experiment results.
    
    Args:
        project_root: Path to the project root
        custom_name: Optional custom name for the archive
        dry_run: If True, show what would be done without actually doing it
        
    Returns:
        Tuple of (success, message)
    """
    # Define paths
    results_dir = project_root / "experiments/llm_postfilter/data/llm_results"
    archive_base = project_root / "results/llm_postfilter"
    
    # Check if results directory exists and has content
    if not results_dir.exists():
        return False, f"❌ Results directory not found: {results_dir}"
    
    if not any(results_dir.iterdir()):
        return False, f"❌ No results to archive in: {results_dir}"
    
    # Get experiment timestamp
    timestamp = get_experiment_timestamp(results_dir)
    if not timestamp:
        return False, "❌ Could not determine experiment timestamp"
    
    # Generate archive name
    archive_name = get_archive_name(timestamp, custom_name)
    archive_dir = archive_base / archive_name
    
    # Check if archive already exists
    if archive_dir.exists():
        return False, f"❌ Archive already exists: {archive_dir}"
    
    # Get experiment summary
    summary = get_experiment_summary(results_dir)
    
    if dry_run:
        message = f"""🔍 DRY RUN - Would archive:
📁 Source: {results_dir}
📁 Target: {archive_dir}
📊 Summary:
  - Files: {summary['total_files']}
  - Size: {summary['total_size_mb']} MB
  - Tools: {', '.join(summary['tools']) if summary['tools'] else 'None'}
  - Smells: {', '.join(summary['smells']) if summary['smells'] else 'None'}
  - Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"""
        return True, message
    
    try:
        # Create archive directory
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all files
        for item in results_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, archive_dir)
            elif item.is_dir():
                shutil.copytree(item, archive_dir / item.name)
        
        # Create metadata
        create_experiment_metadata(archive_dir, summary, timestamp)
        
        message = f"""✅ Successfully archived experiment results!
📁 Archive: {archive_dir}
📊 Summary:
  - Files: {summary['total_files']}
  - Size: {summary['total_size_mb']} MB
  - Tools: {', '.join(summary['tools']) if summary['tools'] else 'None'}
  - Smells: {', '.join(summary['smells']) if summary['smells'] else 'None'}
  - Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}

🗂️  Archive structure:
  - results/llm_postfilter/{archive_name}/
  - README.md (human-readable summary)
  - experiment_metadata.json (machine-readable metadata)
  - [experiment files...]

💡 To clear working directory: rm -rf experiments/llm_postfilter/data/llm_results/*"""
        
        return True, message
        
    except Exception as e:
        return False, f"❌ Failed to archive results: {str(e)}"

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Archive LLM Post-Filter experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python archive_results.py                    # Archive with auto-generated name
  python archive_results.py --dry-run          # Show what would be archived
  python archive_results.py --custom-name "baseline_experiment"  # Custom name
        """
    )
    
    parser.add_argument(
        '--dry-run', 
        action='store_true', 
        help='Show what would be archived without actually doing it'
    )
    
    parser.add_argument(
        '--custom-name',
        type=str,
        help='Custom name to append to the archive directory'
    )
    
    args = parser.parse_args()
    
    # Get project root (3 levels up from this script)
    project_root = Path(__file__).parent.parent.parent
    
    print("🗂️  LLM Post-Filter Results Archiver")
    print("=" * 50)
    
    success, message = archive_experiment_results(
        project_root=project_root,
        custom_name=args.custom_name,
        dry_run=args.dry_run
    )
    
    print(message)
    
    if not success:
        sys.exit(1)
    
    if not args.dry_run:
        print(f"\n📋 To view archived experiments:")
        print(f"    ls -la results/llm_postfilter/")
        print(f"\n🧹 To clean working directory after archiving:")
        print(f"    rm -rf experiments/llm_postfilter/data/llm_results/*")

if __name__ == "__main__":
    main()