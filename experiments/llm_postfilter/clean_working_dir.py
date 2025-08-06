#!/usr/bin/env python3
"""
Clean LLM Post-Filter Working Directory

This script safely cleans the working results directory after archiving.
It preserves the directory structure but removes all result files.

Usage:
    python clean_working_dir.py [--dry-run]
"""

import argparse
import shutil
from pathlib import Path

def clean_working_directory(project_root: Path, dry_run: bool = False) -> tuple[bool, str]:
    """
    Clean the working results directory.
    
    Args:
        project_root: Path to the project root
        dry_run: If True, show what would be done without actually doing it
        
    Returns:
        Tuple of (success, message)
    """
    working_dir = project_root / "experiments/llm_postfilter/data/llm_results"
    
    if not working_dir.exists():
        return False, f"❌ Working directory not found: {working_dir}"
    
    # Count files to be removed
    files_to_remove = []
    dirs_to_remove = []
    
    for item in working_dir.rglob('*'):
        if item.is_file():
            files_to_remove.append(item)
        elif item.is_dir() and item != working_dir:
            dirs_to_remove.append(item)
    
    if not files_to_remove and not dirs_to_remove:
        return True, "✅ Working directory is already clean"
    
    if dry_run:
        message = f"""🔍 DRY RUN - Would clean:
📁 Directory: {working_dir}
📄 Files to remove: {len(files_to_remove)}
📁 Subdirectories to remove: {len(dirs_to_remove)}

Files:"""
        for file_path in files_to_remove[:10]:  # Show first 10 files
            message += f"\n  - {file_path.name}"
        if len(files_to_remove) > 10:
            message += f"\n  - ... and {len(files_to_remove) - 10} more files"
        
        return True, message
    
    try:
        # Remove all contents but keep the directory
        for item in working_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        
        message = f"""✅ Working directory cleaned successfully!
📁 Directory: {working_dir}
📄 Files removed: {len(files_to_remove)}
📁 Subdirectories removed: {len(dirs_to_remove)}

💡 Ready for next experiment!"""
        
        return True, message
        
    except Exception as e:
        return False, f"❌ Failed to clean working directory: {str(e)}"

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Clean LLM Post-Filter working directory",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--dry-run', 
        action='store_true', 
        help='Show what would be cleaned without actually doing it'
    )
    
    args = parser.parse_args()
    
    # Get project root (3 levels up from this script)
    project_root = Path(__file__).parent.parent.parent
    
    print("🧹 LLM Post-Filter Working Directory Cleaner")
    print("=" * 50)
    
    success, message = clean_working_directory(
        project_root=project_root,
        dry_run=args.dry_run
    )
    
    print(message)
    
    if not success:
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())