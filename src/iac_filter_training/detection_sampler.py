"""
IaC Detection Sampler for IaC Filter Training

This module samples GLITCH detections upfront to get exactly the target number of samples
for each IaC technology while preserving natural smell proportions.

Supports Chef, Ansible, and Puppet IaC technologies.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IaCDetectionSampler:
    """Samples GLITCH detections to meet exact target counts per technology."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data" / "iac_filter_training"

        # Target security smells for pseudo-labeling (same for all IaC technologies)
        self.target_smells = [
            'hardcoded-secret',
            'suspicious comment',
            'weak cryptography algorithms',
            'use of http'
        ]

        # Mapping for display names
        self.smell_display_names = {
            'hardcoded-secret': 'Hard-coded secret',
            'suspicious comment': 'Suspicious comment',
            'weak cryptography algorithms': 'Use of weak cryptography algorithms',
            'use of http': 'Use of HTTP without SSL/TLS'
        }

        # Target sample sizes per technology (based on test set sizes and 8:1:1 ratio)
        # Total = train + val = 8x + 1x test size
        self.target_totals = {
            'chef': 675,    # 600 train + 75 val (75 test * 9)
            'puppet': 963,  # 856 train + 107 val (107 test * 9)
            'ansible': 432  # 384 train + 48 val (48 test * 9)
        }

        # Fixed target breakdowns - adjusted to match exact totals
        self.target_breakdowns = {
            'chef': {
                'hardcoded-secret': 360,  # Adjusted from 361 to reach exactly 675
                'suspicious comment': 202,
                'weak cryptography algorithms': 7,
                'use of http': 106
            },
            'puppet': {
                'hardcoded-secret': 652,  # Adjusted from 651 to reach exactly 963
                'suspicious comment': 235,
                'weak cryptography algorithms': 11,
                'use of http': 65
            },
            'ansible': {
                'hardcoded-secret': 321,
                'suspicious comment': 76,
                'weak cryptography algorithms': 11,
                'use of http': 24
            }
        }

        # IaC technology specific configurations
        self.iac_config = {
            'chef': {
                'glitch_file': 'GLITCH-chef-oracle.csv'
            },
            'ansible': {
                'glitch_file': 'GLITCH-ansible-oracle.csv'
            },
            'puppet': {
                'glitch_file': 'GLITCH-puppet-oracle.csv'
            }
        }

    def load_glitch_detections(self, iac_tech: str) -> pd.DataFrame:
        """Load GLITCH detections from the IaC oracle dataset."""
        glitch_filename = self.iac_config[iac_tech]['glitch_file']
        glitch_file = self.data_dir / glitch_filename

        if not glitch_file.exists():
            raise FileNotFoundError(f"GLITCH detection file not found: {glitch_file}")

        glitch_df = pd.read_csv(glitch_file)
        logger.info(f"Loaded {len(glitch_df)} GLITCH detections for {iac_tech}")

        # Filter for target smells only
        glitch_filtered = glitch_df[glitch_df['ERROR'].isin(self.target_smells)].copy()
        logger.info(f"Filtered to {len(glitch_filtered)} target smell detections")

        return glitch_filtered

    def get_available_counts(self, df: pd.DataFrame, iac_tech: str) -> Dict[str, int]:
        """Get actual available counts per smell in the dataset."""
        available = {}
        for smell in self.target_smells:
            count = len(df[df['ERROR'] == smell])
            available[smell] = count
            logger.info(f"  {iac_tech} {smell}: {count} available")

        return available

    def calculate_proportional_targets(self, df: pd.DataFrame, iac_tech: str) -> Dict[str, int]:
        """Calculate proportional target counts based on available data."""
        available = self.get_available_counts(df, iac_tech)
        target_total = self.target_totals[iac_tech]

        # Calculate total available target smells
        total_available = sum(available[smell] for smell in self.target_smells)

        targets = {}
        allocated = 0

        # Calculate proportional allocation for each smell
        for smell in self.target_smells:
            if total_available == 0:
                targets[smell] = 0
                continue

            proportion = available[smell] / total_available
            target_count = round(target_total * proportion)
            targets[smell] = min(target_count, available[smell])  # Don't exceed available
            allocated += targets[smell]

        # Adjust to meet exact total (distribute remainder to largest categories)
        remainder = target_total - allocated
        if remainder != 0:
            # Sort smells by available count descending
            sorted_smells = sorted(self.target_smells, key=lambda s: available[s], reverse=True)
            for smell in sorted_smells:
                if remainder == 0:
                    break
                if targets[smell] < available[smell]:
                    targets[smell] += 1
                    remainder -= 1
                    allocated += 1

        return targets

    def sample_detections(self, df: pd.DataFrame, iac_tech: str) -> pd.DataFrame:
        """Sample detections to meet exact target counts."""
        targets = self.target_breakdowns[iac_tech]
        available = self.get_available_counts(df, iac_tech)

        logger.info(f"\n{iac_tech.upper()} Target Breakdown:")
        for smell in self.target_smells:
            target = targets[smell]
            avail = available[smell]
            logger.info(f"  {smell}: {target} (available: {avail})")
            if target > avail:
                logger.warning(f"  ⚠ Target {target} > available {avail} for {smell}")

        sampled_dfs = []

        for smell in self.target_smells:
            target_count = targets[smell]
            available_count = available[smell]

            # Use available count if less than target
            sample_count = min(target_count, available_count)

            if sample_count == 0:
                logger.warning(f"No samples available for {iac_tech} {smell}")
                continue

            smell_df = df[df['ERROR'] == smell].copy()
            sampled = smell_df.sample(n=sample_count, random_state=42)
            sampled_dfs.append(sampled)

            logger.info(f"Sampled {sample_count} {smell} detections for {iac_tech}")

        # Combine all sampled detections
        if sampled_dfs:
            result = pd.concat(sampled_dfs, ignore_index=True)
            # Sort by PATH for consistency
            result = result.sort_values(['PATH', 'LINE']).reset_index(drop=True)
            return result
        else:
            return pd.DataFrame()

    def verify_sample_counts(self, sampled_df: pd.DataFrame, iac_tech: str, targets: Dict[str, int]) -> bool:
        """Verify that sampled counts match targets."""
        actual_counts = sampled_df['ERROR'].value_counts().to_dict()

        logger.info(f"\n{iac_tech.upper()} Sampling Results:")
        total_actual = 0

        for smell in self.target_smells:
            target = targets[smell]
            actual = actual_counts.get(smell, 0)
            total_actual += actual

            if actual == target:
                logger.info(f"  ✓ {smell}: {actual}/{target}")
            else:
                logger.warning(f"  ⚠ {smell}: {actual}/{target} (mismatch)")

        expected_total = self.target_totals[iac_tech]
        if total_actual == expected_total:
            logger.info(f"  ✓ Total: {total_actual}/{expected_total}")
            return True
        else:
            logger.warning(f"  ⚠ Total: {total_actual}/{expected_total} (mismatch)")
            return False

    def save_sampled_detections(self, sampled_df: pd.DataFrame, iac_tech: str) -> Path:
        """Save sampled detections to CSV."""
        output_file = self.data_dir / f"GLITCH-{iac_tech}-oracle_sampled.csv"
        sampled_df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(sampled_df)} sampled detections to {output_file}")
        return output_file

    def run_single_tech(self, iac_tech: str) -> Dict:
        """Run sampling for a single IaC technology."""
        logger.info(f"\n=== Sampling {iac_tech.upper()} detections ===")

        # Load detections
        glitch_df = self.load_glitch_detections(iac_tech)

        # Get fixed targets
        targets = self.target_breakdowns[iac_tech]

        # Sample detections
        sampled_df = self.sample_detections(glitch_df, iac_tech)

        if sampled_df.empty:
            logger.error(f"No samples generated for {iac_tech}")
            return {'success': False}

        # Verify counts
        counts_match = self.verify_sample_counts(sampled_df, iac_tech, targets)

        # Save results
        output_file = self.save_sampled_detections(sampled_df, iac_tech)

        return {
            'success': True,
            'counts_match': counts_match,
            'output_file': output_file,
            'total_samples': len(sampled_df),
            'targets': targets
        }

    def run_all(self) -> Dict[str, Dict]:
        """Run sampling for all IaC technologies."""
        results = {}

        for iac_tech in ['chef', 'ansible', 'puppet']:
            results[iac_tech] = self.run_single_tech(iac_tech)

        # Summary
        logger.info("\n=== SAMPLING SUMMARY ===")
        total_samples = 0
        all_match = True

        for iac_tech, result in results.items():
            if result['success']:
                samples = result['total_samples']
                total_samples += samples
                match_status = "✓" if result['counts_match'] else "⚠"
                logger.info(f"{match_status} {iac_tech}: {samples} samples")
                if not result['counts_match']:
                    all_match = False
            else:
                logger.error(f"✗ {iac_tech}: Failed")
                all_match = False

        expected_total = sum(self.target_totals[tech] for tech in ['chef', 'ansible', 'puppet'])
        logger.info(f"\nTotal: {total_samples}/{expected_total} samples")

        return {
            'results': results,
            'total_samples': total_samples,
            'expected_total': expected_total,
            'all_counts_match': all_match
        }


def main():
    """Main function to sample IaC detections."""
    parser = argparse.ArgumentParser(description="Sample IaC detections for training")
    parser.add_argument("--iac-tech", type=str, choices=["chef", "ansible", "puppet"],
                       help="IaC technology to process (default: all)")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only verify current sampling without re-sampling")

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    sampler = IaCDetectionSampler(project_root)

    if args.iac_tech:
        logger.info(f"Sampling detections for {args.iac_tech}")
        result = sampler.run_single_tech(args.iac_tech)
        if result['success']:
            logger.info("Sampling completed successfully!")
        else:
            logger.error("Sampling failed!")
    else:
        logger.info("Sampling detections for all technologies")
        summary = sampler.run_all()
        if summary['all_counts_match']:
            logger.info("All sampling targets met!")
        else:
            logger.warning("Some sampling targets not met - check warnings above")


if __name__ == "__main__":
    main()
