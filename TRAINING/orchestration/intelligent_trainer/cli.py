# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Intelligent Trainer CLI

Command-line interface parsing for intelligent training orchestrator.
"""

import argparse
from pathlib import Path
from typing import List, Optional


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for intelligent trainer CLI.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description='Intelligent Training Orchestrator with Target Ranking and Feature Selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-select top 5 targets and top 100 features per target
  python -m TRAINING.orchestration.intelligent_trainer \\
      --data-dir data/data_labeled/interval=5m \\
      --symbols AAPL MSFT GOOGL \\
      --auto-targets --top-n-targets 5 \\
      --auto-features --top-m-features 100

  # Manual targets, auto features
  python -m TRAINING.orchestration.intelligent_trainer \\
      --data-dir data/data_labeled/interval=5m \\
      --symbols AAPL MSFT \\
      --targets fwd_ret_5m fwd_ret_15m \\
      --auto-features --top-m-features 50

  # Use cached rankings (faster)
  python -m TRAINING.orchestration.intelligent_trainer \\
      --data-dir data/data_labeled/interval=5m \\
      --symbols AAPL MSFT \\
      --auto-targets --top-n-targets 5 \\
      --no-refresh-cache
        """
    )

    # Core arguments (now optional - can come from config)
    parser.add_argument('--data-dir', type=Path, required=False,
                       help='Data directory (overrides config, required if not in config)')
    parser.add_argument('--symbols', nargs='+', required=False,
                       help='Symbols to train on (overrides config, required if not in config)')
    parser.add_argument('--output-dir', type=Path, required=False,
                       help='Output directory (overrides config, default: intelligent_output)')
    parser.add_argument('--cache-dir', type=Path,
                       help='Cache directory (overrides config, default: output_dir/cache)')

    # Simple config-based mode
    parser.add_argument('--config', type=str,
                       help='Config profile name (loads from CONFIG/pipeline/training/intelligent.yaml)')

    # Target/feature selection (moved to config - CLI only for manual overrides)
    parser.add_argument('--targets', nargs='+',
                       help='Manual target list (overrides config auto_targets)')
    parser.add_argument('--features', nargs='+',
                       help='Manual feature list (overrides config auto_features)')

    # Training arguments (moved to config - CLI only for manual overrides)
    parser.add_argument('--families', nargs='+',
                       help='Model families to train (overrides config)')

    # Quick presets
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode: 3 targets, 50 features, limited evaluation')
    parser.add_argument('--full', action='store_true',
                       help='Full production mode: all defaults from config')

    # Testing/debugging overrides (use sparingly - prefer config)
    parser.add_argument('--override-max-samples', type=int,
                       help='OVERRIDE: Max samples per symbol (testing only, overrides config)')
    parser.add_argument('--override-max-rows', type=int,
                       help='OVERRIDE: Max rows per symbol (testing only, overrides config)')

    # Cache control
    parser.add_argument('--force-refresh', action='store_true',
                       help='Force refresh of cached rankings/selections')
    parser.add_argument('--no-refresh-cache', action='store_true',
                       help='Never refresh cache (use existing only)')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching entirely')

    # Config files
    parser.add_argument('--target-ranking-config', type=Path,
                       help='Path to target ranking config YAML (default: CONFIG/ranking/targets/configs.yaml)')
    parser.add_argument('--multi-model-config', type=Path,
                       help='Path to multi-model feature selection config YAML (default: CONFIG/multi_model_feature_selection.yaml)')
    parser.add_argument('--experiment-config', type=str,
                       help='Experiment config name (without .yaml) from CONFIG/experiments/ [NEW - preferred]')

    # Decision application
    parser.add_argument('--apply-decisions', type=str, choices=['off', 'dry_run', 'apply'], default='off',
                       help='Decision application mode: off (assist mode), dry_run (show patch without applying), apply (auto-apply patches)')

    return parser


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command line arguments.

    Args:
        args: Optional list of arguments (for testing). If None, uses sys.argv.

    Returns:
        Parsed argument namespace
    """
    parser = create_argument_parser()
    return parser.parse_args(args)
