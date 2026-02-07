# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Persistence operations for reproducibility tracking.

Provides functions for:
- Loading previous run data from logs and indices
- Saving run data to cohort directories
- Atomic file writes with locking

Note: The full implementation is currently in the parent reproducibility_tracker.py
file. This module provides the interface and will be fully extracted in a future phase.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def load_previous_run(
    log_file: Path,
    stage: str,
    target: str,
    search_previous_runs: bool = False,
    output_dir: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """Load the previous run's summary for a stage/item combination.

    This is a delegating wrapper that calls the method on ReproducibilityTracker.

    Args:
        log_file: Path to the reproducibility log file
        stage: Pipeline stage name (e.g., "target_ranking", "feature_selection")
        target: Name of the item (e.g., target name, symbol name)
        search_previous_runs: If True, search parent directories for previous runs
        output_dir: Output directory (required if search_previous_runs is True)

    Returns:
        Dictionary with previous run results, or None if no previous run exists
    """
    from ..reproducibility_tracker import ReproducibilityTracker

    # Create a tracker instance
    if output_dir is None:
        output_dir = log_file.parent

    tracker = ReproducibilityTracker.__new__(ReproducibilityTracker)
    tracker.log_file = log_file
    tracker.output_dir = output_dir
    tracker.search_previous_runs = search_previous_runs

    return tracker.load_previous_run(stage, target)


def save_run(
    log_file: Path,
    output_dir: Path,
    stage: str,
    target: str,
    metrics: Dict[str, Any],
    additional_data: Optional[Dict[str, Any]] = None,
    max_runs_per_item: int = 10
) -> None:
    """Save the current run's summary to the reproducibility log.

    This is a delegating wrapper that calls the method on ReproducibilityTracker.

    Args:
        log_file: Path to the reproducibility log file
        output_dir: Output directory for the log
        stage: Pipeline stage name
        target: Name of the item (e.g., target name)
        metrics: Dictionary of metrics to track
        additional_data: Optional additional data to store with the run
        max_runs_per_item: Maximum number of runs to keep per item
    """
    from ..reproducibility_tracker import ReproducibilityTracker

    # Create a tracker instance
    tracker = ReproducibilityTracker.__new__(ReproducibilityTracker)
    tracker.log_file = log_file
    tracker.output_dir = output_dir
    tracker.max_runs_per_item = max_runs_per_item

    return tracker.save_run(stage, target, metrics, additional_data)


def read_log_file(log_file: Path) -> Dict[str, Any]:
    """Read a reproducibility log file.

    Args:
        log_file: Path to the log file

    Returns:
        Dictionary with all runs data, or empty dict if file doesn't exist
    """
    if not log_file.exists():
        return {}

    try:
        with open(log_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.debug(f"Could not read log file {log_file}: {e}")
        return {}


def get_run_history(
    log_file: Path,
    stage: str,
    target: str
) -> List[Dict[str, Any]]:
    """Get all historical runs for a stage/target combination.

    Args:
        log_file: Path to the log file
        stage: Pipeline stage name
        target: Target name

    Returns:
        List of run dictionaries, sorted by timestamp (newest first)
    """
    all_runs = read_log_file(log_file)
    key = f"{stage}:{target}"

    runs = all_runs.get(key, [])
    runs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

    return runs


__all__ = [
    'load_previous_run',
    'save_run',
    'read_log_file',
    'get_run_history',
]
