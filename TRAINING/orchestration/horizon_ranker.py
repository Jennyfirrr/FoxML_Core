# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Horizon Bundle Ranker

Ranks bundles by diversity and predictability for multi-horizon training.
Provides the full pipeline from target list to ranked, filtered bundles.

Usage:
    from TRAINING.orchestration.horizon_ranker import select_top_bundles

    bundles = select_top_bundles(
        targets=all_targets,
        y_dict=target_values,
        target_scores=predictability_scores,
        top_n=3,
        min_diversity=0.3
    )
"""

from __future__ import annotations

import TRAINING.common.repro_bootstrap  # noqa: F401 - must be first (after __future__)

import logging
from typing import Dict, List, Optional

import numpy as np

from TRAINING.common.horizon_bundle import (
    HorizonBundle,
    compute_bundle_diversity,
    create_bundles_from_targets,
)

logger = logging.getLogger(__name__)


def rank_bundles(
    bundles: List[HorizonBundle],
    y_dict: Dict[str, np.ndarray],
    target_scores: Optional[Dict[str, float]] = None,
    diversity_weight: float = 0.3,
    predictability_weight: float = 0.7,
) -> List[HorizonBundle]:
    """
    Rank horizon bundles by combined diversity and predictability.

    The ranking formula is:
        combined_score = diversity_weight * diversity + predictability_weight * predictability

    High diversity means targets are uncorrelated (good for multi-task learning).
    High predictability means targets have good signal-to-noise ratio.

    Args:
        bundles: List of bundles to rank
        y_dict: Target values for diversity calculation
        target_scores: Optional dict of target → predictability score (0-1)
        diversity_weight: Weight for diversity in ranking (default: 0.3)
        predictability_weight: Weight for predictability in ranking (default: 0.7)

    Returns:
        Bundles sorted by combined score (best first)
    """
    for bundle in bundles:
        # Compute diversity
        diversity = compute_bundle_diversity(bundle, y_dict)

        # Compute average predictability (if scores provided)
        if target_scores:
            predictabilities = [
                target_scores.get(t, 0.5) for t in bundle.targets  # 0.5 = neutral
            ]
            predictability = np.mean(predictabilities) if predictabilities else 0.5
        else:
            predictability = 0.5  # Neutral if no scores

        # Combined score
        bundle.combined_score = (
            diversity_weight * diversity + predictability_weight * predictability
        )

        logger.debug(
            f"Bundle {bundle.base_name}: diversity={diversity:.3f}, "
            f"predictability={predictability:.3f}, combined={bundle.combined_score:.3f}"
        )

    # Sort by combined score (descending)
    # DETERMINISM: Use base_name as tie-breaker for equal scores
    return sorted(bundles, key=lambda b: (-b.combined_score, b.base_name))


def filter_bundles_by_diversity(
    bundles: List[HorizonBundle],
    min_diversity: float = 0.3,
) -> List[HorizonBundle]:
    """
    Filter out bundles with low diversity (too correlated).

    Bundles with highly correlated targets don't benefit from multi-task learning,
    as the model can't learn distinct features for each horizon.

    Args:
        bundles: Bundles to filter
        min_diversity: Minimum diversity score (default: 0.3)

    Returns:
        Bundles with diversity >= min_diversity
    """
    filtered = [b for b in bundles if b.diversity_score >= min_diversity]

    if len(filtered) < len(bundles):
        removed = len(bundles) - len(filtered)
        logger.info(
            f"Filtered {removed} bundles with diversity < {min_diversity}"
        )

    return filtered


def select_top_bundles(
    targets: List[str],
    y_dict: Dict[str, np.ndarray],
    target_scores: Optional[Dict[str, float]] = None,
    top_n: int = 3,
    min_diversity: float = 0.3,
    min_horizons: int = 2,
    max_horizons: int = 5,
    diversity_weight: float = 0.3,
    predictability_weight: float = 0.7,
) -> List[HorizonBundle]:
    """
    Full bundle selection pipeline.

    Steps:
    1. Create bundles from targets (group by base name)
    2. Compute diversity scores
    3. Rank by combined diversity + predictability
    4. Filter by minimum diversity
    5. Return top N

    Args:
        targets: All available targets
        y_dict: Dict of target_name → target_values
        target_scores: Optional predictability scores from ranking pipeline
        top_n: Number of bundles to return (default: 3)
        min_diversity: Minimum diversity threshold (default: 0.3)
        min_horizons: Minimum horizons per bundle (default: 2)
        max_horizons: Maximum horizons per bundle (default: 5)
        diversity_weight: Weight for diversity in ranking (default: 0.3)
        predictability_weight: Weight for predictability in ranking (default: 0.7)

    Returns:
        Top N bundles, ranked by quality
    """
    # Create bundles
    bundles = create_bundles_from_targets(
        targets, min_horizons=min_horizons, max_horizons=max_horizons
    )

    if not bundles:
        logger.warning(
            f"No horizon bundles found in {len(targets)} targets "
            f"(min_horizons={min_horizons})"
        )
        return []

    logger.info(
        f"Created {len(bundles)} horizon bundles from {len(targets)} targets"
    )

    # Rank bundles
    ranked = rank_bundles(
        bundles,
        y_dict,
        target_scores,
        diversity_weight=diversity_weight,
        predictability_weight=predictability_weight,
    )

    # Filter by diversity
    filtered = filter_bundles_by_diversity(ranked, min_diversity)

    # Return top N
    result = filtered[:top_n]

    if result:
        logger.info(f"Selected {len(result)} bundles for multi-horizon training:")
        for i, bundle in enumerate(result):
            logger.info(
                f"  {i + 1}. {bundle.base_name}: {bundle.n_horizons} horizons "
                f"({bundle.horizons}), diversity={bundle.diversity_score:.3f}, "
                f"combined={bundle.combined_score:.3f}"
            )

    return result


def get_all_bundle_targets(bundles: List[HorizonBundle]) -> List[str]:
    """
    Get all unique targets from a list of bundles.

    Args:
        bundles: List of HorizonBundle objects

    Returns:
        Sorted list of unique target names
    """
    all_targets = set()
    for bundle in bundles:
        all_targets.update(bundle.targets)
    return sorted(all_targets)


def validate_bundles_for_training(
    bundles: List[HorizonBundle],
    y_dict: Dict[str, np.ndarray],
) -> List[HorizonBundle]:
    """
    Validate that bundles have all required data for training.

    Args:
        bundles: Bundles to validate
        y_dict: Available target data

    Returns:
        List of valid bundles (with all targets present in y_dict)
    """
    valid_bundles = []

    for bundle in bundles:
        missing = [t for t in bundle.targets if t not in y_dict]
        if missing:
            logger.warning(
                f"Bundle {bundle.base_name} missing targets: {missing}. Skipping."
            )
            continue

        # Check for sufficient data
        min_samples = min(len(y_dict[t]) for t in bundle.targets)
        if min_samples < 100:
            logger.warning(
                f"Bundle {bundle.base_name} has only {min_samples} samples. Skipping."
            )
            continue

        valid_bundles.append(bundle)

    return valid_bundles
