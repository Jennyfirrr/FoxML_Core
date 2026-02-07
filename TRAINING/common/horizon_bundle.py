# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Horizon Bundle Types

Groups related targets across prediction horizons for multi-horizon training.
This enables training a single model with shared encoder + per-horizon heads.

Usage:
    from TRAINING.common.horizon_bundle import (
        HorizonBundle,
        parse_horizon_from_target,
        create_bundles_from_targets,
        compute_bundle_diversity,
    )

    # Parse horizon from target name
    base_name, horizon = parse_horizon_from_target("fwd_ret_60m")
    # Returns: ("fwd_ret", 60)

    # Create bundles automatically
    targets = ["fwd_ret_5m", "fwd_ret_15m", "fwd_ret_60m"]
    bundles = create_bundles_from_targets(targets)

    # Compute diversity for ranking
    diversity = compute_bundle_diversity(bundle, y_dict)
"""

from __future__ import annotations

import TRAINING.common.repro_bootstrap  # noqa: F401 - must be first (after __future__)

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Minimum valid samples required for diversity calculation
# Below this threshold, target data is considered insufficient
MIN_SAMPLES_FOR_DIVERSITY = 10


@dataclass
class HorizonBundle:
    """
    A bundle of related targets across multiple horizons.

    Attributes:
        base_name: Base target name (e.g., "fwd_ret", "will_peak")
        horizons: List of horizons in minutes (e.g., [5, 15, 60])
        targets: List of target names (e.g., ["fwd_ret_5m", "fwd_ret_15m", "fwd_ret_60m"])
        correlation_matrix: Pairwise correlations between targets (populated during ranking)
        diversity_score: Diversity metric (0-1, higher = more diverse = better for multi-task)
        loss_weights: Per-target loss weights for training
        combined_score: Overall ranking score (diversity + predictability)

    Example:
        bundle = HorizonBundle(
            base_name="fwd_ret",
            horizons=[5, 15, 60],
            targets=["fwd_ret_5m", "fwd_ret_15m", "fwd_ret_60m"]
        )
    """

    base_name: str
    horizons: List[int]
    targets: List[str]

    # Diversity metrics (populated during ranking)
    correlation_matrix: Optional[np.ndarray] = None
    diversity_score: float = 0.0
    combined_score: float = 0.0

    # Training config
    loss_weights: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and set default loss weights."""
        if len(self.horizons) != len(self.targets):
            raise ValueError(
                f"horizons ({len(self.horizons)}) and targets ({len(self.targets)}) "
                "must have the same length"
            )

        if not self.loss_weights:
            # Default: equal weights
            self.loss_weights = {t: 1.0 for t in self.targets}

    @property
    def n_horizons(self) -> int:
        """Number of horizons in the bundle."""
        return len(self.horizons)

    @property
    def primary_horizon(self) -> int:
        """
        Primary horizon for the bundle (middle horizon, or shortest if 2).

        This is used when a single representative horizon is needed.
        """
        if len(self.horizons) == 1:
            return self.horizons[0]
        return sorted(self.horizons)[len(self.horizons) // 2]

    @property
    def primary_target(self) -> str:
        """Target corresponding to the primary horizon."""
        idx = self.horizons.index(self.primary_horizon)
        return self.targets[idx]

    @property
    def min_horizon(self) -> int:
        """Shortest horizon in the bundle."""
        return min(self.horizons)

    @property
    def max_horizon(self) -> int:
        """Longest horizon in the bundle."""
        return max(self.horizons)

    def get_target_for_horizon(self, horizon: int) -> Optional[str]:
        """Get target name for a specific horizon."""
        try:
            idx = self.horizons.index(horizon)
            return self.targets[idx]
        except ValueError:
            return None

    def to_dict(self) -> Dict:
        """Serialize bundle for artifacts/config."""
        return {
            "base_name": self.base_name,
            "horizons": self.horizons,
            "targets": self.targets,
            "diversity_score": self.diversity_score,
            "combined_score": self.combined_score,
            "loss_weights": self.loss_weights,
            "n_horizons": self.n_horizons,
            "primary_horizon": self.primary_horizon,
            "primary_target": self.primary_target,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "HorizonBundle":
        """Deserialize bundle from dict."""
        bundle = cls(
            base_name=data["base_name"],
            horizons=data["horizons"],
            targets=data["targets"],
            loss_weights=data.get("loss_weights", {}),
        )
        bundle.diversity_score = data.get("diversity_score", 0.0)
        bundle.combined_score = data.get("combined_score", 0.0)
        return bundle


def parse_horizon_from_target(target: str) -> Tuple[str, Optional[int]]:
    """
    Extract base name and horizon from target string.

    Handles various target naming conventions:
    - "fwd_ret_60m" → ("fwd_ret", 60)
    - "will_peak_5m_0.8" → ("will_peak", 5)
    - "mdd_15m_0.001" → ("mdd", 15)
    - "custom_target" → ("custom_target", None)

    Args:
        target: Target name string

    Returns:
        (base_name, horizon_minutes) tuple, or (target, None) if not parseable

    Examples:
        >>> parse_horizon_from_target("fwd_ret_60m")
        ("fwd_ret", 60)
        >>> parse_horizon_from_target("will_peak_5m_0.8")
        ("will_peak", 5)
        >>> parse_horizon_from_target("custom_target")
        ("custom_target", None)
    """
    # Pattern: base_name_Nm or base_name_Nm_threshold
    # Handles: fwd_ret_5m, will_peak_5m_0.8, mdd_15m_0.001
    pattern = r"^(.+?)_(\d+)m(?:_[\d.]+)?$"
    match = re.match(pattern, target)

    if match:
        base_name = match.group(1)
        horizon = int(match.group(2))
        return base_name, horizon

    return target, None


def group_targets_by_base(targets: List[str]) -> Dict[str, List[Tuple[str, int]]]:
    """
    Group targets by their base name.

    Args:
        targets: List of target names

    Returns:
        Dict mapping base_name → [(target, horizon), ...], sorted by horizon

    Example:
        >>> targets = ["fwd_ret_5m", "fwd_ret_15m", "will_peak_5m"]
        >>> groups = group_targets_by_base(targets)
        >>> groups["fwd_ret"]
        [("fwd_ret_5m", 5), ("fwd_ret_15m", 15)]
    """
    groups: Dict[str, List[Tuple[str, int]]] = {}

    for target in targets:
        base_name, horizon = parse_horizon_from_target(target)
        if horizon is not None:
            if base_name not in groups:
                groups[base_name] = []
            groups[base_name].append((target, horizon))

    # Sort each group by horizon
    for base_name in groups:
        groups[base_name] = sorted(groups[base_name], key=lambda x: x[1])

    return groups


def create_bundles_from_targets(
    targets: List[str],
    min_horizons: int = 2,
    max_horizons: int = 5,
) -> List[HorizonBundle]:
    """
    Auto-create horizon bundles from target list.

    Args:
        targets: All available targets
        min_horizons: Minimum horizons required for a bundle (default: 2)
        max_horizons: Maximum horizons per bundle (default: 5)

    Returns:
        List of HorizonBundle objects, sorted by base_name for determinism

    Example:
        >>> targets = ["fwd_ret_5m", "fwd_ret_15m", "fwd_ret_60m", "will_peak_5m"]
        >>> bundles = create_bundles_from_targets(targets, min_horizons=2)
        >>> len(bundles)
        1  # Only fwd_ret has >= 2 horizons
    """
    groups = group_targets_by_base(targets)
    bundles = []

    # Sort base_names for determinism
    for base_name in sorted(groups.keys()):
        target_horizons = groups[base_name]

        if len(target_horizons) < min_horizons:
            logger.debug(
                f"Skipping {base_name}: only {len(target_horizons)} horizons "
                f"(min: {min_horizons})"
            )
            continue

        # Limit to max_horizons (take evenly spaced)
        if len(target_horizons) > max_horizons:
            indices = np.linspace(0, len(target_horizons) - 1, max_horizons, dtype=int)
            target_horizons = [target_horizons[i] for i in indices]
            logger.debug(
                f"Limiting {base_name} from {len(groups[base_name])} to "
                f"{max_horizons} horizons"
            )

        bundle = HorizonBundle(
            base_name=base_name,
            horizons=[h for _, h in target_horizons],
            targets=[t for t, _ in target_horizons],
        )
        bundles.append(bundle)

    logger.info(f"Created {len(bundles)} horizon bundles from {len(targets)} targets")
    return bundles


def compute_bundle_diversity(
    bundle: HorizonBundle,
    y_dict: Dict[str, np.ndarray],
) -> float:
    """
    Compute diversity score for a bundle based on target correlations.

    Low correlation = high diversity = good for multi-task learning.
    High correlation means the targets contain redundant information.

    Args:
        bundle: HorizonBundle to score
        y_dict: Dict of target_name → target_values

    Returns:
        Diversity score (0-1, higher = more diverse)

    Note:
        Modifies bundle.correlation_matrix and bundle.diversity_score in place.
    """
    if bundle.n_horizons < 2:
        bundle.diversity_score = 1.0  # Single horizon is trivially "diverse"
        return 1.0

    # Gather target data
    targets_data = []
    valid_targets = []
    for target in bundle.targets:
        if target in y_dict:
            data = y_dict[target].flatten()
            # Filter out NaN values
            valid_mask = ~np.isnan(data)
            if np.sum(valid_mask) > MIN_SAMPLES_FOR_DIVERSITY:  # Need enough data
                targets_data.append(data[valid_mask])
                valid_targets.append(target)

    if len(targets_data) < 2:
        logger.warning(
            f"Bundle {bundle.base_name}: not enough valid targets for diversity calculation"
        )
        bundle.diversity_score = 0.0
        return 0.0

    # Align lengths (use minimum length)
    min_len = min(len(d) for d in targets_data)
    targets_data = [d[:min_len] for d in targets_data]

    # Stack and compute correlation
    try:
        data_matrix = np.column_stack(targets_data)
        corr_matrix = np.corrcoef(data_matrix, rowvar=False)

        bundle.correlation_matrix = corr_matrix

        # Diversity = 1 - mean absolute off-diagonal correlation
        n = corr_matrix.shape[0]
        off_diag_mask = ~np.eye(n, dtype=bool)
        mean_abs_corr = np.mean(np.abs(corr_matrix[off_diag_mask]))

        bundle.diversity_score = 1.0 - mean_abs_corr

    except (ValueError, np.linalg.LinAlgError) as e:
        # ValueError: Invalid data (e.g., all NaN), LinAlgError: Singular matrix
        logger.warning(f"Error computing diversity for {bundle.base_name}: {e}")
        bundle.diversity_score = 0.0
    except TypeError as e:
        # TypeError: Invalid data types
        logger.warning(f"Type error computing diversity for {bundle.base_name}: {e}")
        bundle.diversity_score = 0.0

    return bundle.diversity_score


def compute_horizon_based_weights(
    bundle: HorizonBundle,
    decay_half_life_minutes: float = 30.0,
    primary_horizon: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute loss weights based on horizon distance from primary.

    Shorter horizons (closer to primary) get higher weights.

    Args:
        bundle: HorizonBundle to compute weights for
        decay_half_life_minutes: Half-life for exponential decay
        primary_horizon: Reference horizon (default: bundle.primary_horizon)

    Returns:
        Dict of target_name → weight

    Example:
        >>> bundle = HorizonBundle(
        ...     base_name="fwd_ret",
        ...     horizons=[5, 15, 60],
        ...     targets=["fwd_ret_5m", "fwd_ret_15m", "fwd_ret_60m"]
        ... )
        >>> weights = compute_horizon_based_weights(bundle, decay_half_life_minutes=30)
        >>> # fwd_ret_15m (primary) gets highest weight
    """
    if primary_horizon is None:
        primary_horizon = bundle.primary_horizon

    weights = {}
    for target, horizon in zip(bundle.targets, bundle.horizons):
        # Distance from primary horizon
        distance = abs(horizon - primary_horizon)
        # Exponential decay
        weight = np.exp(-np.log(2) * distance / decay_half_life_minutes)
        weights[target] = float(weight)

    # Normalize so sum = n_horizons
    total = sum(weights.values())
    if total > 0:
        scale = bundle.n_horizons / total
        weights = {t: w * scale for t, w in weights.items()}

    bundle.loss_weights = weights
    return weights
