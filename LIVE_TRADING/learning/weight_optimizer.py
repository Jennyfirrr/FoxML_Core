"""
Ensemble Weight Optimizer
=========================

Combines bandit learning with static blending weights.

The optimizer blends:
- Static weights (e.g., from ridge regression on historical data)
- Bandit-learned weights (from online P&L feedback)

Final weights = (1 - blend_ratio) * static + blend_ratio * bandit

SST Compliance:
- Uses get_cfg() for configuration
- Deterministic weight calculations
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from CONFIG.config_loader import get_cfg
from LIVE_TRADING.learning.bandit import Exp3IXBandit

logger = logging.getLogger(__name__)

# Default parameters
DEFAULT_BLEND_RATIO = 0.3  # 30% bandit, 70% static
DEFAULT_MIN_SAMPLES = 100  # Minimum trades before bandit influence


class EnsembleWeightOptimizer:
    """
    Combines bandit learning with static blending weights.

    The optimizer provides a smooth transition from static weights
    (based on historical performance) to bandit-adapted weights
    (based on live P&L feedback).

    Key features:
    - Blend ratio controls bandit influence
    - Minimum samples guard prevents early instability
    - Gradual ramp-up of bandit influence

    Example:
        >>> arm_names = ["5m", "10m", "15m"]
        >>> optimizer = EnsembleWeightOptimizer(arm_names=arm_names)
        >>> static_weights = {"5m": 0.3, "10m": 0.4, "15m": 0.3}
        >>> final = optimizer.get_ensemble_weights(static_weights)
        >>> # Early on: final ≈ static_weights
        >>> # After many trades: final blends bandit-learned weights
    """

    def __init__(
        self,
        arm_names: List[str],
        bandit: Optional[Exp3IXBandit] = None,
        blend_ratio: Optional[float] = None,
        min_samples: Optional[int] = None,
        ramp_up_samples: Optional[int] = None,
    ) -> None:
        """
        Initialize ensemble weight optimizer.

        Args:
            arm_names: Names of arms (horizons, models, etc.)
            bandit: Exp3IXBandit instance (created if None)
            blend_ratio: Final blend ratio for bandit weights (default: 0.3)
            min_samples: Minimum samples before bandit has influence
            ramp_up_samples: Samples over which to ramp up bandit influence
        """
        self._arm_names = list(arm_names)
        self._n_arms = len(arm_names)

        if self._n_arms < 2:
            raise ValueError(f"Need at least 2 arms, got {self._n_arms}")

        # Load config parameters
        self._blend_ratio = blend_ratio if blend_ratio is not None else get_cfg(
            "live_trading.online_learning.blend_ratio", default=DEFAULT_BLEND_RATIO
        )
        self._min_samples = min_samples if min_samples is not None else get_cfg(
            "live_trading.online_learning.min_samples", default=DEFAULT_MIN_SAMPLES
        )
        self._ramp_up_samples = ramp_up_samples if ramp_up_samples is not None else get_cfg(
            "live_trading.online_learning.ramp_up_samples", default=self._min_samples
        )

        # Create name to index mapping
        self._name_to_idx: Dict[str, int] = {
            name: i for i, name in enumerate(arm_names)
        }

        # Create or use provided bandit
        if bandit is not None:
            if bandit.n_arms != self._n_arms:
                raise ValueError(
                    f"Bandit has {bandit.n_arms} arms but optimizer has {self._n_arms}"
                )
            self._bandit = bandit
        else:
            self._bandit = Exp3IXBandit(
                n_arms=self._n_arms,
                arm_names=self._arm_names.copy(),  # Copy to avoid shared mutation
            )

        logger.info(
            f"EnsembleWeightOptimizer initialized: {self._n_arms} arms, "
            f"blend_ratio={self._blend_ratio:.2f}, min_samples={self._min_samples}"
        )

    def get_ensemble_weights(
        self,
        static_weights: Dict[str, float],
        force_static: bool = False,
    ) -> Dict[str, float]:
        """
        Get blended ensemble weights.

        Blends static weights with bandit-learned weights:
            final = (1 - effective_blend) * static + effective_blend * bandit

        The effective blend ratio ramps up from 0 after min_samples
        to full blend_ratio after min_samples + ramp_up_samples.

        Args:
            static_weights: Static weights from ridge regression or similar
            force_static: If True, return only static weights (for debugging)

        Returns:
            Dict mapping arm name to blended weight (sum to 1.0)
        """
        # Validate static weights
        validated_static = self._validate_and_normalize_weights(static_weights)

        if force_static:
            return validated_static

        # Get bandit weights
        bandit_weights_array = self._bandit.get_weights()
        bandit_weights = {
            name: float(bandit_weights_array[i])
            for i, name in enumerate(self._arm_names)
        }

        # Calculate effective blend ratio based on samples
        effective_blend = self._calculate_effective_blend()

        if effective_blend == 0:
            return validated_static

        # Blend weights
        blended = {}
        for name in self._arm_names:
            static_w = validated_static.get(name, 1.0 / self._n_arms)
            bandit_w = bandit_weights.get(name, 1.0 / self._n_arms)
            blended[name] = (1 - effective_blend) * static_w + effective_blend * bandit_w

        # Normalize to ensure sum to 1.0
        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}

        logger.debug(
            f"Ensemble blend: effective_ratio={effective_blend:.3f}, "
            f"bandit_steps={self._bandit.total_steps}"
        )

        return blended

    def _validate_and_normalize_weights(
        self,
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """Validate and normalize static weights."""
        # Fill missing arms with uniform weight
        result = {}
        for name in self._arm_names:
            w = weights.get(name, 1.0 / self._n_arms)
            result[name] = max(0, w)  # Ensure non-negative

        # Normalize
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}
        else:
            result = {name: 1.0 / self._n_arms for name in self._arm_names}

        return result

    def _calculate_effective_blend(self) -> float:
        """
        Calculate effective blend ratio based on samples.

        Returns 0 if below min_samples, then ramps up linearly
        to blend_ratio over ramp_up_samples.

        Returns:
            Effective blend ratio in [0, blend_ratio]
        """
        total_steps = self._bandit.total_steps

        if total_steps < self._min_samples:
            return 0.0

        if self._ramp_up_samples <= 0:
            return self._blend_ratio

        # Linear ramp from min_samples to min_samples + ramp_up_samples
        excess = total_steps - self._min_samples
        ramp_progress = min(1.0, excess / self._ramp_up_samples)

        return self._blend_ratio * ramp_progress

    def update_from_pnl(self, arm_name: str, net_pnl_bps: float) -> None:
        """
        Update bandit after observing realized P&L.

        Args:
            arm_name: Name of the arm (horizon/model)
            net_pnl_bps: Net P&L in basis points

        Raises:
            ValueError: If arm_name not recognized
        """
        if arm_name not in self._name_to_idx:
            raise ValueError(
                f"Unknown arm '{arm_name}', expected one of {self._arm_names}"
            )

        arm_idx = self._name_to_idx[arm_name]
        self._bandit.update(arm_idx, net_pnl_bps)

        logger.debug(f"Updated bandit: arm={arm_name}, reward={net_pnl_bps:.2f}bps")

    def select_arm(self) -> str:
        """
        Select an arm using bandit probabilities.

        Can be used when pure bandit selection is needed
        (e.g., for exploration during warm-up).

        Returns:
            Name of selected arm
        """
        arm_idx = self._bandit.select_arm()
        return self._arm_names[arm_idx]

    def get_arm_index(self, arm_name: str) -> int:
        """Get index for an arm name."""
        if arm_name not in self._name_to_idx:
            raise ValueError(f"Unknown arm '{arm_name}'")
        return self._name_to_idx[arm_name]

    def add_arm(self, arm_name: str) -> int:
        """
        Dynamically add a new arm (horizon/target) to the optimizer.

        This allows the system to discover new trained models at runtime
        and start learning if they're profitable.

        Args:
            arm_name: Name for the new arm (e.g., "30m", "2h")

        Returns:
            Index of the new arm

        Raises:
            ValueError: If arm already exists
        """
        if arm_name in self._name_to_idx:
            raise ValueError(f"Arm '{arm_name}' already exists")

        # Add to bandit
        new_idx = self._bandit.add_arm(arm_name)

        # Update our tracking
        self._arm_names.append(arm_name)
        self._name_to_idx[arm_name] = new_idx
        self._n_arms += 1

        logger.info(
            f"EnsembleWeightOptimizer added arm '{arm_name}' "
            f"(now {self._n_arms} arms)"
        )

        return new_idx

    def remove_arm(self, arm_name: str) -> None:
        """
        Remove an arm from the optimizer.

        Args:
            arm_name: Name of arm to remove

        Raises:
            ValueError: If arm doesn't exist or minimum arms reached
        """
        if arm_name not in self._name_to_idx:
            raise ValueError(f"Arm '{arm_name}' not found")

        # Remove from bandit
        self._bandit.remove_arm(arm_name)

        # Update our tracking
        self._arm_names.remove(arm_name)
        self._n_arms -= 1

        # Rebuild index mapping
        self._name_to_idx = {
            name: i for i, name in enumerate(self._arm_names)
        }

        logger.info(
            f"EnsembleWeightOptimizer removed arm '{arm_name}' "
            f"(now {self._n_arms} arms)"
        )

    def has_arm(self, arm_name: str) -> bool:
        """Check if an arm exists."""
        return arm_name in self._name_to_idx

    def get_stats(self) -> Dict[str, any]:
        """
        Get optimizer statistics.

        Returns:
            Dict with stats for analysis/debugging
        """
        bandit_stats = self._bandit.get_stats()

        return {
            "n_arms": self._n_arms,
            "arm_names": self._arm_names,
            "blend_ratio": self._blend_ratio,
            "min_samples": self._min_samples,
            "ramp_up_samples": self._ramp_up_samples,
            "effective_blend": self._calculate_effective_blend(),
            "bandit_total_steps": self._bandit.total_steps,
            "bandit_stats": bandit_stats,
        }

    def get_bandit_weights(self) -> Dict[str, float]:
        """Get raw bandit weights (without blending)."""
        return self._bandit.get_weights_dict()

    def get_bandit_probabilities(self) -> Dict[str, float]:
        """Get bandit selection probabilities."""
        probs = self._bandit.get_probabilities()
        return {name: float(probs[i]) for i, name in enumerate(self._arm_names)}

    @property
    def bandit(self) -> Exp3IXBandit:
        """Get underlying bandit instance."""
        return self._bandit

    @property
    def arm_names(self) -> List[str]:
        """Get arm names."""
        return self._arm_names.copy()

    @property
    def n_arms(self) -> int:
        """Number of arms."""
        return self._n_arms

    @property
    def blend_ratio(self) -> float:
        """Target blend ratio."""
        return self._blend_ratio

    @property
    def effective_blend_ratio(self) -> float:
        """Current effective blend ratio."""
        return self._calculate_effective_blend()

    def to_dict(self) -> Dict[str, any]:
        """
        Serialize state for persistence.

        Returns:
            Dict that can be saved to JSON
        """
        return {
            "arm_names": self._arm_names,
            "blend_ratio": self._blend_ratio,
            "min_samples": self._min_samples,
            "ramp_up_samples": self._ramp_up_samples,
            "bandit": self._bandit.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> "EnsembleWeightOptimizer":
        """
        Restore from serialized state.

        Args:
            data: Dict from to_dict()

        Returns:
            Restored EnsembleWeightOptimizer
        """
        # Restore bandit first
        bandit = Exp3IXBandit.from_dict(data["bandit"])

        # Create optimizer with restored bandit
        optimizer = cls(
            arm_names=data["arm_names"],
            bandit=bandit,
            blend_ratio=data["blend_ratio"],
            min_samples=data["min_samples"],
            ramp_up_samples=data.get("ramp_up_samples", data["min_samples"]),
        )

        logger.info(
            f"Restored EnsembleWeightOptimizer: {optimizer._n_arms} arms, "
            f"{bandit.total_steps} bandit steps"
        )

        return optimizer

    def reset(self) -> None:
        """Reset optimizer and bandit state."""
        self._bandit.reset()
        logger.info("EnsembleWeightOptimizer reset")
