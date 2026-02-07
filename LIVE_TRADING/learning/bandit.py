"""
Exp3-IX Multi-Armed Bandit
==========================

Implementation of the Exp3-IX algorithm for online weight adaptation.

This bandit algorithm provides:
- Exploration/exploitation tradeoff via gamma parameter
- Implicit exploration for improved high-probability regret bounds
- Adaptive learning rate based on problem horizon

Reference:
- Neu, Gergely. "Explore no more: Improved high-probability regret bounds
  for non-stochastic bandits"
- DOCS/03_technical/trading/architecture/MATHEMATICAL_FOUNDATIONS.md (Section 5)

SST Compliance:
- Uses get_cfg() for configuration
- Deterministic with seed control
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import numpy as np

from CONFIG.config_loader import get_cfg

logger = logging.getLogger(__name__)

# Default parameters from Mathematical Foundations doc
DEFAULT_GAMMA = 0.05  # Exploration rate
DEFAULT_ETA_MAX = 0.07  # Maximum learning rate


class Exp3IXBandit:
    """
    Exp3-IX multi-armed bandit for online weight adaptation.

    Arms can represent:
    - Model families (LightGBM, XGBoost, etc.)
    - Horizons (5m, 10m, 15m, etc.)
    - Model-horizon pairs

    The algorithm maintains unnormalized weights for each arm and
    updates them based on observed rewards using importance-weighted
    estimators.

    Weight update:
        u_i <- u_i * exp(eta * r_hat_i)
        r_hat_i = r_i / p_i

    Probability selection:
        p_i = (1 - gamma) * (u_i / sum(u)) + gamma / K

    Example:
        >>> bandit = Exp3IXBandit(n_arms=5)
        >>> arm = bandit.select_arm()  # Returns 0-4
        >>> bandit.update(arm, reward=10.5)  # Update with P&L in bps
        >>> weights = bandit.get_weights()  # Get normalized weights
    """

    def __init__(
        self,
        n_arms: int,
        arm_names: Optional[List[str]] = None,
        gamma: Optional[float] = None,
        eta: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize Exp3-IX bandit.

        Args:
            n_arms: Number of arms (models/horizons)
            arm_names: Optional names for arms (for logging)
            gamma: Exploration rate (default: 0.05 from config)
            eta: Learning rate (default: auto-computed)
            seed: Random seed for reproducibility
        """
        if n_arms < 2:
            raise ValueError(f"Need at least 2 arms, got {n_arms}")

        self._n_arms = n_arms
        self._arm_names = arm_names or [f"arm_{i}" for i in range(n_arms)]

        if len(self._arm_names) != n_arms:
            raise ValueError(f"arm_names length {len(self._arm_names)} != n_arms {n_arms}")

        # Load parameters from config with fallbacks
        self._gamma = gamma if gamma is not None else get_cfg(
            "live_trading.online_learning.gamma", default=DEFAULT_GAMMA
        )
        self._eta_auto = eta is None
        self._eta = eta if eta is not None else self._compute_initial_eta()

        # Initialize weights uniformly
        self._weights = np.ones(n_arms, dtype=np.float64)

        # Tracking
        self._total_steps = 0
        self._arm_pulls = np.zeros(n_arms, dtype=np.int64)
        self._cumulative_rewards = np.zeros(n_arms, dtype=np.float64)

        # Random state for reproducibility
        seed = seed if seed is not None else get_cfg(
            "pipeline.determinism.base_seed", default=42
        )
        self._rng = np.random.default_rng(seed)

        logger.info(
            f"Exp3IXBandit initialized: n_arms={n_arms}, gamma={self._gamma:.3f}, "
            f"eta={self._eta:.4f} ({'auto' if self._eta_auto else 'fixed'})"
        )

    def _compute_initial_eta(self) -> float:
        """
        Compute initial learning rate.

        For Exp3-IX: eta = min(eta_max, sqrt(ln(K) / (K * T)))

        At initialization T=0, we use eta_max and adapt as T grows.
        """
        return get_cfg(
            "live_trading.online_learning.eta_max", default=DEFAULT_ETA_MAX
        )

    def _compute_adaptive_eta(self) -> float:
        """
        Compute adaptive learning rate based on total steps.

        eta = min(eta_max, sqrt(ln(K) / (K * T)))
        """
        if self._total_steps == 0:
            return self._compute_initial_eta()

        eta_max = get_cfg(
            "live_trading.online_learning.eta_max", default=DEFAULT_ETA_MAX
        )
        K = self._n_arms
        T = self._total_steps

        # Exp3-IX learning rate formula
        eta_computed = math.sqrt(math.log(K) / (K * T))

        return min(eta_max, eta_computed)

    def get_probabilities(self) -> np.ndarray:
        """
        Get selection probabilities for all arms.

        p_i = (1 - gamma) * (u_i / sum(u)) + gamma / K

        Returns:
            Array of probabilities summing to 1
        """
        # Normalize weights
        weight_sum = np.sum(self._weights)
        if weight_sum == 0:
            # Fallback to uniform
            return np.ones(self._n_arms) / self._n_arms

        normalized = self._weights / weight_sum

        # Mix with uniform exploration
        K = self._n_arms
        probs = (1 - self._gamma) * normalized + self._gamma / K

        # Ensure valid probability distribution
        probs = np.clip(probs, 1e-10, 1.0)
        probs /= np.sum(probs)

        return probs

    def select_arm(self) -> int:
        """
        Select an arm using the probability distribution.

        Returns:
            Index of selected arm (0 to n_arms-1)
        """
        probs = self.get_probabilities()
        arm = int(self._rng.choice(self._n_arms, p=probs))

        logger.debug(
            f"Exp3IX selected arm {arm} ({self._arm_names[arm]}) "
            f"with p={probs[arm]:.4f}"
        )

        return arm

    def update(self, arm: int, reward: float) -> None:
        """
        Update weights after observing reward.

        Uses importance-weighted estimator:
            r_hat_i = r_i / p_i
            u_i <- u_i * exp(eta * r_hat_i)

        Args:
            arm: Index of arm that was pulled
            reward: Observed reward (typically net P&L in bps)
        """
        if not 0 <= arm < self._n_arms:
            raise ValueError(f"Invalid arm {arm}, must be 0 to {self._n_arms - 1}")

        self._total_steps += 1
        self._arm_pulls[arm] += 1
        self._cumulative_rewards[arm] += reward

        # Get probability for importance weighting
        probs = self.get_probabilities()
        p_arm = probs[arm]

        # Importance-weighted reward estimate
        r_hat = reward / p_arm

        # Adaptive learning rate if enabled
        if self._eta_auto:
            self._eta = self._compute_adaptive_eta()

        # Weight update
        self._weights[arm] *= math.exp(self._eta * r_hat)

        # Numerical stability: prevent weights from exploding or vanishing
        max_weight = np.max(self._weights)
        if max_weight > 1e6:
            self._weights /= max_weight
        min_weight = np.min(self._weights)
        if min_weight < 1e-10 and min_weight > 0:
            self._weights = np.maximum(self._weights, 1e-10)

        logger.debug(
            f"Exp3IX update: arm={arm}, reward={reward:.2f}bps, "
            f"r_hat={r_hat:.2f}, eta={self._eta:.4f}, "
            f"new_weight={self._weights[arm]:.4f}"
        )

    def get_weights(self) -> np.ndarray:
        """
        Get normalized weights for blending.

        Returns:
            Array of weights summing to 1
        """
        weight_sum = np.sum(self._weights)
        if weight_sum == 0:
            return np.ones(self._n_arms) / self._n_arms
        return self._weights / weight_sum

    def get_weights_dict(self) -> Dict[str, float]:
        """
        Get weights as a dictionary with arm names.

        Returns:
            Dict mapping arm name to normalized weight
        """
        normalized = self.get_weights()
        return {name: float(w) for name, w in zip(self._arm_names, normalized)}

    def get_stats(self) -> Dict[str, any]:
        """
        Get statistics about bandit state.

        Returns:
            Dict with stats for analysis/debugging
        """
        probs = self.get_probabilities()
        weights = self.get_weights()

        return {
            "n_arms": self._n_arms,
            "total_steps": self._total_steps,
            "gamma": self._gamma,
            "eta": self._eta,
            "eta_auto": self._eta_auto,
            "arm_stats": [
                {
                    "name": self._arm_names[i],
                    "pulls": int(self._arm_pulls[i]),
                    "cumulative_reward": float(self._cumulative_rewards[i]),
                    "avg_reward": (
                        float(self._cumulative_rewards[i] / self._arm_pulls[i])
                        if self._arm_pulls[i] > 0 else 0.0
                    ),
                    "weight": float(weights[i]),
                    "probability": float(probs[i]),
                }
                for i in range(self._n_arms)
            ],
        }

    def to_dict(self) -> Dict[str, any]:
        """
        Serialize bandit state to dictionary.

        Returns:
            Dict that can be saved to JSON
        """
        return {
            "n_arms": self._n_arms,
            "arm_names": self._arm_names,
            "gamma": self._gamma,
            "eta": self._eta,
            "eta_auto": self._eta_auto,
            "weights": self._weights.tolist(),
            "total_steps": self._total_steps,
            "arm_pulls": self._arm_pulls.tolist(),
            "cumulative_rewards": self._cumulative_rewards.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, any], seed: Optional[int] = None) -> "Exp3IXBandit":
        """
        Restore bandit from serialized state.

        Args:
            data: Dict from to_dict()
            seed: Optional new seed for RNG

        Returns:
            Restored Exp3IXBandit instance
        """
        bandit = cls(
            n_arms=data["n_arms"],
            arm_names=data["arm_names"],
            gamma=data["gamma"],
            eta=data["eta"],
            seed=seed,
        )
        bandit._eta_auto = data["eta_auto"]
        bandit._weights = np.array(data["weights"], dtype=np.float64)
        bandit._total_steps = data["total_steps"]
        bandit._arm_pulls = np.array(data["arm_pulls"], dtype=np.int64)
        bandit._cumulative_rewards = np.array(data["cumulative_rewards"], dtype=np.float64)

        logger.info(f"Restored Exp3IXBandit with {bandit._total_steps} steps")
        return bandit

    def reset(self) -> None:
        """Reset bandit to initial state."""
        self._weights = np.ones(self._n_arms, dtype=np.float64)
        self._total_steps = 0
        self._arm_pulls = np.zeros(self._n_arms, dtype=np.int64)
        self._cumulative_rewards = np.zeros(self._n_arms, dtype=np.float64)
        self._eta = self._compute_initial_eta() if self._eta_auto else self._eta
        logger.info("Exp3IXBandit reset to initial state")

    def add_arm(
        self,
        arm_name: str,
        initial_weight: Optional[float] = None,
    ) -> int:
        """
        Dynamically add a new arm to the bandit.

        The new arm starts with a weight that gives it fair exploration
        opportunity while not disrupting learned weights too much.

        Args:
            arm_name: Name for the new arm
            initial_weight: Initial weight (default: average of existing weights)

        Returns:
            Index of the new arm

        Raises:
            ValueError: If arm_name already exists
        """
        if arm_name in self._arm_names:
            raise ValueError(f"Arm '{arm_name}' already exists")

        # Calculate initial weight for new arm
        if initial_weight is None:
            # Use average of existing weights to give fair chance
            initial_weight = float(np.mean(self._weights))

        # Expand arrays
        self._weights = np.append(self._weights, initial_weight)
        self._arm_pulls = np.append(self._arm_pulls, 0)
        self._cumulative_rewards = np.append(self._cumulative_rewards, 0.0)
        self._arm_names.append(arm_name)
        self._n_arms += 1

        new_index = self._n_arms - 1

        logger.info(
            f"Added new arm '{arm_name}' at index {new_index} "
            f"with initial_weight={initial_weight:.4f}"
        )

        return new_index

    def remove_arm(self, arm_name: str) -> None:
        """
        Remove an arm from the bandit.

        Use with caution - removes all learned data for this arm.

        Args:
            arm_name: Name of arm to remove

        Raises:
            ValueError: If arm_name doesn't exist or only 2 arms remain
        """
        if arm_name not in self._arm_names:
            raise ValueError(f"Arm '{arm_name}' not found")

        if self._n_arms <= 2:
            raise ValueError("Cannot remove arm: minimum 2 arms required")

        idx = self._arm_names.index(arm_name)

        # Remove from all arrays
        self._weights = np.delete(self._weights, idx)
        self._arm_pulls = np.delete(self._arm_pulls, idx)
        self._cumulative_rewards = np.delete(self._cumulative_rewards, idx)
        self._arm_names.pop(idx)
        self._n_arms -= 1

        logger.info(f"Removed arm '{arm_name}' (was index {idx})")

    def has_arm(self, arm_name: str) -> bool:
        """Check if an arm exists."""
        return arm_name in self._arm_names

    def get_arm_index(self, arm_name: str) -> int:
        """
        Get index for an arm name.

        Args:
            arm_name: Name of arm

        Returns:
            Index of arm

        Raises:
            ValueError: If arm not found
        """
        if arm_name not in self._arm_names:
            raise ValueError(f"Arm '{arm_name}' not found")
        return self._arm_names.index(arm_name)

    @property
    def n_arms(self) -> int:
        """Number of arms."""
        return self._n_arms

    @property
    def arm_names(self) -> List[str]:
        """Names of arms."""
        return self._arm_names.copy()

    @property
    def total_steps(self) -> int:
        """Total update steps."""
        return self._total_steps

    @property
    def gamma(self) -> float:
        """Exploration rate."""
        return self._gamma

    @property
    def eta(self) -> float:
        """Current learning rate."""
        return self._eta
