"""
Barrier Probability Gate
========================

Uses peak/valley predictions to gate entries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from CONFIG.config_loader import get_cfg

from LIVE_TRADING.common.constants import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Result of gate evaluation."""

    allowed: bool
    gate_value: float  # 0-1, 1 = fully allowed
    reason: str
    p_peak: Optional[float] = None
    p_valley: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "allowed": self.allowed,
            "gate_value": self.gate_value,
            "reason": self.reason,
            "p_peak": self.p_peak,
            "p_valley": self.p_valley,
        }


class BarrierGate:
    """
    Gates entries using barrier probability predictions.

    g = max(g_min, (1-p_peak)^γ × (0.5+0.5×p_valley)^δ)

    This gate prevents entering positions right before local
    peaks (tops) and encourages entry near valleys (bottoms).
    """

    def __init__(
        self,
        g_min: float | None = None,
        gamma: float | None = None,
        delta: float | None = None,
        peak_threshold: float | None = None,
        valley_threshold: float | None = None,
    ):
        """
        Initialize barrier gate.

        Args:
            g_min: Minimum gate value
            gamma: Peak sensitivity exponent
            delta: Valley sensitivity exponent
            peak_threshold: Block threshold for peak probability
            valley_threshold: Prefer threshold for valley probability
        """
        self.g_min = g_min if g_min is not None else get_cfg(
            "live_trading.barrier_gate.g_min",
            default=DEFAULT_CONFIG["g_min"],
        )
        self.gamma = gamma if gamma is not None else get_cfg(
            "live_trading.barrier_gate.gamma",
            default=DEFAULT_CONFIG["gamma"],
        )
        self.delta = delta if delta is not None else get_cfg(
            "live_trading.barrier_gate.delta",
            default=DEFAULT_CONFIG["delta"],
        )
        self.peak_threshold = peak_threshold if peak_threshold is not None else get_cfg(
            "live_trading.barrier_gate.peak_threshold",
            default=DEFAULT_CONFIG["peak_threshold"],
        )
        self.valley_threshold = valley_threshold if valley_threshold is not None else get_cfg(
            "live_trading.barrier_gate.valley_threshold",
            default=DEFAULT_CONFIG["valley_threshold"],
        )

        logger.info(f"BarrierGate: g_min={self.g_min}, γ={self.gamma}, δ={self.delta}")

    def evaluate_long_entry(
        self,
        p_peak: float,
        p_valley: float = 0.0,
        y_p_peak: float | None = None,
    ) -> GateResult:
        """
        Evaluate gate for long entry.

        Block if: P(will_peak_5m) > peak_threshold OR P(y_will_peak_5m) > peak_threshold
        Prefer if: P(will_valley_5m) > valley_threshold

        Args:
            p_peak: P(will_peak_5m)
            p_valley: P(will_valley_5m)
            y_p_peak: P(y_will_peak_5m) - optional auxiliary predictor

        Returns:
            GateResult
        """
        # Check hard block
        if p_peak > self.peak_threshold:
            return GateResult(
                allowed=False,
                gate_value=0.0,
                reason=f"peak_prob {p_peak:.2f} > {self.peak_threshold}",
                p_peak=p_peak,
                p_valley=p_valley,
            )

        if y_p_peak is not None and y_p_peak > self.peak_threshold:
            return GateResult(
                allowed=False,
                gate_value=0.0,
                reason=f"y_peak_prob {y_p_peak:.2f} > {self.peak_threshold}",
                p_peak=p_peak,
                p_valley=p_valley,
            )

        # Calculate gate value
        gate_value = self._calculate_gate(p_peak, p_valley)

        # Prefer entry if valley detected
        prefer = p_valley > self.valley_threshold
        reason = "valley_bounce" if prefer else "normal_entry"

        return GateResult(
            allowed=True,
            gate_value=gate_value,
            reason=reason,
            p_peak=p_peak,
            p_valley=p_valley,
        )

    def evaluate_short_entry(
        self,
        p_valley: float,
        p_peak: float = 0.0,
        y_p_valley: float | None = None,
    ) -> GateResult:
        """
        Evaluate gate for short entry.

        Block if: P(will_valley_5m) > valley_threshold (for shorts)
        Prefer if: P(will_peak_5m) > peak_threshold (approaching top)

        Args:
            p_valley: P(will_valley_5m)
            p_peak: P(will_peak_5m)
            y_p_valley: P(y_will_valley_5m) - optional auxiliary

        Returns:
            GateResult
        """
        # For shorts, block near valleys (bottoms)
        if p_valley > self.valley_threshold:
            return GateResult(
                allowed=False,
                gate_value=0.0,
                reason=f"valley_prob {p_valley:.2f} > {self.valley_threshold}",
                p_peak=p_peak,
                p_valley=p_valley,
            )

        if y_p_valley is not None and y_p_valley > self.valley_threshold:
            return GateResult(
                allowed=False,
                gate_value=0.0,
                reason=f"y_valley_prob {y_p_valley:.2f} > {self.valley_threshold}",
                p_peak=p_peak,
                p_valley=p_valley,
            )

        # Calculate gate value (inverse of long gate)
        gate_value = self._calculate_gate(p_valley, p_peak)

        # Prefer entry if peak detected (approaching top)
        prefer = p_peak > self.peak_threshold * 0.8  # Slightly lower threshold
        reason = "peak_rejection" if prefer else "normal_entry"

        return GateResult(
            allowed=True,
            gate_value=gate_value,
            reason=reason,
            p_peak=p_peak,
            p_valley=p_valley,
        )

    def _calculate_gate(self, p_peak: float, p_valley: float) -> float:
        """
        Calculate gate value.

        g = max(g_min, (1-p_peak)^γ × (0.5+0.5×p_valley)^δ)

        Args:
            p_peak: Peak probability (penalizes gate)
            p_valley: Valley probability (boosts gate)

        Returns:
            Gate value between g_min and 1.0
        """
        peak_factor = (1 - p_peak) ** self.gamma
        valley_factor = (0.5 + 0.5 * p_valley) ** self.delta
        g = peak_factor * valley_factor
        return max(self.g_min, min(1.0, g))

    def evaluate_long_exit(
        self,
        p_peak: float,
        alpha: float,
    ) -> GateResult:
        """
        Evaluate exit signal for long position.

        Exit if: P(will_peak_5m) > exit_threshold OR alpha < 0

        Args:
            p_peak: Peak probability
            alpha: Current alpha/expected return

        Returns:
            GateResult where allowed=True means "should exit"
        """
        exit_threshold = self.peak_threshold + 0.05  # Slightly higher for exit

        if p_peak > exit_threshold:
            return GateResult(
                allowed=True,
                gate_value=1.0,
                reason=f"peak_exit: {p_peak:.2f} > {exit_threshold}",
                p_peak=p_peak,
            )

        if alpha < 0:
            return GateResult(
                allowed=True,
                gate_value=1.0,
                reason=f"alpha_exit: {alpha:.4f} < 0",
                p_peak=p_peak,
            )

        return GateResult(
            allowed=False,
            gate_value=0.0,
            reason="hold_position",
            p_peak=p_peak,
        )

    def evaluate_short_exit(
        self,
        p_valley: float,
        alpha: float,
    ) -> GateResult:
        """
        Evaluate exit signal for short position.

        Exit if: P(will_valley_5m) > exit_threshold OR alpha > 0 (shorts profit from negative alpha)

        Args:
            p_valley: Valley probability
            alpha: Current alpha/expected return

        Returns:
            GateResult where allowed=True means "should exit"
        """
        exit_threshold = self.valley_threshold + 0.05

        if p_valley > exit_threshold:
            return GateResult(
                allowed=True,
                gate_value=1.0,
                reason=f"valley_exit: {p_valley:.2f} > {exit_threshold}",
                p_valley=p_valley,
            )

        if alpha > 0:
            return GateResult(
                allowed=True,
                gate_value=1.0,
                reason=f"alpha_exit: {alpha:.4f} > 0",
                p_valley=p_valley,
            )

        return GateResult(
            allowed=False,
            gate_value=0.0,
            reason="hold_position",
            p_valley=p_valley,
        )

    def get_size_reduction(self, p_peak: float) -> float:
        """
        Calculate position size reduction based on peak probability.

        size_reduction = (1 - p_peak)

        Args:
            p_peak: Peak probability

        Returns:
            Multiplier for position size (0.0 to 1.0)
        """
        return max(0.0, min(1.0, 1.0 - p_peak))

    def get_analysis(
        self,
        p_peak: float,
        p_valley: float,
    ) -> Dict[str, Any]:
        """
        Get detailed analysis of barrier signals.

        Args:
            p_peak: Peak probability
            p_valley: Valley probability

        Returns:
            Analysis dict with gate breakdown
        """
        gate_value = self._calculate_gate(p_peak, p_valley)
        peak_factor = (1 - p_peak) ** self.gamma
        valley_factor = (0.5 + 0.5 * p_valley) ** self.delta

        return {
            "p_peak": p_peak,
            "p_valley": p_valley,
            "peak_factor": peak_factor,
            "valley_factor": valley_factor,
            "raw_gate": peak_factor * valley_factor,
            "gate_value": gate_value,
            "size_reduction": self.get_size_reduction(p_peak),
            "long_blocked": p_peak > self.peak_threshold,
            "valley_preferred": p_valley > self.valley_threshold,
        }
