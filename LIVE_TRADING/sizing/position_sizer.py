"""
Position Sizer
==============

Main position sizing engine combining all components.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from CONFIG.config_loader import get_cfg
from TRAINING.common.utils.determinism_ordering import sorted_items

from LIVE_TRADING.common.constants import DEFAULT_CONFIG, SIDE_BUY, SIDE_SELL
from LIVE_TRADING.gating.barrier_gate import GateResult
from .vol_scaling import VolatilityScaler
from .turnover import TurnoverManager

logger = logging.getLogger(__name__)


@dataclass
class SizingResult:
    """Result of position sizing."""

    symbol: str
    target_weight: float
    current_weight: float
    trade_weight: float  # Actual weight change
    shares: int
    notional: float
    side: str  # "BUY" or "SELL"
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "target_weight": self.target_weight,
            "current_weight": self.current_weight,
            "trade_weight": self.trade_weight,
            "shares": self.shares,
            "notional": self.notional,
            "side": self.side,
            "reason": self.reason,
        }


class PositionSizer:
    """
    Main position sizing engine.

    Pipeline:
    1. Scale alpha by volatility
    2. Apply gate reduction
    3. Apply no-trade band
    4. Normalize to gross target
    5. Calculate shares

    This converts alpha signals to actionable trade sizes
    while respecting risk limits and turnover constraints.
    """

    def __init__(
        self,
        vol_scaler: VolatilityScaler | None = None,
        turnover_manager: TurnoverManager | None = None,
        gross_target: float | None = None,
        max_position_pct: float | None = None,
    ):
        """
        Initialize position sizer.

        Args:
            vol_scaler: VolatilityScaler instance (created if not provided)
            turnover_manager: TurnoverManager instance (created if not provided)
            gross_target: Target gross exposure (e.g., 0.5 = 50%)
            max_position_pct: Maximum single position as percentage
        """
        self.vol_scaler = vol_scaler or VolatilityScaler()
        self.turnover_manager = turnover_manager or TurnoverManager()

        self.gross_target = gross_target if gross_target is not None else get_cfg(
            "live_trading.sizing.gross_target",
            default=DEFAULT_CONFIG["gross_target"],
        )
        max_pos_config = max_position_pct if max_position_pct is not None else get_cfg(
            "live_trading.risk.max_position_pct",
            default=DEFAULT_CONFIG["max_position_pct"],
        )
        self.max_position_pct = max_pos_config / 100  # Convert to decimal

        logger.info(
            f"PositionSizer: gross_target={self.gross_target}, "
            f"max_position={self.max_position_pct:.1%}"
        )

    def calculate_target_weight(
        self,
        alpha: float,
        volatility: float,
        gate_result: Optional[GateResult] = None,
    ) -> float:
        """
        Calculate target weight for a symbol.

        Args:
            alpha: Alpha signal
            volatility: Volatility estimate
            gate_result: Optional gate result for reduction

        Returns:
            Target weight
        """
        # Vol-scaled weight
        weight = self.vol_scaler.scale(alpha, volatility)

        # Apply gate reduction
        if gate_result is not None:
            weight *= gate_result.gate_value

        # Cap at max position
        weight = max(-self.max_position_pct, min(self.max_position_pct, weight))

        return weight

    def size_single(
        self,
        symbol: str,
        alpha: float,
        volatility: float,
        current_weight: float = 0.0,
        price: float = 100.0,
        portfolio_value: float = 100_000.0,
        gate_result: Optional[GateResult] = None,
    ) -> SizingResult:
        """
        Size a single position.

        Args:
            symbol: Trading symbol
            alpha: Alpha signal
            volatility: Volatility estimate
            current_weight: Current position weight
            price: Current price
            portfolio_value: Total portfolio value
            gate_result: Optional gate result

        Returns:
            SizingResult
        """
        target = self.calculate_target_weight(alpha, volatility, gate_result)
        trade_weight = target - current_weight

        # Check no-trade band
        if abs(trade_weight) < self.turnover_manager.no_trade_band:
            return SizingResult(
                symbol=symbol,
                target_weight=current_weight,  # Keep current
                current_weight=current_weight,
                trade_weight=0.0,
                shares=0,
                notional=0.0,
                side="HOLD",
                reason="no_trade_band",
            )

        # Calculate shares
        notional = abs(trade_weight) * portfolio_value
        shares = int(notional / price) if price > 0 else 0
        side = SIDE_BUY if trade_weight > 0 else SIDE_SELL

        return SizingResult(
            symbol=symbol,
            target_weight=target,
            current_weight=current_weight,
            trade_weight=trade_weight,
            shares=shares,
            notional=notional,
            side=side,
            reason="sized",
        )

    def size_portfolio(
        self,
        alphas: Dict[str, float],
        volatilities: Dict[str, float],
        current_weights: Dict[str, float],
        gates: Dict[str, GateResult] | None = None,
        portfolio_value: float = 100_000.0,
        prices: Dict[str, float] | None = None,
    ) -> Dict[str, SizingResult]:
        """
        Size all positions in portfolio.

        Args:
            alphas: Alpha signals per symbol
            volatilities: Volatilities per symbol
            current_weights: Current position weights
            gates: Gate results per symbol
            portfolio_value: Total portfolio value
            prices: Current prices per symbol

        Returns:
            Dict mapping symbol to SizingResult
        """
        gates = gates or {}
        prices = prices or {}

        # Calculate raw target weights
        raw_targets = {}
        for symbol, alpha in sorted_items(alphas):
            vol = volatilities.get(symbol, 0.01)
            gate = gates.get(symbol)
            raw_targets[symbol] = self.calculate_target_weight(alpha, vol, gate)

        # Apply no-trade band
        targets = self.turnover_manager.apply_no_trade_band(
            raw_targets, current_weights
        )

        # Normalize to gross target
        targets = self._normalize_to_gross(targets)

        # Calculate results
        results = {}
        for symbol, target in sorted_items(targets):
            current = current_weights.get(symbol, 0.0)
            trade = target - current
            price = prices.get(symbol, 100.0)

            notional = abs(trade) * portfolio_value
            shares = int(notional / price) if price > 0 else 0

            if abs(trade) < 1e-8:
                side = "HOLD"
                reason = "no_trade_band"
            else:
                side = SIDE_BUY if trade > 0 else SIDE_SELL
                reason = "sized"

            results[symbol] = SizingResult(
                symbol=symbol,
                target_weight=target,
                current_weight=current,
                trade_weight=trade,
                shares=shares,
                notional=notional,
                side=side,
                reason=reason,
            )

        return results

    def _normalize_to_gross(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize weights to gross target.

        If gross exposure exceeds target, scale down proportionally.

        Args:
            weights: Position weights

        Returns:
            Normalized weights
        """
        gross = sum(abs(w) for w in weights.values())

        if gross <= 0 or gross <= self.gross_target:
            return weights

        scale = self.gross_target / gross
        return {s: w * scale for s, w in sorted_items(weights)}

    def get_gross_exposure(self, weights: Dict[str, float]) -> float:
        """
        Calculate gross exposure.

        Args:
            weights: Position weights

        Returns:
            Gross exposure (sum of absolute weights)
        """
        return sum(abs(w) for w in weights.values())

    def get_net_exposure(self, weights: Dict[str, float]) -> float:
        """
        Calculate net exposure.

        Args:
            weights: Position weights

        Returns:
            Net exposure (sum of signed weights)
        """
        return sum(weights.values())

    def get_portfolio_analysis(
        self,
        alphas: Dict[str, float],
        volatilities: Dict[str, float],
        current_weights: Dict[str, float],
        gates: Dict[str, GateResult] | None = None,
    ) -> Dict[str, Any]:
        """
        Get detailed portfolio sizing analysis.

        Args:
            alphas: Alpha signals
            volatilities: Volatilities
            current_weights: Current weights
            gates: Gate results

        Returns:
            Analysis dict
        """
        gates = gates or {}

        # Calculate raw targets
        raw_targets = {}
        for symbol, alpha in sorted_items(alphas):
            vol = volatilities.get(symbol, 0.01)
            gate = gates.get(symbol)
            raw_targets[symbol] = self.calculate_target_weight(alpha, vol, gate)

        # Apply no-trade band
        targets = self.turnover_manager.apply_no_trade_band(
            raw_targets, current_weights
        )

        # Normalize
        normalized = self._normalize_to_gross(targets)

        return {
            "raw_gross": self.get_gross_exposure(raw_targets),
            "after_band_gross": self.get_gross_exposure(targets),
            "final_gross": self.get_gross_exposure(normalized),
            "gross_target": self.gross_target,
            "net_exposure": self.get_net_exposure(normalized),
            "num_positions": len([w for w in normalized.values() if abs(w) > 1e-8]),
            "max_position": max(abs(w) for w in normalized.values()) if normalized else 0,
            "turnover": self.turnover_manager.calculate_turnover(normalized, current_weights),
        }
