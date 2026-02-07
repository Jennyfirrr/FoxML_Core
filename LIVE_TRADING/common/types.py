"""
Common Type Definitions
=======================

Shared dataclasses used across LIVE_TRADING modules.
These types provide:
1. Type safety across module boundaries
2. Explainability through PipelineTrace
3. Serialization support for audit logging

SST Compliance:
- All to_dict() methods use sorted keys
- datetime fields serialize to ISO format
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from LIVE_TRADING.common.time_utils import parse_iso

# =============================================================================
# Market Data Types
# =============================================================================


@dataclass
class MarketSnapshot:
    """
    Market data at decision time.

    Captures OHLCV plus quote data for explainability.
    Used to understand what market conditions triggered a decision.
    """

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: float
    ask: float
    spread_bps: float
    volatility: float  # Annualized volatility

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict with sorted keys."""
        return {
            "ask": self.ask,
            "bid": self.bid,
            "close": self.close,
            "high": self.high,
            "low": self.low,
            "open": self.open,
            "spread_bps": self.spread_bps,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "volatility": self.volatility,
            "volume": self.volume,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MarketSnapshot":
        """Create from dict."""
        return cls(
            symbol=d["symbol"],
            timestamp=parse_iso(d["timestamp"]),
            open=d["open"],
            high=d["high"],
            low=d["low"],
            close=d["close"],
            volume=d["volume"],
            bid=d["bid"],
            ask=d["ask"],
            spread_bps=d["spread_bps"],
            volatility=d["volatility"],
        )


@dataclass
class Quote:
    """Real-time quote data."""

    symbol: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    timestamp: datetime
    spread_bps: float = 0.0

    def __post_init__(self) -> None:
        """Calculate spread if not provided."""
        if self.spread_bps == 0.0 and (self.bid + self.ask) > 0:
            self.spread_bps = (self.ask - self.bid) / ((self.ask + self.bid) / 2) * 10000

    @property
    def mid(self) -> float:
        """Mid price."""
        return (self.bid + self.ask) / 2

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "ask": self.ask,
            "ask_size": self.ask_size,
            "bid": self.bid,
            "bid_size": self.bid_size,
            "spread_bps": self.spread_bps,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# Prediction Types
# =============================================================================


@dataclass
class ModelPrediction:
    """Prediction from a single model."""

    family: str
    target: str
    horizon: str
    raw_value: float
    standardized_value: float = 0.0
    confidence: float = 1.0
    ic: float = 0.0  # Information coefficient
    freshness: float = 1.0  # Decay factor for staleness

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "confidence": self.confidence,
            "family": self.family,
            "freshness": self.freshness,
            "horizon": self.horizon,
            "ic": self.ic,
            "raw_value": self.raw_value,
            "standardized_value": self.standardized_value,
            "target": self.target,
        }


@dataclass
class HorizonPredictions:
    """All predictions for a single horizon."""

    horizon: str
    predictions: Dict[str, ModelPrediction] = field(default_factory=dict)  # family -> pred
    blended_alpha: float = 0.0
    blend_weights: Dict[str, float] = field(default_factory=dict)

    def add_prediction(self, pred: ModelPrediction) -> None:
        """Add a prediction."""
        self.predictions[pred.family] = pred

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "blend_weights": dict(sorted(self.blend_weights.items())),
            "blended_alpha": self.blended_alpha,
            "horizon": self.horizon,
            "predictions": {
                k: v.to_dict() for k, v in sorted(self.predictions.items())
            },
        }


# =============================================================================
# Gate Types
# =============================================================================


@dataclass
class GateResult:
    """Result from a gate evaluation."""

    gate_name: str
    allowed: bool
    gate_value: float  # 0.0 to 1.0, where 1.0 = fully open
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "allowed": self.allowed,
            "gate_name": self.gate_name,
            "gate_value": self.gate_value,
            "metadata": dict(sorted(self.metadata.items())),
            "reason": self.reason,
        }


@dataclass
class BarrierGateResult(GateResult):
    """Result from barrier gate evaluation."""

    p_peak: float = 0.0
    p_valley: float = 0.0

    def __post_init__(self) -> None:
        """Set gate name."""
        object.__setattr__(self, "gate_name", "barrier")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        d = super().to_dict()
        d["p_peak"] = self.p_peak
        d["p_valley"] = self.p_valley
        return dict(sorted(d.items()))


@dataclass
class SpreadGateResult(GateResult):
    """Result from spread gate evaluation."""

    spread_bps: float = 0.0
    max_spread_bps: float = 0.0

    def __post_init__(self) -> None:
        """Set gate name."""
        object.__setattr__(self, "gate_name", "spread")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        d = super().to_dict()
        d["max_spread_bps"] = self.max_spread_bps
        d["spread_bps"] = self.spread_bps
        return dict(sorted(d.items()))


# =============================================================================
# Arbitration Types
# =============================================================================


@dataclass
class CostBreakdown:
    """Breakdown of trading costs."""

    spread_cost: float = 0.0
    timing_cost: float = 0.0
    impact_cost: float = 0.0
    total_cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "impact_cost": self.impact_cost,
            "spread_cost": self.spread_cost,
            "timing_cost": self.timing_cost,
            "total_cost": self.total_cost,
        }


@dataclass
class ArbitrationResult:
    """Result from horizon arbitration."""

    selected_horizon: Optional[str]
    horizon_scores: Dict[str, float]  # horizon -> net score
    cost_breakdown: Dict[str, CostBreakdown]  # horizon -> costs
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "cost_breakdown": {
                k: v.to_dict() for k, v in sorted(self.cost_breakdown.items())
            },
            "horizon_scores": dict(sorted(self.horizon_scores.items())),
            "reason": self.reason,
            "selected_horizon": self.selected_horizon,
        }


# =============================================================================
# Sizing Types
# =============================================================================


@dataclass
class SizingResult:
    """Result from position sizing."""

    symbol: str
    raw_weight: float
    gate_adjusted_weight: float
    final_weight: float
    shares: int
    side: str  # BUY or SELL
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "final_weight": self.final_weight,
            "gate_adjusted_weight": self.gate_adjusted_weight,
            "raw_weight": self.raw_weight,
            "reason": self.reason,
            "shares": self.shares,
            "side": self.side,
            "symbol": self.symbol,
        }


# =============================================================================
# Pipeline Trace (Explainability)
# =============================================================================


@dataclass
class PipelineTrace:
    """
    Full trace of decision pipeline for explainability.

    This captures every stage of the trading pipeline, allowing
    post-hoc analysis of why a trade was made or not made.
    """

    market_snapshot: MarketSnapshot

    # Stage 1: Predictions (horizon -> family -> prediction)
    raw_predictions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    standardized_predictions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confidences: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Stage 2: Blending
    blend_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)
    blended_alphas: Dict[str, float] = field(default_factory=dict)

    # Stage 3: Arbitration
    horizon_scores: Dict[str, float] = field(default_factory=dict)
    cost_breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)
    selected_horizon: Optional[str] = None

    # Stage 4: Gating
    barrier_gate_result: Dict[str, Any] = field(default_factory=dict)
    spread_gate_result: Dict[str, Any] = field(default_factory=dict)

    # Stage 5: Sizing
    raw_weight: float = 0.0
    gate_adjusted_weight: float = 0.0
    final_weight: float = 0.0

    # Stage 6: Risk
    risk_checks: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict with sorted keys."""
        return {
            "barrier_gate_result": dict(sorted(self.barrier_gate_result.items())),
            "blend_weights": {
                h: dict(sorted(w.items()))
                for h, w in sorted(self.blend_weights.items())
            },
            "blended_alphas": dict(sorted(self.blended_alphas.items())),
            "confidences": {
                h: dict(sorted(c.items()))
                for h, c in sorted(self.confidences.items())
            },
            "cost_breakdown": {
                h: dict(sorted(c.items()))
                for h, c in sorted(self.cost_breakdown.items())
            },
            "final_weight": self.final_weight,
            "gate_adjusted_weight": self.gate_adjusted_weight,
            "horizon_scores": dict(sorted(self.horizon_scores.items())),
            "market_snapshot": self.market_snapshot.to_dict(),
            "raw_predictions": {
                h: dict(sorted(p.items()))
                for h, p in sorted(self.raw_predictions.items())
            },
            "raw_weight": self.raw_weight,
            "risk_checks": dict(sorted(self.risk_checks.items())),
            "selected_horizon": self.selected_horizon,
            "spread_gate_result": dict(sorted(self.spread_gate_result.items())),
            "standardized_predictions": {
                h: dict(sorted(p.items()))
                for h, p in sorted(self.standardized_predictions.items())
            },
        }

    def explain(self) -> str:
        """Generate human-readable explanation."""
        lines = [
            f"Pipeline Trace: {self.market_snapshot.symbol}",
            "=" * 60,
            "",
            "MARKET CONTEXT",
            f"  Time: {self.market_snapshot.timestamp}",
            f"  OHLCV: O={self.market_snapshot.open:.2f} "
            f"H={self.market_snapshot.high:.2f} "
            f"L={self.market_snapshot.low:.2f} "
            f"C={self.market_snapshot.close:.2f} "
            f"V={self.market_snapshot.volume:,.0f}",
            f"  Bid/Ask: {self.market_snapshot.bid:.2f}/{self.market_snapshot.ask:.2f}",
            f"  Spread: {self.market_snapshot.spread_bps:.1f} bps",
            f"  Volatility: {self.market_snapshot.volatility:.2%}",
            "",
            "HORIZON ANALYSIS",
        ]

        for h in sorted(self.horizon_scores.keys()):
            alpha = self.blended_alphas.get(h, 0) * 10000
            score = self.horizon_scores.get(h, 0)
            marker = " <--" if h == self.selected_horizon else ""
            lines.append(f"  {h}: alpha={alpha:.1f}bps, score={score:.2f}{marker}")

        lines.extend(
            [
                "",
                f"SELECTED HORIZON: {self.selected_horizon or 'None'}",
                "",
                "COSTS",
            ]
        )

        if self.cost_breakdown:
            for h, costs in sorted(self.cost_breakdown.items()):
                lines.append(
                    f"  {h}: spread={costs.get('spread', 0):.1f} "
                    f"timing={costs.get('timing', 0):.1f} "
                    f"impact={costs.get('impact', 0):.1f}"
                )

        lines.extend(
            [
                "",
                "GATING",
                f"  Barrier: p_peak={self.barrier_gate_result.get('p_peak', 0):.2f}, "
                f"p_valley={self.barrier_gate_result.get('p_valley', 0):.2f}",
                f"  Spread: allowed={self.spread_gate_result.get('allowed', True)}",
                "",
                "SIZING",
                f"  Raw Weight: {self.raw_weight:.4f}",
                f"  Gate Adjusted: {self.gate_adjusted_weight:.4f}",
                f"  Final Weight: {self.final_weight:.4f}",
            ]
        )

        return "\n".join(lines)


# =============================================================================
# Trade Decision
# =============================================================================


@dataclass
class TradeDecision:
    """
    Result of a trading cycle with full context.

    Includes the decision, sizing, and optionally the full
    pipeline trace for explainability and audit.
    """

    symbol: str
    decision: str  # TRADE, HOLD, BLOCKED
    horizon: Optional[str]
    target_weight: float
    current_weight: float
    alpha: float
    shares: int
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)
    trace: Optional[PipelineTrace] = None

    def explain(self) -> str:
        """Get human-readable explanation."""
        if self.trace:
            return self.trace.explain()
        return f"{self.decision}: {self.reason}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "alpha": self.alpha,
            "current_weight": self.current_weight,
            "decision": self.decision,
            "horizon": self.horizon,
            "reason": self.reason,
            "shares": self.shares,
            "symbol": self.symbol,
            "target_weight": self.target_weight,
            "timestamp": self.timestamp.isoformat(),
            "trace": self.trace.to_dict() if self.trace else None,
        }

    def to_audit_record(self) -> Dict[str, Any]:
        """Convert to audit log record (always includes trace)."""
        return {
            "alpha": self.alpha,
            "current_weight": self.current_weight,
            "decision": self.decision,
            "horizon": self.horizon,
            "reason": self.reason,
            "shares": self.shares,
            "symbol": self.symbol,
            "target_weight": self.target_weight,
            "timestamp": self.timestamp.isoformat(),
            "trace": self.trace.to_dict() if self.trace else {},
        }


# =============================================================================
# Position State
# =============================================================================


@dataclass
class PositionState:
    """State of a single position."""

    symbol: str
    shares: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    weight: float = 0.0

    def update_price(self, price: float, portfolio_value: float) -> None:
        """Update with current price."""
        self.current_price = price
        self.unrealized_pnl = (price - self.entry_price) * self.shares
        if portfolio_value > 0:
            self.weight = (self.shares * price) / portfolio_value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "current_price": self.current_price,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat(),
            "shares": self.shares,
            "symbol": self.symbol,
            "unrealized_pnl": self.unrealized_pnl,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PositionState":
        """Create from dict."""
        return cls(
            symbol=d["symbol"],
            shares=d["shares"],
            entry_price=d["entry_price"],
            entry_time=parse_iso(d["entry_time"]),
            current_price=d.get("current_price", 0.0),
            unrealized_pnl=d.get("unrealized_pnl", 0.0),
            weight=d.get("weight", 0.0),
        )
