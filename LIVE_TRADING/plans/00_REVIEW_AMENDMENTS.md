# Plan Review: Amendments and Recommendations

## Executive Summary

After reviewing all 11 sub-plans, this document identifies:
1. **6 integration issues** between modules that could break the pipeline
2. **4 explainability gaps** that prevent understanding why trades were made
3. **5 extension points** needed for future plugin/hook architecture
4. **MCP server design** for development tracking

---

## Part 1: Integration Issues Found

### Issue 1: Missing `PositionState` Import in Trading Engine

**Location:** `LIVE_TRADING/engine/trading_engine.py` (Plan 10, line 480)

**Problem:** Code references `PositionState` class but doesn't import it.

**Fix:**
```python
from .state import EngineState, PositionState
```

---

### Issue 2: Barrier Gate Predictions Not Integrated

**Location:** `LIVE_TRADING/engine/trading_engine.py` (Plan 10, line 407)

**Problem:** There's a `TODO` comment but no actual integration with barrier model predictions.

**Fix:** Add to `_process_symbol()`:
```python
# Load barrier predictions
barrier_target = "will_peak_5m"
if barrier_target in self.targets:
    barrier_preds = self.predictor.predict_all_horizons(
        target=barrier_target,
        prices=prices,
        symbol=symbol,
    )
    p_peak = barrier_preds.get_horizon("5m").mean_calibrated if barrier_preds.get_horizon("5m") else 0.0
else:
    p_peak = 0.0

gate_result = self.barrier_gate.evaluate_long_entry(p_peak=p_peak, p_valley=0.0)
```

---

### Issue 3: HorizonPredictions None Check Missing

**Location:** `LIVE_TRADING/engine/trading_engine.py` (Plan 10, line 383-384)

**Problem:** Code calls `all_preds.get_horizon(h)` without checking for None properly.

**Fix:**
```python
blended = self.blender.blend_all_horizons({
    h: hp for h in HORIZONS
    if (hp := all_preds.get_horizon(h)) is not None and hp.predictions
})
```

---

### Issue 4: Inconsistent Determinism in Dict Iteration

**Locations:** Multiple files

**Problem:** Several loops don't use `sorted_items()`:
- `LIVE_TRADING/engine/trading_engine.py` line 317: `for symbol in symbols`
- `LIVE_TRADING/risk/exposure.py` lines not using `sorted_items()`

**Fix:** Replace all `for k, v in dict.items()` with `for k, v in sorted_items(dict)`

---

### Issue 5: DateTime Serialization in State

**Location:** `LIVE_TRADING/engine/state.py` (Plan 10)

**Problem:** `PositionState.entry_time` is a datetime but `asdict()` won't serialize it properly.

**Fix:** Add custom serialization:
```python
def save(self, path: Path) -> None:
    """Save state to file."""
    positions_data = {}
    for s, p in sorted_items(self.positions):
        pos_dict = asdict(p)
        pos_dict["entry_time"] = p.entry_time.isoformat()
        positions_data[s] = pos_dict

    data = {
        "portfolio_value": self.portfolio_value,
        "cash": self.cash,
        "positions": positions_data,
        # ... rest
    }
```

---

### Issue 6: Missing Cross-Module Type Definitions

**Problem:** Several dataclasses are duplicated or have inconsistent definitions:
- `GateResult` in gating module needs to be importable by sizing module
- `ArbitrationResult` needs richer context

**Fix:** Create `LIVE_TRADING/common/types.py` with shared dataclasses.

---

## Part 2: Explainability Gaps

### Gap 1: No OHLCV Context Capture

**Problem:** When a trade is made, we don't capture the OHLCV data that triggered it.

**Solution:** Create `DecisionContext` dataclass:

```python
# LIVE_TRADING/common/types.py

@dataclass
class MarketSnapshot:
    """Market data at decision time."""
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
    volatility: float  # Annualized

    @classmethod
    def from_quote_and_prices(cls, symbol: str, quote: dict, prices: pd.DataFrame) -> "MarketSnapshot":
        return cls(
            symbol=symbol,
            timestamp=datetime.now(),
            open=float(prices["Open"].iloc[-1]),
            high=float(prices["High"].iloc[-1]),
            low=float(prices["Low"].iloc[-1]),
            close=float(prices["Close"].iloc[-1]),
            volume=float(prices["Volume"].iloc[-1]),
            bid=quote["bid"],
            ask=quote["ask"],
            spread_bps=quote.get("spread_bps", 0),
            volatility=float(prices["Close"].pct_change().std() * (252 ** 0.5)),
        )
```

---

### Gap 2: No Pipeline Stage Trace

**Problem:** Can't see what each stage contributed to the decision.

**Solution:** Create `PipelineTrace`:

```python
@dataclass
class PipelineTrace:
    """Full trace of decision pipeline."""
    market_snapshot: MarketSnapshot

    # Stage 1: Predictions
    predictions: Dict[str, Dict[str, float]]  # horizon -> family -> value
    standardized: Dict[str, Dict[str, float]]
    confidences: Dict[str, Dict[str, float]]

    # Stage 2: Blending
    blend_weights: Dict[str, Dict[str, float]]  # horizon -> family -> weight
    blended_alphas: Dict[str, float]  # horizon -> alpha

    # Stage 3: Arbitration
    horizon_scores: Dict[str, float]
    selected_horizon: str
    costs: Dict[str, float]  # cost breakdown

    # Stage 4: Gating
    barrier_gate: Dict[str, float]  # p_peak, p_valley, gate_value
    spread_gate: Dict[str, Any]

    # Stage 5: Sizing
    raw_weight: float
    gate_adjusted_weight: float
    final_weight: float

    # Stage 6: Risk
    risk_checks: Dict[str, bool]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return asdict(self)

    def explain(self) -> str:
        """Generate human-readable explanation."""
        lines = [
            f"Decision Trace for {self.market_snapshot.symbol}",
            f"=" * 50,
            f"Market: Close={self.market_snapshot.close:.2f}, Vol={self.market_snapshot.volatility:.2%}",
            f"Spread: {self.market_snapshot.spread_bps:.1f} bps",
            f"",
            f"Selected Horizon: {self.selected_horizon}",
            f"Horizon Scores: {self.horizon_scores}",
            f"",
            f"Costs: spread={self.costs.get('spread', 0):.1f}, timing={self.costs.get('timing', 0):.1f}",
            f"",
            f"Barrier Gate: p_peak={self.barrier_gate.get('p_peak', 0):.2f}",
            f"",
            f"Final Weight: {self.final_weight:.2%}",
        ]
        return "\n".join(lines)
```

---

### Gap 3: TradeDecision Too Simple

**Problem:** Current `TradeDecision` doesn't include enough context.

**Solution:** Enhance with trace:

```python
@dataclass
class TradeDecision:
    """Result of a trading cycle with full context."""
    symbol: str
    decision: str  # TRADE, HOLD, BLOCKED
    horizon: Optional[str]
    target_weight: float
    current_weight: float
    alpha: float
    shares: int
    reason: str

    # NEW: Full explainability
    trace: Optional[PipelineTrace] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def explain(self) -> str:
        """Get human-readable explanation."""
        if self.trace:
            return self.trace.explain()
        return f"{self.decision}: {self.reason}"

    def to_audit_record(self) -> Dict[str, Any]:
        """Convert to audit log record."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "decision": self.decision,
            "reason": self.reason,
            "alpha": self.alpha,
            "shares": self.shares,
            "trace": self.trace.to_dict() if self.trace else None,
        }
```

---

### Gap 4: No Audit Trail

**Problem:** No persistent log of decisions with context.

**Solution:** Add `DecisionLogger`:

```python
# LIVE_TRADING/common/audit.py

class DecisionLogger:
    """Logs all trade decisions with full context."""

    def __init__(self, log_dir: Path = Path("logs/decisions")):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_decision(self, decision: TradeDecision) -> None:
        """Log a decision to daily file."""
        date_str = decision.timestamp.strftime("%Y-%m-%d")
        log_file = self.log_dir / f"{date_str}.jsonl"

        record = decision.to_audit_record()

        with log_file.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def get_decisions(
        self,
        date: str,
        symbol: Optional[str] = None,
        decision_type: Optional[str] = None,
    ) -> List[Dict]:
        """Retrieve decisions from log."""
        log_file = self.log_dir / f"{date}.jsonl"
        if not log_file.exists():
            return []

        decisions = []
        with log_file.open() as f:
            for line in f:
                record = json.loads(line)
                if symbol and record["symbol"] != symbol:
                    continue
                if decision_type and record["decision"] != decision_type:
                    continue
                decisions.append(record)

        return decisions
```

---

## Part 3: Plugin/Hook Architecture

### New Plan: 00_hooks_and_plugins.md

Create extensibility through hooks and plugins:

```python
# LIVE_TRADING/common/hooks.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

@dataclass
class HookResult:
    """Result from a hook execution."""
    continue_pipeline: bool = True
    modified_data: Optional[Any] = None
    metadata: Dict[str, Any] = None


class Hook(ABC):
    """Base class for pipeline hooks."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Hook identifier."""
        ...

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> HookResult:
        """Execute hook logic."""
        ...


class HookRegistry:
    """Registry for pipeline hooks."""

    # Hook points in the pipeline
    HOOK_POINTS = [
        "pre_cycle",
        "post_data_fetch",
        "post_prediction",
        "post_blend",
        "post_arbitration",
        "post_gate",
        "post_sizing",
        "pre_trade",
        "post_trade",
        "post_cycle",
    ]

    def __init__(self):
        self._hooks: Dict[str, List[Hook]] = {p: [] for p in self.HOOK_POINTS}

    def register(self, point: str, hook: Hook) -> None:
        """Register a hook at a specific point."""
        if point not in self.HOOK_POINTS:
            raise ValueError(f"Unknown hook point: {point}")
        self._hooks[point].append(hook)

    def execute(self, point: str, context: Dict[str, Any]) -> HookResult:
        """Execute all hooks at a point."""
        for hook in self._hooks[point]:
            result = hook.execute(context)
            if not result.continue_pipeline:
                return result
            if result.modified_data is not None:
                context["data"] = result.modified_data

        return HookResult(continue_pipeline=True)


# Example hook implementations

class LoggingHook(Hook):
    """Logs pipeline events."""

    @property
    def name(self) -> str:
        return "logging"

    def execute(self, context: Dict[str, Any]) -> HookResult:
        import logging
        logger = logging.getLogger("hooks")
        logger.info(f"Hook: {context.get('hook_point')} | Symbol: {context.get('symbol')}")
        return HookResult()


class MetricsHook(Hook):
    """Collects metrics for monitoring."""

    def __init__(self, metrics_collector):
        self.metrics = metrics_collector

    @property
    def name(self) -> str:
        return "metrics"

    def execute(self, context: Dict[str, Any]) -> HookResult:
        self.metrics.record(
            hook_point=context.get("hook_point"),
            symbol=context.get("symbol"),
            timestamp=context.get("timestamp"),
        )
        return HookResult()


class CustomGateHook(Hook):
    """Custom gating logic via hook."""

    def __init__(self, gate_fn: Callable[[Dict], bool]):
        self.gate_fn = gate_fn

    @property
    def name(self) -> str:
        return "custom_gate"

    def execute(self, context: Dict[str, Any]) -> HookResult:
        if not self.gate_fn(context):
            return HookResult(
                continue_pipeline=False,
                metadata={"reason": "custom_gate_blocked"}
            )
        return HookResult()
```

### Plugin Interface for Custom Components

```python
# LIVE_TRADING/common/plugins.py

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class GatePlugin(ABC):
    """Plugin interface for custom gates."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def evaluate(
        self,
        symbol: str,
        alpha: float,
        context: Dict[str, Any],
    ) -> tuple[bool, float, str]:
        """
        Evaluate gate.

        Returns:
            (allowed, gate_value, reason)
        """
        ...


class SizerPlugin(ABC):
    """Plugin interface for custom sizing."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def calculate_weight(
        self,
        alpha: float,
        volatility: float,
        context: Dict[str, Any],
    ) -> float:
        """Calculate target weight."""
        ...


class PluginRegistry:
    """Registry for plugins."""

    def __init__(self):
        self._gates: Dict[str, GatePlugin] = {}
        self._sizers: Dict[str, SizerPlugin] = {}

    def register_gate(self, plugin: GatePlugin) -> None:
        self._gates[plugin.name] = plugin

    def register_sizer(self, plugin: SizerPlugin) -> None:
        self._sizers[plugin.name] = plugin

    def get_gate(self, name: str) -> Optional[GatePlugin]:
        return self._gates.get(name)

    def get_sizer(self, name: str) -> Optional[SizerPlugin]:
        return self._sizers.get(name)
```

---

## Part 4: MCP Server Design for Development Tracking

### Purpose

An MCP server to help track:
1. Implementation progress (files created, tests passing)
2. SST compliance verification
3. Determinism checks
4. Integration test status
5. Explainability verification

### Server Structure

```
MCP_SERVERS/foxml-live-dev/
├── __init__.py
├── server.py
├── tools/
│   ├── __init__.py
│   ├── progress.py      # Implementation progress tracking
│   ├── compliance.py    # SST compliance checks
│   ├── determinism.py   # Determinism verification
│   ├── explain.py       # Decision explainability tools
│   └── integration.py   # Integration test tools
└── data/
    └── progress.json    # Persistent progress state
```

### Tool Definitions

```python
# MCP_SERVERS/foxml-live-dev/server.py

TOOLS = [
    # Progress Tracking
    {
        "name": "check_implementation_progress",
        "description": "Check which files from the plans have been implemented",
        "parameters": {
            "plan": {"type": "string", "description": "Plan number (e.g., '01', '02') or 'all'"}
        }
    },
    {
        "name": "mark_file_complete",
        "description": "Mark a file as implemented",
        "parameters": {
            "file_path": {"type": "string", "description": "Relative path to file"},
            "tests_passing": {"type": "boolean", "description": "Whether tests pass"}
        }
    },

    # SST Compliance
    {
        "name": "check_sst_compliance",
        "description": "Check a file for SST compliance issues",
        "parameters": {
            "file_path": {"type": "string", "description": "Path to check"}
        }
    },
    {
        "name": "find_hardcoded_config",
        "description": "Find hardcoded config values that should use get_cfg()",
        "parameters": {
            "directory": {"type": "string", "description": "Directory to scan"}
        }
    },

    # Determinism
    {
        "name": "check_determinism_violations",
        "description": "Find potential determinism violations in LIVE_TRADING code",
        "parameters": {}
    },
    {
        "name": "verify_sorted_items_usage",
        "description": "Verify all dict iterations use sorted_items()",
        "parameters": {
            "file_path": {"type": "string", "description": "File to check"}
        }
    },

    # Explainability
    {
        "name": "get_decision_trace",
        "description": "Get full decision trace for a trade",
        "parameters": {
            "date": {"type": "string", "description": "Date (YYYY-MM-DD)"},
            "symbol": {"type": "string", "description": "Symbol"},
            "index": {"type": "integer", "description": "Decision index", "default": 0}
        }
    },
    {
        "name": "explain_trade",
        "description": "Get human-readable explanation of a trade decision",
        "parameters": {
            "trace_id": {"type": "string", "description": "Trace ID from decision log"}
        }
    },

    # Integration
    {
        "name": "run_integration_check",
        "description": "Check integration between two modules",
        "parameters": {
            "module1": {"type": "string", "description": "First module"},
            "module2": {"type": "string", "description": "Second module"}
        }
    },
    {
        "name": "verify_interfaces",
        "description": "Verify that module interfaces match expected signatures",
        "parameters": {
            "module": {"type": "string", "description": "Module to check"}
        }
    }
]
```

### Implementation Plan for MCP Server

```python
# MCP_SERVERS/foxml-live-dev/tools/progress.py

import json
from pathlib import Path
from typing import Dict, List, Any

# Expected files from each plan
PLAN_FILES = {
    "01": [
        "LIVE_TRADING/__init__.py",
        "LIVE_TRADING/common/__init__.py",
        "LIVE_TRADING/common/exceptions.py",
        "LIVE_TRADING/common/constants.py",
    ],
    "02": [
        "LIVE_TRADING/brokers/__init__.py",
        "LIVE_TRADING/brokers/interface.py",
        "LIVE_TRADING/brokers/paper.py",
        "LIVE_TRADING/brokers/data_provider.py",
    ],
    # ... etc for all plans
}


def check_implementation_progress(plan: str = "all") -> Dict[str, Any]:
    """Check which files have been implemented."""
    progress = {
        "total_files": 0,
        "implemented": 0,
        "missing": [],
        "by_plan": {}
    }

    plans_to_check = PLAN_FILES.keys() if plan == "all" else [plan]

    for p in plans_to_check:
        files = PLAN_FILES.get(p, [])
        plan_progress = {"total": len(files), "done": 0, "missing": []}

        for file_path in files:
            progress["total_files"] += 1
            if Path(file_path).exists():
                progress["implemented"] += 1
                plan_progress["done"] += 1
            else:
                progress["missing"].append(file_path)
                plan_progress["missing"].append(file_path)

        progress["by_plan"][p] = plan_progress

    progress["completion_pct"] = (
        progress["implemented"] / progress["total_files"] * 100
        if progress["total_files"] > 0 else 0
    )

    return progress
```

```python
# MCP_SERVERS/foxml-live-dev/tools/compliance.py

import ast
import re
from pathlib import Path
from typing import List, Dict, Any


def check_sst_compliance(file_path: str) -> Dict[str, Any]:
    """Check a file for SST compliance issues."""
    issues = []
    path = Path(file_path)

    if not path.exists():
        return {"error": f"File not found: {file_path}"}

    content = path.read_text()

    # Check 1: repro_bootstrap import (for entry points)
    if "if __name__" in content or file_path.endswith("__init__.py"):
        if "import TRAINING.common.repro_bootstrap" not in content:
            # Check if it's the module's main __init__
            if "LIVE_TRADING/__init__.py" in file_path:
                issues.append({
                    "type": "missing_repro_bootstrap",
                    "message": "Entry point missing repro_bootstrap import",
                    "severity": "error"
                })

    # Check 2: Hardcoded config values
    # Look for common hardcoded patterns
    hardcoded_patterns = [
        (r'= 0\.\d+', "Possible hardcoded float"),
        (r'= \d+\.?\d*  # (?!shares|qty)', "Possible hardcoded number"),
    ]

    # Check 3: Dict iteration without sorted_items
    if "for " in content and ".items()" in content:
        if "sorted_items" not in content:
            issues.append({
                "type": "unsorted_dict_iteration",
                "message": "Dict iteration without sorted_items()",
                "severity": "warning"
            })

    # Check 4: get_cfg usage
    if "config" in content.lower() or "cfg" in content.lower():
        if "get_cfg" not in content:
            issues.append({
                "type": "missing_get_cfg",
                "message": "Config access without get_cfg()",
                "severity": "warning"
            })

    return {
        "file": file_path,
        "issues": issues,
        "compliant": len([i for i in issues if i["severity"] == "error"]) == 0
    }
```

```python
# MCP_SERVERS/foxml-live-dev/tools/explain.py

import json
from pathlib import Path
from typing import Dict, Any, Optional


def get_decision_trace(
    date: str,
    symbol: str,
    index: int = 0,
    log_dir: str = "logs/decisions"
) -> Dict[str, Any]:
    """Get full decision trace for a trade."""
    log_file = Path(log_dir) / f"{date}.jsonl"

    if not log_file.exists():
        return {"error": f"No decisions logged for {date}"}

    matching = []
    with log_file.open() as f:
        for line in f:
            record = json.loads(line)
            if record.get("symbol") == symbol:
                matching.append(record)

    if not matching:
        return {"error": f"No decisions for {symbol} on {date}"}

    if index >= len(matching):
        return {"error": f"Only {len(matching)} decisions, requested index {index}"}

    return matching[index]


def explain_trade(trace: Dict[str, Any]) -> str:
    """Generate human-readable explanation."""
    if "error" in trace:
        return trace["error"]

    lines = [
        f"Trade Decision Explanation",
        f"=" * 50,
        f"",
        f"Symbol: {trace.get('symbol')}",
        f"Time: {trace.get('timestamp')}",
        f"Decision: {trace.get('decision')}",
        f"Reason: {trace.get('reason')}",
        f"",
    ]

    if trace.get("trace"):
        t = trace["trace"]

        lines.extend([
            f"Market Context:",
            f"  Close: ${t.get('market_snapshot', {}).get('close', 'N/A')}",
            f"  Spread: {t.get('market_snapshot', {}).get('spread_bps', 'N/A')} bps",
            f"  Volatility: {t.get('market_snapshot', {}).get('volatility', 'N/A'):.2%}",
            f"",
            f"Horizon Analysis:",
        ])

        for h, score in sorted(t.get("horizon_scores", {}).items()):
            lines.append(f"  {h}: {score:.2f}")

        lines.extend([
            f"",
            f"Selected: {t.get('selected_horizon')}",
            f"Alpha: {trace.get('alpha', 0) * 10000:.1f} bps",
            f"Shares: {trace.get('shares')}",
        ])

    return "\n".join(lines)
```

---

## Part 5: New Files to Add

Based on this review, add these files to the plans:

### New File: `LIVE_TRADING/common/types.py`

Add to Plan 01:

```python
"""
Common Type Definitions
=======================

Shared dataclasses used across LIVE_TRADING modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class MarketSnapshot:
    """Market data at decision time."""
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
    volatility: float

    @classmethod
    def from_quote_and_prices(
        cls,
        symbol: str,
        quote: Dict[str, Any],
        prices: pd.DataFrame,
    ) -> "MarketSnapshot":
        """Create from quote and price data."""
        return cls(
            symbol=symbol,
            timestamp=datetime.now(),
            open=float(prices["Open"].iloc[-1]),
            high=float(prices["High"].iloc[-1]),
            low=float(prices["Low"].iloc[-1]),
            close=float(prices["Close"].iloc[-1]),
            volume=float(prices["Volume"].iloc[-1]),
            bid=quote["bid"],
            ask=quote["ask"],
            spread_bps=quote.get("spread_bps", 0),
            volatility=float(prices["Close"].pct_change().std() * (252 ** 0.5)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass
class PipelineTrace:
    """Full trace of decision pipeline for explainability."""
    market_snapshot: MarketSnapshot

    # Predictions
    raw_predictions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    standardized_predictions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confidences: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Blending
    blend_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)
    blended_alphas: Dict[str, float] = field(default_factory=dict)

    # Arbitration
    horizon_scores: Dict[str, float] = field(default_factory=dict)
    cost_breakdown: Dict[str, float] = field(default_factory=dict)
    selected_horizon: Optional[str] = None

    # Gating
    barrier_gate_result: Dict[str, Any] = field(default_factory=dict)
    spread_gate_result: Dict[str, Any] = field(default_factory=dict)

    # Sizing
    raw_weight: float = 0.0
    gate_adjusted_weight: float = 0.0
    final_weight: float = 0.0

    # Risk
    risk_checks: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        d = asdict(self)
        d["market_snapshot"] = self.market_snapshot.to_dict()
        return d

    def explain(self) -> str:
        """Generate human-readable explanation."""
        lines = [
            f"Pipeline Trace: {self.market_snapshot.symbol}",
            "=" * 50,
            "",
            "MARKET CONTEXT",
            f"  Time: {self.market_snapshot.timestamp}",
            f"  OHLCV: O={self.market_snapshot.open:.2f} H={self.market_snapshot.high:.2f} "
            f"L={self.market_snapshot.low:.2f} C={self.market_snapshot.close:.2f} "
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
            lines.append(f"  {h}: alpha={alpha:.1f}bps, score={score:.2f}")

        lines.extend([
            "",
            f"SELECTED: {self.selected_horizon}",
            "",
            "COSTS",
            f"  Spread: {self.cost_breakdown.get('spread', 0):.1f} bps",
            f"  Timing: {self.cost_breakdown.get('timing', 0):.1f} bps",
            f"  Impact: {self.cost_breakdown.get('impact', 0):.1f} bps",
            "",
            "GATING",
            f"  Barrier: p_peak={self.barrier_gate_result.get('p_peak', 0):.2f}",
            f"  Spread: {self.spread_gate_result.get('allowed', True)}",
            "",
            "SIZING",
            f"  Raw Weight: {self.raw_weight:.4f}",
            f"  Gate Adjusted: {self.gate_adjusted_weight:.4f}",
            f"  Final Weight: {self.final_weight:.4f}",
        ])

        return "\n".join(lines)
```

### New File: `LIVE_TRADING/common/hooks.py`

Add to Plan 01 (hooks and plugin system - see Part 3 above).

### New File: `LIVE_TRADING/common/audit.py`

Add to Plan 01 (decision logging - see Gap 4 above).

---

## Summary of Required Changes

### Plan 01 Amendments
- Add `types.py` with `MarketSnapshot`, `PipelineTrace`
- Add `hooks.py` with hook system
- Add `audit.py` with `DecisionLogger`
- Add `plugins.py` with plugin interfaces

### Plan 10 Amendments
- Fix `PositionState` import
- Integrate barrier predictions properly
- Add `PipelineTrace` capture in `_process_symbol()`
- Add hook execution at each pipeline stage
- Add decision logging

### All Plans
- Verify `sorted_items()` usage for all dict iterations
- Verify proper datetime serialization

---

## MCP Server Implementation Priority

1. **Phase 1:** Progress tracking tools (check_implementation_progress)
2. **Phase 2:** SST compliance tools (check_sst_compliance, find_hardcoded_config)
3. **Phase 3:** Determinism tools (check_determinism_violations)
4. **Phase 4:** Explainability tools (get_decision_trace, explain_trade)
