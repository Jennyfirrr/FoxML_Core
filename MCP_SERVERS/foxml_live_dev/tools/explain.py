"""
Trade Decision Explainability
=============================

Tools for understanding why trade decisions were made.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs" / "decisions"


def get_decision_trace(
    date: str,
    symbol: str,
    index: int = 0,
    log_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get full decision trace for a trade.

    Args:
        date: Date string (YYYY-MM-DD)
        symbol: Trading symbol
        index: Decision index for that symbol on that date (0 = first)
        log_dir: Optional custom log directory

    Returns:
        Dict with decision trace
    """
    log_path = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
    log_file = log_path / f"{date}.jsonl"

    if not log_file.exists():
        return {
            "error": f"No decision log for {date}",
            "log_file": str(log_file),
            "exists": False,
        }

    matching = []
    with log_file.open() as f:
        for line in f:
            try:
                record = json.loads(line)
                if record.get("symbol", "").upper() == symbol.upper():
                    matching.append(record)
            except json.JSONDecodeError:
                continue

    if not matching:
        return {
            "error": f"No decisions for {symbol} on {date}",
            "symbol": symbol,
            "date": date,
        }

    if index >= len(matching):
        return {
            "error": f"Only {len(matching)} decisions for {symbol}, requested index {index}",
            "total_decisions": len(matching),
        }

    return matching[index]


def explain_trade(trace: Dict[str, Any]) -> str:
    """
    Generate human-readable explanation of a trade decision.

    Args:
        trace: Decision trace dict

    Returns:
        Human-readable explanation string
    """
    if "error" in trace:
        return f"Error: {trace['error']}"

    lines = [
        "=" * 60,
        "TRADE DECISION EXPLANATION",
        "=" * 60,
        "",
        f"Symbol: {trace.get('symbol', 'N/A')}",
        f"Time: {trace.get('timestamp', 'N/A')}",
        f"Decision: {trace.get('decision', 'N/A')}",
        f"Reason: {trace.get('reason', 'N/A')}",
        "",
    ]

    # Alpha and shares
    alpha = trace.get("alpha", 0)
    lines.append(f"Alpha: {alpha * 10000:.1f} bps")
    lines.append(f"Shares: {trace.get('shares', 0)}")
    lines.append("")

    # If we have a full trace
    if trace.get("trace"):
        t = trace["trace"]

        # Market context
        if t.get("market_snapshot"):
            ms = t["market_snapshot"]
            lines.extend([
                "MARKET CONTEXT",
                "-" * 40,
                f"  Close: ${ms.get('close', 0):.2f}",
                f"  Bid/Ask: ${ms.get('bid', 0):.2f} / ${ms.get('ask', 0):.2f}",
                f"  Spread: {ms.get('spread_bps', 0):.1f} bps",
                f"  Volatility: {ms.get('volatility', 0):.2%}",
                "",
            ])

        # Horizon analysis
        if t.get("horizon_scores"):
            lines.extend([
                "HORIZON ANALYSIS",
                "-" * 40,
            ])
            for h, score in sorted(t["horizon_scores"].items()):
                alpha_h = t.get("blended_alphas", {}).get(h, 0) * 10000
                lines.append(f"  {h}: alpha={alpha_h:.1f}bps, score={score:.2f}")
            lines.append(f"  Selected: {t.get('selected_horizon', 'N/A')}")
            lines.append("")

        # Cost breakdown
        if t.get("cost_breakdown"):
            costs = t["cost_breakdown"]
            lines.extend([
                "TRADING COSTS",
                "-" * 40,
                f"  Spread cost: {costs.get('spread', 0):.1f} bps",
                f"  Timing cost: {costs.get('timing', 0):.1f} bps",
                f"  Impact cost: {costs.get('impact', 0):.1f} bps",
                f"  Total cost: {costs.get('total', 0):.1f} bps",
                "",
            ])

        # Gating
        if t.get("barrier_gate_result"):
            bg = t["barrier_gate_result"]
            lines.extend([
                "BARRIER GATE",
                "-" * 40,
                f"  P(peak): {bg.get('p_peak', 0):.2f}",
                f"  P(valley): {bg.get('p_valley', 0):.2f}",
                f"  Gate value: {bg.get('gate_value', 1):.2f}",
                f"  Allowed: {bg.get('allowed', True)}",
                "",
            ])

        # Sizing
        lines.extend([
            "POSITION SIZING",
            "-" * 40,
            f"  Raw weight: {t.get('raw_weight', 0):.4f}",
            f"  Gate adjusted: {t.get('gate_adjusted_weight', 0):.4f}",
            f"  Final weight: {t.get('final_weight', 0):.4f}",
            "",
        ])

        # Risk checks
        if t.get("risk_checks"):
            lines.extend([
                "RISK CHECKS",
                "-" * 40,
            ])
            for check, passed in t["risk_checks"].items():
                status = "PASS" if passed else "FAIL"
                lines.append(f"  {check}: {status}")
            lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)


def list_recent_decisions(
    days: int = 7,
    symbol: Optional[str] = None,
    decision_type: Optional[str] = None,
    log_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    List recent trading decisions.

    Args:
        days: Number of days to look back
        symbol: Filter by symbol
        decision_type: Filter by decision type (TRADE, HOLD, BLOCKED)
        log_dir: Optional custom log directory

    Returns:
        Dict with decision summaries
    """
    log_path = Path(log_dir) if log_dir else DEFAULT_LOG_DIR

    if not log_path.exists():
        return {
            "error": "Decision log directory not found",
            "log_dir": str(log_path),
        }

    decisions = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        log_file = log_path / f"{date_str}.jsonl"

        if log_file.exists():
            with log_file.open() as f:
                for line in f:
                    try:
                        record = json.loads(line)

                        # Apply filters
                        if symbol and record.get("symbol", "").upper() != symbol.upper():
                            continue
                        if decision_type and record.get("decision") != decision_type:
                            continue

                        decisions.append({
                            "date": date_str,
                            "timestamp": record.get("timestamp"),
                            "symbol": record.get("symbol"),
                            "decision": record.get("decision"),
                            "reason": record.get("reason"),
                            "alpha": record.get("alpha"),
                            "shares": record.get("shares"),
                        })
                    except json.JSONDecodeError:
                        continue

        current += timedelta(days=1)

    # Summary stats
    by_decision = {}
    by_symbol = {}

    for d in decisions:
        dec = d["decision"]
        sym = d["symbol"]

        by_decision[dec] = by_decision.get(dec, 0) + 1
        by_symbol[sym] = by_symbol.get(sym, 0) + 1

    return {
        "period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        "total_decisions": len(decisions),
        "by_decision_type": by_decision,
        "by_symbol": by_symbol,
        "recent_decisions": decisions[-20:],  # Last 20
    }


def compare_decisions(
    date1: str,
    date2: str,
    symbol: str,
    log_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare decisions for a symbol across two dates.

    Args:
        date1: First date
        date2: Second date
        symbol: Symbol to compare
        log_dir: Optional custom log directory

    Returns:
        Comparison dict
    """
    trace1 = get_decision_trace(date1, symbol, log_dir=log_dir)
    trace2 = get_decision_trace(date2, symbol, log_dir=log_dir)

    if "error" in trace1:
        return {"error": f"Date 1: {trace1['error']}"}
    if "error" in trace2:
        return {"error": f"Date 2: {trace2['error']}"}

    return {
        "symbol": symbol,
        "date1": {
            "date": date1,
            "decision": trace1.get("decision"),
            "alpha": trace1.get("alpha"),
            "horizon": trace1.get("trace", {}).get("selected_horizon"),
            "reason": trace1.get("reason"),
        },
        "date2": {
            "date": date2,
            "decision": trace2.get("decision"),
            "alpha": trace2.get("alpha"),
            "horizon": trace2.get("trace", {}).get("selected_horizon"),
            "reason": trace2.get("reason"),
        },
        "alpha_change": (
            (trace2.get("alpha", 0) - trace1.get("alpha", 0)) * 10000
            if trace1.get("alpha") and trace2.get("alpha")
            else None
        ),
    }
