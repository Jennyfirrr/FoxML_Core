# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Target Routing Logic

Determines which view (CROSS_SECTIONAL, SYMBOL_SPECIFIC, BOTH, or BLOCKED) each target
should use based on dual-view evaluation results.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

# DETERMINISM: Atomic writes and sorted iteration for reproducibility
from TRAINING.common.utils.file_utils import write_atomic_json
from TRAINING.common.utils.determinism_ordering import sorted_items

logger = logging.getLogger(__name__)


def _compute_target_routing_decisions(
    results_cs: list[Any],  # List of TargetPredictabilityScore (cross-sectional)
    results_sym: dict[str, dict[str, Any]],  # {target: {symbol: TargetPredictabilityScore}}
    results_loso: dict[str, dict[str, Any]],  # {target: {symbol: TargetPredictabilityScore}} (optional)
    symbol_skip_reasons: dict[str, dict[str, dict[str, Any]]] = None,  # {target: {symbol: {reason, status, ...}}}
    experiment_config: Any = None,  # Optional experiment config for SS gate overrides
    total_symbols: int = None  # FIX: Total universe size for SS gate (not just evaluated symbols)
) -> dict[str, dict[str, Any]]:
    """
    Compute routing decisions for each target based on dual-view scores.

    Uses skill01 (normalized [0,1] score) for unified routing across task types:
    - Regression: skill01 = 0.5 * (IC + 1.0) where IC ∈ [-1, 1]
    - Classification: skill01 = 0.5 * (AUC-excess + 1.0) where AUC-excess ∈ [-0.5, 0.5]

    Routing rules (using skill01 thresholds):
    - CROSS_SECTIONAL only: skill01 >= T_cs AND frac_symbols_good >= T_frac
    - SYMBOL_SPECIFIC only: skill01 < T_cs AND exists symbol with skill01 >= T_sym
    - BOTH: skill01 >= T_cs BUT performance is concentrated (high IQR / low frac_symbols_good)
    - BLOCKED: skill01 >= 0.90 (suspicious) UNLESS tstat > 3.0 (stable signal)

    Args:
        results_cs: Cross-sectional results (TargetPredictabilityScore objects)
        results_sym: Symbol-specific results by target
        results_loso: LOSO results by target (optional)
        symbol_skip_reasons: Skip reasons by target and symbol (optional)
        experiment_config: Optional experiment config for SS gate overrides

    Returns:
        Dict mapping target -> routing decision dict (includes skill01_cs, skill01_sym_mean)
    """
    # Load thresholds from config
    try:
        from CONFIG.config_loader import get_cfg
        routing_cfg = get_cfg("target_ranking.routing", default={}, config_name="target_ranking_config")
        T_cs = float(routing_cfg.get('auc_threshold', 0.65))
        T_frac = float(routing_cfg.get('frac_symbols_good_threshold', 0.5))
        T_sym = float(routing_cfg.get('symbol_auc_threshold', 0.60))
        T_suspicious_cs = float(routing_cfg.get('suspicious_auc', 0.90))
        T_suspicious_sym = float(routing_cfg.get('suspicious_symbol_auc', 0.95))

        # Dev mode: relax thresholds for testing with small datasets (10 symbols, 40k CS samples)
        dev_mode = routing_cfg.get('dev_mode', False)
        if dev_mode:
            T_cs = max(0.40, T_cs - 0.25)  # Lower from 0.65 to 0.40 (allows more CS routes)
            T_sym = max(0.35, T_sym - 0.25)  # Lower from 0.60 to 0.35 (allows more symbol routes)
            T_frac = max(0.2, T_frac - 0.3)  # Lower from 0.5 to 0.2 (less strict symbol coverage)
            T_suspicious_cs = min(0.98, T_suspicious_cs + 0.05)  # Raise from 0.90 to 0.95 (less blocking)
            T_suspicious_sym = min(0.98, T_suspicious_sym + 0.03)  # Raise from 0.95 to 0.98 (less blocking)
            logger.info(f"[DEV] Dev mode enabled: relaxed thresholds (CS={T_cs:.2f}, SYM={T_sym:.2f}, frac={T_frac:.2f}, suspicious_cs={T_suspicious_cs:.2f})")
    except Exception as e:
        # EH-005: Fail-closed in strict mode for config load failures
        from TRAINING.common.determinism import is_strict_mode
        if is_strict_mode():
            from TRAINING.common.exceptions import ConfigError
            raise ConfigError(
                f"Failed to load routing thresholds: {e}",
                config_key="target_ranking.routing",
                stage="ROUTING"
            ) from e
        # Fallback defaults (documented in CONFIG/defaults.yaml)
        T_cs = 0.65
        T_frac = 0.5
        T_sym = 0.60
        T_suspicious_cs = 0.90
        T_suspicious_sym = 0.95
        logger.warning(f"EH-005: Using fallback thresholds T_cs={T_cs}, T_frac={T_frac}: {e}")

    routing_decisions = {}

    # Collect all target names (from both CS results and symbol-specific results)
    # This ensures we process targets even if CS failed but symbol-specific succeeded
    all_targets = set()
    for result_cs in results_cs:
        all_targets.add(result_cs.target)
    for target in results_sym:
        all_targets.add(target)

    # Process each target (whether it has CS results or not)
    for target in all_targets:
        # Find cross-sectional result if it exists
        result_cs = None
        skill01_cs = 0.0  # Default to failed score (skill01 is [0,1], so 0.0 = failed)
        auc = -999.0  # Keep for backward compatibility (deprecated for routing)
        for r in results_cs:
            if r.target == target:
                result_cs = r
                # Use skill01 for routing (normalized [0,1] score works for both regression and classification)
                skill01_cs = r.skill01 if hasattr(r, 'skill01') and r.skill01 is not None else 0.0
                auc = r.auc  # Keep for backward compatibility
                break

        # Get symbol-specific results for this target
        sym_results = results_sym.get(target, {})

        # Compute symbol distribution stats using skill01
        symbol_skill01s = []
        if sym_results:
            for symbol, result_sym in sym_results.items():
                skill01_val = result_sym.skill01 if hasattr(result_sym, 'skill01') and result_sym.skill01 is not None else None
                if skill01_val is not None and skill01_val > 0.0:  # Valid result (skill01 > 0)
                    symbol_skill01s.append(skill01_val)

        if symbol_skill01s:
            symbol_skill01_mean = np.mean(symbol_skill01s)
            symbol_skill01_median = np.median(symbol_skill01s)
            symbol_skill01_min = np.min(symbol_skill01s)
            symbol_skill01_max = np.max(symbol_skill01s)
            symbol_skill01_iqr = np.percentile(symbol_skill01s, 75) - np.percentile(symbol_skill01s, 25)
            frac_symbols_good = sum(1 for s01 in symbol_skill01s if s01 >= T_sym) / len(symbol_skill01s)
            # DETERMINISM: Sort winner_symbols for artifact consistency
            winner_symbols = sorted([sym for sym, result in sym_results.items()
                            if hasattr(result, 'skill01') and result.skill01 is not None and result.skill01 >= T_sym])
        else:
            symbol_skill01_mean = None
            symbol_skill01_median = None
            symbol_skill01_min = None
            symbol_skill01_max = None
            symbol_skill01_iqr = None
            frac_symbols_good = 0.0
            winner_symbols = []

        # Initialize route to None (will be set by conditions below)
        route = None
        reason = None

        # Initialize route to None (will be set by conditions below)
        route = None
        reason = None

        # Handle case where CS failed (skill01 = 0.0 or result_cs is None)
        if result_cs is None or skill01_cs <= 0.0:
            # CS failed - check if symbol-specific works
            if symbol_skill01s and max(symbol_skill01s) >= T_sym:
                route = "SYMBOL_SPECIFIC"
                reason = f"cs_failed (skill01={skill01_cs:.3f}) BUT exists symbol with skill01 >= {T_sym}"
                winner_symbols_str = ', '.join(winner_symbols[:5])
                if len(winner_symbols) > 5:
                    winner_symbols_str += f", ... ({len(winner_symbols)} total)"
                reason += f" (winners: {winner_symbols_str})"
            else:
                # CS failed and no good symbol-specific results
                route = "BLOCKED"
                if symbol_skill01s:
                    reason = f"cs_failed AND max_symbol_skill01={max(symbol_skill01s):.3f} < {T_sym} (no viable route)"
                else:
                    reason = "cs_failed AND no symbol-specific results (no viable route)"

        # Check for suspicious scores (BLOCKED) - task-aware: high skill01 + low tstat = suspicious
        if route is None and (skill01_cs >= T_suspicious_cs or (symbol_skill01s and max(symbol_skill01s) >= T_suspicious_sym)):
            # Additional check: if tstat available, verify signal stability
            is_suspicious = True
            if result_cs and hasattr(result_cs, 'primary_metric_tstat'):
                tstat = result_cs.primary_metric_tstat
                if tstat is not None and tstat > 3.0:  # Strong, stable signal
                    # High skill01 + high tstat = legitimate strong signal
                    is_suspicious = False
                    logger.debug(f"High skill01 ({skill01_cs:.3f}) but stable (tstat={tstat:.2f}), not blocking")

            if is_suspicious:
                route = "BLOCKED"
                if skill01_cs >= T_suspicious_cs:
                    reason = f"skill01={skill01_cs:.3f} >= {T_suspicious_cs} (suspicious high score)"
                elif symbol_skill01s and max(symbol_skill01s) >= T_suspicious_sym:
                    reason = f"max_symbol_skill01={max(symbol_skill01s):.3f} >= {T_suspicious_sym} (suspicious high score)"
                else:
                    reason = f"skill01={skill01_cs:.3f} or symbol_skill01 >= suspicious threshold"

        # CROSS_SECTIONAL only: strong CS performance + good symbol coverage
        if route is None and skill01_cs >= T_cs and frac_symbols_good >= T_frac:
            route = "CROSS_SECTIONAL"
            reason = f"skill01={skill01_cs:.3f} >= {T_cs} AND frac_symbols_good={frac_symbols_good:.2f} >= {T_frac}"

        # SYMBOL_SPECIFIC only: weak CS but some symbols work
        if route is None and skill01_cs < T_cs and symbol_skill01s and max(symbol_skill01s) >= T_sym:
            route = "SYMBOL_SPECIFIC"
            reason = f"skill01={skill01_cs:.3f} < {T_cs} BUT exists symbol with skill01 >= {T_sym}"
            winner_symbols_str = ', '.join(winner_symbols[:5])
            if len(winner_symbols) > 5:
                winner_symbols_str += f", ... ({len(winner_symbols)} total)"
            reason += f" (winners: {winner_symbols_str})"

        # BOTH: strong CS but concentrated performance
        if route is None and skill01_cs >= T_cs and symbol_skill01s and symbol_skill01_iqr is not None and (symbol_skill01_iqr > 0.15 or frac_symbols_good < T_frac):
            route = "BOTH"
            reason = f"skill01={skill01_cs:.3f} >= {T_cs} BUT concentrated (IQR={symbol_skill01_iqr:.3f}, frac_good={frac_symbols_good:.2f})"

        # Default: CROSS_SECTIONAL (fallback)
        if route is None:
            route = "CROSS_SECTIONAL"
            if len(symbol_skill01s) == 0:
                reason = f"default (skill01={skill01_cs:.3f}, symbol_eval=0 symbols evaluable)"
            else:
                reason = f"default (skill01={skill01_cs:.3f}, no strong symbol-specific signal)"

        # SS Universe Size Gate: Disable SS routing for large universes
        # SS creates one model per symbol, which negates memory savings for large universes
        ss_gated = False
        # FIX: Use total universe size, not just evaluated symbols
        # This ensures the gate applies consistently regardless of evaluation success rate
        universe_size = total_symbols if total_symbols is not None else (len(sym_results) if sym_results else 0)
        if route in ("SYMBOL_SPECIFIC", "BOTH") and universe_size > 0:
            # CRITICAL: Check experiment config first (has precedence over base config)
            max_symbols_for_ss = None
            ss_fallback = None

            if experiment_config:
                exp_name = experiment_config.name if hasattr(experiment_config, 'name') else str(experiment_config)
                try:
                    exp_file = Path("CONFIG/experiments") / f"{exp_name}.yaml"
                    if exp_file.exists():
                        import yaml
                        with open(exp_file, 'r') as f:
                            exp_yaml = yaml.safe_load(f) or {}
                        routing_cfg = exp_yaml.get('target_routing', {})
                        if 'max_symbols_for_ss' in routing_cfg:
                            max_symbols_for_ss = int(routing_cfg['max_symbols_for_ss'])
                            ss_fallback = str(routing_cfg.get('ss_fallback_route', 'CROSS_SECTIONAL'))
                            logger.debug(f"SS gate config from experiment: max_symbols_for_ss={max_symbols_for_ss}")
                except Exception as e:
                    logger.debug(f"Could not load SS gate config from experiment: {e}")

            # Fallback to base config if not set by experiment
            if max_symbols_for_ss is None:
                try:
                    max_symbols_for_ss = int(get_cfg("target_routing.max_symbols_for_ss", default=100))
                    ss_fallback = str(get_cfg("target_routing.ss_fallback_route", default="CROSS_SECTIONAL"))
                except Exception:
                    max_symbols_for_ss = 100
                    ss_fallback = "CROSS_SECTIONAL"

            if universe_size > max_symbols_for_ss:
                original_route = route
                route = ss_fallback
                ss_gated = True
                logger.warning(
                    f"🚫 SS gated for {target}: universe size {universe_size} > {max_symbols_for_ss}. "
                    f"Route changed: {original_route} → {route}"
                )
                reason = f"SS_GATED (was {original_route}): universe={universe_size} > max={max_symbols_for_ss}. " + reason

        # Get skip reasons for this target
        target_skip_reasons = {}
        if symbol_skip_reasons and target in symbol_skip_reasons:
            target_skip_reasons = symbol_skip_reasons[target]

        # Extract auto_fix_reason and T-stat from result_cs using existing SST solutions
        auto_fix_reason_for_routing = None
        tstat_cs_for_routing = None
        if result_cs:
            auto_fix_reason_for_routing = getattr(result_cs, 'auto_fix_reason', None)
            tstat_cs_for_routing = getattr(result_cs, 'primary_metric_tstat', None)

        routing_decisions[target] = {
            'route': route,
            'reason': reason,  # Keep existing reason string unchanged (backward compat)
            'skill01_cs': skill01_cs,  # New: normalized skill score for routing
            'skill01_sym_mean': symbol_skill01_mean,  # New: mean symbol skill01
            'auc': auc,  # Deprecated: kept for backward compatibility (R² for regression, AUC for classification)
            'symbol_auc_mean': symbol_skill01_mean,  # Deprecated: now contains skill01_mean, kept for backward compat
            'symbol_auc_median': symbol_skill01_median,  # Deprecated: now contains skill01_median
            'symbol_auc_min': symbol_skill01_min,  # Deprecated: now contains skill01_min
            'symbol_auc_max': symbol_skill01_max,  # Deprecated: now contains skill01_max
            'symbol_auc_iqr': symbol_skill01_iqr,  # Deprecated: now contains skill01_iqr
            'frac_symbols_good': frac_symbols_good,
            'winner_symbols': winner_symbols,
            'n_symbols_evaluated': len(symbol_skill01s) if symbol_skill01s else 0,
            'symbol_skip_reasons': target_skip_reasons if target_skip_reasons else None,
            'auto_fix_reason': auto_fix_reason_for_routing,  # NEW: Structured skip reason
            'tstat_cs': tstat_cs_for_routing  # NEW: T-stat for routing traceability
        }

    return routing_decisions


def _compute_single_target_routing_decision(
    target: str,
    result_cs: Any | None,  # TargetPredictabilityScore or None
    sym_results: dict[str, Any],  # {symbol: TargetPredictabilityScore}
    symbol_skip_reasons: dict[str, dict[str, Any]] | None = None,  # {symbol: {reason, status, ...}}
    experiment_config: Any = None,  # Optional experiment config for SS gate overrides
    total_symbols: int = None  # FIX: Total universe size for SS gate (not just evaluated symbols)
) -> dict[str, Any]:
    """
    Compute routing decision for a single target.

    This is a single-target version of _compute_target_routing_decisions for incremental saving.
    Uses skill01 (normalized [0,1] score) for unified routing across task types.

    Args:
        target: Target name
        result_cs: Cross-sectional result (TargetPredictabilityScore or None if failed)
        sym_results: Symbol-specific results for this target
        symbol_skip_reasons: Skip reasons for symbols (optional)
        experiment_config: Optional experiment config for SS gate overrides

    Returns:
        Routing decision dict for this target (includes skill01_cs, skill01_sym_mean)
    """
    # Load thresholds from config (support both new skill01_threshold and legacy auc_threshold)
    try:
        from CONFIG.config_loader import get_cfg
        routing_cfg = get_cfg("target_ranking.routing", default={}, config_name="target_ranking_config")
        # New unified skill01 thresholds (works for both regression IC and classification AUC)
        T_cs = float(routing_cfg.get('skill01_threshold', routing_cfg.get('auc_threshold', 0.65)))
        T_frac = float(routing_cfg.get('frac_symbols_good_threshold', 0.5))
        T_sym = float(routing_cfg.get('symbol_skill01_threshold', routing_cfg.get('symbol_auc_threshold', 0.60)))
        T_suspicious_cs = float(routing_cfg.get('suspicious_skill01', routing_cfg.get('suspicious_auc', 0.90)))
        T_suspicious_sym = float(routing_cfg.get('suspicious_symbol_skill01', routing_cfg.get('suspicious_symbol_auc', 0.95)))

        # Dev mode: relax thresholds for testing with small datasets
        dev_mode = routing_cfg.get('dev_mode', False)
        if dev_mode:
            T_cs = max(0.40, T_cs - 0.25)  # Lower from 0.65 to 0.40 (allows more CS routes)
            T_sym = max(0.35, T_sym - 0.25)  # Lower from 0.60 to 0.35 (allows more symbol routes)
            T_frac = max(0.2, T_frac - 0.3)  # Lower from 0.5 to 0.2 (less strict symbol coverage)
            T_suspicious_cs = min(0.98, T_suspicious_cs + 0.05)  # Raise from 0.90 to 0.95 (less blocking)
            T_suspicious_sym = min(0.98, T_suspicious_sym + 0.03)  # Raise from 0.95 to 0.98 (less blocking)
            logger.debug(f"Dev mode enabled for single target: relaxed thresholds (CS={T_cs:.2f}, SYM={T_sym:.2f})")
    except Exception as e:
        # EH-005: Fail-closed in strict mode for config load failures
        from TRAINING.common.determinism import is_strict_mode
        if is_strict_mode():
            from TRAINING.common.exceptions import ConfigError
            raise ConfigError(
                f"Failed to load routing thresholds: {e}",
                config_key="target_ranking.routing",
                stage="ROUTING"
            ) from e
        # Fallback defaults (documented in CONFIG/defaults.yaml)
        T_cs = 0.65
        T_frac = 0.5
        T_sym = 0.60
        T_suspicious_cs = 0.90
        T_suspicious_sym = 0.95
        logger.warning(f"EH-005: Using fallback thresholds T_cs={T_cs}, T_frac={T_frac}: {e}")

    # Get CS skill01 (normalized [0,1] score for unified routing)
    skill01_cs = 0.0  # Default to failed score
    auc = -999.0  # Keep for backward compatibility (deprecated for routing)
    if result_cs:
        skill01_cs = result_cs.skill01 if hasattr(result_cs, 'skill01') and result_cs.skill01 is not None else 0.0
        auc = result_cs.auc  # Keep for backward compatibility

    # Compute symbol distribution stats using skill01
    symbol_skill01s = []
    if sym_results:
        for symbol, result_sym in sym_results.items():
            skill01_val = result_sym.skill01 if hasattr(result_sym, 'skill01') and result_sym.skill01 is not None else None
            if skill01_val is not None and skill01_val > 0.0:  # Valid result (skill01 > 0)
                symbol_skill01s.append(skill01_val)

    if symbol_skill01s:
        symbol_skill01_mean = np.mean(symbol_skill01s)
        symbol_skill01_median = np.median(symbol_skill01s)
        symbol_skill01_min = np.min(symbol_skill01s)
        symbol_skill01_max = np.max(symbol_skill01s)
        symbol_skill01_iqr = np.percentile(symbol_skill01s, 75) - np.percentile(symbol_skill01s, 25)
        frac_symbols_good = sum(1 for s01 in symbol_skill01s if s01 >= T_sym) / len(symbol_skill01s)
        # DETERMINISM: Sort winner_symbols for artifact consistency
        winner_symbols = sorted([sym for sym, result in sym_results.items()
                        if hasattr(result, 'skill01') and result.skill01 is not None and result.skill01 >= T_sym])
    else:
        symbol_skill01_mean = None
        symbol_skill01_median = None
        symbol_skill01_min = None
        symbol_skill01_max = None
        symbol_skill01_iqr = None
        frac_symbols_good = 0.0
        winner_symbols = []

    # Initialize route to None (will be set by conditions below)
    route = None
    reason = None

    # Handle case where CS failed (skill01 = 0.0 or result_cs is None)
    if result_cs is None or skill01_cs <= 0.0:
        # CS failed - check if symbol-specific works
        if symbol_skill01s and max(symbol_skill01s) >= T_sym:
            route = "SYMBOL_SPECIFIC"
            reason = f"cs_failed (skill01={skill01_cs:.3f}) BUT exists symbol with skill01 >= {T_sym}"
            winner_symbols_str = ', '.join(winner_symbols[:5])
            if len(winner_symbols) > 5:
                winner_symbols_str += f", ... ({len(winner_symbols)} total)"
            reason += f" (winners: {winner_symbols_str})"
        else:
            # CS failed and no good symbol-specific results
            route = "BLOCKED"
            if symbol_skill01s:
                reason = f"cs_failed AND max_symbol_skill01={max(symbol_skill01s):.3f} < {T_sym} (no viable route)"
            else:
                reason = "cs_failed AND no symbol-specific results (no viable route)"

    # Check for suspicious scores (BLOCKED) - task-aware: high skill01 + low tstat = suspicious
    if route is None and (skill01_cs >= T_suspicious_cs or (symbol_skill01s and max(symbol_skill01s) >= T_suspicious_sym)):
        # Additional check: if tstat available, verify signal stability
        is_suspicious = True
        if result_cs and hasattr(result_cs, 'primary_metric_tstat'):
            tstat = result_cs.primary_metric_tstat
            if tstat is not None and tstat > 3.0:  # Strong, stable signal
                # High skill01 + high tstat = legitimate strong signal
                is_suspicious = False
                logger.debug(f"High skill01 ({skill01_cs:.3f}) but stable (tstat={tstat:.2f}), not blocking")

        if is_suspicious:
            route = "BLOCKED"
            if skill01_cs >= T_suspicious_cs:
                reason = f"skill01={skill01_cs:.3f} >= {T_suspicious_cs} (suspicious high score)"
            elif symbol_skill01s and max(symbol_skill01s) >= T_suspicious_sym:
                reason = f"max_symbol_skill01={max(symbol_skill01s):.3f} >= {T_suspicious_sym} (suspicious high score)"
            else:
                reason = f"skill01={skill01_cs:.3f} or symbol_skill01 >= suspicious threshold"

    # CROSS_SECTIONAL only: strong CS performance + good symbol coverage
    if route is None and skill01_cs >= T_cs and frac_symbols_good >= T_frac:
        route = "CROSS_SECTIONAL"
        reason = f"skill01={skill01_cs:.3f} >= {T_cs} AND frac_symbols_good={frac_symbols_good:.2f} >= {T_frac}"

    # SYMBOL_SPECIFIC only: weak CS but some symbols work
    if route is None and skill01_cs < T_cs and symbol_skill01s and max(symbol_skill01s) >= T_sym:
        route = "SYMBOL_SPECIFIC"
        reason = f"skill01={skill01_cs:.3f} < {T_cs} BUT exists symbol with skill01 >= {T_sym}"
        winner_symbols_str = ', '.join(winner_symbols[:5])
        if len(winner_symbols) > 5:
            winner_symbols_str += f", ... ({len(winner_symbols)} total)"
        reason += f" (winners: {winner_symbols_str})"

    # BOTH: strong CS but concentrated performance
    if route is None and skill01_cs >= T_cs and symbol_skill01s and symbol_skill01_iqr is not None and (symbol_skill01_iqr > 0.15 or frac_symbols_good < T_frac):
        route = "BOTH"
        reason = f"skill01={skill01_cs:.3f} >= {T_cs} BUT concentrated (IQR={symbol_skill01_iqr:.3f}, frac_good={frac_symbols_good:.2f})"

    # Default: CROSS_SECTIONAL (fallback)
    if route is None:
        route = "CROSS_SECTIONAL"
        if len(symbol_skill01s) == 0:
            reason = f"default (skill01={skill01_cs:.3f}, symbol_eval=0 symbols evaluable)"
        else:
            reason = f"default (skill01={skill01_cs:.3f}, no strong symbol-specific signal)"

    # SS Universe Size Gate: Disable SS routing for large universes
    # SS creates one model per symbol, which negates memory savings for large universes
    ss_gated = False
    # FIX: Use total universe size, not just evaluated symbols
    # This ensures the gate applies consistently regardless of evaluation success rate
    universe_size = total_symbols if total_symbols is not None else (len(sym_results) if sym_results else 0)
    if route in ("SYMBOL_SPECIFIC", "BOTH") and universe_size > 0:
        # CRITICAL: Check experiment config first (has precedence over base config)
        max_symbols_for_ss = None
        ss_fallback = None

        if experiment_config:
            exp_name = experiment_config.name if hasattr(experiment_config, 'name') else str(experiment_config)
            try:
                exp_file = Path("CONFIG/experiments") / f"{exp_name}.yaml"
                if exp_file.exists():
                    import yaml
                    with open(exp_file, 'r') as f:
                        exp_yaml = yaml.safe_load(f) or {}
                    routing_cfg = exp_yaml.get('target_routing', {})
                    if 'max_symbols_for_ss' in routing_cfg:
                        max_symbols_for_ss = int(routing_cfg['max_symbols_for_ss'])
                        ss_fallback = str(routing_cfg.get('ss_fallback_route', 'CROSS_SECTIONAL'))
                        logger.debug(f"SS gate config from experiment: max_symbols_for_ss={max_symbols_for_ss}")
            except Exception as e:
                logger.debug(f"Could not load SS gate config from experiment: {e}")

        # Fallback to base config if not set by experiment
        if max_symbols_for_ss is None:
            try:
                from CONFIG.config_loader import get_cfg
                max_symbols_for_ss = int(get_cfg("target_routing.max_symbols_for_ss", default=100))
                ss_fallback = str(get_cfg("target_routing.ss_fallback_route", default="CROSS_SECTIONAL"))
            except Exception:
                max_symbols_for_ss = 100
                ss_fallback = "CROSS_SECTIONAL"

        if universe_size > max_symbols_for_ss:
            original_route = route
            route = ss_fallback
            ss_gated = True
            logger.warning(
                f"🚫 SS gated for {target}: universe size {universe_size} > {max_symbols_for_ss}. "
                f"Route changed: {original_route} → {route}"
            )
            reason = f"SS_GATED (was {original_route}): universe={universe_size} > max={max_symbols_for_ss}. " + reason

    # Get skip reasons for this target
    target_skip_reasons = symbol_skip_reasons if symbol_skip_reasons else {}

    # Extract auto_fix_reason and T-stat from result_cs using existing SST solutions
    auto_fix_reason_for_routing = None
    tstat_cs_for_routing = None
    if result_cs:
        auto_fix_reason_for_routing = getattr(result_cs, 'auto_fix_reason', None)
        tstat_cs_for_routing = getattr(result_cs, 'primary_metric_tstat', None)

    return {
        'route': route,
        'reason': reason,  # Keep existing reason string unchanged (backward compat)
        'skill01_cs': skill01_cs,  # New: normalized skill score for routing
        'skill01_sym_mean': symbol_skill01_mean,  # New: mean symbol skill01
        'auc': auc,  # Deprecated: kept for backward compatibility (R² for regression, AUC for classification)
        'symbol_auc_mean': symbol_skill01_mean,  # Deprecated: now contains skill01_mean, kept for backward compat
        'symbol_auc_median': symbol_skill01_median,  # Deprecated: now contains skill01_median
        'symbol_auc_min': symbol_skill01_min,  # Deprecated: now contains skill01_min
        'symbol_auc_max': symbol_skill01_max,  # Deprecated: now contains skill01_max
        'symbol_auc_iqr': symbol_skill01_iqr,  # Deprecated: now contains skill01_iqr
        'frac_symbols_good': frac_symbols_good,
        'winner_symbols': winner_symbols,
        'n_symbols_evaluated': len(symbol_skill01s) if symbol_skill01s else 0,
        'symbol_skip_reasons': target_skip_reasons if target_skip_reasons else None,
        'auto_fix_reason': auto_fix_reason_for_routing,  # NEW: Structured skip reason
        'tstat_cs': tstat_cs_for_routing  # NEW: T-stat for routing traceability
    }


def _save_single_target_decision(
    target: str,
    decision: dict[str, Any],
    output_dir: Path | None
) -> None:
    """
    Save routing decision for a single target immediately after evaluation.
    
    This allows incremental saving so decisions are available as soon as each target completes.
    
    Args:
        target: Target name
        decision: Routing decision dict for this target
        output_dir: Base output directory (RESULTS/{run}/) - can be None, will try to infer
    """
    import json

    from TRAINING.orchestration.utils.target_first_paths import ensure_target_structure, get_target_decision_dir

    if output_dir is None:
        logger.warning(f"[WARN]  Cannot save routing decision for {target}: output_dir is None")
        return

    # Determine base output directory
    if output_dir.name == "target_rankings":
        base_output_dir = output_dir.parent
    else:
        base_output_dir = output_dir

    # Walk up to find run directory using SST helper
    from TRAINING.orchestration.utils.target_first_paths import run_root as get_run_root
    base_output_dir = get_run_root(base_output_dir)

    try:
        ensure_target_structure(base_output_dir, target)
        target_decision_dir = get_target_decision_dir(base_output_dir, target)
        target_decision_file = target_decision_dir / "routing_decision.json"
        # DETERMINISM: Use atomic write for crash consistency
        write_atomic_json(target_decision_file, {target: decision}, default=str)
        logger.debug(f"[OK] Saved routing decision for {target} to {target_decision_file}")
    except Exception as e:
        # EH-004: Fail-closed in strict mode for artifact write failures
        from TRAINING.common.determinism import is_strict_mode
        if is_strict_mode():
            from TRAINING.common.exceptions import ArtifactError
            raise ArtifactError(
                f"Failed to save routing decision for {target}",
                artifact_path=str(target_decision_file) if 'target_decision_file' in dir() else str(base_output_dir),
                stage="ROUTING"
            ) from e
        logger.warning(f"[WARN] EH-004: Failed to save routing decision for {target}: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")


def _save_dual_view_rankings(
    results_cs: list[Any],
    results_sym: dict[str, dict[str, Any]],
    results_loso: dict[str, dict[str, Any]],
    routing_decisions: dict[str, dict[str, Any]],
    output_dir: Path
):
    """
    Save dual-view ranking results and routing decisions.
    
    Target-first structure:
    - Global routing decisions → globals/routing_decisions.json (global summary)
    - Per-target routing decision → targets/<target>/decision/routing_decision.json (optional, for fast local inspection)
    
    Reading logic maintains backward compatibility (reads from legacy locations if needed):
    - DECISION/TARGET_RANKING/routing_decisions.json (legacy location)
    - REPRODUCIBILITY/TARGET_RANKING/routing_decisions.json (legacy location)
    
    Args:
        output_dir: Base output directory (RESULTS/{run}/), not target_rankings subdirectory
    """
    import json

    # Path is already imported globally at line 13
    from TRAINING.orchestration.utils.target_first_paths import (
        ensure_target_structure,
        get_globals_dir,
        get_target_decision_dir,
    )

    # Determine base output directory (handle both old and new call patterns)
    if output_dir.name == "target_rankings":
        base_output_dir = output_dir.parent
    else:
        base_output_dir = output_dir

    # Ensure we have the actual run directory using SST helper
    from TRAINING.orchestration.utils.target_first_paths import run_root as get_run_root
    base_output_dir = get_run_root(base_output_dir)

    # Compute fingerprint for routing decisions to prevent stale data reuse
    # Fingerprint includes: target set, symbol set, and config hash
    import hashlib
    target_set = sorted(set(routing_decisions.keys()))
    # Extract symbols from routing decisions (if available)
    symbols_set = set()
    for decision in routing_decisions.values():
        if 'symbols' in decision and isinstance(decision['symbols'], dict):
            symbols_set.update(decision['symbols'].keys())
    symbols_list = sorted(symbols_set) if symbols_set else []

    # SST: Determine view from actual symbols count (same pattern as resolve_write_scope validation)
    # This is the reliable method that works - don't trust legacy global view
    from TRAINING.orchestration.utils.scope_resolution import View
    if symbols_list and len(symbols_list) > 1:
        run_view = View.CROSS_SECTIONAL.value  # Multi-symbol → CROSS_SECTIONAL
    elif symbols_list and len(symbols_list) == 1:
        run_view = View.SYMBOL_SPECIFIC.value  # Single-symbol → SYMBOL_SPECIFIC
    else:
        # Fallback: try universe-specific view cache (already validated in get_view_for_universe)
        run_view = View.CROSS_SECTIONAL.value  # Default
        try:
            from TRAINING.orchestration.utils.run_context import load_run_context
            context = load_run_context(output_dir)
            if context:
                # Try to get from first universe in cache (already validated)
                views = context.get("views", {})
                if views:
                    first_entry = next(iter(views.values()))
                    cached_view = first_entry.get('view')
                    # Use cached view (get_view_for_universe already validated it)
                    if cached_view:
                        run_view = cached_view
        except Exception:
            pass

    # Create fingerprint from targets, symbols, and view
    fingerprint_data = {
        'targets': target_set,
        'symbols': symbols_list,
        'target_count': len(target_set),
        'symbol_count': len(symbols_list),
        'view': run_view  # Include view in fingerprint (SST)
    }
    fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
    fingerprint_hash = hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]

    # Prepare routing data
    routing_data = {
        'routing_decisions': routing_decisions,
        'fingerprint': fingerprint_hash,
        'fingerprint_data': fingerprint_data,  # Store for debugging
        'summary': {
            'total_targets': len(routing_decisions),
            'cross_sectional_only': sum(1 for r in routing_decisions.values() if r.get('route') == 'CROSS_SECTIONAL'),
            'symbol_specific_only': sum(1 for r in routing_decisions.values() if r.get('route') == 'SYMBOL_SPECIFIC'),
            'both': sum(1 for r in routing_decisions.values() if r.get('route') == 'BOTH'),
            'blocked': sum(1 for r in routing_decisions.values() if r.get('route') == 'BLOCKED')
        }
    }

    # Save to globals/ (target-first primary location)
    globals_dir = get_globals_dir(base_output_dir)
    globals_dir.mkdir(parents=True, exist_ok=True)
    # SST: Sanitize routing data to normalize enums to strings before JSON serialization
    from TRAINING.orchestration.utils.diff_telemetry import _sanitize_for_json
    sanitized_routing_data = _sanitize_for_json(routing_data)

    globals_file = globals_dir / "routing_decisions.json"
    # DETERMINISM: Use atomic write for crash consistency
    write_atomic_json(globals_file, sanitized_routing_data, default=str)
    logger.info(f"Saved routing decisions to {globals_file}")

    # Save per-target slices for fast local inspection
    # CRITICAL: This ensures ALL targets in routing_decisions get decision files,
    # even if incremental save failed earlier. This is a safety net.
    # DETERMINISM: Use sorted_items for deterministic iteration order
    for target, decision in sorted_items(routing_decisions):
        try:
            ensure_target_structure(base_output_dir, target)
            target_decision_dir = get_target_decision_dir(base_output_dir, target)
            target_decision_file = target_decision_dir / "routing_decision.json"
            # DETERMINISM: Use atomic write for crash consistency
            write_atomic_json(target_decision_file, {target: decision}, default=str)
            logger.debug(f"Saved per-target routing decision to {target_decision_file}")
        except Exception as e:
            logger.warning(f"[WARN]  Failed to save per-target routing decision for {target}: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")

    # Target-first structure only - no legacy writes

    # Note: Individual view results are already saved by evaluate_target_predictability
    # via reproducibility tracker (with view/symbol metadata in RunContext)


def load_routing_decisions(
    routing_file: Path | None = None,
    output_dir: Path | None = None,
    expected_targets: list[str] | None = None,
    validate_fingerprint: bool = True
) -> dict[str, dict[str, Any]]:
    """
    Load routing decisions from file.
    
    Tries multiple locations in order:
    1. Target-first structure: globals/routing_decisions.json
    2. Legacy structure: DECISION/TARGET_RANKING/routing_decisions.json
    3. Legacy structure: REPRODUCIBILITY/TARGET_RANKING/routing_decisions.json
    4. Explicit routing_file path if provided
    
    Args:
        routing_file: Optional explicit path to routing_decisions.json
        output_dir: Optional base output directory (will search for routing decisions)
        expected_targets: Optional list of expected targets (for fingerprint validation)
        validate_fingerprint: If True, validate fingerprint matches expected targets (default: True)
    
    Returns:
        Dict mapping target -> routing decision dict
    """
    import hashlib
    import json

    # If explicit file provided, use it
    if routing_file and routing_file.exists():
        try:
            with open(routing_file) as f:
                data = json.load(f)
            routing_decisions = data.get('routing_decisions', {})

            # Validate fingerprint if expected_targets provided
            if validate_fingerprint and expected_targets:
                stored_fingerprint = data.get('fingerprint')
                if stored_fingerprint:
                    # SST: Determine view from symbols in stored fingerprint_data (same pattern as generation)
                    from TRAINING.orchestration.utils.scope_resolution import View
                    stored_fingerprint_data = data.get('fingerprint_data', {})
                    stored_symbols = stored_fingerprint_data.get('symbols', [])

                    # Determine view from stored symbols (same logic as generation)
                    if stored_symbols and len(stored_symbols) > 1:
                        run_view = View.CROSS_SECTIONAL.value  # Multi-symbol → CROSS_SECTIONAL
                    elif stored_symbols and len(stored_symbols) == 1:
                        run_view = View.SYMBOL_SPECIFIC.value  # Single-symbol → SYMBOL_SPECIFIC
                    else:
                        # Fallback: try universe-specific view cache (already validated in get_view_for_universe)
                        run_view = View.CROSS_SECTIONAL.value  # Default
                        try:
                            from TRAINING.orchestration.utils.run_context import load_run_context
                            context = load_run_context(output_dir)
                            if context:
                                # Try to get from first universe in cache (already validated)
                                views = context.get("views", {})
                                if views:
                                    first_entry = next(iter(views.values()))
                                    cached_view = first_entry.get('view')
                                    # Use cached view (get_view_for_universe already validated it)
                                    if cached_view:
                                        run_view = cached_view
                        except Exception:
                            pass

                    # Compute expected fingerprint (include view)
                    target_set = sorted(set(expected_targets))
                    fingerprint_data = {
                        'targets': target_set,
                        'target_count': len(target_set),
                        'view': run_view  # Include view in fingerprint (SST)
                    }
                    fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
                    expected_fingerprint = hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]

                    if stored_fingerprint != expected_fingerprint:
                        # Check dev_mode to decide whether to raise or return empty
                        dev_mode = False
                        try:
                            from CONFIG.config_loader import get_cfg
                            routing_config = get_cfg("training_config.routing", default={}, config_name="training_config")
                            dev_mode = routing_config.get("dev_mode", False)
                        except Exception:
                            pass

                        error_msg = (
                            f"[ERROR] Routing decisions fingerprint mismatch: stored={stored_fingerprint[:8]}... "
                            f"expected={expected_fingerprint[:8]}... "
                            f"This indicates stale routing decisions. "
                            f"Re-run feature selection to generate fresh routing decisions."
                        )

                        if dev_mode:
                            logger.warning(f"{error_msg} Dev mode: Ignoring stale decisions, attempting regeneration...")
                            # Try to regenerate from current candidates
                            try:
                                # Check if fresh candidates exist
                                from TRAINING.orchestration.utils.target_first_paths import get_globals_dir
                                globals_dir = get_globals_dir(output_dir) if output_dir else None
                                if globals_dir:
                                    candidates_file = globals_dir / "routing" / "routing_candidates.parquet"
                                    if candidates_file.exists():
                                        logger.info("Fresh routing candidates found, but regeneration not yet implemented. Returning empty decisions.")
                                        # TODO: Implement regeneration from candidates
                                        return {"_STALE_DECISIONS_IGNORED": True, "_regenerate": True}
                            except Exception as e:
                                logger.warning(f"Failed to check for fresh candidates: {e}")
                            # Return marker indicating stale decisions were ignored
                            return {"_STALE_DECISIONS_IGNORED": True}
                        else:
                            logger.error(error_msg)
                            raise ValueError(
                                "Stale routing decisions detected. Re-run feature selection to generate fresh decisions. "
                                f"Fingerprint mismatch: stored={stored_fingerprint[:8]}... expected={expected_fingerprint[:8]}..."
                            )
                else:
                    logger.debug("Routing decisions file has no fingerprint - skipping validation")

            return routing_decisions
        except Exception as e:
            logger.error(f"Failed to load routing decisions from {routing_file}: {e}")
            return {}

    # If output_dir provided, enforce single known path (globals/routing_decisions.json)
    if output_dir:
        output_dir = Path(output_dir)

        # PRIMARY: globals/routing_decisions.json (current run only)
        globals_file = output_dir / "globals" / "routing_decisions.json"
        if globals_file.exists():
            try:
                with open(globals_file) as f:
                    data = json.load(f)
                routing_decisions = data.get('routing_decisions', {})

                # Validate fingerprint if expected_targets provided
                if validate_fingerprint and expected_targets:
                    stored_fingerprint = data.get('fingerprint')
                    if stored_fingerprint:
                        # SST: Determine view from symbols in stored fingerprint_data (same pattern as generation)
                        from TRAINING.orchestration.utils.scope_resolution import View
                        stored_fingerprint_data = data.get('fingerprint_data', {})
                        stored_symbols = stored_fingerprint_data.get('symbols', [])

                        # Determine view from stored symbols (same logic as generation)
                        if stored_symbols and len(stored_symbols) > 1:
                            run_view = View.CROSS_SECTIONAL.value  # Multi-symbol → CROSS_SECTIONAL
                        elif stored_symbols and len(stored_symbols) == 1:
                            run_view = View.SYMBOL_SPECIFIC.value  # Single-symbol → SYMBOL_SPECIFIC
                        else:
                            # Fallback: try universe-specific view cache (already validated in get_view_for_universe)
                            run_view = View.CROSS_SECTIONAL.value  # Default
                            try:
                                from TRAINING.orchestration.utils.run_context import load_run_context
                                context = load_run_context(output_dir)
                                if context:
                                    # Try to get from first universe in cache (already validated)
                                    views = context.get("views", {})
                                    if views:
                                        first_entry = next(iter(views.values()))
                                        cached_view = first_entry.get('view')
                                        # Use cached view (get_view_for_universe already validated it)
                                        if cached_view:
                                            run_view = cached_view
                            except Exception:
                                pass

                        target_set = sorted(set(expected_targets))
                        # CRITICAL: Expected fingerprint must match stored fingerprint structure
                        # Stored fingerprint includes symbols and symbol_count (see _save_dual_view_rankings)
                        # Use stored symbols from fingerprint_data to ensure consistency
                        expected_symbols = stored_symbols if stored_symbols else []
                        fingerprint_data = {
                            'targets': target_set,
                            'symbols': expected_symbols,  # Must match stored fingerprint structure
                            'target_count': len(target_set),
                            'symbol_count': len(expected_symbols),  # Must match stored fingerprint structure
                            'view': run_view  # Include view in fingerprint (SST)
                        }
                        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
                        expected_fingerprint = hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]

                        if stored_fingerprint != expected_fingerprint:
                            # Check dev_mode to decide whether to raise or return empty
                            dev_mode = False
                            try:
                                from CONFIG.config_loader import get_cfg
                                routing_config = get_cfg("training_config.routing", default={}, config_name="training_config")
                                dev_mode = routing_config.get("dev_mode", False)
                            except Exception:
                                pass

                            error_msg = (
                                f"[ERROR] Routing decisions fingerprint mismatch: stored={stored_fingerprint[:8]}... "
                                f"expected={expected_fingerprint[:8]}... "
                                f"Loaded {len(routing_decisions)} decisions but fingerprint doesn't match expected targets. "
                                f"This indicates stale routing decisions. "
                                f"Re-run feature selection to generate fresh routing decisions."
                            )

                            if dev_mode:
                                logger.warning(f"{error_msg} Dev mode: Ignoring stale decisions, attempting regeneration...")
                                # Try to regenerate from current candidates
                                try:
                                    from TRAINING.orchestration.utils.target_first_paths import get_globals_dir
                                    globals_dir = get_globals_dir(output_dir) if output_dir else None
                                    if globals_dir:
                                        candidates_file = globals_dir / "routing" / "routing_candidates.parquet"
                                        if candidates_file.exists():
                                            logger.info("Fresh routing candidates found, but regeneration not yet implemented. Returning empty decisions.")
                                            # TODO: Implement regeneration from candidates
                                            return {"_STALE_DECISIONS_IGNORED": True, "_regenerate": True}
                                except Exception as e:
                                    logger.warning(f"Failed to check for fresh candidates: {e}")
                                # Return marker indicating stale decisions were ignored
                                return {"_STALE_DECISIONS_IGNORED": True}
                            else:
                                logger.error(error_msg)
                                raise ValueError(
                                    "Stale routing decisions detected. Re-run feature selection to generate fresh decisions. "
                                    f"Fingerprint mismatch: stored={stored_fingerprint[:8]}... expected={expected_fingerprint[:8]}..."
                                )
                    else:
                        logger.debug("Routing decisions file has no fingerprint - skipping validation")

                logger.debug(f"Loaded routing decisions from target-first structure: {globals_file}")
                return routing_decisions
            except Exception as e:
                logger.debug(f"Failed to load from globals: {e}")

        # If not found, fail loudly (no legacy fallback to prevent stale decisions)
        if validate_fingerprint and expected_targets:
            raise FileNotFoundError(
                f"Routing decisions not found for current run at {globals_file}. "
                f"Expected targets: {expected_targets}. "
                f"Re-run feature selection to generate fresh decisions."
            )
        else:
            logger.warning(f"Routing decisions not found at {globals_file}. Returning empty dict.")
            return {}

    # If routing_file was provided but doesn't exist, warn
    if routing_file:
        logger.warning(f"Routing decisions file not found: {routing_file}")

    return {}
