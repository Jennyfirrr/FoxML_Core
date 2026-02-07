# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Automatic Leakage Detection and Fixing

Wrapper around evaluate_target_predictability with auto-rerun logic.
Automatically re-evaluates targets after leakage fixes are applied.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from TRAINING.orchestration.utils.scope_resolution import View
from TRAINING.common.utils.task_types import TaskType, TargetConfig
from TRAINING.ranking.predictability.scoring import TargetPredictabilityScore

logger = logging.getLogger(__name__)

# Try to import config loader
_CONFIG_AVAILABLE = False
try:
    from config_loader import get_cfg, get_safety_config
    _CONFIG_AVAILABLE = True
except ImportError:
    pass


def evaluate_target_with_autofix(
    target: str,
    target_config: Dict[str, Any] | TargetConfig,
    symbols: List[str],
    data_dir: Path,
    model_families: List[str],
    multi_model_config: Dict[str, Any] = None,
    output_dir: Path = None,
    min_cs: int = 10,
    max_cs_samples: Optional[int] = None,
    max_rows_per_symbol: int = None,
    max_reruns: int = 3,
    rerun_on_perfect_train_acc: bool = True,
    rerun_on_high_auc_only: bool = False,
    explicit_interval: Optional[Union[int, str]] = None,
    experiment_config: Optional[Any] = None,
    view: Union[str, View] = View.CROSS_SECTIONAL,
    symbol: Optional[str] = None,
    run_identity: Optional[Any] = None,
    **kwargs
) -> TargetPredictabilityScore:
    """
    Wrapper around evaluate_target_predictability() with auto-rerun logic.

    Automatically re-evaluates targets after leakage fixes are applied.
    Uses patches from current run on reruns (allow_current_run_overlay=True).

    Config Precedence (SST):
    1. Experiment config (if exists) - target_ranking_overrides.auto_rerun
    2. Safety config - safety.leakage_detection.auto_rerun
    3. Function parameters (defaults)

    Args:
        target: Target column name
        target_config: TargetConfig or dict
        symbols: List of symbols
        data_dir: Data directory
        model_families: List of model family names
        multi_model_config: Multi-model config
        output_dir: Output directory (for patches)
        min_cs: Minimum cross-sectional samples
        max_cs_samples: Maximum CS samples
        max_rows_per_symbol: Maximum rows per symbol
        max_reruns: Maximum reruns (overridden by config if present)
        rerun_on_perfect_train_acc: Rerun on perfect train acc (overridden by config)
        rerun_on_high_auc_only: Rerun on high AUC only (overridden by config)
        explicit_interval: Explicit bar interval
        experiment_config: ExperimentConfig (SST - overrides safety config)
        view: View enum
        symbol: Symbol name (for SYMBOL_SPECIFIC)
        run_identity: RunIdentity object
        **kwargs: Additional kwargs passed to evaluate_target_predictability()

    Returns:
        TargetPredictabilityScore from final attempt
    """
    # Import evaluate_target_predictability at function level to avoid circular imports
    # Import from parent module directly (ranking.py is a stub that also imports from parent)
    import importlib.util
    from pathlib import Path
    _parent_file = Path(__file__).parent.parent / "model_evaluation.py"
    if _parent_file.exists():
        spec = importlib.util.spec_from_file_location("model_evaluation_main", _parent_file)
        _model_evaluation_main = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_model_evaluation_main)
        evaluate_target_predictability = _model_evaluation_main.evaluate_target_predictability
    else:
        raise ImportError(f"Could not find model_evaluation.py at {_parent_file}")

    # Load auto-rerun settings (SST: experiment config → safety config → defaults)
    auto_rerun_enabled = False
    max_reruns_config = max_reruns
    rerun_on_perfect_train_acc_config = rerun_on_perfect_train_acc
    rerun_on_high_auc_only_config = rerun_on_high_auc_only

    # 1. Try experiment config first (SST if exists)
    if experiment_config:
        try:
            # Check target_ranking_overrides.auto_rerun
            target_ranking_overrides = getattr(experiment_config, 'target_ranking_overrides', {})
            if not target_ranking_overrides:
                # Try dict access if it's a dict-like object
                target_ranking_overrides = experiment_config.get('target_ranking_overrides', {}) if isinstance(experiment_config, dict) else {}

            auto_rerun_cfg = target_ranking_overrides.get('auto_rerun', {})
            if auto_rerun_cfg:
                auto_rerun_enabled = auto_rerun_cfg.get('enabled', False)
                max_reruns_config = auto_rerun_cfg.get('max_reruns', max_reruns)
                rerun_on_perfect_train_acc_config = auto_rerun_cfg.get('rerun_on_perfect_train_acc', rerun_on_perfect_train_acc)
                rerun_on_high_auc_only_config = auto_rerun_cfg.get('rerun_on_high_auc_only', rerun_on_high_auc_only)
                logger.debug(f"Loaded auto-rerun settings from experiment config: enabled={auto_rerun_enabled}, max_reruns={max_reruns_config}")
        except Exception as e:
            logger.debug(f"Could not load auto-rerun from experiment config: {e}")

    # 2. Fallback to safety config if experiment config didn't provide settings
    if not auto_rerun_enabled and _CONFIG_AVAILABLE:
        try:
            from config_loader import get_safety_config
            safety_cfg = get_safety_config()
            safety_section = safety_cfg.get('safety', {})
            leakage_cfg = safety_section.get('leakage_detection', {})
            auto_rerun_cfg = leakage_cfg.get('auto_rerun', {})
            if auto_rerun_cfg:
                auto_rerun_enabled = auto_rerun_cfg.get('enabled', False)
                max_reruns_config = auto_rerun_cfg.get('max_reruns', max_reruns)
                rerun_on_perfect_train_acc_config = auto_rerun_cfg.get('rerun_on_perfect_train_acc', rerun_on_perfect_train_acc)
                rerun_on_high_auc_only_config = auto_rerun_cfg.get('rerun_on_high_auc_only', rerun_on_high_auc_only)
                logger.debug(f"Loaded auto-rerun settings from safety config: enabled={auto_rerun_enabled}, max_reruns={max_reruns_config}")
        except Exception as e:
            logger.debug(f"Could not load auto-rerun from safety config: {e}")

    # If auto-rerun disabled, just call evaluate_target_predictability once
    if not auto_rerun_enabled:
        return evaluate_target_predictability(
            target=target,
            target_config=target_config,
            symbols=symbols,
            data_dir=data_dir,
            model_families=model_families,
            multi_model_config=multi_model_config,
            output_dir=output_dir,
            min_cs=min_cs,
            max_cs_samples=max_cs_samples,
            max_rows_per_symbol=max_rows_per_symbol,
            explicit_interval=explicit_interval,
            experiment_config=experiment_config,
            view=view,
            symbol=symbol,
            run_identity=run_identity,
            allow_current_run_overlay=False,  # First attempt: no patches loaded
            **kwargs
        )

    # Auto-rerun enabled: loop until no leakage or max_reruns reached
    attempt = 0
    last_result = None
    attempt_history = []  # Track state for each attempt

    while attempt < max_reruns_config:
        # Track attempt state
        attempt_state = {
            'attempt': attempt + 1,
            'config_hash_before': None,
            'config_hash_after': None,
            'excluded_features_count': None,
            'detected_leaks': [],
            'exclusions_added': [],
            'metrics': {},
            'status': None,
            'decision': None
        }

        # Get config hash before attempt
        fixer = None
        try:
            from TRAINING.common.utils.config_hashing import compute_config_hash_from_file
            from TRAINING.common.leakage_auto_fixer import LeakageAutoFixer
            fixer = LeakageAutoFixer(backup_configs=False)
            if fixer.excluded_features_path.exists():
                attempt_state['config_hash_before'] = compute_config_hash_from_file(
                    fixer.excluded_features_path, short=False
                )
                logger.info(
                    f"📊 Attempt {attempt + 1} starting: "
                    f"config_hash={attempt_state['config_hash_before'][:16]}..."
                )
        except Exception as e:
            logger.debug(f"Could not compute config hash before attempt: {e}")

        # First attempt: don't load patches (they don't exist yet)
        # Subsequent attempts: load patches from current run
        allow_overlay = (attempt > 0)

        logger.info(f"🔁 Auto-rerun attempt {attempt + 1}/{max_reruns_config} for {target}")
        if allow_overlay:
            logger.debug(f"   Loading patches from current run: {output_dir / 'registry_patches' if output_dir else 'N/A'}")

        result = evaluate_target_predictability(
            target=target,
            target_config=target_config,
            symbols=symbols,
            data_dir=data_dir,
            model_families=model_families,
            multi_model_config=multi_model_config,
            output_dir=output_dir,
            min_cs=min_cs,
            max_cs_samples=max_cs_samples,
            max_rows_per_symbol=max_rows_per_symbol,
            explicit_interval=explicit_interval,
            experiment_config=experiment_config,
            view=view,
            symbol=symbol,
            run_identity=run_identity,
            allow_current_run_overlay=allow_overlay,  # KEY: True on reruns
            attempt_id=attempt,  # Pass attempt_id for path grouping
            **kwargs
        )

        last_result = result

        # After attempt completes, capture state
        attempt_state['status'] = getattr(result, 'status', None)
        attempt_state['metrics'] = {
            'auc': getattr(result, 'auc', None),
            'composite': getattr(result, 'composite_score', None),
            'leakage_flag': getattr(result, 'leakage_flag', None)
        }

        # Get config hash after attempt (if auto-fixer ran)
        try:
            if fixer and fixer.excluded_features_path.exists():
                attempt_state['config_hash_after'] = compute_config_hash_from_file(
                    fixer.excluded_features_path, short=False
                )
                if attempt_state['config_hash_before'] != attempt_state['config_hash_after']:
                    logger.info(
                        f"📊 Attempt {attempt + 1} config changed: "
                        f"{attempt_state['config_hash_before'][:16] if attempt_state['config_hash_before'] else 'new'} -> "
                        f"{attempt_state['config_hash_after'][:16]}"
                    )
        except Exception as e:
            logger.debug(f"Could not compute config hash after attempt: {e}")

        # Compare with previous attempt
        if attempt > 0 and attempt_history:
            prev_state = attempt_history[-1]
            logger.info("=" * 80)
            logger.info(f"📊 COMPARISON: Attempt {attempt} vs Attempt {attempt + 1}")
            logger.info(f"   Status: {prev_state['status']} -> {attempt_state['status']}")
            prev_auc = prev_state['metrics'].get('auc', None)
            curr_auc = attempt_state['metrics'].get('auc', None)
            auc_str = f"{prev_auc:.4f}" if prev_auc is not None else "N/A"
            auc_str2 = f"{curr_auc:.4f}" if curr_auc is not None else "N/A"
            logger.info(f"   AUC: {auc_str} -> {auc_str2}")
            prev_comp = prev_state['metrics'].get('composite', None)
            curr_comp = attempt_state['metrics'].get('composite', None)
            comp_str = f"{prev_comp:.4f}" if prev_comp is not None else "N/A"
            comp_str2 = f"{curr_comp:.4f}" if curr_comp is not None else "N/A"
            logger.info(f"   Composite: {comp_str} -> {comp_str2}")
            if prev_state['config_hash_after'] and attempt_state['config_hash_before']:
                if prev_state['config_hash_after'] == attempt_state['config_hash_before']:
                    logger.info(f"   Config hash: MATCH (rerun using same config)")
                else:
                    logger.warning(f"   Config hash: MISMATCH (config changed between attempts!)")
                    logger.warning(f"      Attempt {attempt}: {prev_state['config_hash_after'][:16]}")
                    logger.warning(f"      Attempt {attempt + 1}: {attempt_state['config_hash_before'][:16]}")
            logger.info("=" * 80)

        attempt_history.append(attempt_state)

        # Check if leakage was resolved
        status = getattr(result, 'status', None)
        if status and status not in ("LEAKAGE_DETECTED", "LEAKAGE_UNRESOLVED", "LEAKAGE_UNRESOLVED_MAX_RETRIES"):
            logger.info(f"✅ DECISION: Leakage resolved after {attempt + 1} attempt(s)")
            logger.info(f"   Final status: {status}")
            auc_final = attempt_state['metrics'].get('auc', None)
            comp_final = attempt_state['metrics'].get('composite', None)
            # Use canonical metric name from SST (don't branch on task type)
            # Get metric name from result if available, otherwise infer from task type
            if hasattr(result, 'primary_metric_name'):
                metric_display_name = result.primary_metric_name
            elif hasattr(result, 'task_type'):
                if result.task_type == TaskType.REGRESSION:
                    metric_display_name = "R²"
                elif result.task_type == TaskType.BINARY_CLASSIFICATION:
                    metric_display_name = "AUC"
                else:  # MULTICLASS_CLASSIFICATION
                    metric_display_name = "Accuracy"
            else:
                metric_display_name = "Score"  # Fallback
            auc_final_str = f"{auc_final:.4f}" if auc_final is not None else "N/A"
            comp_final_str = f"{comp_final:.4f}" if comp_final is not None else "N/A"
            logger.info(f"   Final metrics: {metric_display_name}={auc_final_str}, Composite={comp_final_str}")
            return result

        # Enhanced decision logging
        logger.warning(f"⚠️  DECISION: Rerun needed (status={status})")
        if attempt + 1 >= max_reruns_config:
            logger.warning(f"   DECISION: Stopping (max reruns reached: {max_reruns_config})")
        else:
            logger.info(f"   DECISION: Continuing to attempt {attempt + 2}")

        # Check if patches were written (configs modified)
        if output_dir:
            patch_dir = output_dir / "registry_patches"
            if patch_dir.exists():
                from TRAINING.common.registry_patch_naming import find_patch_file
                target_column = getattr(target_config, 'target_column', target) if hasattr(target_config, 'target_column') else target
                if find_patch_file(patch_dir, target_column):
                    # Patches exist, reload configs and rerun
                    try:
                        from TRAINING.ranking.utils.leakage_filtering import reload_feature_configs
                        reload_feature_configs()
                        logger.info(f"   Reloaded feature configs (patches available for next attempt)")
                    except Exception as e:
                        logger.warning(f"   Failed to reload feature configs: {e}")
                else:
                    # No patches written, can't fix
                    logger.warning(f"   DECISION: No patches written for {target}, stopping reruns")
                    return result
            else:
                # No patch directory, can't fix
                logger.warning(f"   DECISION: No patch directory exists, stopping reruns")
                return result
        else:
            # No output_dir, can't write/load patches
            logger.warning(f"   DECISION: No output_dir provided, stopping reruns")
            return result

        attempt += 1

    # Max reruns reached
    logger.warning(f"⚠️  DECISION: Reached max reruns ({max_reruns_config}) for {target}")
    return last_result if last_result else result
