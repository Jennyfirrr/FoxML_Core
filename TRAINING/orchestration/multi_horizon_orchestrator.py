# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Multi-Horizon Bundle Training Orchestrator

Orchestrates the training of multi-horizon models for horizon bundles.
Integrates with the intelligent trainer when strategy="multi_horizon_bundle".

Usage:
    from TRAINING.orchestration.multi_horizon_orchestrator import (
        run_multi_horizon_training,
        load_multi_horizon_config,
    )

    # Run multi-horizon training
    results = run_multi_horizon_training(
        mtf_data=mtf_data,
        targets=targets,
        y_dict=y_dict,
        target_scores=target_scores,
        output_dir=output_dir,
        config=config,
    )
"""

from __future__ import annotations

import TRAINING.common.repro_bootstrap  # noqa: F401 - must be first (after __future__)

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from CONFIG.config_loader import get_cfg

logger = logging.getLogger(__name__)


def load_multi_horizon_config(
    experiment_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load multi-horizon bundle configuration.

    Precedence: experiment_config > intelligent.yaml defaults

    Args:
        experiment_config: Optional experiment-level overrides

    Returns:
        Merged configuration dict
    """
    # Load defaults from intelligent.yaml
    defaults = {
        "enabled": get_cfg(
            "strategy_configs.multi_horizon_bundle.enabled", default=False
        ),
        "auto_discover": get_cfg(
            "strategy_configs.multi_horizon_bundle.auto_discover", default=True
        ),
        "min_horizons": get_cfg(
            "strategy_configs.multi_horizon_bundle.min_horizons", default=2
        ),
        "max_horizons": get_cfg(
            "strategy_configs.multi_horizon_bundle.max_horizons", default=5
        ),
        "min_diversity": get_cfg(
            "strategy_configs.multi_horizon_bundle.min_diversity", default=0.3
        ),
        "top_n_bundles": get_cfg(
            "strategy_configs.multi_horizon_bundle.top_n_bundles", default=3
        ),
        "diversity_weight": get_cfg(
            "strategy_configs.multi_horizon_bundle.diversity_weight", default=0.3
        ),
        "predictability_weight": get_cfg(
            "strategy_configs.multi_horizon_bundle.predictability_weight", default=0.7
        ),
        "shared_layers": get_cfg(
            "strategy_configs.multi_horizon_bundle.shared_layers", default=[256, 128]
        ),
        "head_layers": get_cfg(
            "strategy_configs.multi_horizon_bundle.head_layers", default=[64]
        ),
        "dropout": get_cfg(
            "strategy_configs.multi_horizon_bundle.dropout", default=0.2
        ),
        "batch_norm": get_cfg(
            "strategy_configs.multi_horizon_bundle.batch_norm", default=True
        ),
        "backend": get_cfg(
            "strategy_configs.multi_horizon_bundle.backend", default="tensorflow"
        ),
        "epochs": get_cfg(
            "strategy_configs.multi_horizon_bundle.epochs", default=100
        ),
        "batch_size": get_cfg(
            "strategy_configs.multi_horizon_bundle.batch_size", default=256
        ),
        "patience": get_cfg(
            "strategy_configs.multi_horizon_bundle.patience", default=10
        ),
        "lr": get_cfg("strategy_configs.multi_horizon_bundle.lr", default=0.001),
        "loss_weighting": get_cfg(
            "strategy_configs.multi_horizon_bundle.loss_weighting", default="equal"
        ),
        "horizon_decay_half_life_minutes": get_cfg(
            "strategy_configs.multi_horizon_bundle.horizon_decay_half_life_minutes",
            default=30,
        ),
    }

    # Apply experiment overrides
    if experiment_config:
        mh_overrides = experiment_config.get("strategy_configs", {}).get(
            "multi_horizon_bundle", {}
        )
        defaults.update(mh_overrides)

    return defaults


def prepare_target_data(
    mtf_data: Dict[str, Any],
    targets: List[str],
) -> Dict[str, np.ndarray]:
    """
    Extract target values from MTF data.

    Args:
        mtf_data: Multi-timeframe data dict
        targets: List of target names

    Returns:
        Dict of target_name → target_values array
    """
    import pandas as pd

    y_dict = {}

    # Get combined data
    from TRAINING.common.utils.determinism_ordering import sorted_items
    df = mtf_data.get("combined_df")
    if df is None:
        # Try to combine from symbol data (sorted for determinism)
        symbol_dfs = []
        for symbol, sdata in sorted_items(mtf_data.get("symbols", {})):
            if isinstance(sdata, pd.DataFrame):
                symbol_dfs.append(sdata)
            elif isinstance(sdata, dict) and "df" in sdata:
                symbol_dfs.append(sdata["df"])

        if symbol_dfs:
            df = pd.concat(symbol_dfs, ignore_index=True)

    if df is None:
        logger.warning("Could not extract combined dataframe from mtf_data")
        return {}

    # Extract target columns
    for target in targets:
        if target in df.columns:
            values = df[target].values
            # Drop NaN values for diversity calculation
            valid_mask = ~np.isnan(values)
            if np.sum(valid_mask) > 10:
                y_dict[target] = values[valid_mask]
            else:
                logger.warning(f"Target {target} has too few valid samples, skipping")
        else:
            logger.debug(f"Target {target} not found in data")

    return y_dict


def run_multi_horizon_training(
    mtf_data: Dict[str, Any],
    targets: List[str],
    output_dir: str,
    target_scores: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
    experiment_config: Optional[Dict[str, Any]] = None,
    X_train: Optional[np.ndarray] = None,
    X_val: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run multi-horizon bundle training.

    Steps:
    1. Discover/create horizon bundles from targets
    2. Rank bundles by diversity + predictability
    3. Train multi-horizon models for top bundles
    4. Save models and metadata

    Args:
        mtf_data: Multi-timeframe data dict
        targets: All available targets
        output_dir: Output directory for artifacts
        target_scores: Optional predictability scores from ranking
        config: Multi-horizon config (loaded if None)
        experiment_config: Optional experiment-level config
        X_train: Training features (if pre-prepared)
        X_val: Validation features (if pre-prepared)
        feature_names: Feature column names

    Returns:
        Results dict with trained bundles and metadata
    """
    # Load config
    if config is None:
        config = load_multi_horizon_config(experiment_config)

    if not config.get("enabled", False):
        logger.info("Multi-horizon bundle training is disabled")
        return {"status": "disabled", "bundles_trained": 0}

    logger.info("=" * 80)
    logger.info("MULTI-HORIZON BUNDLE TRAINING")
    logger.info("=" * 80)

    # Import here to avoid circular imports
    from TRAINING.common.horizon_bundle import (
        HorizonBundle,
        compute_horizon_based_weights,
    )
    from TRAINING.orchestration.horizon_ranker import (
        select_top_bundles,
        validate_bundles_for_training,
    )

    # Prepare target data
    y_dict = prepare_target_data(mtf_data, targets)
    if not y_dict:
        logger.error("Could not extract target data for multi-horizon training")
        return {"status": "error", "reason": "no_target_data", "bundles_trained": 0}

    logger.info(f"Prepared data for {len(y_dict)} targets")

    # Select top bundles
    bundles = select_top_bundles(
        targets=targets,
        y_dict=y_dict,
        target_scores=target_scores,
        top_n=config.get("top_n_bundles", 3),
        min_diversity=config.get("min_diversity", 0.3),
        min_horizons=config.get("min_horizons", 2),
        max_horizons=config.get("max_horizons", 5),
        diversity_weight=config.get("diversity_weight", 0.3),
        predictability_weight=config.get("predictability_weight", 0.7),
    )

    if not bundles:
        logger.warning("No valid horizon bundles found")
        return {
            "status": "no_bundles",
            "reason": "no_bundles_meet_criteria",
            "bundles_trained": 0,
        }

    # Validate bundles have data
    valid_bundles = validate_bundles_for_training(bundles, y_dict)
    if not valid_bundles:
        logger.warning("No bundles have sufficient data for training")
        return {
            "status": "no_valid_bundles",
            "reason": "insufficient_data",
            "bundles_trained": 0,
        }

    logger.info(f"Training {len(valid_bundles)} horizon bundles")

    # Prepare training data
    import pandas as pd

    df = mtf_data.get("combined_df")
    if df is None:
        # Combine from symbols (sorted for determinism)
        from TRAINING.common.utils.determinism_ordering import sorted_items
        symbol_dfs = []
        for symbol, sdata in sorted_items(mtf_data.get("symbols", {})):
            if isinstance(sdata, pd.DataFrame):
                symbol_dfs.append(sdata)
            elif isinstance(sdata, dict) and "df" in sdata:
                symbol_dfs.append(sdata["df"])
        if symbol_dfs:
            df = pd.concat(symbol_dfs, ignore_index=True)

    if df is None:
        logger.error("Could not prepare training data")
        return {"status": "error", "reason": "no_training_data", "bundles_trained": 0}

    # Get feature columns (exclude targets and metadata)
    target_cols = set(targets)
    meta_cols = {"symbol", "timestamp", "date", "datetime", "open", "high", "low", "close", "volume"}
    feature_cols = [
        c for c in df.columns
        if c not in target_cols and c.lower() not in meta_cols and not c.startswith("fwd_") and not c.startswith("will_")
    ]

    if not feature_cols:
        logger.error("No feature columns found")
        return {"status": "error", "reason": "no_features", "bundles_trained": 0}

    logger.info(f"Using {len(feature_cols)} features for training")

    # Prepare X matrix
    X = df[feature_cols].values.astype(np.float32)

    # Train each bundle
    results = {
        "status": "success",
        "bundles_trained": 0,
        "bundles": [],
        "models": {},
    }

    output_path = Path(output_dir)
    mh_output = output_path / "multi_horizon"
    mh_output.mkdir(parents=True, exist_ok=True)

    # Import trainer (direct import instead of fragile importlib.util)
    try:
        from TRAINING.model_fun.multi_horizon_trainer import MultiHorizonTrainer
    except ImportError as e:
        logger.error(f"Failed to import MultiHorizonTrainer: {e}")
        return {"status": "error", "reason": f"import_error: {e}", "bundles_trained": 0}

    for bundle in valid_bundles:
        logger.info(f"Training bundle: {bundle.base_name} ({bundle.n_horizons} horizons)")

        # Prepare y dict for this bundle
        bundle_y = {}
        valid_idx = None

        for target in bundle.targets:
            if target in df.columns:
                target_vals = df[target].values
                # Build valid index (non-NaN across all targets)
                mask = ~np.isnan(target_vals)
                if valid_idx is None:
                    valid_idx = mask
                else:
                    valid_idx = valid_idx & mask
                bundle_y[target] = target_vals

        if valid_idx is None or np.sum(valid_idx) < 100:
            logger.warning(f"Bundle {bundle.base_name} has insufficient valid samples")
            continue

        # Filter to valid samples
        X_bundle = X[valid_idx]
        for target in bundle_y:
            bundle_y[target] = bundle_y[target][valid_idx]

        # Split train/val
        n_samples = X_bundle.shape[0]
        split_idx = int(n_samples * 0.8)
        X_tr = X_bundle[:split_idx]
        X_va = X_bundle[split_idx:]
        y_tr = {t: v[:split_idx] for t, v in bundle_y.items()}
        y_va = {t: v[split_idx:] for t, v in bundle_y.items()}

        # Compute loss weights
        if config.get("loss_weighting", "equal") == "horizon_decay":
            loss_weights = compute_horizon_based_weights(
                bundle,
                decay_half_life_minutes=config.get("horizon_decay_half_life_minutes", 30),
            )
        else:
            loss_weights = {t: 1.0 for t in bundle.targets}

        # Build trainer config
        trainer_config = {
            "shared_layers": config.get("shared_layers", [256, 128]),
            "head_layers": config.get("head_layers", [64]),
            "dropout": config.get("dropout", 0.2),
            "batch_norm": config.get("batch_norm", True),
            "epochs": config.get("epochs", 100),
            "batch_size": config.get("batch_size", 256),
            "patience": config.get("patience", 10),
            "lr": config.get("lr", 0.001),
            "backend": config.get("backend", "tensorflow"),
        }

        # Train
        try:
            trainer = MultiHorizonTrainer(trainer_config)
            train_result = trainer.train(
                X_tr,
                y_tr,
                X_va=X_va,
                y_va_dict=y_va,
                loss_weights=loss_weights,
                verbose=1,
            )

            # Save model
            bundle_save_path = mh_output / bundle.base_name
            trainer.save(str(bundle_save_path))

            # Record results
            bundle_result = {
                "bundle": bundle.to_dict(),
                "training": train_result,
                "model_path": str(bundle_save_path),
                "n_train_samples": X_tr.shape[0],
                "n_val_samples": X_va.shape[0],
                "n_features": X_tr.shape[1],
                "loss_weights": loss_weights,
            }

            results["bundles"].append(bundle_result)
            results["models"][bundle.base_name] = bundle_result
            results["bundles_trained"] += 1

            logger.info(
                f"  ✓ Trained {bundle.base_name}: "
                f"loss={train_result['final_loss']:.6f}, "
                f"epochs={train_result['epochs_trained']}"
            )

        except Exception as e:
            logger.error(f"Failed to train bundle {bundle.base_name}: {e}")
            import traceback

            traceback.print_exc()

    # Save summary (atomic write for crash consistency)
    from TRAINING.common.utils.file_utils import write_atomic_json

    summary_path = mh_output / "training_summary.json"
    # Convert numpy types for JSON serialization
    json_results = {
        "status": results["status"],
        "bundles_trained": results["bundles_trained"],
        "bundles": [
            {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in b.items()
            }
            for b in results["bundles"]
        ],
    }
    write_atomic_json(summary_path, json_results, default=str)

    logger.info("=" * 80)
    logger.info(f"Multi-horizon training complete: {results['bundles_trained']} bundles trained")
    logger.info(f"Results saved to: {mh_output}")
    logger.info("=" * 80)

    return results
