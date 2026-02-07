# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Index Manager Mixin for ReproducibilityTracker.

Contains methods for managing the global index.parquet file.
Extracted from reproducibility_tracker.py for maintainability.
"""

import fcntl
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# SST: Import Stage enum for consistent handling
from TRAINING.orchestration.utils.scope_resolution import Stage

# Import helper extractors from reproducibility module
from TRAINING.orchestration.utils.reproducibility.utils import (
    extract_auc,
    extract_pos_rate,
    extract_feature_counts,
    extract_purge_minutes,
    extract_embargo_minutes,
    extract_folds,
)

# Import RouteType for view handling (if available)
try:
    from TRAINING.orchestration.utils.scope_resolution import RouteType
except ImportError:
    RouteType = None

logger = logging.getLogger(__name__)


def _extract_horizon_minutes_sst(metadata: Dict[str, Any], cv_details: Dict[str, Any]) -> Optional[int]:
    """
    Extract horizon_minutes using SST pattern (multiple source fallback).

    Args:
        metadata: Run metadata dict
        cv_details: CV details dict

    Returns:
        horizon_minutes if found, else None
    """
    # Try cv_details first (schema v2)
    if cv_details:
        hm = cv_details.get("horizon_minutes")
        if hm is not None:
            return int(hm)

    # Try metadata (schema v1)
    hm = metadata.get("horizon_minutes")
    if hm is not None:
        return int(hm)

    # Try to derive from target name
    target = metadata.get("target")
    if target:
        # Pattern: ..._Nm_... where N is the horizon
        match = re.search(r'_(\d+)m_', target)
        if match:
            return int(match.group(1))

    return None


class IndexManagerMixin:
    """
    Mixin class providing index management methods for ReproducibilityTracker.

    This mixin contains methods related to:
    - Updating the global index.parquet file
    - Parsing run timestamps
    - Computing relative paths

    Methods in this mixin expect the following attributes on self:
    - _repro_base_dir: Path to the reproducibility base directory
    - _calculate_cohort_relative_path(cohort_dir: Path) -> str: Method to compute relative path
    """

    def _parse_run_started_at(self, run_id: str, created_at: Optional[str] = None) -> str:
        """
        Parse run_started_at from run_id or use created_at.

        run_id formats:
        - YYYYMMDD_HHMMSS_* (preferred)
        - YYYY-MM-DDTHH:MM:SS* (ISO format)
        - Other formats fall back to created_at

        Args:
            run_id: Run identifier string
            created_at: Optional ISO timestamp string

        Returns:
            ISO format timestamp string
        """
        # Try to parse from run_id first
        # Format: YYYYMMDD_HHMMSS_*
        match = re.match(r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})', run_id)
        if match:
            year, month, day, hour, minute, second = match.groups()
            try:
                dt = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
                return dt.isoformat() + 'Z'
            except ValueError:
                pass

        # Try ISO format: YYYY-MM-DDTHH:MM:SS*
        match = re.match(r'(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})', run_id)
        if match:
            year, month, day, hour, minute, second = match.groups()
            try:
                dt = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
                return dt.isoformat() + 'Z'
            except ValueError:
                pass

        # Fall back to created_at
        if created_at:
            try:
                # Ensure it's in ISO format with timezone
                dt = pd.to_datetime(created_at, utc=True)
                return dt.isoformat()
            except Exception:
                pass

        # Last resort: use current time
        return datetime.now().isoformat() + 'Z'

    def _update_index(
        self,
        stage: str,
        target: str,
        view: Optional[str],
        symbol: Optional[str],
        model_family: Optional[str],
        cohort_id: str,
        run_id: str,
        metadata: Dict[str, Any],
        metrics: Dict[str, Any],
        cohort_dir: Path
    ) -> None:
        """
        Update the global index.parquet file.

        Args:
            stage: Pipeline stage (e.g., "TARGET_RANKING", "FEATURE_SELECTION")
            target: Target name
            view: Optional view type ("CROSS_SECTIONAL" or "SYMBOL_SPECIFIC")
            symbol: Optional symbol name
            model_family: Optional model family
            cohort_id: Cohort identifier
            run_id: Run identifier
            metadata: Run metadata dict
            metrics: Run metrics dict
            cohort_dir: Path to cohort directory
        """
        # Write index to globals/ instead of REPRODUCIBILITY/
        from TRAINING.orchestration.utils.target_first_paths import get_globals_dir
        globals_dir = get_globals_dir(self._repro_base_dir)
        globals_dir.mkdir(parents=True, exist_ok=True)
        index_file = globals_dir / "index.parquet"

        # Normalize stage
        if isinstance(stage, Stage):
            phase = stage.value
        else:
            # Normalize stage to enum, then to string
            stage_enum = Stage.from_string(stage) if isinstance(stage, str) else stage
            phase = str(stage_enum)  # Stage enum's __str__ returns .value

        # Normalize view
        if RouteType is not None and view and isinstance(view, RouteType):
            mode = view.value
        else:
            mode = view.upper() if view else None

        # Compute segment_id for decision-making (segments reset on identity breaks)
        segment_id = self._compute_segment_id(index_file, cohort_id, metadata, metrics)

        # Extract CV details from metadata
        cv_details = metadata.get("cv_details", {})

        # Extract regression features for cohort-based tracking
        auc = extract_auc(metrics)
        logloss = metrics.get("logloss")
        pr_auc = metrics.get("pr_auc")

        # Symbol-specific metrics
        sym_auc_mean, sym_auc_median, sym_auc_iqr, sym_auc_min, sym_auc_max, frac_symbols_good = \
            self._extract_symbol_metrics(metadata, metrics)

        # Route information
        route = metadata.get("view") or mode

        # Class balance and feature counts
        pos_rate = extract_pos_rate(metrics)
        n_features_pre, n_features_post_prune, n_features_selected = extract_feature_counts(metrics, metadata)

        # Temporal safety
        purge_minutes_used = extract_purge_minutes(metadata, cv_details)
        embargo_minutes_used = extract_embargo_minutes(metadata, cv_details)

        # Feature stability metrics
        jaccard_topK = metrics.get("jaccard_top_k") or metrics.get("jaccard_topK")
        rank_corr_spearman = metrics.get("rank_corr") or metrics.get("rank_correlation") or metrics.get("spearman_corr")
        importance_concentration = metrics.get("importance_concentration") or metrics.get("top10_importance_share")

        # Operational metrics
        runtime_sec = metrics.get("runtime_sec") or metrics.get("train_time_sec") or metrics.get("wall_clock_time")
        peak_ram_mb = metrics.get("peak_ram_mb") or metrics.get("peak_memory_mb")
        folds_executed = extract_folds(metadata, cv_details)

        # Identity fields
        data_fingerprint = metadata.get("data_fingerprint")
        featureset_hash = metadata.get("featureset_hash") or metrics.get("featureset_hash")
        config_hash = metadata.get("config_hash")
        git_commit = metadata.get("git_commit")

        # Create new row with all regression features
        new_row = {
            # Identity (categorical)
            "phase": phase,
            "mode": mode,
            "target": target,
            "symbol": symbol,
            "model_family": model_family,
            "cohort_id": cohort_id,
            "run_id": run_id,
            "segment_id": segment_id,
            "data_fingerprint": data_fingerprint,
            "featureset_hash": featureset_hash,
            "config_hash": config_hash,
            "git_commit": git_commit,

            # Sample size
            "n_effective": metadata.get("n_effective", 0),
            "n_symbols": metadata.get("n_symbols", 0),

            # Target ranking metrics
            "auc": auc,
            "logloss": logloss,
            "pr_auc": pr_auc,
            "sym_auc_mean": sym_auc_mean,
            "sym_auc_median": sym_auc_median,
            "sym_auc_iqr": sym_auc_iqr,
            "sym_auc_min": sym_auc_min,
            "sym_auc_max": sym_auc_max,
            "frac_symbols_good": frac_symbols_good,
            "composite_score": metrics.get("composite_score"),
            "mean_importance": metrics.get("mean_importance"),

            # Route stability
            "route": route,
            "route_changed": None,
            "route_entropy": None,

            # Class balance
            "pos_rate": pos_rate,

            # Feature counts
            "n_features_pre": n_features_pre,
            "n_features_post_prune": n_features_post_prune,
            "n_features_selected": n_features_selected,

            # Temporal safety
            "purge_minutes_used": purge_minutes_used,
            "embargo_minutes_used": embargo_minutes_used,
            "horizon_minutes": _extract_horizon_minutes_sst(metadata, cv_details),

            # Feature stability
            "jaccard_topK": jaccard_topK,
            "rank_corr_spearman": rank_corr_spearman,
            "importance_concentration": importance_concentration,

            # Operational metrics
            "runtime_sec": runtime_sec,
            "peak_ram_mb": peak_ram_mb,
            "folds_executed": folds_executed,

            # Timestamps
            "date": metadata.get("date_start"),
            "created_at": metadata.get("created_at", datetime.now().isoformat()),
            "run_started_at": self._parse_run_started_at(run_id, metadata.get("created_at")),

            # Decision fields
            "decision_level": metrics.get("decision_level") or 0,
            "decision_action_mask": json.dumps(metrics.get("decision_action_mask") or []) if metrics.get("decision_action_mask") else None,
            "decision_reason_codes": json.dumps(metrics.get("decision_reason_codes") or []) if metrics.get("decision_reason_codes") else None,

            # Path
            "path": self._calculate_cohort_relative_path(cohort_dir)
        }

        # Load existing index or create new
        if index_file.exists():
            try:
                df = pd.read_parquet(index_file)
            except Exception:
                df = pd.DataFrame()
        else:
            df = pd.DataFrame()

        # Append new row
        new_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_df], ignore_index=True)

        # Remove duplicates (keep latest)
        df = df.drop_duplicates(
            subset=["phase", "mode", "target", "symbol", "model_family", "cohort_id", "run_id"],
            keep="last"
        )

        # Save with file locking for concurrency safety
        self._save_index_with_lock(index_file, df, new_row)

    def _compute_segment_id(
        self,
        index_file: Path,
        cohort_id: str,
        metadata: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> Optional[int]:
        """Compute segment_id for decision-making (segments reset on identity breaks)."""
        try:
            from TRAINING.common.utils.regression_analysis import prepare_segments

            if not index_file.exists():
                return 0

            try:
                df_existing = pd.read_parquet(index_file)
                cohort_mask = df_existing['cohort_id'] == cohort_id
                if not cohort_mask.any():
                    return 0

                df_cohort = df_existing[cohort_mask].copy()
                df_cohort = prepare_segments(df_cohort, time_col='run_started_at')

                if len(df_cohort) == 0:
                    return 0

                last_segment = df_cohort['segment_id'].iloc[-1]
                last_row = df_cohort.iloc[-1]

                # Check if identity fields changed
                identity_cols = ["data_fingerprint", "featureset_hash", "config_hash", "git_commit"]
                for col in identity_cols:
                    new_val = metadata.get(col) or (metrics.get(col) if col in metrics else None)
                    old_val = last_row.get(col)
                    if new_val != old_val and (new_val is not None or old_val is not None):
                        return int(last_segment) + 1

                return int(last_segment)

            except Exception:
                return 0

        except ImportError:
            return None

    def _extract_symbol_metrics(
        self,
        metadata: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> tuple:
        """Extract symbol-specific AUC metrics."""
        per_symbol_stats = metadata.get("per_symbol_stats", {})
        sym_aucs = None

        if per_symbol_stats and isinstance(per_symbol_stats, dict):
            sym_aucs = [
                stats.get("auc")
                for stats in per_symbol_stats.values()
                if isinstance(stats, dict) and "auc" in stats
            ]
        elif "sym_aucs" in metrics:
            sym_aucs = metrics.get("sym_aucs")

        sym_auc_mean = sym_auc_median = sym_auc_iqr = sym_auc_min = sym_auc_max = None
        frac_symbols_good = None

        if sym_aucs and len(sym_aucs) > 0:
            sym_aucs_clean = [a for a in sym_aucs if a is not None and not np.isnan(a)]
            if len(sym_aucs_clean) > 0:
                sym_auc_mean = float(np.mean(sym_aucs_clean))
                sym_auc_median = float(np.median(sym_aucs_clean))
                q75, q25 = np.percentile(sym_aucs_clean, [75, 25])
                sym_auc_iqr = float(q75 - q25)
                sym_auc_min = float(np.min(sym_aucs_clean))
                sym_auc_max = float(np.max(sym_aucs_clean))

            # Fraction of symbols with good AUC
            try:
                from CONFIG.config_loader import get_cfg
                threshold = float(get_cfg(
                    "training.target_routing.auc_threshold",
                    default=0.65,
                    config_name="training_config"
                ))
            except Exception:
                threshold = 0.65

            good_count = sum(1 for a in sym_aucs if a is not None and a >= threshold)
            frac_symbols_good = good_count / len(sym_aucs) if len(sym_aucs) > 0 else None

        return sym_auc_mean, sym_auc_median, sym_auc_iqr, sym_auc_min, sym_auc_max, frac_symbols_good

    def _save_index_with_lock(
        self,
        index_file: Path,
        df: pd.DataFrame,
        new_row: Dict[str, Any]
    ) -> None:
        """Save index with file locking for concurrency safety."""
        try:
            index_file.parent.mkdir(parents=True, exist_ok=True)
            lock_file = index_file.with_suffix('.lock')

            with open(lock_file, 'w') as lock_f:
                try:
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)

                    # Re-read in case another process updated
                    if index_file.exists():
                        try:
                            df_existing = pd.read_parquet(index_file)
                            df = pd.concat([df_existing, df], ignore_index=True)
                            df = df.drop_duplicates(subset=['run_id', 'phase'], keep='last')

                            existing_mask = (
                                (df_existing['run_id'] == new_row['run_id']) &
                                (df_existing['phase'] == new_row['phase'])
                            )
                            if existing_mask.any():
                                logger.debug(
                                    f"Idempotency: Updating existing index entry for "
                                    f"run_id={new_row['run_id']}, phase={new_row['phase']}"
                                )
                        except Exception:
                            pass

                    df.to_parquet(index_file, index=False)

                    if hasattr(os, 'sync'):
                        try:
                            os.sync()
                        except AttributeError:
                            pass

                except Exception as e:
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
                    raise

        except Exception as e:
            error_type = (
                "IO_ERROR" if isinstance(e, (IOError, OSError))
                else "SERIALIZATION_ERROR" if isinstance(e, (json.JSONDecodeError, TypeError))
                else "UNKNOWN_ERROR"
            )
            logger.warning(f"Failed to save index.parquet to {index_file}: {e}, error_type={error_type}")
