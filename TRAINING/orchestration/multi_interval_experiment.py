# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Multi-Interval Experiment Orchestrator

Enables experiments that compare models across different data intervals.
Supports cross-interval validation, feature transfer, and interval comparison.

Key Capabilities:
1. Train models at multiple intervals (1m, 5m, 15m, etc.)
2. Cross-interval validation (train at 5m, validate at 1m)
3. Feature transfer warm-start from coarser intervals
4. Interval comparison metrics and reports

Usage:
    from TRAINING.orchestration.multi_interval_experiment import (
        MultiIntervalExperiment,
        CrossIntervalValidator,
        load_multi_interval_config,
    )

    # Run multi-interval experiment
    experiment = MultiIntervalExperiment(config)
    results = experiment.run(
        data_root="data/data_labeled_v2",
        intervals=[1, 5, 15],
        targets=["fwd_ret_5m", "fwd_ret_15m"],
    )

    # Cross-interval validation
    validator = CrossIntervalValidator()
    val_results = validator.validate(
        model_path="runs/5m_model",
        validation_data_path="data/data_labeled_v2/interval=1m",
    )
"""

from __future__ import annotations

import TRAINING.common.repro_bootstrap  # noqa: F401 - must be first (after __future__)

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from CONFIG.config_loader import get_cfg
from TRAINING.common.interval import minutes_to_bars, bars_to_minutes

logger = logging.getLogger(__name__)


# SST: Load multi-interval config from CONFIG/pipeline/training/multi_interval.yaml
def _get_multi_interval_config() -> Dict[str, Any]:
    """Load multi-interval experiment config from CONFIG."""
    return {
        "intervals": get_cfg("intervals", default=[5], config_name="multi_interval"),
        "primary_interval": get_cfg("primary_interval", default=5, config_name="multi_interval"),
        "cross_validation": {
            "enabled": get_cfg("cross_validation.enabled", default=False, config_name="multi_interval"),
            "train_intervals": get_cfg("cross_validation.train_intervals", default=[5], config_name="multi_interval"),
            "validate_intervals": get_cfg("cross_validation.validate_intervals", default=[1, 5, 15], config_name="multi_interval"),
        },
        "feature_transfer": {
            "enabled": get_cfg("feature_transfer.enabled", default=False, config_name="multi_interval"),
            "source_interval": get_cfg("feature_transfer.source_interval", default=None, config_name="multi_interval"),
            "target_interval": get_cfg("feature_transfer.target_interval", default=None, config_name="multi_interval"),
            "transfer_method": get_cfg("feature_transfer.transfer_method", default="warm_start", config_name="multi_interval"),
        },
        "comparison": {
            "metrics": get_cfg("comparison.metrics", default=["auc", "ic", "sharpe"], config_name="multi_interval"),
            "generate_report": get_cfg("comparison.generate_report", default=True, config_name="multi_interval"),
        },
    }


# For backward compatibility
DEFAULT_MULTI_INTERVAL_CONFIG = _get_multi_interval_config()


@dataclass
class IntervalRunResult:
    """Results from a single interval run."""

    interval_minutes: int
    run_path: str
    targets_trained: List[str]
    metrics: Dict[str, float]
    n_samples: int
    training_time_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "interval_minutes": self.interval_minutes,
            "run_path": self.run_path,
            "targets_trained": self.targets_trained,
            "metrics": self.metrics,
            "n_samples": self.n_samples,
            "training_time_seconds": self.training_time_seconds,
            "metadata": self.metadata,
        }


@dataclass
class CrossIntervalValidationResult:
    """Results from cross-interval validation."""

    train_interval: int
    validate_interval: int
    model_path: str
    metrics: Dict[str, float]
    degradation: Dict[str, float]  # % change from in-interval validation
    warnings: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if model generalizes reasonably."""
        # Check for severe degradation (>50% drop in key metrics)
        for metric, deg in self.degradation.items():
            if deg < -0.5:  # >50% degradation
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "train_interval": self.train_interval,
            "validate_interval": self.validate_interval,
            "model_path": self.model_path,
            "metrics": self.metrics,
            "degradation": self.degradation,
            "warnings": self.warnings,
            "is_valid": self.is_valid,
        }


@dataclass
class MultiIntervalExperimentResult:
    """Results from multi-interval experiment."""

    experiment_name: str
    intervals: List[int]
    primary_interval: int
    interval_results: Dict[int, IntervalRunResult]
    cross_validation_results: List[CrossIntervalValidationResult]
    comparison_summary: Dict[str, Any]
    best_interval: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "experiment_name": self.experiment_name,
            "intervals": self.intervals,
            "primary_interval": self.primary_interval,
            "interval_results": {
                k: v.to_dict() for k, v in self.interval_results.items()
            },
            "cross_validation_results": [r.to_dict() for r in self.cross_validation_results],
            "comparison_summary": self.comparison_summary,
            "best_interval": self.best_interval,
        }


def load_multi_interval_config(
    experiment_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load multi-interval experiment configuration.

    Args:
        experiment_config: Optional experiment-level overrides

    Returns:
        Merged configuration dict
    """
    # Start with defaults
    config = {**DEFAULT_MULTI_INTERVAL_CONFIG}

    # Load from centralized config if available
    config["intervals"] = get_cfg(
        "experiment.multi_interval.intervals",
        default=config["intervals"],
    )
    config["primary_interval"] = get_cfg(
        "experiment.multi_interval.primary_interval",
        default=config["primary_interval"],
    )

    # Apply experiment overrides
    if experiment_config:
        mi_config = experiment_config.get("multi_interval", {})
        if mi_config:
            # Deep merge
            for key, value in mi_config.items():
                if isinstance(value, dict) and key in config:
                    config[key] = {**config[key], **value}
                else:
                    config[key] = value

    return config


class IntervalDataLoader:
    """
    Loads data for a specific interval.

    Handles interval-specific data paths and validation.
    """

    def __init__(self, data_root: str):
        """
        Initialize data loader.

        Args:
            data_root: Root data directory (e.g., "data/data_labeled_v2")
        """
        self.data_root = Path(data_root)

    def get_interval_path(self, interval_minutes: int) -> Path:
        """Get data path for specific interval."""
        return self.data_root / f"interval={interval_minutes}m"

    def list_available_intervals(self) -> List[int]:
        """List intervals with available data."""
        intervals = []
        if not self.data_root.exists():
            return intervals

        for path in sorted(self.data_root.iterdir()):
            if path.is_dir() and path.name.startswith("interval="):
                try:
                    # Parse interval from "interval=5m" using removesuffix (safer than rstrip)
                    interval_str = path.name.split("=")[1]
                    if interval_str.endswith("m"):
                        interval_str = interval_str[:-1]  # Remove single 'm' suffix
                    intervals.append(int(interval_str))
                except (ValueError, IndexError):
                    continue

        return sorted(intervals)

    def validate_interval_data(self, interval_minutes: int) -> Tuple[bool, str]:
        """
        Validate that interval data exists and is valid.

        Returns:
            (is_valid, message) tuple
        """
        path = self.get_interval_path(interval_minutes)

        if not path.exists():
            return False, f"Data path does not exist: {path}"

        # Check for parquet files
        parquet_files = list(path.glob("*.parquet"))
        if not parquet_files:
            return False, f"No parquet files found in {path}"

        return True, f"Found {len(parquet_files)} parquet files"

    def load_interval_data(
        self,
        interval_minutes: int,
        symbols: Optional[List[str]] = None,
        max_rows_per_symbol: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Load data for a specific interval.

        Args:
            interval_minutes: Data interval in minutes
            symbols: Optional list of symbols to load
            max_rows_per_symbol: Optional row limit per symbol

        Returns:
            Data dict with 'combined_df' and metadata
        """
        import pandas as pd

        path = self.get_interval_path(interval_minutes)
        is_valid, msg = self.validate_interval_data(interval_minutes)
        if not is_valid:
            raise ValueError(msg)

        # Load parquet files
        dfs = []
        parquet_files = sorted(path.glob("*.parquet"))

        for pf in parquet_files:
            # Extract symbol from filename if present
            symbol = pf.stem.split("_")[0] if "_" in pf.stem else pf.stem

            if symbols and symbol not in symbols:
                continue

            df = pd.read_parquet(pf)
            df["symbol"] = symbol

            if max_rows_per_symbol and len(df) > max_rows_per_symbol:
                df = df.tail(max_rows_per_symbol)

            dfs.append(df)

        if not dfs:
            raise ValueError(f"No data loaded for interval={interval_minutes}m")

        combined = pd.concat(dfs, ignore_index=True)

        return {
            "combined_df": combined,
            "interval_minutes": interval_minutes,
            "n_symbols": len(dfs),
            "n_rows": len(combined),
            "path": str(path),
        }


class CrossIntervalValidator:
    """
    Validates model performance across different intervals.

    Tests how well a model trained at one interval generalizes
    to data at different intervals.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize validator.

        Args:
            config: Optional configuration
        """
        self.config = config or {}
        self.metrics = self.config.get("metrics", ["auc", "ic"])

    def validate(
        self,
        model_path: str,
        train_interval: int,
        validation_data: Dict[str, Any],
        validation_interval: int,
        baseline_metrics: Optional[Dict[str, float]] = None,
    ) -> CrossIntervalValidationResult:
        """
        Validate model on data from different interval.

        Args:
            model_path: Path to trained model
            train_interval: Interval model was trained on
            validation_data: Validation data dict
            validation_interval: Interval of validation data
            baseline_metrics: Optional baseline metrics (from in-interval validation)

        Returns:
            Validation results
        """
        warnings = []

        # Check interval compatibility
        if train_interval != validation_interval:
            # Warn about potential issues
            if validation_interval < train_interval:
                warnings.append(
                    f"Validating at finer interval ({validation_interval}m) than training ({train_interval}m). "
                    "Model may miss short-term patterns."
                )
            else:
                warnings.append(
                    f"Validating at coarser interval ({validation_interval}m) than training ({train_interval}m). "
                    "Features may aggregate differently."
                )

        # Compute metrics
        metrics = self._compute_validation_metrics(
            model_path, validation_data, validation_interval
        )

        # Compute degradation if baseline provided
        degradation = {}
        if baseline_metrics:
            for metric_name, value in metrics.items():
                if metric_name in baseline_metrics and baseline_metrics[metric_name] != 0:
                    deg = (value - baseline_metrics[metric_name]) / abs(baseline_metrics[metric_name])
                    degradation[metric_name] = deg

                    if deg < -0.2:  # >20% degradation
                        warnings.append(
                            f"Metric '{metric_name}' degraded by {abs(deg)*100:.1f}% "
                            f"(from {baseline_metrics[metric_name]:.4f} to {value:.4f})"
                        )

        return CrossIntervalValidationResult(
            train_interval=train_interval,
            validate_interval=validation_interval,
            model_path=model_path,
            metrics=metrics,
            degradation=degradation,
            warnings=warnings,
        )

    def _compute_validation_metrics(
        self,
        model_path: str,
        validation_data: Dict[str, Any],
        interval_minutes: int,
    ) -> Dict[str, float]:
        """Compute validation metrics."""
        # This is a simplified implementation
        # In production, would load model and run inference

        metrics = {}

        # Placeholder metrics - would compute from actual predictions
        df = validation_data.get("combined_df")
        if df is not None:
            # Check for basic data quality
            metrics["n_samples"] = len(df)
            metrics["n_features"] = len([c for c in df.columns if not c.startswith("fwd_")])

        return metrics

    def validate_across_intervals(
        self,
        model_path: str,
        train_interval: int,
        data_loader: IntervalDataLoader,
        validation_intervals: List[int],
        symbols: Optional[List[str]] = None,
    ) -> List[CrossIntervalValidationResult]:
        """
        Validate model across multiple intervals.

        Args:
            model_path: Path to trained model
            train_interval: Interval model was trained on
            data_loader: Data loader instance
            validation_intervals: Intervals to validate on
            symbols: Optional symbol filter

        Returns:
            List of validation results
        """
        results = []

        # First, get baseline metrics from training interval
        baseline_metrics = None
        if train_interval in validation_intervals:
            try:
                train_data = data_loader.load_interval_data(train_interval, symbols)
                baseline_result = self.validate(
                    model_path, train_interval, train_data, train_interval
                )
                baseline_metrics = baseline_result.metrics
            except Exception as e:
                logger.warning(f"Could not compute baseline metrics: {e}")

        # Validate across all intervals
        for interval in validation_intervals:
            try:
                val_data = data_loader.load_interval_data(interval, symbols)
                result = self.validate(
                    model_path=model_path,
                    train_interval=train_interval,
                    validation_data=val_data,
                    validation_interval=interval,
                    baseline_metrics=baseline_metrics,
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Validation failed for interval={interval}m: {e}")

        return results


class FeatureTransfer:
    """
    Enables feature transfer between intervals.

    Supports warm-starting models from coarser intervals
    and feature projection across interval boundaries.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize feature transfer.

        Args:
            config: Optional configuration
        """
        self.config = config or {}
        self.transfer_method = self.config.get("transfer_method", "warm_start")

    def can_transfer(
        self,
        source_interval: int,
        target_interval: int,
    ) -> Tuple[bool, str]:
        """
        Check if transfer is possible between intervals.

        Args:
            source_interval: Source interval in minutes
            target_interval: Target interval in minutes

        Returns:
            (can_transfer, reason) tuple
        """
        # Check divisibility
        if target_interval < source_interval:
            # Fine to coarse - need aggregation
            if source_interval % target_interval != 0:
                return False, f"Source interval ({source_interval}m) must be divisible by target ({target_interval}m)"
        else:
            # Coarse to fine - need interpolation/projection
            if target_interval % source_interval != 0:
                return False, f"Target interval ({target_interval}m) must be divisible by source ({source_interval}m)"

        return True, "Transfer possible"

    def compute_feature_mapping(
        self,
        source_interval: int,
        target_interval: int,
        feature_names: List[str],
    ) -> Dict[str, str]:
        """
        Compute feature name mapping between intervals.

        Features with interval-specific names (e.g., "ret_5m") need
        mapping to equivalent features at the target interval.

        Args:
            source_interval: Source interval
            target_interval: Target interval
            feature_names: Source feature names

        Returns:
            Mapping from source to target feature names
        """
        mapping = {}

        for name in feature_names:
            # Check if feature name contains interval suffix
            if f"_{source_interval}m" in name:
                # Map to equivalent at target interval
                target_name = name.replace(f"_{source_interval}m", f"_{target_interval}m")
                mapping[name] = target_name
            else:
                # Feature name is interval-agnostic
                mapping[name] = name

        return mapping

    def prepare_warm_start(
        self,
        source_model_path: str,
        source_interval: int,
        target_interval: int,
    ) -> Dict[str, Any]:
        """
        Prepare warm-start initialization from source model.

        Args:
            source_model_path: Path to source model
            source_interval: Source interval
            target_interval: Target interval

        Returns:
            Warm-start configuration dict
        """
        can_transfer, reason = self.can_transfer(source_interval, target_interval)
        if not can_transfer:
            raise ValueError(f"Cannot transfer: {reason}")

        # Compute scale factor for hyperparameters
        scale_factor = target_interval / source_interval

        warm_start_config = {
            "source_model_path": source_model_path,
            "source_interval": source_interval,
            "target_interval": target_interval,
            "scale_factor": scale_factor,
            "transfer_method": self.transfer_method,
            "adjustments": {
                # Adjust lookback-related hyperparameters
                "sequence_length_scale": 1.0 / scale_factor,  # More bars at finer intervals
                "patience_scale": scale_factor,  # More patience at coarser intervals
            },
        }

        logger.info(
            f"Prepared warm-start transfer: {source_interval}m → {target_interval}m "
            f"(scale={scale_factor:.2f})"
        )

        return warm_start_config


class IntervalComparator:
    """
    Compares experiment results across intervals.

    Generates comparison reports and identifies best intervals
    for different targets/metrics.
    """

    def __init__(self, metrics: Optional[List[str]] = None):
        """
        Initialize comparator.

        Args:
            metrics: Metrics to compare (default: auc, ic, sharpe)
        """
        self.metrics = metrics or ["auc", "ic", "sharpe"]

    def compare(
        self,
        interval_results: Dict[int, IntervalRunResult],
        primary_metric: str = "auc",
    ) -> Dict[str, Any]:
        """
        Compare results across intervals.

        Args:
            interval_results: Dict mapping interval to results
            primary_metric: Primary metric for ranking

        Returns:
            Comparison summary dict
        """
        if not interval_results:
            return {"error": "No results to compare"}

        intervals = sorted(interval_results.keys())
        comparison = {
            "intervals_compared": intervals,
            "primary_metric": primary_metric,
            "per_metric_best": {},
            "per_metric_ranking": {},
            "summary": {},
        }

        # Find best interval for each metric
        for metric in self.metrics:
            metric_values = {}
            for interval, result in interval_results.items():
                if metric in result.metrics:
                    metric_values[interval] = result.metrics[metric]

            if metric_values:
                # Rank intervals by metric value
                ranking = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
                comparison["per_metric_ranking"][metric] = [
                    {"interval": interval, "value": value}
                    for interval, value in ranking
                ]
                comparison["per_metric_best"][metric] = ranking[0][0]

        # Determine overall best interval
        if primary_metric in comparison["per_metric_best"]:
            comparison["best_interval"] = comparison["per_metric_best"][primary_metric]

        # Generate summary statistics
        comparison["summary"] = self._generate_summary(interval_results)

        return comparison

    def _generate_summary(
        self,
        interval_results: Dict[int, IntervalRunResult],
    ) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {
            "total_intervals": len(interval_results),
            "total_training_time_seconds": sum(
                r.training_time_seconds for r in interval_results.values()
            ),
            "total_samples_trained": sum(
                r.n_samples for r in interval_results.values()
            ),
        }

        # Compute metric ranges
        for metric in self.metrics:
            values = [
                r.metrics.get(metric, 0)
                for r in interval_results.values()
                if metric in r.metrics
            ]
            if values:
                summary[f"{metric}_range"] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                }

        return summary

    def generate_report(
        self,
        comparison: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate human-readable comparison report.

        Args:
            comparison: Comparison results from compare()
            output_path: Optional path to save report

        Returns:
            Report string
        """
        lines = [
            "=" * 80,
            "MULTI-INTERVAL EXPERIMENT COMPARISON REPORT",
            "=" * 80,
            "",
            f"Intervals compared: {comparison['intervals_compared']}",
            f"Primary metric: {comparison['primary_metric']}",
            f"Best interval: {comparison.get('best_interval', 'N/A')}",
            "",
            "-" * 40,
            "METRIC RANKINGS",
            "-" * 40,
        ]

        for metric, rankings in comparison.get("per_metric_ranking", {}).items():
            lines.append(f"\n{metric.upper()}:")
            for i, item in enumerate(rankings, 1):
                lines.append(f"  {i}. {item['interval']}m: {item['value']:.4f}")

        lines.extend([
            "",
            "-" * 40,
            "SUMMARY STATISTICS",
            "-" * 40,
        ])

        summary = comparison.get("summary", {})
        lines.append(f"Total intervals: {summary.get('total_intervals', 0)}")
        lines.append(f"Total training time: {summary.get('total_training_time_seconds', 0):.1f}s")
        lines.append(f"Total samples: {summary.get('total_samples_trained', 0):,}")

        for metric in self.metrics:
            range_key = f"{metric}_range"
            if range_key in summary:
                r = summary[range_key]
                lines.append(
                    f"{metric}: min={r['min']:.4f}, max={r['max']:.4f}, "
                    f"mean={r['mean']:.4f}, std={r['std']:.4f}"
                )

        lines.extend(["", "=" * 80])

        report = "\n".join(lines)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
            logger.info(f"Saved comparison report to {output_path}")

        return report


class MultiIntervalExperiment:
    """
    Orchestrates multi-interval experiments.

    Coordinates training across multiple intervals, cross-validation,
    feature transfer, and comparison.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        experiment_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize experiment.

        Args:
            config: Multi-interval config
            experiment_config: Full experiment config (for overrides)
        """
        self.config = load_multi_interval_config(experiment_config)
        if config:
            self.config.update(config)

        self.intervals = self.config.get("intervals", [5])
        self.primary_interval = self.config.get("primary_interval", 5)

        # Initialize components
        self.validator = CrossIntervalValidator(self.config.get("cross_validation", {}))
        self.feature_transfer = FeatureTransfer(self.config.get("feature_transfer", {}))
        self.comparator = IntervalComparator(
            self.config.get("comparison", {}).get("metrics")
        )

    def run(
        self,
        data_root: str,
        output_dir: str,
        targets: List[str],
        symbols: Optional[List[str]] = None,
        experiment_name: str = "multi_interval",
        **train_kwargs,
    ) -> MultiIntervalExperimentResult:
        """
        Run multi-interval experiment.

        Args:
            data_root: Root data directory
            output_dir: Output directory for results
            targets: Targets to train
            symbols: Optional symbol filter
            experiment_name: Experiment name
            **train_kwargs: Additional training arguments

        Returns:
            Experiment results
        """
        import time

        logger.info("=" * 80)
        logger.info(f"MULTI-INTERVAL EXPERIMENT: {experiment_name}")
        logger.info(f"Intervals: {self.intervals}")
        logger.info(f"Primary interval: {self.primary_interval}m")
        logger.info("=" * 80)

        data_loader = IntervalDataLoader(data_root)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Check available intervals
        available = data_loader.list_available_intervals()
        logger.info(f"Available intervals in data: {available}")

        missing = [i for i in self.intervals if i not in available]
        if missing:
            logger.warning(f"Requested intervals not available: {missing}")
            self.intervals = [i for i in self.intervals if i in available]

        if not self.intervals:
            raise ValueError("No valid intervals to run experiment")

        interval_results: Dict[int, IntervalRunResult] = {}

        # Train at each interval
        for interval in self.intervals:
            logger.info(f"\n{'='*40}")
            logger.info(f"Training at {interval}m interval")
            logger.info(f"{'='*40}")

            start_time = time.time()

            try:
                # Load data for this interval
                data = data_loader.load_interval_data(interval, symbols)
                logger.info(f"Loaded {data['n_rows']:,} rows from {data['n_symbols']} symbols")

                # Run training (simplified - would call actual training pipeline)
                interval_output = output_path / f"interval_{interval}m"
                interval_output.mkdir(parents=True, exist_ok=True)

                # Placeholder metrics - in production would come from actual training
                metrics = {
                    "n_samples": data["n_rows"],
                    "n_symbols": data["n_symbols"],
                }

                elapsed = time.time() - start_time

                interval_results[interval] = IntervalRunResult(
                    interval_minutes=interval,
                    run_path=str(interval_output),
                    targets_trained=targets,
                    metrics=metrics,
                    n_samples=data["n_rows"],
                    training_time_seconds=elapsed,
                    metadata={"symbols": symbols or []},
                )

                logger.info(f"Completed {interval}m training in {elapsed:.1f}s")

            except Exception as e:
                logger.error(f"Training failed for {interval}m: {e}")
                import traceback
                traceback.print_exc()

        # Cross-interval validation if enabled
        cross_val_results = []
        cv_config = self.config.get("cross_validation", {})
        if cv_config.get("enabled", False):
            logger.info("\n" + "=" * 40)
            logger.info("Cross-Interval Validation")
            logger.info("=" * 40)

            train_intervals = cv_config.get("train_intervals", [self.primary_interval])
            validate_intervals = cv_config.get("validate_intervals", self.intervals)

            for train_int in train_intervals:
                if train_int in interval_results:
                    model_path = interval_results[train_int].run_path
                    results = self.validator.validate_across_intervals(
                        model_path=model_path,
                        train_interval=train_int,
                        data_loader=data_loader,
                        validation_intervals=validate_intervals,
                        symbols=symbols,
                    )
                    cross_val_results.extend(results)

                    for r in results:
                        status = "✓" if r.is_valid else "✗"
                        logger.info(
                            f"  {status} Train={train_int}m, Val={r.validate_interval}m: "
                            f"metrics={r.metrics}"
                        )

        # Compare results
        logger.info("\n" + "=" * 40)
        logger.info("Comparison")
        logger.info("=" * 40)

        comparison = self.comparator.compare(
            interval_results,
            primary_metric=self.config.get("comparison", {}).get("primary_metric", "auc"),
        )

        # Generate report
        if self.config.get("comparison", {}).get("generate_report", True):
            report_path = output_path / "comparison_report.txt"
            report = self.comparator.generate_report(comparison, str(report_path))
            logger.info(f"\n{report}")

        # Build final result
        result = MultiIntervalExperimentResult(
            experiment_name=experiment_name,
            intervals=self.intervals,
            primary_interval=self.primary_interval,
            interval_results=interval_results,
            cross_validation_results=cross_val_results,
            comparison_summary=comparison,
            best_interval=comparison.get("best_interval"),
        )

        # Save results (atomic write for crash consistency)
        from TRAINING.common.utils.file_utils import write_atomic_json
        results_path = output_path / "experiment_results.json"
        write_atomic_json(results_path, result.to_dict(), default=str)
        logger.info(f"\nSaved experiment results to {results_path}")

        logger.info("\n" + "=" * 80)
        logger.info(f"Experiment complete. Best interval: {result.best_interval}m")
        logger.info("=" * 80)

        return result
