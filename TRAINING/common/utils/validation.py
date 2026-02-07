# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Validation Utilities

Validation functions for model training and data quality.
"""


import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class ValidationUtils:
    """Validation utilities for model training"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
    def validate_training_data(self, X: np.ndarray, y_dict: Dict[str, np.ndarray], 
                             feature_names: List[str]) -> Dict[str, Any]:
        """Validate training data quality"""
        
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'stats': {}
        }
        
        # Check X
        x_validation = self._validate_features(X, feature_names)
        validation_results['stats']['features'] = x_validation
        
        if not x_validation['valid']:
            validation_results['valid'] = False
            validation_results['errors'].extend(x_validation['errors'])
        
        # Check y_dict
        for target, y in y_dict.items():
            y_validation = self._validate_target(y, target)
            validation_results['stats'][target] = y_validation
            
            if not y_validation['valid']:
                validation_results['valid'] = False
                validation_results['errors'].extend(y_validation['errors'])
            else:
                validation_results['warnings'].extend(y_validation['warnings'])
        
        return validation_results
    
    def _validate_features(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Validate feature matrix"""
        
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check dimensions
        if X.shape[0] == 0:
            validation['valid'] = False
            validation['errors'].append("X is empty")
            return validation
        
        if X.shape[1] != len(feature_names):
            validation['valid'] = False
            validation['errors'].append(f"Feature count mismatch: X has {X.shape[1]} columns, {len(feature_names)} names")
            return validation
        
        # Check for NaN values
        n_nan = np.sum(np.isnan(X))
        n_total = X.size
        nan_ratio = n_nan / n_total if n_total > 0 else 0
        
        validation['stats']['n_samples'] = X.shape[0]
        validation['stats']['n_features'] = X.shape[1]
        validation['stats']['n_nan'] = n_nan
        validation['stats']['nan_ratio'] = nan_ratio
        
        if nan_ratio > 0.1:  # More than 10% NaN
            validation['warnings'].append(f"High NaN ratio: {nan_ratio:.2%}")
        
        # Check for constant features
        constant_features = []
        for i in range(X.shape[1]):
            feature_data = X[:, i]
            valid_data = feature_data[~np.isnan(feature_data)]
            if len(valid_data) > 0 and len(np.unique(valid_data)) <= 1:
                constant_features.append(feature_names[i])
        
        if constant_features:
            validation['warnings'].append(f"Constant features: {constant_features}")
        
        # Check for infinite values
        n_inf = np.sum(np.isinf(X))
        if n_inf > 0:
            validation['warnings'].append(f"Infinite values found: {n_inf}")
        
        return validation
    
    def _validate_target(self, y: np.ndarray, target: str) -> Dict[str, Any]:
        """Validate target data"""
        
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check for empty target
        if len(y) == 0:
            validation['valid'] = False
            validation['errors'].append(f"Target {target} is empty")
            return validation
        
        # Check for NaN values
        n_nan = np.sum(np.isnan(y))
        n_total = len(y)
        nan_ratio = n_nan / n_total if n_total > 0 else 0
        
        validation['stats']['n_samples'] = n_total
        validation['stats']['n_nan'] = n_nan
        validation['stats']['nan_ratio'] = nan_ratio
        
        if nan_ratio > 0.5:  # More than 50% NaN
            validation['valid'] = False
            validation['errors'].append(f"Target {target} has too many NaN values: {nan_ratio:.2%}")
        elif nan_ratio > 0.1:  # More than 10% NaN
            validation['warnings'].append(f"Target {target} has high NaN ratio: {nan_ratio:.2%}")
        
        # Check for constant values
        valid_data = y[~np.isnan(y)]
        if len(valid_data) > 0:
            n_unique = len(np.unique(valid_data))
            validation['stats']['n_unique'] = n_unique
            
            if n_unique <= 1:
                validation['valid'] = False
                validation['errors'].append(f"Target {target} is constant")
            elif n_unique <= 2:
                validation['warnings'].append(f"Target {target} has only {n_unique} unique values")
            
            # Check data range
            data_min = np.min(valid_data)
            data_max = np.max(valid_data)
            validation['stats']['range'] = (data_min, data_max)
            
            # Check for infinite values
            n_inf = np.sum(np.isinf(valid_data))
            if n_inf > 0:
                validation['warnings'].append(f"Target {target} has infinite values: {n_inf}")
            
            # Check for extreme values
            if target.startswith('fwd_ret_'):
                # Forward returns should be reasonable
                if abs(data_max) > 1.0:  # More than 100% return
                    validation['warnings'].append(f"Target {target} has extreme values: max={data_max:.3f}")
        else:
            validation['valid'] = False
            validation['errors'].append(f"Target {target} has no valid data")
        
        return validation
    
    def validate_model_predictions(self, predictions: Dict[str, np.ndarray], 
                                 expected_targets: List[str]) -> Dict[str, Any]:
        """Validate model predictions"""
        
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check if all expected targets are present
        missing_targets = set(expected_targets) - set(predictions.keys())
        if missing_targets:
            validation['valid'] = False
            validation['errors'].append(f"Missing predictions for targets: {missing_targets}")
        
        # Check each prediction
        for target, pred in predictions.items():
            pred_validation = self._validate_prediction(pred, target)
            validation['stats'][target] = pred_validation
            
            if not pred_validation['valid']:
                validation['valid'] = False
                validation['errors'].extend(pred_validation['errors'])
            else:
                validation['warnings'].extend(pred_validation['warnings'])
        
        return validation
    
    def _validate_prediction(self, pred: np.ndarray, target: str) -> Dict[str, Any]:
        """Validate a single prediction"""
        
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check for NaN values
        n_nan = np.sum(np.isnan(pred))
        n_total = len(pred)
        nan_ratio = n_nan / n_total if n_total > 0 else 0
        
        validation['stats']['n_samples'] = n_total
        validation['stats']['n_nan'] = n_nan
        validation['stats']['nan_ratio'] = nan_ratio
        
        if nan_ratio > 0.1:  # More than 10% NaN
            validation['warnings'].append(f"Prediction {target} has high NaN ratio: {nan_ratio:.2%}")
        
        # Check for infinite values
        n_inf = np.sum(np.isinf(pred))
        if n_inf > 0:
            validation['warnings'].append(f"Prediction {target} has infinite values: {n_inf}")
        
        # Check data range
        valid_data = pred[~np.isnan(pred) & ~np.isinf(pred)]
        if len(valid_data) > 0:
            data_min = np.min(valid_data)
            data_max = np.max(valid_data)
            validation['stats']['range'] = (data_min, data_max)
            
            # Check for reasonable ranges based on target type
            if target.startswith('fwd_ret_'):
                # Forward returns should be reasonable
                if abs(data_max) > 2.0:  # More than 200% return
                    validation['warnings'].append(f"Prediction {target} has extreme values: max={data_max:.3f}")
            elif any(target.startswith(prefix) for prefix in 
                    ['will_peak', 'will_valley', 'mdd', 'mfe']):
                # Probability-like targets should be in [0, 1]
                if data_min < 0 or data_max > 1:
                    validation['warnings'].append(f"Prediction {target} outside [0,1] range: [{data_min:.3f}, {data_max:.3f}]")
        
        return validation
    
    def validate_training_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate training configuration"""
        
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check required fields
        required_fields = ['strategy', 'targets']
        for field in required_fields:
            if field not in config:
                validation['valid'] = False
                validation['errors'].append(f"Missing required field: {field}")
        
        # Check strategy
        if 'strategy' in config:
            valid_strategies = ['single_task', 'multi_task', 'cascade']
            if config['strategy'] not in valid_strategies:
                validation['valid'] = False
                validation['errors'].append(f"Invalid strategy: {config['strategy']}. Must be one of {valid_strategies}")
        
        # Check targets
        if 'targets' in config:
            if not isinstance(config['targets'], list) or len(config['targets']) == 0:
                validation['valid'] = False
                validation['errors'].append("Targets must be a non-empty list")
        
        # Check model configuration
        if 'models' in config:
            model_config = config['models']
            if not isinstance(model_config, dict):
                validation['warnings'].append("Model configuration should be a dictionary")
        
        return validation
    
    def create_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """Create a human-readable validation report"""
        
        report = []
        report.append("=== Validation Report ===")
        
        if validation_results['valid']:
            report.append("✅ Validation PASSED")
        else:
            report.append("❌ Validation FAILED")
        
        if validation_results['errors']:
            report.append("\n🚨 Errors:")
            for error in validation_results['errors']:
                report.append(f"  - {error}")
        
        if validation_results['warnings']:
            report.append("\n⚠️  Warnings:")
            for warning in validation_results['warnings']:
                report.append(f"  - {warning}")
        
        # Add statistics
        if 'stats' in validation_results:
            report.append("\n📊 Statistics:")
            for key, value in validation_results['stats'].items():
                if isinstance(value, dict):
                    report.append(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        report.append(f"    {sub_key}: {sub_value}")
                else:
                    report.append(f"  {key}: {value}")
        
        return "\n".join(report)


# =============================================================================
# Phase 14-15: Alignment Validation and Gap Detection (Interval-Agnostic Pipeline)
# =============================================================================

def validate_timestamp_alignment(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    timestamp_col: str = 'ts',
    strict: bool = False
) -> Dict[str, Any]:
    """
    Phase 14: Validate feature-target timestamp alignment before merge.

    Args:
        features_df: DataFrame with features
        targets_df: DataFrame with targets
        timestamp_col: Name of timestamp column
        strict: If True, raise error on misalignment; if False, log warning

    Returns:
        Dict with alignment validation results

    Raises:
        ValueError: If strict=True and alignment check fails
    """
    result = {
        'valid': True,
        'feature_rows': len(features_df),
        'target_rows': len(targets_df),
        'common_timestamps': 0,
        'feature_only': 0,
        'target_only': 0,
        'warnings': [],
        'errors': []
    }

    # Check timestamp column exists
    if timestamp_col not in features_df.columns:
        result['valid'] = False
        result['errors'].append(f"Feature DataFrame missing timestamp column: {timestamp_col}")
        if strict:
            raise ValueError(result['errors'][-1])
        return result

    if timestamp_col not in targets_df.columns:
        result['valid'] = False
        result['errors'].append(f"Target DataFrame missing timestamp column: {timestamp_col}")
        if strict:
            raise ValueError(result['errors'][-1])
        return result

    # Get unique timestamps
    feature_ts = set(features_df[timestamp_col].unique())
    target_ts = set(targets_df[timestamp_col].unique())

    # Compute alignment stats
    common_ts = feature_ts & target_ts
    feature_only_ts = feature_ts - target_ts
    target_only_ts = target_ts - feature_ts

    result['common_timestamps'] = len(common_ts)
    result['feature_only'] = len(feature_only_ts)
    result['target_only'] = len(target_only_ts)

    # Check row counts match for common timestamps
    if len(common_ts) == 0:
        result['valid'] = False
        result['errors'].append("No common timestamps between features and targets")
        if strict:
            raise ValueError(result['errors'][-1])
        return result

    # Warn if significant mismatch
    total_unique = len(feature_ts | target_ts)
    alignment_ratio = len(common_ts) / total_unique if total_unique > 0 else 0

    if alignment_ratio < 0.9:
        msg = f"Low timestamp alignment: {alignment_ratio:.1%} ({len(common_ts)}/{total_unique})"
        result['warnings'].append(msg)
        if strict and alignment_ratio < 0.5:
            result['valid'] = False
            result['errors'].append(msg)
            raise ValueError(msg)

    logger.info(
        f"Phase 14: Timestamp alignment validated | "
        f"common={len(common_ts)} feature_only={len(feature_only_ts)} target_only={len(target_only_ts)}"
    )

    return result


def detect_data_gaps(
    df: pd.DataFrame,
    timestamp_col: str = 'ts',
    expected_interval_minutes: Optional[float] = None,
    interval_minutes: float = 5.0,
    max_gap_multiple: float = 2.0,
    log_gaps: bool = True
) -> Dict[str, Any]:
    """
    Phase 15: Detect gaps in timestamp sequence.

    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column
        expected_interval_minutes: Expected interval between rows (auto-detected if None)
        interval_minutes: Fallback interval in minutes
        max_gap_multiple: Gaps larger than interval * max_gap_multiple are flagged
        log_gaps: Whether to log gap information

    Returns:
        Dict with gap detection results including:
        - has_gaps: bool
        - n_gaps: int
        - gap_indices: list of row indices where gaps occur
        - gap_durations_minutes: list of gap durations
        - total_gap_minutes: total time lost to gaps
    """
    result = {
        'has_gaps': False,
        'n_gaps': 0,
        'gap_indices': [],
        'gap_durations_minutes': [],
        'total_gap_minutes': 0.0,
        'expected_interval_minutes': None,
        'rows_checked': len(df)
    }

    if len(df) < 2:
        return result

    if timestamp_col not in df.columns:
        logger.warning(f"Phase 15: Timestamp column '{timestamp_col}' not found in DataFrame")
        return result

    # Convert timestamps to datetime if needed
    ts = df[timestamp_col]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        try:
            ts = pd.to_datetime(ts, unit='ns')
        except Exception:
            ts = pd.to_datetime(ts)

    # Sort by timestamp
    ts_sorted = ts.sort_values()

    # Compute diffs
    diffs = ts_sorted.diff().dropna()
    diffs_minutes = diffs.dt.total_seconds() / 60.0

    # Auto-detect expected interval if not provided
    if expected_interval_minutes is None:
        # Use median diff as expected interval
        expected_interval_minutes = float(diffs_minutes.median())
        if expected_interval_minutes <= 0:
            expected_interval_minutes = interval_minutes

    result['expected_interval_minutes'] = expected_interval_minutes

    # Detect gaps (diffs larger than expected * max_gap_multiple)
    gap_threshold = expected_interval_minutes * max_gap_multiple
    gaps_mask = diffs_minutes > gap_threshold

    if gaps_mask.any():
        result['has_gaps'] = True
        result['n_gaps'] = int(gaps_mask.sum())
        result['gap_indices'] = list(gaps_mask[gaps_mask].index)
        result['gap_durations_minutes'] = list(diffs_minutes[gaps_mask].values)
        result['total_gap_minutes'] = float(diffs_minutes[gaps_mask].sum())

        if log_gaps:
            logger.warning(
                f"Phase 15: Detected {result['n_gaps']} data gaps | "
                f"total_gap={result['total_gap_minutes']:.1f}m | "
                f"expected_interval={expected_interval_minutes:.1f}m"
            )
    else:
        if log_gaps:
            logger.debug(
                f"Phase 15: No gaps detected | "
                f"rows={len(df)} expected_interval={expected_interval_minutes:.1f}m"
            )

    return result


def validate_interval_consistency(
    df: pd.DataFrame,
    timestamp_col: str = 'ts',
    expected_interval_minutes: float = 5.0,
    tolerance_pct: float = 0.1,
) -> Dict[str, Any]:
    """
    Phase 14/15: Validate that data has consistent interval.

    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column
        expected_interval_minutes: Expected interval in minutes
        tolerance_pct: Tolerance percentage for interval validation

    Returns:
        Dict with validation results
    """
    result = {
        'valid': True,
        'expected_interval_minutes': expected_interval_minutes,
        'detected_interval_minutes': None,
        'consistency_ratio': 0.0,
        'warnings': [],
        'errors': []
    }

    gap_info = detect_data_gaps(
        df, timestamp_col,
        expected_interval_minutes=expected_interval_minutes,
        log_gaps=False
    )

    result['detected_interval_minutes'] = gap_info['expected_interval_minutes']

    if gap_info['rows_checked'] < 2:
        result['warnings'].append("Not enough rows to validate interval consistency")
        return result

    # Check if detected interval matches expected
    detected = gap_info['expected_interval_minutes']
    diff_pct = abs(detected - expected_interval_minutes) / expected_interval_minutes

    if diff_pct > tolerance_pct:
        result['valid'] = False
        result['errors'].append(
            f"Interval mismatch: expected {expected_interval_minutes}m, "
            f"detected {detected:.2f}m (diff={diff_pct:.1%})"
        )

    # Check gap ratio
    if gap_info['has_gaps']:
        total_expected_minutes = gap_info['rows_checked'] * expected_interval_minutes
        gap_ratio = gap_info['total_gap_minutes'] / total_expected_minutes
        result['consistency_ratio'] = 1.0 - gap_ratio

        if gap_ratio > 0.1:  # More than 10% gaps
            result['warnings'].append(
                f"High gap ratio: {gap_ratio:.1%} of data is gaps"
            )
    else:
        result['consistency_ratio'] = 1.0

    return result
