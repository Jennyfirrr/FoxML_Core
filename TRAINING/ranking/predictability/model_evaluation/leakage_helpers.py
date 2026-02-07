# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Leakage Detection Helper Functions

Helper functions for detecting and analyzing data leakage.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set, Tuple, Any, List

from TRAINING.common.utils.task_types import TaskType

logger = logging.getLogger(__name__)

# Try to import config loader
_CONFIG_AVAILABLE = False
try:
    from config_loader import get_cfg, get_safety_config
    _CONFIG_AVAILABLE = True
except ImportError:
    pass


def compute_suspicion_score(
    train_score: float,
    cv_score: Optional[float],
    feature_importances: Dict[str, float],
    task_type: str = 'classification'
) -> float:
    """
    Compute suspicion score for perfect train accuracy.
    
    Higher score = more suspicious (likely real leakage, not just overfitting).
    
    Signals that increase suspicion:
    - CV too good to be true (cv_mean >= 0.85)
    - Generalization gap too small with perfect train (gap < 0.05)
    - Single feature domination (top1_importance / sum >= 0.40)
    
    Signals that decrease suspicion:
    - CV is normal-ish (0.55-0.75)
    - Large gap (classic overfit)
    - Feature dominance not extreme
    """
    suspicion = 0.0
    
    # Signal 1: CV too good to be true
    if cv_score is not None:
        if cv_score >= 0.85:
            suspicion += 0.4  # High suspicion
        elif cv_score >= 0.75:
            suspicion += 0.2  # Medium suspicion
        elif cv_score < 0.55:
            suspicion -= 0.2  # Low suspicion (normal performance)
    
    # Signal 2: Generalization gap (small gap with perfect train = suspicious)
    if cv_score is not None:
        gap = train_score - cv_score
        if gap < 0.05 and train_score >= 0.99:
            suspicion += 0.3  # Very suspicious: perfect train but CV also high
        elif gap > 0.20:
            suspicion -= 0.2  # Large gap = classic overfit (less suspicious)
    
    # Signal 3: Feature dominance
    if feature_importances:
        importances = list(feature_importances.values())
        if importances:
            total_importance = sum(importances)
            if total_importance > 0:
                top1_importance = max(importances)
                dominance_ratio = top1_importance / total_importance
                if dominance_ratio >= 0.50:
                    suspicion += 0.3  # Single feature dominates
                elif dominance_ratio >= 0.40:
                    suspicion += 0.2  # High dominance
                elif dominance_ratio < 0.20:
                    suspicion -= 0.1  # Low dominance (less suspicious)
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, suspicion))


@dataclass
class LeakageArtifacts:
    """Artifacts from model training needed for leakage detection."""
    model_metrics: Dict[str, Dict[str, float]]
    primary_scores: Dict[str, float]  # CV scores (same as model_scores from train_and_evaluate_models)
    feature_importances: Dict[str, Dict[str, float]]
    perfect_correlation_models: Set[str]
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    safe_columns: Optional[List[str]] = None  # For scan scope logging


def detect_and_fix_leakage(
    target_ctx: Any,  # TargetContext from leakage_auto_fixer (or None)
    artifacts: LeakageArtifacts,
    io_context: Dict[str, Any],  # output_dir, experiment_config, run_identity, target_column, symbols_array, task_type, detected_interval
    autofix_enabled: bool = True  # Phase 2: report-only mode (False) vs autofix (True)
) -> Tuple[bool, Optional[Any], bool]:
    """
    Detect leakage and run auto-fixer if needed.
    
    Args:
        target_ctx: TargetContext (target, interval, task_type, symbols_array)
        artifacts: LeakageArtifacts (model outputs, data, features)
        io_context: Dict with output_dir, experiment_config, run_identity
        autofix_enabled: If False, only detect and report (no fixes applied)
    
    Returns:
        (should_rerun, autofix_info, detector_failed)
        - should_rerun: True if configs were modified and rerun is needed
        - autofix_info: AutoFixInfo if fixes were applied, None otherwise
        - detector_failed: True if detection failed (non-fatal, continue execution)
    """
    detector_failed = False
    autofix_info = None
    should_rerun = False
    
    try:
        # Load thresholds from config (with sensible defaults)
        if _CONFIG_AVAILABLE:
            try:
                safety_cfg = get_safety_config()
                # safety_config.yaml has a top-level 'safety' key
                safety_section = safety_cfg.get('safety', {})
                leakage_cfg = safety_section.get('leakage_detection', {})
                auto_fix_cfg = leakage_cfg.get('auto_fix_thresholds', {})
                cv_threshold = float(auto_fix_cfg.get('cv_score', 0.99))
                accuracy_threshold = float(auto_fix_cfg.get('training_accuracy', 0.999))
                r2_threshold = float(auto_fix_cfg.get('training_r2', 0.999))
                correlation_threshold = float(auto_fix_cfg.get('perfect_correlation', 0.999))
                auto_fix_enabled_config = leakage_cfg.get('auto_fix_enabled', True)
                auto_fix_min_confidence = float(leakage_cfg.get('auto_fix_min_confidence', 0.8))
                auto_fix_max_features = int(leakage_cfg.get('auto_fix_max_features_per_run', 20))
            except Exception as e:
                logger.debug(f"Failed to load leakage detection config: {e}, using defaults")
                cv_threshold = 0.99
                accuracy_threshold = 0.999
                r2_threshold = 0.999
                correlation_threshold = 0.999
                auto_fix_enabled_config = True
                auto_fix_min_confidence = 0.8
                auto_fix_max_features = 20
        else:
            cv_threshold = 0.99
            accuracy_threshold = 0.999
            r2_threshold = 0.999
            correlation_threshold = 0.999
            auto_fix_enabled_config = True
            auto_fix_min_confidence = 0.8
            auto_fix_max_features = 20
        
        # Check if auto-fixer is enabled (both config and parameter)
        if not auto_fix_enabled_config or not autofix_enabled:
            logger.debug("Auto-fixer is disabled in config or parameter")
            should_auto_fix = False
        else:
            should_auto_fix = False
            
            # Check 1: Perfect CV scores (cross-validation)
            # CRITICAL: Use actual CV scores from primary_scores, not model_metrics
            from TRAINING.common.utils.determinism_ordering import sorted_items
            max_cv_score = None
            if artifacts.primary_scores:
                valid_cv_scores = [s for _k, s in sorted_items(artifacts.primary_scores) if s is not None and not np.isnan(s)]
                if valid_cv_scores:
                    max_cv_score = max(valid_cv_scores)

            # Fallback: try to extract from model_metrics if primary_scores unavailable
            if max_cv_score is None and artifacts.model_metrics:
                for model_name, metrics in sorted_items(artifacts.model_metrics):
                    if isinstance(metrics, dict):
                        cv_score_val = metrics.get('roc_auc') or metrics.get('r2') or metrics.get('accuracy')
                        if cv_score_val is not None and not np.isnan(cv_score_val):
                            # Skip if this looks like a training score
                            if 'training_accuracy' in metrics and abs(cv_score_val - metrics['training_accuracy']) < 0.001:
                                continue
                            if max_cv_score is None or cv_score_val > max_cv_score:
                                max_cv_score = cv_score_val
            
            if max_cv_score is not None and max_cv_score >= cv_threshold:
                should_auto_fix = True
                logger.warning(f"🚨 Perfect CV scores detected (max_cv={max_cv_score:.4f} >= {cv_threshold:.1%}) - enabling auto-fix mode")
            
            # Check 2: Perfect in-sample training accuracy with suspicion score gating
            if not should_auto_fix and artifacts.model_metrics:
                logger.debug(f"Checking model_metrics for perfect scores: {sorted(artifacts.model_metrics.keys())}")

                for model_name, metrics in sorted_items(artifacts.model_metrics):
                    if isinstance(metrics, dict):
                        logger.debug(f"  {model_name} metrics: {list(metrics.keys())}")
                        
                        train_acc = metrics.get('training_accuracy')
                        cv_acc = metrics.get('accuracy')
                        train_r2 = metrics.get('training_r2')
                        cv_r2 = metrics.get('r2')
                        
                        # Check classification
                        if train_acc is not None and train_acc >= accuracy_threshold:
                            logger.debug(f"    {model_name} training_accuracy: {train_acc:.4f}")
                            
                            suspicion = compute_suspicion_score(
                                train_score=train_acc,
                                cv_score=cv_acc,
                                feature_importances=artifacts.feature_importances.get(model_name, {}) if artifacts.feature_importances else {},
                                task_type='classification'
                            )
                            
                            suspicion_threshold = 0.5
                            if suspicion >= suspicion_threshold:
                                should_auto_fix = True
                                cv_acc_str = f"{cv_acc:.3f}" if cv_acc is not None else "N/A"
                                logger.warning(f"🚨 Suspicious perfect training accuracy in {model_name} "
                                            f"(train={train_acc:.1%}, cv={cv_acc_str}, "
                                            f"suspicion={suspicion:.2f}) - enabling auto-fix mode")
                                break
                            else:
                                cv_acc_str = f"{cv_acc:.3f}" if cv_acc is not None else "N/A"
                                logger.info(f"⚠️  {model_name} memorized training data (train={train_acc:.1%}, "
                                         f"cv={cv_acc_str}, suspicion={suspicion:.2f}). "
                                         f"Ignoring; check CV metrics.")
                        
                        elif cv_acc is not None and cv_acc >= accuracy_threshold:
                            should_auto_fix = True
                            logger.warning(f"🚨 Perfect CV accuracy detected in {model_name} "
                                        f"({cv_acc:.1%} >= {accuracy_threshold:.1%}) - enabling auto-fix mode")
                            break
                        
                        # Check regression
                        if train_r2 is not None and train_r2 >= r2_threshold:
                            logger.debug(f"    {model_name} training_r2 (correlation): {train_r2:.4f}")
                            
                            suspicion = compute_suspicion_score(
                                train_score=train_r2,
                                cv_score=cv_r2,
                                feature_importances=artifacts.feature_importances.get(model_name, {}) if artifacts.feature_importances else {},
                                task_type='regression'
                            )
                            
                            suspicion_threshold = 0.5
                            if suspicion >= suspicion_threshold:
                                should_auto_fix = True
                                cv_r2_str = f"{cv_r2:.4f}" if cv_r2 is not None else "N/A"
                                logger.warning(f"🚨 Suspicious perfect training correlation in {model_name} "
                                            f"(train={train_r2:.4f}, cv={cv_r2_str}, "
                                            f"suspicion={suspicion:.2f}) - enabling auto-fix mode")
                                break
                            else:
                                cv_r2_str = f"{cv_r2:.4f}" if cv_r2 is not None else "N/A"
                                logger.info(f"⚠️  {model_name} memorized training data (train={train_r2:.4f}, "
                                         f"cv={cv_r2_str}, suspicion={suspicion:.2f}). "
                                         f"Ignoring; check CV metrics.")
                        
                        elif cv_r2 is not None and cv_r2 >= r2_threshold:
                            should_auto_fix = True
                            logger.warning(f"🚨 Perfect CV R² detected in {model_name} "
                                        f"({cv_r2:.4f} >= {r2_threshold:.4f}) - enabling auto-fix mode")
                            break
            
            # Check 3: Models that triggered perfect correlation warnings
            if not should_auto_fix and artifacts.perfect_correlation_models:
                should_auto_fix = True
                logger.warning(f"🚨 Perfect correlation detected in models: {', '.join(artifacts.perfect_correlation_models)} (>= {correlation_threshold:.1%}) - enabling auto-fix mode")
        
        if should_auto_fix:
            try:
                from TRAINING.common.leakage_auto_fixer import LeakageAutoFixer, TargetContext
                
                logger.info("🔧 Auto-fixing detected leaks...")
                logger.info(f"   Initializing LeakageAutoFixer (backups disabled)...")
                output_dir = io_context.get('output_dir')
                fixer = LeakageAutoFixer(backup_configs=False, output_dir=output_dir)
                
                # Create TargetContext if not provided (from SST - detected_interval is SST-driven bar_minutes)
                if target_ctx is None:
                    target_column = io_context.get('target_column')
                    detected_interval = io_context.get('detected_interval')
                    experiment_config = io_context.get('experiment_config')
                    if target_column and detected_interval:
                        target_ctx = TargetContext.from_target(
                            target=target_column,
                            bar_minutes=detected_interval,  # SST-driven (from resolved_config or auto-detection)
                            experiment_config=experiment_config
                        )
                        if target_ctx is None:
                            logger.warning(f"Could not create TargetContext for {target_column}, skipping per-target registry updates")
                
                # Get run_id for evidence tracking
                run_id = None
                run_identity = io_context.get('run_identity')
                if run_identity:
                    run_id = getattr(run_identity, 'run_id', None) or str(run_identity)
                
                # Convert X to DataFrame if needed
                if not isinstance(artifacts.X, pd.DataFrame):
                    X_df = pd.DataFrame(artifacts.X, columns=artifacts.feature_names)
                else:
                    X_df = artifacts.X
                
                # Convert y to Series if needed
                if not isinstance(artifacts.y, pd.Series):
                    y_series = pd.Series(artifacts.y)
                else:
                    y_series = artifacts.y
                
                # Aggregate feature importances across all models (deterministic order)
                aggregated_importance = {}
                if artifacts.feature_importances:
                    for model_name in sorted(artifacts.feature_importances.keys()):
                        importances = artifacts.feature_importances[model_name]
                        if isinstance(importances, dict):
                            for feat, imp in importances.items():
                                if feat not in aggregated_importance:
                                    aggregated_importance[feat] = []
                                aggregated_importance[feat].append(imp)
                
                # Average importance across models (sort features for deterministic order)
                avg_importance = {feat: np.mean(imps) for feat, imps in sorted(aggregated_importance.items())} if aggregated_importance else {}
                
                # Get actual training accuracy from model_metrics
                actual_train_score = None
                if artifacts.model_metrics:
                    for model_name, metrics in artifacts.model_metrics.items():
                        if isinstance(metrics, dict):
                            if 'training_accuracy' in metrics and metrics['training_accuracy'] >= accuracy_threshold:
                                actual_train_score = metrics['training_accuracy']
                                logger.debug(f"Using training accuracy {actual_train_score:.4f} from {model_name} for auto-fixer")
                                break
                            elif 'accuracy' in metrics and metrics['accuracy'] >= accuracy_threshold:
                                actual_train_score = metrics['accuracy']
                                logger.debug(f"Using CV accuracy {actual_train_score:.4f} from {model_name} for auto-fixer")
                                break
                            elif 'training_r2' in metrics and metrics['training_r2'] >= r2_threshold:
                                actual_train_score = metrics['training_r2']
                                logger.debug(f"Using training correlation {actual_train_score:.4f} from {model_name} for auto-fixer")
                                break
                            elif 'r2' in metrics and metrics['r2'] >= r2_threshold:
                                actual_train_score = metrics['r2']
                                logger.debug(f"Using CV R² {actual_train_score:.4f} from {model_name} for auto-fixer")
                                break
                
                # Fallback to CV score if no perfect training score found
                if actual_train_score is None:
                    if max_cv_score is not None:
                        actual_train_score = max_cv_score
                        logger.debug(f"Using CV score {actual_train_score:.4f} as fallback for auto-fixer (from model_metrics)")
                    else:
                        actual_train_score = max(artifacts.primary_scores.values()) if artifacts.primary_scores else None
                        logger.debug(f"Using CV score {actual_train_score:.4f} as fallback for auto-fixer (from primary_scores)")
                
                # Log what we're passing to auto-fixer
                train_feature_set_size = len(artifacts.feature_names)
                scan_feature_set_size = len(artifacts.safe_columns) if artifacts.safe_columns else len(artifacts.feature_names)
                scan_scope = "full_safe" if scan_feature_set_size > train_feature_set_size else "trained_only"
                
                train_score_str = f"{actual_train_score:.4f}" if actual_train_score is not None else "None"
                logger.info(f"🔧 Auto-fixer inputs: train_score={train_score_str}, "
                           f"train_feature_set_size={train_feature_set_size}, "
                           f"scan_feature_set_size={scan_feature_set_size}, "
                           f"scan_scope={scan_scope}, "
                           f"model_importance keys={len(avg_importance)}")
                if avg_importance:
                    top_5 = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                    logger.debug(f"   Top 5 features by importance: {', '.join([f'{f}={imp:.4f}' for f, imp in top_5])}")
                else:
                    logger.warning(f"   ⚠️  No aggregated importance available! feature_importances keys: {list(artifacts.feature_importances.keys()) if artifacts.feature_importances else 'None'}")
                
                # Get task type from io_context (TargetContext doesn't have task_type attribute)
                task_type = io_context.get('task_type')
                # Convert TaskType enum to string (handle None case)
                if task_type is None:
                    task_type_str = 'regression'  # Default fallback
                    logger.warning("⚠️  task_type is None in io_context, defaulting to 'regression'")
                elif isinstance(task_type, TaskType):
                    task_type_str = 'classification' if (task_type == TaskType.BINARY_CLASSIFICATION or 
                                                         task_type == TaskType.MULTICLASS_CLASSIFICATION) else 'regression'
                else:
                    # Already a string (shouldn't happen, but handle gracefully)
                    task_type_str = str(task_type) if task_type else 'regression'
                
                # Get symbols_array from io_context (TargetContext doesn't have symbols_array attribute)
                symbols_array = io_context.get('symbols_array')
                
                # Get target_column from io_context (fallback to target_ctx.target if available)
                target_column = io_context.get('target_column')
                if target_column is None and target_ctx:
                    target_column = target_ctx.target  # TargetContext has 'target' attribute
                
                # Get detected_interval from io_context (fallback to target_ctx.bar_minutes if available)
                detected_interval = io_context.get('detected_interval')
                if detected_interval is None and target_ctx:
                    detected_interval = target_ctx.bar_minutes  # TargetContext has 'bar_minutes' attribute
                
                # Validate required parameters before calling auto-fixer
                if not target_column:
                    logger.warning("⚠️  Cannot run auto-fixer: target_column is None. Skipping auto-fix.")
                    detector_failed = True
                elif detected_interval is None or detected_interval <= 0:
                    logger.warning(f"⚠️  Cannot run auto-fixer: detected_interval is invalid ({detected_interval}). Skipping auto-fix.")
                    detector_failed = True
                else:
                    # All required parameters available, proceed with detection
                    try:
                        detections = fixer.detect_leaking_features(
                            X=X_df, y=y_series, feature_names=artifacts.feature_names,
                            target_column=target_column,
                            symbols=pd.Series(symbols_array) if symbols_array is not None else None,
                            task_type=task_type_str,
                            data_interval_minutes=detected_interval,
                            model_importance=avg_importance if avg_importance else None,
                            train_score=actual_train_score,
                            test_score=None  # CV scores are already validation scores
                        )
                        
                        if detections:
                            logger.warning(f"🔧 Auto-detected {len(detections)} leaking features")
                            
                            # Log detailed detection breakdown
                            for i, detection in enumerate(sorted(detections, key=lambda d: d.confidence, reverse=True)[:10], 1):
                                logger.info(
                                    f"   Detection {i}: {detection.feature_name} "
                                    f"(confidence={detection.confidence:.3f}, reason={detection.reason}, source={detection.source})"
                                )
                            if len(detections) > 10:
                                logger.info(f"   ... and {len(detections) - 10} more detections")
                            
                            # Apply fixes (with high confidence threshold to avoid false positives)
                            updates, autofix_info = fixer.apply_fixes(
                                detections, 
                                min_confidence=auto_fix_min_confidence, 
                                max_features=auto_fix_max_features,
                                dry_run=False,
                                target=target_column,
                                target_ctx=target_ctx,  # May be None if creation failed
                                run_id=run_id
                            )
                            if autofix_info.modified_configs:
                                logger.info(f"✅ Auto-fixed leaks. Configs updated.")
                                
                                # Log detailed exclusion changes
                                excluded_updates = updates.get('excluded_features_updates', {})
                                exact_patterns = excluded_updates.get('exact_patterns', [])
                                prefix_patterns = excluded_updates.get('prefix_patterns', [])
                                registry_rejects = updates.get('feature_registry_updates', {}).get('rejected_features', [])
                                
                                logger.info(f"   📝 EXCLUSIONS ADDED:")
                                logger.info(f"      Exact patterns: {len(exact_patterns)}")
                                if exact_patterns:
                                    logger.info(f"         {', '.join(exact_patterns[:10])}{'...' if len(exact_patterns) > 10 else ''}")
                                logger.info(f"      Prefix patterns: {len(prefix_patterns)}")
                                if prefix_patterns:
                                    logger.info(f"         {', '.join(prefix_patterns[:10])}{'...' if len(prefix_patterns) > 10 else ''}")
                                logger.info(f"      Registry rejections: {len(registry_rejects)}")
                                if registry_rejects:
                                    logger.info(f"         {', '.join(registry_rejects[:10])}{'...' if len(registry_rejects) > 10 else ''}")
                                
                                should_rerun = True  # Only rerun if configs were modified
                            else:
                                logger.warning("⚠️  Auto-fix detected leaks but no configs were modified")
                                logger.warning("   This usually means all detections were below confidence threshold")
                                logger.warning(f"   Check logs above for confidence distribution details")
                            # Log backup info if available
                            if autofix_info.backup_files:
                                logger.info(f"📦 Backup created: {len(autofix_info.backup_files)} backup file(s)")
                        else:
                            logger.info("🔍 Auto-fix detected no leaks (may need manual review)")
                    except Exception as e:
                        logger.error(f"❌ Auto-fixer failed during detection/fix: {e}", exc_info=True)
                        detector_failed = True
            except Exception as e:
                logger.error(f"❌ Auto-fixer initialization/setup failed: {e}", exc_info=True)
                detector_failed = True
        
    except Exception as e:
        logger.warning(f"Leakage detection failed: {e}", exc_info=True)
        detector_failed = True
    
    return should_rerun, autofix_info, detector_failed


def detect_leakage(
    auc: float,
    composite_score: float,
    mean_importance: float,
    target: str = "",
    model_scores: Dict[str, float] = None,
    task_type: TaskType = TaskType.REGRESSION
) -> str:
    """
    Detect potential data leakage based on suspicious patterns.
    
    Returns:
        "OK" - No signs of leakage
        "HIGH_R2" - R² > threshold (suspiciously high)
        "INCONSISTENT" - Composite score too high for R² (possible leakage)
        "SUSPICIOUS" - Multiple warning signs
    """
    flags = []
    
    # Load thresholds from config
    if _CONFIG_AVAILABLE:
        try:
            safety_cfg = get_safety_config()
            # safety_config.yaml has a top-level 'safety' key
            safety_section = safety_cfg.get('safety', {})
            leakage_cfg = safety_section.get('leakage_detection', {})
            warning_cfg = leakage_cfg.get('warning_thresholds', {})
        except Exception:
            warning_cfg = {}
    else:
        warning_cfg = {}
    
    # Determine threshold based on task type and target name
    if task_type == TaskType.REGRESSION:
        is_forward_return = target.startswith('fwd_ret_')
        if is_forward_return:
            # For forward returns: R² > 0.50 is suspicious
            reg_cfg = warning_cfg.get('regression', {}).get('forward_return', {})
            high_threshold = float(reg_cfg.get('high', 0.50))
            very_high_threshold = float(reg_cfg.get('very_high', 0.60))
            metric_name = "R²"
        else:
            # For barrier targets: R² > 0.70 is suspicious
            reg_cfg = warning_cfg.get('regression', {}).get('barrier', {})
            high_threshold = float(reg_cfg.get('high', 0.70))
            very_high_threshold = float(reg_cfg.get('very_high', 0.80))
            metric_name = "R²"
    elif task_type == TaskType.BINARY_CLASSIFICATION:
        # ROC-AUC > 0.95 is suspicious (near-perfect classification)
        class_cfg = warning_cfg.get('classification', {})
        high_threshold = float(class_cfg.get('high', 0.90))
        very_high_threshold = float(class_cfg.get('very_high', 0.95))
        metric_name = "ROC-AUC"
    else:  # MULTICLASS_CLASSIFICATION
        # Accuracy > 0.95 is suspicious
        class_cfg = warning_cfg.get('classification', {})
        high_threshold = float(class_cfg.get('high', 0.90))
        very_high_threshold = float(class_cfg.get('very_high', 0.95))
        metric_name = "Accuracy"
    
    # Check 1: Suspiciously high mean score
    if auc > very_high_threshold:
        flags.append("HIGH_SCORE")
        logger.warning(
            f"LEAKAGE WARNING: {metric_name}={auc:.3f} > {very_high_threshold:.2f} "
            f"(extremely high - likely leakage)"
        )
    elif auc > high_threshold:
        flags.append("HIGH_SCORE")
        logger.warning(
            f"LEAKAGE WARNING: {metric_name}={auc:.3f} > {high_threshold:.2f} "
            f"(suspiciously high - investigate)"
        )
    
    # Check 2: Individual model scores too high (even if mean is lower)
    if model_scores:
        high_model_count = sum(1 for score in model_scores.values() 
                              if not np.isnan(score) and score > high_threshold)
        if high_model_count >= 3:  # 3+ models with high scores
            flags.append("HIGH_SCORE")
            logger.warning(
                f"LEAKAGE WARNING: {high_model_count} models have {metric_name} > {high_threshold:.2f} "
                f"(models: {[k for k, v in model_scores.items() if not np.isnan(v) and v > high_threshold]})"
            )
    
    # Check 3: Composite score inconsistent with mean score
    # Load thresholds from config
    try:
        from CONFIG.config_loader import get_cfg
        composite_high_threshold = float(get_cfg("safety.leakage_detection.model_evaluation.composite_score_high_threshold", default=0.5, config_name="safety_config"))
        regression_score_low = float(get_cfg("safety.leakage_detection.model_evaluation.regression_score_low_threshold", default=0.2, config_name="safety_config"))
        classification_score_low = float(get_cfg("safety.leakage_detection.model_evaluation.classification_score_low_threshold", default=0.6, config_name="safety_config"))
    except Exception:
        composite_high_threshold = 0.5
        regression_score_low = 0.2
        classification_score_low = 0.6
    
    score_low_threshold = regression_score_low if task_type == TaskType.REGRESSION else classification_score_low
    if composite_score > composite_high_threshold and auc < score_low_threshold:
        flags.append("INCONSISTENT")
        # Reserve "LEAKAGE" for actual leak_scan results - use "METRIC INCONSISTENCY" for heuristic checks
        logger.warning(
            f"METRIC INCONSISTENCY: Composite={composite_score:.3f} but {metric_name}={auc:.3f} "
            f"(high composite with low {metric_name} - may indicate data quality issues or feature importance inflation). "
            f"Thresholds: composite > {composite_high_threshold}, {metric_name} < {score_low_threshold}. "
            f"Note: This is a heuristic check; actual leak detection is performed by leak_scan."
        )
    
    # Check 4: Very high importance with low score (might indicate leaked features)
    # Load thresholds from config
    try:
        from CONFIG.config_loader import get_cfg
        importance_high_threshold = float(get_cfg("safety.leakage_detection.model_evaluation.importance_high_threshold", default=0.7, config_name="safety_config"))
        regression_score_very_low = float(get_cfg("safety.leakage_detection.model_evaluation.regression_score_very_low_threshold", default=0.1, config_name="safety_config"))
        classification_score_very_low = float(get_cfg("safety.leakage_detection.model_evaluation.classification_score_very_low_threshold", default=0.5, config_name="safety_config"))
    except Exception:
        importance_high_threshold = 0.7
        regression_score_very_low = 0.1
        classification_score_very_low = 0.5
    
    score_very_low_threshold = regression_score_very_low if task_type == TaskType.REGRESSION else classification_score_very_low
    if mean_importance > importance_high_threshold and auc < score_very_low_threshold:
        flags.append("INCONSISTENT")
        logger.warning(
            f"LEAKAGE WARNING: Importance={mean_importance:.2f} but {metric_name}={auc:.3f} "
            f"(high importance with low {metric_name} - check for leaked features)"
        )
    
    if len(flags) > 1:
        return "SUSPICIOUS"
    elif len(flags) == 1:
        return flags[0]
    else:
        return "OK"

