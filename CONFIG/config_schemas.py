# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Typed Configuration Schemas

Defines dataclasses for all pipeline configuration types to ensure
type safety and prevent config "crossing" between modules.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional


# =============================================================================
# Interval Parsing Helper (Phase 24)
# =============================================================================

def _parse_interval_to_minutes(interval_str: str) -> Optional[float]:
    """
    Parse interval string to minutes for validation.

    Supports formats:
    - "5m", "15m" -> minutes
    - "1h", "2h" -> hours to minutes
    - "300s" -> seconds to minutes
    - "5" (integer) -> assumed minutes

    Args:
        interval_str: Interval string

    Returns:
        Minutes as float, or None if parsing fails
    """
    if interval_str is None:
        return None

    try:
        interval_str = str(interval_str).lower().strip()

        if interval_str.endswith('m'):
            return float(interval_str[:-1])
        elif interval_str.endswith('h'):
            return float(interval_str[:-1]) * 60
        elif interval_str.endswith('s'):
            return float(interval_str[:-1]) / 60
        else:
            # Assume integer is minutes
            return float(interval_str)
    except (ValueError, TypeError):
        return None


@dataclass
class DataConfig:
    """Data loading configuration"""
    timestamp_column: str = "ts"
    bar_interval: Optional[str] = "5m"  # Normalized interval string (e.g., "5m", "15m", "1h")
    base_interval_minutes: Optional[float] = None  # NEW: Explicit base interval for training grid (overrides bar_interval if set)
    base_interval_source: str = "auto"  # NEW: "config" (use base_interval_minutes) or "auto" (infer from timestamps)
    asof_strategy: str = "backward"  # NEW: As-of join strategy (only "backward" supported for now)
    default_embargo_minutes: float = 0.0  # NEW: Default embargo for features without explicit metadata
    default_max_staleness_minutes: Optional[float] = None  # NEW: Optional staleness cap (e.g., 1440 for 1-day)
    max_samples_per_symbol: int = 50000
    validation_split: float = 0.2
    seed: int = 42
    symbol_batch_size: Optional[int] = None  # Limit auto-discovered symbols to N (random sample)

    def __post_init__(self):
        """Validate and normalize bar_interval and symbol_batch_size"""
        # Validate symbol_batch_size
        if self.symbol_batch_size is not None and self.symbol_batch_size < 1:
            raise ValueError(
                f"DataConfig.symbol_batch_size must be >= 1 if provided, got {self.symbol_batch_size}"
            )

        if self.bar_interval is not None:
            # Validate format using regex (avoid circular import)
            import re
            interval_str = str(self.bar_interval).lower().strip()

            # Check format: "5m", "15m", "1h", "300s", or integer
            valid_patterns = [
                r'^\d+[mh]$',  # "5m", "15m", "1h"
                r'^\d+s$',     # "300s"
                r'^\d+$'       # integer
            ]

            if not any(re.match(pattern, interval_str) for pattern in valid_patterns):
                raise ValueError(
                    f"Invalid bar_interval format '{self.bar_interval}'. "
                    f"Expected: '5m', '15m', '1h', '300s', or integer"
                )

        # Phase 24: Validate bar_interval and base_interval_minutes consistency
        if self.bar_interval is not None and self.base_interval_minutes is not None:
            parsed_minutes = _parse_interval_to_minutes(self.bar_interval)
            if parsed_minutes is not None and abs(parsed_minutes - self.base_interval_minutes) > 0.001:
                raise ValueError(
                    f"Interval conflict: bar_interval='{self.bar_interval}' ({parsed_minutes}m) "
                    f"!= base_interval_minutes={self.base_interval_minutes}m. "
                    f"Use only one method to specify interval, or ensure they match."
                )


@dataclass
class ExperimentConfig:
    """Experiment-level configuration (what are we running?)"""
    name: str
    data_dir: Path
    symbols: List[str]
    target: str = ""  # Optional: can be empty if auto_targets=true
    data: DataConfig = field(default_factory=lambda: DataConfig())  # Data configuration
    max_samples_per_symbol: int = 5000
    description: Optional[str] = None
    
    # Optional overrides for specific modules
    feature_selection_overrides: Dict[str, Any] = field(default_factory=dict)
    target_ranking_overrides: Dict[str, Any] = field(default_factory=dict)
    training_overrides: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Convert string paths to Path objects and validate"""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        
        # Validation
        if not self.name:
            raise ValueError("ExperimentConfig.name cannot be empty")
        # symbols can be empty - will be resolved via auto-discovery in IntelligentTrainer.__init__()
        # target is optional when auto_targets=true (will be discovered)
        # Only validate if target is explicitly set (non-empty string)
        # Empty string is allowed and will be overridden by auto_targets
        if self.max_samples_per_symbol < 1:
            raise ValueError(f"ExperimentConfig.max_samples_per_symbol must be >= 1, got {self.max_samples_per_symbol}")
    
    @property
    def bar_interval(self) -> Optional[str]:
        """Convenience property to access data.bar_interval"""
        return self.data.bar_interval if self.data else None


@dataclass
class FeatureSelectionConfig:
    """Feature selection module configuration"""
    top_n: int
    model_families: Dict[str, Dict[str, Any]]
    aggregation: Dict[str, Any]
    sampling: Dict[str, Any] = field(default_factory=dict)
    shap: Dict[str, Any] = field(default_factory=dict)
    permutation: Dict[str, Any] = field(default_factory=dict)
    cross_validation: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)
    compute: Dict[str, Any] = field(default_factory=dict)
    
    # Target info (from experiment config)
    target: Optional[str] = None
    data_dir: Optional[Path] = None
    symbols: Optional[List[str]] = None
    max_samples_per_symbol: Optional[int] = None
    
    def __post_init__(self):
        """Validate feature selection config"""
        if self.top_n < 1:
            raise ValueError(f"FeatureSelectionConfig.top_n must be >= 1, got {self.top_n}")
        if not isinstance(self.model_families, dict):
            raise ValueError(f"FeatureSelectionConfig.model_families must be a dict, got {type(self.model_families)}")


@dataclass
class TargetRankingConfig:
    """Target ranking module configuration"""
    model_families: Dict[str, Dict[str, Any]]
    ranking: Dict[str, Any] = field(default_factory=dict)
    sampling: Dict[str, Any] = field(default_factory=dict)
    cross_validation: Dict[str, Any] = field(default_factory=dict)
    
    # Target discovery settings
    min_samples: int = 100
    min_class_samples: int = 10
    
    # Data info (from experiment config)
    data_dir: Optional[Path] = None
    symbols: Optional[List[str]] = None
    max_samples_per_symbol: Optional[int] = None
    
    def __post_init__(self):
        """Validate target ranking config"""
        if not isinstance(self.model_families, dict):
            raise ValueError(f"TargetRankingConfig.model_families must be a dict, got {type(self.model_families)}")
        if self.min_samples < 1:
            raise ValueError(f"TargetRankingConfig.min_samples must be >= 1, got {self.min_samples}")
        if self.min_class_samples < 1:
            raise ValueError(f"TargetRankingConfig.min_class_samples must be >= 1, got {self.min_class_samples}")


@dataclass
class TrainingConfig:
    """Training module configuration"""
    model_families: Dict[str, Dict[str, Any]]
    folds: int = 5
    pipeline: Dict[str, Any] = field(default_factory=dict)
    gpu: Dict[str, Any] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    threading: Dict[str, Any] = field(default_factory=dict)
    callbacks: Dict[str, Any] = field(default_factory=dict)
    optimizer: Dict[str, Any] = field(default_factory=dict)
    
    # Data info (from experiment config)
    target: Optional[str] = None
    data_dir: Optional[Path] = None
    symbols: Optional[List[str]] = None
    max_samples_per_symbol: Optional[int] = None
    
    def __post_init__(self):
        """Validate training config"""
        if not isinstance(self.model_families, dict):
            raise ValueError(f"TrainingConfig.model_families must be a dict, got {type(self.model_families)}")
        if self.folds < 2:
            raise ValueError(f"TrainingConfig.folds must be >= 2, got {self.folds}")


@dataclass
class LeakageConfig:
    """Leakage detection and auto-fix configuration"""
    safety: Dict[str, Any] = field(default_factory=dict)
    auto_fix: Dict[str, Any] = field(default_factory=dict)
    auto_rerun: Dict[str, Any] = field(default_factory=dict)
    pre_scan: Dict[str, Any] = field(default_factory=dict)
    warning_thresholds: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModuleLoggingConfig:
    """Per-module logging configuration"""
    level: str = "INFO"  # DEBUG / INFO / WARNING / ERROR
    gpu_detail: bool = False  # GPU confirmations, device info
    cv_detail: bool = False  # Cross-validation fold details, timestamps
    edu_hints: bool = False  # Educational hints (💡 messages)
    detail: bool = False  # General detailed logging


@dataclass
class BackendLoggingConfig:
    """Backend library logging configuration"""
    native_verbosity: int = -1  # -1=silent, 0=info, >0=more verbose
    show_sparse_warnings: bool = True  # Show sparse matrix warnings


@dataclass
class LoggingConfig:
    """Structured logging configuration"""
    global_level: str = "INFO"  # DEBUG / INFO / WARNING / ERROR
    modules: Dict[str, ModuleLoggingConfig] = field(default_factory=dict)
    backends: Dict[str, BackendLoggingConfig] = field(default_factory=dict)
    profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def get_module_config(self, module_name: str) -> ModuleLoggingConfig:
        """Get module config, with defaults if not specified"""
        if module_name in self.modules:
            return self.modules[module_name]
        return ModuleLoggingConfig()
    
    def get_backend_config(self, backend_name: str) -> BackendLoggingConfig:
        """Get backend config, with defaults if not specified"""
        if backend_name in self.backends:
            return self.backends[backend_name]
        return BackendLoggingConfig()


@dataclass
class SystemConfig:
    """System-level configuration (paths, logging, etc.)"""
    paths: Dict[str, Any] = field(default_factory=dict)
    logging: Dict[str, Any] = field(default_factory=dict)
    defaults: Dict[str, Any] = field(default_factory=dict)
    backup: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Config Validation Functions (for catching silent failures)
# ============================================================================

def validate_safety_config(cfg: Dict[str, Any], strict: bool = True) -> None:
    """
    Validate safety_config structure.
    
    This prevents silent failures where config values fall back to defaults
    due to incorrect key paths (e.g., missing 'safety.' prefix).
    
    Args:
        cfg: Safety config dictionary (from get_safety_config())
        strict: If True, raise ValueError on validation failure.
                If False, log warning and continue (graceful degradation).
    
    Raises:
        ValueError: If strict=True and validation fails
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if not isinstance(cfg, dict):
        msg = f"safety_config must be a dict, got {type(cfg)}"
        if strict:
            raise ValueError(msg)
        logger.warning(f"[CONFIG VALIDATION] {msg}")
        return
    
    safety_section = cfg.get('safety', {})
    if not isinstance(safety_section, dict):
        msg = f"safety_config.safety must be a dict, got {type(safety_section)}"
        if strict:
            raise ValueError(msg)
        logger.warning(f"[CONFIG VALIDATION] {msg}")
        return
    
    leakage_detection = safety_section.get('leakage_detection', {})
    if not isinstance(leakage_detection, dict):
        msg = f"safety_config.safety.leakage_detection must be a dict, got {type(leakage_detection)}"
        if strict:
            raise ValueError(msg)
        logger.warning(f"[CONFIG VALIDATION] {msg}")
        return
    
    # Validate critical keys exist
    required_keys = [
        'auto_fix_max_features_per_run',
        'auto_fix_min_confidence',
        'auto_fix_enabled'
    ]
    
    missing = [k for k in required_keys if k not in leakage_detection]
    if missing:
        msg = (
            f"safety_config.safety.leakage_detection missing required keys: {missing}. "
            f"Available keys: {list(leakage_detection.keys())}"
        )
        if strict:
            raise ValueError(msg)
        logger.warning(f"[CONFIG VALIDATION] {msg}")


def validate_interval_config(
    bar_interval: Optional[str] = None,
    base_interval_minutes: Optional[float] = None,
    detected_interval_minutes: Optional[float] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Phase 24: Validate interval configuration consistency.

    Checks for conflicts between different interval specifications:
    - bar_interval (string like "5m")
    - base_interval_minutes (explicit float)
    - detected_interval_minutes (auto-detected from data)

    Args:
        bar_interval: Interval string from config (e.g., "5m", "1h")
        base_interval_minutes: Explicit interval in minutes
        detected_interval_minutes: Interval detected from data timestamps
        strict: If True, raise ValueError on conflict. If False, return warnings.

    Returns:
        Dict with:
            - valid: bool
            - resolved_minutes: float (the resolved interval, or None if invalid)
            - source: str ("config_string", "config_minutes", "detected", "default")
            - warnings: List[str]

    Raises:
        ValueError: If strict=True and there's an unresolvable conflict
    """
    import logging
    logger = logging.getLogger(__name__)

    result = {
        'valid': True,
        'resolved_minutes': None,
        'source': 'default',
        'warnings': []
    }

    # Parse bar_interval to minutes
    parsed_bar_interval = None
    if bar_interval is not None:
        parsed_bar_interval = _parse_interval_to_minutes(bar_interval)
        if parsed_bar_interval is None:
            msg = f"Could not parse bar_interval='{bar_interval}' to minutes"
            result['warnings'].append(msg)
            if strict:
                raise ValueError(msg)

    # Check for conflicts between bar_interval and base_interval_minutes
    if parsed_bar_interval is not None and base_interval_minutes is not None:
        if abs(parsed_bar_interval - base_interval_minutes) > 0.001:
            msg = (
                f"Interval conflict: bar_interval='{bar_interval}' ({parsed_bar_interval}m) "
                f"!= base_interval_minutes={base_interval_minutes}m"
            )
            result['warnings'].append(msg)
            result['valid'] = False
            if strict:
                raise ValueError(msg)

    # Check for conflicts with detected interval
    config_interval = base_interval_minutes or parsed_bar_interval
    if config_interval is not None and detected_interval_minutes is not None:
        if abs(config_interval - detected_interval_minutes) > 0.5:  # Allow small tolerance
            msg = (
                f"Config interval ({config_interval}m) differs from detected interval "
                f"({detected_interval_minutes}m). Using detected value."
            )
            result['warnings'].append(msg)
            logger.warning(f"[INTERVAL VALIDATION] {msg}")

    # Resolve the final interval (precedence: detected > config_minutes > config_string > default)
    if detected_interval_minutes is not None:
        result['resolved_minutes'] = detected_interval_minutes
        result['source'] = 'detected'
    elif base_interval_minutes is not None:
        result['resolved_minutes'] = base_interval_minutes
        result['source'] = 'config_minutes'
    elif parsed_bar_interval is not None:
        result['resolved_minutes'] = parsed_bar_interval
        result['source'] = 'config_string'
    else:
        result['resolved_minutes'] = 5.0  # Default fallback
        result['source'] = 'default'
        result['warnings'].append("No interval specified, using default 5m")

    return result

