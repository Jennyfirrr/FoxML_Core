# MIT License

"""
Target Ranking Module

Extracted from SCRIPTS/rank_target_predictability.py to enable integration
into the training pipeline. All leakage-free behavior is preserved by
reusing the original functions.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import warnings

# Add project root to path for imports
# TRAINING/ranking/target_ranker.py -> parents[2] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import original functions to preserve leakage-free behavior
from TRAINING.ranking.rank_target_predictability import (
    TargetPredictabilityScore,
    evaluate_target_predictability as _evaluate_target_predictability,
    discover_all_targets as _discover_all_targets,
    load_target_configs as _load_target_configs,
    save_rankings as _save_rankings
)
from TRAINING.ranking.target_routing import (
    _compute_target_routing_decisions,
    _save_dual_view_rankings
)

# Import auto-rerun wrapper if available
# Note: evaluate_target_with_autofix may not accept view/symbol, so we'll handle that
try:
    from TRAINING.ranking.rank_target_predictability import evaluate_target_with_autofix
    _AUTOFIX_AVAILABLE = True
    # FIX: Check signature once at module load instead of catching TypeError each time
    import inspect
    _AUTOFIX_HAS_VIEW_PARAM = 'view' in inspect.signature(evaluate_target_with_autofix).parameters
except ImportError:
    # Fallback: use regular evaluation
    evaluate_target_with_autofix = None
    _AUTOFIX_AVAILABLE = False
    _AUTOFIX_HAS_VIEW_PARAM = False
from TRAINING.common.utils.task_types import TargetConfig, TaskType
from TRAINING.ranking.utils.leakage_filtering import reload_feature_configs

# SST: Import View enum for consistent view handling
from TRAINING.orchestration.utils.scope_resolution import View, Stage
# DETERMINISM_CRITICAL: Symbol aggregation order must be deterministic
from TRAINING.common.utils.determinism_ordering import sorted_items, sorted_keys, collect_and_sort_parallel_results

# Import parallel execution utilities
try:
    from TRAINING.common.parallel_exec import execute_parallel, execute_parallel_with_context, get_max_workers
    _PARALLEL_AVAILABLE = True
except ImportError:
    _PARALLEL_AVAILABLE = False
    logger.warning("Parallel execution utilities not available; will run sequentially")

# Try to import config loader
_CONFIG_AVAILABLE = False
try:
    from config_loader import get_safety_config, get_experiment_config_path, load_experiment_config
    _CONFIG_AVAILABLE = True
except ImportError:
    pass

# Import new config system (optional - for backward compatibility)
try:
    from CONFIG.config_builder import build_target_ranking_config, load_yaml  # CH-009: Add load_yaml
    from CONFIG.config_schemas import ExperimentConfig, TargetRankingConfig
    _NEW_CONFIG_AVAILABLE = True
except ImportError:
    _NEW_CONFIG_AVAILABLE = False
    # Logger not yet initialized, will be set up below
    pass

# Suppress expected warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')

logger = logging.getLogger(__name__)


def evaluate_target_predictability(
    target: str,
    target_config: Dict[str, Any] | TargetConfig,
    symbols: List[str],
    data_dir: Path,
    model_families: List[str],
    multi_model_config: Dict[str, Any] = None,
    output_dir: Path = None,
    min_cs: Optional[int] = None,  # Load from config if None
    max_cs_samples: Optional[int] = None,  # Load from config if None
    max_rows_per_symbol: Optional[int] = None,  # Load from config if None
    explicit_interval: Optional[Union[int, str]] = None,  # Explicit interval from config
    experiment_config: Optional[Any] = None,  # Optional ExperimentConfig (for data.bar_interval)
    view: Union[str, View] = View.CROSS_SECTIONAL,  # View enum or "CROSS_SECTIONAL", "SYMBOL_SPECIFIC", or "LOSO"
    symbol: Optional[str] = None,  # Required for SYMBOL_SPECIFIC and LOSO views
    scope_purpose: str = "ROUTING_EVAL",  # Default to ROUTING_EVAL for target ranking
    run_identity: Optional[Any] = None,  # NEW: RunIdentity SST object for authoritative signatures
    registry: Optional[Any] = None,  # NEW: Pass registry instance (reuse same)
    coverage_breakdowns_dict: Optional[Dict[str, Any]] = None,  # NEW: Optional dict to collect coverage breakdowns
) -> TargetPredictabilityScore:
    """
    Evaluate predictability of a single target across symbols.
    
    This is a wrapper around the original function to preserve all
    leakage-free behavior (PurgedTimeSeriesSplit, leakage filtering, etc.).
    
    Args:
        target: Display name of target
        target_config: TargetConfig object or dict with target config
        symbols: List of symbols to evaluate on
        data_dir: Directory containing symbol data
        model_families: List of model family names to use
        multi_model_config: Multi-model config dict
        output_dir: Optional output directory for results
        min_cs: Minimum cross-sectional size per timestamp (loads from config if None)
        max_cs_samples: Maximum samples per timestamp for cross-sectional sampling (loads from config if None)
        max_rows_per_symbol: Maximum rows to load per symbol (loads from config if None)
    
    Returns:
        TargetPredictabilityScore object with predictability metrics
    """
    # Load from config if not provided
    if min_cs is None:
        try:
            from CONFIG.config_loader import get_cfg
            min_cs = int(get_cfg("pipeline.data_limits.min_cross_sectional_samples", default=10, config_name="pipeline_config"))
        except Exception as e:
            # CH-010: Strict mode fails on config errors
            from TRAINING.common.determinism import is_strict_mode
            if is_strict_mode():
                from TRAINING.common.exceptions import ConfigError
                raise ConfigError(f"CH-010: Failed to load min_cs config in strict mode: {e}") from e
            min_cs = 10

    if max_cs_samples is None:
        try:
            from CONFIG.config_loader import get_cfg
            max_cs_samples = int(get_cfg("pipeline.data_limits.max_cs_samples", default=1000, config_name="pipeline_config"))
        except Exception as e:
            # CH-010: Strict mode fails on config errors
            from TRAINING.common.determinism import is_strict_mode
            if is_strict_mode():
                from TRAINING.common.exceptions import ConfigError
                raise ConfigError(f"CH-010: Failed to load max_cs_samples config in strict mode: {e}") from e
            max_cs_samples = 1000
    
    if max_rows_per_symbol is None:
        # First check experiment config if available (same pattern as model_evaluation.py)
        if experiment_config and hasattr(experiment_config, 'max_samples_per_symbol'):
            max_rows_per_symbol = experiment_config.max_samples_per_symbol
            logger.debug(f"Using max_rows_per_symbol={max_rows_per_symbol} from experiment config attribute")
        else:
            # Try reading from experiment config YAML directly (same as model_evaluation.py)
            if experiment_config:
                try:
                    exp_name = experiment_config.name
                    if _CONFIG_AVAILABLE:
                        exp_yaml = load_experiment_config(exp_name)
                    else:
                        # CH-009: Use load_yaml from config_builder when available
                        exp_file = Path("CONFIG/experiments") / f"{exp_name}.yaml"
                        if exp_file.exists():
                            if _NEW_CONFIG_AVAILABLE:
                                exp_yaml = load_yaml(exp_file) or {}
                            else:
                                import yaml
                                with open(exp_file, 'r') as f:
                                    exp_yaml = yaml.safe_load(f) or {}
                        else:
                            exp_yaml = {}
                        exp_data = exp_yaml.get('data', {})
                        if 'max_samples_per_symbol' in exp_data:
                            max_rows_per_symbol = exp_data['max_samples_per_symbol']
                            logger.debug(f"Using max_rows_per_symbol={max_rows_per_symbol} from experiment config YAML")
                except Exception:
                    pass

            # Fallback to pipeline config
            if max_rows_per_symbol is None:
                try:
                    from CONFIG.config_loader import get_cfg
                    max_rows_per_symbol = int(get_cfg("pipeline.data_limits.default_max_rows_per_symbol_ranking", default=50000, config_name="pipeline_config"))
                except Exception as e:
                    # CH-010: Strict mode fails on config errors
                    from TRAINING.common.determinism import is_strict_mode
                    if is_strict_mode():
                        from TRAINING.common.exceptions import ConfigError
                        raise ConfigError(f"CH-010: Failed to load max_rows_per_symbol config in strict mode: {e}") from e
                    max_rows_per_symbol = 50000

    return _evaluate_target_predictability(
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
        scope_purpose=scope_purpose,
        run_identity=run_identity,  # NEW: Pass RunIdentity SST object
        registry=registry,  # NEW: Pass registry instance
        coverage_breakdowns_dict=coverage_breakdowns_dict,  # NEW: Collect breakdowns
    )


def discover_targets(
    symbol: str,
    data_dir: Path
) -> Dict[str, TargetConfig]:
    """
    Auto-discover all valid targets from data (non-degenerate).
    
    Preserves all leakage-free filtering (excludes first_touch targets, etc.).
    
    Args:
        symbol: Symbol to use for discovery
        data_dir: Directory containing symbol data
    
    Returns:
        Dict mapping target -> TargetConfig
    """
    return _discover_all_targets(symbol, data_dir)


def load_target_configs() -> Dict[str, Dict]:
    """
    Load target configurations from CONFIG/target_configs.yaml.
    
    Returns:
        Dict mapping target -> target config dict
    """
    return _load_target_configs()


def rank_targets(
    targets: Dict[str, TargetConfig | Dict[str, Any]],
    symbols: List[str],
    data_dir: Path,
    model_families: List[str],
    multi_model_config: Dict[str, Any] = None,
    output_dir: Path = None,
    min_cs: Optional[int] = None,  # Load from config if None
    max_cs_samples: Optional[int] = None,  # Load from config if None
    max_rows_per_symbol: Optional[int] = None,  # Load from config if None
    top_n: Optional[int] = None,
    max_targets_to_evaluate: Optional[int] = None,  # Limit number of targets to evaluate (for faster testing)
    target_ranking_config: Optional['TargetRankingConfig'] = None,  # New typed config (optional)
    explicit_interval: Optional[Union[int, str]] = None,  # Explicit interval from config (e.g., "5m")
    experiment_config: Optional[Any] = None,  # Optional ExperimentConfig (for data.bar_interval)
    run_identity: Optional[Any] = None,  # NEW: RunIdentity SST object for authoritative signatures
    registry: Optional[Any] = None,  # NEW: Pass registry instance (reuse same)
    coverage_breakdowns_dict: Optional[Dict[str, Any]] = None,  # NEW: Optional dict to collect coverage breakdowns
) -> List[TargetPredictabilityScore]:
    """
    Rank multiple targets by predictability.
    
    This function evaluates all targets and returns them sorted by
    composite predictability score. All leakage-free behavior is preserved.
    
    Args:
        targets: Dict mapping target -> TargetConfig or config dict
        symbols: List of symbols to evaluate on
        data_dir: Directory containing symbol data
        model_families: List of model family names to use
        multi_model_config: Multi-model config dict [LEGACY]
        output_dir: Optional output directory for results
        min_cs: Minimum cross-sectional size per timestamp (loads from config if None)
        max_cs_samples: Maximum samples per timestamp for cross-sectional sampling (loads from config if None)
        max_rows_per_symbol: Maximum rows to load per symbol (loads from config if None)
        top_n: Optional limit on number of top targets to return (after ranking)
        max_targets_to_evaluate: Optional limit on number of targets to evaluate (for faster testing)
        target_ranking_config: Optional TargetRankingConfig object [NEW - preferred]
    
    Returns:
        List of TargetPredictabilityScore objects, sorted by composite_score (descending)
    """
    # Initialize coverage_breakdowns_dict if not provided
    if coverage_breakdowns_dict is None:
        coverage_breakdowns_dict = {}
    
    # Registry should be provided by caller (intelligent_trainer.py)
    # Early validation: fail fast in strict mode
    from TRAINING.common.determinism import is_strict_mode
    from TRAINING.common.exceptions import RegistryLoadError
    
    fail_closed_registry = is_strict_mode()
    
    if registry is None:
        if fail_closed_registry:
            raise RegistryLoadError(
                message="registry is None in rank_targets() (strict mode). "
                        "Registry must be loaded in IntelligentTrainer.rank_targets_auto().",
                registry_path=None,
                stage="TARGET_RANKING",
                error_code="REGISTRY_LOAD_FAILED"
            )
        else:
            logger.warning(
                "registry is None in rank_targets() (best-effort mode). "
                "Coverage computation will be skipped."
            )
    
    # Load from config if not provided - check experiment config first
    if min_cs is None:
        # First check experiment config if available (read from YAML data section)
        if experiment_config:
            try:
                exp_name = experiment_config.name
                exp_file = Path("CONFIG/experiments") / f"{exp_name}.yaml"
                if exp_file.exists():
                    # CH-009: Use load_yaml from config_builder when available
                    if _NEW_CONFIG_AVAILABLE:
                        exp_yaml = load_yaml(exp_file) or {}
                    else:
                        import yaml
                        with open(exp_file, 'r') as f:
                            exp_yaml = yaml.safe_load(f) or {}
                    exp_data = exp_yaml.get('data', {})
                    if 'min_cs' in exp_data:
                        min_cs = exp_data['min_cs']
                        logger.debug(f"Using min_cs={min_cs} from experiment config")
            except Exception:
                pass

        # Fallback to pipeline config
        if min_cs is None:
            try:
                from CONFIG.config_loader import get_cfg
                min_cs = int(get_cfg("pipeline.data_limits.min_cross_sectional_samples", default=10, config_name="pipeline_config"))
            except Exception as e:
                # CH-010: Strict mode fails on config errors
                from TRAINING.common.determinism import is_strict_mode
                if is_strict_mode():
                    from TRAINING.common.exceptions import ConfigError
                    raise ConfigError(f"CH-010: Failed to load min_cs config in strict mode: {e}") from e
                min_cs = 10

    if max_cs_samples is None:
        # First check experiment config if available (read from YAML data section)
        if experiment_config:
            try:
                exp_name = experiment_config.name
                exp_file = Path("CONFIG/experiments") / f"{exp_name}.yaml"
                if exp_file.exists():
                    # CH-009: Use load_yaml from config_builder when available
                    if _NEW_CONFIG_AVAILABLE:
                        exp_yaml = load_yaml(exp_file) or {}
                    else:
                        import yaml
                        with open(exp_file, 'r') as f:
                            exp_yaml = yaml.safe_load(f) or {}
                    exp_data = exp_yaml.get('data', {})
                    if 'max_cs_samples' in exp_data:
                        max_cs_samples = exp_data['max_cs_samples']
                        logger.debug(f"Using max_cs_samples={max_cs_samples} from experiment config")
            except Exception:
                pass

        # Fallback to pipeline config
        if max_cs_samples is None:
            try:
                from CONFIG.config_loader import get_cfg
                max_cs_samples = int(get_cfg("pipeline.data_limits.max_cs_samples", default=1000, config_name="pipeline_config"))
            except Exception as e:
                # CH-010: Strict mode fails on config errors
                from TRAINING.common.determinism import is_strict_mode
                if is_strict_mode():
                    from TRAINING.common.exceptions import ConfigError
                    raise ConfigError(f"CH-010: Failed to load max_cs_samples config in strict mode: {e}") from e
                max_cs_samples = 1000

    if max_rows_per_symbol is None:
        # First check experiment config if available (same pattern as model_evaluation.py)
        if experiment_config and hasattr(experiment_config, 'max_samples_per_symbol'):
            max_rows_per_symbol = experiment_config.max_samples_per_symbol
            logger.debug(f"Using max_rows_per_symbol={max_rows_per_symbol} from experiment config attribute")
        else:
            # Try reading from experiment config YAML directly (same as model_evaluation.py)
            if experiment_config:
                try:
                    exp_name = experiment_config.name
                    if _CONFIG_AVAILABLE:
                        exp_yaml = load_experiment_config(exp_name)
                    else:
                        # CH-009: Use load_yaml from config_builder when available
                        exp_file = Path("CONFIG/experiments") / f"{exp_name}.yaml"
                        if exp_file.exists():
                            if _NEW_CONFIG_AVAILABLE:
                                exp_yaml = load_yaml(exp_file) or {}
                            else:
                                import yaml
                                with open(exp_file, 'r') as f:
                                    exp_yaml = yaml.safe_load(f) or {}
                        else:
                            exp_yaml = {}
                        exp_data = exp_yaml.get('data', {})
                        if 'max_samples_per_symbol' in exp_data:
                            max_rows_per_symbol = exp_data['max_samples_per_symbol']
                            logger.debug(f"Using max_rows_per_symbol={max_rows_per_symbol} from experiment config YAML")
                except Exception:
                    pass

            # Fallback to pipeline config
            if max_rows_per_symbol is None:
                try:
                    from CONFIG.config_loader import get_cfg
                    max_rows_per_symbol = int(get_cfg("pipeline.data_limits.default_max_rows_per_symbol_ranking", default=50000, config_name="pipeline_config"))
                except Exception as e:
                    # CH-010: Strict mode fails on config errors
                    from TRAINING.common.determinism import is_strict_mode
                    if is_strict_mode():
                        from TRAINING.common.exceptions import ConfigError
                        raise ConfigError(f"CH-010: Failed to load max_rows_per_symbol config in strict mode: {e}") from e
                    max_rows_per_symbol = 50000

    # Results storage: separate by view
    results_cs = []  # Cross-sectional results
    results_sym = {}  # Symbol-specific results: {target: {symbol: result}}
    symbol_skip_reasons = {}  # Skip reasons: {target: {symbol: {reason, status, ...}}}
    results_loso = {}  # LOSO results: {symbol: [results]} (optional)
    
    # Load dual-view config (experiment config takes precedence over global config)
    enable_symbol_specific = True  # Default: enable symbol-specific evaluation
    enable_loso = False  # Default: disable LOSO (optional, high value)
    
    # First, try to load from experiment config (per-experiment control)
    exp_target_ranking = {}
    if experiment_config:
        try:
            exp_name = experiment_config.name
            exp_file = Path("CONFIG/experiments") / f"{exp_name}.yaml"
            if exp_file.exists():
                # CH-009: Use load_yaml from config_builder when available
                if _NEW_CONFIG_AVAILABLE:
                    exp_yaml = load_yaml(exp_file) or {}
                else:
                    import yaml
                    with open(exp_file, 'r') as f:
                        exp_yaml = yaml.safe_load(f) or {}
                exp_target_ranking = exp_yaml.get('target_ranking', {})
                if 'enable_symbol_specific' in exp_target_ranking:
                    enable_symbol_specific = bool(exp_target_ranking['enable_symbol_specific'])
                    logger.debug(f"Using enable_symbol_specific={enable_symbol_specific} from experiment config")
                if 'enable_loso' in exp_target_ranking:
                    enable_loso = bool(exp_target_ranking['enable_loso'])
                    logger.debug(f"Using enable_loso={enable_loso} from experiment config")
        except Exception as e:
            logger.debug(f"Failed to load target_ranking from experiment config: {e}")
    
    # Fallback to global config if not set in experiment config
    if _CONFIG_AVAILABLE:
        try:
            from CONFIG.config_loader import get_cfg
            ranking_cfg = get_cfg("target_ranking", default={}, config_name="target_ranking_config")
            # Only use global config if experiment config didn't set these
            if 'enable_symbol_specific' not in exp_target_ranking:
                enable_symbol_specific = ranking_cfg.get('enable_symbol_specific', enable_symbol_specific)
            if 'enable_loso' not in exp_target_ranking:
                enable_loso = ranking_cfg.get('enable_loso', enable_loso)
        except Exception:
            pass

    # FIX: Early SS gate - disable symbol-specific evaluation for large universes
    # This prevents wasted compute when max_symbols_for_ss would gate the routing anyway
    # Check BEFORE any SS evaluation runs (not just at routing time)
    if enable_symbol_specific:
        try:
            from CONFIG.config_loader import get_cfg as _get_cfg
            # Check experiment config first (has precedence)
            # NOTE: target_routing is at ROOT level of experiment YAML, not under target_ranking
            max_symbols_for_ss = None
            if experiment_config:
                try:
                    exp_name = experiment_config.name
                    exp_file = Path("CONFIG/experiments") / f"{exp_name}.yaml"
                    if exp_file.exists():
                        if _NEW_CONFIG_AVAILABLE:
                            exp_yaml = load_yaml(exp_file) or {}
                        else:
                            import yaml
                            with open(exp_file, 'r') as f:
                                exp_yaml = yaml.safe_load(f) or {}
                        # target_routing is at ROOT level, not under target_ranking
                        routing_cfg = exp_yaml.get('target_routing', {}) or {}
                        if 'max_symbols_for_ss' in routing_cfg:
                            max_symbols_for_ss = int(routing_cfg['max_symbols_for_ss'])
                            logger.debug(f"SS gate: using experiment config max_symbols_for_ss={max_symbols_for_ss}")
                except Exception as e:
                    logger.debug(f"Could not load SS gate config from experiment: {e}")
            # Fallback to base routing config
            if max_symbols_for_ss is None:
                max_symbols_for_ss = int(_get_cfg("target_routing.max_symbols_for_ss", default=100))
                logger.debug(f"SS gate: using base config max_symbols_for_ss={max_symbols_for_ss}")

            universe_size = len(symbols)
            if max_symbols_for_ss == 0 or universe_size > max_symbols_for_ss:
                enable_symbol_specific = False
                logger.info(
                    f"🚫 SS evaluation DISABLED at ranking stage: "
                    f"universe_size={universe_size}, max_symbols_for_ss={max_symbols_for_ss}. "
                    f"Skipping all per-symbol evaluation to save compute."
                )
        except Exception as e:
            logger.debug(f"Could not check max_symbols_for_ss gate: {e}")

    # Results list for backward compatibility (will contain cross-sectional + aggregated symbol results)
    results = []
    
    # Track all evaluated targets (for ensuring decision files are created)
    all_evaluated_targets = set()
    # Track all CS results (including failed ones) for decision computation
    all_cs_results = {}  # {target: TargetPredictabilityScore}
    
    # NEW: Use typed config if provided
    if target_ranking_config is not None and _NEW_CONFIG_AVAILABLE:
        # Extract values from typed config
        if target_ranking_config.model_families:
            # Convert to list of enabled family names
            # DETERMINISM_CRITICAL: Model family order must be deterministic
            model_families = [
                name for name, cfg in sorted_items(target_ranking_config.model_families)
                if cfg.get('enabled', False)
            ]
        if target_ranking_config.data_dir:
            data_dir = target_ranking_config.data_dir
        if target_ranking_config.symbols:
            symbols = target_ranking_config.symbols
        if target_ranking_config.max_samples_per_symbol:
            max_rows_per_symbol = target_ranking_config.max_samples_per_symbol
        # Build multi_model_config dict from typed config for backward compat
        multi_model_config = {
            'model_families': target_ranking_config.model_families,
            'cross_validation': target_ranking_config.cross_validation,
            'sampling': target_ranking_config.sampling,
            'ranking': target_ranking_config.ranking
        }
    
    # Limit targets to evaluate if specified (for faster testing)
    all_targets_count = len(targets)
    targets_to_evaluate = targets
    if max_targets_to_evaluate is not None and max_targets_to_evaluate > 0:
        # Sort targets alphabetically for consistent ordering
        sorted_target_items = sorted(targets.items(), key=lambda x: x[0])
        target_items = sorted_target_items[:max_targets_to_evaluate]
        targets_to_evaluate = dict(target_items)
        logger.info(f"Limiting evaluation to {len(targets_to_evaluate)} targets (out of {all_targets_count} total) for faster testing")
        # DETERMINISM: Use sorted_keys for deterministic iteration order
        selected_targets = list(sorted_keys(targets_to_evaluate))
        if len(selected_targets) <= 10:
            logger.debug(f"Selected targets: {selected_targets}")
        else:
            logger.debug(f"Selected targets: {selected_targets[:10]}... (showing first 10 of {len(selected_targets)})")
    
    total_to_evaluate = len(targets_to_evaluate)
    logger.info(f"Ranking {total_to_evaluate} targets across {len(symbols)} symbols")
    logger.info(f"Model families: {', '.join(model_families)}")
    
    # Load auto-rerun config (SST: experiment config → safety config → defaults)
    auto_rerun_enabled = False
    max_reruns = 3
    rerun_on_perfect_train_acc = True
    rerun_on_high_auc_only = False
    
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
                max_reruns = int(auto_rerun_cfg.get('max_reruns', 3))
                rerun_on_perfect_train_acc = auto_rerun_cfg.get('rerun_on_perfect_train_acc', True)
                rerun_on_high_auc_only = auto_rerun_cfg.get('rerun_on_high_auc_only', False)
                logger.debug(f"Loaded auto-rerun settings from experiment config: enabled={auto_rerun_enabled}, max_reruns={max_reruns}")
        except Exception as e:
            logger.debug(f"Could not load auto-rerun from experiment config: {e}")
    
    # 2. Fallback to safety config if experiment config didn't provide settings
    if not auto_rerun_enabled and _CONFIG_AVAILABLE:
        try:
            safety_cfg = get_safety_config()
            # safety_config.yaml has a top-level 'safety' key
            safety_section = safety_cfg.get('safety', {})
            leakage_cfg = safety_section.get('leakage_detection', {})
            auto_rerun_cfg = leakage_cfg.get('auto_rerun', {})
            if auto_rerun_cfg:
                auto_rerun_enabled = auto_rerun_cfg.get('enabled', False)
                max_reruns = int(auto_rerun_cfg.get('max_reruns', 3))
                rerun_on_perfect_train_acc = auto_rerun_cfg.get('rerun_on_perfect_train_acc', True)
                rerun_on_high_auc_only = auto_rerun_cfg.get('rerun_on_high_auc_only', False)
                logger.debug(f"Loaded auto-rerun settings from safety config: enabled={auto_rerun_enabled}, max_reruns={max_reruns}")
        except Exception as e:
            logger.debug(f"Could not load auto-rerun from safety config: {e}")
    
    # Load parallel execution config
    parallel_targets = False
    if _CONFIG_AVAILABLE:
        try:
            from CONFIG.config_loader import get_cfg
            multi_target_cfg = get_cfg("multi_target", default={}, config_name="target_configs")
            parallel_targets = multi_target_cfg.get('parallel_targets', False)
        except Exception:
            pass
    
    # Check if parallel execution is globally enabled
    parallel_enabled = _PARALLEL_AVAILABLE and parallel_targets
    if parallel_enabled:
        try:
            from CONFIG.config_loader import get_cfg
            parallel_global = get_cfg("threading.parallel.enabled", default=True, config_name="threading_config")
            parallel_enabled = parallel_enabled and parallel_global
        except Exception:
            pass
    
    # Helper function for parallel target evaluation (must be picklable)
    def _evaluate_single_target(item, context):
        """Evaluate a single target - wrapper for parallel execution"""
        target, target_config = item
        
        # Extract shared variables from context (avoids closure pickling issues)
        coverage_breakdowns_dict = context.get('coverage_breakdowns_dict')
        run_identity = context.get('run_identity')
        symbols = context['symbols']
        data_dir = context['data_dir']
        model_families = context['model_families']
        multi_model_config = context.get('multi_model_config')
        output_dir = context.get('output_dir')
        min_cs = context['min_cs']
        max_cs_samples = context.get('max_cs_samples')
        max_rows_per_symbol = context.get('max_rows_per_symbol')
        explicit_interval = context.get('explicit_interval')
        experiment_config = context.get('experiment_config')
        auto_rerun_enabled = context.get('auto_rerun_enabled', False)
        max_reruns = context.get('max_reruns', 3)
        rerun_on_perfect_train_acc = context.get('rerun_on_perfect_train_acc', True)
        rerun_on_high_auc_only = context.get('rerun_on_high_auc_only', False)
        enable_symbol_specific = context.get('enable_symbol_specific', True)
        enable_loso = context.get('enable_loso', False)
        
        # Reconstruct registry from provenance in worker process
        from TRAINING.common.utils.registry_provenance import load_registry_from_provenance, RegistryProvenance
        import multiprocessing
        import os
        from pathlib import Path
        
        registry_provenance_dict = context.get('registry_provenance')
        fail_closed = context.get('fail_closed', True)  # Default to fail-closed for safety
        
        registry = None
        if registry_provenance_dict:
            # Deserialize provenance
            try:
                prov = RegistryProvenance(
                    registry_path_abs=Path(registry_provenance_dict['registry_path_abs']),
                    overlay_paths={
                        k: Path(v) if v else None 
                        for k, v in registry_provenance_dict['overlay_paths'].items()
                    },
                    overlay_interval_minutes=registry_provenance_dict.get('overlay_interval_minutes'),
                    current_bar_minutes=registry_provenance_dict['current_bar_minutes'],
                    allow_overwrite=registry_provenance_dict['allow_overwrite'],
                    registry_path_exists=registry_provenance_dict['registry_path_exists'],  # Parent observed
                    registry_path_readable=registry_provenance_dict['registry_path_readable'],  # Parent observed
                    registry_identity_hash=registry_provenance_dict.get('registry_identity_hash'),
                )
                
                # Load registry (raises on failure in fail-closed mode, returns degraded in best-effort)
                registry = load_registry_from_provenance(prov, fail_closed=fail_closed)
            except Exception as e:
                # Log structured diagnostics
                logger.error(
                    "Registry reconstruction failed in worker: %s. "
                    "registry_path=%s exists=%s readable=%s overlay_paths=%s "
                    "overlay_interval=%s cwd=%s pid=%s start_method=%s identity_hash=%s",
                    e,
                    registry_provenance_dict.get('registry_path_abs'),
                    registry_provenance_dict.get('registry_path_exists'),
                    registry_provenance_dict.get('registry_path_readable'),
                    registry_provenance_dict.get('overlay_paths'),
                    registry_provenance_dict.get('overlay_interval_minutes'),
                    Path.cwd(),
                    os.getpid(),
                    multiprocessing.get_start_method(),
                    registry_provenance_dict.get('registry_identity_hash', '')[:16] if registry_provenance_dict.get('registry_identity_hash') else None
                )
                if fail_closed:
                    raise  # Fail closed
                else:
                    # Best-effort: load_registry_from_provenance should have returned degraded registry
                    # If we reach here, it means load_registry_from_provenance didn't handle best-effort correctly
                    raise ValueError(f"Registry reconstruction failed (best-effort mode should return degraded registry): {e}") from e
        else:
            # No provenance provided - try to load with defaults (fallback)
            logger.warning("No registry_provenance in context. Attempting to load registry with defaults.")
            try:
                from TRAINING.common.feature_registry import get_registry
                registry = get_registry(strict=fail_closed)
                if registry is None:
                    if fail_closed:
                        raise ValueError("get_registry() returned None in strict mode (this should be impossible)")
                    else:
                        raise ValueError("get_registry() returned None in best-effort mode (empty registry)")
            except Exception as e:
                if fail_closed:
                    raise
                logger.warning(f"Failed to load registry with defaults: {e}")
                # Best-effort: would need degraded registry wrapper here
                raise ValueError(f"Registry reconstruction failed (no provenance and defaults failed): {e}") from e
        
        # Defensive: validate registry is not None (this is a bug, not a runtime condition)
        if registry is None:
            raise ValueError(
                "registry is None after reconstruction (this should be impossible). "
                f"Check load_registry_from_provenance() implementation. "
                f"target={target if 'target' in locals() else 'unknown'}, worker_pid={os.getpid()}"
            )
        
        result_data = {
            'target': target,
            'result_cs': None,
            'result_sym_dict': {},
            'result_loso_dict': {},
            'error': None,
            'coverage_summary': None  # NEW: Store coverage summary for parent aggregation
        }
        
        try:
            # View A: Cross-sectional evaluation (always run)
            if auto_rerun_enabled and _AUTOFIX_AVAILABLE:
                # FIX: Use signature check instead of try/except TypeError
                if _AUTOFIX_HAS_VIEW_PARAM:
                    result_cs = evaluate_target_with_autofix(
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
                        max_reruns=max_reruns,
                        rerun_on_perfect_train_acc=rerun_on_perfect_train_acc,
                        rerun_on_high_auc_only=rerun_on_high_auc_only,
                        explicit_interval=explicit_interval,
                        experiment_config=experiment_config,
                        view=View.CROSS_SECTIONAL,
                        symbol=None,
                        registry=registry,
                        coverage_breakdowns_dict=coverage_breakdowns_dict,
                    )
                else:
                    result_cs = evaluate_target_with_autofix(
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
                        max_reruns=max_reruns,
                        rerun_on_perfect_train_acc=rerun_on_perfect_train_acc,
                        rerun_on_high_auc_only=rerun_on_high_auc_only,
                        explicit_interval=explicit_interval,
                        experiment_config=experiment_config,
                        registry=registry,
                        coverage_breakdowns_dict=coverage_breakdowns_dict,
                    )
            else:
                result_cs = evaluate_target_predictability(
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
                        view=View.CROSS_SECTIONAL,
                    symbol=None,
                    run_identity=run_identity,  # NEW: Pass RunIdentity
                    registry=registry,  # NEW: Pass same instance
                    coverage_breakdowns_dict=coverage_breakdowns_dict,  # NEW: Collect breakdowns
                )
            
            result_data['result_cs'] = result_cs
            
            # Extract coverage summary from result (if available)
            # Coverage breakdown is stored in coverage_breakdowns_dict for CS view
            if coverage_breakdowns_dict and target in coverage_breakdowns_dict:
                try:
                    from TRAINING.ranking.utils.registry_coverage import summarize_coverage_breakdown
                    coverage_breakdown = coverage_breakdowns_dict[target]
                    if coverage_breakdown and coverage_breakdown.coverage_mode in ("horizon_ok", "membership_only"):
                        result_data['coverage_summary'] = summarize_coverage_breakdown(coverage_breakdown)
                except Exception as e:
                    logger.debug(f"Failed to compute coverage summary for {target}: {e}")
            
            # View B: Symbol-specific evaluation (if enabled and cross-sectional succeeded)
            skip_statuses = ["LEAKAGE_UNRESOLVED", "LEAKAGE_UNRESOLVED_MAX_RETRIES", "SUSPICIOUS", "SUSPICIOUS_STRONG"]
            cs_succeeded = result_cs.auc != -999.0 and result_cs.status not in skip_statuses
            
            if enable_symbol_specific and cs_succeeded:
                result_sym_dict = {}
                for symbol in symbols:
                    try:
                        if auto_rerun_enabled and _AUTOFIX_AVAILABLE:
                            # FIX: Use signature check instead of try/except TypeError
                            if _AUTOFIX_HAS_VIEW_PARAM:
                                result_sym = evaluate_target_with_autofix(
                                    target=target,
                                    target_config=target_config,
                                    symbols=[symbol],
                                    data_dir=data_dir,
                                    model_families=model_families,
                                    multi_model_config=multi_model_config,
                                    output_dir=output_dir,
                                    min_cs=1,
                                    max_cs_samples=max_cs_samples,
                                    max_rows_per_symbol=max_rows_per_symbol,
                                    max_reruns=max_reruns,
                                    rerun_on_perfect_train_acc=rerun_on_perfect_train_acc,
                                    rerun_on_high_auc_only=rerun_on_high_auc_only,
                                    explicit_interval=explicit_interval,
                                    experiment_config=experiment_config,
                                    view=View.SYMBOL_SPECIFIC,
                                    symbol=symbol,
                                    registry=registry,
                                    coverage_breakdowns_dict=coverage_breakdowns_dict,
                                )
                            else:
                                result_sym = evaluate_target_with_autofix(
                                    target=target,
                                    target_config=target_config,
                                    symbols=[symbol],
                                    data_dir=data_dir,
                                    model_families=model_families,
                                    multi_model_config=multi_model_config,
                                    output_dir=output_dir,
                                    min_cs=1,
                                    max_cs_samples=max_cs_samples,
                                    max_rows_per_symbol=max_rows_per_symbol,
                                    max_reruns=max_reruns,
                                    rerun_on_perfect_train_acc=rerun_on_perfect_train_acc,
                                    rerun_on_high_auc_only=rerun_on_high_auc_only,
                                    explicit_interval=explicit_interval,
                                    experiment_config=experiment_config,
                                    registry=registry,
                                    coverage_breakdowns_dict=coverage_breakdowns_dict,
                                )
                        else:
                            result_sym = evaluate_target_predictability(
                                target=target,
                                target_config=target_config,
                                symbols=[symbol],
                                data_dir=data_dir,
                                model_families=model_families,
                                multi_model_config=multi_model_config,
                                output_dir=output_dir,
                                min_cs=1,
                                max_cs_samples=max_cs_samples,
                                max_rows_per_symbol=max_rows_per_symbol,
                                explicit_interval=explicit_interval,
                                experiment_config=experiment_config,
                                view=View.SYMBOL_SPECIFIC,
                                symbol=symbol,
                                run_identity=run_identity,  # NEW: Pass RunIdentity
                                registry=registry,  # NEW: Pass same instance
                                coverage_breakdowns_dict=coverage_breakdowns_dict,  # NEW: Collect breakdowns (SS won't be stored)
                            )
                        
                        if result_sym.auc != -999.0:
                            result_sym_dict[symbol] = result_sym
                    except Exception as e:
                        logger.warning(f"    Failed to evaluate {target} for symbol {symbol}: {e}")
                        continue
                
                result_data['result_sym_dict'] = result_sym_dict
            
            # View C: LOSO evaluation (if enabled)
            if enable_loso:
                result_loso_dict = {}
                for symbol in symbols:
                    try:
                        result_loso_sym = evaluate_target_predictability(
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
                            view="LOSO",
                            symbol=symbol,
                            run_identity=run_identity,  # NEW: Pass RunIdentity
                            registry=registry,  # NEW: Pass same instance
                            coverage_breakdowns_dict=coverage_breakdowns_dict,  # NEW: Collect breakdowns (LOSO won't be stored)
                        )
                        result_loso_dict[symbol] = result_loso_sym
                    except Exception as e:
                        logger.warning(f"    Failed LOSO evaluation for {target} on symbol {symbol}: {e}")
                        continue
                
                result_data['result_loso_dict'] = result_loso_dict
                
        except Exception as e:
            result_data['error'] = str(e)
            logger.exception(f"  Failed to evaluate {target}: {e}")
        
        return result_data
    
    # Evaluate targets (parallel or sequential)
    if parallel_enabled and len(targets_to_evaluate) > 1:
        logger.info(f"🚀 Parallel target evaluation enabled ({len(targets_to_evaluate)} targets)")
        # DETERMINISM_CRITICAL: Target evaluation order must be deterministic
        
        # Resolve registry provenance once in parent (before spawning workers)
        from TRAINING.common.utils.registry_provenance import resolve_registry_provenance, RegistryProvenance
        from TRAINING.common.determinism import is_strict_mode
        
        fail_closed = is_strict_mode()
        registry_provenance = None
        if registry is not None:
            try:
                registry_provenance = resolve_registry_provenance(
                    registry=registry,
                    explicit_interval=explicit_interval,
                    experiment_config=experiment_config,
                    strict=fail_closed
                )
            except Exception as e:
                if fail_closed:
                    raise  # Fail closed in strict mode
                logger.warning(
                    f"Failed to resolve registry provenance: {e}. "
                    "Creating fallback provenance for workers."
                )
                # Create fallback provenance even if resolve failed
                try:
                    from TRAINING.common.feature_registry import _resolve_registry_path_base
                    default_path = _resolve_registry_path_base()
                    registry_provenance = RegistryProvenance(
                        registry_path_abs=default_path.resolve() if default_path else Path("CONFIG/data/feature_registry.yaml"),
                        overlay_paths={'global': None, 'per_target': None},
                        overlay_interval_minutes=None,
                        current_bar_minutes=explicit_interval if explicit_interval else None,
                        allow_overwrite=False,
                        registry_path_exists=default_path.exists() if default_path else False,
                        registry_path_readable=False,
                        registry_identity_hash=None
                    )
                except Exception as e2:
                    logger.warning(f"Failed to create fallback registry provenance: {e2}")
                    # registry_provenance stays None - workers will try defaults
        else:
            # Registry is None - create minimal provenance for worker fallback
            # Workers will attempt to load with defaults
            try:
                from TRAINING.common.feature_registry import _resolve_registry_path_base
                default_path = _resolve_registry_path_base()
                registry_provenance = RegistryProvenance(
                    registry_path_abs=default_path.resolve() if default_path else Path("CONFIG/data/feature_registry.yaml"),
                    overlay_paths={'global': None, 'per_target': None},
                    overlay_interval_minutes=None,
                    current_bar_minutes=explicit_interval if explicit_interval else None,
                    allow_overwrite=False,
                    registry_path_exists=default_path.exists() if default_path else False,
                    registry_path_readable=False,
                    registry_identity_hash=None
                )
            except Exception as e:
                logger.warning(f"Failed to create fallback registry provenance: {e}")
        
        # Pass provenance to workers (serialize as dict)
        registry_provenance_dict = None
        if registry_provenance:
            registry_provenance_dict = {
                'registry_path_abs': str(registry_provenance.registry_path_abs),
                'overlay_paths': {
                    k: str(v) if v else None 
                    for k, v in registry_provenance.overlay_paths.items()
                },
                'overlay_interval_minutes': registry_provenance.overlay_interval_minutes,
                'current_bar_minutes': registry_provenance.current_bar_minutes,
                'allow_overwrite': registry_provenance.allow_overwrite,
                'registry_path_exists': registry_provenance.registry_path_exists,  # Parent observed (worker revalidates)
                'registry_path_readable': registry_provenance.registry_path_readable,  # Parent observed (worker revalidates)
                'registry_identity_hash': registry_provenance.registry_identity_hash,
            }
        
        context = {
            'registry_provenance': registry_provenance_dict,  # Pass provenance instead of ad-hoc path
            'fail_closed': fail_closed,  # Pass fail_closed flag explicitly
            # NOTE: Shared dict relies on CPython GIL for thread-safe dict updates.
            # If using multiprocessing or non-CPython, use Manager().dict() instead.
            'coverage_breakdowns_dict': coverage_breakdowns_dict,
            'run_identity': run_identity,
            'symbols': symbols,
            'data_dir': data_dir,
            'model_families': model_families,
            'multi_model_config': multi_model_config,
            'output_dir': output_dir,
            'min_cs': min_cs,
            'max_cs_samples': max_cs_samples,
            'max_rows_per_symbol': max_rows_per_symbol,
            'explicit_interval': explicit_interval,
            'experiment_config': experiment_config,
            'auto_rerun_enabled': auto_rerun_enabled,
            'max_reruns': max_reruns,
            'rerun_on_perfect_train_acc': rerun_on_perfect_train_acc,
            'rerun_on_high_auc_only': rerun_on_high_auc_only,
            'enable_symbol_specific': enable_symbol_specific,
            'enable_loso': enable_loso,
        }
        
        parallel_results = execute_parallel_with_context(
            _evaluate_single_target,
            sorted_items(targets_to_evaluate),
            context=context,
            max_workers=None,  # Auto-detect from config
            task_type="process",  # CPU-bound
            desc="Evaluating targets",
            show_progress=True
        )
        
        # Process parallel results
        # DETERMINISM_CRITICAL: Sort results by target name (parallel execution returns in completion order)
        # Sort by target name from result_data dict
        sorted_parallel_results = collect_and_sort_parallel_results(
            parallel_results,
            sort_key=lambda x: x[1].get('target', '') if isinstance(x[1], dict) and 'target' in x[1] else '',
            tie_breaker=lambda x: str(x[0])  # Break ties by original item (target name)
        )
        for item, result_data in sorted_parallel_results:
            target = result_data['target']
            all_evaluated_targets.add(target)  # Track that this target was evaluated
            if result_data['error']:
                logger.error(f"  ❌ {target}: {result_data['error']}")
                continue
            
            result_cs = result_data['result_cs']
            result_sym_dict = result_data.get('result_sym_dict', {})
            result_loso_dict = result_data.get('result_loso_dict', {})
            
            # Track CS result (even if failed) for decision computation
            if result_cs:
                all_cs_results[target] = result_cs
            
            # Process cross-sectional result
            skip_statuses = ["LEAKAGE_UNRESOLVED", "LEAKAGE_UNRESOLVED_MAX_RETRIES", "SUSPICIOUS", "SUSPICIOUS_STRONG"]
            cs_succeeded = result_cs.auc != -999.0 and result_cs.status not in skip_statuses
            if cs_succeeded:
                results_cs.append(result_cs)
                results.append(result_cs)
            else:
                reason = result_cs.status if result_cs.status in skip_statuses else (result_cs.leakage_flag if result_cs.leakage_flag != "OK" else "degenerate/failed")
                if result_cs.status in ["SUSPICIOUS", "SUSPICIOUS_STRONG"]:
                    logger.warning(f"  ⚠️  Excluded {target} CROSS_SECTIONAL ({reason}) - High score suggests structural leakage")
                else:
                    logger.info(f"  Skipped {target} CROSS_SECTIONAL ({reason})")
            
            # Store symbol-specific results
            # DETERMINISM_CRITICAL: Symbol aggregation order must be deterministic
            if enable_symbol_specific:
                if target not in results_sym:
                    results_sym[target] = {}
                for symbol, result_sym in sorted_items(result_sym_dict):
                    if result_sym.auc != -999.0 and result_sym.status not in skip_statuses:
                        results_sym[target][symbol] = result_sym
                    else:
                        reason = result_sym.status if result_sym.status in skip_statuses else "degenerate/failed"
                        logger.debug(f"    Skipped {target} SYMBOL_SPECIFIC ({symbol}): {reason}")
            
            # Store LOSO results
            # DETERMINISM_CRITICAL: Symbol aggregation order must be deterministic
            if enable_loso:
                if target not in results_loso:
                    results_loso[target] = {}
                for symbol, result_loso_sym in sorted_items(result_loso_dict):
                    if result_loso_sym.auc != -999.0 and result_loso_sym.status not in skip_statuses:
                        results_loso[target][symbol] = result_loso_sym
            
            # Save routing decision immediately after target evaluation (incremental)
            if output_dir:
                from TRAINING.ranking.target_routing import (
                    _compute_single_target_routing_decision, _save_single_target_decision
                )
                target_sym_results = results_sym.get(target, {})
                target_skip_reasons = symbol_skip_reasons.get(target, {}) if symbol_skip_reasons else {}
                decision = _compute_single_target_routing_decision(
                    target=target,
                    result_cs=result_cs if cs_succeeded else None,
                    sym_results=target_sym_results,
                    symbol_skip_reasons=target_skip_reasons,
                    experiment_config=experiment_config,
                    total_symbols=len(symbols)  # FIX: Pass total universe size for SS gate
                )
                _save_single_target_decision(target, decision, output_dir)

        # Aggregate coverage summaries by target (dedupe across views) and log warnings once per target
        # Note: sorted_items is already imported at module level (line 51)
        coverage_summaries_by_target = {}
        for item, result_data in sorted_parallel_results:
            target = result_data['target']
            coverage_summary = result_data.get('coverage_summary')  # From worker
            if coverage_summary and coverage_summary.get('coverage_in_registry', 1.0) < 0.5:
                # Only store if coverage is low (avoid storing all summaries)
                if target not in coverage_summaries_by_target:
                    coverage_summaries_by_target[target] = coverage_summary
        
        # Log warnings once per target (not per view)
        for target, coverage_summary in sorted_items(coverage_summaries_by_target):
            # Compute autopatch path (SST helper)
            autopatch_path = None
            if output_dir:
                try:
                    from TRAINING.orchestration.utils.target_first_paths import run_root
                    run_root_dir = run_root(output_dir)
                    autopatch_path = run_root_dir / "registry_patches"  # SST: standard location
                except Exception:
                    pass
            
            logger.warning(
                "Low registry coverage for target=%s: coverage=%.2f mode=%s missing_count=%d. "
                "Missing sample (first 10): %s. "
                "Blocked by reason: %s. "
                "%s",
                target,
                coverage_summary['coverage_in_registry'],
                coverage_summary['mode'],
                coverage_summary['missing_count'],
                coverage_summary['missing_sample_sorted'][:10],
                coverage_summary['blocked_counts_by_reason'],
                f"Run registry autopatch to add missing features. Output: {autopatch_path}" if autopatch_path else "Run registry autopatch to add missing features."
            )
    else:
        # Sequential evaluation (original code path)
        # Validate registry before sequential evaluation
        if registry is None:
            if fail_closed_registry:
                raise RegistryLoadError(
                    message="registry is None in sequential path (strict mode). "
                            "Registry must be loaded before calling rank_targets().",
                    registry_path=None,
                    stage="TARGET_RANKING",
                    error_code="REGISTRY_LOAD_FAILED"
                )
            else:
                logger.warning(
                    "registry is None in sequential path (best-effort mode). "
                    "Coverage computation will be skipped."
                )
        
        if parallel_enabled and len(targets_to_evaluate) == 1:
            logger.info("Running sequentially (only 1 target)")
        elif not parallel_enabled:
            logger.info("Parallel execution disabled (parallel_targets=false or not available)")
        
        # Evaluate each target in dual views
        # FIX ISSUE-022: Sort for determinism - ensures consistent evaluation order regardless of dict construction order
        for idx, (target, target_config) in enumerate(sorted(targets_to_evaluate.items()), 1):
            all_evaluated_targets.add(target)  # Track that this target was evaluated
            logger.info(f"[{idx}/{total_to_evaluate}] Evaluating {target}...")

            # Emit progress event for dashboard monitoring
            try:
                from TRAINING.orchestration.utils.training_events import emit_progress
                emit_progress(
                    stage="ranking",
                    progress_pct=((idx - 1) / total_to_evaluate) * 100,
                    current_target=target,
                    targets_complete=idx - 1,
                    targets_total=total_to_evaluate,
                    message=f"Evaluating target: {target}"
                )
            except ImportError:
                pass  # Training events module not available
            
            try:
                # View A: Cross-sectional evaluation (always run)
                logger.info(f"  View A: CROSS_SECTIONAL")
                result_cs = None
                cs_error = None
                try:
                    if auto_rerun_enabled and _AUTOFIX_AVAILABLE:
                        # FIX: Use signature check instead of try/except TypeError
                        if _AUTOFIX_HAS_VIEW_PARAM:
                            result_cs = evaluate_target_with_autofix(
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
                                max_reruns=max_reruns,
                                rerun_on_perfect_train_acc=rerun_on_perfect_train_acc,
                                rerun_on_high_auc_only=rerun_on_high_auc_only,
                                explicit_interval=explicit_interval,
                                experiment_config=experiment_config,
                                view=View.CROSS_SECTIONAL,
                                symbol=None,
                                registry=registry,
                                coverage_breakdowns_dict=coverage_breakdowns_dict,
                            )
                        else:
                            result_cs = evaluate_target_with_autofix(
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
                                max_reruns=max_reruns,
                                rerun_on_perfect_train_acc=rerun_on_perfect_train_acc,
                                rerun_on_high_auc_only=rerun_on_high_auc_only,
                                explicit_interval=explicit_interval,
                                experiment_config=experiment_config,
                                registry=registry,
                                coverage_breakdowns_dict=coverage_breakdowns_dict,
                            )
                    else:
                        result_cs = evaluate_target_predictability(
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
                            view=View.CROSS_SECTIONAL,
                            symbol=None,
                            run_identity=run_identity,  # NEW: Pass RunIdentity
                            registry=registry,  # NEW: Pass same instance
                            coverage_breakdowns_dict=coverage_breakdowns_dict,  # NEW: Collect breakdowns
                        )
                except Exception as e:
                    # FIX: Catch exceptions in cross-sectional evaluation so symbol-specific can still run
                    cs_error = e
                    logger.error(f"  ❌ Failed to evaluate {target} CROSS_SECTIONAL: {e}", exc_info=True)
                    # Create a failed result so symbol-specific can still run
                    # NOTE: TargetPredictabilityScore is already imported at top of file (line 26)
                    # Derive task_type from context (don't guess silently)
                    try:
                        task_type = TaskType.from_target_column(target) if target else TaskType.REGRESSION
                        task_type_source = "derived_from_target"
                    except Exception as e_task:
                        task_type = TaskType.REGRESSION  # Default fallback
                        task_type_source = "default_fallback"
                        logger.warning(
                            f"⚠️  Fallback TargetPredictabilityScore: Could not derive task_type from target '{target}' "
                            f"(error: {e_task}), defaulting to REGRESSION. This may be incorrect for classification targets."
                        )
                    
                    result_cs = TargetPredictabilityScore(
                        target=target,
                        target_column=target,  # Use target as fallback
                        task_type=task_type,
                        auc=-999.0,  # Invalid sentinel (consistent)
                        std_score=-999.0,  # Invalid sentinel (consistent)
                        mean_importance=-999.0,  # Invalid sentinel (consistent)
                        consistency=-999.0,  # Invalid sentinel (consistent)
                        n_models=0,
                        model_scores={},
                        composite_score=-999.0,  # Invalid sentinel (consistent, not 0.0)
                        status="ERROR",
                        leakage_flag="ERROR"
                    )
                
                # Store cross-sectional result FIRST (needed for gating symbol-specific)
                skip_statuses = [
                    "LEAKAGE_UNRESOLVED", 
                    "LEAKAGE_UNRESOLVED_MAX_RETRIES",
                    "SUSPICIOUS",
                    "SUSPICIOUS_STRONG",
                    "ERROR"  # Include ERROR status in skip_statuses
                ]
                
                # Track CS result (even if failed) for decision computation
                if result_cs:
                    all_cs_results[target] = result_cs
                
                # Handle case where result_cs might be None (shouldn't happen, but defensive)
                if result_cs is None:
                    logger.warning(f"  ⚠️  Cross-sectional evaluation returned None for {target}, treating as failed")
                    # NOTE: TargetPredictabilityScore is already imported at top of file (line 26)
                    # Derive task_type from context (don't guess silently)
                    try:
                        task_type = TaskType.from_target_column(target) if target else TaskType.REGRESSION
                        task_type_source = "derived_from_target"
                    except Exception as e_task:
                        task_type = TaskType.REGRESSION  # Default fallback
                        task_type_source = "default_fallback"
                        logger.warning(
                            f"⚠️  Fallback TargetPredictabilityScore: Could not derive task_type from target '{target}' "
                            f"(error: {e_task}), defaulting to REGRESSION. This may be incorrect for classification targets."
                        )
                    
                    result_cs = TargetPredictabilityScore(
                        target=target,
                        target_column=target,  # Use target as fallback
                        task_type=task_type,
                        auc=-999.0,  # Invalid sentinel (consistent)
                        std_score=-999.0,  # Invalid sentinel (consistent)
                        mean_importance=-999.0,  # Invalid sentinel (consistent)
                        consistency=-999.0,  # Invalid sentinel (consistent)
                        n_models=0,
                        model_scores={},
                        composite_score=-999.0,  # Invalid sentinel (consistent, not 0.0)
                        status="ERROR",
                        leakage_flag="ERROR"
                    )
                    all_cs_results[target] = result_cs
                
                cs_succeeded = result_cs.auc != -999.0 and result_cs.status not in skip_statuses
                if cs_succeeded:
                    results_cs.append(result_cs)
                    results.append(result_cs)  # Backward compatibility
                else:
                    reason = result_cs.status if result_cs.status in skip_statuses else (result_cs.leakage_flag if result_cs.leakage_flag != "OK" else "degenerate/failed")
                    if result_cs.status in ["SUSPICIOUS", "SUSPICIOUS_STRONG"]:
                        logger.warning(f"  ⚠️  Excluded {target} CROSS_SECTIONAL ({reason}) - High score suggests structural leakage")
                    else:
                        logger.info(f"  Skipped {target} CROSS_SECTIONAL ({reason})")
                
                # View B: Symbol-specific evaluation (if enabled)
                result_sym_dict = {}
                if enable_symbol_specific:
                    logger.info(f"  View B: SYMBOL_SPECIFIC (evaluating {len(symbols)} symbols)")
                    
                    # Always evaluate symbol-specific, even if cross-sectional failed
                    # Some targets may work symbol-specifically even if cross-sectional fails
                    # Routing logic needs symbol-specific results to make decisions (e.g., "SYMBOL_SPECIFIC only: weak CS but some symbols work")
                    if not cs_succeeded:
                        logger.info(f"  ℹ️  Cross-sectional failed for {target} (auc={result_cs.auc}, status={result_cs.status}), but evaluating symbol-specific anyway (target may work per-symbol)")
                    else:
                        logger.info(f"  ✅ Cross-sectional succeeded for {target}, proceeding with symbol-specific evaluation")
                    
                    # Track per-symbol skip reasons and diagnostics (local to this target evaluation)
                    local_symbol_skip_reasons = {}  # {symbol: {reason, n_rows, n_train, n_val, n_pos_train, n_neg_train, n_pos_val, n_neg_val}}
                    
                    for symbol in symbols:
                        logger.info(f"    Evaluating {target} for symbol {symbol}...")
                        try:
                            if auto_rerun_enabled and _AUTOFIX_AVAILABLE:
                                # FIX: Use signature check instead of try/except TypeError
                                if _AUTOFIX_HAS_VIEW_PARAM:
                                    result_sym = evaluate_target_with_autofix(
                                        target=target,
                                        target_config=target_config,
                                        symbols=[symbol],
                                        data_dir=data_dir,
                                        model_families=model_families,
                                        multi_model_config=multi_model_config,
                                        output_dir=output_dir,
                                        min_cs=1,
                                        max_cs_samples=max_cs_samples,
                                        max_rows_per_symbol=max_rows_per_symbol,
                                        max_reruns=max_reruns,
                                        rerun_on_perfect_train_acc=rerun_on_perfect_train_acc,
                                        rerun_on_high_auc_only=rerun_on_high_auc_only,
                                        explicit_interval=explicit_interval,
                                        experiment_config=experiment_config,
                                        view=View.SYMBOL_SPECIFIC,
                                        symbol=symbol,
                                        registry=registry,
                                        coverage_breakdowns_dict=coverage_breakdowns_dict,
                                    )
                                else:
                                    result_sym = evaluate_target_with_autofix(
                                        target=target,
                                        target_config=target_config,
                                        symbols=[symbol],
                                        data_dir=data_dir,
                                        model_families=model_families,
                                        multi_model_config=multi_model_config,
                                        output_dir=output_dir,
                                        min_cs=1,
                                        max_cs_samples=max_cs_samples,
                                        max_rows_per_symbol=max_rows_per_symbol,
                                        max_reruns=max_reruns,
                                        rerun_on_perfect_train_acc=rerun_on_perfect_train_acc,
                                        rerun_on_high_auc_only=rerun_on_high_auc_only,
                                        explicit_interval=explicit_interval,
                                        experiment_config=experiment_config,
                                        registry=registry,
                                        coverage_breakdowns_dict=coverage_breakdowns_dict,
                                    )
                            else:
                                result_sym = evaluate_target_predictability(
                                    target=target,
                                    target_config=target_config,
                                    symbols=[symbol],  # Single symbol
                                    data_dir=data_dir,
                                    model_families=model_families,
                                    multi_model_config=multi_model_config,
                                    output_dir=output_dir,
                                    min_cs=1,  # Single symbol, min_cs=1
                                    max_cs_samples=max_cs_samples,
                                    max_rows_per_symbol=max_rows_per_symbol,
                                    explicit_interval=explicit_interval,
                                    experiment_config=experiment_config,
                                    view=View.SYMBOL_SPECIFIC,
                                    symbol=symbol,
                                    run_identity=run_identity,  # NEW: Pass RunIdentity
                                    registry=registry,  # NEW: Pass same instance
                                    coverage_breakdowns_dict=coverage_breakdowns_dict,  # NEW: Collect breakdowns (SS won't be stored)
                                )
                            
                            # Gate: Skip if result is degenerate (auc = -999)
                            if result_sym.auc == -999.0:
                                skip_reason = result_sym.status if result_sym.status != "OK" else "degenerate"
                                logger.warning(f"    ⚠️  Skipped {target} for symbol {symbol}: {skip_reason} (auc=-999.0, status={result_sym.status})")
                                local_symbol_skip_reasons[symbol] = {
                                    'reason': skip_reason,
                                    'status': result_sym.status,
                                    'leakage_flag': result_sym.leakage_flag,
                                    'auc': result_sym.auc
                                }
                                continue
                            
                            logger.info(f"    ✅ {target} for {symbol}: auc={result_sym.auc:.4f}, status={result_sym.status}")
                            result_sym_dict[symbol] = result_sym
                        except Exception as e:
                            skip_reason = f"exception: {type(e).__name__}"
                            logger.error(f"    ❌ Failed to evaluate {target} for symbol {symbol}: {e}", exc_info=True)
                            local_symbol_skip_reasons[symbol] = {
                                'reason': skip_reason,
                                'error': str(e),
                                'error_type': type(e).__name__
                            }
                            continue
                    
                    # Log summary of skip reasons
                    if local_symbol_skip_reasons:
                        logger.warning(f"  📋 Symbol-specific skip reasons for {target}: {len(local_symbol_skip_reasons)}/{len(symbols)} symbols skipped")
                        # DETERMINISM: Use sorted_items for deterministic iteration order
                        for sym, skip_info in sorted_items(local_symbol_skip_reasons):
                            reason = skip_info.get('reason', 'unknown')
                            logger.debug(f"    {sym}: {reason}")
                    
                    # Store skip reasons for routing decisions (use global symbol_skip_reasons dict)
                    if local_symbol_skip_reasons:
                        symbol_skip_reasons[target] = local_symbol_skip_reasons
                    
                    logger.info(f"  📊 Symbol-specific results for {target}: {len(result_sym_dict)}/{len(symbols)} symbols succeeded")
                
                # View C: LOSO evaluation (optional, if enabled)
                result_loso_dict = {}
                if enable_loso:
                    logger.info(f"  View C: LOSO (evaluating {len(symbols)} symbols)")
                    for symbol in symbols:
                        try:
                            result_loso_sym = evaluate_target_predictability(
                                target=target,
                                target_config=target_config,
                                symbols=symbols,  # All symbols for training
                                data_dir=data_dir,
                                model_families=model_families,
                                multi_model_config=multi_model_config,
                                output_dir=output_dir,
                                min_cs=min_cs,
                                max_cs_samples=max_cs_samples,
                                max_rows_per_symbol=max_rows_per_symbol,
                                explicit_interval=explicit_interval,
                                experiment_config=experiment_config,
                                view="LOSO",
                                symbol=symbol,
                                run_identity=run_identity,  # NEW: Pass RunIdentity
                                registry=registry,  # NEW: Pass same instance
                                coverage_breakdowns_dict=coverage_breakdowns_dict,  # NEW: Collect breakdowns (LOSO won't be stored)
                            )
                            result_loso_dict[symbol] = result_loso_sym
                        except Exception as e:
                            logger.warning(f"    Failed LOSO evaluation for {target} on symbol {symbol}: {e}")
                            continue
                
                # Store symbol-specific results
                if enable_symbol_specific and result_sym_dict:
                    if target not in results_sym:
                        results_sym[target] = {}
                    
                    stored_count = 0
                    # DETERMINISM_CRITICAL: Symbol aggregation order must be deterministic
                    for symbol, result_sym in sorted_items(result_sym_dict):
                        if result_sym.auc != -999.0 and result_sym.status not in skip_statuses:
                            results_sym[target][symbol] = result_sym
                            stored_count += 1
                        else:
                            reason = result_sym.status if result_sym.status in skip_statuses else "degenerate/failed"
                            logger.warning(f"    ⚠️  Filtered out {target} SYMBOL_SPECIFIC ({symbol}): {reason} (auc={result_sym.auc})")
                            # Add to skip reasons if not already there (use global symbol_skip_reasons dict)
                            if target not in symbol_skip_reasons:
                                symbol_skip_reasons[target] = {}
                            if symbol not in symbol_skip_reasons[target]:
                                symbol_skip_reasons[target][symbol] = {
                                    'reason': reason,
                                    'status': result_sym.status,
                                    'auc': result_sym.auc
                                }
                    if stored_count > 0:
                        logger.info(f"  ✅ Stored {stored_count} symbol-specific results for {target}")
                    elif len(result_sym_dict) > 0:
                        logger.warning(f"  ⚠️  All {len(result_sym_dict)} symbol-specific results for {target} were filtered out")
                    
                    # Save routing decision immediately after target evaluation (incremental)
                    if output_dir:
                        from TRAINING.ranking.target_routing import (
                            _compute_single_target_routing_decision, _save_single_target_decision
                        )
                        target_sym_results = results_sym.get(target, {})
                        target_skip_reasons = symbol_skip_reasons.get(target, {}) if symbol_skip_reasons else {}
                        decision = _compute_single_target_routing_decision(
                            target=target,
                            result_cs=result_cs if cs_succeeded else None,
                            sym_results=target_sym_results,
                            symbol_skip_reasons=target_skip_reasons,
                            experiment_config=experiment_config,
                            total_symbols=len(symbols)  # FIX: Pass total universe size for SS gate
                        )
                        _save_single_target_decision(target, decision, output_dir)

                # Store LOSO results
                # DETERMINISM_CRITICAL: Symbol aggregation order must be deterministic
                if enable_loso:
                    if target not in results_loso:
                        results_loso[target] = {}
                    for symbol, result_loso_sym in sorted_items(result_loso_dict):
                        if result_loso_sym.auc != -999.0 and result_loso_sym.status not in skip_statuses:
                            results_loso[target][symbol] = result_loso_sym
            
            except Exception as e:
                logger.exception(f"  Failed to evaluate {target}: {e}")  # Better error logging with traceback
                # Continue with next target
    
    # Compute routing decisions and aggregate symbol-specific results
    logger.info("=" * 60)
    logger.info("DUAL-VIEW TARGET RANKING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Cross-sectional targets evaluated: {len(results_cs)}")
    if enable_symbol_specific:
        total_sym_results = sum(len(sym_results) for sym_results in results_sym.values())
        logger.info(f"Symbol-specific evaluations: {total_sym_results} (across {len(results_sym)} targets)")
    
    # Compute routing decisions for each target
    # Use global symbol_skip_reasons dict (already initialized at top of function)
    
    routing_decisions = _compute_target_routing_decisions(
        results_cs=results_cs,
        results_sym=results_sym,
        results_loso=results_loso if enable_loso else {},
        symbol_skip_reasons=symbol_skip_reasons,
        experiment_config=experiment_config,
        total_symbols=len(symbols)  # FIX: Pass total universe size for SS gate
    )
    
    # CRITICAL: Ensure ALL evaluated targets have routing decisions, even if they're not in routing_decisions
    # This handles cases where CS failed and all symbols failed (target won't be in routing_decisions)
    if output_dir:
        from TRAINING.ranking.target_routing import (
            _compute_single_target_routing_decision, _save_single_target_decision
        )
        for target in all_evaluated_targets:
            if target not in routing_decisions:
                # Target was evaluated but not in routing_decisions (CS failed + all symbols failed)
                # Compute and save decision anyway
                logger.debug(f"Computing routing decision for {target} (not in routing_decisions, likely all evaluations failed)")
                target_sym_results = results_sym.get(target, {})
                target_skip_reasons = symbol_skip_reasons.get(target, {}) if symbol_skip_reasons else {}
                # Find CS result even if it failed (from all_cs_results which includes failed ones)
                result_cs = all_cs_results.get(target)
                decision = _compute_single_target_routing_decision(
                    target=target,
                    result_cs=result_cs,  # Will be None if CS was never evaluated or failed
                    sym_results=target_sym_results,
                    symbol_skip_reasons=target_skip_reasons,
                    experiment_config=experiment_config,
                    total_symbols=len(symbols)  # FIX: Pass total universe size for SS gate
                )
                _save_single_target_decision(target, decision, output_dir)
                # Also add to routing_decisions so it's in the global file
                routing_decisions[target] = decision
    
    # Log routing summary
    cs_only = sum(1 for r in routing_decisions.values() if r.get('route') == View.CROSS_SECTIONAL.value)
    sym_only = sum(1 for r in routing_decisions.values() if r.get('route') == View.SYMBOL_SPECIFIC.value)
    both = sum(1 for r in routing_decisions.values() if r.get('route') == 'BOTH')
    blocked = sum(1 for r in routing_decisions.values() if r.get('route') == 'BLOCKED')
    logger.info(f"Routing decisions: {cs_only} CROSS_SECTIONAL, {sym_only} SYMBOL_SPECIFIC, {both} BOTH, {blocked} BLOCKED")
    
    # Sort cross-sectional results by composite score (descending) for backward compatibility
    # DETERMINISM_CRITICAL: Tie-breaker required for score-based sorting
    results.sort(key=lambda r: (-r.composite_score, r.target), reverse=False)
    
    # === DUAL RANKING: Compute rank_delta after sorting ===
    # rank_delta = rank_screen - rank_strict (positive = screen ranks higher)
    # Compute ranks for both screen and strict scores
    # Initialize valid_results to handle empty results case
    valid_results = []
    if results:
        # Filter to only valid targets for ranking
        valid_results = [r for r in results if getattr(r, 'valid_for_ranking', True)]
        invalid_results = [r for r in results if not getattr(r, 'valid_for_ranking', True)]
        
        if invalid_results:
            # Add dev_mode indicator
            dev_mode_indicator = ""
            try:
                from CONFIG.dev_mode import get_dev_mode
                if get_dev_mode():
                    dev_mode_indicator = " [DEV_MODE]"
            except Exception:
                pass
            logger.warning(f"Excluding {len(invalid_results)} targets from ranking due to eligibility gates{dev_mode_indicator}:")
            for r in invalid_results:
                invalid_reasons = getattr(r, 'invalid_reasons', [])
                logger.warning(f"  {r.target}: {', '.join(invalid_reasons) if invalid_reasons else 'UNKNOWN_REASON'}")
        
        # Sort by screen score (composite_score is screen score) - only valid targets
        # DETERMINISM_CRITICAL: Tie-breaker required for score-based sorting
        screen_ranked = sorted(
            valid_results,
            key=lambda r: (
                -(r.score_screen if (hasattr(r, 'score_screen') and r.score_screen is not None) else r.composite_score),
                r.target  # Tie-breaker: ascending target name for equal scores
            ),
            reverse=False  # Note: negative score means reverse=False gives desc order
        )
        screen_ranks = {r.target: idx for idx, r in enumerate(screen_ranked)}
        
        # Sort by strict score - only valid targets
        # DETERMINISM_CRITICAL: Tie-breaker required for score-based sorting
        strict_ranked = sorted(
            [r for r in valid_results if hasattr(r, 'score_strict') and r.score_strict is not None],
            key=lambda r: (-r.score_strict, r.target),  # Desc score, asc target name for ties
            reverse=False  # Note: negative score means reverse=False gives desc order
        )
        strict_ranks = {r.target: idx for idx, r in enumerate(strict_ranked)}
        
        # Compute rank_delta and strict_viability_flag for each result
        # strict_viability_flag: True if target would be in top N when ranked by strict score
        top_n_for_viability = top_n if top_n is not None and top_n > 0 else len(results)
        for r in results:
            screen_rank = screen_ranks.get(r.target, len(results))
            strict_rank = strict_ranks.get(r.target, len(results))
            r.rank_delta = screen_rank - strict_rank
            
            # Set strict_viability_flag: True if strict rank is within top N
            if hasattr(r, 'score_strict') and r.score_strict is not None:
                r.strict_viability_flag = strict_rank < top_n_for_viability
            else:
                # If strict score not available, assume viable (backward compatibility)
                r.strict_viability_flag = True
    
    # Update results to only include valid targets (for downstream processing)
    results = valid_results
    
    # Apply top_n limit if specified (to cross-sectional results)
    if top_n is not None and top_n > 0:
        results = results[:top_n]
        logger.info(f"Returning top {len(results)} targets (cross-sectional, after eligibility filtering)")
    
    # Save rankings if output_dir provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_rankings(results, output_dir)
        
        # Save dual-view results and routing decisions
        _save_dual_view_rankings(
            results_cs=results_cs,
            results_sym=results_sym,
            results_loso=results_loso if enable_loso else {},
            routing_decisions=routing_decisions,
            output_dir=output_dir
        )
        
        # Generate metrics rollups after all targets are evaluated
        try:
            from TRAINING.orchestration.utils.reproducibility_tracker import ReproducibilityTracker
            from datetime import datetime
            
            # Find the REPRODUCIBILITY directory (could be in output_dir or parent)
            repro_dir = output_dir / "REPRODUCIBILITY"
            if not repro_dir.exists() and output_dir.parent.exists():
                repro_dir = output_dir.parent / "REPRODUCIBILITY"
            
            if repro_dir.exists():
                # Use output_dir parent as base (where RESULTS/runs/ typically is)
                base_dir = output_dir.parent if (output_dir / "REPRODUCIBILITY").exists() else output_dir
                tracker = ReproducibilityTracker(output_dir=base_dir)
                # Generate run_id from RunIdentity if available, otherwise use unstable run_id
                run_id = None
                if run_identity is not None:
                    try:
                        from TRAINING.orchestration.utils.manifest import derive_run_id_from_identity
                        run_id = derive_run_id_from_identity(run_identity=run_identity)
                    except (ValueError, AttributeError):
                        pass  # Fall through to unstable run_id
                
                if run_id is None:
                    from TRAINING.orchestration.utils.manifest import derive_unstable_run_id, generate_run_instance_id
                    run_id = derive_unstable_run_id(generate_run_instance_id())
                
                tracker.generate_metrics_rollups(stage=Stage.TARGET_RANKING, run_id=run_id)
                logger.debug("✅ Generated metrics rollups for TARGET_RANKING")
        except Exception as e:
            logger.debug(f"Failed to generate metrics rollups: {e}")
    
    return results

