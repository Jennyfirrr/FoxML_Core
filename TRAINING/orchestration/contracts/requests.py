# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Request dataclasses for pipeline APIs.

Consolidates function parameters into typed request objects to:
1. Reduce function parameter counts (from 17 to 1)
2. Enable type checking and validation
3. Provide clear documentation of required vs optional params

API-004: FeatureSelectionRequest - for select_features_for_target()
API-005: RankingRequest - for rank_targets()
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from TRAINING.common.feature_registry import FeatureRegistry
    from TRAINING.orchestration.utils.scope_resolution import View


@dataclass
class FeatureSelectionRequest:
    """
    Request object for select_features_for_target().

    API-004: Consolidates 17 parameters into a typed dataclass.

    Required fields:
        target_column: The target variable to select features for
        symbols: List of symbols to process
        data_dir: Directory containing input data

    Optional config (precedence: typed config > dict config > defaults):
        config: Typed FeatureSelectionConfig object
        multi_model_config: Legacy dict config (deprecated)
        model_families_config: Model family configurations

    Optional behavior:
        max_samples_per_symbol: Limit samples per symbol (default from config)
        top_n: Number of features to select (default from config)
        output_dir: Directory for output artifacts
        view: View type (CROSS_SECTIONAL or SYMBOL_SPECIFIC)
        symbol: Required for SYMBOL_SPECIFIC view
        force_refresh: Bypass cache and re-run
        explicit_interval: Override data interval

    Identity tracking:
        run_identity: Finalized RunIdentity for artifact storage
        universe_sig: Universe signature from SST
        experiment_config: Experiment configuration for context

    Example:
        >>> request = FeatureSelectionRequest(
        ...     target_column="fwd_ret_60m",
        ...     symbols=["AAPL", "GOOGL"],
        ...     data_dir=Path("/data"),
        ...     top_n=50
        ... )
        >>> features, summary = select_features_for_target(request)
    """
    # Required
    target_column: str
    symbols: List[str]
    data_dir: Path

    # Config (precedence: typed config > dict config)
    config: Optional[Any] = None  # FeatureSelectionConfig when available
    multi_model_config: Optional[Dict[str, Any]] = None
    model_families_config: Optional[Dict[str, Dict[str, Any]]] = None

    # Optional behavior
    max_samples_per_symbol: Optional[int] = None
    top_n: Optional[int] = None
    output_dir: Optional[Path] = None
    view: Optional[Any] = None  # View enum or string
    symbol: Optional[str] = None
    force_refresh: bool = False
    explicit_interval: Optional[Any] = None  # int, str, or None

    # Identity tracking
    run_identity: Optional[Any] = None  # RunIdentity
    universe_sig: Optional[str] = None
    experiment_config: Optional[Any] = None

    def __post_init__(self):
        """Validate request after initialization."""
        if not self.target_column:
            raise ValueError("target_column is required")
        if not self.symbols:
            raise ValueError("symbols cannot be empty")
        if not self.data_dir:
            raise ValueError("data_dir is required")

        # Validate view/symbol consistency
        if self.view is not None:
            view_str = str(self.view).upper()
            if view_str == "SYMBOL_SPECIFIC" and self.symbol is None:
                raise ValueError("symbol is required for SYMBOL_SPECIFIC view")

    def to_kwargs(self) -> Dict[str, Any]:
        """
        Convert to kwargs dict for legacy function signature.

        Use during transition period when calling select_features_for_target()
        with the old signature.
        """
        return {
            'target_column': self.target_column,
            'symbols': self.symbols,
            'data_dir': self.data_dir,
            'model_families_config': self.model_families_config,
            'multi_model_config': self.multi_model_config,
            'max_samples_per_symbol': self.max_samples_per_symbol,
            'top_n': self.top_n,
            'output_dir': self.output_dir,
            'feature_selection_config': self.config,
            'explicit_interval': self.explicit_interval,
            'experiment_config': self.experiment_config,
            'view': self.view,
            'symbol': self.symbol,
            'force_refresh': self.force_refresh,
            'universe_sig': self.universe_sig,
            'run_identity': self.run_identity,
        }


@dataclass
class RankingRequest:
    """
    Request object for rank_targets().

    API-005: Consolidates 17 parameters into a typed dataclass.

    Required fields:
        targets: Target definitions dict (target_name -> target_config)
        symbols: List of symbols to evaluate
        data_dir: Directory containing input data
        model_families: List of model families to use for ranking

    Optional config:
        target_ranking_config: Typed TargetRankingConfig object
        multi_model_config: Dict config for multi-model ranking
        registry: Feature registry instance

    Optional limits:
        top_n: Number of top targets to return
        max_targets_to_evaluate: Maximum targets to evaluate
        min_cs: Minimum cross-sectional samples
        max_cs_samples: Maximum cross-sectional samples
        max_rows_per_symbol: Maximum rows per symbol

    Optional behavior:
        output_dir: Directory for output artifacts
        explicit_interval: Override data interval

    Identity tracking:
        run_identity: Finalized RunIdentity for artifact storage
        experiment_config: Experiment configuration for context

    Example:
        >>> request = RankingRequest(
        ...     targets={"fwd_ret_60m": {...}, "fwd_ret_120m": {...}},
        ...     symbols=["AAPL", "GOOGL"],
        ...     data_dir=Path("/data"),
        ...     model_families=["lightgbm", "xgboost"],
        ...     top_n=10
        ... )
        >>> ranked_targets = rank_targets(**request.to_kwargs())
    """
    # Required
    targets: Dict[str, Any]
    symbols: List[str]
    data_dir: Path
    model_families: List[str]

    # Config
    target_ranking_config: Optional[Any] = None  # TargetRankingConfig when available
    multi_model_config: Optional[Dict[str, Any]] = None
    registry: Optional[Any] = None  # FeatureRegistry

    # Limits
    top_n: Optional[int] = None
    max_targets_to_evaluate: Optional[int] = None
    min_cs: Optional[int] = None
    max_cs_samples: Optional[int] = None
    max_rows_per_symbol: Optional[int] = None

    # Optional behavior
    output_dir: Optional[Path] = None
    explicit_interval: Optional[Any] = None

    # Identity tracking
    run_identity: Optional[Any] = None
    experiment_config: Optional[Any] = None

    def __post_init__(self):
        """Validate request after initialization."""
        if not self.targets:
            raise ValueError("targets cannot be empty")
        if not self.symbols:
            raise ValueError("symbols cannot be empty")
        if not self.data_dir:
            raise ValueError("data_dir is required")
        if not self.model_families:
            raise ValueError("model_families cannot be empty")

    def to_kwargs(self) -> Dict[str, Any]:
        """
        Convert to kwargs dict for rank_targets() function signature.

        Use during transition period when calling rank_targets()
        with the current signature.
        """
        return {
            'targets': self.targets,
            'symbols': self.symbols,
            'data_dir': self.data_dir,
            'model_families': self.model_families,
            'multi_model_config': self.multi_model_config,
            'output_dir': self.output_dir,
            'min_cs': self.min_cs,
            'max_cs_samples': self.max_cs_samples,
            'max_rows_per_symbol': self.max_rows_per_symbol,
            'top_n': self.top_n,
            'max_targets_to_evaluate': self.max_targets_to_evaluate,
            'target_ranking_config': self.target_ranking_config,
            'explicit_interval': self.explicit_interval,
            'experiment_config': self.experiment_config,
            'run_identity': self.run_identity,
            'registry': self.registry,
        }
