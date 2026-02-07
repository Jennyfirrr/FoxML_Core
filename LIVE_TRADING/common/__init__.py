"""
Common infrastructure for LIVE_TRADING.

Provides exceptions, constants, and shared types.
"""

from .clock import (
    Clock,
    SystemClock,
    SimulatedClock,
    OffsetClock,
    get_clock,
    set_clock,
    reset_clock,
)
from .constants import (
    DEFAULT_CONFIG,
    FAMILIES,
    HORIZON_MINUTES,
    HORIZONS,
    SEQUENTIAL_FAMILIES,
    TREE_FAMILIES,
)
from .exceptions import (
    BrokerError,
    GatingError,
    InferenceError,
    InsufficientFundsError,
    KillSwitchTriggered,
    LiveTradingError,
    ModelLoadError,
    OrderRejectedError,
    RiskError,
    SizingError,
    SpreadTooWideError,
    StaleDataError,
)
from .order import (
    Order,
    OrderBook,
    OrderFill,
    OrderSide,
    OrderStatus,
    OrderStateTransition,
    OrderType,
    TimeInForce,
)
from .persistence import (
    StateManager,
    StatePersistence,
    WALEntry,
    WriteAheadLog,
    compute_checksum,
    verify_checksum,
)
from .reconciliation import (
    CashDiscrepancy,
    DiscrepancyType,
    PositionDiscrepancy,
    ReconciliationError,
    ReconciliationMode,
    ReconciliationResult,
    Reconciler,
)
from .time_utils import (
    utc_now,
    ensure_utc,
    require_utc,
    parse_iso,
    to_iso,
    to_iso_z,
    timestamp_ms,
    from_timestamp_ms,
    trading_day,
    is_same_trading_day,
    market_open_utc,
    market_close_utc,
    validate_utc_params,
)

__all__ = [
    # Clock
    "Clock",
    "SystemClock",
    "SimulatedClock",
    "OffsetClock",
    "get_clock",
    "set_clock",
    "reset_clock",
    # Order management
    "Order",
    "OrderBook",
    "OrderFill",
    "OrderSide",
    "OrderStatus",
    "OrderStateTransition",
    "OrderType",
    "TimeInForce",
    # Time utilities
    "utc_now",
    "ensure_utc",
    "require_utc",
    "parse_iso",
    "to_iso",
    "to_iso_z",
    "timestamp_ms",
    "from_timestamp_ms",
    "trading_day",
    "is_same_trading_day",
    "market_open_utc",
    "market_close_utc",
    "validate_utc_params",
    # Exceptions
    "LiveTradingError",
    "BrokerError",
    "OrderRejectedError",
    "InsufficientFundsError",
    "ModelLoadError",
    "InferenceError",
    "GatingError",
    "SizingError",
    "RiskError",
    "KillSwitchTriggered",
    "StaleDataError",
    "SpreadTooWideError",
    # Persistence
    "StateManager",
    "StatePersistence",
    "WALEntry",
    "WriteAheadLog",
    "compute_checksum",
    "verify_checksum",
    # Reconciliation
    "CashDiscrepancy",
    "DiscrepancyType",
    "PositionDiscrepancy",
    "ReconciliationError",
    "ReconciliationMode",
    "ReconciliationResult",
    "Reconciler",
    # Constants
    "HORIZONS",
    "HORIZON_MINUTES",
    "FAMILIES",
    "SEQUENTIAL_FAMILIES",
    "TREE_FAMILIES",
    "DEFAULT_CONFIG",
]
