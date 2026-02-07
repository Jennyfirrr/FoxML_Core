# Configuration Reference

This document provides a complete reference for all LIVE_TRADING configuration options.

---

## Configuration Files

| File | Purpose |
|------|---------|
| `CONFIG/live_trading/live_trading.yaml` | Main configuration |
| `CONFIG/live_trading/symbols.yaml` | Symbol universe |

---

## Main Configuration

### Complete Example

```yaml
# CONFIG/live_trading/live_trading.yaml

live_trading:
  #############################################
  # HORIZONS
  #############################################
  horizons:
    - "5m"
    - "10m"
    - "15m"
    - "30m"
    - "60m"
    - "1d"

  #############################################
  # MODEL FAMILIES
  #############################################
  families:
    - "lightgbm"
    - "xgboost"
    - "catboost"
    - "ridge"
    - "mlp"
    - "lstm"

  #############################################
  # BLENDING
  #############################################
  blending:
    # Ridge regularization
    ridge_lambda: 0.15

    # Temperature compression for short horizons
    temperature:
      "5m": 0.75
      "10m": 0.85
      "15m": 0.90
      "30m": 1.0
      "60m": 1.0
      "1d": 1.0

    # Minimum samples for correlation estimation
    min_correlation_samples: 50

  #############################################
  # COST MODEL
  #############################################
  cost_model:
    # Spread penalty coefficient
    k1: 1.0

    # Volatility timing coefficient
    k2: 0.15

    # Market impact coefficient
    k3: 1.0

    # Impact model exponent (sqrt = 0.5)
    impact_exponent: 0.5

  #############################################
  # ARBITRATION
  #############################################
  arbitration:
    # Apply sqrt(h/5) horizon penalty
    horizon_penalty: true

    # Minimum net score to trade
    min_net_score: 0.001

    # Additional reserve buffer (bps)
    reserve_bps: 2.0

  #############################################
  # PREDICTION
  #############################################
  prediction:
    # Z-score standardization
    z_clip: 3.0
    rolling_window_days: 10
    min_samples: 100

    # Freshness decay
    freshness_tau:
      "5m": 150     # seconds
      "10m": 300
      "15m": 450
      "30m": 900
      "60m": 1800
      "1d": 14400

    # Confidence thresholds
    min_confidence: 0.3

  #############################################
  # GATING
  #############################################
  gating:
    # Spread gate
    max_spread_bps: 12
    max_quote_age_ms: 200

    # Barrier gate
    barrier_enabled: true
    g_min: 0.2
    gamma: 1.0
    delta: 0.5

    # Peak/valley thresholds
    peak_threshold: 0.6
    valley_threshold: 0.55

    # Cost sanity
    max_slippage_ratio: 0.6

  #############################################
  # SIZING
  #############################################
  sizing:
    # Volatility scaling
    z_max: 3.0
    max_weight: 0.05  # 5% per position

    # Gross exposure
    gross_target: 0.50  # 50%

    # No-trade band
    no_trade_band: 0.008  # 80 bps

    # Share rounding
    round_to_lot: 1  # Round to nearest share

  #############################################
  # RISK MANAGEMENT
  #############################################
  risk:
    # Kill switches
    max_daily_loss_pct: 0.02   # 2%
    max_drawdown_pct: 0.10     # 10%
    max_position_pct: 0.20     # 20%

    # Exposure limits
    max_gross_exposure: 0.50   # 50%
    max_net_exposure: 0.30     # 30%
    max_sector_exposure: 0.30  # 30%

    # Recovery
    cool_off_minutes: 30
    gradual_recovery: true

  #############################################
  # ONLINE LEARNING
  #############################################
  online_learning:
    enabled: true
    algorithm: "exp3ix"

    # Bandit parameters
    gamma: 0.05
    eta_max: 0.07
    eta_auto: true

    # Weight blending
    blend_alpha: 0.3  # 30% bandit, 70% ridge

    # Warm start
    warm_start_steps: 500

    # Persistence
    save_interval: 100
    state_file: "state/bandit_state.json"

  #############################################
  # BROKERS
  #############################################
  brokers:
    paper:
      initial_cash: 100000
      slippage_bps: 5
      fee_bps: 1

    ibkr:
      host: "127.0.0.1"
      port: 7497
      client_id: 1

    alpaca:
      paper_trading: true
      # API keys from environment

  #############################################
  # DATA PROVIDERS
  #############################################
  data_providers:
    simulated:
      volatility: 0.02
      drift: 0.0001

    cached:
      cache_dir: "cache/market_data"
      ttl_seconds: 60

  #############################################
  # ENGINE
  #############################################
  engine:
    # State management
    state_path: "state/engine_state.json"
    history_path: "state/history.json"
    save_state: true
    save_history: true

    # Pipeline configuration
    default_target: "ret_5m"
    include_trace: true  # Include full pipeline trace in decisions

    # Gates enabled
    enable_barrier_gate: true
    enable_spread_gate: true

    # Trade safety (C4: duplicate prevention)
    trade_cooldown_seconds: 5.0

    # Position reconciliation (C2: broker sync)
    reconciliation_interval_cycles: 100
    reconciliation_mode: "warn"  # "strict", "warn", "auto_sync"

  #############################################
  # ONLINE LEARNING (CILS)
  #############################################
  online_learning:
    enabled: true
    blend_ratio: 0.3        # 30% bandit weights, 70% static
    enable_horizon_discovery: true
    discovery_interval_cycles: 1000

  bandit:
    state_dir: "state/cils"
    save_interval: 100

  #############################################
  # LOGGING
  #############################################
  logging:
    level: "INFO"
    log_dir: "logs"
    log_trades: true
    log_predictions: false

  #############################################
  # OBSERVABILITY
  #############################################
  observability:
    events_enabled: true
    metrics_enabled: true
    prometheus_port: 9090

  #############################################
  # ALERTING
  #############################################
  alerting:
    enabled: true
    channels:
      - slack
      - email

    # Alert routing
    kill_switch: ["slack", "email"]
    warning: ["slack"]
    info: ["email"]
```

---

## Parameter Reference

### Horizons

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `horizons` | list | ["5m"..."1d"] | Prediction horizons to use |

### Blending

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ridge_lambda` | float | 0.15 | Ridge regularization parameter |
| `temperature.*` | float | 0.75-1.0 | Temperature per horizon |
| `min_correlation_samples` | int | 50 | Minimum samples for correlation |

### Cost Model

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k1` | float | 1.0 | Spread penalty coefficient |
| `k2` | float | 0.15 | Volatility timing coefficient |
| `k3` | float | 1.0 | Market impact coefficient |
| `impact_exponent` | float | 0.5 | Impact model exponent |

### Gating

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_spread_bps` | float | 12 | Maximum spread (bps) |
| `max_quote_age_ms` | int | 200 | Maximum quote age (ms) |
| `barrier_enabled` | bool | true | Enable barrier gates |
| `g_min` | float | 0.2 | Minimum gate factor |
| `gamma` | float | 1.0 | Peak penalty exponent |
| `delta` | float | 0.5 | Valley reward exponent |
| `peak_threshold` | float | 0.6 | Peak probability threshold |

### Sizing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `z_max` | float | 3.0 | Maximum Z-score |
| `max_weight` | float | 0.05 | Maximum position weight |
| `gross_target` | float | 0.50 | Target gross exposure |
| `no_trade_band` | float | 0.008 | No-trade band (80 bps) |

### Risk

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_daily_loss_pct` | float | 0.02 | Maximum daily loss (2%) |
| `max_drawdown_pct` | float | 0.10 | Maximum drawdown (10%) |
| `max_position_pct` | float | 0.20 | Maximum position (20%) |
| `max_gross_exposure` | float | 0.50 | Maximum gross exposure |
| `cool_off_minutes` | int | 30 | Cool-off after kill switch |

### Online Learning

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | true | Enable online learning |
| `gamma` | float | 0.05 | Exploration rate |
| `eta_max` | float | 0.07 | Maximum learning rate |
| `blend_alpha` | float | 0.3 | Bandit weight blend |
| `warm_start_steps` | int | 500 | Warm-up period |

---

## Symbol Configuration

### symbols.yaml

```yaml
# CONFIG/live_trading/symbols.yaml

symbols:
  # ETF universe
  universe:
    - SPY
    - QQQ
    - IWM
    - DIA
    - VTI
    - VOO

  # Sector ETFs
  sectors:
    - XLF   # Financials
    - XLK   # Technology
    - XLE   # Energy
    - XLV   # Healthcare
    - XLI   # Industrials
    - XLC   # Communication
    - XLY   # Consumer Discretionary
    - XLP   # Consumer Staples
    - XLB   # Materials
    - XLU   # Utilities
    - XLRE  # Real Estate

  # Individual stocks
  stocks:
    - AAPL
    - MSFT
    - GOOGL
    - AMZN
    - META
    - NVDA
    - TSLA
    - JPM
    - V
    - JNJ

  # Default symbols (used if none specified)
  default:
    - SPY
    - QQQ
    - AAPL
    - MSFT

  # Excluded symbols
  excluded:
    - []  # Add symbols to exclude
```

---

## CLI Configuration

### CLIConfig

```python
@dataclass
class CLIConfig:
    # Run identification
    run_id: Optional[str] = None
    run_root: Optional[str] = None

    # Broker selection
    broker: str = "paper"

    # Symbol selection
    symbols: List[str] = field(default_factory=lambda: ["SPY", "QQQ"])
    symbols_file: Optional[str] = None

    # Cycle settings
    cycle_interval: int = 60  # seconds
    max_cycles: int = 0       # 0 = unlimited

    # Mode flags
    dry_run: bool = False
    no_state: bool = False

    # Logging
    log_level: str = "INFO"
    log_dir: str = "logs"
```

### CLI Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--run-id` | str | Required | TRAINING run ID |
| `--run-root` | str | None | Full path to run directory |
| `--broker` | str | "paper" | Broker to use |
| `--symbols` | list | ["SPY", "QQQ"] | Symbols to trade |
| `--symbols-file` | str | None | YAML file with symbols |
| `--interval` | int | 60 | Cycle interval (seconds) |
| `--max-cycles` | int | 0 | Maximum cycles (0=unlimited) |
| `--dry-run` | bool | False | Use simulated data |
| `--no-state` | bool | False | Don't persist state |
| `--log-level` | str | "INFO" | Logging level |
| `--log-dir` | str | "logs" | Log directory |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ALPACA_API_KEY` | Alpaca API key |
| `ALPACA_API_SECRET` | Alpaca API secret |
| `IBKR_HOST` | IBKR Gateway host |
| `IBKR_PORT` | IBKR Gateway port |
| `SLACK_WEBHOOK_URL` | Slack webhook for alerts |
| `SMTP_SERVER` | SMTP server for email alerts |

---

## Config Access Patterns

### Using get_cfg()

```python
from CONFIG.config_loader import get_cfg

# Get with default
ridge_lambda = get_cfg("live_trading.blending.ridge_lambda", default=0.15)

# Nested access
max_spread = get_cfg("live_trading.gating.max_spread_bps", default=12)

# List access
horizons = get_cfg("live_trading.horizons", default=["5m", "15m", "60m"])
```

### Using DEFAULT_CONFIG

```python
from LIVE_TRADING.common.constants import DEFAULT_CONFIG

# Fallback pattern
value = get_cfg("live_trading.sizing.z_max", default=DEFAULT_CONFIG["sizing"]["z_max"])
```

---

## Validation

### Config Validation

```python
from LIVE_TRADING.cli.config import validate_config

errors = validate_config(config)
if errors:
    for error in errors:
        print(f"Config error: {error}")
    sys.exit(1)
```

### Validation Rules

| Field | Rule |
|-------|------|
| `horizons` | Non-empty list of valid horizon strings |
| `ridge_lambda` | 0 < value < 1 |
| `temperature.*` | 0 < value <= 1 |
| `k1, k2, k3` | value >= 0 |
| `max_spread_bps` | value > 0 |
| `z_max` | value > 0 |
| `max_weight` | 0 < value <= 1 |
| `gross_target` | 0 < value <= 1 |
| `max_daily_loss_pct` | 0 < value < 1 |

---

## Related Documentation

- [../architecture/SYSTEM_ARCHITECTURE.md](../architecture/SYSTEM_ARCHITECTURE.md) - Architecture overview
- [../architecture/MATHEMATICAL_FORMULAS.md](../architecture/MATHEMATICAL_FORMULAS.md) - Formula parameters
- [PLAN_REFERENCES.md](PLAN_REFERENCES.md) - Source plans
