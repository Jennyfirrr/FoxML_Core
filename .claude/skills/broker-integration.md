# Broker Integration

Guidelines for implementing broker interfaces.

## Broker Protocol

All brokers implement this Protocol:

```python
class Broker(Protocol):
    def submit_order(self, symbol: str, side: str, qty: float,
                     order_type: str = "market") -> dict[str, Any]: ...
    def cancel_order(self, order_id: str) -> dict[str, Any]: ...
    def get_positions(self) -> dict[str, float]: ...
    def get_cash(self) -> float: ...
    def get_fills(self, since: datetime | None = None) -> list[dict]: ...
    def now(self) -> datetime: ...
```

## Implementations

### Paper Broker
- Simulated execution with configurable slippage (5-10 bps)
- Fee calculation (1 bps default)
- Trade logging to JSONL files
- No external connections

### IBKR Broker
- Uses `ib_insync` library
- Connection to TWS/Gateway (port 7497 paper, 7496 live)
- Order types: Market, Limit, Stop, Stop-Limit
- Account info and position queries

### Alpaca Broker (Future)
- REST API with alpaca-trade-api
- WebSocket for streaming

## Order Flow

```
1. Validate order (risk checks)
2. Submit via broker.submit_order()
3. Track order_id
4. Poll/subscribe for fills
5. Update positions on fill
6. Log trade record
```

## Configuration

**Planned structure** (not yet implemented):
- Location: `CONFIG/brokers/{broker}.yaml`
- Contains connection settings, API keys, account configs
- Excluded from git (sensitive credentials)

Example planned config:
```yaml
# CONFIG/brokers/ibkr.yaml (planned)
ibkr:
  host: "127.0.0.1"
  port: 7497  # Paper trading (7496 for live)
  client_id: 123
  account: null  # Auto-detect

# CONFIG/brokers/alpaca.yaml (planned)
alpaca:
  api_key: "${ALPACA_API_KEY}"  # From environment
  secret_key: "${ALPACA_SECRET_KEY}"
  paper: true  # Paper trading mode
```

**Security notes:**
- Never commit API keys to git
- Use environment variable substitution (`${VAR}`)
- Add `CONFIG/brokers/*.yaml` to `.gitignore`

## Related Skills

- `execution-engine.md` - Engine using broker interface
- `risk-management.md` - Pre-trade validation
- `signal-generation.md` - Signals to orders

## Related Documentation

- `LIVE_TRADING/brokers/` - Broker implementations
