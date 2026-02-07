# Phase 2: Timestamp-Grouped Data Structure

**Parent**: `cross-sectional-ranking-objective.md`
**Status**: Complete ✅
**Estimated Effort**: 2-3 hours
**Actual Effort**: ~1.5 hours

## Objective

Restructure data from flat `(N_samples, L, F)` to timestamp-grouped `(T, M, L, F)` where:
- T = number of timestamps (batch dimension)
- M = symbols per timestamp
- L = lookback length (64 bars)
- F = channels (5 for OHLCV)

This enables computing ranking losses **within** each timestamp.

## Current State

```python
# Current: Flat structure
X.shape = (728000, 64, 5)  # All (symbol, time) pairs flattened
y.shape = (728000,)        # Targets

# DataLoader returns random samples, mixing timestamps
batch_X.shape = (32, 64, 5)  # 32 random samples from any timestamp
```

## Proposed Structure

```python
# New: Grouped by timestamp
# Each timestamp is a "query" with M "candidates" (symbols)

class CrossSectionalBatch:
    X: Tensor       # (B, M, L, F) - B timestamps, M symbols each
    y: Tensor       # (B, M) - CS percentile targets
    mask: Tensor    # (B, M) - 1 if symbol present at this t, 0 if missing
    timestamps: List[datetime]  # (B,) - actual timestamp values
    symbols: List[List[str]]    # (B, M) - symbol names per position
```

## Implementation

### CrossSectionalDataset

```python
# TRAINING/data/cross_sectional_dataset.py

class CrossSectionalDataset(torch.utils.data.Dataset):
    """
    Dataset that yields cross-sections (all symbols at same timestamp).

    Each __getitem__ returns one timestamp's worth of data.
    DataLoader batches these into (B, M, L, F).
    """

    def __init__(
        self,
        sequences: Dict[str, np.ndarray],  # symbol -> (T_sym, L, F)
        targets: Dict[str, np.ndarray],    # symbol -> (T_sym,) CS targets
        timestamps: Dict[str, np.ndarray], # symbol -> (T_sym,) timestamps
        symbols: List[str],
        min_symbols_per_timestamp: int = 50,
    ):
        """
        Args:
            sequences: Per-symbol sequence arrays
            targets: Per-symbol CS target arrays
            timestamps: Per-symbol timestamp arrays
            symbols: List of all symbols
            min_symbols_per_timestamp: Skip timestamps with fewer symbols
        """
        self.symbols = sorted(symbols)
        self.symbol_to_idx = {s: i for i, s in enumerate(self.symbols)}
        self.M = len(symbols)

        # Build timestamp index: which symbols exist at each unique timestamp
        self._build_timestamp_index(sequences, targets, timestamps, min_symbols_per_timestamp)

    def _build_timestamp_index(self, sequences, targets, timestamps, min_symbols):
        """
        Create index: timestamp -> {symbol: (seq_idx, target_value)}
        """
        from collections import defaultdict

        ts_to_data = defaultdict(dict)

        for symbol in self.symbols:
            if symbol not in timestamps:
                continue
            ts_arr = timestamps[symbol]
            for idx, ts in enumerate(ts_arr):
                ts_to_data[ts][symbol] = {
                    'seq_idx': idx,
                    'target': targets[symbol][idx],
                    'sequence': sequences[symbol][idx],  # Or store idx and fetch later
                }

        # Filter timestamps with too few symbols
        self.valid_timestamps = sorted([
            ts for ts, data in ts_to_data.items()
            if len(data) >= min_symbols
        ])

        self.ts_to_data = ts_to_data
        self.T = len(self.valid_timestamps)

    def __len__(self):
        return self.T

    def __getitem__(self, idx) -> Dict:
        """
        Return one timestamp's cross-section.

        Returns:
            {
                'X': (M, L, F) tensor, padded for missing symbols
                'y': (M,) tensor, NaN for missing symbols
                'mask': (M,) tensor, 1 if present, 0 if missing
                'timestamp': datetime
            }
        """
        ts = self.valid_timestamps[idx]
        data = self.ts_to_data[ts]

        L, F = next(iter(data.values()))['sequence'].shape
        X = np.zeros((self.M, L, F), dtype=np.float32)
        y = np.full((self.M,), np.nan, dtype=np.float32)
        mask = np.zeros((self.M,), dtype=np.float32)

        for symbol, item in data.items():
            sym_idx = self.symbol_to_idx[symbol]
            X[sym_idx] = item['sequence']
            y[sym_idx] = item['target']
            mask[sym_idx] = 1.0

        return {
            'X': torch.from_numpy(X),
            'y': torch.from_numpy(y),
            'mask': torch.from_numpy(mask),
            'timestamp': ts,
        }


def collate_cross_sectional(batch: List[Dict]) -> Dict:
    """
    Collate function for DataLoader.

    Stacks individual timestamp cross-sections into batch.
    """
    return {
        'X': torch.stack([b['X'] for b in batch]),      # (B, M, L, F)
        'y': torch.stack([b['y'] for b in batch]),      # (B, M)
        'mask': torch.stack([b['mask'] for b in batch]), # (B, M)
        'timestamps': [b['timestamp'] for b in batch],
    }
```

### Memory-Efficient Variant

For large universes, loading all sequences into memory may be prohibitive.

```python
class CrossSectionalDatasetLazy(CrossSectionalDataset):
    """
    Lazy-loading variant that reads sequences on demand.

    Stores only metadata in memory, loads sequences from disk per batch.
    """

    def __init__(
        self,
        sequence_dir: Path,  # Directory with {symbol}.npy files
        targets_file: Path,  # Parquet with [ts, symbol, target]
        ...
    ):
        # Build index from targets_file
        # Don't load sequences into memory
        pass

    def __getitem__(self, idx):
        ts = self.valid_timestamps[idx]
        # Load sequences for this timestamp's symbols on demand
        ...
```

### Symbol Sampling (Optional)

If M=728 is too large, sample a subset per batch:

```python
class CrossSectionalDatasetSampled(CrossSectionalDataset):
    """
    Sample M_sample < M symbols per timestamp.

    Useful for memory constraints or faster iteration.
    """

    def __init__(self, ..., symbols_per_timestamp: int = 200):
        super().__init__(...)
        self.M_sample = symbols_per_timestamp

    def __getitem__(self, idx):
        ts = self.valid_timestamps[idx]
        data = self.ts_to_data[ts]

        # Sample M_sample symbols (deterministic based on idx for reproducibility)
        rng = np.random.RandomState(seed=idx)
        available = list(data.keys())
        sampled = rng.choice(available, size=min(self.M_sample, len(available)), replace=False)

        # Build tensors for sampled symbols only
        ...
```

## Integration with Existing Pipeline

### Data Preparation Flow

```python
# In data_preparation.py or new module

def prepare_cross_sectional_data(
    mtf_data: Dict[str, pd.DataFrame],
    target: str,
    seq_config: Dict,
    cs_config: Dict,
    interval_minutes: int,
) -> CrossSectionalDataset:
    """
    Prepare data for cross-sectional ranking training.

    1. Build sequences per symbol (existing)
    2. Compute CS targets (Phase 1)
    3. Create CrossSectionalDataset (this phase)
    """
    from TRAINING.training_strategies.utils import build_sequences_from_ohlcv
    from TRAINING.common.targets.cross_sectional import compute_cs_percentile_target

    sequences = {}
    all_targets = {}
    all_timestamps = {}

    for symbol, df in sorted(mtf_data.items()):
        # Build sequences
        X, ts, idx = build_sequences_from_ohlcv(
            df,
            seq_len_minutes=seq_config['length_minutes'],
            interval_minutes=interval_minutes,
            normalization=seq_config['normalization'],
        )

        if len(X) == 0:
            continue

        # Extract raw returns for target computation
        raw_returns = df.loc[idx, target].values

        sequences[symbol] = X
        all_timestamps[symbol] = ts
        # Store raw returns; CS normalization happens in target computation

    # Compute CS targets across all symbols
    # Need to join into single DataFrame for groupby
    target_df = _build_target_dataframe(all_timestamps, mtf_data, target)
    cs_targets = compute_cs_percentile_target(
        target_df,
        return_col=target,
        residualize=cs_config['target']['residualize'],
        winsorize_pct=tuple(cs_config['target']['winsorize']),
    )

    # Split back to per-symbol
    for symbol in sequences:
        mask = target_df['symbol'] == symbol
        all_targets[symbol] = cs_targets[mask].values

    return CrossSectionalDataset(
        sequences=sequences,
        targets=all_targets,
        timestamps=all_timestamps,
        symbols=list(sequences.keys()),
        min_symbols_per_timestamp=cs_config['batching']['min_symbols_per_timestamp'],
    )
```

## Memory Analysis

| Configuration | Memory (approx) |
|---------------|-----------------|
| Full: (5000, 728, 64, 5) float32 | ~4.7 GB |
| Batched: (32, 728, 64, 5) float32 | ~300 MB |
| Sampled: (32, 200, 64, 5) float32 | ~82 MB |

Recommendation: Use full M=728 if GPU memory allows, else sample M=200-400.

## Deliverables

1. [x] `TRAINING/data/datasets/cs_dataset.py`:
   - `CrossSectionalDataset`
   - `CrossSectionalDatasetSampled`
   - `CrossSectionalDataModule`
   - `CrossSectionalBatch` (dataclass)
   - `collate_cross_sectional()`
   - `create_cs_dataloader()`
   - `prepare_cross_sectional_data()`

2. [x] Updated `TRAINING/data/datasets/__init__.py` with exports

3. [x] Unit tests in `tests/test_cs_dataset.py` (21 tests passing)

## Definition of Done

- [x] Dataset yields `(B, M, L, F)` batches
- [x] Missing symbols handled with mask
- [x] Timestamps never mixed within ranking loss computation
- [x] Memory-efficient options available (CrossSectionalDatasetSampled)
- [x] Deterministic ordering (sorted symbols, sorted timestamps)

## Session Log

### 2026-01-21: Implementation Complete
- Created `TRAINING/data/datasets/cs_dataset.py` with all dataset classes
- Implemented `CrossSectionalBatch` dataclass for typed batch handling
- Added `CrossSectionalDataModule` for train/val/test splitting
- Created `prepare_cross_sectional_data()` integration function
- Written 21 unit tests covering:
  - Basic dataset creation and iteration
  - Mask handling for missing symbols
  - Symbol sampling (CrossSectionalDatasetSampled)
  - Collation and batching
  - DataLoader iteration
  - DataModule temporal splitting
  - Determinism verification
- Note: Skipped lazy-loading variant (not needed for typical universe sizes)

**Next**: Phase 3 (Ranking Loss Functions) - see `cs-ranking-phase3-losses.md`
