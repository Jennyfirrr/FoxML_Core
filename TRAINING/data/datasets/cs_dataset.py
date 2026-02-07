# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
PyTorch Dataset for Cross-Sectional Ranking
============================================

Dataset classes for cross-sectional ranking models. Restructures data from
flat (N, L, F) to timestamp-grouped (T, M, L, F) to enable ranking losses
that compare symbols at the same timestamp.

Key Shapes:
- T: number of unique timestamps (batch dimension when iterating)
- M: number of symbols (cross-section size)
- L: lookback length in bars (e.g., 64)
- F: feature channels (e.g., 5 for OHLCV)

See .claude/plans/cs-ranking-phase2-batching.md for design details.
"""

import TRAINING.common.repro_bootstrap  # noqa: F401 - must be first

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


@dataclass
class CrossSectionalBatch:
    """
    Batch of cross-sectional data for ranking models.

    Attributes:
        X: Sequences tensor of shape (B, M, L, F)
        y: Targets tensor of shape (B, M)
        mask: Valid symbol mask of shape (B, M), 1.0 where symbol present
        timestamps: List of timestamp values, length B
        n_valid: Number of valid symbols per timestamp, shape (B,)
    """

    X: torch.Tensor  # (B, M, L, F)
    y: torch.Tensor  # (B, M)
    mask: torch.Tensor  # (B, M)
    timestamps: List[Any]  # (B,)
    n_valid: torch.Tensor  # (B,)

    def to(self, device: torch.device) -> "CrossSectionalBatch":
        """Move batch to device."""
        return CrossSectionalBatch(
            X=self.X.to(device),
            y=self.y.to(device),
            mask=self.mask.to(device),
            timestamps=self.timestamps,
            n_valid=self.n_valid.to(device),
        )

    @property
    def batch_size(self) -> int:
        """Number of timestamps in batch."""
        return self.X.shape[0]

    @property
    def n_symbols(self) -> int:
        """Total number of symbol slots (M)."""
        return self.X.shape[1]


class CrossSectionalDataset(Dataset):
    """
    Dataset that yields cross-sections (all symbols at same timestamp).

    Each __getitem__ returns one timestamp's worth of data across all symbols.
    DataLoader batches these into (B, M, L, F) tensors.

    Args:
        sequences: Dict mapping symbol -> (T_sym, L, F) array of sequences
        targets: Dict mapping symbol -> (T_sym,) array of CS-normalized targets
        timestamps: Dict mapping symbol -> (T_sym,) array of timestamp values
        symbols: List of all symbols (determines fixed ordering)
        min_symbols_per_timestamp: Skip timestamps with fewer symbols
    """

    def __init__(
        self,
        sequences: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        timestamps: Dict[str, np.ndarray],
        symbols: Optional[List[str]] = None,
        min_symbols_per_timestamp: int = 5,
    ):
        """Initialize cross-sectional dataset."""
        # Use sorted symbols for deterministic ordering
        if symbols is None:
            symbols = sorted(sequences.keys())
        self.symbols = sorted(symbols)
        self.symbol_to_idx = {s: i for i, s in enumerate(self.symbols)}
        self.M = len(self.symbols)

        # Validate inputs
        if len(sequences) == 0:
            logger.warning("CrossSectionalDataset: No sequences provided")
            self.valid_timestamps = []
            self.ts_to_data = {}
            self.T = 0
            self.L = 0
            self.F = 0
            return

        # Infer sequence shape from first symbol
        first_sym = next(iter(sequences.keys()))
        sample_seq = sequences[first_sym]
        if sample_seq.ndim != 3:
            raise ValueError(f"Sequences should be 3D (T, L, F), got shape {sample_seq.shape}")
        self.L = sample_seq.shape[1]
        self.F = sample_seq.shape[2]

        # Build timestamp index
        self._build_timestamp_index(
            sequences, targets, timestamps, min_symbols_per_timestamp
        )

        logger.info(
            f"CrossSectionalDataset created: "
            f"T={self.T} timestamps, M={self.M} symbols, "
            f"L={self.L} lookback, F={self.F} features"
        )

    def _build_timestamp_index(
        self,
        sequences: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        timestamps: Dict[str, np.ndarray],
        min_symbols: int,
    ) -> None:
        """
        Build index mapping timestamp -> symbol data.

        Creates:
            self.ts_to_data: Dict[timestamp -> Dict[symbol -> {sequence, target}]]
            self.valid_timestamps: Sorted list of timestamps with >= min_symbols
        """
        ts_to_data: Dict[Any, Dict[str, Dict[str, Any]]] = defaultdict(dict)

        for symbol in self.symbols:
            if symbol not in sequences:
                continue
            if symbol not in timestamps:
                continue
            if symbol not in targets:
                continue

            ts_arr = timestamps[symbol]
            seq_arr = sequences[symbol]
            tgt_arr = targets[symbol]

            # Validate shapes match
            if len(ts_arr) != len(seq_arr) or len(ts_arr) != len(tgt_arr):
                logger.warning(
                    f"Shape mismatch for {symbol}: "
                    f"ts={len(ts_arr)}, seq={len(seq_arr)}, tgt={len(tgt_arr)}"
                )
                continue

            for idx in range(len(ts_arr)):
                ts_val = ts_arr[idx]
                ts_to_data[ts_val][symbol] = {
                    "sequence": seq_arr[idx],  # (L, F)
                    "target": tgt_arr[idx],  # scalar
                }

        # Filter timestamps with too few symbols
        self.valid_timestamps = sorted(
            [ts for ts, data in ts_to_data.items() if len(data) >= min_symbols]
        )

        # Only keep data for valid timestamps
        self.ts_to_data = {ts: ts_to_data[ts] for ts in self.valid_timestamps}
        self.T = len(self.valid_timestamps)

        n_skipped = len(ts_to_data) - self.T
        if n_skipped > 0:
            logger.debug(
                f"Skipped {n_skipped} timestamps with fewer than {min_symbols} symbols"
            )

    def __len__(self) -> int:
        """Number of valid timestamps."""
        return self.T

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get one timestamp's cross-section.

        Args:
            idx: Timestamp index

        Returns:
            Dict with:
                'X': (M, L, F) tensor, zero-padded for missing symbols
                'y': (M,) tensor, NaN for missing symbols
                'mask': (M,) tensor, 1.0 if symbol present, 0.0 if missing
                'timestamp': Timestamp value
                'n_valid': Number of valid symbols at this timestamp
        """
        ts = self.valid_timestamps[idx]
        data = self.ts_to_data[ts]

        # Pre-allocate tensors
        X = np.zeros((self.M, self.L, self.F), dtype=np.float32)
        y = np.full((self.M,), np.nan, dtype=np.float32)
        mask = np.zeros((self.M,), dtype=np.float32)

        # Fill in data for present symbols
        n_valid = 0
        for symbol, item in data.items():
            if symbol not in self.symbol_to_idx:
                continue
            sym_idx = self.symbol_to_idx[symbol]
            X[sym_idx] = item["sequence"]
            y[sym_idx] = item["target"]
            mask[sym_idx] = 1.0
            n_valid += 1

        return {
            "X": torch.from_numpy(X),
            "y": torch.from_numpy(y),
            "mask": torch.from_numpy(mask),
            "timestamp": ts,
            "n_valid": n_valid,
        }

    def get_symbol_list(self) -> List[str]:
        """Get ordered list of symbols."""
        return self.symbols.copy()

    def get_timestamp_list(self) -> List[Any]:
        """Get ordered list of valid timestamps."""
        return self.valid_timestamps.copy()


class CrossSectionalDatasetSampled(CrossSectionalDataset):
    """
    Cross-sectional dataset with symbol sampling for memory efficiency.

    Instead of returning all M symbols per timestamp, samples a fixed number
    of symbols. Useful when M is large (e.g., 700+) and GPU memory is limited.

    The sampling is deterministic based on timestamp index to ensure
    reproducibility across runs.
    """

    def __init__(
        self,
        sequences: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        timestamps: Dict[str, np.ndarray],
        symbols: Optional[List[str]] = None,
        min_symbols_per_timestamp: int = 5,
        symbols_per_timestamp: int = 200,
        seed: int = 42,
    ):
        """
        Initialize sampled cross-sectional dataset.

        Args:
            sequences: Dict mapping symbol -> (T_sym, L, F) array
            targets: Dict mapping symbol -> (T_sym,) array
            timestamps: Dict mapping symbol -> (T_sym,) array
            symbols: List of all symbols
            min_symbols_per_timestamp: Skip timestamps with fewer symbols
            symbols_per_timestamp: Number of symbols to sample per timestamp
            seed: Base seed for deterministic sampling
        """
        super().__init__(
            sequences=sequences,
            targets=targets,
            timestamps=timestamps,
            symbols=symbols,
            min_symbols_per_timestamp=min_symbols_per_timestamp,
        )
        self.M_sample = symbols_per_timestamp
        self.seed = seed

        logger.info(f"CrossSectionalDatasetSampled: sampling {self.M_sample} of {self.M} symbols")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get one timestamp's cross-section with sampled symbols.

        Uses deterministic sampling based on idx for reproducibility.
        """
        ts = self.valid_timestamps[idx]
        data = self.ts_to_data[ts]

        # Deterministic sampling based on timestamp index
        rng = np.random.RandomState(seed=self.seed + idx)
        available_symbols = sorted(data.keys())
        n_to_sample = min(self.M_sample, len(available_symbols))
        sampled_symbols = rng.choice(
            available_symbols, size=n_to_sample, replace=False
        ).tolist()

        # Build tensors for sampled symbols only
        # Use M_sample slots, even if we have fewer symbols
        X = np.zeros((self.M_sample, self.L, self.F), dtype=np.float32)
        y = np.full((self.M_sample,), np.nan, dtype=np.float32)
        mask = np.zeros((self.M_sample,), dtype=np.float32)

        for i, symbol in enumerate(sorted(sampled_symbols)):
            item = data[symbol]
            X[i] = item["sequence"]
            y[i] = item["target"]
            mask[i] = 1.0

        return {
            "X": torch.from_numpy(X),
            "y": torch.from_numpy(y),
            "mask": torch.from_numpy(mask),
            "timestamp": ts,
            "n_valid": len(sampled_symbols),
            "sampled_symbols": sampled_symbols,
        }


def collate_cross_sectional(batch: List[Dict[str, Any]]) -> CrossSectionalBatch:
    """
    Collate function for CrossSectionalDataset.

    Stacks individual timestamp cross-sections into a batch.

    Args:
        batch: List of dicts from CrossSectionalDataset.__getitem__

    Returns:
        CrossSectionalBatch with stacked tensors
    """
    return CrossSectionalBatch(
        X=torch.stack([b["X"] for b in batch]),  # (B, M, L, F)
        y=torch.stack([b["y"] for b in batch]),  # (B, M)
        mask=torch.stack([b["mask"] for b in batch]),  # (B, M)
        timestamps=[b["timestamp"] for b in batch],  # (B,)
        n_valid=torch.tensor([b["n_valid"] for b in batch], dtype=torch.long),  # (B,)
    )


def create_cs_dataloader(
    dataset: CrossSectionalDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Create DataLoader for cross-sectional data with deterministic settings.

    Args:
        dataset: CrossSectionalDataset instance
        batch_size: Number of timestamps per batch
        shuffle: Whether to shuffle timestamps
        num_workers: Number of worker processes (0 for deterministic)
        pin_memory: Whether to pin memory for GPU transfer
        drop_last: Whether to drop incomplete final batch

    Returns:
        DataLoader configured for cross-sectional ranking
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_cross_sectional,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=False,  # For deterministic behavior
    )


class CrossSectionalDataModule:
    """
    Data module for cross-sectional ranking with train/val/test splits.

    Performs temporal split to ensure no data leakage from future to past.
    """

    def __init__(
        self,
        sequences: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        timestamps: Dict[str, np.ndarray],
        symbols: Optional[List[str]] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        batch_size: int = 32,
        min_symbols_per_timestamp: int = 5,
        symbols_per_timestamp: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Initialize cross-sectional data module.

        Args:
            sequences: Dict mapping symbol -> (T_sym, L, F) array
            targets: Dict mapping symbol -> (T_sym,) array
            timestamps: Dict mapping symbol -> (T_sym,) array
            symbols: List of all symbols
            train_ratio: Training set ratio (by timestamps)
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            batch_size: Timestamps per batch
            min_symbols_per_timestamp: Minimum symbols required
            symbols_per_timestamp: If set, use sampled dataset
            seed: Random seed for sampling
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

        self.batch_size = batch_size
        self.min_symbols = min_symbols_per_timestamp
        self.symbols_per_timestamp = symbols_per_timestamp
        self.seed = seed

        # Build full dataset to get valid timestamps
        full_dataset = CrossSectionalDataset(
            sequences=sequences,
            targets=targets,
            timestamps=timestamps,
            symbols=symbols,
            min_symbols_per_timestamp=min_symbols_per_timestamp,
        )

        self.symbols = full_dataset.get_symbol_list()
        all_ts = full_dataset.get_timestamp_list()
        T = len(all_ts)

        if T == 0:
            logger.warning("CrossSectionalDataModule: No valid timestamps")
            self.train_ts = self.val_ts = self.test_ts = []
            self._sequences = sequences
            self._targets = targets
            self._timestamps = timestamps
            return

        # Temporal split (chronological)
        train_end = int(T * train_ratio)
        val_end = int(T * (train_ratio + val_ratio))

        self.train_ts = set(all_ts[:train_end])
        self.val_ts = set(all_ts[train_end:val_end])
        self.test_ts = set(all_ts[val_end:])

        # Store data for creating split datasets
        self._sequences = sequences
        self._targets = targets
        self._timestamps = timestamps

        logger.info(
            f"CrossSectionalDataModule split: "
            f"train={len(self.train_ts)}, val={len(self.val_ts)}, test={len(self.test_ts)} timestamps"
        )

    def _filter_by_timestamps(
        self, ts_set: set
    ) -> tuple:
        """Filter sequences/targets/timestamps to those in ts_set."""
        filtered_seq = {}
        filtered_tgt = {}
        filtered_ts = {}

        for symbol in self.symbols:
            if symbol not in self._timestamps:
                continue

            ts_arr = self._timestamps[symbol]
            mask = np.array([ts in ts_set for ts in ts_arr])

            if mask.sum() == 0:
                continue

            filtered_seq[symbol] = self._sequences[symbol][mask]
            filtered_tgt[symbol] = self._targets[symbol][mask]
            filtered_ts[symbol] = ts_arr[mask]

        return filtered_seq, filtered_tgt, filtered_ts

    def _create_dataset(self, ts_set: set) -> CrossSectionalDataset:
        """Create dataset for a timestamp set."""
        seq, tgt, ts = self._filter_by_timestamps(ts_set)

        if self.symbols_per_timestamp is not None:
            return CrossSectionalDatasetSampled(
                sequences=seq,
                targets=tgt,
                timestamps=ts,
                symbols=self.symbols,
                min_symbols_per_timestamp=self.min_symbols,
                symbols_per_timestamp=self.symbols_per_timestamp,
                seed=self.seed,
            )
        else:
            return CrossSectionalDataset(
                sequences=seq,
                targets=tgt,
                timestamps=ts,
                symbols=self.symbols,
                min_symbols_per_timestamp=self.min_symbols,
            )

    def get_train_dataloader(self, shuffle: bool = True) -> DataLoader:
        """Get training dataloader."""
        dataset = self._create_dataset(self.train_ts)
        return create_cs_dataloader(dataset, self.batch_size, shuffle)

    def get_val_dataloader(self, shuffle: bool = False) -> DataLoader:
        """Get validation dataloader."""
        dataset = self._create_dataset(self.val_ts)
        return create_cs_dataloader(dataset, self.batch_size, shuffle)

    def get_test_dataloader(self, shuffle: bool = False) -> DataLoader:
        """Get test dataloader."""
        dataset = self._create_dataset(self.test_ts)
        return create_cs_dataloader(dataset, self.batch_size, shuffle)


# ==============================================================================
# DATA PREPARATION INTEGRATION
# ==============================================================================


def prepare_cross_sectional_data(
    mtf_data: Dict[str, "pd.DataFrame"],
    target: str,
    seq_config: Optional[Dict[str, Any]] = None,
    cs_config: Optional[Dict[str, Any]] = None,
    interval_minutes: Optional[int] = None,
    min_symbols_per_timestamp: int = 50,
) -> CrossSectionalDataset:
    """
    Prepare data for cross-sectional ranking training.

    This function integrates:
    1. Sequence building from raw OHLCV (existing utils)
    2. Cross-sectional target computation (Phase 1)
    3. CrossSectionalDataset creation (Phase 2)

    Args:
        mtf_data: Dict mapping symbol -> DataFrame with OHLCV + target columns
        target: Target column name (e.g., "fwd_ret_5m")
        seq_config: Sequence configuration:
            - length_minutes: Sequence length in minutes (default: 320)
            - channels: OHLCV columns (default: standard 5)
            - normalization: Normalization method (default: "returns")
            - gap_tolerance: Gap detection threshold (default: 1.5)
        cs_config: Cross-sectional ranking config:
            - target.type: "cs_percentile", "cs_zscore", or "vol_scaled"
            - target.residualize: Whether to subtract market mean (default: true)
            - target.winsorize: Percentile bounds (default: [0.01, 0.99])
            - target.vol_col: Vol column for vol_scaled type
        interval_minutes: Data interval in minutes
        min_symbols_per_timestamp: Minimum symbols required per timestamp

    Returns:
        CrossSectionalDataset ready for DataLoader

    Example:
        >>> dataset = prepare_cross_sectional_data(
        ...     mtf_data=mtf_data,
        ...     target="fwd_ret_5m",
        ...     seq_config={"length_minutes": 320},
        ...     cs_config={"target": {"type": "cs_percentile"}},
        ... )
        >>> dataloader = create_cs_dataloader(dataset, batch_size=32)
    """
    import pandas as pd

    from TRAINING.training_strategies.utils import build_sequences_from_ohlcv
    from TRAINING.common.targets import compute_cs_target, CrossSectionalTargetType

    # Default configs
    if seq_config is None:
        seq_config = {}
    if cs_config is None:
        cs_config = {}

    # Get config values with defaults
    seq_len_minutes = seq_config.get("length_minutes", 320)
    channels = seq_config.get("channels", ["open", "high", "low", "close", "volume"])
    normalization = seq_config.get("normalization", "returns")
    gap_tolerance = seq_config.get("gap_tolerance", 1.5)

    target_config = cs_config.get("target", {})
    target_type = target_config.get("type", "cs_percentile")
    residualize = target_config.get("residualize", True)
    winsorize = target_config.get("winsorize", [0.01, 0.99])
    vol_col = target_config.get("vol_col", "rolling_vol_20")

    # Get interval from config if not provided
    if interval_minutes is None:
        try:
            from CONFIG.config_loader import get_cfg
            interval_minutes = int(get_cfg("pipeline.data.interval_minutes", default=5))
        except ImportError:
            interval_minutes = 5

    logger.info(
        f"Preparing cross-sectional data: target={target}, "
        f"target_type={target_type}, seq_len={seq_len_minutes}min"
    )

    # Step 1: Build sequences per symbol
    sequences: Dict[str, np.ndarray] = {}
    timestamps_dict: Dict[str, np.ndarray] = {}
    raw_returns_list: List[Dict[str, Any]] = []

    sorted_symbols = sorted(mtf_data.keys())

    for symbol in sorted_symbols:
        df = mtf_data[symbol]

        # Check required columns
        if target not in df.columns:
            logger.debug(f"Target {target} not in {symbol}, skipping")
            continue

        required_cols = ["ts"] + channels
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.debug(f"Missing columns {missing} in {symbol}, skipping")
            continue

        # Build sequences
        try:
            X_seq, ts_arr, indices = build_sequences_from_ohlcv(
                df,
                seq_len_minutes=seq_len_minutes,
                interval_minutes=interval_minutes,
                channels=channels,
                normalization=normalization,
                symbol=symbol,
                handle_gaps=True,
                gap_tolerance=gap_tolerance,
            )

            if len(X_seq) == 0:
                continue

            sequences[symbol] = X_seq
            timestamps_dict[symbol] = ts_arr

            # Collect raw returns for CS target computation
            # Use .iloc with indices to get the aligned returns
            returns = df[target].iloc[indices].values
            for i, (ts_val, ret) in enumerate(zip(ts_arr, returns)):
                raw_returns_list.append({
                    "ts": ts_val,
                    "symbol": symbol,
                    target: ret,
                    "_seq_idx": i,
                })

                # Include vol_col if needed for vol_scaled
                if target_type == "vol_scaled" and vol_col in df.columns:
                    raw_returns_list[-1][vol_col] = df[vol_col].iloc[indices[i]]

        except Exception as e:
            logger.warning(f"Failed to build sequences for {symbol}: {e}")
            continue

    if len(sequences) == 0:
        logger.error("No valid sequences built for any symbol")
        return CrossSectionalDataset(
            sequences={}, targets={}, timestamps={}, symbols=[], min_symbols_per_timestamp=1
        )

    # Step 2: Compute cross-sectional targets
    returns_df = pd.DataFrame(raw_returns_list)

    logger.debug(
        f"Computing CS targets: {len(returns_df)} samples, "
        f"{returns_df['symbol'].nunique()} symbols, "
        f"{returns_df['ts'].nunique()} timestamps"
    )

    # Compute CS target
    vol_col_arg = vol_col if target_type == "vol_scaled" else None
    cs_targets_series = compute_cs_target(
        returns_df,
        target_type=target_type,
        return_col=target,
        time_col="ts",
        vol_col=vol_col_arg,
        residualize=residualize,
        winsorize_pct=tuple(winsorize),
        min_symbols=5,  # Inner minimum for target computation
    )

    # Add CS targets back to DataFrame
    returns_df["cs_target"] = cs_targets_series

    # Step 3: Split targets back to per-symbol arrays
    targets_dict: Dict[str, np.ndarray] = {}

    for symbol in sorted(sequences.keys()):
        mask = returns_df["symbol"] == symbol
        symbol_df = returns_df[mask].sort_values("_seq_idx")
        targets_dict[symbol] = symbol_df["cs_target"].values

    # Step 4: Create CrossSectionalDataset
    dataset = CrossSectionalDataset(
        sequences=sequences,
        targets=targets_dict,
        timestamps=timestamps_dict,
        symbols=list(sequences.keys()),
        min_symbols_per_timestamp=min_symbols_per_timestamp,
    )

    logger.info(
        f"CrossSectionalDataset created: "
        f"T={dataset.T} timestamps, M={dataset.M} symbols"
    )

    return dataset


def cs_collate_fn(batch: List[Dict[str, Any]]) -> CrossSectionalBatch:
    """
    Custom collate function for CrossSectionalDataset.

    Takes a list of dicts from __getitem__ and stacks them into a CrossSectionalBatch.

    Args:
        batch: List of dicts from CrossSectionalDataset.__getitem__()
            Each dict has: X (M, L, F), y (M,), mask (M,), timestamp, n_valid

    Returns:
        CrossSectionalBatch with tensors of shape (B, M, ...)
    """
    # Stack tensors
    X = torch.stack([item["X"] for item in batch], dim=0)  # (B, M, L, F)
    y = torch.stack([item["y"] for item in batch], dim=0)  # (B, M)
    mask = torch.stack([item["mask"] for item in batch], dim=0)  # (B, M)

    # Collect scalars
    timestamps = [item["timestamp"] for item in batch]  # List of B
    n_valid = torch.tensor([item["n_valid"] for item in batch], dtype=torch.int64)  # (B,)

    return CrossSectionalBatch(
        X=X,
        y=y,
        mask=mask,
        timestamps=timestamps,
        n_valid=n_valid,
    )


def create_cs_dataset_from_mtf(
    mtf_data: Dict[str, "pd.DataFrame"],
    target: str,
    target_type: str = "cs_percentile",
    residualize: bool = True,
    winsorize: Optional[List[float]] = None,
    min_symbols_per_timestamp: int = 50,
    sequence_length: int = 64,
    interval_minutes: Optional[int] = None,
) -> CrossSectionalDataset:
    """
    Create CrossSectionalDataset from MTF data (simplified interface).

    This is a convenience wrapper around prepare_cross_sectional_data() for
    use in training loops where direct parameter passing is preferred over
    nested config dicts.

    Args:
        mtf_data: Dict mapping symbol -> DataFrame with OHLCV + target columns
        target: Target column name (e.g., "fwd_ret_5m")
        target_type: CS target type ("cs_percentile", "cs_zscore", "vol_scaled")
        residualize: Whether to subtract market mean before ranking
        winsorize: Percentile bounds for winsorization [low, high]
        min_symbols_per_timestamp: Minimum symbols required per timestamp
        sequence_length: Number of bars in each sequence
        interval_minutes: Data interval in minutes (auto-detected if None)

    Returns:
        CrossSectionalDataset ready for training

    Example:
        >>> dataset = create_cs_dataset_from_mtf(
        ...     mtf_data=mtf_data,
        ...     target="fwd_ret_5m",
        ...     target_type="cs_percentile",
        ...     residualize=True,
        ... )
    """
    if winsorize is None:
        winsorize = [0.01, 0.99]

    # Get interval from config if not provided
    if interval_minutes is None:
        try:
            from CONFIG.config_loader import get_cfg
            interval_minutes = int(get_cfg("pipeline.data.interval_minutes", default=5))
        except ImportError:
            interval_minutes = 5

    # Calculate sequence length in minutes
    seq_len_minutes = sequence_length * interval_minutes

    # Build config dicts for prepare_cross_sectional_data
    seq_config = {
        "length_minutes": seq_len_minutes,
        "channels": ["open", "high", "low", "close", "volume"],
        "normalization": "returns",
        "gap_tolerance": 1.5,
    }

    cs_config = {
        "target": {
            "type": target_type,
            "residualize": residualize,
            "winsorize": list(winsorize),
        },
    }

    return prepare_cross_sectional_data(
        mtf_data=mtf_data,
        target=target,
        seq_config=seq_config,
        cs_config=cs_config,
        interval_minutes=interval_minutes,
        min_symbols_per_timestamp=min_symbols_per_timestamp,
    )
