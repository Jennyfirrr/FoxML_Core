# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Cross-Sectional Ranking Trainer
===============================

Training loop for cross-sectional ranking models. Wraps existing sequence
models (LSTM, Transformer, CNN1D) with ranking-specific data batching,
loss functions, and metrics.

Key differences from pointwise training:
1. Data: Uses CrossSectionalDataset for (T, M, L, F) batching
2. Loss: Uses ranking losses (pairwise, listwise) instead of MSE
3. Metrics: Uses ranking metrics (Spearman IC, spread) instead of MAE

See .claude/plans/cs-ranking-phase5-integration.md for design details.
"""

import TRAINING.common.repro_bootstrap  # noqa: F401 - must be first

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from TRAINING.data.datasets.cs_dataset import (
    CrossSectionalBatch,
    CrossSectionalDataset,
    cs_collate_fn,
)
from TRAINING.losses.ranking_losses import (
    get_ranking_loss,
    pairwise_logistic_loss,
)
from TRAINING.models.specialized.metrics import (
    compute_ranking_metrics,
    spearman_ic_matrix,
)

logger = logging.getLogger(__name__)


def train_cs_ranking_model(
    model: nn.Module,
    train_dataset: CrossSectionalDataset,
    val_dataset: Optional[CrossSectionalDataset],
    cs_ranking_config: Dict[str, Any],
    output_dir: Path,
    target: str,
    family: str,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Train a model with cross-sectional ranking objective.

    Args:
        model: PyTorch model that takes (B, M, L, F) input and outputs (B, M) scores
        train_dataset: CrossSectionalDataset for training
        val_dataset: CrossSectionalDataset for validation (optional)
        cs_ranking_config: CS ranking config from pipeline.yaml
        output_dir: Directory to save model and metrics
        target: Target name
        family: Model family name
        device: Device to train on (default: auto-detect)

    Returns:
        Dict with:
            'model': Trained model
            'best_metrics': Best validation metrics
            'training_history': Per-epoch metrics
            'epochs_trained': Number of epochs completed
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"🎯 Training CS ranking model: {family} for {target}")
    logger.info(f"   Device: {device}")
    logger.info(f"   Train timestamps: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"   Val timestamps: {len(val_dataset)}")

    # Extract config
    loss_config = cs_ranking_config.get("loss", {})
    batching_config = cs_ranking_config.get("batching", {})
    metrics_config = cs_ranking_config.get("metrics", {})

    # Training hyperparameters
    epochs = cs_ranking_config.get("epochs", 30)
    lr = cs_ranking_config.get("learning_rate", 1e-3)
    patience = cs_ranking_config.get("patience", 5)
    batch_size = batching_config.get("timestamps_per_batch", 32)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=cs_collate_fn,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=device.type == "cuda",
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=cs_collate_fn,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

    # Move model to device
    model = model.to(device)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Create loss function
    loss_type = loss_config.get("type", "pairwise_logistic")
    loss_kwargs = {
        k: v for k, v in loss_config.items()
        if k not in ("type",)
    }
    loss_fn = get_ranking_loss(loss_type, **loss_kwargs)

    # Training loop
    best_ic = -float("inf")
    best_metrics = {}
    best_state = None
    patience_counter = 0
    training_history = []

    for epoch in range(epochs):
        # Train epoch
        model.train()
        train_losses = []

        for batch in train_loader:
            batch = batch.to(device)

            # Forward pass
            optimizer.zero_grad()
            scores = model(batch.X)  # (B, M)

            # Compute ranking loss
            loss = loss_fn(scores, batch.y, batch.mask)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # Validation
        val_metrics = {}
        if val_loader is not None:
            val_metrics = _evaluate_cs_model(model, val_loader, device, metrics_config)

            # Check for improvement
            val_ic = val_metrics.get("spearman_ic", 0.0)
            if val_ic > best_ic:
                best_ic = val_ic
                best_metrics = val_metrics.copy()
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            logger.info(
                f"  Epoch {epoch + 1}/{epochs}: "
                f"train_loss={avg_train_loss:.4f}, "
                f"val_ic={val_ic:.4f}, "
                f"val_spread={val_metrics.get('spread', 0.0):.4f}"
            )
        else:
            logger.info(f"  Epoch {epoch + 1}/{epochs}: train_loss={avg_train_loss:.4f}")

        # Record history
        training_history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))},
        })

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"  Early stopping at epoch {epoch + 1}")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"  Restored best model (IC={best_ic:.4f})")

    return {
        "model": model,
        "best_metrics": best_metrics,
        "training_history": training_history,
        "epochs_trained": len(training_history),
    }


def _evaluate_cs_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    metrics_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate cross-sectional ranking model.

    Args:
        model: Model to evaluate
        dataloader: Validation data loader
        device: Device
        metrics_config: Metrics configuration

    Returns:
        Dict with ranking metrics
    """
    model.eval()

    all_scores = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            scores = model(batch.X)  # (B, M)

            all_scores.append(scores.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())
            all_masks.append(batch.mask.cpu().numpy())

    # Concatenate all batches
    scores_arr = np.concatenate(all_scores, axis=0)  # (T, M)
    targets_arr = np.concatenate(all_targets, axis=0)  # (T, M)
    masks_arr = np.concatenate(all_masks, axis=0)  # (T, M)

    # Compute ranking metrics
    metrics = compute_ranking_metrics(
        scores_arr,
        targets_arr,
        masks_arr,
        config=metrics_config,
    )

    # Return flattened metrics (without nested 'details')
    return {
        "spearman_ic": metrics["spearman_ic"],
        "ic_ir": metrics["ic_ir"],
        "ic_hit_rate": metrics["ic_hit_rate"],
        "spread": metrics["spread"],
        "spread_sharpe": metrics["spread_sharpe"],
        "net_spread": metrics["net_spread"],
        "turnover": metrics["turnover"],
    }


def create_cs_model_metadata(
    model: nn.Module,
    training_result: Dict[str, Any],
    cs_ranking_config: Dict[str, Any],
    family: str,
    target: str,
    model_path: Path,
) -> Dict[str, Any]:
    """
    Create model_meta.json for CS ranking model.

    Follows INTEGRATION_CONTRACTS.md schema with CS ranking fields.
    """
    from TRAINING.models.specialized.core import _compute_model_checksum

    # Compute model checksum
    model_checksum = None
    if model_path.exists():
        model_checksum = _compute_model_checksum(model_path)

    best_metrics = training_result.get("best_metrics", {})

    return {
        # Required fields
        "family": family,
        "target": target,
        "feature_list": [],  # CS ranking uses raw OHLCV, not features
        "n_features": 0,
        "metrics": {
            "mean_IC": best_metrics.get("spearman_ic", 0.0),
            "mean_RankIC": best_metrics.get("spearman_ic", 0.0),
            "IC_IR": best_metrics.get("ic_ir", 0.0),
        },
        "model_checksum": model_checksum,
        "interval_minutes": cs_ranking_config.get("interval_minutes", 5),

        # Raw sequence mode fields
        "input_mode": "raw_sequence",
        "sequence_length": cs_ranking_config.get("sequence", {}).get("length_bars", 64),
        "sequence_channels": ["open", "high", "low", "close", "volume"],
        "sequence_normalization": cs_ranking_config.get("sequence", {}).get(
            "normalization", "log_returns"
        ),

        # CS ranking fields (per INTEGRATION_CONTRACTS.md v1.4)
        "cross_sectional_ranking": {
            "enabled": True,
            "target_type": cs_ranking_config.get("target", {}).get("type", "cs_percentile"),
            "loss_type": cs_ranking_config.get("loss", {}).get("type", "pairwise_logistic"),
            "sequence_length": cs_ranking_config.get("sequence", {}).get("length_bars", 64),
            "normalization": cs_ranking_config.get("sequence", {}).get(
                "normalization", "log_returns"
            ),
            "training_metrics": {
                "best_ic": best_metrics.get("spearman_ic", 0.0),
                "best_spread": best_metrics.get("spread", 0.0),
                "epochs_trained": training_result.get("epochs_trained", 0),
                "ic_ir": best_metrics.get("ic_ir"),
                "ic_hit_rate": best_metrics.get("ic_hit_rate"),
                "turnover": best_metrics.get("turnover"),
                "net_spread": best_metrics.get("net_spread"),
            },
        },
    }
