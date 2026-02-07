# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Cross-Sectional Model Wrappers
==============================

Model wrappers that adapt existing sequence models (LSTM, Transformer, CNN1D)
to work with cross-sectional ranking data.

The key transformation:
- Input: (B, M, L, F) - B timestamps, M symbols, L sequence length, F features
- Output: (B, M) - One score per symbol per timestamp

See .claude/plans/cs-ranking-phase3-losses.md for design details.
"""

import TRAINING.common.repro_bootstrap  # noqa: F401 - must be first

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CrossSectionalSequenceModel(nn.Module):
    """
    Wrapper that adapts a sequence model for cross-sectional ranking.

    Takes a base sequence model that processes individual sequences and applies
    it to all symbols in a cross-section, producing one score per symbol.

    Input shape: (B, M, L, F)
        - B: batch size (number of timestamps)
        - M: number of symbols
        - L: sequence length
        - F: number of features/channels

    Output shape: (B, M)
        - One score per symbol per timestamp

    Example:
        >>> # Base LSTM model: (batch, L, F) -> (batch, 1)
        >>> base_model = SimpleLSTM(input_size=5, hidden_size=64, output_size=1)
        >>> # Wrap for cross-sectional use
        >>> cs_model = CrossSectionalSequenceModel(base_model)
        >>> # Process cross-sectional batch
        >>> scores = cs_model(X, mask)  # (B, M, L, F) -> (B, M)
    """

    def __init__(
        self,
        seq_model: nn.Module,
        apply_mask: bool = False,
        mask_value: float = 0.0,
    ):
        """
        Initialize cross-sectional model wrapper.

        Args:
            seq_model: Base sequence model that takes (batch, L, F) and returns
                      (batch, 1) or (batch,). Will be applied to each symbol.
            apply_mask: If True, multiply output by mask to zero out missing symbols
            mask_value: Value to use for masked positions (default: 0.0)
        """
        super().__init__()
        self.seq_model = seq_model
        self.apply_mask = apply_mask
        self.mask_value = mask_value

    def forward(
        self,
        X: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process cross-sectional batch through sequence model.

        Args:
            X: Input tensor of shape (B, M, L, F)
            mask: Optional mask of shape (B, M), 1.0 for valid, 0.0 for missing

        Returns:
            Scores tensor of shape (B, M)
        """
        B, M, L, F = X.shape

        # Flatten batch and symbol dimensions
        X_flat = X.view(B * M, L, F)  # (B*M, L, F)

        # Apply base model
        scores_flat = self.seq_model(X_flat)  # (B*M, 1) or (B*M,)

        # Handle different output shapes
        if scores_flat.dim() == 2 and scores_flat.shape[1] == 1:
            scores_flat = scores_flat.squeeze(-1)  # (B*M,)

        # Reshape back to (B, M)
        scores = scores_flat.view(B, M)

        # Apply mask if requested
        if self.apply_mask and mask is not None:
            scores = scores * mask + self.mask_value * (1 - mask)

        return scores

    def extra_repr(self) -> str:
        return f"apply_mask={self.apply_mask}, mask_value={self.mask_value}"


class CrossSectionalMLPModel(nn.Module):
    """
    Simple MLP baseline for cross-sectional ranking.

    Flattens the sequence dimension and applies MLP to each symbol independently.
    Useful as a baseline or for ablation studies.

    Input: (B, M, L, F) -> flatten last two dims -> (B, M, L*F) -> MLP -> (B, M)
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        hidden_dims: list = None,
        dropout: float = 0.1,
    ):
        """
        Initialize MLP model.

        Args:
            seq_len: Sequence length (L)
            n_features: Number of features (F)
            hidden_dims: List of hidden layer dimensions (default: [128, 64])
            dropout: Dropout rate
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        input_dim = seq_len * n_features
        layers = []

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.mlp = nn.Sequential(*layers)
        self.seq_len = seq_len
        self.n_features = n_features

    def forward(
        self,
        X: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process cross-sectional batch.

        Args:
            X: Input tensor of shape (B, M, L, F)
            mask: Optional mask of shape (B, M)

        Returns:
            Scores tensor of shape (B, M)
        """
        B, M, L, F = X.shape

        # Flatten sequence and feature dimensions
        X_flat = X.view(B * M, L * F)  # (B*M, L*F)

        # Apply MLP
        scores_flat = self.mlp(X_flat).squeeze(-1)  # (B*M,)

        # Reshape
        scores = scores_flat.view(B, M)

        return scores


class CrossSectionalAttentionModel(nn.Module):
    """
    Cross-sectional model with attention across symbols.

    Unlike CrossSectionalSequenceModel which processes each symbol independently,
    this model allows information flow between symbols at the same timestamp.

    This can capture relative patterns like "this symbol is cheap relative to peers".
    """

    def __init__(
        self,
        seq_model: nn.Module,
        embed_dim: int,
        n_heads: int = 4,
        cross_symbol_layers: int = 1,
        dropout: float = 0.1,
    ):
        """
        Initialize attention-based cross-sectional model.

        Args:
            seq_model: Base sequence model (batch, L, F) -> (batch, embed_dim)
            embed_dim: Embedding dimension from seq_model output
            n_heads: Number of attention heads for cross-symbol attention
            cross_symbol_layers: Number of cross-symbol transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        self.seq_model = seq_model
        self.embed_dim = embed_dim

        # Cross-symbol attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_symbol_attn = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cross_symbol_layers,
        )

        # Output projection
        self.output_proj = nn.Linear(embed_dim, 1)

    def forward(
        self,
        X: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process cross-sectional batch with cross-symbol attention.

        Args:
            X: Input tensor of shape (B, M, L, F)
            mask: Optional mask of shape (B, M)

        Returns:
            Scores tensor of shape (B, M)
        """
        B, M, L, F = X.shape

        # Step 1: Process each symbol's sequence independently
        X_flat = X.view(B * M, L, F)  # (B*M, L, F)
        embeddings_flat = self.seq_model(X_flat)  # (B*M, embed_dim)

        # Handle different output shapes
        if embeddings_flat.dim() == 1:
            embeddings_flat = embeddings_flat.unsqueeze(-1)
        if embeddings_flat.shape[-1] != self.embed_dim:
            raise ValueError(
                f"seq_model output dim {embeddings_flat.shape[-1]} != embed_dim {self.embed_dim}"
            )

        embeddings = embeddings_flat.view(B, M, self.embed_dim)  # (B, M, embed_dim)

        # Step 2: Cross-symbol attention
        # Create attention mask from symbol mask (True = ignore)
        if mask is not None:
            # mask is (B, M), 1.0 = valid, 0.0 = missing
            # Transformer expects True = ignore
            attn_mask = (mask < 0.5)  # (B, M)
        else:
            attn_mask = None

        # Apply cross-symbol transformer
        # TransformerEncoder expects src_key_padding_mask of shape (B, M)
        embeddings = self.cross_symbol_attn(
            embeddings,
            src_key_padding_mask=attn_mask,
        )  # (B, M, embed_dim)

        # Step 3: Project to scores
        scores = self.output_proj(embeddings).squeeze(-1)  # (B, M)

        return scores
