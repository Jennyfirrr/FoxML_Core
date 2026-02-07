# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""Dataset classes for sequential and time-series data"""

from .seq_dataset import (
    SeqDataset,
    VariableSeqDataset,
    pad_collate,
    create_seq_dataloader,
    SeqDataModule,
)
from .cs_dataset import (
    CrossSectionalBatch,
    CrossSectionalDataset,
    CrossSectionalDatasetSampled,
    CrossSectionalDataModule,
    collate_cross_sectional,
    create_cs_dataloader,
    prepare_cross_sectional_data,
)

__all__ = [
    # Sequential datasets
    "SeqDataset",
    "VariableSeqDataset",
    "pad_collate",
    "create_seq_dataloader",
    "SeqDataModule",
    # Cross-sectional ranking datasets
    "CrossSectionalBatch",
    "CrossSectionalDataset",
    "CrossSectionalDatasetSampled",
    "CrossSectionalDataModule",
    "collate_cross_sectional",
    "create_cs_dataloader",
    "prepare_cross_sectional_data",
]

