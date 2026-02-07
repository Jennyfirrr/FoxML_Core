"""
Bandit State Persistence
========================

Save and load CILS state (bandit, reward tracker, optimizer).

Features:
- Atomic file writes (prevents corruption)
- Automatic backups
- Recovery on startup

SST Compliance:
- Uses atomic writes for persistence
- Uses get_cfg() for paths
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

from CONFIG.config_loader import get_cfg
from LIVE_TRADING.common.clock import Clock, get_clock
from LIVE_TRADING.learning.bandit import Exp3IXBandit
from LIVE_TRADING.learning.reward_tracker import RewardTracker
from LIVE_TRADING.learning.weight_optimizer import EnsembleWeightOptimizer

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_STATE_DIR = "state/cils"
DEFAULT_STATE_FILE = "cils_state.json"
DEFAULT_BACKUP_COUNT = 5


class BanditPersistence:
    """
    Manages CILS state persistence.

    Handles saving and loading of:
    - Bandit state (weights, steps, etc.)
    - Reward tracker (pending trades, arm stats)
    - Optimizer state (blend settings)

    Example:
        >>> persistence = BanditPersistence()
        >>> persistence.save(optimizer, reward_tracker)
        >>> # Later...
        >>> optimizer, tracker = persistence.load()
    """

    def __init__(
        self,
        state_dir: Optional[str | Path] = None,
        state_file: Optional[str] = None,
        backup_count: int = DEFAULT_BACKUP_COUNT,
        clock: Optional[Clock] = None,
    ) -> None:
        """
        Initialize persistence manager.

        Args:
            state_dir: Directory for state files
            state_file: Name of state file
            backup_count: Number of backups to keep
            clock: Clock instance for timestamps
        """
        self._clock = clock or get_clock()

        # Resolve paths from config
        default_dir = get_cfg(
            "live_trading.bandit.state_dir", default=DEFAULT_STATE_DIR
        )
        self._state_dir = Path(state_dir or default_dir)

        self._state_file = state_file or get_cfg(
            "live_trading.bandit.state_file", default=DEFAULT_STATE_FILE
        )
        self._backup_count = backup_count

        # Full path to state file
        self._state_path = self._state_dir / self._state_file

        # Ensure directory exists
        self._state_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"BanditPersistence initialized: {self._state_path}")

    def save(
        self,
        optimizer: EnsembleWeightOptimizer,
        reward_tracker: Optional[RewardTracker] = None,
        create_backup: bool = True,
    ) -> Path:
        """
        Save CILS state to file.

        Uses atomic write to prevent corruption.

        Args:
            optimizer: EnsembleWeightOptimizer to save
            reward_tracker: Optional RewardTracker to save
            create_backup: Whether to create backup of existing file

        Returns:
            Path to saved state file
        """
        # Create state dict
        state = {
            "version": 1,
            "saved_at": self._clock.now().isoformat(),
            "optimizer": optimizer.to_dict(),
        }

        if reward_tracker is not None:
            state["reward_tracker"] = reward_tracker.to_dict()

        # Backup existing file if requested
        if create_backup and self._state_path.exists():
            self._rotate_backups()

        # Atomic write
        self._write_atomic(self._state_path, state)

        logger.info(
            f"Saved CILS state: {self._state_path} "
            f"(steps={optimizer.bandit.total_steps})"
        )

        return self._state_path

    def load(
        self,
        clock: Optional[Clock] = None,
    ) -> Tuple[Optional[EnsembleWeightOptimizer], Optional[RewardTracker]]:
        """
        Load CILS state from file.

        Args:
            clock: Clock instance for reward tracker

        Returns:
            Tuple of (optimizer, reward_tracker), either may be None
        """
        if not self._state_path.exists():
            logger.info(f"No CILS state file found at {self._state_path}")
            return None, None

        try:
            with open(self._state_path) as f:
                state = json.load(f)

            version = state.get("version", 1)
            if version != 1:
                logger.warning(f"Unknown CILS state version: {version}")

            # Restore optimizer
            optimizer = None
            if "optimizer" in state:
                optimizer = EnsembleWeightOptimizer.from_dict(state["optimizer"])

            # Restore reward tracker
            tracker = None
            if "reward_tracker" in state:
                tracker = RewardTracker.from_dict(
                    state["reward_tracker"],
                    clock=clock or self._clock,
                )

            saved_at = state.get("saved_at", "unknown")
            logger.info(f"Loaded CILS state from {self._state_path} (saved_at={saved_at})")

            return optimizer, tracker

        except Exception as e:
            logger.error(f"Failed to load CILS state: {e}")
            return None, None

    def _write_atomic(self, path: Path, data: Dict) -> None:
        """Write JSON atomically using temp file and rename."""
        temp_path = path.with_suffix(".tmp")

        try:
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=2)

            # Atomic rename
            temp_path.replace(path)

        except Exception:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise

    def _rotate_backups(self) -> None:
        """Rotate backup files, keeping backup_count most recent."""
        if not self._state_path.exists():
            return

        # Shift existing backups
        for i in range(self._backup_count - 1, 0, -1):
            old_backup = self._state_path.with_suffix(f".bak{i}")
            new_backup = self._state_path.with_suffix(f".bak{i+1}")

            if old_backup.exists():
                if i == self._backup_count - 1:
                    old_backup.unlink()  # Delete oldest
                else:
                    old_backup.rename(new_backup)

        # Create new backup from current
        backup_path = self._state_path.with_suffix(".bak1")
        shutil.copy2(self._state_path, backup_path)

    def get_backup_paths(self) -> list[Path]:
        """Get list of existing backup files."""
        backups = []
        for i in range(1, self._backup_count + 1):
            backup_path = self._state_path.with_suffix(f".bak{i}")
            if backup_path.exists():
                backups.append(backup_path)
        return backups

    def load_from_backup(
        self,
        backup_number: int = 1,
        clock: Optional[Clock] = None,
    ) -> Tuple[Optional[EnsembleWeightOptimizer], Optional[RewardTracker]]:
        """
        Load CILS state from a backup file.

        Args:
            backup_number: Which backup to load (1 = most recent)
            clock: Clock instance for reward tracker

        Returns:
            Tuple of (optimizer, reward_tracker)
        """
        backup_path = self._state_path.with_suffix(f".bak{backup_number}")

        if not backup_path.exists():
            logger.warning(f"Backup {backup_number} not found: {backup_path}")
            return None, None

        # Temporarily swap paths
        original_path = self._state_path
        self._state_path = backup_path

        try:
            return self.load(clock=clock)
        finally:
            self._state_path = original_path

    def delete_state(self, include_backups: bool = False) -> None:
        """
        Delete state file and optionally backups.

        Args:
            include_backups: Whether to delete backup files too
        """
        if self._state_path.exists():
            self._state_path.unlink()
            logger.info(f"Deleted CILS state: {self._state_path}")

        if include_backups:
            for backup in self.get_backup_paths():
                backup.unlink()
                logger.info(f"Deleted backup: {backup}")

    def state_exists(self) -> bool:
        """Check if state file exists."""
        return self._state_path.exists()

    def get_state_info(self) -> Optional[Dict[str, any]]:
        """
        Get info about saved state without loading full state.

        Returns:
            Dict with metadata or None if no state
        """
        if not self._state_path.exists():
            return None

        try:
            with open(self._state_path) as f:
                state = json.load(f)

            return {
                "path": str(self._state_path),
                "version": state.get("version"),
                "saved_at": state.get("saved_at"),
                "has_optimizer": "optimizer" in state,
                "has_reward_tracker": "reward_tracker" in state,
                "bandit_steps": (
                    state.get("optimizer", {}).get("bandit", {}).get("total_steps")
                ),
                "file_size_bytes": self._state_path.stat().st_size,
            }
        except Exception as e:
            return {"error": str(e)}

    @property
    def state_path(self) -> Path:
        """Get path to state file."""
        return self._state_path

    @property
    def state_dir(self) -> Path:
        """Get state directory."""
        return self._state_dir


def save_cils_state(
    optimizer: EnsembleWeightOptimizer,
    reward_tracker: Optional[RewardTracker] = None,
    state_path: Optional[str | Path] = None,
) -> Path:
    """
    Convenience function to save CILS state.

    Args:
        optimizer: EnsembleWeightOptimizer to save
        reward_tracker: Optional RewardTracker to save
        state_path: Optional custom path

    Returns:
        Path to saved state
    """
    persistence = BanditPersistence()
    if state_path:
        persistence._state_path = Path(state_path)
    return persistence.save(optimizer, reward_tracker)


def load_cils_state(
    state_path: Optional[str | Path] = None,
    clock: Optional[Clock] = None,
) -> Tuple[Optional[EnsembleWeightOptimizer], Optional[RewardTracker]]:
    """
    Convenience function to load CILS state.

    Args:
        state_path: Optional custom path
        clock: Clock instance for reward tracker

    Returns:
        Tuple of (optimizer, reward_tracker)
    """
    persistence = BanditPersistence()
    if state_path:
        persistence._state_path = Path(state_path)
    return persistence.load(clock=clock)
