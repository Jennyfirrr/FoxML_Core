"""
State Persistence
=================

Robust state persistence with atomic writes, WAL, and recovery.

Features:
- Atomic file writes (temp file + rename)
- Write-ahead log for crash recovery
- Automatic backups
- Checksum verification
- Configurable retention

SST Compliance:
- All operations logged
- Recovery is deterministic
- Timezone-aware timestamps
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

from CONFIG.config_loader import get_cfg

from LIVE_TRADING.common.clock import Clock, get_clock

logger = logging.getLogger(__name__)

T = TypeVar("T")


def compute_checksum(data: Dict[str, Any]) -> str:
    """
    Compute SHA256 checksum of JSON data.

    Uses sorted keys for deterministic output.

    Args:
        data: Dictionary to checksum

    Returns:
        16-character hex checksum
    """
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def verify_checksum(data: Dict[str, Any], expected: str) -> bool:
    """
    Verify checksum matches.

    Args:
        data: Dictionary to verify
        expected: Expected checksum

    Returns:
        True if checksum matches
    """
    actual = compute_checksum(data)
    return actual == expected


@dataclass
class WALEntry:
    """Write-ahead log entry."""

    sequence: int
    operation: str  # "save", "update_position", "record_trade", etc.
    timestamp: datetime
    data: Dict[str, Any]
    checksum: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "checksum": self.checksum,
            "data": self.data,
            "op": self.operation,
            "seq": self.sequence,
            "ts": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WALEntry":
        """Reconstruct from dictionary."""
        from LIVE_TRADING.common.time_utils import parse_iso

        return cls(
            sequence=d["seq"],
            operation=d["op"],
            timestamp=parse_iso(d["ts"]),
            data=d["data"],
            checksum=d["checksum"],
        )


class WriteAheadLog:
    """
    Write-ahead log for state durability.

    Every state change is logged before being applied.
    On crash recovery, replay the WAL to reconstruct state.

    Example:
        >>> wal = WriteAheadLog(Path("state.wal"))
        >>> wal.append("update_position", {"symbol": "AAPL", "shares": 100})
        >>> wal.checkpoint()  # After successful state save
    """

    def __init__(
        self,
        path: Path,
        max_entries: Optional[int] = None,
        clock: Optional[Clock] = None,
    ):
        """
        Initialize WAL.

        Args:
            path: Path to WAL file
            max_entries: Max entries before auto-truncate (default from config)
            clock: Clock for timestamps
        """
        self._path = path
        self._max_entries = max_entries if max_entries is not None else get_cfg(
            "live_trading.persistence.wal_max_entries",
            default=10000,
        )
        self._clock = clock or get_clock()
        self._sequence = 0
        self._entries: List[WALEntry] = []

        # Load existing WAL if present
        if self._path.exists():
            self._load()

    @property
    def path(self) -> Path:
        """Get WAL file path."""
        return self._path

    @property
    def sequence(self) -> int:
        """Get current sequence number."""
        return self._sequence

    @property
    def entry_count(self) -> int:
        """Get number of entries."""
        return len(self._entries)

    def _load(self) -> None:
        """Load WAL from disk."""
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = WALEntry.from_dict(json.loads(line))
                        self._entries.append(entry)
                        self._sequence = max(self._sequence, entry.sequence)

            logger.info(f"Loaded WAL with {len(self._entries)} entries")
        except Exception as e:
            logger.warning(f"Failed to load WAL: {e}")
            self._entries = []

    def append(self, operation: str, data: Dict[str, Any]) -> WALEntry:
        """
        Append entry to WAL.

        Args:
            operation: Operation type
            data: Operation data

        Returns:
            WAL entry
        """
        self._sequence += 1
        entry = WALEntry(
            sequence=self._sequence,
            operation=operation,
            timestamp=self._clock.now(),
            data=data,
            checksum=compute_checksum(data),
        )

        # Write to disk immediately (append mode)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), default=str) + "\n")
            f.flush()

        self._entries.append(entry)

        # Auto-truncate if too large
        if len(self._entries) > self._max_entries:
            self._truncate()

        return entry

    def checkpoint(self) -> None:
        """
        Mark checkpoint after successful state save.

        Clears WAL since state is now persisted.
        """
        self._entries.clear()
        self._path.unlink(missing_ok=True)
        logger.debug("WAL checkpoint - cleared")

    def get_entries_since(self, sequence: int) -> List[WALEntry]:
        """Get all entries after given sequence number."""
        return [e for e in self._entries if e.sequence > sequence]

    def get_all_entries(self) -> List[WALEntry]:
        """Get all entries."""
        return list(self._entries)

    def _truncate(self) -> None:
        """Truncate old entries to manage size."""
        # Keep only last half
        keep = self._max_entries // 2
        self._entries = self._entries[-keep:]

        # Rewrite WAL file
        with open(self._path, "w", encoding="utf-8") as f:
            for entry in self._entries:
                f.write(json.dumps(entry.to_dict(), default=str) + "\n")

        logger.info(f"WAL truncated to {len(self._entries)} entries")


class StatePersistence:
    """
    Robust state persistence manager.

    Features:
    - Atomic writes (temp file + rename)
    - Automatic backups
    - WAL for crash recovery
    - Checksum verification

    Example:
        >>> persistence = StatePersistence(Path("engine_state.json"))
        >>> persistence.save(state.to_dict())
        >>> data = persistence.load()
    """

    def __init__(
        self,
        path: Path,
        backup_count: Optional[int] = None,
        use_wal: Optional[bool] = None,
        clock: Optional[Clock] = None,
    ):
        """
        Initialize persistence manager.

        Args:
            path: Primary state file path
            backup_count: Number of backups to keep (default from config)
            use_wal: Enable write-ahead logging (default from config)
            clock: Clock for timestamps
        """
        self._path = path
        self._backup_count = backup_count if backup_count is not None else get_cfg(
            "live_trading.persistence.backup_count",
            default=5,
        )
        self._clock = clock or get_clock()

        # Setup WAL
        use_wal_cfg = use_wal if use_wal is not None else get_cfg(
            "live_trading.persistence.use_wal",
            default=True,
        )
        self._wal: Optional[WriteAheadLog] = None
        if use_wal_cfg:
            wal_path = path.with_suffix(".wal")
            self._wal = WriteAheadLog(wal_path, clock=clock)

        # Metadata
        self._last_save: Optional[datetime] = None
        self._last_checksum: Optional[str] = None

    @property
    def path(self) -> Path:
        """Get primary state file path."""
        return self._path

    @property
    def wal(self) -> Optional[WriteAheadLog]:
        """Get WAL instance."""
        return self._wal

    @property
    def last_save(self) -> Optional[datetime]:
        """Get timestamp of last successful save."""
        return self._last_save

    @property
    def last_checksum(self) -> Optional[str]:
        """Get checksum of last saved state."""
        return self._last_checksum

    @property
    def backup_count(self) -> int:
        """Get number of backups to keep."""
        return self._backup_count

    def save(self, data: Dict[str, Any]) -> bool:
        """
        Save state atomically with backup.

        Args:
            data: State data to save

        Returns:
            True if save succeeded
        """
        try:
            # Compute checksum
            checksum = compute_checksum(data)
            data_with_meta = {
                **data,
                "_meta": {
                    "checksum": checksum,
                    "saved_at": self._clock.now().isoformat(),
                    "version": 1,
                },
            }

            # Create backup of existing state
            if self._path.exists():
                self._rotate_backups()

            # Atomic write: write to temp file, then rename
            temp_path = self._path.with_suffix(".tmp")
            self._path.parent.mkdir(parents=True, exist_ok=True)

            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data_with_meta, f, indent=2, sort_keys=True, default=str)
                f.flush()

            # Atomic rename
            temp_path.rename(self._path)

            # Update metadata
            self._last_save = self._clock.now()
            self._last_checksum = checksum

            # Checkpoint WAL (state is now safe)
            if self._wal:
                self._wal.checkpoint()

            logger.debug(f"State saved to {self._path} (checksum: {checksum})")
            return True

        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False

    def load(self, validate: bool = True) -> Optional[Dict[str, Any]]:
        """
        Load state from file.

        Args:
            validate: Verify checksum

        Returns:
            State data, or None if load failed
        """
        if not self._path.exists():
            logger.info(f"State file not found: {self._path}")
            return None

        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Verify checksum
            if validate and "_meta" in data:
                expected = data["_meta"].get("checksum")
                # Remove meta before checksum verification
                data_without_meta = {k: v for k, v in data.items() if k != "_meta"}
                if expected and not verify_checksum(data_without_meta, expected):
                    logger.error("State checksum mismatch - file may be corrupted")
                    return self._try_recover()

            # Remove metadata before returning
            data.pop("_meta", None)

            logger.debug(f"State loaded from {self._path}")
            return data

        except json.JSONDecodeError as e:
            logger.error(f"State file corrupt (invalid JSON): {e}")
            return self._try_recover()

        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return self._try_recover()

    def _try_recover(self) -> Optional[Dict[str, Any]]:
        """Attempt recovery from backup or WAL."""
        logger.info("Attempting state recovery...")

        # Try backups first (most recent to oldest)
        for i in range(1, self._backup_count + 1):
            backup_path = self._path.with_suffix(f".bak{i}")
            if backup_path.exists():
                logger.info(f"Trying backup: {backup_path}")
                try:
                    with open(backup_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    data.pop("_meta", None)
                    logger.info(f"Recovered from backup: {backup_path}")
                    return data
                except Exception:
                    continue

        # Try WAL replay
        if self._wal and self._wal._entries:
            logger.info("Attempting WAL replay...")
            return self._replay_wal()

        logger.error("All recovery attempts failed")
        return None

    def _replay_wal(self) -> Optional[Dict[str, Any]]:
        """Replay WAL to reconstruct state."""
        if not self._wal:
            return None

        # Find last known good state (from backup)
        base_state: Dict[str, Any] = {}
        base_seq = 0

        for i in range(1, self._backup_count + 1):
            backup_path = self._path.with_suffix(f".bak{i}")
            if backup_path.exists():
                try:
                    with open(backup_path, "r", encoding="utf-8") as f:
                        base_state = json.load(f)
                    base_state.pop("_meta", None)
                    break
                except Exception:
                    continue

        # Replay WAL entries
        entries = self._wal.get_entries_since(base_seq)
        for entry in entries:
            # Apply operation to state
            if entry.operation == "full_state":
                base_state = entry.data
            elif entry.operation == "update_position":
                if "positions" not in base_state:
                    base_state["positions"] = {}
                symbol = entry.data.get("symbol")
                if symbol:
                    base_state["positions"][symbol] = entry.data

        logger.info(f"Replayed {len(entries)} WAL entries")
        return base_state if base_state else None

    def _rotate_backups(self) -> None:
        """Rotate backup files."""
        # Delete oldest backup
        oldest = self._path.with_suffix(f".bak{self._backup_count}")
        oldest.unlink(missing_ok=True)

        # Rotate existing backups
        for i in range(self._backup_count - 1, 0, -1):
            src = self._path.with_suffix(f".bak{i}")
            dst = self._path.with_suffix(f".bak{i + 1}")
            if src.exists():
                src.rename(dst)

        # Current state becomes backup 1
        if self._path.exists():
            shutil.copy2(self._path, self._path.with_suffix(".bak1"))

    def log_operation(self, operation: str, data: Dict[str, Any]) -> None:
        """Log operation to WAL (before applying)."""
        if self._wal:
            self._wal.append(operation, data)

    def get_backup_paths(self) -> List[Path]:
        """Get list of available backup files."""
        backups = []
        for i in range(1, self._backup_count + 1):
            backup_path = self._path.with_suffix(f".bak{i}")
            if backup_path.exists():
                backups.append(backup_path)
        return backups

    def delete_all(self) -> None:
        """Delete state file and all backups (for testing)."""
        self._path.unlink(missing_ok=True)
        for i in range(1, self._backup_count + 1):
            backup_path = self._path.with_suffix(f".bak{i}")
            backup_path.unlink(missing_ok=True)
        if self._wal:
            self._wal._path.unlink(missing_ok=True)


class StateManager(Generic[T]):
    """
    High-level state manager for trading engine.

    Wraps StatePersistence with typed operations.

    Example:
        >>> manager = StateManager(Path("state"), state_factory=EngineState)
        >>> state = manager.load_or_create()
        >>> manager.update_position("AAPL", 100, 150.0)
        >>> manager.save()
    """

    def __init__(
        self,
        state_dir: Path,
        state_factory: Callable[[], T],
        state_from_dict: Callable[[Dict[str, Any]], T],
        clock: Optional[Clock] = None,
    ):
        """
        Initialize state manager.

        Args:
            state_dir: Directory for state files
            state_factory: Factory function to create new state
            state_from_dict: Function to reconstruct state from dict
            clock: Clock for timestamps
        """
        self._state_dir = state_dir
        self._state_factory = state_factory
        self._state_from_dict = state_from_dict
        self._clock = clock or get_clock()

        # Setup persistence
        state_path = state_dir / "engine_state.json"
        self._persistence = StatePersistence(state_path, clock=clock)

        self._state: Optional[T] = None
        self._dirty = False

    @property
    def state(self) -> T:
        """Get current state (loads if needed)."""
        if self._state is None:
            self._state = self.load_or_create()
        return self._state

    @property
    def persistence(self) -> StatePersistence:
        """Get underlying persistence manager."""
        return self._persistence

    def load_or_create(self) -> T:
        """Load existing state or create new."""
        data = self._persistence.load()

        if data is not None:
            # Reconstruct state from dict
            return self._state_from_dict(data)
        else:
            logger.info("Creating new state")
            return self._state_factory()

    def save(self) -> bool:
        """Save current state."""
        if self._state is None:
            return False

        # Get state as dict (assumes state has to_dict method)
        state_dict = self._state.to_dict()  # type: ignore

        # Log to WAL first
        self._persistence.log_operation("full_state", state_dict)

        # Then save
        success = self._persistence.save(state_dict)
        if success:
            self._dirty = False

        return success

    def mark_dirty(self) -> None:
        """Mark state as having unsaved changes."""
        self._dirty = True

    @property
    def has_unsaved_changes(self) -> bool:
        """Check if there are unsaved changes."""
        return self._dirty

    def reload(self) -> T:
        """Force reload state from disk."""
        self._state = self.load_or_create()
        self._dirty = False
        return self._state
