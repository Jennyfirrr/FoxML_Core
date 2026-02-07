# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""Run indexing system for FoxML Artifact Server."""

import json
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class RunMetadata:
    """Metadata for a single run."""
    run_id: str
    run_instance_id: str
    is_comparable: bool
    created_at: datetime
    experiment_name: Optional[str]
    git_sha: str
    config_fingerprint: str
    deterministic_config_fingerprint: str
    targets: List[str]
    manifest_path: Path
    run_dir: Path

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "run_instance_id": self.run_instance_id,
            "is_comparable": self.is_comparable,
            "created_at": self.created_at.isoformat(),
            "experiment_name": self.experiment_name,
            "git_sha": self.git_sha,
            "config_fingerprint": self.config_fingerprint[:16] + "..." if self.config_fingerprint else None,
            "deterministic_config_fingerprint": self.deterministic_config_fingerprint[:16] + "..." if self.deterministic_config_fingerprint else None,
            "targets": self.targets,
            "target_count": len(self.targets)
        }


class RunIndex:
    """
    Index of all runs in RESULTS/runs directory.

    Scans run directories and caches metadata for fast querying.
    """

    def __init__(self, results_dir: Optional[Path] = None):
        if results_dir is None:
            # Default to RESULTS directory
            results_dir = Path(__file__).resolve().parents[2] / "RESULTS"
        self.results_dir = Path(results_dir)
        self.runs_dir = self.results_dir / "runs"
        self._index: Dict[str, RunMetadata] = {}
        self._last_build: Optional[float] = None
        self._cache_ttl: float = 300.0  # 5 minutes

    def _needs_refresh(self) -> bool:
        """Check if index needs refresh."""
        if not self._index or self._last_build is None:
            return True
        return (time.time() - self._last_build) > self._cache_ttl

    def build_index(self, force: bool = False) -> Dict[str, RunMetadata]:
        """
        Scan RESULTS/runs and build index.

        Args:
            force: Force rebuild even if cache is valid

        Returns:
            Dict mapping run_id to RunMetadata
        """
        if not force and not self._needs_refresh():
            return self._index

        if not self.runs_dir.exists():
            self._index = {}
            self._last_build = time.time()
            return self._index

        index = {}

        # Scan all comparison_group directories (or directly run directories)
        for item in sorted(self.runs_dir.iterdir()):
            if not item.is_dir():
                continue

            # Check if this is a run directory (has manifest.json)
            manifest_path = item / "manifest.json"
            if manifest_path.exists():
                # Direct run directory
                metadata = self._parse_manifest(manifest_path, item)
                if metadata:
                    index[metadata.run_id] = metadata
            else:
                # Might be a comparison_group directory
                for run_dir in sorted(item.iterdir()):
                    if not run_dir.is_dir():
                        continue
                    manifest_path = run_dir / "manifest.json"
                    if manifest_path.exists():
                        metadata = self._parse_manifest(manifest_path, run_dir)
                        if metadata:
                            index[metadata.run_id] = metadata

        self._index = index
        self._last_build = time.time()
        return index

    def _parse_manifest(self, manifest_path: Path, run_dir: Path) -> Optional[RunMetadata]:
        """Parse a manifest.json file into RunMetadata."""
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            # Extract required fields
            run_id = manifest.get("run_id", "")
            if not run_id:
                return None

            # Parse created_at
            created_at_str = manifest.get("created_at", "")
            try:
                created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                created_at = datetime.now()

            return RunMetadata(
                run_id=run_id,
                run_instance_id=manifest.get("run_instance_id", ""),
                is_comparable=manifest.get("is_comparable", False),
                created_at=created_at,
                experiment_name=manifest.get("experiment", {}).get("name") if manifest.get("experiment") else None,
                git_sha=manifest.get("git_sha", ""),
                config_fingerprint=manifest.get("config_digest", "") or "",
                deterministic_config_fingerprint=manifest.get("deterministic_config_fingerprint", "") or "",
                targets=manifest.get("targets", []),
                manifest_path=manifest_path,
                run_dir=run_dir
            )

        except Exception:
            return None

    def query(
        self,
        experiment_name: Optional[str] = None,
        git_sha: Optional[str] = None,
        config_fingerprint: Optional[str] = None,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None,
        is_comparable: Optional[bool] = None,
        limit: int = 20
    ) -> List[RunMetadata]:
        """
        Filter runs by criteria.

        Args:
            experiment_name: Filter by experiment name
            git_sha: Filter by git SHA prefix
            config_fingerprint: Filter by config fingerprint prefix
            date_start: Filter by start date (ISO format)
            date_end: Filter by end date (ISO format)
            is_comparable: Filter by comparability
            limit: Maximum results

        Returns:
            List of matching RunMetadata, sorted by created_at descending
        """
        self.build_index()

        results = list(self._index.values())

        # Apply filters
        if experiment_name:
            results = [r for r in results if r.experiment_name == experiment_name]

        if git_sha:
            results = [r for r in results if r.git_sha.startswith(git_sha)]

        if config_fingerprint:
            results = [r for r in results if r.config_fingerprint.startswith(config_fingerprint)]

        if is_comparable is not None:
            results = [r for r in results if r.is_comparable == is_comparable]

        if date_start:
            try:
                start = datetime.fromisoformat(date_start)
                results = [r for r in results if r.created_at >= start]
            except ValueError:
                pass

        if date_end:
            try:
                end = datetime.fromisoformat(date_end)
                results = [r for r in results if r.created_at <= end]
            except ValueError:
                pass

        # Sort by created_at descending (newest first)
        results.sort(key=lambda r: r.created_at, reverse=True)

        return results[:limit]

    def get_run(self, run_id: str) -> Optional[RunMetadata]:
        """Get a specific run by ID."""
        self.build_index()
        return self._index.get(run_id)

    def get_manifest(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load full manifest for a run."""
        metadata = self.get_run(run_id)
        if not metadata:
            return None

        try:
            with open(metadata.manifest_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    def get_resolved_config(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load resolved config for a run."""
        metadata = self.get_run(run_id)
        if not metadata:
            return None

        config_path = metadata.run_dir / "globals" / "config.resolved.json"
        if not config_path.exists():
            return None

        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    def list_experiments(self) -> Dict[str, List[str]]:
        """List all experiments and their run IDs."""
        self.build_index()

        experiments: Dict[str, List[str]] = {}
        for run in self._index.values():
            exp_name = run.experiment_name or "(unnamed)"
            if exp_name not in experiments:
                experiments[exp_name] = []
            experiments[exp_name].append(run.run_id)

        return experiments


# Singleton instance
_index_instance: Optional[RunIndex] = None


def get_index() -> RunIndex:
    """Get the singleton index instance."""
    global _index_instance
    if _index_instance is None:
        _index_instance = RunIndex()
    return _index_instance
