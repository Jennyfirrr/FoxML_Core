# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Determinism Ordering Helpers

Canonical helpers for deterministic filesystem and container iteration.
These functions ensure consistent ordering across runs, preventing non-determinism
in financial outputs (target rankings, feature selection, routing decisions).

CRITICAL: Use these helpers in all Tier A files (files that affect financial outputs).
"""

from pathlib import Path
from typing import Iterator, List, Dict, Tuple, Any, Optional, Callable, Iterable, Union
from datetime import datetime


def iterdir_sorted(
    path: Path,
    *,
    key: Optional[Callable[[Path], Any]] = None,
    filter_fn: Optional[Callable[[Path], bool]] = None
) -> Iterator[Path]:
    """
    Deterministic directory iteration.
    
    Args:
        path: Directory path (must exist and be a directory)
        key: Optional sort key function (default: lambda p: (not p.is_dir(), p.name))
        filter_fn: Optional filter function (e.g., skip hidden files)
    
    Yields:
        Path objects in sorted order
    
    Raises:
        ValueError: If path doesn't exist or isn't a directory
    """
    if not path.exists():
        raise ValueError(f"Path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")
    
    if key is None:
        # Default: directories first, then files, both sorted by name
        key = lambda p: (not p.is_dir(), p.name)
    
    items = list(path.iterdir())
    
    if filter_fn:
        items = [item for item in items if filter_fn(item)]
    
    for item in sorted(items, key=key):
        yield item


def glob_sorted(
    path: Path,
    pattern: str,
    *,
    key: Optional[Callable[[Path], Any]] = None,
    filter_fn: Optional[Callable[[Path], bool]] = None
) -> List[Path]:
    """
    Deterministic glob results.
    
    Args:
        path: Base path
        pattern: Glob pattern
        key: Optional sort key function (default: lambda p: p.name)
        filter_fn: Optional filter function
    
    Returns:
        Sorted list of matching paths
    """
    if key is None:
        key = lambda p: p.name
    
    matches = list(path.glob(pattern))
    
    if filter_fn:
        matches = [m for m in matches if filter_fn(m)]
    
    return sorted(matches, key=key)


def rglob_sorted(
    path: Path,
    pattern: str,
    *,
    key: Optional[Callable[[Path], Any]] = None,
    filter_fn: Optional[Callable[[Path], bool]] = None
) -> List[Path]:
    """
    Deterministic recursive glob results.
    
    CRITICAL: Sorts by relative path parts (relative to base) to avoid machine-dependent
    absolute path prefixes. Hard-fails if any path cannot be relativized to base.
    
    Args:
        path: Base path
        pattern: Glob pattern
        key: Optional sort key function (default: relative path parts)
        filter_fn: Optional filter function
    
    Returns:
        Sorted list of matching paths
    
    Raises:
        ValueError: If any matched path cannot be relativized to base path
    """
    # Resolve stable base once (normalize symlinks and absolute paths)
    base = path.resolve()
    
    if key is None:
        # Natural sort: compare relative path parts (not absolute)
        def relative_sort_key(p: Path) -> tuple:
            """Sort key using relative path parts only."""
            try:
                # Resolve p to normalize symlinks, then relativize to base
                rel = p.resolve().relative_to(base)
                return rel.parts
            except ValueError:
                # Path cannot be relativized to base - hard-fail for determinism
                raise ValueError(
                    f"Path {p} cannot be relativized to base {base}. "
                    f"All paths returned by rglob_sorted must be under the base directory "
                    f"to ensure deterministic sorting across machines."
                )
        key = relative_sort_key
    
    matches = list(path.rglob(pattern))
    
    if filter_fn:
        matches = [m for m in matches if filter_fn(m)]
    
    return sorted(matches, key=key)


def sorted_items(d: Dict, key: Optional[Callable] = None) -> Iterator[Tuple]:
    """
    Deterministic dictionary items iteration.
    
    Args:
        d: Dictionary
        key: Optional sort key function (default: lambda kv: kv[0] for key-based sorting)
    
    Yields:
        (key, value) tuples in sorted order
    """
    if key is None:
        key = lambda kv: kv[0]  # Sort by key
    for item in sorted(d.items(), key=key):
        yield item


def sorted_keys(d: Dict) -> Iterator:
    """
    Deterministic dictionary keys iteration.
    
    Args:
        d: Dictionary
    
    Yields:
        Keys in sorted order
    """
    for key in sorted(d.keys()):
        yield key


def sorted_unique(
    seq: Iterable,
    key: Optional[Callable] = None
) -> List:
    """
    Deterministic unique sequence (replaces set() when order matters).
    
    CRITICAL: This function ensures deterministic ordering even when
    input sequence order is non-deterministic (e.g., from set iteration).
    
    Args:
        seq: Input sequence
        key: Optional deduplication key function
    
    Returns:
        Sorted list of unique items (deterministic ordering)
    
    Implementation Notes:
    - If key is None: dedupe by identity, sort by item value
    - If key is provided: dedupe by key, return items sorted by key
    - For key-based deduplication: if multiple items map to same key,
      keep the first encountered (but this is deterministic because
      we sort by key, not by input order)
    """
    if key is None:
        # Simple case: dedupe by identity, sort by value
        return sorted(set(seq))
    else:
        # Key-based deduplication: build {key: item} dict, then sort by key
        # This ensures deterministic ordering even if input order varies
        keyed_dict = {}
        for item in seq:
            item_key = key(item)
            # Keep first item for each key (deterministic: we'll sort by key anyway)
            if item_key not in keyed_dict:
                keyed_dict[item_key] = item
        
        # Return items sorted by their deduplication key (deterministic)
        return sorted(keyed_dict.values(), key=key)


def select_latest_by_semantic_key(
    items: List[Path],
    key_extractor: Callable[[Path], Union[int, float, str, datetime]],
    reverse: bool = True,
    tie_breaker: Optional[Callable[[Path], Any]] = None
) -> Optional[Path]:
    """
    Select "latest" item by semantic key (not mtime).
    
    CRITICAL: key_extractor must return a comparable type (int, float, datetime, str)
    that has a monotonic ordering matching "latest" semantics.
    
    Examples of valid semantic keys:
    - attempt_id (int): 0, 1, 2, ... (higher = later)
    - run_id (int or zero-padded string): "0001", "0002", ... (higher = later)
    - timestamp (datetime or ISO string): "2026-01-01T00:00:00" (later = later)
    - cohort_id (zero-padded string): "cohort=001", "cohort=002" (higher = later)
    
    Examples of INVALID semantic keys:
    - Hash strings (e.g., "abc123"): lexical max != latest
    - Non-zero-padded numbers as strings: "10" < "2" lexically
    
    Args:
        items: List of paths
        key_extractor: Function to extract semantic key (must return comparable type)
        reverse: If True, select maximum key; if False, select minimum
        tie_breaker: Optional secondary sort key for ties (e.g., lambda p: p.name)
    
    Returns:
        Path with highest/lowest semantic key, or None if empty
    
    Raises:
        ValueError: If key_extractor returns non-comparable types or None
    """
    if not items:
        return None
    
    # Extract semantic keys
    keyed_items = []
    for item in items:
        key = key_extractor(item)
        if key is None:
            continue
        
        # Validate key is comparable
        try:
            # Test comparability
            _ = key < key  # Will raise TypeError if not comparable
        except TypeError:
            raise ValueError(
                f"key_extractor returned non-comparable type {type(key)}. "
                f"Must return int, float, datetime, or str with monotonic ordering."
            )
        
        keyed_items.append((key, item))
    
    if not keyed_items:
        return None
    
    # Sort by semantic key (with tie-breaker if provided)
    if tie_breaker:
        # Sort by (key, tie_breaker) tuple
        keyed_items.sort(
            key=lambda x: (x[0], tie_breaker(x[1])),
            reverse=reverse
        )
    else:
        # Sort by key only
        keyed_items.sort(key=lambda x: x[0], reverse=reverse)
    
    return keyed_items[0][1]


def collect_and_sort_parallel_results(
    results: Iterable[Tuple[Any, Any]],  # (key, result) pairs
    sort_key: Optional[Callable] = None,
    tie_breaker: Optional[Callable] = None
) -> List[Tuple[Any, Any]]:
    """
    Collect parallel results and sort deterministically.
    
    Args:
        results: Iterable of (key, result) tuples from parallel execution
        sort_key: Optional sort key function (default: lambda x: x[0] for key-based sorting)
        tie_breaker: Optional secondary sort key for ties
    
    Returns:
        Sorted list of (key, result) tuples
    """
    collected = list(results)
    if sort_key is None:
        sort_key = lambda x: x[0]  # Sort by key
    
    if tie_breaker:
        # Sort by (primary_key, tie_breaker) tuple
        collected.sort(key=lambda x: (sort_key(x), tie_breaker(x)))
    else:
        collected.sort(key=sort_key)
    
    return collected
