# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""Tool implementations for FoxML Artifact MCP Server."""

import json
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime

from .index import get_index, RunMetadata


def query_runs(
    experiment_name: Optional[str] = None,
    git_sha: Optional[str] = None,
    config_fingerprint: Optional[str] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    is_comparable: Optional[bool] = None,
    limit: int = 20
) -> Dict[str, Any]:
    """
    Query runs with filters.

    Args:
        experiment_name: Filter by experiment name
        git_sha: Filter by git SHA prefix
        config_fingerprint: Filter by config fingerprint prefix
        date_start: Filter by start date (ISO format)
        date_end: Filter by end date (ISO format)
        is_comparable: Filter by comparability
        limit: Maximum results

    Returns:
        Dict with runs list and count
    """
    index = get_index()
    runs = index.query(
        experiment_name=experiment_name,
        git_sha=git_sha,
        config_fingerprint=config_fingerprint,
        date_start=date_start,
        date_end=date_end,
        is_comparable=is_comparable,
        limit=limit
    )

    return {
        "runs": [r.to_dict() for r in runs],
        "count": len(runs),
        "filters_applied": {
            k: v for k, v in {
                "experiment_name": experiment_name,
                "git_sha": git_sha,
                "config_fingerprint": config_fingerprint,
                "date_start": date_start,
                "date_end": date_end,
                "is_comparable": is_comparable
            }.items() if v is not None
        },
        "limit": limit
    }


def get_run_details(
    run_id: str,
    include_config: bool = False,
    include_targets: bool = True
) -> Dict[str, Any]:
    """
    Get detailed information about a specific run.

    Args:
        run_id: Run identifier
        include_config: Whether to include resolved config
        include_targets: Whether to include target index

    Returns:
        Dict with run details
    """
    index = get_index()
    metadata = index.get_run(run_id)

    if not metadata:
        # Try to find similar run IDs
        all_runs = index.build_index()
        suggestions = [
            rid for rid in all_runs.keys()
            if run_id.lower() in rid.lower() or rid.lower() in run_id.lower()
        ][:5]

        return {
            "run_id": run_id,
            "error": f"Run '{run_id}' not found",
            "suggestions": suggestions if suggestions else None
        }

    manifest = index.get_manifest(run_id)

    result = {
        "run_id": run_id,
        "run_instance_id": metadata.run_instance_id,
        "is_comparable": metadata.is_comparable,
        "created_at": metadata.created_at.isoformat(),
        "experiment_name": metadata.experiment_name,
        "git_sha": metadata.git_sha,
        "run_dir": str(metadata.run_dir)
    }

    if manifest:
        result["manifest"] = {
            "config_digest": manifest.get("config_digest"),
            "plan_hashes": manifest.get("plan_hashes"),
            "run_hash": manifest.get("run_hash"),
            "run_metadata": manifest.get("run_metadata")
        }

    if include_targets and manifest:
        result["target_index"] = manifest.get("target_index", {})
        result["targets"] = metadata.targets

    if include_config:
        config = index.get_resolved_config(run_id)
        if config:
            result["config"] = config

    return result


def query_targets(
    run_id: str,
    target: Optional[str] = None
) -> Dict[str, Any]:
    """
    Query targets for a run.

    Args:
        run_id: Run identifier
        target: Optional specific target to query

    Returns:
        Dict with targets information
    """
    index = get_index()
    manifest = index.get_manifest(run_id)

    if not manifest:
        return {
            "run_id": run_id,
            "error": f"Run '{run_id}' not found"
        }

    target_index = manifest.get("target_index", {})

    if target:
        # Filter to specific target
        if target not in target_index:
            return {
                "run_id": run_id,
                "target": target,
                "error": f"Target '{target}' not found in run",
                "available_targets": list(target_index.keys())
            }

        target_info = target_index[target]
        return {
            "run_id": run_id,
            "target": target,
            "decision": target_info.get("decision", []),
            "models": target_info.get("models", {}),
            "metrics": target_info.get("metrics", []),
            "trends": target_info.get("trends", []),
            "reproducibility": target_info.get("reproducibility", [])
        }

    # Return all targets
    targets_summary = []
    for tgt_name, tgt_info in sorted(target_index.items()):
        model_families = list(tgt_info.get("models", {}).keys())
        targets_summary.append({
            "target": tgt_name,
            "model_families": model_families,
            "model_count": len(model_families),
            "has_decision": bool(tgt_info.get("decision")),
            "has_metrics": bool(tgt_info.get("metrics"))
        })

    return {
        "run_id": run_id,
        "targets": targets_summary,
        "count": len(targets_summary)
    }


def compare_runs(
    run_id_1: str,
    run_id_2: str,
    compare_config: bool = True,
    compare_targets: bool = True
) -> Dict[str, Any]:
    """
    Compare two runs.

    Args:
        run_id_1: First run identifier
        run_id_2: Second run identifier
        compare_config: Whether to compare configs
        compare_targets: Whether to compare targets

    Returns:
        Dict with comparison results
    """
    index = get_index()

    run1 = index.get_run(run_id_1)
    run2 = index.get_run(run_id_2)

    if not run1:
        return {"error": f"Run '{run_id_1}' not found"}
    if not run2:
        return {"error": f"Run '{run_id_2}' not found"}

    differences = {}

    # Compare basic metadata
    if run1.git_sha != run2.git_sha:
        differences["git_sha"] = {
            "run_1": run1.git_sha,
            "run_2": run2.git_sha
        }

    if run1.experiment_name != run2.experiment_name:
        differences["experiment_name"] = {
            "run_1": run1.experiment_name,
            "run_2": run2.experiment_name
        }

    if run1.deterministic_config_fingerprint != run2.deterministic_config_fingerprint:
        differences["deterministic_config_fingerprint"] = {
            "run_1": run1.deterministic_config_fingerprint[:16] + "..." if run1.deterministic_config_fingerprint else None,
            "run_2": run2.deterministic_config_fingerprint[:16] + "..." if run2.deterministic_config_fingerprint else None
        }

    # Compare targets
    if compare_targets:
        targets1 = set(run1.targets)
        targets2 = set(run2.targets)

        if targets1 != targets2:
            differences["targets"] = {
                "only_in_run_1": sorted(targets1 - targets2),
                "only_in_run_2": sorted(targets2 - targets1),
                "common": sorted(targets1 & targets2)
            }

    # Compare configs
    if compare_config:
        config1 = index.get_resolved_config(run_id_1)
        config2 = index.get_resolved_config(run_id_2)

        if config1 and config2:
            config_diffs = _diff_configs(config1, config2)
            if config_diffs:
                differences["config_diff"] = config_diffs

    return {
        "run_1": run1.to_dict(),
        "run_2": run2.to_dict(),
        "differences": differences,
        "are_identical": len(differences) == 0,
        "time_difference": str(run1.created_at - run2.created_at)
    }


def _diff_configs(config1: Dict, config2: Dict, path: str = "") -> List[Dict]:
    """Recursively diff two config dicts."""
    diffs = []

    # Get all keys
    all_keys = set(config1.keys()) | set(config2.keys())

    for key in sorted(all_keys):
        current_path = f"{path}.{key}" if path else key

        # Skip run_id, timestamp, and fingerprint fields
        if key in ("run_id", "timestamp", "config_fingerprint", "deterministic_config_fingerprint"):
            continue

        in_1 = key in config1
        in_2 = key in config2

        if not in_1:
            diffs.append({
                "path": current_path,
                "type": "added_in_run_2",
                "value": config2[key]
            })
        elif not in_2:
            diffs.append({
                "path": current_path,
                "type": "removed_in_run_2",
                "value": config1[key]
            })
        elif isinstance(config1[key], dict) and isinstance(config2[key], dict):
            # Recurse into nested dicts
            nested_diffs = _diff_configs(config1[key], config2[key], current_path)
            diffs.extend(nested_diffs)
        elif config1[key] != config2[key]:
            diffs.append({
                "path": current_path,
                "type": "changed",
                "run_1": config1[key],
                "run_2": config2[key]
            })

    return diffs


def get_target_stage_history(
    run_id: str,
    target: str
) -> Dict[str, Any]:
    """
    Get stage progression history for a target.

    Args:
        run_id: Run identifier
        target: Target name

    Returns:
        Dict with stage history
    """
    index = get_index()
    metadata = index.get_run(run_id)

    if not metadata:
        return {
            "run_id": run_id,
            "error": f"Run '{run_id}' not found"
        }

    target_dir = metadata.run_dir / "targets" / target

    if not target_dir.exists():
        manifest = index.get_manifest(run_id)
        target_index = manifest.get("target_index", {}) if manifest else {}
        return {
            "run_id": run_id,
            "target": target,
            "error": f"Target directory not found",
            "available_targets": list(target_index.keys())
        }

    # Look for reproducibility directory with stage information
    repro_dir = target_dir / "reproducibility"
    stages = []

    if repro_dir.exists():
        # Look for view directories (CROSS_SECTIONAL, SYMBOL_SPECIFIC)
        for view_dir in sorted(repro_dir.iterdir()):
            if view_dir.is_dir():
                view_name = view_dir.name

                # Look for cohort directories
                for cohort_path in sorted(view_dir.rglob("cohort=*")):
                    if cohort_path.is_dir():
                        cohort_id = cohort_path.name.replace("cohort=", "")
                        metadata_file = cohort_path / "metadata.json"

                        stage_info = {
                            "cohort_id": cohort_id,
                            "view": view_name,
                            "path": str(cohort_path.relative_to(metadata.run_dir))
                        }

                        if metadata_file.exists():
                            try:
                                with open(metadata_file, 'r') as f:
                                    cohort_meta = json.load(f)
                                    stage_info["stage"] = cohort_meta.get("stage")
                                    stage_info["n_effective"] = cohort_meta.get("n_effective")
                                    stage_info["date_range"] = {
                                        "start": cohort_meta.get("date_start"),
                                        "end": cohort_meta.get("date_end")
                                    }
                            except Exception:
                                pass

                        stages.append(stage_info)

    return {
        "run_id": run_id,
        "target": target,
        "stages": stages,
        "stage_count": len(stages)
    }


def search_experiments(
    name_pattern: Optional[str] = None,
    limit: int = 20
) -> Dict[str, Any]:
    """
    Search and list experiments.

    Args:
        name_pattern: Optional pattern to filter experiment names
        limit: Maximum results

    Returns:
        Dict with experiments list
    """
    index = get_index()
    experiments = index.list_experiments()

    # Filter by pattern if provided
    if name_pattern:
        pattern_lower = name_pattern.lower()
        experiments = {
            name: runs for name, runs in experiments.items()
            if pattern_lower in name.lower()
        }

    # Build experiment summaries
    experiment_summaries = []
    for name, run_ids in sorted(experiments.items()):
        # Get details for the latest run
        runs = [index.get_run(rid) for rid in run_ids]
        runs = [r for r in runs if r is not None]

        if runs:
            # Sort by created_at
            runs.sort(key=lambda r: r.created_at, reverse=True)
            latest = runs[0]

            git_shas = list(set(r.git_sha for r in runs if r.git_sha))

            experiment_summaries.append({
                "name": name,
                "run_count": len(runs),
                "latest_run": latest.run_id,
                "latest_created_at": latest.created_at.isoformat(),
                "git_shas": git_shas[:5],  # Limit to 5 unique SHAs
                "targets": latest.targets[:5] if latest.targets else []  # Sample targets
            })

    # Sort by latest run date and limit
    experiment_summaries.sort(key=lambda e: e["latest_created_at"], reverse=True)
    experiment_summaries = experiment_summaries[:limit]

    return {
        "experiments": experiment_summaries,
        "count": len(experiment_summaries),
        "name_pattern": name_pattern
    }


def get_model_metrics(
    run_id: str,
    target: str,
    family: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get model performance metrics for a target.

    Args:
        run_id: Run identifier
        target: Target name
        family: Optional specific model family to query

    Returns:
        Dict with model metrics
    """
    index = get_index()
    metadata = index.get_run(run_id)

    if not metadata:
        return {
            "run_id": run_id,
            "error": f"Run '{run_id}' not found"
        }

    target_dir = metadata.run_dir / "targets" / target

    if not target_dir.exists():
        manifest = index.get_manifest(run_id)
        target_index = manifest.get("target_index", {}) if manifest else {}
        return {
            "run_id": run_id,
            "target": target,
            "error": f"Target directory not found",
            "available_targets": list(target_index.keys())
        }

    metrics_dir = target_dir / "metrics"
    models_dir = target_dir / "models"

    result = {
        "run_id": run_id,
        "target": target,
        "metrics": [],
        "model_families": []
    }

    # Collect metrics from metrics directory
    if metrics_dir.exists():
        for view_dir in sorted(metrics_dir.iterdir()):
            if view_dir.is_dir() and view_dir.name.startswith("view="):
                view_name = view_dir.name.replace("view=", "")
                metrics_file = view_dir / "metrics.json"

                if metrics_file.exists():
                    try:
                        with open(metrics_file, 'r') as f:
                            metrics_data = json.load(f)
                            result["metrics"].append({
                                "view": view_name,
                                "auc": metrics_data.get("auc") or metrics_data.get("primary_metric", {}).get("value"),
                                "std": metrics_data.get("std_score") or metrics_data.get("primary_metric", {}).get("std"),
                                "composite_score": metrics_data.get("composite_score") or metrics_data.get("score", {}).get("composite"),
                                "metric_name": metrics_data.get("metric_name"),
                                "sample_size": metrics_data.get("sample_size") or metrics_data.get("n_samples")
                            })
                    except Exception:
                        pass

    # Collect model family information
    if models_dir.exists():
        for family_dir in sorted(models_dir.iterdir()):
            if family_dir.is_dir():
                family_name = family_dir.name

                # Filter by family if specified
                if family and family.lower() != family_name.lower():
                    continue

                family_info = {
                    "family": family_name,
                    "artifacts": []
                }

                # Look for model artifacts
                for artifact in sorted(family_dir.rglob("*")):
                    if artifact.is_file():
                        rel_path = str(artifact.relative_to(family_dir))
                        family_info["artifacts"].append(rel_path)

                # Look for metrics specific to this family
                family_metrics_file = family_dir / "metrics.json"
                if family_metrics_file.exists():
                    try:
                        with open(family_metrics_file, 'r') as f:
                            family_metrics = json.load(f)
                            family_info["metrics"] = family_metrics
                    except Exception:
                        pass

                result["model_families"].append(family_info)

    result["family_count"] = len(result["model_families"])
    return result


def diff_target_results(
    run_id_1: str,
    run_id_2: str,
    target: str
) -> Dict[str, Any]:
    """
    Compare target results between two runs.

    Args:
        run_id_1: First run identifier
        run_id_2: Second run identifier
        target: Target name to compare

    Returns:
        Dict with target comparison results
    """
    index = get_index()

    run1 = index.get_run(run_id_1)
    run2 = index.get_run(run_id_2)

    if not run1:
        return {"error": f"Run '{run_id_1}' not found"}
    if not run2:
        return {"error": f"Run '{run_id_2}' not found"}

    # Get target directories
    target_dir_1 = run1.run_dir / "targets" / target
    target_dir_2 = run2.run_dir / "targets" / target

    result = {
        "run_id_1": run_id_1,
        "run_id_2": run_id_2,
        "target": target,
        "comparison": {}
    }

    # Check if target exists in both runs
    if not target_dir_1.exists():
        result["comparison"]["run_1_missing"] = True
    if not target_dir_2.exists():
        result["comparison"]["run_2_missing"] = True

    if result["comparison"].get("run_1_missing") or result["comparison"].get("run_2_missing"):
        return result

    # Compare model families
    models_dir_1 = target_dir_1 / "models"
    models_dir_2 = target_dir_2 / "models"

    families_1 = set()
    families_2 = set()

    if models_dir_1.exists():
        families_1 = {d.name for d in models_dir_1.iterdir() if d.is_dir()}
    if models_dir_2.exists():
        families_2 = {d.name for d in models_dir_2.iterdir() if d.is_dir()}

    result["comparison"]["model_families"] = {
        "only_in_run_1": sorted(families_1 - families_2),
        "only_in_run_2": sorted(families_2 - families_1),
        "common": sorted(families_1 & families_2)
    }

    # Compare metrics
    metrics_1 = _load_target_metrics(target_dir_1)
    metrics_2 = _load_target_metrics(target_dir_2)

    metric_diffs = []
    all_views = set(metrics_1.keys()) | set(metrics_2.keys())

    for view in sorted(all_views):
        m1 = metrics_1.get(view, {})
        m2 = metrics_2.get(view, {})

        if m1 or m2:
            auc_1 = m1.get("auc")
            auc_2 = m2.get("auc")

            diff = {
                "view": view,
                "run_1_auc": auc_1,
                "run_2_auc": auc_2
            }

            if auc_1 is not None and auc_2 is not None:
                diff["auc_delta"] = round(auc_2 - auc_1, 6)
                diff["auc_pct_change"] = round((auc_2 - auc_1) / auc_1 * 100, 2) if auc_1 != 0 else None

            metric_diffs.append(diff)

    result["comparison"]["metrics"] = metric_diffs

    # Compare decisions
    decision_dir_1 = target_dir_1 / "decision"
    decision_dir_2 = target_dir_2 / "decision"

    decisions_1 = _load_decisions(decision_dir_1)
    decisions_2 = _load_decisions(decision_dir_2)

    if decisions_1 or decisions_2:
        result["comparison"]["decisions"] = {
            "run_1": decisions_1,
            "run_2": decisions_2,
            "changed": decisions_1 != decisions_2
        }

    return result


def _load_target_metrics(target_dir: Path) -> Dict[str, Dict]:
    """Load metrics from target metrics directory."""
    metrics_dir = target_dir / "metrics"
    metrics = {}

    if not metrics_dir.exists():
        return metrics

    for view_dir in sorted(metrics_dir.iterdir()):
        if view_dir.is_dir() and view_dir.name.startswith("view="):
            view_name = view_dir.name.replace("view=", "")
            metrics_file = view_dir / "metrics.json"

            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r') as f:
                        data = json.load(f)
                        metrics[view_name] = {
                            "auc": data.get("auc") or data.get("primary_metric", {}).get("value"),
                            "std": data.get("std_score") or data.get("primary_metric", {}).get("std"),
                            "composite_score": data.get("composite_score") or data.get("score", {}).get("composite")
                        }
                except Exception:
                    pass

    return metrics


def _load_decisions(decision_dir: Path) -> Dict[str, Any]:
    """Load decision files from target decision directory."""
    decisions = {}

    if not decision_dir.exists():
        return decisions

    for decision_file in sorted(decision_dir.iterdir()):
        if decision_file.is_file() and decision_file.suffix in [".json", ".yaml"]:
            decision_type = decision_file.stem
            try:
                with open(decision_file, 'r') as f:
                    if decision_file.suffix == ".json":
                        decisions[decision_type] = json.load(f)
                    else:
                        import yaml
                        decisions[decision_type] = yaml.safe_load(f)
            except Exception:
                decisions[decision_type] = {"error": "Failed to load"}

    return decisions
