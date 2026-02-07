# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""Tool implementations for FoxML SST MCP Server."""

from typing import Any, Dict, List, Optional
from dataclasses import asdict

from .catalog_parser import get_parser, SSTHelper


# Task description to helper mapping for recommendations
TASK_PATTERNS = {
    # Config access
    "config": ["get_cfg", "load_threshold", "load_routing_thresholds"],
    "configuration": ["get_cfg", "load_threshold", "load_routing_thresholds"],
    "setting": ["get_cfg", "load_threshold"],
    "threshold": ["load_threshold", "load_routing_thresholds", "apply_dev_mode_relaxation"],
    "routing threshold": ["load_routing_thresholds", "apply_dev_mode_relaxation"],

    # Path construction
    "path": ["get_target_dir", "get_scoped_artifact_dir", "get_globals_dir"],
    "target directory": ["get_target_dir", "get_target_decision_dir", "get_target_models_dir"],
    "target path": ["get_target_dir", "normalize_target_name"],
    "artifact": ["get_scoped_artifact_dir", "get_target_metrics_dir"],
    "model directory": ["get_target_models_dir", "model_output_dir"],
    "reproducibility": ["get_target_reproducibility_dir", "target_repro_dir"],
    "cohort": ["find_cohort_dir_by_id", "build_target_cohort_dir", "find_cohort_dirs"],
    "globals": ["get_globals_dir", "get_global_trends_dir", "globals_dir"],
    "run root": ["run_root", "training_results_root"],

    # Normalization
    "normalize": ["normalize_family", "normalize_target_name"],
    "family": ["normalize_family", "is_trainer_family", "filter_trainers"],
    "model family": ["normalize_family", "is_trainer_family", "filter_trainers"],
    "target name": ["normalize_target_name", "resolve_target_horizon_minutes"],
    "horizon": ["resolve_target_horizon_minutes"],

    # Determinism
    "determinism": ["iterdir_sorted", "sorted_items", "glob_sorted", "rglob_sorted"],
    "iterate": ["iterdir_sorted", "sorted_items"],
    "iteration": ["iterdir_sorted", "sorted_items"],
    "dict": ["sorted_items"],
    "filesystem": ["iterdir_sorted", "glob_sorted", "rglob_sorted"],
    "glob": ["glob_sorted", "rglob_sorted"],

    # Feature tracking
    "feature drop": ["track_feature_drops", "validate_feature_drops"],
    "feature": ["track_feature_drops", "validate_feature_drops"],

    # Tracker
    "tracker": ["tracker_input_adapter"],
    "enum": ["tracker_input_adapter"],
}


def search_sst_helpers(
    query: str,
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    max_results: int = 10
) -> Dict[str, Any]:
    """
    Search SST helpers by query.

    Args:
        query: Search query string
        category: Optional category filter
        subcategory: Optional subcategory filter (requires category)
        max_results: Maximum results to return

    Returns:
        Dict with query, results list, and count
    """
    parser = get_parser()

    if subcategory and category:
        # Filter by subcategory
        helpers = parser.get_helpers_by_subcategory(category, subcategory)
        if not helpers:
            return {
                "query": query,
                "category": category,
                "subcategory": subcategory,
                "results": [],
                "count": 0,
                "error": f"Subcategory '{subcategory}' not found in '{category}'"
            }

        # Score and filter
        scored = []
        for helper in helpers:
            score = helper.matches_query(query)
            if score > 0:
                scored.append((score, helper))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [h for _, h in scored[:max_results]]
    elif category:
        # Filter by category first, then search within
        helpers = parser.get_helpers_by_category(category)
        if not helpers:
            return {
                "query": query,
                "category": category,
                "results": [],
                "count": 0,
                "error": f"Category '{category}' not found"
            }

        # Score and filter
        scored = []
        for helper in helpers:
            score = helper.matches_query(query)
            if score > 0:
                scored.append((score, helper))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [h for _, h in scored[:max_results]]
    else:
        results = parser.search(query, max_results)

    return {
        "query": query,
        "category": category,
        "subcategory": subcategory,
        "results": [
            {
                "name": h.name,
                "import": h.import_path,
                "when_to_use": h.when_to_use,
                "example": h.example,
                "category": h.category,
                "subcategory": h.subcategory
            }
            for h in results
        ],
        "count": len(results)
    }


def list_sst_categories(include_subcategories: bool = False) -> Dict[str, Any]:
    """
    List all SST helper categories.

    Args:
        include_subcategories: Whether to include subcategory breakdown

    Returns:
        Dict with categories list and counts
    """
    parser = get_parser()
    categories = parser.get_categories()

    result_categories = []
    for name, count in sorted(categories.items()):
        cat_info = {"name": name, "count": count}
        if include_subcategories:
            subcats = parser.get_subcategories(name)
            if subcats:
                cat_info["subcategories"] = [
                    {"name": subcat_name, "count": subcat_count}
                    for subcat_name, subcat_count in sorted(subcats.items())
                ]
        result_categories.append(cat_info)

    return {
        "categories": result_categories,
        "total_helpers": sum(categories.values())
    }


def get_sst_helper_details(
    helper_name: str,
    show_example: bool = True
) -> Dict[str, Any]:
    """
    Get detailed information about an SST helper.

    Args:
        helper_name: Name of the helper function
        show_example: Whether to include usage example

    Returns:
        Dict with full helper details
    """
    parser = get_parser()
    helper = parser.get_helper(helper_name)

    if not helper:
        # Try to find similar helpers
        all_helpers = parser.parse()
        suggestions = [
            name for name in all_helpers.keys()
            if helper_name.lower() in name.lower() or name.lower() in helper_name.lower()
        ]

        return {
            "name": helper_name,
            "error": f"Helper '{helper_name}' not found",
            "suggestions": suggestions[:5] if suggestions else None
        }

    result = {
        "name": helper.name,
        "category": helper.category,
        "subcategory": helper.subcategory,
        "import": helper.import_path,
        "signature": helper.signature,
        "when_to_use": helper.when_to_use,
        "determinism_impact": helper.determinism_impact,
        "common_misuse": helper.common_misuse
    }

    if helper.returns:
        result["returns"] = helper.returns

    if show_example and helper.example:
        result["example"] = helper.example

    return result


def recommend_sst_helper(task_description: str) -> Dict[str, Any]:
    """
    Recommend SST helpers based on task description.

    Uses heuristic matching to suggest appropriate helpers.

    Args:
        task_description: Description of what you want to do

    Returns:
        Dict with task and recommendations list
    """
    parser = get_parser()
    task_lower = task_description.lower()

    recommendations = []
    seen_helpers = set()

    # Check for pattern matches
    for pattern, helper_names in TASK_PATTERNS.items():
        if pattern in task_lower:
            for name in helper_names:
                if name not in seen_helpers:
                    helper = parser.get_helper(name)
                    if helper:
                        confidence = 0.9 if pattern in task_lower.split() else 0.7
                        recommendations.append({
                            "helper": name,
                            "confidence": confidence,
                            "reason": f"Matches pattern '{pattern}'",
                            "import": helper.import_path,
                            "when_to_use": helper.when_to_use
                        })
                        seen_helpers.add(name)

    # Fallback to search if no pattern matches
    if not recommendations:
        search_results = parser.search(task_description, max_results=5)
        for helper in search_results:
            if helper.name not in seen_helpers:
                recommendations.append({
                    "helper": helper.name,
                    "confidence": 0.5,
                    "reason": "Search match",
                    "import": helper.import_path,
                    "when_to_use": helper.when_to_use
                })

    # Sort by confidence
    recommendations.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "task": task_description,
        "recommendations": recommendations[:5]
    }


def list_sst_helpers_by_category(category: str) -> Dict[str, Any]:
    """
    List all helpers in a specific category.

    Args:
        category: Category name

    Returns:
        Dict with category, helpers list, and count
    """
    parser = get_parser()
    helpers = parser.get_helpers_by_category(category)

    if not helpers:
        # List available categories
        categories = parser.get_categories()
        return {
            "category": category,
            "helpers": [],
            "count": 0,
            "error": f"Category '{category}' not found",
            "available_categories": list(categories.keys())
        }

    return {
        "category": category,
        "helpers": [
            {
                "name": h.name,
                "import": h.import_path,
                "when_to_use": h.when_to_use,
                "signature": h.signature
            }
            for h in helpers
        ],
        "count": len(helpers)
    }
