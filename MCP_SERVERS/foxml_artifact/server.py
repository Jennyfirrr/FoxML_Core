#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
FoxML Artifact MCP Server

Query and explore run artifacts in RESULTS/runs/ (manifests, configs, snapshots, targets).
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add MCP_SERVERS to path for local imports
MCP_SERVERS_DIR = Path(__file__).resolve().parent.parent
if str(MCP_SERVERS_DIR) not in sys.path:
    sys.path.insert(0, str(MCP_SERVERS_DIR))

from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

from foxml_artifact import tools

# Create MCP server
app = Server("foxml-artifact")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="query_runs",
            description="Query runs with filters. List recent runs, filter by experiment, git SHA, or date range.",
            inputSchema={
                "type": "object",
                "properties": {
                    "experiment_name": {
                        "type": "string",
                        "description": "Filter by experiment name"
                    },
                    "git_sha": {
                        "type": "string",
                        "description": "Filter by git SHA prefix"
                    },
                    "config_fingerprint": {
                        "type": "string",
                        "description": "Filter by config fingerprint prefix"
                    },
                    "date_start": {
                        "type": "string",
                        "description": "Filter by start date (ISO format)"
                    },
                    "date_end": {
                        "type": "string",
                        "description": "Filter by end date (ISO format)"
                    },
                    "is_comparable": {
                        "type": "boolean",
                        "description": "Filter by comparability status"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Maximum results to return"
                    }
                }
            }
        ),
        Tool(
            name="get_run_details",
            description="Get detailed information about a specific run including manifest, config, and target index.",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Run identifier"
                    },
                    "include_config": {
                        "type": "boolean",
                        "default": False,
                        "description": "Whether to include resolved config"
                    },
                    "include_targets": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to include target index"
                    }
                },
                "required": ["run_id"]
            }
        ),
        Tool(
            name="query_targets",
            description="Query targets for a run. Get target information including models, metrics, and decisions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Run identifier"
                    },
                    "target": {
                        "type": "string",
                        "description": "Optional specific target to query"
                    }
                },
                "required": ["run_id"]
            }
        ),
        Tool(
            name="compare_runs",
            description="Compare two runs. Shows differences in git SHA, config, targets, and other metadata.",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id_1": {
                        "type": "string",
                        "description": "First run identifier"
                    },
                    "run_id_2": {
                        "type": "string",
                        "description": "Second run identifier"
                    },
                    "compare_config": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to compare configs"
                    },
                    "compare_targets": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to compare targets"
                    }
                },
                "required": ["run_id_1", "run_id_2"]
            }
        ),
        Tool(
            name="get_target_stage_history",
            description="Get stage progression history for a target. Shows cohorts, views, and stage metadata.",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Run identifier"
                    },
                    "target": {
                        "type": "string",
                        "description": "Target name"
                    }
                },
                "required": ["run_id", "target"]
            }
        ),
        Tool(
            name="search_experiments",
            description="Search and list experiments. Shows experiment names, run counts, and latest runs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name_pattern": {
                        "type": "string",
                        "description": "Optional pattern to filter experiment names"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Maximum results to return"
                    }
                }
            }
        ),
        Tool(
            name="get_model_metrics",
            description="Get model performance metrics for a target. Shows AUC, composite scores, and model family details.",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Run identifier"
                    },
                    "target": {
                        "type": "string",
                        "description": "Target name"
                    },
                    "family": {
                        "type": "string",
                        "description": "Optional specific model family to query (e.g., 'lightgbm')"
                    }
                },
                "required": ["run_id", "target"]
            }
        ),
        Tool(
            name="diff_target_results",
            description="Compare target results between two runs. Shows AUC delta, metric changes, and decision differences.",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id_1": {
                        "type": "string",
                        "description": "First run identifier"
                    },
                    "run_id_2": {
                        "type": "string",
                        "description": "Second run identifier"
                    },
                    "target": {
                        "type": "string",
                        "description": "Target name to compare"
                    }
                },
                "required": ["run_id_1", "run_id_2", "target"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    try:
        if name == "query_runs":
            result = tools.query_runs(
                experiment_name=arguments.get("experiment_name"),
                git_sha=arguments.get("git_sha"),
                config_fingerprint=arguments.get("config_fingerprint"),
                date_start=arguments.get("date_start"),
                date_end=arguments.get("date_end"),
                is_comparable=arguments.get("is_comparable"),
                limit=arguments.get("limit", 20)
            )
        elif name == "get_run_details":
            result = tools.get_run_details(
                run_id=arguments["run_id"],
                include_config=arguments.get("include_config", False),
                include_targets=arguments.get("include_targets", True)
            )
        elif name == "query_targets":
            result = tools.query_targets(
                run_id=arguments["run_id"],
                target=arguments.get("target")
            )
        elif name == "compare_runs":
            result = tools.compare_runs(
                run_id_1=arguments["run_id_1"],
                run_id_2=arguments["run_id_2"],
                compare_config=arguments.get("compare_config", True),
                compare_targets=arguments.get("compare_targets", True)
            )
        elif name == "get_target_stage_history":
            result = tools.get_target_stage_history(
                run_id=arguments["run_id"],
                target=arguments["target"]
            )
        elif name == "search_experiments":
            result = tools.search_experiments(
                name_pattern=arguments.get("name_pattern"),
                limit=arguments.get("limit", 20)
            )
        elif name == "get_model_metrics":
            result = tools.get_model_metrics(
                run_id=arguments["run_id"],
                target=arguments["target"],
                family=arguments.get("family")
            )
        elif name == "diff_target_results":
            result = tools.diff_target_results(
                run_id_1=arguments["run_id_1"],
                run_id_2=arguments["run_id_2"],
                target=arguments["target"]
            )
        else:
            result = {"error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    except Exception as e:
        error_result = {
            "error": str(e),
            "tool": name,
            "arguments": arguments
        }
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]


async def main():
    """Run the MCP server."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
