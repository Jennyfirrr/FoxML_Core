#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
FoxML SST MCP Server

Exposes the SST helper catalog from INTERNAL/docs/references/SST_SOLUTIONS.md
for discovery and recommendation.
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

from foxml_sst import tools

# Create MCP server
app = Server("foxml-sst")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="search_sst_helpers",
            description="Search SST helpers by query. Use to find the right helper for a task like 'path construction' or 'config access'.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'config', 'path construction', 'determinism')"
                    },
                    "category": {
                        "type": "string",
                        "description": "Optional category filter (e.g., 'Path Construction Helpers')"
                    },
                    "subcategory": {
                        "type": "string",
                        "description": "Optional subcategory filter (e.g., 'Target Paths'). Requires category."
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 10,
                        "description": "Maximum results to return"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="list_sst_categories",
            description="List all SST helper categories with counts. Use to discover what categories of helpers exist.",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_subcategories": {
                        "type": "boolean",
                        "default": False,
                        "description": "Whether to include subcategory breakdown"
                    }
                }
            }
        ),
        Tool(
            name="get_sst_helper_details",
            description="Get detailed information about a specific SST helper including import path, usage, and examples.",
            inputSchema={
                "type": "object",
                "properties": {
                    "helper_name": {
                        "type": "string",
                        "description": "Name of the helper function (e.g., 'get_cfg', 'get_target_dir')"
                    },
                    "show_example": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to include usage example"
                    }
                },
                "required": ["helper_name"]
            }
        ),
        Tool(
            name="recommend_sst_helper",
            description="Get helper recommendations based on a task description. Describe what you want to do and get relevant helpers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": "Description of task (e.g., 'construct a target directory path', 'normalize model family name')"
                    }
                },
                "required": ["task_description"]
            }
        ),
        Tool(
            name="list_sst_helpers_by_category",
            description="List all helpers in a specific category.",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Category name (e.g., 'Config Access Helpers', 'Path Construction Helpers')"
                    }
                },
                "required": ["category"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    try:
        if name == "search_sst_helpers":
            result = tools.search_sst_helpers(
                query=arguments["query"],
                category=arguments.get("category"),
                subcategory=arguments.get("subcategory"),
                max_results=arguments.get("max_results", 10)
            )
        elif name == "list_sst_categories":
            result = tools.list_sst_categories(
                include_subcategories=arguments.get("include_subcategories", False)
            )
        elif name == "get_sst_helper_details":
            result = tools.get_sst_helper_details(
                helper_name=arguments["helper_name"],
                show_example=arguments.get("show_example", True)
            )
        elif name == "recommend_sst_helper":
            result = tools.recommend_sst_helper(
                task_description=arguments["task_description"]
            )
        elif name == "list_sst_helpers_by_category":
            result = tools.list_sst_helpers_by_category(
                category=arguments["category"]
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
