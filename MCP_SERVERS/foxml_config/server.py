#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
FoxML Config MCP Server

Exposes the centralized config system (CONFIG/config_loader.py) with precedence chain visibility.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

from foxml_config import tools

# Create MCP server
app = Server("foxml-config")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="get_config_value",
            description="Get configuration value by path with optional precedence chain. Use for accessing any config setting.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Dot-notation config path (e.g., 'pipeline.determinism.base_seed')"
                    },
                    "config_name": {
                        "type": "string",
                        "default": "pipeline_config",
                        "description": "Config file name (e.g., 'pipeline_config', 'intelligent_training_config')"
                    },
                    "show_precedence": {
                        "type": "boolean",
                        "default": False,
                        "description": "Whether to show full precedence chain showing where value comes from"
                    }
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="list_config_keys",
            description="List available configuration keys in a config file. Use to discover what settings exist.",
            inputSchema={
                "type": "object",
                "properties": {
                    "config_name": {
                        "type": "string",
                        "default": "pipeline_config",
                        "description": "Config file name"
                    },
                    "prefix": {
                        "type": "string",
                        "default": "",
                        "description": "Filter by key prefix (e.g., 'pipeline.determinism')"
                    },
                    "max_depth": {
                        "type": "integer",
                        "default": 3,
                        "description": "Maximum nesting depth to traverse"
                    }
                }
            }
        ),
        Tool(
            name="load_experiment_config",
            description="Load an experiment configuration by name. Shows experiment overrides and optionally the effective merged config.",
            inputSchema={
                "type": "object",
                "properties": {
                    "experiment_name": {
                        "type": "string",
                        "description": "Experiment name (without .yaml extension)"
                    },
                    "show_effective": {
                        "type": "boolean",
                        "default": False,
                        "description": "Whether to show effective (merged with base) config"
                    }
                },
                "required": ["experiment_name"]
            }
        ),
        Tool(
            name="show_config_precedence",
            description="Show full precedence chain for a config path. Shows all config layers and which one provides the effective value.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Dot-notation config path"
                    }
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="validate_config_structure",
            description="Validate configuration structure and check for required keys.",
            inputSchema={
                "type": "object",
                "properties": {
                    "config_name": {
                        "type": "string",
                        "description": "Config file name to validate"
                    },
                    "expected_keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of keys that must be present"
                    }
                },
                "required": ["config_name"]
            }
        ),
        Tool(
            name="list_available_configs",
            description="List all available configuration files (model configs and training configs).",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    try:
        if name == "get_config_value":
            result = tools.get_config_value(
                path=arguments["path"],
                config_name=arguments.get("config_name", "pipeline_config"),
                show_precedence=arguments.get("show_precedence", False)
            )
        elif name == "list_config_keys":
            result = tools.list_config_keys(
                config_name=arguments.get("config_name", "pipeline_config"),
                prefix=arguments.get("prefix", ""),
                max_depth=arguments.get("max_depth", 3)
            )
        elif name == "load_experiment_config":
            result = tools.load_experiment_config(
                experiment_name=arguments["experiment_name"],
                show_effective=arguments.get("show_effective", False)
            )
        elif name == "show_config_precedence":
            result = tools.show_config_precedence(
                path=arguments["path"]
            )
        elif name == "validate_config_structure":
            result = tools.validate_config_structure(
                config_name=arguments["config_name"],
                expected_keys=arguments.get("expected_keys")
            )
        elif name == "list_available_configs":
            result = tools.list_available_configs()
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
