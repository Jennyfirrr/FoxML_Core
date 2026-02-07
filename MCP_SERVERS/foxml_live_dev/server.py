"""
FoxML LIVE_TRADING Development Tracking MCP Server
===================================================

MCP server for tracking LIVE_TRADING module development,
SST compliance, determinism verification, and explainability.
"""

import json
import logging
from typing import Any

from mcp.server import Server
from mcp.types import (
    Tool,
    TextContent,
)

from .tools import (
    # Progress
    check_implementation_progress,
    get_plan_status,
    mark_file_complete,
    # Compliance
    check_sst_compliance,
    find_hardcoded_config,
    check_sorted_items_usage,
    # Determinism
    check_determinism_violations,
    verify_repro_bootstrap,
    # Explain
    get_decision_trace,
    explain_trade,
    list_recent_decisions,
)
from .tools.progress import get_next_files_to_implement
from .tools.compliance import check_all_live_trading_compliance
from .tools.determinism import check_random_seed_usage
from .tools.explain import compare_decisions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
server = Server("foxml-live-dev")


# Tool definitions
TOOLS = [
    # Progress Tracking
    Tool(
        name="check_progress",
        description="Check implementation progress for LIVE_TRADING plans. "
                    "Use plan='all' for overview or specific plan number (01-11).",
        inputSchema={
            "type": "object",
            "properties": {
                "plan": {
                    "type": "string",
                    "description": "Plan number (e.g., '01', '02') or 'all'",
                    "default": "all",
                }
            },
        },
    ),
    Tool(
        name="get_plan_details",
        description="Get detailed status for a specific plan including file list.",
        inputSchema={
            "type": "object",
            "properties": {
                "plan": {
                    "type": "string",
                    "description": "Plan number (01-11)",
                }
            },
            "required": ["plan"],
        },
    ),
    Tool(
        name="get_next_files",
        description="Get the next files that should be implemented based on dependencies.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="mark_complete",
        description="Mark a file as implemented with test status.",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative path to file",
                },
                "tests_passing": {
                    "type": "boolean",
                    "description": "Whether tests pass",
                    "default": False,
                },
            },
            "required": ["file_path"],
        },
    ),

    # SST Compliance
    Tool(
        name="check_sst",
        description="Check a file for SST compliance issues (repro_bootstrap, get_cfg, sorted_items).",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to file to check",
                }
            },
            "required": ["file_path"],
        },
    ),
    Tool(
        name="check_all_sst",
        description="Check all LIVE_TRADING files for SST compliance.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="find_hardcoded",
        description="Find hardcoded config values that should use get_cfg().",
        inputSchema={
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory to scan",
                    "default": "LIVE_TRADING",
                }
            },
        },
    ),
    Tool(
        name="check_dict_iteration",
        description="Check a file for proper sorted_items() usage in dict iterations.",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to file to check",
                }
            },
            "required": ["file_path"],
        },
    ),

    # Determinism
    Tool(
        name="check_determinism",
        description="Find potential determinism violations in code.",
        inputSchema={
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory to scan",
                    "default": "LIVE_TRADING",
                }
            },
        },
    ),
    Tool(
        name="verify_bootstrap",
        description="Verify repro_bootstrap is imported correctly in entry points.",
        inputSchema={
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory to check",
                    "default": "LIVE_TRADING",
                }
            },
        },
    ),
    Tool(
        name="check_random_seeds",
        description="Check for proper random seed management.",
        inputSchema={
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory to check",
                    "default": "LIVE_TRADING",
                }
            },
        },
    ),

    # Explainability
    Tool(
        name="get_decision",
        description="Get full decision trace for a trade decision.",
        inputSchema={
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Date (YYYY-MM-DD)",
                },
                "symbol": {
                    "type": "string",
                    "description": "Trading symbol",
                },
                "index": {
                    "type": "integer",
                    "description": "Decision index (0 = first)",
                    "default": 0,
                },
            },
            "required": ["date", "symbol"],
        },
    ),
    Tool(
        name="explain_decision",
        description="Get human-readable explanation of a trade decision.",
        inputSchema={
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Date (YYYY-MM-DD)",
                },
                "symbol": {
                    "type": "string",
                    "description": "Trading symbol",
                },
                "index": {
                    "type": "integer",
                    "description": "Decision index",
                    "default": 0,
                },
            },
            "required": ["date", "symbol"],
        },
    ),
    Tool(
        name="list_decisions",
        description="List recent trading decisions with optional filters.",
        inputSchema={
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Number of days to look back",
                    "default": 7,
                },
                "symbol": {
                    "type": "string",
                    "description": "Filter by symbol",
                },
                "decision_type": {
                    "type": "string",
                    "description": "Filter by type (TRADE, HOLD, BLOCKED)",
                    "enum": ["TRADE", "HOLD", "BLOCKED"],
                },
            },
        },
    ),
    Tool(
        name="compare_decisions",
        description="Compare decisions for a symbol across two dates.",
        inputSchema={
            "type": "object",
            "properties": {
                "date1": {
                    "type": "string",
                    "description": "First date",
                },
                "date2": {
                    "type": "string",
                    "description": "Second date",
                },
                "symbol": {
                    "type": "string",
                    "description": "Symbol to compare",
                },
            },
            "required": ["date1", "date2", "symbol"],
        },
    ),
]


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    try:
        if name == "check_progress":
            result = check_implementation_progress(arguments.get("plan", "all"))
        elif name == "get_plan_details":
            result = get_plan_status(arguments["plan"])
        elif name == "get_next_files":
            result = get_next_files_to_implement()
        elif name == "mark_complete":
            result = mark_file_complete(
                arguments["file_path"],
                arguments.get("tests_passing", False),
            )
        elif name == "check_sst":
            result = check_sst_compliance(arguments["file_path"])
        elif name == "check_all_sst":
            result = check_all_live_trading_compliance()
        elif name == "find_hardcoded":
            result = find_hardcoded_config(arguments.get("directory", "LIVE_TRADING"))
        elif name == "check_dict_iteration":
            result = check_sorted_items_usage(arguments["file_path"])
        elif name == "check_determinism":
            result = check_determinism_violations(arguments.get("directory", "LIVE_TRADING"))
        elif name == "verify_bootstrap":
            result = verify_repro_bootstrap(arguments.get("directory", "LIVE_TRADING"))
        elif name == "check_random_seeds":
            result = check_random_seed_usage(arguments.get("directory", "LIVE_TRADING"))
        elif name == "get_decision":
            result = get_decision_trace(
                arguments["date"],
                arguments["symbol"],
                arguments.get("index", 0),
            )
        elif name == "explain_decision":
            trace = get_decision_trace(
                arguments["date"],
                arguments["symbol"],
                arguments.get("index", 0),
            )
            result = explain_trade(trace)
        elif name == "list_decisions":
            result = list_recent_decisions(
                arguments.get("days", 7),
                arguments.get("symbol"),
                arguments.get("decision_type"),
            )
        elif name == "compare_decisions":
            result = compare_decisions(
                arguments["date1"],
                arguments["date2"],
                arguments["symbol"],
            )
        else:
            result = {"error": f"Unknown tool: {name}"}

        # Format output
        if isinstance(result, str):
            output = result
        else:
            output = json.dumps(result, indent=2, default=str)

        return [TextContent(type="text", text=output)]

    except Exception as e:
        logger.exception(f"Error in tool {name}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Run the MCP server."""
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
