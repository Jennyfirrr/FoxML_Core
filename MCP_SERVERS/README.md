# FoxML MCP Servers

Model Context Protocol (MCP) servers exposing FoxML Core domain knowledge to AI assistants.

## Overview

These servers provide structured access to FoxML's configuration system, SST helper catalog, and run artifacts, making AI assistants significantly more effective at architectural questions and development tasks.

## Servers

### 1. FoxML Config Server (`foxml_config`)

Exposes the centralized config system with precedence chain visibility.

**Tools:**
- `get_config_value` - Get config value by path with optional precedence chain
- `list_config_keys` - List available keys in a config file
- `load_experiment_config` - Load experiment configuration by name
- `show_config_precedence` - Show full precedence chain for a config path
- `validate_config_structure` - Validate configuration structure
- `list_available_configs` - List all available config files

### 2. FoxML SST Server (`foxml_sst`)

Exposes the SST helper catalog from `INTERNAL/docs/references/SST_SOLUTIONS.md`.

**Tools:**
- `search_sst_helpers` - Search helpers by query
- `list_sst_categories` - List all helper categories
- `get_sst_helper_details` - Get detailed helper information
- `recommend_sst_helper` - Get helper recommendations for a task
- `list_sst_helpers_by_category` - List helpers in a category

### 3. FoxML Artifact Server (`foxml_artifact`)

Query and explore run artifacts in `RESULTS/runs/`.

**Tools:**
- `query_runs` - Query runs with filters (experiment, git SHA, date range)
- `get_run_details` - Get detailed run information
- `query_targets` - Query targets for a run
- `compare_runs` - Compare two runs
- `get_target_stage_history` - Get stage progression for a target
- `search_experiments` - Search and list experiments
- `get_model_metrics` - Get model performance metrics (AUC, composite scores)
- `diff_target_results` - Compare target results between two runs

## Installation

```bash
cd /home/Jennifer/trader/MCP_SERVERS
pip install -e .
```

**Dependencies:**
- `mcp>=0.9.0`
- `pydantic>=2.0`
- `pyyaml>=6.0`

## Usage

### Running Servers Directly

```bash
# Config server
python -m foxml_config.server

# SST server
python -m foxml_sst.server

# Artifact server
python -m foxml_artifact.server
```

### Claude Desktop Integration

Add to `~/.config/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "foxml-config": {
      "command": "python",
      "args": ["-m", "foxml_config.server"],
      "cwd": "/home/Jennifer/trader/MCP_SERVERS"
    },
    "foxml-sst": {
      "command": "python",
      "args": ["-m", "foxml_sst.server"],
      "cwd": "/home/Jennifer/trader/MCP_SERVERS"
    },
    "foxml-artifact": {
      "command": "python",
      "args": ["-m", "foxml_artifact.server"],
      "cwd": "/home/Jennifer/trader/MCP_SERVERS"
    }
  }
}
```

Restart Claude Desktop to activate the servers.

### Claude Code Integration

Add to `.claude/settings.local.json`:

```json
{
  "mcpServers": {
    "foxml-config": {
      "command": "python",
      "args": ["-m", "foxml_config.server"],
      "cwd": "/home/Jennifer/trader/MCP_SERVERS"
    },
    "foxml-sst": {
      "command": "python",
      "args": ["-m", "foxml_sst.server"],
      "cwd": "/home/Jennifer/trader/MCP_SERVERS"
    },
    "foxml-artifact": {
      "command": "python",
      "args": ["-m", "foxml_artifact.server"],
      "cwd": "/home/Jennifer/trader/MCP_SERVERS"
    }
  }
}
```

## Testing

### Manual Testing

```python
# Test config server
from mcp import stdio_client
import asyncio

async def test():
    async with stdio_client(["python", "-m", "foxml_config.server"]) as client:
        tools = await client.list_tools()
        print(f"Available tools: {[t.name for t in tools]}")

        result = await client.call_tool("get_config_value", {
            "path": "pipeline.determinism.base_seed"
        })
        print(f"Result: {result}")

asyncio.run(test())
```

### Expected Results

1. **Config Server:** Query `pipeline.determinism.base_seed` → returns 42
2. **SST Server:** Search "path construction" → returns `get_target_dir`, `get_scoped_artifact_dir`
3. **Artifact Server:** Query recent runs → lists runs from `RESULTS/runs/`

## Development

### Project Structure

```
MCP_SERVERS/
├── README.md
├── pyproject.toml
├── foxml_config/
│   ├── __init__.py
│   ├── server.py           # MCP server
│   └── tools.py            # Tool implementations
├── foxml_sst/
│   ├── __init__.py
│   ├── server.py           # MCP server
│   ├── tools.py            # Tool implementations
│   └── catalog_parser.py   # SST_SOLUTIONS.md parser
└── foxml_artifact/
    ├── __init__.py
    ├── server.py           # MCP server
    ├── tools.py            # Tool implementations
    └── index.py            # Run indexing system
```

### Critical Files Referenced

- `CONFIG/config_loader.py` - Config loading implementation
- `INTERNAL/docs/references/SST_SOLUTIONS.md` - SST helper catalog
- `TRAINING/orchestration/utils/manifest.py` - Manifest structure
- `RESULTS/runs/` - Run artifact directories

### Caching

- Config values: 60s TTL
- SST catalog: 5min TTL (parses SST_SOLUTIONS.md once)
- Run index: 5min TTL (scans RESULTS/runs/)

### Adding New Tools

1. Add tool implementation in `tools.py`
2. Register tool in `server.py` under `@app.list_tools()`
3. Add handler in `@app.call_tool()`
4. Update this README

## License

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
Copyright (c) 2025-2026 Fox ML Infrastructure LLC
