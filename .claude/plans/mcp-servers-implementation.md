# MCP Servers Implementation Plan for FoxML Core

## Objective
Build 3 Model Context Protocol (MCP) servers to expose FoxML Core's domain knowledge to AI assistants (Claude, GPT, etc.), making your $220/mo AI tooling investment 3-5x more effective by giving structured access to:
1. Configuration system (precedence, thresholds, experiments)
2. SST helper catalog (path construction, atomic writes, determinism)
3. Run artifacts (historical experiments, metrics, comparisons)

**Total effort:** ~7 hours (2h + 2h + 3h)

---

## Directory Structure

```
MCP_SERVERS/
├── README.md
├── pyproject.toml                  # Dependencies: mcp, pydantic, pyyaml
├── foxml_config/
│   ├── __init__.py
│   ├── server.py                   # Main MCP server
│   ├── tools.py                    # Tool implementations
│   └── cache.py                    # LRU cache with mtime tracking
├── foxml_sst/
│   ├── __init__.py
│   ├── server.py                   # Main MCP server
│   ├── tools.py                    # Tool implementations
│   └── catalog_parser.py           # Parse SST_SOLUTIONS.md
└── foxml_artifact/
    ├── __init__.py
    ├── server.py                   # Main MCP server
    ├── tools.py                    # Tool implementations
    └── index.py                    # Run indexing system
```

---

## Server 1: FoxML Config Server (~2 hours)

### Purpose
Expose the centralized config system (CONFIG/config_loader.py) with precedence chain visibility.

### Tools to Implement

**1. `get_config_value`**
- **Params:** `path: str, config_name: str = "pipeline_config", show_precedence: bool = False`
- **Returns:** `{"value": any, "source": str, "precedence_chain": list}`
- **Implementation:** Import and call `get_cfg()` from CONFIG.config_loader

**2. `list_config_keys`**
- **Params:** `config_name: str = "pipeline_config", prefix: str = "", max_depth: int = 3`
- **Returns:** `{"keys": list, "structure": dict, "count": int}`
- **Implementation:** Load config YAML, recursively extract keys with dot-notation

**3. `load_experiment_config`**
- **Params:** `experiment_name: str, show_effective: bool = False`
- **Returns:** `{"name": str, "config": dict, "effective_config": dict}`
- **Implementation:** Call `load_experiment_config()` from CONFIG.config_loader

**4. `show_config_precedence`**
- **Params:** `path: str`
- **Returns:** `{"path": str, "effective_value": any, "source": str, "precedence_chain": list}`
- **Implementation:** Load all config levels (intelligent → pipeline → defaults), show which provides the value

**5. `validate_config_structure`**
- **Params:** `config_name: str, expected_keys: list[str] = None`
- **Returns:** `{"valid": bool, "missing_keys": list, "warnings": list}`
- **Implementation:** Load config, check for required keys (e.g., determinism.base_seed)

### Critical Files
- `/home/Jennifer/trader/CONFIG/config_loader.py` - Core implementation
- `/home/Jennifer/trader/TRAINING/common/utils/config_helpers.py` - Helper patterns
- `/home/Jennifer/trader/CONFIG/pipeline/training/intelligent.yaml` - Example config
- `/home/Jennifer/trader/CONFIG/defaults.yaml` - Default values

---

## Server 2: FoxML SST Server (~2 hours)

### Purpose
Expose the SST helper catalog from INTERNAL/docs/references/SST_SOLUTIONS.md for discovery and recommendation.

### Tools to Implement

**1. `search_sst_helpers`**
- **Params:** `query: str, category: str = None, max_results: int = 10`
- **Returns:** `{"query": str, "results": list[{name, import, when_to_use, example}], "count": int}`
- **Implementation:** Parse SST_SOLUTIONS.md, search name/when_to_use/common_misuse fields

**2. `list_sst_categories`**
- **Params:** None
- **Returns:** `{"categories": list[{name, count}]}`
- **Implementation:** Extract section headers from SST_SOLUTIONS.md

**3. `get_sst_helper_details`**
- **Params:** `helper_name: str, show_example: bool = True`
- **Returns:** `{"name": str, "category": str, "import": str, "when_to_use": str, "determinism_impact": str, "common_misuse": str, "example": str}`
- **Implementation:** Parse SST_SOLUTIONS.md, extract structured fields for helper

**4. `recommend_sst_helper`**
- **Params:** `task_description: str`
- **Returns:** `{"task": str, "recommendations": list[{helper, confidence, reason}]}`
- **Implementation:** Heuristic matching:
  - "config" → `get_cfg`
  - "path" + "target" → `get_target_dir`
  - "normalize" + "family" → `normalize_family`
  - "iterate" → `iterdir_sorted`

**5. `list_sst_helpers_by_category`**
- **Params:** `category: str`
- **Returns:** `{"category": str, "helpers": list, "count": int}`
- **Implementation:** Filter parsed catalog by category

### Catalog Parser

```python
@dataclass
class SSTHelper:
    name: str
    category: str
    import_path: str
    when_to_use: str
    determinism_impact: str
    common_misuse: str
    example: Optional[str]

class SSTCatalogParser:
    def parse(self, md_path: Path) -> dict[str, SSTHelper]:
        # Split by ### helper_name
        # Extract fields: Import, When to use, Determinism impact, etc.
        # Return dict keyed by helper name
```

### Critical Files
- `/home/Jennifer/trader/INTERNAL/docs/references/SST_SOLUTIONS.md` - Primary source (parse this)
- `/home/Jennifer/trader/TRAINING/orchestration/utils/target_first_paths.py` - Path helper implementations
- `/home/Jennifer/trader/TRAINING/common/utils/file_utils.py` - Atomic write implementations
- `/home/Jennifer/trader/TRAINING/common/utils/determinism_ordering.py` - Determinism implementations

---

## Server 3: FoxML Artifact Server (~3 hours)

### Purpose
Query and explore run artifacts in RESULTS/runs/ (manifests, configs, snapshots, targets).

### Tools to Implement

**1. `query_runs`**
- **Params:** `experiment_name: str = None, git_sha: str = None, config_fingerprint: str = None, date_start: str = None, date_end: str = None, is_comparable: bool = None, limit: int = 20`
- **Returns:** `{"runs": list[{run_id, created_at, experiment_name, git_sha, targets, is_comparable}], "count": int}`
- **Implementation:** Scan RESULTS/runs/*/* for manifest.json, filter by criteria

**2. `get_run_details`**
- **Params:** `run_id: str, include_config: bool = False, include_targets: bool = True`
- **Returns:** `{"run_id": str, "manifest": dict, "config": dict, "target_index": dict}`
- **Implementation:** Find run by run_id, load manifest.json and optionally config.resolved.json

**3. `query_targets`**
- **Params:** `run_id: str, target: str = None`
- **Returns:** `{"run_id": str, "targets": list[{target, decision, models, metrics}]}`
- **Implementation:** Parse manifest.target_index, list artifact paths per target

**4. `compare_runs`**
- **Params:** `run_id_1: str, run_id_2: str, compare_config: bool = True, compare_targets: bool = True`
- **Returns:** `{"run_1": dict, "run_2": dict, "differences": {git_sha, targets, config_diff}}`
- **Implementation:** Load both manifests, deep diff configs and target lists

**5. `get_target_stage_history`**
- **Params:** `run_id: str, target: str`
- **Returns:** `{"run_id": str, "target": str, "stages": list[{stage, completed, snapshots, metrics}]}`
- **Implementation:** Parse globals/snapshot_index.json, extract stage progression

**6. `search_experiments`**
- **Params:** `name_pattern: str = None, limit: int = 20`
- **Returns:** `{"experiments": list[{name, runs, latest_run, git_shas}]}`
- **Implementation:** Group runs by experiment_name from manifests

### Run Index System

```python
@dataclass
class RunMetadata:
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

class RunIndex:
    def build_index(self) -> dict[str, RunMetadata]:
        # Scan RESULTS/runs/*/* for manifest.json
        # Parse each manifest
        # Index by run_id
        # Refresh every 5 minutes (background)
```

### Critical Files
- `/home/Jennifer/trader/TRAINING/orchestration/utils/manifest.py` - Manifest structure
- `/home/Jennifer/trader/RESULTS/runs/` - Run directories (scan these)
- Example manifests in recent runs (for schema validation)
- `/home/Jennifer/trader/TRAINING/common/utils/file_utils.py` - Atomic read helpers

---

## Implementation Steps

### 1. Project Setup (30 min)
```bash
cd /home/Jennifer/trader
mkdir -p MCP_SERVERS/{foxml_config,foxml_sst,foxml_artifact}
cd MCP_SERVERS

# Create pyproject.toml
cat > pyproject.toml << 'EOF'
[project]
name = "foxml-mcp-servers"
version = "0.1.0"
dependencies = [
    "mcp>=0.9.0",
    "pydantic>=2.0",
    "pyyaml>=6.0"
]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
EOF

# Install dependencies
pip install -e .
```

### 2. Implement Config Server (2 hours)

**Structure:**
```python
# foxml_config/server.py
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

app = Server("foxml-config")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_config_value",
            description="Get configuration value with optional precedence chain",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Config path (e.g. 'pipeline.determinism.base_seed')"},
                    "config_name": {"type": "string", "default": "pipeline_config"},
                    "show_precedence": {"type": "boolean", "default": False}
                },
                "required": ["path"]
            }
        ),
        # ... other tools
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "get_config_value":
        from .tools import get_config_value
        result = get_config_value(**arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    # ... other tools

if __name__ == "__main__":
    mcp.server.stdio.stdio_server(app)
```

**Implement tools.py with all 5 tools**, importing from CONFIG/config_loader.py.

### 3. Implement SST Server (2 hours)

**Key component: catalog_parser.py**
```python
import re
from pathlib import Path
from dataclasses import dataclass

@dataclass
class SSTHelper:
    name: str
    category: str
    import_path: str
    when_to_use: str
    determinism_impact: str
    common_misuse: str
    example: str

class SSTCatalogParser:
    def __init__(self, sst_solutions_path: Path):
        self.path = sst_solutions_path
        self._catalog = None

    def parse(self) -> dict[str, SSTHelper]:
        if self._catalog:
            return self._catalog

        content = self.path.read_text()

        # Split by category sections (## header)
        categories = re.split(r'\n## (.+)\n', content)

        catalog = {}
        current_category = None

        for i, section in enumerate(categories):
            if i % 2 == 1:  # Category name
                current_category = section.strip()
            elif i % 2 == 0 and current_category:  # Category content
                # Split by helper sections (### helper_name)
                helpers = re.split(r'\n### `(.+?)`\n', section)

                for j in range(1, len(helpers), 2):
                    name = helpers[j].strip()
                    content = helpers[j + 1] if j + 1 < len(helpers) else ""

                    # Extract fields with regex
                    import_match = re.search(r'\*\*Import:\*\*\s*```python\n(.+?)\n```', content, re.DOTALL)
                    when_to_use_match = re.search(r'\*\*When to use:\*\*\s*(.+?)(?=\n\*\*|\n###|\Z)', content, re.DOTALL)
                    determinism_match = re.search(r'\*\*Determinism impact:\*\*\s*(.+?)(?=\n\*\*|\n###|\Z)', content, re.DOTALL)
                    misuse_match = re.search(r'\*\*Common misuse:\*\*\s*(.+?)(?=\n\*\*|\n###|\Z)', content, re.DOTALL)
                    example_match = re.search(r'\*\*Example:\*\*\s*```python\n(.+?)\n```', content, re.DOTALL)

                    catalog[name] = SSTHelper(
                        name=name,
                        category=current_category,
                        import_path=import_match.group(1).strip() if import_match else "",
                        when_to_use=when_to_use_match.group(1).strip() if when_to_use_match else "",
                        determinism_impact=determinism_match.group(1).strip() if determinism_match else "",
                        common_misuse=misuse_match.group(1).strip() if misuse_match else "",
                        example=example_match.group(1).strip() if example_match else ""
                    )

        self._catalog = catalog
        return catalog
```

**Implement server.py and tools.py** with 5 tools using the parser.

### 4. Implement Artifact Server (3 hours)

**Key component: index.py**
```python
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import json
from typing import Optional

@dataclass
class RunMetadata:
    run_id: str
    run_instance_id: str
    is_comparable: bool
    created_at: datetime
    experiment_name: Optional[str]
    git_sha: str
    config_fingerprint: str
    deterministic_config_fingerprint: str
    targets: list[str]
    manifest_path: Path

class RunIndex:
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir / "runs"
        self._index: dict[str, RunMetadata] = {}
        self._last_build = None

    def build_index(self) -> dict[str, RunMetadata]:
        """Scan RESULTS/runs and build index."""
        index = {}

        # Scan all comparison_group directories
        for cg_dir in self.results_dir.iterdir():
            if not cg_dir.is_dir():
                continue

            # Scan run instance directories
            for run_dir in cg_dir.iterdir():
                if not run_dir.is_dir():
                    continue

                manifest_path = run_dir / "manifest.json"
                if not manifest_path.exists():
                    continue

                try:
                    with open(manifest_path) as f:
                        manifest = json.load(f)

                    metadata = RunMetadata(
                        run_id=manifest["run_id"],
                        run_instance_id=manifest["run_instance_id"],
                        is_comparable=manifest.get("is_comparable", False),
                        created_at=datetime.fromisoformat(manifest["created_at"]),
                        experiment_name=manifest.get("experiment", {}).get("name"),
                        git_sha=manifest.get("git_sha", ""),
                        config_fingerprint=manifest.get("config_digest", ""),
                        deterministic_config_fingerprint=manifest.get("deterministic_config_fingerprint", ""),
                        targets=manifest.get("targets", []),
                        manifest_path=manifest_path
                    )

                    index[metadata.run_id] = metadata

                except Exception as e:
                    # Skip corrupted manifests
                    continue

        self._index = index
        self._last_build = datetime.now()
        return index

    def query(self, **filters) -> list[RunMetadata]:
        """Filter runs by criteria."""
        results = list(self._index.values())

        if "experiment_name" in filters and filters["experiment_name"]:
            results = [r for r in results if r.experiment_name == filters["experiment_name"]]

        if "git_sha" in filters and filters["git_sha"]:
            results = [r for r in results if r.git_sha.startswith(filters["git_sha"])]

        if "is_comparable" in filters and filters["is_comparable"] is not None:
            results = [r for r in results if r.is_comparable == filters["is_comparable"]]

        if "date_start" in filters and filters["date_start"]:
            start = datetime.fromisoformat(filters["date_start"])
            results = [r for r in results if r.created_at >= start]

        if "date_end" in filters and filters["date_end"]:
            end = datetime.fromisoformat(filters["date_end"])
            results = [r for r in results if r.created_at <= end]

        # Sort by created_at descending (newest first)
        results.sort(key=lambda r: r.created_at, reverse=True)

        # Apply limit
        limit = filters.get("limit", 20)
        return results[:limit]
```

**Implement server.py and tools.py** with 6 tools using the index.

### 5. Testing (1 hour total)

**Test each server with MCP stdio_client:**
```python
# test_config_server.py
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

### 6. Integration with Claude (15 min)

**Add to Claude Desktop config (`~/.config/Claude/claude_desktop_config.json`):**
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

**Restart Claude Desktop.** Tools should now appear in tool picker.

---

## Verification

### Manual Testing
1. **Config Server:** Query `pipeline.determinism.base_seed` → should return 42
2. **SST Server:** Search "path construction" → should return `get_target_dir`, `get_scoped_artifact_dir`
3. **Artifact Server:** Query recent runs → should list runs from RESULTS/runs/

### Integration Testing
Ask Claude:
- "What's the base seed for deterministic runs?" → Should call `get_config_value`
- "How do I construct a target directory path?" → Should call `search_sst_helpers` or `recommend_sst_helper`
- "Show me the last 5 runs" → Should call `query_runs`

---

## Critical Files to Reference

1. **CONFIG/config_loader.py** - Config loading implementation
2. **INTERNAL/docs/references/SST_SOLUTIONS.md** - SST helper catalog
3. **TRAINING/orchestration/utils/manifest.py** - Manifest structure
4. **TRAINING/orchestration/utils/target_first_paths.py** - Path helper implementations
5. **TRAINING/common/utils/file_utils.py** - Atomic write patterns

---

## Notes for Opus

- All servers follow the same pattern: server.py (MCP setup) + tools.py (implementations)
- Use atomic read helpers from `TRAINING.common.utils.file_utils` for consistency
- Follow deterministic iteration patterns (sorted_items, iterdir_sorted) when enumerating artifacts
- Cache aggressively (60s TTL for configs, 5min for run index) to minimize I/O
- Return structured errors with suggestions for better UX
- All times should be ISO format for parsing
- Validate inputs before processing (e.g., run_id format, config paths)

The goal is clean, maintainable MCP servers that expose FoxML Core's unique domain knowledge to AI assistants, making them significantly more effective at architectural questions and development tasks.
