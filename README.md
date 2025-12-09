# IntelliJ Tools Testing

Autonomous agents for software engineering tasks using MCP (Model Context Protocol) tools with Claude AI.

## Overview

This project contains autonomous agents that can:
- Work on projects from DPAIA tasks
- Solve SWE-bench tasks
- Analyze tool usage statistics from agent runs

## Prerequisites

- Python 3.8+
- Anthropic API key
- MCP server running (IntelliJ IDEA MCP server for DPAIA agent)

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd "agentic attemps"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env` file:
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

See `.env.example` for configuration options.

## Scripts

### `dpaia_agent.py`
Autonomous agent for Spring Boot projects using IntelliJ MCP tools.

**Usage:**
```bash
# Use default configuration
python dpaia_agent.py

# Specify task via command-line arguments
python dpaia_agent.py --repo owner/repo --issue-url https://github.com/owner/repo/issues/1 \
  --issue-title "Task title" --issue-body "Task description"

# Use JSON configuration file
python dpaia_agent.py --config task.json

# Custom model and steps
python dpaia_agent.py --model claude-sonnet-4-20250514 --max-steps 64 --log-level DEBUG
```

**Features:**
- Clones repositories into isolated instances (`repos/<repo_name>/<timestamp>/`)
- Each run gets a fresh repository copy
- Logs saved to `results_dpaia/`
- Command-line interface for easy configuration
- Proper logging with configurable levels

**Options:**
- `--config FILE`: Path to JSON configuration file
- `--repo REPO`: Repository in format owner/repo
- `--issue-url URL`: GitHub issue URL
- `--issue-title TITLE`: Issue title
- `--issue-body BODY`: Issue description
- `--mcp-url URL`: MCP server URL (overrides env var)
- `--model MODEL`: Claude model to use (default: claude-sonnet-4-20250514)
- `--max-tokens N`: Max tokens per request (default: 8000)
- `--max-steps N`: Maximum agent steps (default: 128)
- `--instance-id ID`: Custom instance ID (default: timestamp)
- `--log-level LEVEL`: Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)

### `swe_agent.py`
Autonomous agent for SWE-bench tasks.

**Usage:**
```bash
# Use default instance ID
python swe_agent.py

# Specify instance ID
python swe_agent.py --instance-id astropy__astropy-13579

# Custom model and steps
python swe_agent.py --instance-id django__django-12345 --model claude-3-5-sonnet-20241022 --max-steps 100

# Custom dataset path
python swe_agent.py --instance-id sympy__sympy-67890 --dataset-path /path/to/swebench_verified.jsonl
```

**Features:**
- Loads tasks from SWE-bench dataset
- Clones repositories to `repos/<repo_name>/`
- Logs saved to `results_swe/<repo_name>/`
- Command-line interface for configuration

**Options:**
- `--instance-id ID`: SWE-bench instance ID (default: astropy__astropy-13579)
- `--dataset-path PATH`: Path to SWE-bench dataset JSONL file
- `--mcp-url URL`: MCP server URL (overrides env var)
- `--model MODEL`: Claude model to use (default: claude-3-5-haiku-20241022)
- `--max-tokens N`: Max tokens per request (default: 4096)
- `--max-steps N`: Maximum agent steps (default: 50)
- `--log-level LEVEL`: Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)

### `download_bench.py`
Downloads SWE-bench Verified dataset.

**Usage:**
```bash
python download_bench.py
```

Saves dataset to `swebench_verified.jsonl` in the project root.

### `clone_repo.py`
Utility to clone and checkout specific commits.

**Usage:**
```bash
python clone_repo.py <owner/repo> <commit_hash>
```

Clones to `repos/<repo_name>/`.

### `tools_stats.py`
Analyzes tool usage statistics from agent log files.

**Usage:**
```bash
# Analyze specific log files
python tools_stats.py results_dpaia/*.txt -v

# Export to JSON/CSV
python tools_stats.py results_dpaia/*.txt -j stats.json -c stats.csv

# Auto-save to statistics folder (default)
python tools_stats.py results_dpaia/*.txt
```

**Options:**
- `-v, --verbose`: Show detailed per-file statistics
- `-j FILE, --json FILE`: Export to JSON (default: `statistics/tool_stats_TIMESTAMP.json`)
- `-c FILE, --csv FILE`: Export to CSV (default: `statistics/tool_stats_TIMESTAMP.csv`)

### `tools_analyse.py`
Collects and analyzes tool usage from SWE agent logs.

**Usage:**
```bash
python tools_analyse.py
```

Scans `results_swe/` for log files and saves statistics to `statistics/tool_usage_stats_TIMESTAMP.json`.

## Configuration

### Environment Variables
Create a `.env` file (see `.env.example`):
- `ANTHROPIC_API_KEY`: Your Anthropic API key (required)
- `MCP_SERVER_URL`: MCP server URL (optional, can be set via CLI)

### DPAIA Agent
Configure via command-line arguments or JSON file:
```json
{
  "repo": "owner/repo",
  "issue_url": "https://github.com/owner/repo/issues/1",
  "issue_title": "Task title",
  "issue_body": "Task description"
}
```

### SWE Agent
Configure via command-line arguments:
- `--instance-id`: SWE-bench instance ID
- Other options available via `--help`

## Results Repository

Results (logs, statistics) should be stored in a **separate repository** for team collaboration.

See [RESULTS_REPO.md](RESULTS_REPO.md) for setup instructions.


