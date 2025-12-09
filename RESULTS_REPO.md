# Results Repository Setup

This project generates results (logs, statistics) that should be stored in a separate repository.

## Setup Instructions

### 1. Create Results Repository

### 2. Configure Results Path (Optional)

You can set an environment variable to automatically push results:

```bash
# In your .env file (optional)
RESULTS_REPO_PATH=../intellij-tools-results
```

### 3. Results Structure

Results are organized as follows:

```
results_dpaia/          # DPAIA agent logs
  dpaia_agent_TIMESTAMP.txt

results_swe/            # SWE agent logs
  <repo_name>/
    agent_log_TIMESTAMP.txt

statistics/             # Generated statistics
  tool_stats_TIMESTAMP.json
  tool_stats_TIMESTAMP.csv
  tool_usage_stats_TIMESTAMP.json
```

### 4. Pushing Results to Repository

Usa a script `push_results.sh`:

```bash
chmod +x push_results.sh
```


