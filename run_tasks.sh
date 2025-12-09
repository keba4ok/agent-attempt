#!/bin/bash
# Run dpaia_agent.py sequentially for all task JSON files in tasks/
# Continues on error and reports failures.

set -u  # unset vars are errors

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="${SCRIPT_DIR}/tasks"
AGENT="${SCRIPT_DIR}/dpaia_agent.py"

if [ ! -f "$AGENT" ]; then
  echo "Agent not found at $AGENT"
  exit 1
fi

shopt -s nullglob
TASK_FILES=("${TASK_DIR}"/task_*.json)
shopt -u nullglob

if [ ${#TASK_FILES[@]} -eq 0 ]; then
  echo "No task_*.json files found in $TASK_DIR"
  exit 0
fi

FAILED=()

for task_file in "${TASK_FILES[@]}"; do
  echo "============================================================"
  echo "Running task: $(basename "$task_file")"
  echo "============================================================"
  if ! python "$AGENT" --config "$task_file"; then
    echo "Task failed: $(basename "$task_file")"
    FAILED+=("$(basename "$task_file")")
  fi
done

if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All tasks completed successfully."
else
  echo "Completed with failures:"
  for f in "${FAILED[@]}"; do
    echo "  - $f"
  done
fi

