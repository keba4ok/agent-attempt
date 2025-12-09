#!/bin/bash
# Script to push results to separate results repository

RESULTS_REPO="${RESULTS_REPO_PATH:-../intellij-tools-results}"

if [ ! -d "$RESULTS_REPO" ]; then
    echo "Error: Results repository not found at $RESULTS_REPO"
    echo "Set RESULTS_REPO_PATH environment variable or create repository at ../intellij-tools-results"
    exit 1
fi

# Check if there are any results to copy
if [ ! -d "results_dpaia" ] && [ ! -d "results_swe" ] && [ ! -d "statistics" ]; then
    echo "No results found to copy"
    exit 0
fi

# Copy results
echo "Copying results to $RESULTS_REPO..."
if [ -d "results_dpaia" ]; then
    cp -r results_dpaia "$RESULTS_REPO/" 2>/dev/null || true
fi
if [ -d "results_swe" ]; then
    cp -r results_swe "$RESULTS_REPO/" 2>/dev/null || true
fi
if [ -d "statistics" ]; then
    cp -r statistics "$RESULTS_REPO/" 2>/dev/null || true
fi

# Commit and push
cd "$RESULTS_REPO"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

git add results_dpaia/ results_swe/ statistics/ 2>/dev/null
if git diff --staged --quiet; then
    echo "No changes to commit"
else
    git commit -m "Add results from $TIMESTAMP"
    git push
    echo "Results pushed successfully to $RESULTS_REPO"
fi

