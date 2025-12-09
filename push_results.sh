#!/bin/bash
# Script to push results to separate results repository
# Supports both submodule (results_repo/) and sibling directory setups

# Check for submodule first, then fall back to sibling directory
# Git submodules have .git as a file, not a directory
if [ -d "results_repo" ] && ( [ -f "results_repo/.git" ] || [ -d "results_repo/.git" ] ) && (cd results_repo && git rev-parse --git-dir > /dev/null 2>&1); then
    RESULTS_REPO="results_repo"
    echo "Using git submodule: results_repo/"
elif [ -d "../intellij-tools-results" ] && ( [ -f "../intellij-tools-results/.git" ] || [ -d "../intellij-tools-results/.git" ] ) && (cd ../intellij-tools-results && git rev-parse --git-dir > /dev/null 2>&1); then
    RESULTS_REPO="../intellij-tools-results"
    echo "Using sibling directory: ../intellij-tools-results/"
else
    RESULTS_REPO="${RESULTS_REPO_PATH:-../intellij-tools-results}"
fi

# Final validation - check if it's a valid git repository
if [ ! -d "$RESULTS_REPO" ] || ! (cd "$RESULTS_REPO" && git rev-parse --git-dir > /dev/null 2>&1); then
    echo "Error: Results repository not found or not a valid git repository"
    echo ""
    echo "Setup options:"
    echo "  1. Git submodule: git submodule add <repo-url> results_repo"
    echo "  2. Sibling directory: Create ../intellij-tools-results and clone repo there"
    echo "  3. Custom path: Set RESULTS_REPO_PATH environment variable"
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

