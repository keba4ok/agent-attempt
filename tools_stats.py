import re
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional


def parse_log_file(log_path: str) -> Dict:
    """Parse a single log file and extract tool usage statistics."""

    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    task_match = re.search(r'TASK: (.+)', content)
    repo_match = re.search(r'REPO: (.+)', content)

    task_name = task_match.group(1) if task_match else "Unknown"
    repo_name = repo_match.group(1) if repo_match else "Unknown"

    tool_pattern = r'Tool: (\w+)'
    tools_used = re.findall(tool_pattern, content)


    error_pattern = r'\[ERROR\] Tool (\w+) failed'
    tools_failed = re.findall(error_pattern, content)


    step_pattern = r'Step (\d+)/(\d+)'
    steps = re.findall(step_pattern, content)
    total_steps = len(steps)

 
    completed = "Task completed successfully!" in content or "TASK_COMPLETE" in content


    tool_counts = Counter(tools_used)
    error_counts = Counter(tools_failed)

    return {
        "log_file": Path(log_path).name,
        "task_name": task_name,
        "repo_name": repo_name,
        "total_steps": total_steps,
        "completed": completed,
        "tool_counts": dict(tool_counts),
        "error_counts": dict(error_counts),
        "total_tool_calls": len(tools_used),
        "unique_tools_used": len(tool_counts),
        "tools_used": list(tool_counts.keys())
    }


def aggregate_statistics(log_files: List[str]) -> Dict:
    """Aggregate statistics from multiple log files."""

    all_stats = []
    overall_tool_counts = Counter()
    overall_error_counts = Counter()

    for log_path in log_files:
        try:
            stats = parse_log_file(log_path)
            all_stats.append(stats)

            # Aggregate overall counts
            for tool, count in stats["tool_counts"].items():
                overall_tool_counts[tool] += count

            for tool, count in stats["error_counts"].items():
                overall_error_counts[tool] += count

        except Exception as e:
            print(f"Error parsing {log_path}: {e}")

    return {
        "individual_stats": all_stats,
        "overall_tool_counts": dict(overall_tool_counts),
        "overall_error_counts": dict(overall_error_counts),
        "total_logs": len(all_stats),
        "completed_tasks": sum(1 for s in all_stats if s["completed"]),
        "total_tool_calls": sum(s["total_tool_calls"] for s in all_stats),
        "all_unique_tools": sorted(overall_tool_counts.keys())
    }


def print_statistics(stats: Dict, verbose: bool = False):
    """Print formatted statistics."""

    print("="*80)
    print("TOOL USAGE STATISTICS")
    print("="*80)

    print(f"\nOVERVIEW:")
    print(f"  Total log files: {stats['total_logs']}")
    print(f"  Completed tasks: {stats['completed_tasks']}/{stats['total_logs']}")
    print(f"  Total tool calls: {stats['total_tool_calls']}")
    print(f"  Unique tools used: {len(stats['all_unique_tools'])}")

    print(f"\nOVERALL TOOL USAGE (sorted by frequency):")
    print(f"{'Tool Name':<40} {'Count':>8} {'Errors':>8}")
    print("-"*60)

    sorted_tools = sorted(
        stats['overall_tool_counts'].items(),
        key=lambda x: x[1],
        reverse=True
    )

    for tool, count in sorted_tools:
        errors = stats['overall_error_counts'].get(tool, 0)
        error_str = f"({errors})" if errors > 0 else ""
        print(f"{tool:<40} {count:>8} {error_str:>8}")

    if verbose:
        print(f"\nINDIVIDUAL LOG FILES:")
        print("-"*80)

        for i, log_stats in enumerate(stats['individual_stats'], 1):
            print(f"\n{i}. {log_stats['log_file']}")
            print(f"   Task: {log_stats['task_name'][:70]}")
            print(f"   Repo: {log_stats['repo_name']}")
            print(f"   Steps: {log_stats['total_steps']}")
            print(f"   Status: {'✓ Completed' if log_stats['completed'] else '✗ Not completed'}")
            print(f"   Tool calls: {log_stats['total_tool_calls']}")
            print(f"   Unique tools: {log_stats['unique_tools_used']}")

            if log_stats['error_counts']:
                print(f"   Errors: {sum(log_stats['error_counts'].values())}")
                for tool, count in log_stats['error_counts'].items():
                    print(f"     - {tool}: {count}")

            print(f"   Top 5 tools:")
            top_tools = sorted(
                log_stats['tool_counts'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            for tool, count in top_tools:
                print(f"     - {tool}: {count}")

    print("\n" + "="*80)


def export_to_json(stats: Dict, output_path: Optional[str] = None):
    """Export statistics to JSON file."""
    if output_path is None:
        script_dir = Path(__file__).parent.resolve()
        stats_dir = script_dir / "statistics"
        stats_dir.mkdir(exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = stats_dir / f"tool_stats_{timestamp}.json"
    else:
        output_path = Path(output_path)
        if not output_path.is_absolute():
            script_dir = Path(__file__).parent.resolve()
            stats_dir = script_dir / "statistics"
            stats_dir.mkdir(exist_ok=True)
            output_path = stats_dir / output_path
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics exported to: {output_path}")


def export_to_csv(stats: Dict, output_path: Optional[str] = None):
    """Export tool usage to CSV file."""
    import csv
    
    if output_path is None:
        script_dir = Path(__file__).parent.resolve()
        stats_dir = script_dir / "statistics"
        stats_dir.mkdir(exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = stats_dir / f"tool_stats_{timestamp}.csv"
    else:
        output_path = Path(output_path)
        if not output_path.is_absolute():
            script_dir = Path(__file__).parent.resolve()
            stats_dir = script_dir / "statistics"
            stats_dir.mkdir(exist_ok=True)
            output_path = stats_dir / output_path

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Tool Name', 'Total Count', 'Error Count', 'Success Rate (%)'])

        sorted_tools = sorted(
            stats['overall_tool_counts'].items(),
            key=lambda x: x[1],
            reverse=True
        )

        for tool, count in sorted_tools:
            errors = stats['overall_error_counts'].get(tool, 0)
            success_rate = ((count - errors) / count * 100) if count > 0 else 0
            writer.writerow([tool, count, errors, f"{success_rate:.1f}"])

    print(f"CSV exported to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Analyze tool usage from agent log files')
    parser.add_argument('log_files', nargs='+', help='Log file paths to analyze')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed per-file statistics')
    parser.add_argument('-j', '--json', help='Export statistics to JSON file')
    parser.add_argument('-c', '--csv', help='Export tool usage to CSV file')

    args = parser.parse_args()

    # Analyze logs
    stats = aggregate_statistics(args.log_files)

    # Print statistics
    print_statistics(stats, verbose=args.verbose)

    if args.json:
        export_to_json(stats, args.json)
    elif not args.csv:
        export_to_json(stats)

    if args.csv:
        export_to_csv(stats, args.csv)


if __name__ == "__main__":

    import sys

    if len(sys.argv) == 1:
        print("Usage: python tools_stats.py <log_file1> [log_file2] ...")
        print("\nOptions:")
        print("  -v, --verbose     Show detailed per-file statistics")
        print("  -j FILE, --json FILE    Export to JSON (default: statistics/tool_stats_TIMESTAMP.json)")
        print("  -c FILE, --csv FILE     Export to CSV (default: statistics/tool_stats_TIMESTAMP.csv)")
        print("\nExample:")
        print("  python tools_stats.py results_dpaia/*.txt -v -j stats.json -c stats.csv")
        print("  python tools_stats.py results_dpaia/*.txt -v  # Auto-saves to statistics/ folder")
        sys.exit(0)

    main()