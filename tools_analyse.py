import os
import re
import json
from collections import Counter, defaultdict
from pathlib import Path

def extract_tool_calls_from_log(log_file_path):
    tool_calls = []

    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = r'Tool call:\s+(\w+)'
    matches = re.findall(pattern, content)
    tool_calls.extend(matches)

    return tool_calls

def collect_tool_stats(base_dir='.'):
    overall_stats = Counter()
    project_stats = defaultdict(Counter)
    file_stats = defaultdict(Counter)

    base_path = Path(base_dir)

    for folder in ['astropy', 'django', 'sympy']:
        folder_path = base_path / folder
        if not folder_path.exists():
            print(f"Warning: {folder} directory not found")
            continue

        for log_file in folder_path.glob('agent_log_*.txt'):
            print(f"Processing: {log_file}")

            tool_calls = extract_tool_calls_from_log(log_file)

            for tool in tool_calls:
                overall_stats[tool] += 1
                project_stats[folder][tool] += 1
                file_stats[str(log_file.relative_to(base_path))][tool] += 1

    return overall_stats, project_stats, file_stats

def print_statistics(overall_stats, project_stats, file_stats):
    print("\n" + "="*80)
    print("OVERALL TOOL USAGE STATISTICS")
    print("="*80)
    print(f"\nTotal unique tools used: {len(overall_stats)}")
    print(f"Total tool calls: {sum(overall_stats.values())}")
    print("\nTool usage (sorted by frequency):")
    print("-" * 60)

    for tool, count in overall_stats.most_common():
        percentage = (count / sum(overall_stats.values())) * 100
        print(f"  {tool:40} {count:6} ({percentage:5.1f}%)")

    print("\n" + "="*80)
    print("TOOL USAGE BY PROJECT")
    print("="*80)

    for project in sorted(project_stats.keys()):
        stats = project_stats[project]
        print(f"\n{project.upper()}:")
        print(f"  Total tool calls: {sum(stats.values())}")
        print(f"  Unique tools: {len(stats)}")
        print("  Top 5 tools:")
        for tool, count in stats.most_common(5):
            print(f"    {tool:38} {count:4}")

    print("\n" + "="*80)
    print("TOOL USAGE BY FILE")
    print("="*80)

    for file_path in sorted(file_stats.keys()):
        stats = file_stats[file_path]
        print(f"\n{file_path}:")
        print(f"  Total tool calls: {sum(stats.values())}")
        for tool, count in stats.most_common():
            print(f"    {tool:38} {count:4}")

def save_statistics_to_json(overall_stats, project_stats, file_stats, output_file='tool_usage_stats.json'):
    stats = {
        'overall': dict(overall_stats),
        'by_project': {project: dict(stats) for project, stats in project_stats.items()},
        'by_file': {file: dict(stats) for file, stats in file_stats.items()},
        'summary': {
            'total_tool_calls': sum(overall_stats.values()),
            'unique_tools_used': len(overall_stats),
            'most_used_tool': overall_stats.most_common(1)[0] if overall_stats else None
        }
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    print(f"\nStatistics saved to {output_file}")

def main():
    print("Collecting tool usage statistics from agent logs...")

    overall_stats, project_stats, file_stats = collect_tool_stats('.')

    if not overall_stats:
        print("No tool usage found in the logs!")
        return

    print_statistics(overall_stats, project_stats, file_stats)

    save_statistics_to_json(overall_stats, project_stats, file_stats)

if __name__ == '__main__':
    main()