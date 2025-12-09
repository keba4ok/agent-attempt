from datasets import load_dataset
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
output_path = script_dir / "swebench_verified.jsonl"

ds = load_dataset("SWE-bench/SWE-bench_Verified", split="test")
ds.to_json(str(output_path))
print(f"Dataset saved to: {output_path}")