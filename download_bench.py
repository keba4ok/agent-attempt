from datasets import load_dataset
ds = load_dataset("SWE-bench/SWE-bench_Verified", split="test")
ds.to_json("swebench_verified.jsonl")