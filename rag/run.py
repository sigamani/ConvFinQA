import json
from benchmark_agent import run_benchmark

with open("dev_converted_full.json") as f:
    examples = json.load(f)

run_benchmark(examples)
