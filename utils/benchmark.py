"""
CUDAgraveyard utils: Benchmarking wrapper using hyperfine
"""

import subprocess
import json
import torch

def benchmark_kernel(binary_path, inputs=None, runs=5, warmup=2):
    """
    Benchmark kernel executable with hyperfine.
    inputs: tuple of args for binary.
    Returns parsed times in seconds.
    """
    cmd = ["hyperfine", "--warmup", str(warmup), "--runs", str(runs), "--export-json", "bench.json", binary_path]
    if inputs:
        cmd.extend(inputs)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Hyperfine failed: {result.stderr}")

    # Parse json
    with open("bench.json", "r") as f:
        data = json.load(f)

    return {
        "mean_time": data["results"][0]["mean"],
        "std_time": data["results"][0]["stddev"],
        "min_time": data["results"][0]["min"],
        "tflops": estimate_tflops_from_time(data["results"][0]["mean"])  # Rough estimate
    }

def estimate_tflops_from_time(time_sec, n=8192, dtype=torch.float16):
    """
    Rough TFLOPS estimate for GEMM nxn.
    """
    ops = n**3 * 2  # Mul-add
    flops = ops / time_sec
    return flops / 1e12

if __name__ == "__main__":
    print(benchmark_kernel("./killer"))
