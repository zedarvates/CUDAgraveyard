"""
CUDAgraveyard utils: Correctness checker vs baseline (cosine similarity)
"""

import numpy as np
import torch
import torch.nn.functional as F

def cosine_similarity(a, b):
    """
    Cosine similarity between tensors.
    Returns 0-1 similarity score.
    """
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return F.cosine_similarity(a_flat, b_flat, dim=0).item()

def check_correctness(output: torch.Tensor, baseline: torch.Tensor, threshold=0.9999):
    """
    Check if output matches baseline within 99.99% cosine similarity.
    Returns (bool passed, float similarity)
    """
    sim = cosine_similarity(output, baseline)
    passed = sim >= threshold
    return passed, sim

def generate_random_inputs(n=8192, dtype=torch.float16):
    """
    Generate random inputs for GEMM testing.
    """
    a = torch.randn(n, n, dtype=dtype).cuda()
    b = torch.randn(n, n, dtype=dtype).cuda()
    return a, b

def test_kernel_correctness(kernel_func, baseline_func=torch.matmul, num_tests=100):
    """
    Run num_tests random GEMM and compare to baseline.
    Returns (passed_ratio, avg_sim)
    """
    passed_count = 0
    similarities = []

    for _ in range(num_tests):
        a, b = generate_random_inputs()
        output = kernel_func(a, b)
        baseline = baseline_func(a, b)

        passed, sim = check_correctness(output, baseline)
        if passed:
            passed_count += 1
        similarities.append(sim)

    ratio = passed_count / num_tests
    avg_sim = np.mean(similarities)
    return ratio, avg_sim

if __name__ == "__main__":
    # Dummy kernel for test
    def dummy_kernel(a, b):
        return torch.matmul(a, b)  # Placeholder for custom kernel

    ratio, sim = test_kernel_correctness(dummy_kernel)
    print(f"Passed: {ratio:.2%}, Avg Sim: {sim:.6f}")
