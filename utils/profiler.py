"""
CUDAgraveyard utils: Profiling wrapper for nsight-compute/nvidia-smi
"""

import subprocess
import os

def get_cpu_power_rapl():
    """
    Read CPU power using RAPL (Linux).
    """
    try:
        with open("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj", "r") as f:
            energy_start = int(f.readline().strip())
        import time; time.sleep(1)  # Sample over 1s
        with open("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj", "r") as f:
            energy_end = int(f.readline().strip())
        power_watts = (energy_end - energy_start) / 1e6  # Watts
        return power_watts
    except FileNotFoundError:
        return None  # Not available

def profile_kernel(kernel_file, nsight_profile=True, power_monitor=True, cpu_power=False):
    """
    Profile kernel with nsight-compute, GPU power, and optional CPU RAPL.
    Returns dict with metrics.
    """
    results = {}

    if nsight_profile:
        try:
            # Run nsight-compute summary
            cmd = ["ncu", "--csv", "--target-processes=all", "--kernel-name=gemm", kernel_file]
            result = subprocess.run(cmd, capture_output=False, text=True)
            results["ncu_summary"] = result.stdout
        except FileNotFoundError:
            results["ncu_error"] = "nsight-compute not found, install NVIDIA Nsight Compute"

    if power_monitor:
        try:
            # Sample power during run
            cmd = ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            results["gpu_power_watts"] = float(result.stdout.strip())
        except (FileNotFoundError, ValueError):
            results["gpu_power_error"] = "nvidia-smi not available or failed"

    if cpu_power:
        cpu_pw = get_cpu_power_rapl()
        if cpu_pw is not None:
            results["cpu_power_watts"] = cpu_pw

    return results

if __name__ == "__main__":
    profile_kernel("temp.cu")
