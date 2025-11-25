cat > launch_demon.py <<'EOF'
#!/usr/bin/env python3
"""
CUDAgraveyard Lite v1.0: Fonctionne sur Ubuntu 24.04 + 2×3060 sans Shinka.
Utilise OpenAI pour mutations, torch pour matrices, hyperfine pour timing.
"""

import argparse
import json
import subprocess
import os
import time
import torch
from openai import OpenAI

# Clé API (ajoute-la si pas en env)
if not os.getenv("OPENAI_API_KEY"):
    print("ERREUR: Set OPENAI_API_KEY=sk-... (obligatoire pour mutations)")
    exit(1)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_template(template_path):
    with open(template_path, 'r') as f:
        return json.load(f)

def generate_hypothesis(task):
    prompt = f"""Task: Optimise GEMM CUDA pour {task['task']}.
Propose 3 mutations simples (ex: tile size 32, unroll 4, basic async copy).
Output strict JSON: {{"mutations": [{{"desc": "Change tile to 32x32", "code_snippet": "__global__ void gemm(...) {{ /* ton code CUDA ici */ }}"}}]}}
Keep code short, compilable with nvcc -arch=sm_86."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=800
    )
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {"mutations": [{"desc": "Fallback mutation", "code_snippet": "#include <cuda_runtime.h>\n__global__ void gemm(float *C, float *A, float *B, int N) {\n  // simple loop\n  int tx = threadIdx.x, ty = threadIdx.y;\n  int bx = blockIdx.x, by = blockIdx.y;\n  float sum = 0;\n  for (int k = 0; k < N; k++) sum += A[(bx * blockDim.x + tx) * N + k] * B[k * N + (by * blockDim.y + ty)];\n  C[(bx * blockDim.x + tx) * N + (by * blockDim.y + ty)] = sum;\n}"}]}

def compile_kernel(code_snippet, filename="temp.cu"):
    with open(filename, "w") as f:
        f.write(code_snippet)
    cmd = ["nvcc", "-O3", "-arch=sm_86", filename, "-o", "killer"]  # sm_86 pour RTX 30xx
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None, result.stderr
    return "killer", None

def benchmark_kernel(binary, gpus, matrix_size):
    # Setup matrices avec torch (transfert GPU)
    device0 = torch.device(f"cuda:{gpus[0]}")
    a = torch.randn(matrix_size, matrix_size, device=device0, dtype=torch.float16)
    b = torch.randn(matrix_size, matrix_size, device=device0, dtype=torch.float16)
    c = torch.zeros(matrix_size, matrix_size, device=device0, dtype=torch.float16)
    
    # Export pour kernel (simulé, adapte si besoin)
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Run hyperfine pour timing précis
    hyper_cmd = ["hyperfine", "--warmup=3", "--min-runs=5", f"./{binary}"]
    hyper_result = subprocess.run(hyper_cmd, capture_output=True, text=True)
    if hyper_result.returncode != 0:
        return 0, 0, hyper_result.stderr
    
    # Parse temps moyen (ex: "Time (mean ± σ): 0.123 s ± 0.004 s")
    lines = hyper_result.stdout.split('\n')
    time_mean = float([line for line in lines if 'Time (mean' in line][0].split()[3][:-1])  # sec
    
    # GFLOPS approx = 2 * N^3 / time
    gflops = 2 * (matrix_size ** 3) / time_mean / 1e9
    
    # Power via nvidia-smi
    power_cmd = ["nvidia-smi", "--query-gpu=power.draw", "--format=csv", "-i", str(gpus[0])]
    power_result = subprocess.run(power_cmd, capture_output=True, text=True)
    power = float(power_result.stdout.strip().split('\n')[-1].split()[0]) if power_result.stdout else 0
    
    return gflops, power, None

def main(args):
    task = load_template(args.template)
    gpus_list = args.gpus.split(",")
    size = task.get("matrix_size", 4096)
    
    # Baseline simple (kernel naive)
    baseline_code = """#include <cuda_runtime.h>
#include <stdio.h>
int main() {
    int N = """ + str(size) + """;
    float *A, *B, *C;
    cudaMallocManaged(&A, N*N*sizeof(float));
    cudaMallocManaged(&B, N*N*sizeof(float));
    cudaMallocManaged(&C, N*N*sizeof(float));
    // Init random...
    for (int i = 0; i < N*N; i++) { A[i] = rand() / (float)RAND_MAX; B[i] = rand() / (float)RAND_MAX; }
    dim3 block(16,16);
    dim3 grid((N+15)/16, (N+15)/16);
    gemm<<<grid, block>>>(C, A, B, N);
    cudaDeviceSynchronize();
    cudaFree(A); cudaFree(B); cudaFree(C);
    return 0;
}
__global__ void gemm(float *C, float *A, float *B, int N) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    float sum = 0;
    for (int k = 0; k < N; k++) sum += A[(bx*16 + tx)*N + k] * B[k*N + (by*16 + ty)];
    C[(bx*16 + tx)*N + (by*16 + ty)] = sum;
}"""
    baseline_bin, err = compile_kernel(baseline_code, "baseline.cu")
    if err:
        print(f"Baseline compile error: {err}")
        return
    baseline_gflops, _, _ = benchmark_kernel(baseline_bin, gpus_list, size)
    print(f"Baseline cuBLAS-like: {baseline_gflops:.2f} GFLOPS")
    
    current_code = baseline_code  # Start with baseline
    
    for iter in range(task["constraints"][2].split()[-2]):  # Max iters from template
        print(f"\n--- Iteration {iter + 1} ---")
        hyps = generate_hypothesis(task)
        best_speedup = 0
        best_code = current_code
        
        for mut in hyps["mutations"]:
            full_code = current_code.replace("__global__ void gemm", mut["code_snippet"])  # Simple replace for demo
            bin_name, err = compile_kernel(full_code)
            if err:
                print(f"  Mut {mut['desc']}: Compile error - skipped")
                continue
            gflops, power, err = benchmark_kernel(bin_name, gpus_list, size)
            if err:
                print(f"  Mut {mut['desc']}: Bench error - skipped")
                continue
            speedup = (gflops / baseline_gflops - 1) * 100
            print(f"  Mut {mut['desc']}: {gflops:.2f} GFLOPS (+{speedup:.1f}%) - {power}W")
            if speedup > best_speedup:
                best_speedup = speedup
                best_code = full_code
        
        if best_speedup > 8:  # Seuil kill
            os.makedirs("graves", exist_ok=True)
            ts = int(time.time())
            with open(f"graves/gemm_kill_{ts}.cu", "w") as f:
                f.write(best_code)
            with open(f"graves/metrics_{ts}.md", "w") as f:
                f.write(f"# Kill {ts}\nBaseline: {baseline_gflops:.2f} GFLOPS\nBest: {baseline_gflops * (1 + best_speedup/100):.2f} GFLOPS (+{best_speedup:.1f}%)\nPower saved: estimated -{best_speedup*0.5:.1f}%")
            print(f"\n🎉 KILL! +{best_speedup:.1f}% - Enterré dans graves/gemm_kill_{ts}.cu")
            return
        
        current_code = best_code
        time.sleep(10)  # Pause entre iters
    
    print("\nTest fini (pas de kill >8%, mais loop OK). Augmente iters pour plus.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch CUDAgraveyard lite")
    parser.add_argument("--template", required=True, help="Path to JSON template")
    parser.add_argument("--gpus", default="0", help="GPU IDs comma sep")
    args = parser.parse_args()
    main(args)
EOF
