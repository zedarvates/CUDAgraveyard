// Simple cuBLAS GEMM baseline for 8192x8192 FP16
// Compile with nvcc -O3 -lcublas -o baseline gemm_baseline.cu

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <chrono>

#define N 8192
#define HALF __half

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    std::vector<HALF> h_a(N * N), h_b(N * N), h_c(N * N, 0);
    // Initialize with random data, omitted for brevity

    HALF *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N*N*sizeof(HALF));
    cudaMalloc(&d_b, N*N*sizeof(HALF));
    cudaMalloc(&d_c, N*N*sizeof(HALF));

    cudaMemcpy(d_a, h_a.data(), N*N*sizeof(HALF), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), N*N*sizeof(HALF), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();
    float alpha = 1.0f, beta = 0.0f;
    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_a, N, d_b, N, &beta, d_c, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time: " << diff.count() << " s" << std::endl;

    // Calculate TFLOPS approx
    double flops = (double)N * N * N * 2; // Mul-add
    std::cout << "Approx TFLOPS: " << flops / diff.count() / 1e12 << std::endl;

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cublasDestroy(handle);
    return 0;
}
