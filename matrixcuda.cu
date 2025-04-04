#include <iostream>
#include <cmath>
#include <stdint.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <cublas_v2.h>

void Sgemm(const int64_t N, const int64_t M, const int64_t K, float *A, float *B, float *C) {
    cublasHandle_t handle;
    float *d_A, *d_B, *d_C;
    size_t size_A = N * K * sizeof(float);
    size_t size_B = M * K * sizeof(float);
    size_t size_C = N * M * sizeof(float);

    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), 
                &alpha, d_B, static_cast<int>(M), 
                d_A, static_cast<int>(K), 
                &beta, d_C, static_cast<int>(M));

    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    cublasDestroy(handle);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
