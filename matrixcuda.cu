#include <iostream>
#include "cuda_runtime.h"

__global__ void sgemm_kernel(const int64_t N, const int64_t M, const int64_t K, float *A, float *B, float *C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = idx / M;
    int m = idx % M;

    if (n < N && m < M) {
        float sum = 0;
        for (int64_t k = 0; k < K; ++k) {
            sum += A[n * K + k] * B[m * K + k];
        }
        C[n * M + m] = sum;
    }
}

void Sgemm(const int64_t N, const int64_t M, const int64_t K, float *A, float *B, float *C) {
    float *d_A, *d_B, *d_C;
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&d_A, N * K * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc d_A failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return;
    }
    cudaStatus = cudaMalloc((void**)&d_B, M * K * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc d_B failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_A);
        return;
    }
    cudaStatus = cudaMalloc((void**)&d_C, N * M * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc d_C failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        return;
    }

    cudaStatus = cudaMemcpy(d_A, A, N * K * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy h_A -> d_A failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }
    cudaStatus = cudaMemcpy(d_B, B, M * K * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy h_B -> d_B failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    int blockSize = 256;
    int gridSize = (N * M + blockSize - 1) / blockSize;

    sgemm_kernel<<<gridSize, blockSize>>>(N, M, K, d_A, d_B, d_C);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    cudaStatus = cudaMemcpy(C, d_C, N * M * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy d_C -> h_C failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
