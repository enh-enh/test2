#include <iostream>
#include <cmath>
#include <stdint.h>
#include <cuda_runtime.h>
#include <omp.h>

__global__ void sgemm_kernel(const int64_t N, const int64_t M, const int64_t K, float *A, float *B, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M) {
        float sum = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[col * K + k];
        }
        C[row * M + col] = sum;
    }
}

void Sgemm(const int64_t N, const int64_t M, const int64_t K, float *A, float *B, float *C) {
    float *d_A, *d_B, *d_C;
    size_t size_A = N * K * sizeof(float);
    size_t size_B = M * K * sizeof(float);
    size_t size_C = N * M * sizeof(float);

    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((M + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    sgemm_kernel<<<gridSize, blockSize>>>(N, M, K, d_A, d_B, d_C);

    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void sgemm(const int64_t N, const int64_t M, const int64_t K, float *A, float *B, float *C) {
    float sum;
    #pragma omp parallel for collapse(2)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t m = 0; m < M; ++m) {
            sum = 0;
            for (int64_t k = 0; k < K; ++k) {
                sum += A[n * K + k] * B[m * K + k];
            }
            C[n * M + m] = sum;
        }
    }
}

int main() {
    const int64_t N = 100;
    const int64_t M = 100;
    const int64_t K = 100;

    float *A = new float[N * K];
    float *B = new float[M * K];
    float *C_cuda = new float[N * M];
    float *C_omp = new float[N * M];

    // Initialize matrices A and B
    for (int64_t i = 0; i < N * K; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int64_t i = 0; i < M * K; ++i) {
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Compute matrix multiplication using CUDA
    Sgemm(N, M, K, A, B, C_cuda);

    // Compute matrix multiplication using OpenMP
    sgemm(N, M, K, A, B, C_omp);

    // Compute the maximum absolute error
    float max_error = 0.0f;
    for (int64_t i = 0; i < N * M; ++i) {
        float error = std::abs(C_cuda[i] - C_omp[i]);
        if (error > max_error) {
            max_error = error;
        }
    }

    std::cout << "Maximum absolute error: " << max_error << std::endl;

    // Clean up memory
    delete[] A;
    delete[] B;
    delete[] C_cuda;
    delete[] C_omp;

    return 0;
}
