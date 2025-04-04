#include <iostream>
#include <cmath>
#include <stdint.h>
#include <cuda_runtime.h>
#include <omp.h>

// CUDA 规约求和的线程块大小
const int TILE_SIZE = 16;

__global__ void sgemm_kernel(const int64_t N, const int64_t M, const int64_t K, float *A, float *B, float *C) {
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < N && t * TILE_SIZE + tx < K) {
            sharedA[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            sharedA[ty][tx] = 0.0f;
        }

        if (col < M && t * TILE_SIZE + ty < K) {
            sharedB[ty][tx] = B[(t * TILE_SIZE + ty) * M + col];
        } else {
            sharedB[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sharedA[ty][k] * sharedB[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < M) {
        C[row * M + col] = sum;
    }
}

void Sgemm(const int64_t N, const int64_t M, const int64_t K, float *A, float *B, float *C) {
    float *d_A, *d_B, *d_C;
    size_t size_A = N * K * sizeof(float);
    size_t size_B = K * M * sizeof(float);
    size_t size_C = N * M * sizeof(float);

    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
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
                sum += A[n * K + k] * B[k * M + m];
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
    float *B = new float[K * M];
    float *C_cuda = new float[N * M];
    float *C_omp = new float[N * M];

    // 初始化矩阵 A 和 B
    for (int64_t i = 0; i < N * K; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int64_t i = 0; i < K * M; ++i) {
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 使用 CUDA 计算矩阵乘法
    Sgemm(N, M, K, A, B, C_cuda);

    // 使用 OpenMP 计算矩阵乘法
    sgemm(N, M, K, A, B, C_omp);

    // 计算最大绝对误差
    float max_error = 0.0f;
    for (int64_t i = 0; i < N * M; ++i) {
        float error = std::abs(C_cuda[i] - C_omp[i]);
        if (error > max_error) {
            max_error = error;
        }
    }

    std::cout << "Maximum absolute error: " << max_error << std::endl;

    // 释放内存
    delete[] A;
    delete[] B;
    delete[] C_cuda;
    delete[] C_omp;

    return 0;
}
