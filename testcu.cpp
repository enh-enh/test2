#include <iostream>
#include <cmath>
#include <stdint.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <cublas_v2.h>

// 封装的 sgemm 函数，使用 CUBLAS 优化
void Sgemm(const int64_t N, const int64_t M, const int64_t K, float *A, float *B, float *C) {
    cublasHandle_t handle;
    float *d_A, *d_B, *d_C;
    size_t size_A = N * K * sizeof(float);
    size_t size_B = M * K * sizeof(float);
    size_t size_C = N * M * sizeof(float);

    // 分配 GPU 内存
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    // 将数据从 CPU 复制到 GPU
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    // 创建 CUBLAS 句柄
    cublasCreate(&handle);

    // 设置矩阵乘法的系数
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 调用 CUBLAS 的 SGEMM 函数执行矩阵乘法
    // CUBLAS_OP_N 表示矩阵不进行转置
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), 
                &alpha, d_B, static_cast<int>(M), 
                d_A, static_cast<int>(K), 
                &beta, d_C, static_cast<int>(M));

    // 将结果从 GPU 复制回 CPU
    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    // 销毁 CUBLAS 句柄
    cublasDestroy(handle);

    // 释放 GPU 内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void sgemm(const int64_t N, const int64_t M, const int64_t K, float *A, float *B, float *C) {
	float sum ;
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
