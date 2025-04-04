// #include "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include\cuda_runtime.h"
#include <stdint.h>
#include <cuda_runtime.h>
#include "utils.h"

__global__ void output_transform_kernel(float *__restrict__ M,
                                        float *__restrict__ Y,
                                        const tiling_info_t ti,
                                        const int64_t collapsed_dim_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < collapsed_dim_size) {
        for (int64_t w = 0; w < ti.tile_in_w; ++w) {
            float z0, z1, z2, z3, z4;

            z4 = M[(0 * ti.tile_in_w + w) * collapsed_dim_size + idx];
            z0 = z4;

            z4 = M[(1 * ti.tile_in_w + w) * collapsed_dim_size + idx];
            z0 = z0 + z4;
            z1 = z4;
            z2 = z4;
            z3 = z4;

            z4 = M[(2 * ti.tile_in_w + w) * collapsed_dim_size + idx];
            z0 += z4;
            z1 += -z4;
            z2 += z4;
            z3 += -z4;

            z4 = M[(3 * ti.tile_in_w + w) * collapsed_dim_size + idx];
            z0 += z4;
            z1 += 2.0f * z4;
            z2 += 4.0f * z4;
            z3 += 8.0f * z4;

            z4 = M[(4 * ti.tile_in_w + w) * collapsed_dim_size + idx];
            z0 += z4;
            z1 += -2.0f * z4;
            z2 += 4.0f * z4;
            z3 += -8.0f * z4;

            z4 = M[(5 * ti.tile_in_w + w) * collapsed_dim_size + idx];
            z3 += z4;

            Y[(0 * ti.tile_in_w + w) * collapsed_dim_size + idx] = z0;
            Y[(1 * ti.tile_in_w + w) * collapsed_dim_size + idx] = z1;
            Y[(2 * ti.tile_in_w + w) * collapsed_dim_size + idx] = z2;
            Y[(3 * ti.tile_in_w + w) * collapsed_dim_size + idx] = z3;
        }

        for (int64_t h = 0; h < ti.tile_out_h; ++h) {
            float z0, z1, z2, z3, z4;

            z4 = Y[(h * ti.tile_in_w + 0) * collapsed_dim_size + idx];
            z0 = z4;

            z4 = Y[(h * ti.tile_in_w + 1) * collapsed_dim_size + idx];
            z0 += z4;
            z1 = z4;
            z2 = z4;
            z3 = z4;

            z4 = Y[(h * ti.tile_in_w + 2) * collapsed_dim_size + idx];
            z0 += z4;
            z1 += -z4;
            z2 += z4;
            z3 += -z4;

            z4 = Y[(h * ti.tile_in_w + 3) * collapsed_dim_size + idx];
            z0 += z4;
            z1 += 2.0f * z4;
            z2 += 4.0f * z4;
            z3 += 8.0f * z4;

            z4 = Y[(h * ti.tile_in_w + 4) * collapsed_dim_size + idx];
            z0 += z4;
            z1 += -2.0f * z4;
            z2 += 4.0f * z4;
            z3 += -8.0f * z4;

            z4 = Y[(h * ti.tile_in_w + 5) * collapsed_dim_size + idx];
            z3 += z4;

            Y[(h * ti.tile_in_w + 0) * collapsed_dim_size + idx] = z0;
            Y[(h * ti.tile_in_w + 1) * collapsed_dim_size + idx] = z1;
            Y[(h * ti.tile_in_w + 2) * collapsed_dim_size + idx] = z2;
            Y[(h * ti.tile_in_w + 3) * collapsed_dim_size + idx] = z3;
        }
    }
}

void output_transform_cuda(float *__restrict__ M,
                      float *__restrict__ Y,
                      const tiling_info_t ti,
                      const int64_t collapsed_dim_size) {
    float *d_M, *d_Y;
    size_t size = ti.tile_in_w * collapsed_dim_size * 6 * sizeof(float);

    cudaMalloc((void**)&d_M, size);
    cudaMalloc((void**)&d_Y, size);

    cudaMemcpy(d_M, M, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (collapsed_dim_size + threadsPerBlock - 1) / threadsPerBlock;

    output_transform_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_M, d_Y, ti, collapsed_dim_size);

    cudaMemcpy(Y, d_Y, size, cudaMemcpyDeviceToHost);

    cudaFree(d_M);
    cudaFree(d_Y);
}