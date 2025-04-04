#include <stdint.h>
#include <cuda_runtime.h>
#include "utils.h"

__global__ void image_transform_kernel(float *__restrict__ packed_image,
                                       float *__restrict__ V,
                                       const V_shape_t vs,
                                       const tiling_info_t ti,
                                       const int64_t collapsed_dim_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < collapsed_dim_size) {
        for (int64_t w = 0; w < ti.tile_in_w; ++w) {
            float z0, z1, z2, z3, z4, z5, z6;

            z6 = packed_image[(0 * ti.tile_in_w + w) * collapsed_dim_size + idx];
            z0 = 4.0f * z6;

            z6 = packed_image[(1 * ti.tile_in_w + w) * collapsed_dim_size + idx];
            z1 = -4.0f * z6;
            z2 = 4.0f * z6;
            z3 = -2.0f * z6;
            z4 = 2.0f * z6;
            z5 = 4.0f * z6;

            z6 = packed_image[(2 * ti.tile_in_w + w) * collapsed_dim_size + idx];
            z0 += -5.0f * z6;
            z1 += -4.0f * z6;
            z2 += -4.0f * z6;
            z3 += -z6;
            z4 += -z6;

            z6 = packed_image[(3 * ti.tile_in_w + w) * collapsed_dim_size + idx];
            z1 += z6;
            z2 += -z6;
            z3 += 2.0f * z6;
            z4 += -2.0f * z6;
            z5 += -5.0f * z6;

            z6 = packed_image[(4 * ti.tile_in_w + w) * collapsed_dim_size + idx];
            z0 += z6;
            z1 += z6;
            z2 += z6;
            z3 += z6;
            z4 += z6;

            z6 = packed_image[(5 * ti.tile_in_w + w) * collapsed_dim_size + idx];
            z5 += z6;

            V[(0 * ti.tile_in_w + w) * collapsed_dim_size + idx] = z0;
            V[(1 * ti.tile_in_w + w) * collapsed_dim_size + idx] = z1;
            V[(2 * ti.tile_in_w + w) * collapsed_dim_size + idx] = z2;
            V[(3 * ti.tile_in_w + w) * collapsed_dim_size + idx] = z3;
            V[(4 * ti.tile_in_w + w) * collapsed_dim_size + idx] = z4;
            V[(5 * ti.tile_in_w + w) * collapsed_dim_size + idx] = z5;
        }

        for (int64_t h = 0; h < ti.tile_in_h; ++h) {
            float z0, z1, z2, z3, z4, z5, z6;

            z6 = V[(h * ti.tile_in_w + 0) * collapsed_dim_size + idx];
            z0 = 4.0f * z6;

            z6 = V[(h * ti.tile_in_w + 1) * collapsed_dim_size + idx];
            z1 = -4.0f * z6;
            z2 = 4.0f * z6;
            z3 = -2.0f * z6;
            z4 = 2.0f * z6;
            z5 = 4.0f * z6;

            z6 = V[(h * ti.tile_in_w + 2) * collapsed_dim_size + idx];
            z0 += -5.0f * z6;
            z1 += -4.0f * z6;
            z2 += -4.0f * z6;
            z3 += -z6;
            z4 += -z6;

            z6 = V[(h * ti.tile_in_w + 3) * collapsed_dim_size + idx];
            z1 += z6;
            z2 += -z6;
            z3 += 2.0f * z6;
            z4 += -2.0f * z6;
            z5 += -5.0f * z6;

            z6 = V[(h * ti.tile_in_w + 4) * collapsed_dim_size + idx];
            z0 += z6;
            z1 += z6;
            z2 += z6;
            z3 += z6;
            z4 += z6;

            z6 = V[(h * ti.tile_in_w + 5) * collapsed_dim_size + idx];
            z5 += z6;

            V[(h * ti.tile_in_w + 0) * collapsed_dim_size + idx] = z0;
            V[(h * ti.tile_in_w + 1) * collapsed_dim_size + idx] = z1;
            V[(h * ti.tile_in_w + 2) * collapsed_dim_size + idx] = z2;
            V[(h * ti.tile_in_w + 3) * collapsed_dim_size + idx] = z3;
            V[(h * ti.tile_in_w + 4) * collapsed_dim_size + idx] = z4;
            V[(h * ti.tile_in_w + 5) * collapsed_dim_size + idx] = z5;
        }
    }
}

void image_transform_cuda(float *__restrict__ packed_image,
                     float *__restrict__ V,
                     const V_shape_t vs,
                     const tiling_info_t ti,
                     const int64_t collapsed_dim_size) {
    float *d_packed_image, *d_V;
    size_t size = ti.tile_in_w * collapsed_dim_size * 6 * sizeof(float);

    cudaMalloc((void**)&d_packed_image, size);
    cudaMalloc((void**)&d_V, size);

    cudaMemcpy(d_packed_image, packed_image, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (collapsed_dim_size + threadsPerBlock - 1) / threadsPerBlock;

    image_transform_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_packed_image, d_V, vs, ti, collapsed_dim_size);

    cudaMemcpy(V, d_V, size, cudaMemcpyDeviceToHost);

    cudaFree(d_packed_image);
    cudaFree(d_V);
}

__global__ void image_packing_kernel(float *__restrict__ image,
                                     float *__restrict__ packed_image,
                                     const image_shape_t is,
                                     const tiling_info_t ti) {
    int tile = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile < ti.num_tiles) {
        for (int64_t ic = 0; ic < is.ic; ic++) {
            for (int64_t h = 0; h < ti.tile_in_h; ++h) {
                for (int64_t w = 0; w < ti.tile_in_w; ++w) {
                    tile_index_t tidx = get_tile_index(tile, ti);
                    int64_t batch = tidx.b, ww = tidx.tw, hh = tidx.th;
                    if (hh * 4 + h < is.h && ww * 4 + w < is.w) {
                        int image_index = batch * is.ic * is.h * is.w + ic * is.h * is.w + (hh * 4 + h) * is.w + (ww * 4 + w);
                        int packed_index = h * ti.tile_in_w * ti.num_tiles * is.ic + w * ti.num_tiles * is.ic + tile * is.ic + ic;
                        packed_image[packed_index] = image[image_index];
                    } else {
                        int packed_index = h * ti.tile_in_w * ti.num_tiles * is.ic + w * ti.num_tiles * is.ic + tile * is.ic + ic;
                        packed_image[packed_index] = 0;
                    }
                }
            }
        }
    }
}

void image_packing_cuda(float *__restrict__ image,
                   float *__restrict__ packed_image,
                   const image_shape_t is,
                   const tiling_info_t ti) {
    float *d_image, *d_packed_image;
    size_t image_size = sizeof(float) * is.ic * is.h * is.w;
    size_t packed_image_size = sizeof(float) * ti.tile_in_h * ti.tile_in_w * ti.num_tiles * is.ic;

    cudaMalloc((void**)&d_image, image_size);
    cudaMalloc((void**)&d_packed_image, packed_image_size);

    cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (ti.num_tiles + threadsPerBlock - 1) / threadsPerBlock;

    image_packing_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_packed_image, is, ti);

    cudaMemcpy(packed_image, d_packed_image, packed_image_size, cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_packed_image);
}    