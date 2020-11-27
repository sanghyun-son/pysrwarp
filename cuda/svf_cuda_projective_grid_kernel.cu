#ifndef NUM_THREADS
#define NUM_THREADS 1024
#endif

#include <cuda.h>
#include <cuda_runtime.h>

#include "svf_cuda_projective_grid_kernel.cuh"

__global__ void SvfProjectiveGridCudaKernel(
    const float* __restrict__ m,
    const int h,
    const int w,
    float* __restrict__ grid,
    const float eps_y,
    const float eps_x
)
{
    int px = blockIdx.y * blockDim.x + threadIdx.x;
    if (px >= w) {
        return;
    }

    int p = blockIdx.x * w + px;
    float pos_i = float(blockIdx.x) + eps_y;
    float pos_j = float(px) + eps_x;
    float x = pos_j * m[0] + pos_i * m[1] + m[2];
    float y = pos_j * m[3] + pos_i * m[4] + m[5];
    float z = pos_j * m[6] + pos_i * m[7] + m[8];

    x /= z;
    y /= z;

    grid[2 * p] = x;
    grid[2 * p + 1] = y;
    return;
}


void SvfProjectiveGridCuda(
    const float* m,
    const int h,
    const int w,
    float* grid,
    const float eps_y,
    const float eps_x
)
{
    dim3 num_blocks(h, int((w + NUM_THREADS - 1) / NUM_THREADS));
    SvfProjectiveGridCudaKernel<<<num_blocks, NUM_THREADS>>>(
        m, h, w, grid, eps_y, eps_x
    );
    return;
}
