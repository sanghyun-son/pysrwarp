#ifndef NUM_THREADS
#define NUM_THREADS 1024
#endif

#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "svf_cuda_kernel.cuh"

/*
What we want to do:
    Args:
        x:              Input images of (B, C, H, W).
        weight:         Spatially-varying filter weights of (N, k^2).
        y:              Output images of (B, C, H', W'). Only N pixels are valid.
        k:              Dimension of the filter weights.
        ix:             Indices of (N,) to indicate which positions to be sampled from x.
        iy:             Indices of (N,)

    Formulation:
        y[iy[i]] = sum(x[ix[i]] * weight[i])
        ix and iy should be converted to spatial coordinates.

    Return:
        y
*/

__global__ void SvfForwardCudaKernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ y,
    const int w,
    const int k,
    const int xdim,
    const int ydim,
    const int kdim,
    const int* __restrict__ xi,
    const int* __restrict__ yi,
    const int n
)
{
    int pi = blockIdx.z * blockDim.x + threadIdx.x;
    if (pi < n) {
        int bci = blockIdx.x * gridDim.y + blockIdx.y;
        int px = bci * xdim + xi[pi];
        int pw = kdim * pi;
        float acc = 0;
        for (int ki = 0; ki < k; ++ki) {
            for (int kj = 0; kj < k; ++kj) {
                acc += x[px] * weight[pw];
                ++px;
                ++pw;
            }
            px += (w - k);
        }
        int py = bci * ydim + yi[pi];
        y[py] = acc;
    }
    return;
}

__global__ void SvfForwardCudaKernelD(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ y,
    const int w,
    const int k,
    const int xdim,
    const int ydim,
    const int kdim,
    const int* __restrict__ xi,
    const int* __restrict__ yi,
    const int n
)
{
    int pi = blockIdx.z * blockDim.x + threadIdx.x;
    if (pi < n) {
        int ci = blockIdx.y;
        int bci = blockIdx.x * gridDim.y + ci;
        int px = bci * xdim + xi[pi];
        int pw = kdim * (pi * gridDim.y + ci);
        float acc = 0;
        for (int ki = 0; ki < k; ++ki) {
            for (int kj = 0; kj < k; ++kj) {
                acc += x[px] * weight[pw];
                ++px;
                ++pw;
            }
            px += (w - k);
        }
        int py = bci * ydim + yi[pi];
        y[py] = acc;
    }
    return;
}

template <int s, int k>
__global__ void SvfForwardCudaKernelT(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ y,
    const int w,
    const int xdim,
    const int ydim,
    const int* __restrict__ xi,
    const int* __restrict__ yi,
    const int n
)
{
    int pi = blockIdx.z * blockDim.x + threadIdx.x;
    if (pi < n) {
        int bci = blockIdx.x * gridDim.y + blockIdx.y;
        int px = bci * xdim + xi[pi];
        int pw = pi << s;
        float acc = 0;
        #pragma unroll (k)
        for (int ki = 0; ki < k; ++ki) {
            #pragma unroll (k)
            for (int kj = 0; kj < k; ++kj) {
                acc += x[px] * weight[pw];
                ++px;
                ++pw;
            }
            px += (w - k);
        }
        int py = bci * ydim + yi[pi];
        y[py] = acc;
    }
    return;
}

template <int s, int k>
__global__ void SvfForwardCudaKernelDT(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ y,
    const int w,
    const int xdim,
    const int ydim,
    const int* __restrict__ xi,
    const int* __restrict__ yi,
    const int n
)
{
    int pi = blockIdx.z * blockDim.x + threadIdx.x;
    if (pi < n) {
        int ci = blockIdx.y;
        int bci = blockIdx.x * gridDim.y + ci;
        int px = bci * xdim + xi[pi];
        int pw = (pi * gridDim.y + ci) << s;
        float acc = 0;
        #pragma unroll (k)
        for (int ki = 0; ki < k; ++ki) {
            #pragma unroll (k)
            for (int kj = 0; kj < k; ++kj) {
                acc += x[px] * weight[pw];
                ++px;
                ++pw;
            }
            px += (w - k);
        }
        int py = bci * ydim + yi[pi];
        y[py] = acc;
    }
    return;
}

__global__ void SvfBackwardWeightCudaKernel(
    const float* __restrict__ x,
    float* __restrict__ dweight,
    const float* __restrict__ dy,
    const int b,
    const int c,
    const int w,
    const int k,
    const int xdim,
    const int ydim,
    const int kdim,
    const int* __restrict__ xi,
    // Same size with weight, dweight
    int* __restrict__ wi,
    const int* __restrict__ yi,
    const int n
)
{
    // gridDim (k, k, n // NUM_THREADS)
    // blockDim (NUM_THREADS)
    // pi: Index of the spatial weight
    int pi = blockIdx.z * blockDim.x + threadIdx.x;
    if (pi < n) {
        int ki = blockIdx.x;
        int kj = blockIdx.y;
        int i = gridDim.y * (gridDim.x * pi + ki) + kj;
        int px = ki * w + kj + xi[pi];
        int py = yi[pi];
        // Backup to differentiate w.r.t input
        wi[i] = px;
        float acc = 0;
        for (int bci = 0; bci < b * c; ++bci) {
            acc += dy[py] * x[px];
            py += ydim;
            px += xdim;
        }
        dweight[i] = acc;
    }
    return;
}

__global__ void SvfBackwardWeightCudaKernelD(
    const float* __restrict__ x,
    float* __restrict__ dweight,
    const float* __restrict__ dy,
    const int b,
    const int c,
    const int w,
    const int k,
    const int xdim,
    const int ydim,
    const int kdim,
    const int* __restrict__ xi,
    // Same size with weight, dweight
    int* __restrict__ wi,
    const int* __restrict__ yi,
    const int n
)
{
    // gridDim (n, c)
    // blockDim (k, k)
    int pi = blockIdx.x;
    int ci = blockIdx.y;
    int ki = threadIdx.x;
    int kj = threadIdx.y;
    // Index of the current weight
    int i = kj + blockDim.y * ki + kdim * (ci + c * pi);
    int px = xdim * ci + kj + w * ki + xi[pi];
    int py = ydim * ci + yi[pi];
    if (ci == 0) {
        // Ignore the channel dimension
        int px_base = kj + w * ki + xi[pi];
        int i_base = kj + blockDim.y * ki + kdim * pi;
        wi[i_base] = px_base;
    }
    float acc = 0;
    for (int bi = 0; bi < b; ++bi) {
        acc += dy[py] * x[px];
        py += ydim * c;
        px += xdim * c;
    }
    dweight[i] = acc;
    return;
}

__global__ void SvfBackwardInputCudaKernel(
    float* __restrict__ dx,
    const float* __restrict__ weight,
    const float* __restrict__ dy,
    const int xdim,
    const int ydim,
    const int kdim,
    const int* __restrict__ wi_sorted,
    const int* __restrict__ wi_idx,
    const int* __restrict__ wi_splitter,
    const int* __restrict__ yi,
    const int n,
    const int m
)
{
    int pi = blockIdx.z * blockDim.x + threadIdx.x;
    if (pi < m) {
        int bci = blockIdx.x * gridDim.y + blockIdx.y;
        int ps = wi_splitter[pi];
        int bound = (pi == m - 1)? n * kdim: wi_splitter[pi + 1];
        int py_offset = bci * ydim;
        float acc = 0;
        for (; ps < bound; ++ps) {
            int pw = wi_idx[ps];
            int py = py_offset + yi[int(pw / kdim)];
            acc += dy[py] * weight[pw];
        }
        int px = bci * xdim + wi_sorted[pi];
        dx[px] = acc;
    }
    return;
}

__global__ void SvfBackwardInputCudaKernelD(
    float* __restrict__ dx,
    const float* __restrict__ weight,
    const float* __restrict__ dy,
    const int xdim,
    const int ydim,
    const int kdim,
    const int* __restrict__ wi_sorted,
    const int* __restrict__ wi_idx,
    const int* __restrict__ wi_splitter,
    const int* __restrict__ yi,
    const int n,
    const int m
)
{
    // gridDim (b, c, m / NUM_THREADS)
    // Kernel layout: (m, c, kdim)
    int pi = blockIdx.z * blockDim.x + threadIdx.x;
    if (pi < m) {
        int bci = blockIdx.x * gridDim.y + blockIdx.y;
        int ps = wi_splitter[pi];
        int bound = (pi == m - 1)? n * kdim: wi_splitter[pi + 1];
        int py_offset = bci * ydim;
        float acc = 0;
        for (; ps < bound; ++ps) {
            int pw_base = wi_idx[ps];
            int ni = int(pw_base / kdim);
            int py = py_offset + yi[ni];
            int pw_offset = ni * (gridDim.y - 1) + blockIdx.y;
            int pw = pw_base + pw_offset * kdim;
            acc += dy[py] * weight[pw];
        }
        int px = bci * xdim + wi_sorted[pi];
        dx[px] = acc;
    }
    return;
}

template <int s>
__global__ void SvfBackwardInputCudaKernelT(
    float* __restrict__ dx,
    const float* __restrict__ weight,
    const float* __restrict__ dy,
    const int xdim,
    const int ydim,
    const int* __restrict__ wi_sorted,
    const int* __restrict__ wi_idx,
    const int* __restrict__ wi_splitter,
    const int* __restrict__ yi,
    const int n,
    const int m
)
{
    int pi = blockIdx.z * blockDim.x + threadIdx.x;
    if (pi < m) {
        int bci = blockIdx.x * gridDim.y + blockIdx.y;
        int ps = wi_splitter[pi];
        int bound = (pi == m - 1)? n << s: wi_splitter[pi + 1];
        int py_offset = bci * ydim;
        float acc = 0;
        for (; ps < bound; ++ps) {
            int pw = wi_idx[ps];
            int py = py_offset + yi[pw >> s];
            acc += dy[py] * weight[pw];
        }
        int px = bci * xdim + wi_sorted[pi];
        dx[px] = acc;
    }
    return;
}

template <int s>
__global__ void SvfBackwardInputCudaKernelDT(
    float* __restrict__ dx,
    const float* __restrict__ weight,
    const float* __restrict__ dy,
    const int xdim,
    const int ydim,
    const int* __restrict__ wi_sorted,
    const int* __restrict__ wi_idx,
    const int* __restrict__ wi_splitter,
    const int* __restrict__ yi,
    const int n,
    const int m
)
{
    // gridDim (b, c, m / NUN_THREADS)
    int pi = blockIdx.z * blockDim.x + threadIdx.x;
    if (pi < m) {
        int bci = blockIdx.x * gridDim.y + blockIdx.y;
        int ps = wi_splitter[pi];
        int bound = (pi == m - 1)? n << s: wi_splitter[pi + 1];
        int py_offset = bci * ydim;
        float acc = 0;
        for (; ps < bound; ++ps) {
            int pw_base = wi_idx[ps];            
            int ni = pw_base >> s;
            int py = py_offset + yi[ni];
            int pw_offset = ni * (gridDim.y - 1) + blockIdx.y;
            int pw = pw_base + (pw_offset << s);
            acc += dy[py] * weight[pw];
        }
        int px = bci * xdim + wi_sorted[pi];
        dx[px] = acc;
    }
    return;
}

void SvfForwardCuda(
    const float* x,
    const float* weight,
    float* y,
    const int b,
    const int c,
    const int h,
    const int w,
    const int hh,
    const int ww,
    const int k,
    const int* xi,
    const int* yi,
    const int n,
    const bool depthwise
)
{
    //std::cout << b << " " << c << " " << h << " " << w << std::endl;
    dim3 num_blocks(b, c, int((n + NUM_THREADS - 1) / NUM_THREADS));
    const int xdim = h * w;
    const int ydim = hh * ww;
    switch (k) {
        case 1:
            if (depthwise) {
                SvfForwardCudaKernelDT<0, 1><<<num_blocks, NUM_THREADS>>>(
                    x, weight, y, w, xdim, ydim, xi, yi, n
                );
            }
            else {
                SvfForwardCudaKernelT<0, 1><<<num_blocks, NUM_THREADS>>>(
                    x, weight, y, w, xdim, ydim, xi, yi, n
                );
            }
            break;
        case 2:
            if (depthwise) {
                SvfForwardCudaKernelDT<2, 2><<<num_blocks, NUM_THREADS>>>(
                    x, weight, y, w, xdim, ydim, xi, yi, n
                );
            }
            else {
                SvfForwardCudaKernelT<2, 2><<<num_blocks, NUM_THREADS>>>(
                    x, weight, y, w, xdim, ydim, xi, yi, n
                );
            }
            break;
        case 4:
            if (depthwise) {
                SvfForwardCudaKernelDT<4, 4><<<num_blocks, NUM_THREADS>>>(
                    x, weight, y, w, xdim, ydim, xi, yi, n
                );
            }
            else {
                SvfForwardCudaKernelT<4, 4><<<num_blocks, NUM_THREADS>>>(
                    x, weight, y, w, xdim, ydim, xi, yi, n
                );
            }
            break;
        default:
            if (depthwise) {
                SvfForwardCudaKernelD<<<num_blocks, NUM_THREADS>>>(
                    x, weight, y, w, k, xdim, ydim, k * k, xi, yi, n
                );
            }
            else {
                SvfForwardCudaKernel<<<num_blocks, NUM_THREADS>>>(
                    x, weight, y, w, k, xdim, ydim, k * k, xi, yi, n
                );
            }
            break;
    }
    return;
}

void SvfBackwardCuda(
    const float* x,
    float* dx,
    const float* weight,
    float* dweight,
    const float* dy,
    const int b,
    const int c,
    const int h,
    const int w,
    const int hh,
    const int ww,
    const int k,
    const int* xi,
    const int* yi,
    const int n,
    const bool depthwise
)
{
    const int xdim = h * w;
    const int ydim = hh * ww;
    const int kdim = k * k;
    thrust::device_vector<int> wi(kdim * n);
    if (depthwise) {
        dim3 num_blocks_w(n, c);
        dim3 num_threads_w(k, k);
        SvfBackwardWeightCudaKernelD<<<num_blocks_w, num_threads_w>>>(
            x, dweight, dy,
            b, c, w, k,
            xdim, ydim, kdim,
            xi, thrust::raw_pointer_cast(wi.data()), yi,
            n
        );
    }
    else {
        dim3 num_blocks_w(k, k, int((n + NUM_THREADS - 1) / NUM_THREADS));
        SvfBackwardWeightCudaKernel<<<num_blocks_w, NUM_THREADS>>>(
            x, dweight, dy,
            b, c, w, k,
            xdim, ydim, kdim,
            xi, thrust::raw_pointer_cast(wi.data()), yi,
            n
        );
    }
    thrust::device_vector<int> wi_idx(kdim * n);
    thrust::sequence(thrust::device, wi_idx.begin(), wi_idx.end());
    thrust::sort_by_key(thrust::device, wi.begin(), wi.end(), wi_idx.begin());
    thrust::device_vector<int> wi_splitter(kdim * n);
    thrust::sequence(thrust::device, wi_splitter.begin(), wi_splitter.end());
    auto end = thrust::unique_by_key(
        thrust::device, wi.begin(), wi.end(), wi_splitter.begin()
    );
    int m = end.second - wi_splitter.begin();

    dim3 num_blocks_x(b, c, int((m + NUM_THREADS - 1) / NUM_THREADS));
    switch (k) {
        case 1:
            if (depthwise) {
                SvfBackwardInputCudaKernelDT<0><<<num_blocks_x, NUM_THREADS>>>(
                    dx, weight, dy,
                    xdim, ydim,
                    thrust::raw_pointer_cast(wi.data()),
                    thrust::raw_pointer_cast(wi_idx.data()),
                    thrust::raw_pointer_cast(wi_splitter.data()),
                    yi,
                    n, m
                );
            }
            else {
                SvfBackwardInputCudaKernelT<0><<<num_blocks_x, NUM_THREADS>>>(
                    dx, weight, dy,
                    xdim, ydim,
                    thrust::raw_pointer_cast(wi.data()),
                    thrust::raw_pointer_cast(wi_idx.data()),
                    thrust::raw_pointer_cast(wi_splitter.data()),
                    yi,
                    n, m
                );
            }
            break;
        case 2:
            if (depthwise) {
                SvfBackwardInputCudaKernelDT<2><<<num_blocks_x, NUM_THREADS>>>(
                    dx, weight, dy,
                    xdim, ydim,
                    thrust::raw_pointer_cast(wi.data()),
                    thrust::raw_pointer_cast(wi_idx.data()),
                    thrust::raw_pointer_cast(wi_splitter.data()),
                    yi,
                    n, m
                );
            }
            else {
                SvfBackwardInputCudaKernelT<2><<<num_blocks_x, NUM_THREADS>>>(
                    dx, weight, dy,
                    xdim, ydim,
                    thrust::raw_pointer_cast(wi.data()),
                    thrust::raw_pointer_cast(wi_idx.data()),
                    thrust::raw_pointer_cast(wi_splitter.data()),
                    yi,
                    n, m
                );
            }
            break;
        case 4:
            if (depthwise) {
                SvfBackwardInputCudaKernelDT<4><<<num_blocks_x, NUM_THREADS>>>(
                    dx, weight, dy,
                    xdim, ydim,
                    thrust::raw_pointer_cast(wi.data()),
                    thrust::raw_pointer_cast(wi_idx.data()),
                    thrust::raw_pointer_cast(wi_splitter.data()),
                    yi,
                    n, m
                );
            }
            else {
                SvfBackwardInputCudaKernelT<4><<<num_blocks_x, NUM_THREADS>>>(
                    dx, weight, dy,
                    xdim, ydim,
                    thrust::raw_pointer_cast(wi.data()),
                    thrust::raw_pointer_cast(wi_idx.data()),
                    thrust::raw_pointer_cast(wi_splitter.data()),
                    yi,
                    n, m
                );
            }
            break;
        default:
            if (depthwise) {
                SvfBackwardInputCudaKernelD<<<num_blocks_x, NUM_THREADS>>>(
                    dx, weight, dy,
                    xdim, ydim, k * k,
                    thrust::raw_pointer_cast(wi.data()),
                    thrust::raw_pointer_cast(wi_idx.data()),
                    thrust::raw_pointer_cast(wi_splitter.data()),
                    yi,
                    n, m
                );
            }
            else {
                SvfBackwardInputCudaKernel<<<num_blocks_x, NUM_THREADS>>>(
                    dx, weight, dy,
                    xdim, ydim, k * k,
                    thrust::raw_pointer_cast(wi.data()),
                    thrust::raw_pointer_cast(wi_idx.data()),
                    thrust::raw_pointer_cast(wi_splitter.data()),
                    yi,
                    n, m
                );
            }
            break;
    }
    return;
}