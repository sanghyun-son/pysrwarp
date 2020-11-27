#ifndef NUM_THREADS
#define NUM_THREADS 1024
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "svf_cuda_half_kernel.cuh"

__global__ void SvfForwardCudaHalfKernel(
    const __half *x,
    const __half *weight,
    __half * __restrict__ y,
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
        __half acc = 0;
        for (int ki = 0; ki < k; ++ki) {
            for (int kj = 0; kj < k; ++kj) {
                acc = __hfma(x[px], weight[pw], acc);
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
__global__ void SvfForwardCudaHalfKernelT(
    const __half *x,
    const __half *weight,
    __half * __restrict__ y,
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
        __half acc = 0;
        #pragma unroll (k)
        for (int ki = 0; ki < k; ++ki) {
            #pragma unroll (k)
            for (int kj = 0; kj < k; ++kj) {
                acc = __hfma(x[px], weight[pw], acc);
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

__global__ void SvfBackwardWeightCudaHalfKernel(
    const __half* __restrict__ x,
    __half* __restrict__ dweight,
    const __half* __restrict__ dy,
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
    // gridDim (k, k, ceil(n / NUM_THREADS))
    // blockDim (NUM_THREADS)
    // pi: Index of the weight
    int pi = blockIdx.z * blockDim.x + threadIdx.x;
    if (pi < n) {
        int ki = blockIdx.x;
        int kj = blockIdx.y;
        int i = gridDim.y * (gridDim.x * pi + ki) + kj;
        int px = ki * w + kj + xi[pi];
        int py = yi[pi];
        // Backup to differentiate w.r.t input
        wi[i] = px;
        __half acc = 0;
        for (int bci = 0; bci < b * c; ++bci) {
            acc = __hfma(dy[py], x[px], acc);
            py += ydim;
            px += xdim;
        }
        dweight[i] = acc;
    }
    return;
}

__global__ void SvfBackwardInputCudaHalfKernel(
    __half* __restrict__ dx,
    const __half* __restrict__ weight,
    const __half* __restrict__ dy,
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
        int px = wi_sorted[pi];
        int bound = (pi == m - 1)? n * kdim: wi_splitter[pi + 1];
        int py_offset = bci * ydim;
        __half acc = 0;
        for (; ps < bound; ++ps) {
            int pw = wi_idx[ps];
            int py = py_offset + yi[int(pw / kdim)];
            acc = __hfma(dy[py], weight[pw], acc);
        }
        dx[bci * xdim + px] = acc;
    }
    return;
}

template <int s>
__global__ void SvfBackwardInputCudaHalfKernelT(
    __half* __restrict__ dx,
    const __half* __restrict__ weight,
    const __half* __restrict__ dy,
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
        int px = wi_sorted[pi];
        int bound = (pi == m - 1)? n << s: wi_splitter[pi + 1];
        int py_offset = bci * ydim;
        __half acc = 0;
        for (; ps < bound; ++ps) {
            int pw = wi_idx[ps];
            int py = py_offset + yi[pw >> s];
            acc = __hfma(dy[py], weight[pw], acc);
        }
        dx[bci * xdim + px] = acc;
    }
    return;
}

void SvfForwardCudaHalf(
    const void* x,
    const void* weight,
    void* y,
    const int b,
    const int c,
    const int h,
    const int w,
    const int hh,
    const int ww,
    const int k,
    const int* xi,
    const int* yi,
    const int n
)
{
    dim3 num_blocks(b, c, int((n + NUM_THREADS - 1) / NUM_THREADS));
    const int xdim = h * w;
    const int ydim = hh * ww;
    switch (k) {
        case 1:
            SvfForwardCudaHalfKernelT<0, 1><<<num_blocks, NUM_THREADS>>>(
                (const __half*)x, (const __half*)weight, (__half*)y,
                w, xdim, ydim, xi, yi, n
            );
            break;
        case 2:
            SvfForwardCudaHalfKernelT<2, 2><<<num_blocks, NUM_THREADS>>>(
                (const __half*)x, (const __half*)weight, (__half*)y,
                w, xdim, ydim, xi, yi, n
            );
            break;
        case 4:
            SvfForwardCudaHalfKernelT<4, 4><<<num_blocks, NUM_THREADS>>>(
                (const __half*)x, (const __half*)weight, (__half*)y,
                w, xdim, ydim, xi, yi, n
            );
            break;
        default:
            SvfForwardCudaHalfKernel<<<num_blocks, NUM_THREADS>>>(
                (const __half*)x, (const __half*)weight, (__half*)y,
                w, k, xdim, ydim, k * k, xi, yi, n
            );
            break;
    }
    return;
}

void SvfBackwardCudaHalf(
    const void* x,
    void* dx,
    const void* weight,
    void* dweight,
    const void* dy,
    const int b,
    const int c,
    const int h,
    const int w,
    const int hh,
    const int ww,
    const int k,
    const int* xi,
    const int* yi,
    const int n
)
{
    dim3 num_blocks_w(k, k, int((n + NUM_THREADS - 1) / NUM_THREADS));
    const int xdim = h * w;
    const int ydim = hh * ww;
    const int kdim = k * k;
    thrust::device_vector<int> wi(kdim * n);
    SvfBackwardWeightCudaHalfKernel<<<num_blocks_w, NUM_THREADS>>>(
        (const __half*)x, (__half*)dweight, (const __half*)dy,
        b, c, w, k,
        xdim, ydim, kdim,
        xi, thrust::raw_pointer_cast(wi.data()), yi,
        n
    );

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
            SvfBackwardInputCudaHalfKernelT<0><<<num_blocks_x, NUM_THREADS>>>(
                (__half*)dx, (const __half*)weight, (const __half*)dy,
                xdim, ydim,
                thrust::raw_pointer_cast(wi.data()),
                thrust::raw_pointer_cast(wi_idx.data()),
                thrust::raw_pointer_cast(wi_splitter.data()),
                yi,
                n, m
            );
            break;
        case 2:
            SvfBackwardInputCudaHalfKernelT<2><<<num_blocks_x, NUM_THREADS>>>(
                (__half*)dx, (const __half*)weight, (const __half*)dy,
                xdim, ydim,
                thrust::raw_pointer_cast(wi.data()),
                thrust::raw_pointer_cast(wi_idx.data()),
                thrust::raw_pointer_cast(wi_splitter.data()),
                yi,
                n, m
            );
            break;
        case 4:
            SvfBackwardInputCudaHalfKernelT<4><<<num_blocks_x, NUM_THREADS>>>(
                (__half*)dx, (const __half*)weight, (const __half*)dy,
                xdim, ydim,
                thrust::raw_pointer_cast(wi.data()),
                thrust::raw_pointer_cast(wi_idx.data()),
                thrust::raw_pointer_cast(wi_splitter.data()),
                yi,
                n, m
            );
            break;
        default:
            SvfBackwardInputCudaHalfKernel<<<num_blocks_x, NUM_THREADS>>>(
                (__half*)dx, (const __half*)weight, (const __half*)dy,
                xdim, ydim, k * k,
                thrust::raw_pointer_cast(wi.data()),
                thrust::raw_pointer_cast(wi_idx.data()),
                thrust::raw_pointer_cast(wi_splitter.data()),
                yi,
                n, m
            );
            break;
    }
    return;

}
