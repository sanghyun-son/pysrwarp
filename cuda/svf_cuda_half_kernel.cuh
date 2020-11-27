#ifndef SVF_CUDA_HALF_KERNEL
#define SVF_CUDA_HALF_KERNEL

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
);

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
);

#endif