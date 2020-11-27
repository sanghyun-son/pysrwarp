#ifndef SVF_CUDA_KERNEL
#define SVF_CUDA_KERNEL

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
);

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
);

/*
// Depthwise version
void SvfBackwardDCuda(
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
    const int n
);
*/
#endif