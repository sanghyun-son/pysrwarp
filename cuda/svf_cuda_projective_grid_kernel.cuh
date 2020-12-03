#ifndef SVF_CUDA_PROJECTIVE_GRID
#define SVF_CUDA_PROJECTIVE_GRID

void SvfProjectiveGridCuda(
    const float* m,
    const int h,
    const int w,
    float* grid,
    const float eps_y,
    const float eps_x
);

void SvfProjectiveGridDoubleCuda(
    const double* m,
    const int h,
    const int w,
    double* grid,
    const double eps_y,
    const double eps_x
);

#endif