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

#endif