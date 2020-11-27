#include <torch/extension.h>
#include "svf_cuda_projective_grid_kernel.cuh"
#include "svf_cuda_kernel.cuh"
#include "svf_cuda_half_kernel.cuh"

void SvfProjectiveGrid(
    const torch::Tensor m,
    const int h,
    const int w,
    torch::Tensor grid,
    const float eps_y,
    const float eps_x
)
{
    SvfProjectiveGridCuda(
        m.data_ptr<float>(), h, w, grid.data_ptr<float>(), eps_y, eps_x
    );
    return;
}

void SvfForward(
    const torch::Tensor x,
    const torch::Tensor weight,
    torch::Tensor y,
    const int k,
    const torch::Tensor xi,
    const torch::Tensor yi,
    const bool is_half
)
{
    if (is_half) {
        SvfForwardCudaHalf(
            x.data_ptr(), weight.data_ptr(), y.data_ptr(),
            x.size(0), x.size(1), x.size(2), x.size(3),
            y.size(-2), y.size(-1),
            k,
            xi.data_ptr<int>(), yi.data_ptr<int>(), yi.size(0)
        );
    }
    else{
        bool depthwise = (weight.dim() == 3 && weight.size(1) > 1);
        SvfForwardCuda(
            x.data_ptr<float>(), weight.data_ptr<float>(), y.data_ptr<float>(),
            x.size(0), x.size(1), x.size(2), x.size(3),
            y.size(-2), y.size(-1),
            k,
            xi.data_ptr<int>(), yi.data_ptr<int>(), yi.size(0),
            depthwise
        );
    }
    return;
}

void SvfBackward(
    const torch::Tensor x,
    torch::Tensor dx,
    const torch::Tensor weight,
    torch::Tensor dweight,
    const torch::Tensor dy,
    const int k,
    const torch::Tensor xi,
    const torch::Tensor yi,
    const bool is_half
)
{
    if (is_half) {
        SvfBackwardCudaHalf(
            x.data_ptr(), dx.data_ptr(),
            weight.data_ptr(), dweight.data_ptr(),
            dy.data_ptr(),
            dx.size(0), dx.size(1), dx.size(2), dx.size(3),
            dy.size(-2), dy.size(-1),
            k,
            xi.data_ptr<int>(), yi.data_ptr<int>(), yi.size(0)
        );
    }
    else {
        bool depthwise = (weight.dim() == 3 && weight.size(1) > 1);
        SvfBackwardCuda(
            x.data_ptr<float>(), dx.data_ptr<float>(),
            weight.data_ptr<float>(), dweight.data_ptr<float>(),
            dy.data_ptr<float>(),
            dx.size(0), dx.size(1), dx.size(2), dx.size(3),
            dy.size(-2), dy.size(-1),
            k,
            xi.data_ptr<int>(), yi.data_ptr<int>(), yi.size(0),
            depthwise
        );
    }
    return;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("projective_grid", &SvfProjectiveGrid, "Projective_grid");
    m.def("forward", &SvfForward, "SVF_forward");
    m.def("backward", &SvfBackward, "SVF_backward");
}
