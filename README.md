# PySRWarp

A core part implementation of the paper \
Sanghyun Son and Kyoung Mu Lee, "SRWarp: Generalized Image Super-Resolution under Arbitrary Transformation," **CVPR** 2021

## Introduction

In the image warping process, the target image can be obtained by applying spatially-varying filters to each local patch in a source image.
Such an operation can be easily implemented with the ```im2col``` function widely used in image processing. However, it is relatively slow and occupies large memory.

This repository provides an efficient PyTorch implementation with a CUDA backend for the spatially-varying filtering with the following specifications:

- A reference implementation of the paper: SRWarp
- Batching
- Channel-wise independent filters
- Automatic differentiation
- Miscellaneous utilities for image warping

Please note that we provide this repository as a standalone so that you can easily utilize it in your project as well.
Full implementations, including training/test scripts, datasets, and models, can be found from [here](https://github.com/sanghyun-son/srwarp).
Since I am a beginner in CUDA programming, there may exist several rooms for improvement.
Any suggestions are welcomed to make this repository better.

## Requirements

This repository is tested under:

- Ubuntu 18.04
- CUDA 10.1 (Compute Capability 7.5)
- PyTorch 1.6

We have found some issues with CUDA 10.0 version.
Please use a proper CUDA version for building this repository.

## Installation

You can easily build this repository by following:
```bash
$ git clone https://github.com/sanghyun-son/pysrwarp
$ cd pysrwarp
$ make
```
If you have trouble building the code, please check ```pysrwarp/cuda/Makefile``` and type your CUDA version as follows:
```bash
NVCC = /usr/local/cuda-${YOUR_CUDA_VERSION}/bin/nvcc
INCFLAGS = -I /usr/local/cuda-${YOUR_CUDA_VERSION}/include
```

## Examples

Once you have installed the repository, you can easily use the warping method in your project:
```python
from torchvision import io
from srwarp import transform
from srwarp import warp

x = io.read_image('image_file.png')
x.unsqueeze_(0)

# Always use torch.DoubleTensor
m = torch.DoubleTensor([
    [1, 0.5, 0],
    [0.4, 1.2, 0],
    [0.0001, 0.0001, 1],
])

# Calculate the output sizes and shift the matrix
# so that the resulting image can be located in regular coordinates.
m, sizes, _ = transform.compensate_matrix(x, m)

# A simple warping process using bicubic interpolation.
# Invalid regions are filled with fill=-255
# mask indicates valid regions where the output image is located.
y, mask_y = warp.warp_by_function(x, m, f_inverse=False, sizes=sizes, fill=-255)

# This MLP is for estimating spatially-varying kernels.
# Please check the paper for more details.
net = some_mlp()
z, mask_z = warp.warp_by_function(
    x,
    m,
    f_inverse=False,
    sizes=sizes,
    fill=-255,
    kernel_type=net,
)

valid = z * mask
valid_mean = valid.sum() / mask.sum()

# Automatic differentiation is supported.
valid_mean.backward()
```

## Citation

If you have found our implementation helpful, please star and cite this repository as follows:

```
@inproceedings{son_2021_srwarp,
    title={{SRW}arp: Generalized Image Super-Resolution under Arbitrary Transformation},
    author={Son, Sanghyun and Lee, Kyoung Mu},
    booktitle={CVPR},
    year={2021}
}
```
