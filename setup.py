from os import path
import setuptools

from torch.utils import cpp_extension

def main() -> None:
    # Spatially-Varying Filtering
    name = 'pysrwarp'
    name_cuda = 'svf_cuda'
    target_dir = 'cuda'
    setuptools.setup(
        name=name,
        version='1.0.0',
        author='Sanghyun Son',
        author_email='sonsang35@gmail.com',
        packages=setuptools.find_packages(),
        ext_modules=[cpp_extension.CppExtension(
            name='srwarp_cuda',
            sources=[path.join('cuda', name_cuda + '.cpp')],
            libraries=[
                name_cuda + '_kernel',
                name_cuda + '_half_kernel',
                name_cuda + '_projective_grid_kernel',
            ],
            library_dirs=[path.join('.', target_dir)],
            extra_compile_args=['-g', '-fPIC'],
        )],
        cmdclass={'build_ext': cpp_extension.BuildExtension},
    )
    return

if __name__ == '__main__':
    main()
