from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='roi_align_2d_cuda',
    ext_modules=[
        CUDAExtension('roi_align_2d_cuda', [
            'src/roi_align_2d_cuda.cpp',
            'src/roi_align_2d_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension}
)

setup(
    name='roi_align_3d_cuda',
    ext_modules=[
        CUDAExtension('roi_align_3d_cuda', [
            'src/roi_align_3d_cuda.cpp',
            'src/roi_align_3d_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension}
)
