import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = os.getenv("NVCC_THREADS") or "4"
    return nvcc_extra_args + ["--threads", nvcc_threads]


SKIP_CUDA_BUILD = os.getenv("MARLIN_SKIP_CUDA_BUILD", "FALSE") == "TRUE"

ext_modules = []
cc_flag = []

if not SKIP_CUDA_BUILD:
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_80,code=sm_80")
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_90,code=sm_90")

    extra_compile_args = {
        "nvcc": append_nvcc_threads(
            [
                "-O3",
                "-std=c++17",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
            ]
            + cc_flag
        )
    }

    ext_modules.append(
        CUDAExtension(
            name="marlin_kernels",
            sources=[
                "marlin_kernels/fp8/fp8_marlin.cu",
                "marlin_kernels/gptq_marlin/awq_marlin_repack.cu",
                "marlin_kernels/gptq_marlin/gptq_marlin.cu",
                "marlin_kernels/gptq_marlin/gptq_marlin_repack.cu",
                "marlin_kernels/marlin/dense/marlin_cuda_kernel.cu",
                "marlin_kernels/marlin/sparse/marlin_24_cuda_kernel.cu",
                "marlin_kernels/ext.cpp",
            ],
            extra_compile_args=extra_compile_args,
        )
    ),

setup(
    name="marlin_kernels",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension} if ext_modules else {},
)
