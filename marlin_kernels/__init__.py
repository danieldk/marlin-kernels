from typing import List, Optional
import torch

# Insure that the ops are registered.
import marlin_kernels._marlin_kernels

from ._core_ext import ScalarType
from .scalar_type import scalar_types


def get_scalar_type(num_bits: int, has_zp: bool):
    if has_zp:
        assert num_bits == 4
        return scalar_types.uint4
    else:
        return scalar_types.uint4b8 if num_bits == 4 else scalar_types.uint8b128


def awq_marlin_repack(
    b_q_weight: torch.Tensor, size_k: int, size_n: int, num_bits: int
) -> torch.Tensor:
    """Repack AWQ parameters for GPTQ-Marlin."""
    return torch.ops._marlin_kernels.awq_marlin_repack(
        b_q_weight, size_k, size_n, num_bits
    )


def machete_supported_schedules(btype: ScalarType) -> List[str]:
    """Get the supported Machete schedules for the given scalar type."""
    return torch.ops._marlin_kernels.machete_supported_schedules(btype)


def machete_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    num_bits: int,
    scales: torch.Tensor,
    zeros: Optional[torch.Tensor] = None,
    group_size: Optional[int] = None,
    C: Optional[torch.Tensor] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    schedule: Optional[str] = None,
) -> torch.Tensor:
    """
    Matrix multiplication using Machette kernels (requires Hopper or later).
    """
    scalar_type = get_scalar_type(num_bits, zeros is not None)
    return torch.ops._marlin_kernels.machete_gemm(
        A, B, scalar_type, scales, zeros, group_size, C, alpha, beta, schedule
    )


def machete_prepack_B(B: torch.Tensor, num_bits: int, has_zp: bool) -> torch.Tensor:
    """Repack quantized weights for Machete kernels."""
    scalar_type = get_scalar_type(num_bits, has_zp)
    return torch.ops._marlin_kernels.machete_prepack_B(B, scalar_type)


def gptq_marlin_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_scales: torch.Tensor,
    b_zeros: torch.Tensor,
    g_idx: torch.Tensor,
    perm: torch.Tensor,
    workspace: torch.Tensor,
    num_bits: int,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool,
    has_zp: bool,
    use_fp32_reduce: bool,
) -> torch.Tensor:
    """
    Matrix multiplication using Marlin kernels. This is an extension of
    `marlin_gemm` that supports converted GPTQ kernels.
    """
    scalar_type = get_scalar_type(num_bits, has_zp)
    return torch.ops._marlin_kernels.gptq_marlin_gemm(
        a,
        b_q_weight,
        b_scales,
        b_zeros,
        g_idx,
        perm,
        workspace,
        scalar_type,
        size_m,
        size_n,
        size_k,
        is_k_full,
        has_zp,
        use_fp32_reduce,
    )


def gptq_marlin_24_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_meta: torch.Tensor,
    b_scales: torch.Tensor,
    workspace: torch.Tensor,
    num_bits: int,
    size_m: int,
    size_n: int,
    size_k: int,
) -> torch.Tensor:
    """
    Matrix multiplication using Marlin kernels. This is an extension of
    `marlin_gemm` that supports 2:4 sparsity.
    """
    scalar_type = get_scalar_type(num_bits, False)
    return torch.ops._marlin_kernels.gptq_marlin_24_gemm(
        a, b_q_weight, b_meta, b_scales, workspace, scalar_type, size_m, size_n, size_k
    )


def gptq_marlin_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> torch.Tensor:
    """Repack GPTQ parameters for Marlin kernels."""
    return torch.ops._marlin_kernels.gptq_marlin_repack(
        b_q_weight, perm, size_k, size_n, num_bits
    )


def marlin_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_scales: torch.Tensor,
    workspace: torch.Tensor,
    size_m: int,
    size_n: int,
    size_k: int,
) -> torch.Tensor:
    """
    Matrix multiplication using Marlin kernels.
    """
    return torch.ops._marlin_kernels.marlin_gemm(
        a, b_q_weight, b_scales, workspace, size_m, size_n, size_k
    )


# fp8 marlin
def fp8_marlin_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_scales: torch.Tensor,
    workspace: torch.Tensor,
    num_bits: int,
    size_m: int,
    size_n: int,
    size_k: int,
) -> torch.Tensor:
    return torch.ops._marlin_kernels.fp8_marlin_gemm(
        a, b_q_weight, b_scales, workspace, num_bits, size_m, size_n, size_k
    )


__all__ = [
    "ScalarType",
    "scalar_types",
    "awq_marlin_repack",
    "machete_supported_schedules",
    "machete_gemm",
    "machete_prepack_B",
    "gptq_marlin_gemm",
    "gptq_marlin_24_gemm",
    "gptq_marlin_repack",
    "marlin_gemm",
    "fp8_marlin_gemm",
]
