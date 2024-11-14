from typing import List, Optional, Tuple, Union
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


# cutlass
def cutlass_scaled_mm_supports_fp8(cuda_device_capability: int) -> bool:
    return torch.ops._marlin_kernels.cutlass_scaled_mm_supports_fp8(
        cuda_device_capability
    )


def cutlass_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
    assert bias is None or bias.shape[0] == b.shape[1] and bias.dtype == out_dtype

    m = a.shape[0]
    n = b.shape[1]
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)

    torch.ops._marlin_kernels.cutlass_scaled_mm(out, a, b, scale_a, scale_b, bias)

    return out


def cutlass_scaled_mm_azp(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    azp_adj: torch.Tensor,
    azp: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
    assert bias is None or bias.numel() == b.shape[1] and bias.dtype == out_dtype

    m = a.shape[0]
    n = b.shape[1]
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)

    torch.ops._marlin_kernels.cutlass_scaled_mm_azp(
        out, a, b, scale_a, scale_b, azp_adj, azp, bias
    )
    return out


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


def scaled_fp8_quant(
    input: torch.Tensor,
    dtype: torch.dtype,
    scale: Optional[torch.Tensor] = None,
    num_token_padding: Optional[int] = None,
    scale_ub: Optional[torch.Tensor] = None,
    use_per_token_if_dynamic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP8 and return quantized tensor and scale.

    This function supports both static and dynamic quantization: If you
    provide the scale, it will use static scaling and if you omit it,
    the scale will be determined dynamically. The function also allows
    optional padding of the output tensors for downstream kernels that
    will benefit from padding.

    Args:
        input: The input tensor to be quantized to FP8
        dtype: quantized data type
        scale: Optional scaling factor for the FP8 quantization
        scale_ub: Optional upper bound for scaling factor in dynamic
            per token case
        num_token_padding: If specified, pad the first dimension
            of the output to at least this value.
        use_per_token_if_dynamic: Whether to do per_tensor or per_token
            in the dynamic quantization case.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The output tensor in FP8 and
            scaling factor.
    """
    # This code assumes batch_dim and num_tokens are flattened
    assert input.ndim == 2
    shape: Union[Tuple[int, int], torch.Size] = input.shape
    if num_token_padding:
        shape = (max(num_token_padding, input.shape[0]), shape[1])
    output = torch.empty(shape, device=input.device, dtype=dtype)

    if scale is None:
        if use_per_token_if_dynamic:
            scale = torch.empty((shape[0], 1), device=input.device, dtype=torch.float32)
            torch.ops._marlin_kernels.dynamic_per_token_scaled_fp8_quant(
                output, input, scale, scale_ub
            )
        else:
            scale = torch.zeros(1, device=input.device, dtype=torch.float32)
            torch.ops._marlin_kernels.dynamic_scaled_fp8_quant(output, input, scale)
    else:
        # num_token_padding not implemented for this case
        assert scale.numel() == 1 or num_token_padding is None
        torch.ops._marlin_kernels.static_scaled_fp8_quant(output, input, scale)

    return output, scale


def scaled_int8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    azp: Optional[torch.Tensor] = None,
    symmetric: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Quantize the input tensor to int8 and return the quantized tensor and scale, and maybe azp.

    Args:
        input: The input tensor to be quantized to int8.
        scale: Optional scaling factor for the int8 quantization.
            When not provided, we invoke dynamic-per-token quantization.
        azp: Optional zero-point for the int8 quantization.
            Must be provided for asymmetric quantization if `scale` is provided.
        symmetric: Whether to use symmetric quantization (scale only, azp ignored).

    Returns:
      Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]] : Output int8 tensor, scales, and optionally azp.
    """
    output = torch.empty_like(input, dtype=torch.int8)
    if scale is not None:
        # static-per-tensor quantization.
        assert symmetric == (
            azp is
            None), "azp must only be provided for asymmetric quantization."
        torch.ops._marlin_kernels.static_scaled_int8_quant(output, input, scale, azp)
        return output, scale, None

    # dynamic-per-token quantization.
    input_scales = torch.empty((input.numel() // input.shape[-1], 1),
                               device=input.device,
                               dtype=torch.float32)
    input_azp = None if symmetric else torch.empty_like(input_scales,
                                                        dtype=torch.int32)
    torch.ops._marlin_kernels.dynamic_scaled_int8_quant(output, input, input_scales,
                                           input_azp)
    return output, input_scales, input_azp

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
    "scaled_fp8_quant",
]
