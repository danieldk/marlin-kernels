#include <torch/all.h>

#include "core/scalar_type.hpp"
#include "core/registration.h"

#include "ops.h"

// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // ScalarType, a custom class for representing data types that supports
  // quantized types, declared here so it can be used when creating interfaces
  // for custom ops.
  marlin_kernels::ScalarTypeTorch::bind_class(ops);

#ifndef USE_ROCM

  // Note about marlin kernel 'workspace' arguments:
  // Technically these should be mutable since they are modified by the kernel.
  // But since they are set back to zero once the kernel is finished we can
  // hand wave and say that they have no net effect.
  //
  // The reason to mark 'workspace' as immutable is so that they don't interfere
  // with using ScalarType arguments in the ops. If they are marked as mutable,
  // pytorch throws an assert in
  // 'torch._higher_order_ops._register_effectful_op' that prevents these
  // kernels from being torch.compile'd.
  // See the following document for more info on custom types and ops that use
  // custom types:
  // https://docs.google.com/document/d/18fBMPuOJ0fY5ZQ6YyrHUppw9FA332CpNtgB6SOIgyuA

  // Marlin (Dense) Optimized Quantized GEMM for GPTQ.
  ops.def(
      "marlin_gemm(Tensor a, Tensor b_q_weight, Tensor b_scales, "
      "Tensor! workspace, int size_m, int size_n, int size_k) -> Tensor");
  // conditionally compiled so impl in source file

  // Marlin_24 (Sparse) Optimized Quantized GEMM for GPTQ.
  ops.def(
      "gptq_marlin_24_gemm(Tensor a, Tensor b_q_weight, Tensor b_meta, "
      "Tensor b_scales, Tensor workspace, "
      "__torch__.torch.classes._marlin_kernels.ScalarType b_q_type, "
      "int size_m, int size_n, int size_k) -> Tensor");
  //  conditionally compiled so impl in source file

  // Machete (Dense) Optimized Mixed Precision GEMM for Hopper.
  ops.def(
      "machete_supported_schedules("
      "   __torch__.torch.classes._marlin_kernels.ScalarType btype"
      ") -> str[]");
  ops.def(
      "machete_gemm(Tensor A, Tensor B,"
      "             __torch__.torch.classes._marlin_kernels.ScalarType btype,"
      "             Tensor? scales, Tensor? zeros, int? group_size,"
      "             Tensor? C, float? alpha, float? beta, str? schedule)"
      "-> Tensor");
  ops.def(
      "machete_prepack_B(Tensor B,"
      "                  __torch__.torch.classes._marlin_kernels.ScalarType btype)"
      "-> Tensor");
  // conditionally compiled so impl registration is in source file

  // gptq_marlin Optimized Quantized GEMM for GPTQ.
  ops.def(
      "gptq_marlin_gemm(Tensor a, Tensor b_q_weight, Tensor b_scales, "
      "Tensor b_zeros, Tensor g_idx, Tensor perm, Tensor workspace, "
      "__torch__.torch.classes._marlin_kernels.ScalarType b_q_type, "
      "int size_m, int size_n, int size_k, bool is_k_full, "
      "bool has_zp, bool use_fp32_reduce) -> Tensor");
  // conditionally compiled so impl registration is in source file

  // gptq_marlin repack from GPTQ.
  ops.def(
      "gptq_marlin_repack(Tensor b_q_weight, Tensor perm, "
      "SymInt size_k, SymInt size_n, int num_bits) -> Tensor");
  // conditionally compiled so impl registrations are in source file

  // awq_marlin repack from AWQ.
  ops.def(
      "awq_marlin_repack(Tensor b_q_weight, SymInt size_k, "
      "SymInt size_n, int num_bits) -> Tensor");
  // conditionally compiled so impl registrations are in source file

  // fp8_marlin Optimized Quantized GEMM for FP8 weight-only.
  ops.def(
      "fp8_marlin_gemm(Tensor a, Tensor b_q_weight, Tensor b_scales, "
      "Tensor! workspace, int num_bits, int size_m, int size_n, "
      "int size_k) -> Tensor");
  // conditionally compiled so impl registration is in source file

  // marlin_qqq_gemm for QQQ.
  ops.def(
      "marlin_qqq_gemm(Tensor a, Tensor b_q_weight, "
      "Tensor s_tok, Tensor s_ch, Tensor s_group, "
      "Tensor! workspace, int size_m, int size_n, "
      "int size_k) -> Tensor");
  // conditionally compiled so impl registration is in source file

  // CUTLASS w8a8 GEMM, supporting symmetric per-tensor or per-row/column
  // quantization, as well as bias
  ops.def(
      "cutlass_scaled_mm(Tensor! out, Tensor a,"
      "                  Tensor b, Tensor a_scales,"
      "                  Tensor b_scales, Tensor? bias) -> ()");
  ops.impl("cutlass_scaled_mm", torch::kCUDA, &cutlass_scaled_mm);

  // CUTLASS w8a8 GEMM, supporting asymmetric per-tensor or per-row/column
  // quantization.
  ops.def(
      "cutlass_scaled_mm_azp(Tensor! out, Tensor a,"
      "                  Tensor b, Tensor a_scales,"
      "                  Tensor b_scales, Tensor azp_adj,"
      "                  Tensor? azp, Tensor? bias) -> ()");
  ops.impl("cutlass_scaled_mm_azp", torch::kCUDA, &cutlass_scaled_mm_azp);

  // Check if cutlass scaled_mm is supported for CUDA devices of the given
  // capability
  ops.def("cutlass_scaled_mm_supports_fp8(int cuda_device_capability) -> bool");
  ops.impl("cutlass_scaled_mm_supports_fp8", &cutlass_scaled_mm_supports_fp8);

#endif

  // Compute FP8 quantized tensor for given scaling factor.
  ops.def(
      "static_scaled_fp8_quant(Tensor! out, Tensor input, Tensor scale) -> ()");
  ops.impl("static_scaled_fp8_quant", torch::kCUDA, &static_scaled_fp8_quant);

  // Compute dynamic-per-tensor FP8 quantized tensor and scaling factor.
  ops.def(
      "dynamic_scaled_fp8_quant(Tensor! out, Tensor input, Tensor! scale) -> "
      "()");
  ops.impl("dynamic_scaled_fp8_quant", torch::kCUDA, &dynamic_scaled_fp8_quant);

  // Compute dynamic-per-token FP8 quantized tensor and scaling factor.
  ops.def(
      "dynamic_per_token_scaled_fp8_quant(Tensor! out, Tensor input, "
      "Tensor! scale, Tensor? scale_ub) -> "
      "()");
  ops.impl("dynamic_per_token_scaled_fp8_quant", torch::kCUDA,
           &dynamic_per_token_scaled_fp8_quant);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
