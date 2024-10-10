#include <torch/extension.h>

#include "ext.h"
#include "core/registration.h"
#include "core/scalar_type.hpp"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  // ScalarType, a custom class for representing data types that supports
  // quantized types, declared here so it can be used when creating interfaces
  // for custom ops.
  marlin_kernels::ScalarTypeTorch::bind_class(m);

#ifndef USE_ROCM

  m.def(
      "awq_marlin_repack(Tensor b_q_weight, SymInt size_k, "
      "SymInt size_n, int num_bits) -> Tensor");
  m.def(
    "gptq_marlin_gemm(Tensor a, Tensor b_q_weight, Tensor b_scales, "
    "Tensor b_zeros, Tensor g_idx, Tensor perm, Tensor workspace, "
    "__torch__.torch.classes._marlin_kernels_ops.ScalarType b_q_type, "
    "int size_m, int size_n, int size_k, bool is_k_full, "
    "bool has_zp, bool use_fp32_reduce) -> Tensor");
  m.def(
    "gptq_marlin_24_gemm(Tensor a, Tensor b_q_weight, Tensor b_meta, "
    "Tensor b_scales, Tensor workspace, "
    "__torch__.torch.classes._marlin_kernels_ops.ScalarType b_q_type, "
    "int size_m, int size_n, int size_k) -> Tensor");
  m.def(
    "gptq_marlin_repack(Tensor b_q_weight, Tensor perm, "
    "SymInt size_k, SymInt size_n, int num_bits) -> Tensor");
  m.def(
    "marlin_gemm(Tensor a, Tensor b_q_weight, Tensor b_scales, "
    "Tensor! workspace, int size_m, int size_n, int size_k) -> Tensor");
  m.def(
    "fp8_marlin_gemm(Tensor a, Tensor b_q_weight, Tensor b_scales, "
    "Tensor! workspace, int num_bits, int size_m, int size_n, "
    "int size_k) -> Tensor");
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
