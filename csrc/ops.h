#pragma once

#include <torch/library.h>

void static_scaled_fp8_quant(torch::Tensor& out, torch::Tensor const& input,
                             torch::Tensor const& scale);

void dynamic_scaled_fp8_quant(torch::Tensor& out, torch::Tensor const& input,
                              torch::Tensor& scale);

void dynamic_per_token_scaled_fp8_quant(
    torch::Tensor& out, torch::Tensor const& input, torch::Tensor& scale,
    c10::optional<torch::Tensor> const& scale_ub);
