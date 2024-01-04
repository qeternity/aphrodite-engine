#include "cache.h"
#include "cuda_utils.h"
#include "ops.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Aphrodite custom ops
  pybind11::module ops = m.def_submodule("ops", "Aphrodite Engine custom operators");

  // Attention ops
  ops.def(
    "paged_attention_v1",
    &paged_attention_v1,
    "Compute the attention between an input query and the cached keys/values using PagedAttention.");
  ops.def(
    "paged_attention_v2",
    &paged_attention_v2,
    "PagedAttention V2.");

  // Activation ops
  ops.def(
    "silu_and_mul",
    &silu_and_mul,
    "Activation function used in SwiGLU.");
  ops.def(
    "gelu_new",
    &gelu_new,
    "GELU implementation used in GPT-2.");
  ops.def(
    "gelu_fast",
    &gelu_fast,
    "Approximate GELU implementation.");
  ops.def(
    "invoke_dequant_silu_and_mul_quant",
    py::overload_cast<torch::Tensor &, torch::Tensor &, float, float, float>(
      &invoke_dequant_silu_and_mul_quant),
      "Dequant input, apply silu act and quant output");
  ops.def("invoke_dequant_silu_and_mul_quant",
        py::overload_cast<torch::Tensor &, torch::Tensor &, float, float,
                          torch::Tensor &, torch::Tensor &>(
            &invoke_dequant_silu_and_mul_quant),
        "Dequant input, apply silu act and quant output");
  ops.def("rotary_embedding", &rotary_embedding, py::arg("positions"),
        py::arg("query"), py::arg("key"), py::arg("head_size"),
        py::arg("cos_sin_cache"), py::arg("is_neox"),
        py::arg("query_out") = torch::empty({}),
        py::arg("key_out") = torch::empty({}), py::arg("use_dequant") = false,
        py::arg("query_scale") = 1.0f, py::arg("key_scale") = 1.0f,
        "Apply GPT-NeoX or GPT-J style rotary embedding to query and key");


  // Layernorm
  ops.def("rms_norm", &rms_norm, py::arg("out"), py::arg("input"),
        py::arg("weight"), py::arg("epsilon"), py::arg("use_quant") = false,
        "Apply Root Mean Square (RMS) Normalization to the input tensor.");

  ops.def("invoke_dequant_add_residual_rms_norm_quant",
        py::overload_cast<torch::Tensor &, torch::Tensor &, torch::Tensor &,
                          torch::Tensor &, float, float>(
            &invoke_dequant_add_residual_rms_norm_quant),
        "Add the dequanted result and residual, then use RMS norm and quant output.");
  ops.def("invoke_dequant_add_residual_rms_norm_quant",
        py::overload_cast<torch::Tensor &, torch::Tensor &, torch::Tensor &,
                          torch::Tensor &, torch::Tensor &, float, float>(
            &invoke_dequant_add_residual_rms_norm_quant),
        "Add the dequanted result and residual, then use RMS norm and quant output.");
  ops.def("invoke_add_residual_rms_norm_quant",
        &invoke_add_residual_rms_norm_quant,
        "Add the result and residual, then use RMS norm and quant output.");

  // Quantization ops
  #ifndef USE_ROCM
  ops.def("gptq_gemm", &gptq_gemm, "Quantized GEMM for GPTQ");
  ops.def("gptq_shuffle", &gptq_shuffle, "Post processing for GPTQ");
  ops.def("awq_gemm", &awq_gemm, "Quantized GEMM for AWQ");
  #endif
  ops.def("squeezellm_gemm", &squeezellm_gemm, "Quantized GEMM for SqueezeLLM");

  // Cache ops
  pybind11::module cache_ops = m.def_submodule("cache_ops", "Aphrodite Engine cache ops");
  cache_ops.def(
    "swap_blocks",
    &swap_blocks,
    "Swap in (out) the cache blocks from src to dst");
  cache_ops.def(
    "copy_blocks",
    &copy_blocks,
    "Copy the cache blocks from src to dst");
  cache_ops.def(
    "reshape_and_cache",
    &reshape_and_cache,
    "Reshape the key and value tensors and cache them");
  cache_ops.def(
    "gather_cached_kv",
    &gather_cached_kv,
    "Gather key and value from the cache into contiguous QKV tensors");
  cache_ops.def(
    "convert_fp8",
    &convert_fp8,
    "Convert the KV cache to FP8 datatype");

  // Cuda utils
  pybind11::module cuda_utils = m.def_submodule("cuda_utils", "Aphrodite Engine cuda utils");
  cuda_utils.def(
    "get_device_attribute",
    &get_device_attribute,
    "Gets the specified device attribute.");
}