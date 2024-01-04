#include <cstdint>
#include <torch/extension.h>

void paged_attention_v1(
  torch::Tensor& out,
  torch::Tensor& query,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  int num_kv_heads,
  float scale,
  torch::Tensor& block_tables,
  torch::Tensor& context_lens,
  int block_size,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes,
  const bool enable_fp8_kv_cache);

void paged_attention_v2(
  torch::Tensor& out,
  torch::Tensor& exp_sums,
  torch::Tensor& max_logits,
  torch::Tensor& tmp_out,
  torch::Tensor& query,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  int num_kv_heads,
  float scale,
  torch::Tensor& block_tables,
  torch::Tensor& context_lens,
  int block_size,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes,
  const bool enable_fp8_kv_cache);

void rms_norm(torch::Tensor &out,    // [num_tokens, hidden_size]
              torch::Tensor &input,  // [num_tokens, hidden_size]
              torch::Tensor &weight, // [hidden_size]
              float epsilon, bool use_quant);

void invoke_dequant_add_residual_rms_norm_quant(
    torch::Tensor &out,      // [..., hidden_size]
    torch::Tensor &input,    // [..., hidden_size]
    torch::Tensor &residual, // [..., hidden_size]
    torch::Tensor &gamma,    // [hidden_size]
    float scale, float epsilon);

void invoke_dequant_add_residual_rms_norm_quant(
    torch::Tensor &out,      // [..., hidden_size]
    torch::Tensor &input,    // [..., hidden_size]
    torch::Tensor &residual, // [..., hidden_size]
    torch::Tensor &gamma,    // [hidden_size]
    torch::Tensor &scale,    // [num_tokens]
    float epsilon,
    float weight_dequant_scale);

void invoke_add_residual_rms_norm_quant(
    torch::Tensor &out,      // [..., hidden_size]
    torch::Tensor &input,    // [..., hidden_size]
    torch::Tensor &residual, // [..., hidden_size]
    torch::Tensor &gamma,    // [hidden_size]
    float epsilon);

void rotary_embedding(
    torch::Tensor &positions, // [batch_size, seq_len] or [num_tokens]
    torch::Tensor &query,     // [batch_size, seq_len, num_heads * head_size] or [num_tokens, num_heads * head_size]
    torch::Tensor &key,       // [batch_size, seq_len, num_heads * head_size] or [num_tokens, num_heads * head_size]
    int head_size,
    torch::Tensor &cos_sin_cache, // [max_position, rot_dim]
    bool is_neox,
    torch::Tensor &query_out, // [batch_size, seq_len, num_heads * head_size] or [num_tokens, num_heads * head_size]
    torch::Tensor &key_out, // [batch_size, seq_len, num_heads * head_size] or [num_tokens, num_heads * head_size]
    bool use_dequant = false,
    float query_scale = 1.0f,
    float key_scale = 1.0f);

void silu_and_mul(torch::Tensor &out,    // [..., d]
                  torch::Tensor &input); // [..., 2 * d]

void gelu_new(torch::Tensor &out, torch::Tensor &input);

void gelu_fast(torch::Tensor &out, torch::Tensor &input);

void invoke_dequant_silu_and_mul_quant(torch::Tensor &out,   // [..., d]
                                       torch::Tensor &input, // [..., 2 * d]
                                       float gate_scale,
                                       float up_scale,
                                       float out_scale);

void invoke_dequant_silu_and_mul_quant(torch::Tensor &out,   // [..., d]
                                       torch::Tensor &input, // [..., 2 * d]
                                       float gate_scale,
                                       float up_scale,
                                       torch::Tensor &out_scale, // [num_tokens]
                                       torch::Tensor &tmp // [num_tokens, d]
);

// The AWQ kernels are only available on CUDA
#ifndef USE_ROCM
torch::Tensor awq_gemm(
  torch::Tensor _in_feats,
  torch::Tensor _kernel,
  torch::Tensor _scaling_factors,
  torch::Tensor _zeros,
  int split_k_iters);
#endif

void squeezellm_gemm(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table);

torch::Tensor gptq_gemm(
  torch::Tensor a,
  torch::Tensor b_q_weight,
  torch::Tensor b_gptq_qzeros,
  torch::Tensor b_gptq_scales,
  torch::Tensor b_g_idx,
  bool use_exllama,
  int bit);

void gptq_shuffle(
  torch::Tensor q_weight,
  torch::Tensor q_perm,
  int bit);
  