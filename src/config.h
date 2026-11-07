// Model hyper-parameters read from a Hugging Face config.json
// (Qwen2 / Llama style decoder: RMSNorm, RoPE, GQA attention, SwiGLU MLP).
#pragma once
#include <string>
#include <vector>

struct ModelConfig {
    int vocab_size = 0;
    int hidden = 0;        // d_model
    int inter = 0;         // SwiGLU inner width
    int n_layers = 0;
    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    float rms_eps = 1e-6f;
    float rope_theta = 10000.0f;
    bool tie_embeddings = false;
    std::vector<int> eos_ids;

    int kv_dim() const { return n_kv_heads * head_dim; }

    static ModelConfig load(const std::string& model_dir);
};
