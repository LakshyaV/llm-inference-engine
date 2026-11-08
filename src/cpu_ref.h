// Straightforward CPU forward pass (Accelerate BLAS, no KV cache).
// Not a fast path: it exists as an oracle for the Metal kernels and to
// separate "weights/architecture wrong" from "GPU kernel wrong".
#pragma once
#include <string>
#include <vector>

#include "config.h"
#include "safetensors.h"

class CpuReference {
public:
    explicit CpuReference(const std::string& model_dir);
    // Returns logits for every position, row-major [n_tokens, vocab].
    std::vector<float> forward(const std::vector<int>& tokens) const;
    const ModelConfig& config() const { return cfg_; }

private:
    std::vector<float> load(const std::string& name) const;
    ModelConfig cfg_;
    SafeTensors st_;
};
