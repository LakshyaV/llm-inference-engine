// The Metal inference engine: weights on the GPU, a paged KV cache, and a
// single entry point (`forward`) that advances any mix of sequences by any
// number of tokens in one GPU pass. Single-stream decode, prefill, continuous
// batching and speculative verification are all just different token lists.
#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "config.h"

enum class WeightFormat {
    F32,  // exact weights; used to validate logits against the reference
    Q4,   // int4 with per-group fp16 scales (4.5 bits/weight); the fast path
};

struct EngineOptions {
    WeightFormat format = WeightFormat::Q4;
    int max_seqs = 16;           // concurrent sequences
    int page_tokens = 16;        // tokens per KV page
    int n_pages = 512;           // KV pool size (n_pages * page_tokens token slots)
    int max_batch_tokens = 1024; // tokens per forward() call
    int max_logit_rows = 64;     // logit rows returned per forward() call
    int max_positions = 8192;    // longest sequence (RoPE table size)
    int tensor_min_tokens = 16;  // passes with at least this many tokens use the matrix-unit kernels
    bool verbose = false;
    bool profile = false;        // time each kernel class separately (slower)
};

struct TokenIn {
    int token;
    int seq;
};

class Engine {
public:
    Engine(const std::string& model_dir, const EngineOptions& opts = {});
    ~Engine();

    const ModelConfig& config() const;
    const EngineOptions& options() const;

    // --- sequences (each owns a list of KV pages) ---
    int seq_create();                    // -1 if all slots are busy
    void seq_free(int seq);
    int seq_len(int seq) const;
    void seq_truncate(int seq, int len); // roll back, e.g. rejected speculative tokens
    int free_pages() const;
    int free_token_slots(int seq) const; // tokens `seq` can still append

    // Append `toks` to their sequences (tokens of one sequence in order) and
    // run the model. `want[i]` indexes into `toks`; row i of the result holds
    // the logits after that token. The pointer is valid until the next call.
    const float* forward(const std::vector<TokenIn>& toks, const std::vector<int>& want);

    double last_gpu_seconds() const;     // GPU execution time of the last forward()
    size_t weight_bytes() const;
    std::map<std::string, double>& stage_seconds();  // accumulated per-stage GPU time (profile mode)

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
