// Speculative decoding (Leviathan et al. 2023, Chen et al. 2023).
//
// A small draft model proposes `gamma` tokens autoregressively; the target
// model scores all of them in ONE batched pass. Proposals are accepted left
// to right with probability min(1, p_target / p_draft); the first rejection
// is replaced by a sample from the residual distribution max(0, p - q), and
// when everything is accepted the target's extra row yields a bonus token.
// The output distribution is exactly the target's; with greedy sampling the
// output is the target's greedy continuation.
//
// Both engines keep their own paged KV cache; rejected tokens are rolled back
// with seq_truncate().
#pragma once
#include <functional>
#include <vector>

#include "engine.h"
#include "sampler.h"

struct SpecStats {
    int generated = 0;      // tokens emitted
    int drafted = 0;        // tokens proposed by the draft model
    int accepted = 0;       // proposals accepted by the target
    int target_passes = 0;  // target forward passes (vs `generated` for plain decoding)
    double acceptance() const { return drafted ? (double)accepted / drafted : 0; }
};

// One verification step of speculative sampling. `x` was drawn from the draft
// distribution q; p is the target distribution for the same position.
// Returns true if x is accepted. Otherwise `replacement` is drawn from the
// residual norm(max(0, p - q)), which makes the emitted token exactly
// p-distributed (tests/test_speculative.cpp checks this empirically).
bool speculative_accept(const std::vector<float>& p, const std::vector<float>& q, int x, Sampler& sampler,
                        int& replacement);

class SpeculativeDecoder {
public:
    SpeculativeDecoder(Engine& target, Engine& draft, int gamma, const SamplerConfig& sampling = {});

    // Generates up to max_tokens after `prompt`; stops at a target EOS token.
    std::vector<int> generate(const std::vector<int>& prompt, int max_tokens,
                              const std::function<void(int)>& on_token = nullptr);
    const SpecStats& stats() const { return stats_; }

private:
    Engine& target_;
    Engine& draft_;
    int gamma_;
    Sampler sampler_;
    SpecStats stats_;
};
