// Token sampling: greedy, temperature, top-k, nucleus (top-p).
#pragma once
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

struct SamplerConfig {
    float temperature = 0.0f;  // 0 = greedy
    int top_k = 0;             // 0 = disabled
    float top_p = 1.0f;
    uint64_t seed = 0;
};

inline int argmax(const float* v, int n) { return (int)(std::max_element(v, v + n) - v); }

class Sampler {
public:
    explicit Sampler(const SamplerConfig& c = {}) : cfg_(c), rng_(c.seed) {}
    bool greedy() const { return cfg_.temperature <= 0; }

    // Full next-token distribution after temperature / top-k / top-p filtering.
    // Speculative decoding needs the distributions, not just a draw.
    void distribution(const float* logits, int n, std::vector<float>& p) const {
        p.assign(n, 0.0f);
        if (greedy()) {
            p[argmax(logits, n)] = 1.0f;
            return;
        }
        const float mx = *std::max_element(logits, logits + n);
        double sum = 0;
        for (int i = 0; i < n; i++) sum += (p[i] = std::exp((logits[i] - mx) / cfg_.temperature));
        for (int i = 0; i < n; i++) p[i] = (float)(p[i] / sum);
        const bool use_k = cfg_.top_k > 0 && cfg_.top_k < n, use_p = cfg_.top_p < 1.0f;
        if (!use_k && !use_p) return;

        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        int keep = use_k ? cfg_.top_k : n;
        // Only the head of the distribution matters; partial sort is enough.
        const int head = std::min(n, use_k ? keep : 4096);
        std::partial_sort(idx.begin(), idx.begin() + head, idx.end(), [&](int a, int b) { return p[a] > p[b]; });
        if (use_p) {
            double cum = 0;
            int k = 0;
            while (k < std::min(keep, head) && cum < cfg_.top_p) cum += p[idx[k++]];
            keep = k;
        }
        std::vector<float> q(n, 0.0f);
        double qs = 0;
        for (int k = 0; k < keep; k++) qs += (q[idx[k]] = p[idx[k]]);
        for (int k = 0; k < keep; k++) q[idx[k]] = (float)(q[idx[k]] / qs);
        p.swap(q);
    }

    int draw(const std::vector<float>& p) {
        double r = std::uniform_real_distribution<double>(0, 1)(rng_), cum = 0;
        int last = 0;
        for (int i = 0; i < (int)p.size(); i++) {
            if (p[i] <= 0) continue;
            last = i;
            cum += p[i];
            if (r < cum) return i;
        }
        return last;
    }

    int sample(const float* logits, int n) {
        if (greedy()) return argmax(logits, n);
        distribution(logits, n, scratch_);
        return draw(scratch_);
    }

    double uniform() { return std::uniform_real_distribution<double>(0, 1)(rng_); }

private:
    SamplerConfig cfg_;
    std::mt19937_64 rng_;
    std::vector<float> scratch_;
};
