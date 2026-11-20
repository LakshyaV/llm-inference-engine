// Statistical test of the speculative-sampling acceptance rule (no GPU, no
// model needed). For toy draft/target distributions, the token that comes out
// of "draw from q, then accept or resample" must be distributed exactly like
// the target p, whatever q is. Also checks the sampler's top-k / top-p filters.
#include <cmath>
#include <cstdio>
#include <vector>

#include "../src/speculative.h"

static int failures = 0;
#define CHECK(cond)                                                    \
    do {                                                               \
        if (!(cond)) {                                                 \
            printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);     \
            failures++;                                                \
        }                                                              \
    } while (0)

// Returns the largest |empirical - p| and reports the acceptance rate.
static double run(const std::vector<float>& p, const std::vector<float>& q, int trials, double& accept_rate) {
    SamplerConfig sc;
    sc.temperature = 1.0f;
    sc.seed = 1234;
    Sampler sampler(sc);
    std::vector<long> hist(p.size(), 0);
    long accepted = 0;
    for (int t = 0; t < trials; t++) {
        const int x = sampler.draw(q);
        int repl = -1;
        if (speculative_accept(p, q, x, sampler, repl)) { hist[x]++; accepted++; }
        else hist[repl]++;
    }
    accept_rate = (double)accepted / trials;
    double worst = 0;
    for (size_t i = 0; i < p.size(); i++) worst = std::max(worst, std::fabs((double)hist[i] / trials - p[i]));
    return worst;
}

int main() {
    const int trials = 400000;
    const double tol = 0.004;  // ~5 sigma for p = 0.5 at this sample size
    double acc;

    // Draft close to target: mostly accepted. Expected rate = sum(min(p, q)).
    std::vector<float> p = {0.50f, 0.20f, 0.15f, 0.10f, 0.05f, 0.00f};
    std::vector<float> q = {0.40f, 0.30f, 0.10f, 0.10f, 0.05f, 0.05f};
    CHECK(run(p, q, trials, acc) < tol);
    CHECK(std::fabs(acc - 0.85) < tol);

    // Draft that puts mass where the target has none, and misses the target's mode.
    q = {0.00f, 0.05f, 0.05f, 0.10f, 0.10f, 0.70f};
    CHECK(run(p, q, trials, acc) < tol);
    CHECK(std::fabs(acc - 0.25) < tol);

    // Identical distributions: everything is accepted.
    CHECK(run(p, p, trials, acc) < tol);
    CHECK(acc == 1.0);

    // Greedy target (one-hot p) against a soft draft.
    std::vector<float> onehot = {0, 0, 1, 0, 0, 0};
    q = {0.2f, 0.2f, 0.2f, 0.2f, 0.1f, 0.1f};
    CHECK(run(onehot, q, trials, acc) == 0.0);

    // --- sampler filters
    const std::vector<float> logits = {2.0f, 1.0f, 0.0f, -1.0f, -2.0f, -30.0f};
    std::vector<float> d;
    SamplerConfig sc;
    Sampler(sc).distribution(logits.data(), 6, d);              // temperature 0 -> one-hot argmax
    CHECK(d[0] == 1.0f && d[1] == 0.0f);

    sc.temperature = 1.0f;
    sc.top_k = 2;
    Sampler(sc).distribution(logits.data(), 6, d);
    CHECK(d[2] == 0 && d[3] == 0 && std::fabs(d[0] + d[1] - 1.0f) < 1e-6f);
    CHECK(std::fabs(d[0] / d[1] - std::exp(1.0f)) < 1e-4f);     // relative odds preserved

    sc.top_k = 0;
    sc.top_p = 0.9f;                                            // softmax = .636 .234 .086 .032 .012 ~0
    Sampler(sc).distribution(logits.data(), 6, d);
    CHECK(d[0] > 0 && d[1] > 0 && d[2] > 0 && d[3] == 0 && d[4] == 0);
    float sum = 0;
    for (float v : d) sum += v;
    CHECK(std::fabs(sum - 1.0f) < 1e-5f);

    printf("speculative: %s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
