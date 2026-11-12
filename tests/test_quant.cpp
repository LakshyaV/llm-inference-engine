// Unit tests for int4 group quantization (no GPU, no model needed).
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "../src/quant.h"

static int failures = 0;
#define CHECK(cond)                                                    \
    do {                                                               \
        if (!(cond)) {                                                 \
            printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);     \
            failures++;                                                \
        }                                                              \
    } while (0)

int main() {
    const size_t rows = 37, cols = 256;
    std::mt19937 rng(7);
    std::normal_distribution<float> gauss(0.0f, 0.02f);
    std::vector<float> w(rows * cols), back(rows * cols);
    for (auto& v : w) v = gauss(rng);
    // One all-zero group and one group with an outlier.
    for (int i = 0; i < kQ4Group; i++) w[i] = 0;
    w[kQ4Group + 3] = 0.5f;

    std::vector<uint32_t> q(rows * cols / 8);
    std::vector<uint16_t> s(rows * cols / kQ4Group);
    quantize_q4(w.data(), rows, cols, q.data(), s.data());
    dequantize_q4(q.data(), s.data(), rows, cols, back.data());

    // Zero groups survive exactly.
    for (int i = 0; i < kQ4Group; i++) CHECK(back[i] == 0.0f);

    // Per-group error is bounded by half a quantization step of that group
    // (plus slack for the fp16 scale), and the overall SNR is what 4 bits buys.
    double sig = 0, err = 0;
    for (size_t g = 0; g < rows * cols / kQ4Group; g++) {
        float amax = 0;
        for (int i = 0; i < kQ4Group; i++) amax = std::fmax(amax, std::fabs(w[g * kQ4Group + i]));
        for (int i = 0; i < kQ4Group; i++) {
            const double d = w[g * kQ4Group + i] - back[g * kQ4Group + i];
            sig += (double)w[g * kQ4Group + i] * w[g * kQ4Group + i];
            err += d * d;
            // Coarsest candidate scale is amax/5, and values may clamp by up to 2 steps at that scale.
            CHECK(std::fabs(d) <= amax / 5.0 * 0.5 * 1.01 + amax * 0.41);
        }
    }
    const double snr_db = 10 * std::log10(sig / err);
    printf("int4 round-trip SNR: %.1f dB\n", snr_db);
    CHECK(snr_db > 17.0);

    // The outlier keeps its sign and most of its magnitude.
    CHECK(back[kQ4Group + 3] > 0.4f);

    printf("quant: %s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
