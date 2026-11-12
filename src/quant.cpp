#include "quant.h"

#include <dispatch/dispatch.h>

#include <cmath>
#include <cstring>

static inline uint16_t f32_to_f16_bits(float f) {
    __fp16 h = (__fp16)f;
    uint16_t b;
    memcpy(&b, &h, 2);
    return b;
}
static inline float f16_bits_to_f32(uint16_t b) {
    __fp16 h;
    memcpy(&h, &b, 2);
    return (float)h;
}

// Pick the scale for one group by searching a few candidates around
// max|w| / 7.5 and keeping the one with the lowest squared error. Negative
// scales are tried too: the integer range is asymmetric ([-8, 7]), so flipping
// the sign lets the largest-magnitude weight use the -8 code.
static void quantize_group(const float* w, uint32_t* q, uint16_t* scale_out) {
    float amax = 0;
    for (int i = 0; i < kQ4Group; i++) amax = std::fmax(amax, std::fabs(w[i]));
    if (amax == 0) {
        for (int i = 0; i < kQ4Group / 8; i++) q[i] = 0x88888888u;
        *scale_out = 0;
        return;
    }
    static const float kDiv[] = {5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f, 8.0f};
    float best_err = INFINITY, best_scale = 0;
    for (float div : kDiv)
        for (int sign = -1; sign <= 1; sign += 2) {
            // Evaluate with the scale as it will actually be stored (fp16).
            const float s = f16_bits_to_f32(f32_to_f16_bits(sign * amax / div));
            if (s == 0) continue;
            const float inv = 1.0f / s;
            float err = 0;
            for (int i = 0; i < kQ4Group; i++) {
                float qi = std::nearbyint(w[i] * inv);
                qi = qi < -8 ? -8 : (qi > 7 ? 7 : qi);
                const float d = w[i] - s * qi;
                err += d * d;
            }
            if (err < best_err) { best_err = err; best_scale = s; }
        }
    const float inv = 1.0f / best_scale;
    for (int word = 0; word < kQ4Group / 8; word++) {
        uint32_t packed = 0;
        for (int i = 0; i < 8; i++) {
            float qi = std::nearbyint(w[word * 8 + i] * inv);
            qi = qi < -8 ? -8 : (qi > 7 ? 7 : qi);
            packed |= (uint32_t)((int)qi + 8) << (4 * i);
        }
        q[word] = packed;
    }
    *scale_out = f32_to_f16_bits(best_scale);
}

void quantize_q4(const float* W, size_t rows, size_t cols, uint32_t* q, uint16_t* scales) {
    const size_t groups = cols / kQ4Group;
    dispatch_apply(rows, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t r) {
        for (size_t g = 0; g < groups; g++)
            quantize_group(W + r * cols + g * kQ4Group, q + (r * groups + g) * (kQ4Group / 8), scales + r * groups + g);
    });
}

void dequantize_q4(const uint32_t* q, const uint16_t* scales, size_t rows, size_t cols, float* W) {
    const size_t groups = cols / kQ4Group;
    for (size_t r = 0; r < rows; r++)
        for (size_t g = 0; g < groups; g++) {
            const float s = f16_bits_to_f32(scales[r * groups + g]);
            for (int i = 0; i < kQ4Group; i++) {
                const uint32_t word = q[(r * groups + g) * (kQ4Group / 8) + i / 8];
                W[r * cols + g * kQ4Group + i] = s * (float)((int)((word >> (4 * (i % 8))) & 0xF) - 8);
            }
        }
}
