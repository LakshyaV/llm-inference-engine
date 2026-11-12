// int4 weight quantization with group scales.
//
// Each row of a weight matrix is split into groups of 32 consecutive weights.
// A group stores 32 signed 4-bit integers q in [-8, 7] (as nibbles q + 8,
// packed 8 per little-endian uint32) and one fp16 scale s, so w ~= s * q.
// That is 4.5 bits per weight.
#pragma once
#include <cstddef>
#include <cstdint>

constexpr int kQ4Group = 32;

// W is row-major [rows, cols], cols % 32 == 0.
// q receives rows * cols / 8 words, scales rows * cols / 32 half floats (raw bits).
void quantize_q4(const float* W, size_t rows, size_t cols, uint32_t* q, uint16_t* scales);

// Inverse, for tests and error measurement.
void dequantize_q4(const uint32_t* q, const uint16_t* scales, size_t rows, size_t cols, float* W);
