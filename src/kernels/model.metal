// ============================================================================
// Transformer kernels: embedding, RMSNorm, linear layers (fp32 and int4),
// RoPE + KV-cache write, fused attention.
//
// Conventions
//   * one row per token:  X[token][feature]
//   * the residual stream, q|k|v and logits are always fp32. The inputs of the
//     linear layers (RMSNorm output, attention output, SwiGLU output) are
//     ACT_T: fp32 for the exact fp32 engine, fp16 for the int4 engine, where
//     halving the activation width is what lets the matrix units run at their
//     fp16 rate
//   * weights are row-major [out][in]; a linear layer is a dot of X rows with
//     W rows, so both are streamed along their contiguous axis
//   * every "one threadgroup = one SIMD-group" kernel is dispatched with 32
//     threads and reduces across the group with simd_sum / simd_max
// ============================================================================

#define SG 32u   // threads per SIMD-group on Apple GPUs

#ifndef ACT_T
#define ACT_T float
#define ACT4 packed_float4
#endif
#define ACT_MAX 65504.0f   // largest finite fp16; SwiGLU outputs are clamped to it

// ------------------------------------------------------------- embedding ---

struct EmbedArgs {
    uint n_tok;
    uint d;
};

kernel void embed_f32(device const float* table [[buffer(0)]],
                      device const int* ids [[buffer(1)]],
                      device float* X [[buffer(2)]],
                      constant EmbedArgs& a [[buffer(3)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= a.d || gid.y >= a.n_tok) return;
    X[gid.y * a.d + gid.x] = table[uint(ids[gid.y]) * a.d + gid.x];
}

// int4 table: one thread expands one uint32 (8 weights) of the selected row.
kernel void embed_q4(device const uint* Wq [[buffer(0)]],
                     device const half* Ws [[buffer(1)]],
                     device const int* ids [[buffer(2)]],
                     device float* X [[buffer(3)]],
                     constant EmbedArgs& a [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]]) {
    const uint U = a.d / 8, G = a.d / 32;
    if (gid.x >= U || gid.y >= a.n_tok) return;
    const uint row = uint(ids[gid.y]);
    const uint w = Wq[row * U + gid.x];
    const float s = float(Ws[row * G + gid.x / 4]);
    device packed_float4* out = (device packed_float4*)(X + gid.y * a.d + gid.x * 8);
    out[0] = (float4((uint4(w) >> uint4(0, 4, 8, 12)) & 0xFu) - 8.0f) * s;
    out[1] = (float4((uint4(w) >> uint4(16, 20, 24, 28)) & 0xFu) - 8.0f) * s;
}

// --------------------------------------------------------------- RMSNorm ---
//
// y = x * rsqrt(mean(x^2) + eps) * w        one threadgroup per token.
// With `gather` set, output row i is the norm of input row rows[i]; the final
// norm uses this to touch only the tokens whose logits were requested.

struct NormArgs {
    uint n_out;
    uint d;
    float eps;
    uint w_off;    // element offset of this layer's gain vector
    uint gather;
};

kernel void rmsnorm(device const float* X [[buffer(0)]],
                    device const float* W [[buffer(1)]],
                    device ACT_T* Y [[buffer(2)]],
                    constant NormArgs& a [[buffer(3)]],
                    device const uint* rows [[buffer(4)]],
                    uint3 tg [[threadgroup_position_in_grid]],
                    ushort tid [[thread_index_in_threadgroup]]) {
    const uint src = a.gather ? rows[tg.x] : tg.x;
    device const packed_float4* x = (device const packed_float4*)(X + src * a.d);
    device const packed_float4* w = (device const packed_float4*)(W + a.w_off);
    device ACT4* y = (device ACT4*)(Y + tg.x * a.d);
    const uint D4 = a.d / 4;
    float ss = 0;
    for (uint i = tid; i < D4; i += SG) {
        const float4 v = x[i];
        ss += dot(v, v);
    }
    ss = simd_sum(ss);
    const float inv = rsqrt(ss / float(a.d) + a.eps);
    for (uint i = tid; i < D4; i += SG) y[i] = ACT4(float4(x[i]) * inv * float4(w[i]));
}

// --------------------------------------------------------- linear layers ---
//
// Y[n][y_col + r] (op)= dot(W[r], X[n])       r < rows, n < n_tok
//
// One SIMD-group computes NR output rows for NT tokens: the 32 threads
// stripe the K axis, keep NR x NT partial sums in registers, and reduce with
// simd_sum. Loading each weight once and using it for NT tokens is what makes
// batched decode and prefill cheaper per token than single-stream decode.
//
//   mode 0  Y  = Wx + b
//   mode 1  Y += Wx                         (residual add fused into the write)
//   mode 2  Y  = silu(W[r] x) * W[r+rows] x (SwiGLU: gate and up rows fused)

struct MMArgs {
    uint n_tok;
    uint rows;
    uint K;
    uint x_stride;
    uint y_stride;
    uint y_col;
    uint mode;
    uint has_bias;
};

static inline void mm_store(device float* Y, device const float* bias, constant MMArgs& a, uint row, uint n, float v) {
    const uint idx = n * a.y_stride + a.y_col + row;
    if (a.has_bias) v += bias[row];
    if (a.mode == 1) Y[idx] += v;
    else if (a.mode == 2) ((device ACT_T*)Y)[idx] = ACT_T(clamp(v, -ACT_MAX, ACT_MAX));
    else Y[idx] = v;
}

// Reduce the per-thread partial sums across the SIMD-group and write results.
// Accumulator slots [NR, 2NR) hold the SwiGLU "up" rows in mode 2.
template <uint NR, uint NT>
static inline void mm_finish(thread float (&acc)[2 * NR][NT], device float* Y, device const float* bias,
                             constant MMArgs& a, uint r0, uint n0, ushort tid) {
    for (uint r = 0; r < NR; r++) {
        for (uint t = 0; t < NT; t++) {
            float v = simd_sum(acc[r][t]);
            if (a.mode == 2) {
                const float up = simd_sum(acc[r + NR][t]);
                v = v / (1.0f + exp(-v)) * up;
            }
            if (tid == 0 && r0 + r < a.rows && n0 + t < a.n_tok) mm_store(Y, bias, a, r0 + r, n0 + t, v);
        }
    }
}

// Weight row behind accumulator slot r.
template <uint NR>
static inline uint mm_row(constant MMArgs& a, uint r0, uint r) {
    const uint last = (a.mode == 2 ? 2 * a.rows : a.rows) - 1;
    return min(r < NR ? r0 + r : a.rows + r0 + r - NR, last);
}

template <uint NR, uint NT>
static inline void mm_f32_impl(device const float* W, device const float* bias, device const ACT_T* X, device float* Y,
                               constant MMArgs& a, uint3 tg, ushort tid) {
    const uint r0 = tg.x * NR, n0 = tg.y * NT;
    const uint K4 = a.K / 4;
    const uint nslots = a.mode == 2 ? 2 * NR : NR;
    float acc[2 * NR][NT];
    for (uint r = 0; r < 2 * NR; r++)
        for (uint t = 0; t < NT; t++) acc[r][t] = 0;

    for (uint k = tid; k < K4; k += SG) {
        float4 x[NT];
        for (uint t = 0; t < NT; t++)
            x[t] = float4(((device const ACT4*)(X + min(n0 + t, a.n_tok - 1) * a.x_stride))[k]);
        for (uint r = 0; r < nslots; r++) {
            const float4 w = ((device const packed_float4*)(W + mm_row<NR>(a, r0, r) * a.K))[k];
            for (uint t = 0; t < NT; t++) acc[r][t] += dot(w, x[t]);
        }
    }
    mm_finish<NR, NT>(acc, Y, bias, a, r0, n0, tid);
}

// int4 weights with group scales: each row is K/8 uint32 (8 nibbles, value =
// nibble - 8) plus one fp16 scale per 32 weights. A thread handles one uint32
// per step, so the SIMD-group consumes 128 contiguous bytes of each row.
template <uint NR, uint NT>
static inline void mm_q4_impl(device const uint* Wq, device const half* Ws, device const float* bias,
                              device const ACT_T* X, device float* Y, constant MMArgs& a, uint3 tg, ushort tid) {
    const uint r0 = tg.x * NR, n0 = tg.y * NT;
    const uint U = a.K / 8, G = a.K / 32;
    const uint nslots = a.mode == 2 ? 2 * NR : NR;
    float acc[2 * NR][NT];
    for (uint r = 0; r < 2 * NR; r++)
        for (uint t = 0; t < NT; t++) acc[r][t] = 0;

    for (uint u = tid; u < U; u += SG) {
        float4 x0[NT], x1[NT];
        for (uint t = 0; t < NT; t++) {
            device const ACT4* p = (device const ACT4*)(X + min(n0 + t, a.n_tok - 1) * a.x_stride + u * 8);
            x0[t] = float4(p[0]);
            x1[t] = float4(p[1]);
        }
        for (uint r = 0; r < nslots; r++) {
            const uint row = mm_row<NR>(a, r0, r);
            const uint w = Wq[row * U + u];
            const float s = float(Ws[row * G + u / 4]);
            const float4 w0 = float4((uint4(w) >> uint4(0, 4, 8, 12)) & 0xFu) - 8.0f;
            const float4 w1 = float4((uint4(w) >> uint4(16, 20, 24, 28)) & 0xFu) - 8.0f;
            for (uint t = 0; t < NT; t++) acc[r][t] += s * (dot(w0, x0[t]) + dot(w1, x1[t]));
        }
    }
    mm_finish<NR, NT>(acc, Y, bias, a, r0, n0, tid);
}

// Instantiate kernels for the (rows per group, tokens per group) tilings the
// engine picks from: mm_{f32,q4}_r<NR>_t<NT>.
#define MM_KERNELS(NR, NT)                                                                                     \
    kernel void mm_f32_r##NR##_t##NT(device const float* W [[buffer(0)]], device const float* bias [[buffer(1)]], \
                             device const ACT_T* X [[buffer(2)]], device float* Y [[buffer(3)]],              \
                             constant MMArgs& a [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],     \
                             ushort tid [[thread_index_in_threadgroup]]) {                                    \
        mm_f32_impl<NR, NT>(W, bias, X, Y, a, tg, tid);                                                        \
    }                                                                                                          \
    kernel void mm_q4_r##NR##_t##NT(device const uint* Wq [[buffer(0)]], device const float* bias [[buffer(1)]], \
                            device const ACT_T* X [[buffer(2)]], device float* Y [[buffer(3)]],               \
                            constant MMArgs& a [[buffer(4)]], device const half* Ws [[buffer(5)]],            \
                            uint3 tg [[threadgroup_position_in_grid]],                                        \
                            ushort tid [[thread_index_in_threadgroup]]) {                                     \
        mm_q4_impl<NR, NT>(Wq, Ws, bias, X, Y, a, tg, tid);                                                    \
    }

MM_KERNELS(1, 1)
MM_KERNELS(2, 1)
MM_KERNELS(4, 1)
MM_KERNELS(8, 1)
MM_KERNELS(1, 4)
MM_KERNELS(2, 4)
MM_KERNELS(4, 4)

// ------------------------------------------------- RoPE + KV-cache write ---
//
// Rotates q in place and writes rotated k and raw v into the paged KV cache.
// One thread per (rotation pair, head, token); heads [0, n_heads) are query
// heads, [n_heads, n_heads + n_kv) are the key/value heads.
//
// Rotation matches the Hugging Face "rotate_half" convention: element i pairs
// with element i + head_dim/2.

struct TokenMeta {
    uint seq;   // slot in the page table
    uint pos;   // absolute position in its sequence
};

struct KVArgs {
    uint n_tok;
    uint n_heads;
    uint n_kv;
    uint head_dim;
    uint qkv_stride;     // floats per token in the fused q|k|v buffer
    uint layer;
    uint n_pages;        // physical pages per layer
    uint page_shift;     // log2(tokens per page)
    uint pt_stride;      // page-table entries per sequence
    float scale;         // 1/sqrt(head_dim)
    uint d;
};

static inline uint kv_slot(constant KVArgs& a, device const uint* page_table, uint seq, uint pos) {
    const uint page = page_table[seq * a.pt_stride + (pos >> a.page_shift)];
    const uint slot = pos & ((1u << a.page_shift) - 1u);
    return ((((a.layer * a.n_pages + page) << a.page_shift) + slot) * a.n_kv) * a.head_dim;
}

kernel void rope_kv_write(device float* QKV [[buffer(0)]],
                          device const float* rope [[buffer(1)]],       // [pos][head_dim/2] (cos, sin)
                          device float* Kc [[buffer(2)]],
                          device float* Vc [[buffer(3)]],
                          device const TokenMeta* meta [[buffer(4)]],
                          device const uint* page_table [[buffer(5)]],
                          constant KVArgs& a [[buffer(6)]],
                          uint3 gid [[thread_position_in_grid]]) {
    const uint half_dim = a.head_dim / 2;
    const uint i = gid.x, h = gid.y, n = gid.z;
    if (i >= half_dim || h >= a.n_heads + a.n_kv || n >= a.n_tok) return;
    const TokenMeta m = meta[n];
    const float2 cs = ((device const packed_float2*)rope)[m.pos * half_dim + i];
    device float* row = QKV + n * a.qkv_stride;

    if (h < a.n_heads) {
        device float* q = row + h * a.head_dim;
        const float x = q[i], y = q[i + half_dim];
        q[i] = x * cs.x - y * cs.y;
        q[i + half_dim] = y * cs.x + x * cs.y;
    } else {
        const uint kh = h - a.n_heads;
        device const float* k = row + a.n_heads * a.head_dim + kh * a.head_dim;
        device const float* v = k + a.n_kv * a.head_dim;
        const uint base = kv_slot(a, page_table, m.seq, m.pos) + kh * a.head_dim;
        const float x = k[i], y = k[i + half_dim];
        Kc[base + i] = x * cs.x - y * cs.y;
        Kc[base + i + half_dim] = y * cs.x + x * cs.y;
        Vc[base + i] = v[i];
        Vc[base + i + half_dim] = v[i + half_dim];
    }
}

