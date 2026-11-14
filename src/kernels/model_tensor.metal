// ============================================================================
// Linear layers on the GPU's matrix units (prefill / batched passes).
//
// Same contract as the mm_* kernels in model.metal, but a threadgroup of
// TT_NSG SIMD-groups computes a TT_TOK x TT_ROWS tile of the output with
// Metal Performance Primitives' matmul2d, the shader-level interface to the
// matrix hardware in M5-class GPUs. The units want a full 32x32 output tile
// per SIMD-group (see bench/gemm_bench), hence 64x64 on four groups.
//
// The result is accumulated in a *cooperative tensor*: thread-private
// registers spread over the group. That lets bias, the residual add and the
// SwiGLU activation be applied in registers before a single store, so the
// fused modes cost no extra pass over memory.
//
// int4 weights are expanded one scale-group (32 weights along K) at a time
// into threadgroup memory as fp16 and multiplied from there, so K is walked
// in steps of 32 and each step is one matrix-unit multiply-accumulate.
// ============================================================================

#define TT_TOK 64     // tokens per tile        (matmul "m")
#define TT_ROWS 64    // output rows per tile   (matmul "n")
#define TT_NSG 4      // SIMD-groups per threadgroup
#define TT_K 64       // K step for int4 (a multiple of the 32-wide quantization group)

typedef tensor<device ACT_T, dextents<int, 2>, tensor_inline> act_tensor;
typedef tensor<device float, dextents<int, 2>, tensor_inline> f32_tensor;
typedef tensor<threadgroup half, dextents<int, 2>, tensor_inline> tg_tensor;

template <class CT>
static inline void tt_zero(thread CT& c) {
    for (ushort i = 0; i < c.get_capacity(); ++i)
        if (c.is_valid_element(i)) c[i] = 0;
}

// Expand one packed word (8 weights) of an int4 row into fp16.
static inline void q4_expand(device const uint* Wq, device const half* Ws, uint word_index, uint scale_index,
                             threadgroup half* dst) {
    const uint w = Wq[word_index];
    const float s = float(Ws[scale_index]);
    const float4 lo = (float4((uint4(w) >> uint4(0, 4, 8, 12)) & 0xFu) - 8.0f) * s;
    const float4 hi = (float4((uint4(w) >> uint4(16, 20, 24, 28)) & 0xFu) - 8.0f) * s;
    dst[0] = lo.x; dst[1] = lo.y; dst[2] = lo.z; dst[3] = lo.w;
    dst[4] = hi.x; dst[5] = hi.y; dst[6] = hi.z; dst[7] = hi.w;
}

// ---- modes 0 and 1:  Y (fp32) = or += W x (+ bias) -------------------------

kernel void mm_q4_tensor(device const uint* Wq [[buffer(0)]],
                         device const float* bias [[buffer(1)]],
                         device ACT_T* X [[buffer(2)]],
                         device float* Y [[buffer(3)]],
                         constant MMArgs& a [[buffer(4)]],
                         device const half* Ws [[buffer(5)]],
                         uint3 tg [[threadgroup_position_in_grid]],
                         ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup half sw[TT_ROWS * TT_K];

    constexpr auto desc = matmul2d_descriptor(TT_TOK, TT_ROWS, TT_K, false, true, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<TT_NSG>> op;

    const uint r0 = tg.x * TT_ROWS, n0 = tg.y * TT_TOK;
    const uint U = a.K / 8, G = a.K / 32;

    act_tensor tX(X, dextents<int, 2>(a.K, a.n_tok), array<int, 2>{1, int(a.x_stride)});
    f32_tensor tY(Y + a.y_col, dextents<int, 2>(a.rows, a.n_tok), array<int, 2>{1, int(a.y_stride)});
    tg_tensor tW(sw, dextents<int, 2>(TT_K, TT_ROWS));
    auto mY = tY.slice(int(r0), int(n0));
    auto mX0 = tX.slice(0, int(n0));

    auto c = op.get_destination_cooperative_tensor<decltype(mX0), decltype(tW), float>();
    if (a.mode == 1) c.load(mY);   // residual: start from the current value of Y
    else tt_zero(c);

    const uint words = TT_ROWS * (TT_K / 8);
    for (uint k0 = 0; k0 < a.K; k0 += TT_K) {
        for (uint e = tid; e < words; e += TT_NSG * SG) {
            const uint r = e / (TT_K / 8), word = e % (TT_K / 8);
            const uint row = min(r0 + r, a.rows - 1);
            q4_expand(Wq, Ws, row * U + k0 / 8 + word, row * G + (k0 + word * 8) / 32, sw + r * TT_K + word * 8);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mX = tX.slice(int(k0), int(n0));
        op.run(mX, tW, c);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (a.has_bias)
        for (ushort i = 0; i < c.get_capacity(); ++i)
            if (c.is_valid_element(i)) c[i] += bias[min(r0 + uint(c.get_multidimensional_index(i)[0]), a.rows - 1)];
    c.store(mY);
}

kernel void mm_f32_tensor(device float* W [[buffer(0)]],
                          device const float* bias [[buffer(1)]],
                          device ACT_T* X [[buffer(2)]],
                          device float* Y [[buffer(3)]],
                          constant MMArgs& a [[buffer(4)]],
                          uint3 tg [[threadgroup_position_in_grid]]) {
    constexpr auto desc = matmul2d_descriptor(TT_TOK, TT_ROWS, static_cast<int>(dynamic_extent), false, true, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<TT_NSG>> op;

    const uint r0 = tg.x * TT_ROWS, n0 = tg.y * TT_TOK;
    act_tensor tX(X, dextents<int, 2>(a.K, a.n_tok), array<int, 2>{1, int(a.x_stride)});
    f32_tensor tW(W, dextents<int, 2>(a.K, a.rows), array<int, 2>{1, int(a.K)});
    f32_tensor tY(Y + a.y_col, dextents<int, 2>(a.rows, a.n_tok), array<int, 2>{1, int(a.y_stride)});
    auto mX = tX.slice(0, int(n0));
    auto mW = tW.slice(0, int(r0));
    auto mY = tY.slice(int(r0), int(n0));

    auto c = op.get_destination_cooperative_tensor<decltype(mX), decltype(mW), float>();
    if (a.mode == 1) c.load(mY);
    else tt_zero(c);
    op.run(mX, mW, c);

    if (a.has_bias)
        for (ushort i = 0; i < c.get_capacity(); ++i)
            if (c.is_valid_element(i)) c[i] += bias[min(r0 + uint(c.get_multidimensional_index(i)[0]), a.rows - 1)];
    c.store(mY);
}

// ---- mode 2, fused SwiGLU:  Y (ACT_T) = silu(W[r] x) * W[r + rows] x --------
//
// Gate and up tiles are accumulated in two cooperative tensors with the same
// layout, combined element-wise in registers, and stored once.

// The accumulators stay fp32; the activation is written element by element
// because the destination (ACT_T) may be narrower than the registers.
template <class CT>
static inline void tt_swiglu_store(thread CT& gate, thread CT& up, device ACT_T* Y, constant MMArgs& a, uint r0, uint n0) {
    for (ushort i = 0; i < gate.get_capacity(); ++i) {
        if (!gate.is_valid_element(i)) continue;
        const auto idx = gate.get_multidimensional_index(i);
        const uint row = r0 + uint(idx[0]), n = n0 + uint(idx[1]);
        if (row >= a.rows || n >= a.n_tok) continue;
        const float g = gate[i];
        Y[n * a.y_stride + a.y_col + row] = ACT_T(clamp(g / (1.0f + exp(-g)) * up[i], -ACT_MAX, ACT_MAX));
    }
}

kernel void mm_q4_tensor_swiglu(device const uint* Wq [[buffer(0)]],
                                device const float* bias [[buffer(1)]],
                                device ACT_T* X [[buffer(2)]],
                                device ACT_T* Y [[buffer(3)]],
                                constant MMArgs& a [[buffer(4)]],
                                device const half* Ws [[buffer(5)]],
                                uint3 tg [[threadgroup_position_in_grid]],
                                ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup half sw0[TT_ROWS * TT_K];
    threadgroup half sw1[TT_ROWS * TT_K];

    constexpr auto desc = matmul2d_descriptor(TT_TOK, TT_ROWS, TT_K, false, true, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<TT_NSG>> op;

    const uint r0 = tg.x * TT_ROWS, n0 = tg.y * TT_TOK;
    const uint U = a.K / 8, G = a.K / 32;

    act_tensor tX(X, dextents<int, 2>(a.K, a.n_tok), array<int, 2>{1, int(a.x_stride)});
    tg_tensor tW0(sw0, dextents<int, 2>(TT_K, TT_ROWS));
    tg_tensor tW1(sw1, dextents<int, 2>(TT_K, TT_ROWS));
    auto mX0 = tX.slice(0, int(n0));

    auto gate = op.get_destination_cooperative_tensor<decltype(mX0), decltype(tW0), float>();
    auto up = op.get_destination_cooperative_tensor<decltype(mX0), decltype(tW0), float>();
    tt_zero(gate);
    tt_zero(up);

    const uint words = TT_ROWS * (TT_K / 8);
    for (uint k0 = 0; k0 < a.K; k0 += TT_K) {
        for (uint e = tid; e < words; e += TT_NSG * SG) {
            const uint r = e / (TT_K / 8), word = e % (TT_K / 8);
            const uint row = min(r0 + r, a.rows - 1);
            q4_expand(Wq, Ws, row * U + k0 / 8 + word, row * G + (k0 + word * 8) / 32, sw0 + r * TT_K + word * 8);
            q4_expand(Wq, Ws, (row + a.rows) * U + k0 / 8 + word, (row + a.rows) * G + (k0 + word * 8) / 32, sw1 + r * TT_K + word * 8);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mX = tX.slice(int(k0), int(n0));
        op.run(mX, tW0, gate);
        op.run(mX, tW1, up);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    tt_swiglu_store(gate, up, Y, a, r0, n0);
}

kernel void mm_f32_tensor_swiglu(device float* W [[buffer(0)]],
                                 device const float* bias [[buffer(1)]],
                                 device ACT_T* X [[buffer(2)]],
                                 device ACT_T* Y [[buffer(3)]],
                                 constant MMArgs& a [[buffer(4)]],
                                 uint3 tg [[threadgroup_position_in_grid]]) {
    constexpr auto desc = matmul2d_descriptor(TT_TOK, TT_ROWS, static_cast<int>(dynamic_extent), false, true, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<TT_NSG>> op;

    const uint r0 = tg.x * TT_ROWS, n0 = tg.y * TT_TOK;
    act_tensor tX(X, dextents<int, 2>(a.K, a.n_tok), array<int, 2>{1, int(a.x_stride)});
    f32_tensor tW(W, dextents<int, 2>(a.K, 2 * a.rows), array<int, 2>{1, int(a.K)});
    auto mX = tX.slice(0, int(n0));
    auto mW0 = tW.slice(0, int(r0));
    auto mW1 = tW.slice(0, int(r0 + a.rows));

    auto gate = op.get_destination_cooperative_tensor<decltype(mX), decltype(mW0), float>();
    auto up = op.get_destination_cooperative_tensor<decltype(mX), decltype(mW0), float>();
    tt_zero(gate);
    tt_zero(up);
    op.run(mX, mW0, gate);
    op.run(mX, mW1, up);
    tt_swiglu_store(gate, up, Y, a, r0, n0);
}
