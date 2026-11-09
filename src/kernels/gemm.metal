// ============================================================================
// Dense GEMM:  C[M,N] = A[M,K] * B[N,K]^T   (row-major, "NT" form)
//
// This is the layout every linear layer wants (activations times weight rows)
// so both operands are read along their contiguous K axis.
//
// Three generations of the kernel are kept so bench/gemm_bench can show what
// each step buys:
//   gemm_naive   one thread per output element
//   gemm_reg     register tiling: one thread owns an RBxCB block of C
//   gemm_tiled   register tiling + threadgroup-memory tiling: a threadgroup
//                stages a tile of A and B in shared memory, then every thread
//                computes its block from the staged copy
// ============================================================================
#include <metal_stdlib>
using namespace metal;

#ifndef REAL
#define REAL float      // storage type of A and B (float or half)
#define REAL4 float4
#endif
#ifndef CREAL
#define CREAL float     // storage type of C for the tensor kernel
#endif
#ifndef RELAXED
#define RELAXED false
#endif

struct GemmArgs {
    uint M, N, K;
    uint a_off, b_off, c_off;   // element offsets into the buffers
    uint lda, ldb, ldc;         // row strides (elements)
};

kernel void gemm_naive(device const REAL* A [[buffer(0)]],
                       device const REAL* B [[buffer(1)]],
                       device float* C [[buffer(2)]],
                       constant GemmArgs& g [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= g.N || gid.y >= g.M) return;
    const device REAL* a = A + g.a_off + gid.y * g.lda;
    const device REAL* b = B + g.b_off + gid.x * g.ldb;
    float acc = 0;
    for (uint k = 0; k < g.K; k++) acc += float(a[k]) * float(b[k]);
    C[g.c_off + gid.y * g.ldc + gid.x] = acc;
}

// ---------------------------------------------------------------------------
// Register-tiled: each thread computes a 4x4 block of C. Per 4 steps of K it
// loads 4 vectors of A and 4 of B and performs 16 dot products, so every
// loaded value is reused 4 times straight from registers.
// K must be a multiple of 4; M and N are handled with clamped rows + guards.
// ---------------------------------------------------------------------------
kernel void gemm_reg(device const REAL* A [[buffer(0)]],
                     device const REAL* B [[buffer(1)]],
                     device float* C [[buffer(2)]],
                     constant GemmArgs& g [[buffer(3)]],
                     uint2 gid [[thread_position_in_grid]]) {
    const uint i0 = gid.y * 4, j0 = gid.x * 4;
    if (i0 >= g.M || j0 >= g.N) return;
    const device REAL4* a[4];
    const device REAL4* b[4];
    for (uint r = 0; r < 4; r++) {
        a[r] = (const device REAL4*)(A + g.a_off + min(i0 + r, g.M - 1) * g.lda);
        b[r] = (const device REAL4*)(B + g.b_off + min(j0 + r, g.N - 1) * g.ldb);
    }
    float4 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    const uint K4 = g.K / 4;
    for (uint k = 0; k < K4; k++) {
        const float4 a0 = float4(a[0][k]), a1 = float4(a[1][k]), a2 = float4(a[2][k]), a3 = float4(a[3][k]);
        const float4 b0 = float4(b[0][k]), b1 = float4(b[1][k]), b2 = float4(b[2][k]), b3 = float4(b[3][k]);
        c0 += float4(dot(a0, b0), dot(a0, b1), dot(a0, b2), dot(a0, b3));
        c1 += float4(dot(a1, b0), dot(a1, b1), dot(a1, b2), dot(a1, b3));
        c2 += float4(dot(a2, b0), dot(a2, b1), dot(a2, b2), dot(a2, b3));
        c3 += float4(dot(a3, b0), dot(a3, b1), dot(a3, b2), dot(a3, b3));
    }
    const float4 c[4] = {c0, c1, c2, c3};
    for (uint r = 0; r < 4 && i0 + r < g.M; r++)
        for (uint q = 0; q < 4 && j0 + q < g.N; q++) C[g.c_off + (i0 + r) * g.ldc + j0 + q] = c[r][q];
}

// ---------------------------------------------------------------------------
// Threadgroup-tiled. A TGxTG threadgroup covers a (4*TG)x(4*TG) tile of C.
// For each K-tile of KT elements the group copies the matching slabs of A and
// B into threadgroup memory (every thread copies its share in parallel),
// synchronises, and then each thread runs the 4x4 register kernel against the
// staged slabs. Slabs are stored as float4 so a row never straddles a vector.
//
// Threadgroup memory is limited (32 KB on Apple GPUs): two slabs of
// (4*TG) rows x KT floats = 2 * 32 * 32 * 4 B = 8 KB at the default TG=8,KT=32.
// ---------------------------------------------------------------------------
#ifndef TG
#define TG 8
#endif
#ifndef KT
#define KT 32
#endif
#define TILE (TG * 4)
#define KT4 (KT / 4)

kernel void gemm_tiled(device const REAL* A [[buffer(0)]],
                       device const REAL* B [[buffer(1)]],
                       device float* C [[buffer(2)]],
                       constant GemmArgs& g [[buffer(3)]],
                       uint2 tg [[threadgroup_position_in_grid]],
                       uint2 lid [[thread_position_in_threadgroup]]) {
    threadgroup REAL4 sa[TILE * KT4];
    threadgroup REAL4 sb[TILE * KT4];

    const uint ti0 = tg.y * TILE, tj0 = tg.x * TILE;   // tile origin in C
    const uint tid = lid.y * TG + lid.x;
    const uint ar = lid.y * 4, br = lid.x * 4;         // this thread's rows inside the slabs

    float4 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    for (uint k0 = 0; k0 < g.K; k0 += KT) {
        // Stage: TILE*KT4 vectors per slab, striped across the TG*TG threads.
        for (uint e = tid; e < TILE * KT4; e += TG * TG) {
            const uint row = e / KT4, kv = e % KT4;
            const uint k = k0 + kv * 4;
            const bool live = k < g.K;
            const uint ia = min(ti0 + row, g.M - 1), ib = min(tj0 + row, g.N - 1);
            sa[e] = live ? *(const device REAL4*)(A + g.a_off + ia * g.lda + k) : REAL4(0);
            sb[e] = live ? *(const device REAL4*)(B + g.b_off + ib * g.ldb + k) : REAL4(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < KT4; k++) {
            const float4 a0 = float4(sa[(ar + 0) * KT4 + k]), a1 = float4(sa[(ar + 1) * KT4 + k]);
            const float4 a2 = float4(sa[(ar + 2) * KT4 + k]), a3 = float4(sa[(ar + 3) * KT4 + k]);
            const float4 b0 = float4(sb[(br + 0) * KT4 + k]), b1 = float4(sb[(br + 1) * KT4 + k]);
            const float4 b2 = float4(sb[(br + 2) * KT4 + k]), b3 = float4(sb[(br + 3) * KT4 + k]);
            c0 += float4(dot(a0, b0), dot(a0, b1), dot(a0, b2), dot(a0, b3));
            c1 += float4(dot(a1, b0), dot(a1, b1), dot(a1, b2), dot(a1, b3));
            c2 += float4(dot(a2, b0), dot(a2, b1), dot(a2, b2), dot(a2, b3));
            c3 += float4(dot(a3, b0), dot(a3, b1), dot(a3, b2), dot(a3, b3));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint i0 = ti0 + ar, j0 = tj0 + br;
    const float4 c[4] = {c0, c1, c2, c3};
    for (uint r = 0; r < 4 && i0 + r < g.M; r++)
        for (uint q = 0; q < 4 && j0 + q < g.N; q++) C[g.c_off + (i0 + r) * g.ldc + j0 + q] = c[r][q];
}
