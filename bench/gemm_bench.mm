// GEMM "boss fight": our Metal kernels vs Apple's MPSMatrixMultiplication
// (the vendor BLAS for the GPU, i.e. the cuBLAS of this platform).
//
//   build/gemm_bench [--half] [M N K]...
//
// Reports GFLOP/s (2*M*N*K / GPU time) and the ratio to MPS.
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "../src/metal_ctx.h"
#include "../src/util.h"

struct GemmArgs {
    uint32_t M, N, K, a_off, b_off, c_off, lda, ldb, ldc;
};

struct Shape {
    int M, N, K;
};

static double gpu_seconds(id<MTLCommandBuffer> cb) { return cb.GPUEndTime - cb.GPUStartTime; }

// Best-of-`reps` GPU time for one multiplication, `inner` multiplications per command buffer.
template <class Encode>
static double time_it(MetalCtx& mc, int reps, int inner, Encode encode) {
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        id<MTLCommandBuffer> cb = [mc.queue commandBuffer];
        for (int i = 0; i < inner; i++) encode(cb);
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) die("gpu error: %s", cb.error.localizedDescription.UTF8String);
        best = std::min(best, gpu_seconds(cb) / inner);
    }
    return best;
}

int main(int argc, char** argv) {
    @autoreleasepool {
        bool half = false, half_out = false;
        auto env = [](const char* k, int d) { return getenv(k) ? atoi(getenv(k)) : d; };
        std::vector<Shape> shapes;
        std::vector<int> nums;
        for (int i = 1; i < argc; i++) {
            if (!strcmp(argv[i], "--half")) half = true;
            else if (!strcmp(argv[i], "--half-out")) half = half_out = true;
            else nums.push_back(atoi(argv[i]));
        }
        for (size_t i = 0; i + 2 < nums.size(); i += 3) shapes.push_back({nums[i], nums[i + 1], nums[i + 2]});
        if (shapes.empty())
            shapes = {{512, 512, 512},   {1024, 1024, 1024}, {2048, 2048, 2048}, {4096, 4096, 4096},
                      {512, 1536, 1536}, {512, 8960, 1536},  {512, 1536, 8960}};

        MetalCtx mc;
        NSString* real = half ? @"half" : @"float";
        const int TM = env("TM", 64), TN = env("TN", 32), NSG = env("NSG", 4);
        id<MTLLibrary> lib = mc.compile(embedded_kernel_source(), @{
            @"REAL" : real, @"REAL4" : [real stringByAppendingString:@"4"], @"CREAL" : half_out ? @"half" : @"float",
            @"TM" : @(TM).stringValue, @"TN" : @(TN).stringValue, @"NSG" : @(NSG).stringValue,
            @"RELAXED" : env("RELAXED", 0) ? @"true" : @"false"
        });
        id<MTLComputePipelineState> p_naive = mc.pipeline(lib, "gemm_naive");
        id<MTLComputePipelineState> p_reg = mc.pipeline(lib, "gemm_reg");
        id<MTLComputePipelineState> p_tiled = mc.pipeline(lib, "gemm_tiled");
        id<MTLComputePipelineState> p_mpp = mc.pipeline(lib, "gemm_mpp");
        const int TG = 8;

        printf("device: %s   precision: %s\n", mc.dev.name.UTF8String, half_out ? "fp16 (tensor kernel writes fp16)" : half ? "fp16 in / fp32 out" : "fp32");
        printf("%-18s %10s %10s %10s %10s %10s   %s\n", "M x N x K", "MPS", "naive", "reg4x4", "tiled", "tensor", "tensor/MPS");

        const size_t esz = half ? 2 : 4;
        for (auto s : shapes) {
            const size_t M = s.M, N = s.N, K = s.K;
            id<MTLBuffer> A = mc.buffer(M * K * esz), B = mc.buffer(N * K * esz);
            id<MTLBuffer> C = mc.buffer(M * N * 4), Cref = mc.buffer(M * N * esz);
            std::mt19937 rng(1);
            std::uniform_real_distribution<float> U(-1, 1);
            auto fill = [&](id<MTLBuffer> buf, size_t n) {
                if (half) { __fp16* p = (__fp16*)buf.contents; for (size_t i = 0; i < n; i++) p[i] = (__fp16)U(rng); }
                else { float* p = (float*)buf.contents; for (size_t i = 0; i < n; i++) p[i] = U(rng); }
            };
            fill(A, M * K);
            fill(B, N * K);

            GemmArgs ga = {(uint32_t)M, (uint32_t)N, (uint32_t)K, 0, 0, 0, (uint32_t)K, (uint32_t)K, (uint32_t)N};
            id<MTLBuffer> args = mc.buffer(&ga, sizeof ga);
            const double flops = 2.0 * M * N * K;
            const int inner = flops < 5e9 ? 8 : 2;

            // --- MPS baseline: C = A * B^T
            MPSDataType dt = half ? MPSDataTypeFloat16 : MPSDataTypeFloat32;
            MPSMatrix* mA = [[MPSMatrix alloc] initWithBuffer:A descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K rowBytes:K * esz dataType:dt]];
            MPSMatrix* mB = [[MPSMatrix alloc] initWithBuffer:B descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:N columns:K rowBytes:K * esz dataType:dt]];
            MPSMatrix* mC = [[MPSMatrix alloc] initWithBuffer:Cref descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:N * esz dataType:dt]];
            MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc] initWithDevice:mc.dev transposeLeft:NO transposeRight:YES resultRows:M resultColumns:N interiorColumns:K alpha:1.0 beta:0.0];
            double t_mps = time_it(mc, 6, inner, [&](id<MTLCommandBuffer> cb) {
                [mm encodeToCommandBuffer:cb leftMatrix:mA rightMatrix:mB resultMatrix:mC];
            });

            auto run = [&](id<MTLComputePipelineState> p, MTLSize groups, MTLSize tpg, bool check, bool c_half = false) {
                memset(C.contents, 0, M * N * 4);
                double t = time_it(mc, 6, inner, [&](id<MTLCommandBuffer> cb) {
                    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                    [e setComputePipelineState:p];
                    [e setBuffer:A offset:0 atIndex:0];
                    [e setBuffer:B offset:0 atIndex:1];
                    [e setBuffer:C offset:0 atIndex:2];
                    [e setBuffer:args offset:0 atIndex:3];
                    [e dispatchThreadgroups:groups threadsPerThreadgroup:tpg];
                    [e endEncoding];
                });
                if (check) {
                    double max_err = 0, max_ref = 0;
                    const float* c = (const float*)C.contents;
                    const __fp16* ch = (const __fp16*)C.contents;
                    for (size_t i = 0; i < M * N; i += 97) {
                        double ref = half ? (double)((const __fp16*)Cref.contents)[i] : (double)((const float*)Cref.contents)[i];
                        max_err = std::max(max_err, std::fabs((c_half ? (double)ch[i] : (double)c[i]) - ref));
                        max_ref = std::max(max_ref, std::fabs(ref));
                    }
                    if (max_err > (half ? 2e-2 : 1e-4) * max_ref) printf("  !! mismatch vs MPS: max err %.4g (ref scale %.4g)\n", max_err, max_ref);
                }
                return t;
            };
            auto ceil_div = [](size_t a, size_t b) { return (a + b - 1) / b; };

            double t_naive = flops > 4e10 ? NAN : run(p_naive, MTLSizeMake(ceil_div(N, 16), ceil_div(M, 16), 1), MTLSizeMake(16, 16, 1), true);
            double t_reg = run(p_reg, MTLSizeMake(ceil_div(N, 32), ceil_div(M, 32), 1), MTLSizeMake(8, 8, 1), true);
            double t_tiled = run(p_tiled, MTLSizeMake(ceil_div(N, 4 * TG), ceil_div(M, 4 * TG), 1), MTLSizeMake(TG, TG, 1), true);

            double t_mpp = run(p_mpp, MTLSizeMake(ceil_div(N, TN), ceil_div(M, TM), 1), MTLSizeMake(p_mpp.threadExecutionWidth * NSG, 1, 1), true, half_out);

            char name[64];
            snprintf(name, sizeof name, "%zux%zux%zu", M, N, K);
            printf("%-18s %10.1f %10.1f %10.1f %10.1f %10.1f   %5.1f%%\n", name, flops / t_mps / 1e9, flops / t_naive / 1e9,
                   flops / t_reg / 1e9, flops / t_tiled / 1e9, flops / t_mpp / 1e9, 100.0 * t_mps / t_mpp);
        }
    }
    return 0;
}
