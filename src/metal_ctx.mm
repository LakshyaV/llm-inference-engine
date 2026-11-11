#include "metal_ctx.h"

#include "util.h"

static const unsigned char kKernelSource[] = {
#include "kernels.inc"
    , 0};

const char* embedded_kernel_source() { return (const char*)kKernelSource; }

MetalCtx::MetalCtx() {
    dev = MTLCreateSystemDefaultDevice();
    if (!dev) die("metal: no default device");
    queue = [dev newCommandQueue];
}

id<MTLLibrary> MetalCtx::compile(const std::string& source, NSDictionary<NSString*, NSObject*>* macros) {
    MTLCompileOptions* opt = [MTLCompileOptions new];
    opt.mathMode = MTLMathModeFast;
    if (macros) opt.preprocessorMacros = macros;
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()] options:opt error:&err];
    if (!lib) die("metal: shader compile failed:\n%s", err.localizedDescription.UTF8String);
    return lib;
}

id<MTLComputePipelineState> MetalCtx::pipeline(id<MTLLibrary> lib, const char* fn) {
    id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:fn]];
    if (!f) die("metal: no kernel named %s", fn);
    NSError* err = nil;
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:&err];
    if (!p) die("metal: pipeline %s: %s", fn, err.localizedDescription.UTF8String);
    return p;
}

id<MTLBuffer> MetalCtx::buffer(size_t bytes) {
    id<MTLBuffer> b = [dev newBufferWithLength:bytes ? bytes : 16 options:MTLResourceStorageModeShared];
    if (!b) die("metal: failed to allocate %zu bytes", bytes);
    return b;
}

id<MTLBuffer> MetalCtx::buffer(const void* data, size_t bytes) {
    id<MTLBuffer> b = [dev newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
    if (!b) die("metal: failed to allocate %zu bytes", bytes);
    return b;
}
