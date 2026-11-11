// Thin Objective-C++ wrapper around a Metal device: runtime shader
// compilation (no Xcode / offline `metal` compiler needed), pipeline cache
// and shared-storage buffers. Include only from .mm files.
#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <map>
#include <string>

struct MetalCtx {
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> queue = nil;

    MetalCtx();

    // Compile MSL source. `macros` become preprocessor definitions.
    id<MTLLibrary> compile(const std::string& source, NSDictionary<NSString*, NSObject*>* macros = nil);
    id<MTLComputePipelineState> pipeline(id<MTLLibrary> lib, const char* fn);

    id<MTLBuffer> buffer(size_t bytes);                    // zero-filled, CPU+GPU visible
    id<MTLBuffer> buffer(const void* data, size_t bytes);  // copy
};

// All kernels shipped with the engine (src/kernels/*.metal, embedded at build time).
const char* embedded_kernel_source();
