# llm-inference-engine

A from-scratch LLM inference engine for Apple GPUs. No PyTorch, no MLX, no
ggml: C++17, Objective-C++ and hand-written Metal compute kernels, built with
nothing but the Xcode Command Line Tools.

It loads real open-weights checkpoints (Qwen2.5, straight from their
`safetensors` files), reproduces the Hugging Face reference logits, and runs
int4-quantized models at speeds comparable to llama.cpp on the same machine.

```
$ build/llm generate -m models/Qwen2.5-1.5B-Instruct --chat -p "Write a haiku about GPUs."
Powerful processors,
Graphics crunching at full speed,
Speeding up games.

prefill: 26 tokens | decode: 16 tokens, 211.1 tok/s
```

## What is in here

| Piece | Where | Notes |
| --- | --- | --- |
| safetensors loader | `src/safetensors.*`, `src/json.h` | mmap, own JSON parser, F32/F16/BF16, sharded checkpoints |
| BPE tokenizer | `src/tokenizer.*` | byte-level BPE from `tokenizer.json`; NFC normalization and the pre-tokenizer regex are hand-written on generated Unicode tables |
| Metal kernels | `src/kernels/model.metal` | embedding, RMSNorm, RoPE, linear layers (fp32 + int4), SwiGLU, attention |
| Tiled GEMM | `src/kernels/gemm.metal`, `bench/gemm_bench.mm` | naive -> register-tiled -> threadgroup-memory-tiled -> matrix-unit ("tensor core") kernel, benchmarked against MPS |
| Fused flash attention | `attention_impl` in `model.metal` | QK^T, causal online softmax and PV in one kernel, tiled over keys, no score matrix |
| Paged KV cache | `src/kv_pager.*` | fixed-size pages + per-sequence page table read by the GPU |
| int4 quantization | `src/quant.*` | 4-bit weights, one fp16 scale per group of 32 (4.5 bits/weight), MSE-optimal scale search |
| Continuous batching | `src/scheduler.*` | requests join/leave the batch every step; prefill and decode share a pass |
| Speculative decoding | `src/speculative.*` | draft + target with exact acceptance/residual sampling and KV rollback |
| Reference tooling | `tools/reference.py` | the only PyTorch in the repo: dumps Hugging Face logits/token ids for the tests |

## CUDA concepts, Metal edition

This was developed on an M5 Pro, so the usual CUDA vocabulary maps like this:

| CUDA | Here |
| --- | --- |
| cuBLAS | `MPSMatrixMultiplication` (Apple's GPU BLAS) is the bar to clear |
| tensor cores / WMMA | the matrix units in M5-class GPU cores, driven from shaders through Metal 4's `matmul2d` tensor ops |
| warp | SIMD-group (32 threads); `simd_sum` / `simd_max` are the warp-level reductions |
| shared memory | threadgroup memory (tiles of A/B, staged int4 weights, attention score tiles) |
| CUDA graphs / streams | one command buffer and one compute encoder per forward pass |

Shaders are compiled at runtime from source embedded in the binary, so neither
Xcode nor the offline `metal` compiler is needed.

## Results

Apple M5 Pro (24 GB), macOS 26.5. All numbers measured on this machine.

### Logits match the reference

`tests/test_logits` runs the fp32 engine against `transformers` float32 logits
for four prompts (201 positions): once with the whole prompt in one pass and
once incrementally through the KV cache.

| | max abs logit diff | argmax agreement |
| --- | --- | --- |
| Metal engine, one-shot | 3.9e-3 | 201 / 201 |
| Metal engine, incremental decode | 6.4e-4 | 4 / 4 |
| plain C++ reference (`src/cpu_ref.cpp`) | 3.1e-3 | 201 / 201 |

(Logits span roughly +-30; the residual is fp32 summation order.)

### Tokens per second vs llama.cpp

`llm bench` against `llama-bench -p 512 -n 128 -ngl 99` (build 10621, Metal).
Ours: int4 + group scales (4.5 bpw). llama.cpp: the official `Q4_K_M` GGUF
(~4.9 bpw, a better quantizer), so the comparison is close but not identical.

| Model | Test | llama.cpp | this engine | ratio |
| --- | --- | --- | --- | --- |
| Qwen2.5-1.5B-Instruct | prefill, 512 tokens | 6448 tok/s | 5812 tok/s | 90% |
| | decode, 128 tokens | 197 tok/s | 215 tok/s | 109% |
| Qwen2.5-0.5B-Instruct | prefill, 512 tokens | 18213 tok/s | 17287 tok/s | 95% |
| | decode, 128 tokens | 349 tok/s | 426 tok/s | 122% |

### GEMM vs MPS (the boss fight)

`build/gemm_bench`, GFLOP/s, C = A * B^T.

| fp32, M x N x K | MPS | naive | register 4x4 | threadgroup-tiled | matrix units | vs MPS |
| --- | --- | --- | --- | --- | --- | --- |
| 1024^3 | 6933 | 802 | 4547 | 4160 | 7433 | 107% |
| 2048^3 | 7074 | 795 | 4610 | 4288 | 7670 | 108% |
| 4096^3 | 7000 | - | 4553 | 4285 | 7327 | 105% |
| 512 x 8960 x 1536 | 7422 | 781 | 4446 | 4360 | 7812 | 105% |

| fp16, M x N x K | MPS | matrix units | vs MPS |
| --- | --- | --- | --- |
| 1024^3 | 30224 | 30690 | 102% |
| 2048^3 | 31785 | 31635 | 100% |
| 4096^3 | 30696 | 20949 | 68% |
| 512 x 8960 x 1536 | 31641 | 31331 | 99% |
| 512 x 1536 x 8960 | 30455 | 26966 | 89% |

What mattered, in order:

1. **Register tiling** (each thread owns a 4x4 block of C): 0.8 -> 4.5 TFLOP/s.
2. **Threadgroup-memory tiling** on top of that did *not* help on this GPU
   (4.2-4.4 TFLOP/s): unlike a discrete card, device memory reads are already
   cheap, and the staging copy plus barriers cost more than they save. The
   kernel is kept for comparison.
3. **Matrix units** via `matmul2d`: fp32 matches MPS. For fp16 the tile shape
   is everything: each SIMD-group needs a full 32x32 output tile. 64x32 on
   four SIMD-groups runs at 16 TFLOP/s; 64x64 on four runs at 31.

Still open: very long K at fp16 (4096^3 is at 68%); the kernel lets the op walk
all of K in one call and would need its own K-blocking there.

### Continuous batching

`llm batch`, 32 chat requests, up to 96 new tokens each, all submitted at once
(Qwen2.5-1.5B, int4):

| concurrent sequences | throughput | mean time to first token |
| --- | --- | --- |
| 1 | 206 tok/s | 7.2 s |
| 4 | 534 tok/s | 2.5 s |
| 16 | 737 tok/s | 1.1 s |
| 32 | 1280 tok/s | 0.1 s |

These were measured with the matrix-unit kernels taking over at 8 tokens per
pass, where a batch of 8 was *slower* (391 tok/s) than a batch of 4. The
threshold has since been raised to 16 but the sweep has not been re-run, so the
8-16 range is untuned. It is a runtime option (`--tensor-min N`), so the sweep
is `for n in 8 12 16 24 32; do build/llm batch -m <model> --tensor-min $n; done`.

### Speculative decoding

`llm spec`, target Qwen2.5-1.5B, draft Qwen2.5-0.5B, greedy, 8 prompts:

| gamma | acceptance | tokens per target pass | speed vs plain decoding | output identical to greedy |
| --- | --- | --- | --- | --- |
| 2 | 55% | 2.07 | 0.83x | 8 / 8 |
| 4 | 40% | 2.53 | 0.58x | 8 / 8 |
| 8 | 24% | 2.80 | 0.36x | 8 / 8 |

The algorithm is correct (identical output, >2 tokens per target pass) but it
is a net **slowdown** with this pair: the draft costs 2.3 ms/token against
4.7 ms for the target, far from the ~10x gap speculative decoding needs. A
larger target (e.g. Qwen2.5-7B with the same draft) is the configuration where
it should pay off; that has not been measured.

## How a forward pass works

`Engine::forward(tokens, want)` takes any mix of `(token, sequence)` pairs and
runs them in one command buffer. Single-stream decode, prefill, a continuous
batch and a speculative verification pass are all the same call.

Per layer: RMSNorm -> fused q|k|v projection -> RoPE + KV-cache write ->
fused attention -> output projection (residual add fused into the store) ->
RMSNorm -> fused SwiGLU (gate and up rows computed together, activation applied
in registers) -> down projection (residual add fused).

Linear layers have two implementations chosen by how many tokens are in the pass:

* **few tokens (decode):** one SIMD-group per pair of output rows; the 32
  threads stripe the K axis, unpack int4 nibbles in registers, and reduce with
  `simd_sum`.
* **many tokens (prefill/batches):** 64x64 output tiles on the matrix units.
  int4 weights are expanded 64 columns at a time into threadgroup memory as
  fp16; results accumulate in a cooperative tensor so bias, residual and
  SwiGLU are applied before a single store.

The int4 engine keeps the residual stream in fp32 and the linear-layer inputs
in fp16, which is what lets the matrix units run at their fp16 rate.

Attention walks the cached keys in tiles with an online softmax (running max
and sum, rescaled per tile), reading K and V through the page table. Decode
uses 1024-thread groups so a long context is scored in parallel; prefill uses
one SIMD-group per (token, head).

## Build and run

```sh
make                       # needs only the Xcode Command Line Tools (macOS 26+, Metal 4)

# weights (any Qwen2.5 size; Llama-style checkpoints with the same layout also load)
mkdir -p models/Qwen2.5-1.5B-Instruct && cd models/Qwen2.5-1.5B-Instruct
for f in config.json generation_config.json tokenizer.json model.safetensors; do
  curl -LO https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/$f
done && cd ../..

build/llm generate -m models/Qwen2.5-1.5B-Instruct --chat -p "Why is the sky blue?"
build/llm bench    -m models/Qwen2.5-1.5B-Instruct [--profile] [--f32]
build/llm batch    -m models/Qwen2.5-1.5B-Instruct --requests 32
build/llm eval     -m models/Qwen2.5-1.5B-Instruct --file some_text.txt   # int4 vs fp32 quality
build/llm spec     -m models/Qwen2.5-1.5B-Instruct --draft models/Qwen2.5-0.5B-Instruct
build/gemm_bench [--half | --half-out] [M N K ...]
```

The first int4 run quantizes the checkpoint and caches the result next to it
(`q4.cache`). `--f32` uses the exact weights instead.

## Tests

```sh
make test                                          # unit tests: allocator, quantizer, safetensors, speculative sampling
python tools/reference.py models/Qwen2.5-1.5B-Instruct tests/data   # needs torch + transformers
make test MODEL=models/Qwen2.5-1.5B-Instruct       # + tokenizer and logit parity
```

## Known limitations

* int4 output quality has only been checked by reading generations. `llm eval`
  computes perplexity, top-1 agreement and KL against the fp32 engine, but it
  was added after the benchmark session and has not been run yet.
* Speculative sampling at temperature > 0 is verified at the level of the
  acceptance rule (statistical unit test), not end to end through two models.
* Sharded checkpoints (`model-0000x-of-0000y.safetensors`) are covered by a
  synthetic unit test, not yet by a real multi-file model.
* The matrix-unit kernels require an M5/A19-class GPU and macOS 26; there is
  no fallback GEMM path for older hardware in the engine.
* No chat-template engine: `--chat` hard-codes the ChatML format Qwen uses.
