#include "engine.h"

#include <cmath>
#include <cstring>
#include <map>

#include "kv_pager.h"
#include "metal_ctx.h"
#include "quant.h"
#include "safetensors.h"
#include "util.h"

// ---- argument structs: must mirror src/kernels/model.metal -----------------

struct EmbedArgs { uint32_t n_tok, d; };
struct NormArgs { uint32_t n_out, d; float eps; uint32_t w_off, gather; };
struct MMArgs { uint32_t n_tok, rows, K, x_stride, y_stride, y_col, mode, has_bias; };
struct TokenMeta { uint32_t seq, pos; };
struct KVArgs {
    uint32_t n_tok, n_heads, n_kv, head_dim, qkv_stride, layer, n_pages, page_shift, pt_stride;
    float scale;
    uint32_t d;
};

enum { kStore = 0, kAccumulate = 1, kSwiGLU = 2 };
constexpr uint32_t kTensorMinTokens = 16;

// One forward pass = one command buffer with one compute encoder. In profile
// mode every stage gets its own command buffer so GPU time can be attributed.
struct Pass {
    MetalCtx& mc;
    bool profile;
    std::map<std::string, double>* times;
    id<MTLCommandBuffer> cb = nil;
    id<MTLComputeCommandEncoder> e = nil;
    double gpu = 0;

    Pass(MetalCtx& m, bool prof, std::map<std::string, double>* t) : mc(m), profile(prof), times(t) { open(); }
    void open() {
        cb = [mc.queue commandBufferWithUnretainedReferences];
        e = [cb computeCommandEncoder];
    }
    void flush() {
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) die("gpu error: %s", cb.error.localizedDescription.UTF8String);
        gpu += cb.GPUEndTime - cb.GPUStartTime;
    }
    // Marks the end of a stage (no-op unless profiling).
    void stage(const char* name) {
        if (!profile) return;
        const double before = gpu;
        flush();
        (*times)[name] += gpu - before;
        open();
    }
};

// One linear layer (possibly several Hugging Face tensors stacked row-wise).
struct Linear {
    id<MTLBuffer> w = nil;       // F32: floats.  Q4: packed nibbles.
    id<MTLBuffer> scales = nil;  // Q4 only
    id<MTLBuffer> bias = nil;
    uint32_t rows = 0, K = 0;
};

struct Layer {
    Linear qkv, o, gate_up, down;
};

// Sequential cache of quantized tensors so int4 conversion runs once per model.
struct Q4Cache {
    FILE* f = nullptr;
    bool writing = false;
    std::string path, tmp;
    static constexpr uint64_t kMagic = 0x3176345155ull;  // "UQ4v1"

    explicit Q4Cache(const std::string& dir) : path(dir + "/q4.cache"), tmp(dir + "/q4.cache.tmp") {
        f = fopen(path.c_str(), "rb");
        uint64_t magic = 0;
        if (f && (fread(&magic, 8, 1, f) != 1 || magic != kMagic)) { fclose(f); f = nullptr; }
        if (!f) {
            f = fopen(tmp.c_str(), "wb");
            if (!f) die("cannot write %s", tmp.c_str());
            writing = true;
            fwrite(&kMagic, 8, 1, f);
        }
    }
    void io(void* p, size_t n) {
        if (writing) fwrite(p, 1, n, f);
        else if (fread(p, 1, n, f) != n) die("q4 cache %s is truncated; delete it and retry", path.c_str());
    }
    ~Q4Cache() {
        if (f) fclose(f);
        if (writing) rename(tmp.c_str(), path.c_str());
    }
};

struct Engine::Impl {
    ModelConfig cfg;
    EngineOptions opt;
    MetalCtx mc;
    std::unique_ptr<KVPager> pager;

    id<MTLComputePipelineState> p_embed, p_norm, p_rope;
    std::map<int, id<MTLComputePipelineState>> p_attn;  // key: SIMD-groups per threadgroup
    std::map<int, id<MTLComputePipelineState>> p_mm;  // key: NR * 16 + NT
    id<MTLComputePipelineState> p_mm_tensor, p_mm_tensor_swiglu;

    std::vector<Layer> layers;
    Linear head;                 // lm_head (shared with the embedding table when tied)
    Linear embed;
    id<MTLBuffer> norm_w;        // [(2L + 1), d]: per layer attn-norm, mlp-norm; then final norm
    id<MTLBuffer> rope;          // [max_positions, head_dim/2] (cos, sin)

    id<MTLBuffer> b_ids, b_meta, b_rows, b_pt;
    id<MTLBuffer> b_x, b_h, b_qkv, b_att, b_act, b_hn, b_logits, b_dummy;
    id<MTLBuffer> b_k, b_v;

    double gpu_seconds = 0;
    std::map<std::string, double> stage_seconds;
    size_t weight_bytes = 0;

    bool q4() const { return opt.format == WeightFormat::Q4; }

    // ---------------------------------------------------------- loading ----

    // Stack `names` row-wise into one matrix and upload it in the engine's weight format.
    Linear load_linear(const SafeTensors& st, const std::vector<std::string>& names, Q4Cache* cache) {
        Linear L;
        L.K = (uint32_t)st.get(names[0]).shape[1];
        for (auto& n : names) L.rows += (uint32_t)st.get(n).shape[0];
        const size_t numel = (size_t)L.rows * L.K;

        auto read_f32 = [&](float* dst) {
            for (auto& n : names) {
                const TensorView& t = st.get(n);
                SafeTensors::to_f32(t, dst);
                dst += t.numel();
            }
        };
        if (!q4()) {
            L.w = mc.buffer(numel * 4);
            read_f32((float*)L.w.contents);
            weight_bytes += numel * 4;
        } else {
            if (L.K % kQ4Group) die("int4 needs input width %% 32 == 0 (got %u)", L.K);
            L.w = mc.buffer(numel / 2);
            L.scales = mc.buffer(numel / kQ4Group * 2);
            if (cache->writing) {
                std::vector<float> tmp(numel);
                read_f32(tmp.data());
                quantize_q4(tmp.data(), L.rows, L.K, (uint32_t*)L.w.contents, (uint16_t*)L.scales.contents);
            }
            cache->io(L.w.contents, numel / 2);
            cache->io(L.scales.contents, numel / kQ4Group * 2);
            weight_bytes += numel / 2 + numel / kQ4Group * 2;
        }
        return L;
    }

    id<MTLBuffer> load_bias(const SafeTensors& st, const std::vector<std::string>& names) {
        size_t total = 0;
        for (auto& n : names) {
            if (!st.has(n)) return nil;
            total += st.get(n).numel();
        }
        id<MTLBuffer> b = mc.buffer(total * 4);
        float* dst = (float*)b.contents;
        for (auto& n : names) {
            SafeTensors::to_f32(st.get(n), dst);
            dst += st.get(n).numel();
        }
        return b;
    }

    void load(const std::string& dir) {
        SafeTensors st(SafeTensors::model_files(dir));
        std::unique_ptr<Q4Cache> cache;
        if (q4()) {
            cache = std::make_unique<Q4Cache>(dir);
            if (cache->writing && opt.verbose) fprintf(stderr, "quantizing to int4 (cached in %s)...\n", cache->path.c_str());
        }
        const int d = cfg.hidden, L = cfg.n_layers;
        norm_w = mc.buffer((size_t)(2 * L + 1) * d * 4);
        float* nw = (float*)norm_w.contents;

        layers.resize(L);
        for (int l = 0; l < L; l++) {
            const std::string p = "model.layers." + std::to_string(l) + ".";
            Layer& ly = layers[l];
            ly.qkv = load_linear(st, {p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"}, cache.get());
            ly.qkv.bias = load_bias(st, {p + "self_attn.q_proj.bias", p + "self_attn.k_proj.bias", p + "self_attn.v_proj.bias"});
            ly.o = load_linear(st, {p + "self_attn.o_proj.weight"}, cache.get());
            ly.gate_up = load_linear(st, {p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"}, cache.get());
            ly.down = load_linear(st, {p + "mlp.down_proj.weight"}, cache.get());
            SafeTensors::to_f32(st.get(p + "input_layernorm.weight"), nw + (size_t)(2 * l) * d);
            SafeTensors::to_f32(st.get(p + "post_attention_layernorm.weight"), nw + (size_t)(2 * l + 1) * d);
        }
        SafeTensors::to_f32(st.get("model.norm.weight"), nw + (size_t)(2 * L) * d);

        embed = load_linear(st, {"model.embed_tokens.weight"}, cache.get());
        head = st.has("lm_head.weight") ? load_linear(st, {"lm_head.weight"}, cache.get()) : embed;
    }

    // ------------------------------------------------------------ setup ----

    Impl(const std::string& dir, const EngineOptions& o) : cfg(ModelConfig::load(dir)), opt(o) {
        if (cfg.head_dim % 4 || cfg.head_dim / 4 > 32) die("unsupported head_dim %d", cfg.head_dim);
        if (cfg.hidden % 32 || cfg.inter % 32) die("hidden/intermediate sizes must be multiples of 32");
        if (opt.page_tokens <= 0 || (opt.page_tokens & (opt.page_tokens - 1))) die("page_tokens must be a power of two");

        // The int4 engine carries linear-layer inputs in fp16 (see model.metal).
        id<MTLLibrary> lib = mc.compile(embedded_kernel_source(),
                                        q4() ? @{@"ACT_T" : @"half", @"ACT4" : @"packed_half4"} : nil);
        p_embed = mc.pipeline(lib, q4() ? "embed_q4" : "embed_f32");
        p_norm = mc.pipeline(lib, "rmsnorm");
        for (auto [nr, nt] : std::vector<std::pair<int, int>>{{1, 1}, {2, 1}, {4, 1}, {8, 1}, {1, 4}, {2, 4}, {4, 4}}) {
            const std::string name = std::string(q4() ? "mm_q4_r" : "mm_f32_r") + std::to_string(nr) + "_t" + std::to_string(nt);
            p_mm[nr * 16 + nt] = mc.pipeline(lib, name.c_str());
        }
        p_mm_tensor = mc.pipeline(lib, q4() ? "mm_q4_tensor" : "mm_f32_tensor");
        p_mm_tensor_swiglu = mc.pipeline(lib, q4() ? "mm_q4_tensor_swiglu" : "mm_f32_tensor_swiglu");
        p_rope = mc.pipeline(lib, "rope_kv_write");
        for (int sgs : {1, 2, 4, 8, 16, 32}) p_attn[sgs] = mc.pipeline(lib, ("attention_sg" + std::to_string(sgs)).c_str());

        load(dir);

        const size_t N = opt.max_batch_tokens, d = cfg.hidden, R = opt.max_logit_rows;
        const size_t qkv = d + 2 * cfg.kv_dim();
        b_ids = mc.buffer(N * 4);
        b_meta = mc.buffer(N * sizeof(TokenMeta));
        b_rows = mc.buffer(R * 4);
        b_x = mc.buffer(N * d * 4);
        b_h = mc.buffer(N * d * 4);
        b_qkv = mc.buffer(N * qkv * 4);
        b_att = mc.buffer(N * d * 4);
        b_act = mc.buffer(N * cfg.inter * 4);
        b_hn = mc.buffer(R * d * 4);
        b_logits = mc.buffer(R * (size_t)cfg.vocab_size * 4);
        b_dummy = mc.buffer(16);

        const size_t kv_floats = (size_t)cfg.n_layers * opt.n_pages * opt.page_tokens * cfg.kv_dim();
        b_k = mc.buffer(kv_floats * 4);
        b_v = mc.buffer(kv_floats * 4);
        b_pt = mc.buffer((size_t)opt.max_seqs * opt.n_pages * 4);
        pager = std::make_unique<KVPager>(opt.max_seqs, opt.n_pages, opt.page_tokens, (uint32_t*)b_pt.contents);

        // RoPE table, computed the way the reference does (float32 angle = pos * inv_freq).
        const int half = cfg.head_dim / 2;
        rope = mc.buffer((size_t)opt.max_positions * half * 8);
        float* rt = (float*)rope.contents;
        for (int i = 0; i < half; i++) {
            const float inv_freq = 1.0f / std::pow(cfg.rope_theta, (float)(2 * i) / (float)cfg.head_dim);
            for (int p = 0; p < opt.max_positions; p++) {
                const float ang = (float)p * inv_freq;
                rt[((size_t)p * half + i) * 2] = std::cos(ang);
                rt[((size_t)p * half + i) * 2 + 1] = std::sin(ang);
            }
        }
    }

    // --------------------------------------------------------- encoding ----

    void linear(id<MTLComputeCommandEncoder> e, const Linear& L, uint32_t rows, id<MTLBuffer> x, uint32_t x_stride,
                id<MTLBuffer> y, uint32_t y_stride, uint32_t n, int mode) {
        // The matrix units work on 64-token tiles, so they only pay off once a
        // pass carries enough tokens to fill a good part of one; below that the
        // SIMD-group dot-product kernels are faster.
        const bool tensor = n >= kTensorMinTokens;
        const uint32_t nt = n > 1 ? 4 : 1;
        const uint32_t nr = 2;
        MMArgs a = {n, rows, L.K, x_stride, y_stride, 0, (uint32_t)mode, L.bias != nil};
        [e setComputePipelineState:tensor ? (mode == kSwiGLU ? p_mm_tensor_swiglu : p_mm_tensor) : p_mm.at(nr * 16 + nt)];
        [e setBuffer:L.w offset:0 atIndex:0];
        [e setBuffer:L.bias ? L.bias : b_dummy offset:0 atIndex:1];
        [e setBuffer:x offset:0 atIndex:2];
        [e setBuffer:y offset:0 atIndex:3];
        [e setBytes:&a length:sizeof a atIndex:4];
        if (q4()) [e setBuffer:L.scales offset:0 atIndex:5];
        if (tensor)  // tile = 64 tokens x 64 rows on 4 SIMD-groups (see model_tensor.metal)
            [e dispatchThreadgroups:MTLSizeMake((rows + 63) / 64, (n + 63) / 64, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        else
            [e dispatchThreadgroups:MTLSizeMake((rows + nr - 1) / nr, (n + nt - 1) / nt, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }

    // SIMD-groups per attention threadgroup. With a handful of tokens the GPU
    // is mostly idle, so use the widest group (1024 threads walk the keys in
    // parallel); it is never slower, and ~30% faster per token at 1k context.
    // When many tokens are in flight the GPU is already saturated and a
    // single SIMD-group per (token, head) launches fastest.
    static int attention_width(uint32_t n_tok) { return n_tok >= 8 ? 1 : 32; }

    void norm(id<MTLComputeCommandEncoder> e, id<MTLBuffer> x, id<MTLBuffer> y, uint32_t n, uint32_t which, bool gather) {
        NormArgs a = {n, (uint32_t)cfg.hidden, cfg.rms_eps, which * (uint32_t)cfg.hidden, gather};
        [e setComputePipelineState:p_norm];
        [e setBuffer:x offset:0 atIndex:0];
        [e setBuffer:norm_w offset:0 atIndex:1];
        [e setBuffer:y offset:0 atIndex:2];
        [e setBytes:&a length:sizeof a atIndex:3];
        [e setBuffer:b_rows offset:0 atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }

    const float* forward(const std::vector<TokenIn>& toks, const std::vector<int>& want) {
        const uint32_t n = (uint32_t)toks.size(), nw = (uint32_t)want.size();
        if (n == 0) return (const float*)b_logits.contents;
        if ((int)n > opt.max_batch_tokens) die("forward: %u tokens exceeds max_batch_tokens", n);
        if ((int)nw > opt.max_logit_rows) die("forward: %u logit rows exceeds max_logit_rows", nw);

        int32_t* ids = (int32_t*)b_ids.contents;
        TokenMeta* meta = (TokenMeta*)b_meta.contents;
        for (uint32_t i = 0; i < n; i++) {
            const int pos = pager->append(toks[i].seq, 1);
            if (pos < 0) die("forward: KV cache pool exhausted");
            if (pos >= opt.max_positions) die("forward: sequence exceeds max_positions");
            ids[i] = toks[i].token;
            meta[i] = {(uint32_t)toks[i].seq, (uint32_t)pos};
        }
        uint32_t* rows = (uint32_t*)b_rows.contents;
        for (uint32_t i = 0; i < nw; i++) rows[i] = (uint32_t)want[i];

        const uint32_t d = cfg.hidden, kvd = cfg.kv_dim(), qkv_stride = d + 2 * kvd;

        @autoreleasepool {
            Pass pass(mc, opt.profile, &stage_seconds);
            id<MTLComputeCommandEncoder> __strong& e = pass.e;

            // token embedding -> X
            EmbedArgs ea = {n, d};
            [e setComputePipelineState:p_embed];
            [e setBuffer:embed.w offset:0 atIndex:0];
            if (q4()) {
                [e setBuffer:embed.scales offset:0 atIndex:1];
                [e setBuffer:b_ids offset:0 atIndex:2];
                [e setBuffer:b_x offset:0 atIndex:3];
                [e setBytes:&ea length:sizeof ea atIndex:4];
                [e dispatchThreadgroups:MTLSizeMake((d / 8 + 31) / 32, n, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            } else {
                [e setBuffer:b_ids offset:0 atIndex:1];
                [e setBuffer:b_x offset:0 atIndex:2];
                [e setBytes:&ea length:sizeof ea atIndex:3];
                [e dispatchThreadgroups:MTLSizeMake((d + 31) / 32, n, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }

            pass.stage("embed");
            KVArgs ka = {n, (uint32_t)cfg.n_heads, (uint32_t)cfg.n_kv_heads, (uint32_t)cfg.head_dim, qkv_stride, 0,
                         (uint32_t)opt.n_pages, (uint32_t)__builtin_ctz(opt.page_tokens), (uint32_t)pager->pages_per_seq(),
                         1.0f / std::sqrt((float)cfg.head_dim), d};
            const uint32_t half = cfg.head_dim / 2, heads_all = cfg.n_heads + cfg.n_kv_heads;

            for (int l = 0; l < cfg.n_layers; l++) {
                const Layer& ly = layers[l];
                ka.layer = (uint32_t)l;

                // --- attention block
                norm(e, b_x, b_h, n, 2 * l, false);
                pass.stage("rmsnorm");
                linear(e, ly.qkv, ly.qkv.rows, b_h, d, b_qkv, qkv_stride, n, kStore);
                pass.stage("qkv proj");

                [e setComputePipelineState:p_rope];
                [e setBuffer:b_qkv offset:0 atIndex:0];
                [e setBuffer:rope offset:0 atIndex:1];
                [e setBuffer:b_k offset:0 atIndex:2];
                [e setBuffer:b_v offset:0 atIndex:3];
                [e setBuffer:b_meta offset:0 atIndex:4];
                [e setBuffer:b_pt offset:0 atIndex:5];
                [e setBytes:&ka length:sizeof ka atIndex:6];
                [e dispatchThreadgroups:MTLSizeMake((half + 31) / 32, heads_all, n) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                pass.stage("rope + kv write");

                const int att_sgs = attention_width(n);
                [e setComputePipelineState:p_attn.at(att_sgs)];
                [e setBuffer:b_qkv offset:0 atIndex:0];
                [e setBuffer:b_k offset:0 atIndex:1];
                [e setBuffer:b_v offset:0 atIndex:2];
                [e setBuffer:b_att offset:0 atIndex:3];
                [e setBuffer:b_meta offset:0 atIndex:4];
                [e setBuffer:b_pt offset:0 atIndex:5];
                [e setBytes:&ka length:sizeof ka atIndex:6];
                [e dispatchThreadgroups:MTLSizeMake(cfg.n_heads, n, 1) threadsPerThreadgroup:MTLSizeMake(att_sgs * 32, 1, 1)];
                pass.stage("attention");

                linear(e, ly.o, ly.o.rows, b_att, d, b_x, d, n, kAccumulate);
                pass.stage("o proj");

                // --- SwiGLU MLP block
                norm(e, b_x, b_h, n, 2 * l + 1, false);
                pass.stage("rmsnorm");
                linear(e, ly.gate_up, cfg.inter, b_h, d, b_act, cfg.inter, n, kSwiGLU);
                pass.stage("swiglu gate/up");
                linear(e, ly.down, ly.down.rows, b_act, cfg.inter, b_x, d, n, kAccumulate);
                pass.stage("mlp down");
            }

            if (nw) {
                norm(e, b_x, b_hn, nw, 2 * cfg.n_layers, true);
                linear(e, head, head.rows, b_hn, d, b_logits, cfg.vocab_size, nw, kStore);
                pass.stage("lm head");
            }
            pass.flush();
            gpu_seconds = pass.gpu;
        }
        return (const float*)b_logits.contents;
    }
};

Engine::Engine(const std::string& dir, const EngineOptions& opts) : impl_(new Impl(dir, opts)) {}
Engine::~Engine() = default;
const ModelConfig& Engine::config() const { return impl_->cfg; }
const EngineOptions& Engine::options() const { return impl_->opt; }
int Engine::seq_create() { return impl_->pager->create(); }
void Engine::seq_free(int seq) { impl_->pager->release(seq); }
int Engine::seq_len(int seq) const { return impl_->pager->length(seq); }
void Engine::seq_truncate(int seq, int len) { impl_->pager->truncate(seq, len); }
int Engine::free_pages() const { return impl_->pager->free_pages(); }
int Engine::free_token_slots(int seq) const {
    const KVPager& p = *impl_->pager;
    const int in_page = p.pages_of(seq) * p.page_tokens() - p.length(seq);
    return std::min(in_page + p.free_pages() * p.page_tokens(), impl_->opt.max_positions - p.length(seq));
}
const float* Engine::forward(const std::vector<TokenIn>& toks, const std::vector<int>& want) {
    return impl_->forward(toks, want);
}
double Engine::last_gpu_seconds() const { return impl_->gpu_seconds; }
size_t Engine::weight_bytes() const { return impl_->weight_bytes; }
std::map<std::string, double>& Engine::stage_seconds() { return impl_->stage_seconds; }
