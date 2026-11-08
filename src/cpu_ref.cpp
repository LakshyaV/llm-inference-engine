#include "cpu_ref.h"

#include <Accelerate/Accelerate.h>

#include <cmath>

#include "util.h"

CpuReference::CpuReference(const std::string& dir) : cfg_(ModelConfig::load(dir)), st_(SafeTensors::model_files(dir)) {}

std::vector<float> CpuReference::load(const std::string& name) const {
    const TensorView& t = st_.get(name);
    std::vector<float> v(t.numel());
    SafeTensors::to_f32(t, v.data());
    return v;
}

// y[T, out] = x[T, in] * W[out, in]^T (+ bias)
static void linear(const float* x, int T, int in, const std::vector<float>& W, int out, const float* bias, float* y) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, out, in, 1.0f, x, in, W.data(), in, 0.0f, y, out);
    if (bias)
        for (int t = 0; t < T; t++)
            for (int o = 0; o < out; o++) y[(size_t)t * out + o] += bias[o];
}

static void rmsnorm(const float* x, int T, int d, const std::vector<float>& w, float eps, float* y) {
    for (int t = 0; t < T; t++) {
        double ss = 0;
        for (int i = 0; i < d; i++) ss += (double)x[t * d + i] * x[t * d + i];
        float inv = 1.0f / std::sqrt((float)(ss / d) + eps);
        for (int i = 0; i < d; i++) y[t * d + i] = x[t * d + i] * inv * w[i];
    }
}

std::vector<float> CpuReference::forward(const std::vector<int>& tokens) const {
    const int T = (int)tokens.size(), d = cfg_.hidden, hd = cfg_.head_dim, H = cfg_.n_heads, KH = cfg_.n_kv_heads;
    const int kvd = cfg_.kv_dim(), half = hd / 2;

    std::vector<float> emb = load("model.embed_tokens.weight");
    std::vector<float> x((size_t)T * d);
    for (int t = 0; t < T; t++) memcpy(&x[(size_t)t * d], &emb[(size_t)tokens[t] * d], d * sizeof(float));

    // RoPE tables
    std::vector<float> rc((size_t)T * half), rs((size_t)T * half);
    for (int t = 0; t < T; t++)
        for (int i = 0; i < half; i++) {
            double ang = t * std::pow((double)cfg_.rope_theta, -2.0 * i / hd);
            rc[t * half + i] = (float)std::cos(ang);
            rs[t * half + i] = (float)std::sin(ang);
        }
    auto rope = [&](float* v, int heads) {
        for (int t = 0; t < T; t++)
            for (int h = 0; h < heads; h++) {
                float* p = v + ((size_t)t * heads + h) * hd;
                for (int i = 0; i < half; i++) {
                    float a = p[i], b = p[i + half], c = rc[t * half + i], s = rs[t * half + i];
                    p[i] = a * c - b * s;
                    p[i + half] = b * c + a * s;
                }
            }
    };

    std::vector<float> h((size_t)T * d), q((size_t)T * d), k((size_t)T * kvd), v((size_t)T * kvd), att((size_t)T * d);
    std::vector<float> o((size_t)T * d), g((size_t)T * cfg_.inter), u((size_t)T * cfg_.inter), sc(T);
    const float scale = 1.0f / std::sqrt((float)hd);

    for (int l = 0; l < cfg_.n_layers; l++) {
        std::string p = "model.layers." + std::to_string(l) + ".";
        auto bias = [&](const std::string& n) -> std::vector<float> {
            return st_.has(p + n) ? load(p + n) : std::vector<float>();
        };
        rmsnorm(x.data(), T, d, load(p + "input_layernorm.weight"), cfg_.rms_eps, h.data());
        auto qb = bias("self_attn.q_proj.bias"), kb = bias("self_attn.k_proj.bias"), vb = bias("self_attn.v_proj.bias");
        linear(h.data(), T, d, load(p + "self_attn.q_proj.weight"), d, qb.empty() ? nullptr : qb.data(), q.data());
        linear(h.data(), T, d, load(p + "self_attn.k_proj.weight"), kvd, kb.empty() ? nullptr : kb.data(), k.data());
        linear(h.data(), T, d, load(p + "self_attn.v_proj.weight"), kvd, vb.empty() ? nullptr : vb.data(), v.data());
        rope(q.data(), H);
        rope(k.data(), KH);

        for (int t = 0; t < T; t++)
            for (int hh = 0; hh < H; hh++) {
                int kh = hh / (H / KH);
                const float* qv = &q[((size_t)t * H + hh) * hd];
                float mx = -1e30f;
                for (int j = 0; j <= t; j++) {
                    sc[j] = cblas_sdot(hd, qv, 1, &k[((size_t)j * KH + kh) * hd], 1) * scale;
                    mx = std::max(mx, sc[j]);
                }
                float sum = 0;
                for (int j = 0; j <= t; j++) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
                float* out = &att[((size_t)t * H + hh) * hd];
                std::fill(out, out + hd, 0.0f);
                for (int j = 0; j <= t; j++) cblas_saxpy(hd, sc[j] / sum, &v[((size_t)j * KH + kh) * hd], 1, out, 1);
            }
        linear(att.data(), T, d, load(p + "self_attn.o_proj.weight"), d, nullptr, o.data());
        for (size_t i = 0; i < x.size(); i++) x[i] += o[i];

        rmsnorm(x.data(), T, d, load(p + "post_attention_layernorm.weight"), cfg_.rms_eps, h.data());
        linear(h.data(), T, d, load(p + "mlp.gate_proj.weight"), cfg_.inter, nullptr, g.data());
        linear(h.data(), T, d, load(p + "mlp.up_proj.weight"), cfg_.inter, nullptr, u.data());
        for (size_t i = 0; i < g.size(); i++) g[i] = g[i] / (1.0f + std::exp(-g[i])) * u[i];  // SwiGLU
        linear(g.data(), T, cfg_.inter, load(p + "mlp.down_proj.weight"), d, nullptr, o.data());
        for (size_t i = 0; i < x.size(); i++) x[i] += o[i];
    }

    rmsnorm(x.data(), T, d, load("model.norm.weight"), cfg_.rms_eps, h.data());
    std::vector<float> logits((size_t)T * cfg_.vocab_size);
    if (st_.has("lm_head.weight")) linear(h.data(), T, d, load("lm_head.weight"), cfg_.vocab_size, nullptr, logits.data());
    else linear(h.data(), T, d, emb, cfg_.vocab_size, nullptr, logits.data());
    return logits;
}
