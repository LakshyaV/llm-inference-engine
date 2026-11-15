// Logit parity against Hugging Face transformers (float32).
//
//   test_logits <model_dir> <logits_*.bin> [--cpu]
//
// For every reference prompt:
//   1. one-shot: all tokens in a single forward, logits at every position
//   2. incremental: half the prompt as prefill, the rest one token at a time
//      through the KV cache; the final logits must agree with (1)
// --cpu additionally checks the plain C++ reference implementation.
#include <cmath>
#include <cstdio>
#include <cstring>

#include "../src/cpu_ref.h"
#include "../src/engine.h"
#include "../src/util.h"

struct Case {
    std::vector<int> ids;
    std::vector<float> logits;
};

static std::vector<Case> load_cases(const std::string& path, int& vocab) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) die("cannot open %s (run tools/reference.py first)", path.c_str());
    char magic[4];
    uint32_t n, v;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "LGT1", 4) || fread(&n, 4, 1, f) != 1 || fread(&v, 4, 1, f) != 1)
        die("bad reference file");
    vocab = (int)v;
    std::vector<Case> cases(n);
    for (auto& c : cases) {
        uint32_t t;
        if (fread(&t, 4, 1, f) != 1) die("bad reference file");
        c.ids.resize(t);
        c.logits.resize((size_t)t * v);
        if (fread(c.ids.data(), 4, t, f) != t || fread(c.logits.data(), 4, c.logits.size(), f) != c.logits.size())
            die("bad reference file");
    }
    fclose(f);
    return cases;
}

struct Diff {
    double max_abs = 0;
    int argmax_mismatch = 0;
};

static Diff compare(const float* got, const float* want, int rows, int vocab) {
    Diff d;
    for (int r = 0; r < rows; r++) {
        int ag = 0, aw = 0;
        for (int i = 0; i < vocab; i++) {
            const size_t k = (size_t)r * vocab + i;
            d.max_abs = std::max(d.max_abs, (double)std::fabs(got[k] - want[k]));
            if (got[k] > got[(size_t)r * vocab + ag]) ag = i;
            if (want[k] > want[(size_t)r * vocab + aw]) aw = i;
        }
        d.argmax_mismatch += ag != aw;
    }
    return d;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model_dir> <logits.bin> [--cpu]\n", argv[0]);
        return 2;
    }
    const bool cpu = argc > 3 && !strcmp(argv[3], "--cpu");
    const double tol = 1e-2;  // max |logit - reference|; logits span roughly +-30, fp32 rounding differs per backend
    try {
        int vocab = 0;
        std::vector<Case> cases = load_cases(argv[2], vocab);
        EngineOptions opt;
        opt.format = WeightFormat::F32;
        opt.max_logit_rows = 160;
        Engine eng(argv[1], opt);
        int fail = 0;

        for (size_t ci = 0; ci < cases.size(); ci++) {
            const Case& c = cases[ci];
            const int T = (int)c.ids.size();

            int seq = eng.seq_create();
            std::vector<TokenIn> toks;
            std::vector<int> want;
            for (int t = 0; t < T; t++) { toks.push_back({c.ids[t], seq}); want.push_back(t); }
            Diff one = compare(eng.forward(toks, want), c.logits.data(), T, vocab);
            eng.seq_free(seq);

            seq = eng.seq_create();
            const int pre = T / 2;
            toks.assign(toks.begin(), toks.begin() + pre);
            for (auto& t : toks) t.seq = seq;
            eng.forward(toks, {});
            const float* last = nullptr;
            for (int t = pre; t < T; t++) last = eng.forward({{c.ids[t], seq}}, {0});
            Diff inc = compare(last, &c.logits[(size_t)(T - 1) * vocab], 1, vocab);
            eng.seq_free(seq);

            const bool ok = one.max_abs < tol && inc.max_abs < tol && !one.argmax_mismatch && !inc.argmax_mismatch;
            fail += !ok;
            printf("case %zu (%3d tokens): one-shot max|d|=%.2e argmax-miss=%d | incremental max|d|=%.2e  %s\n", ci, T,
                   one.max_abs, one.argmax_mismatch, inc.max_abs, ok ? "ok" : "FAIL");

            if (cpu) {
                CpuReference ref(argv[1]);
                Diff cd = compare(ref.forward(c.ids).data(), c.logits.data(), T, vocab);
                printf("         cpu reference: max|d|=%.2e argmax-miss=%d\n", cd.max_abs, cd.argmax_mismatch);
                fail += !(cd.max_abs < tol);
            }
        }
        printf("logits: %zu cases, %d failed (tolerance %.0e)\n", cases.size(), fail, tol);
        return fail ? 1 : 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
