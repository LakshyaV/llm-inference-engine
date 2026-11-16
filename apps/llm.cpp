// Command-line front end.
//
//   llm generate -m <model_dir> -p "prompt" [-n 128] [--chat] [--f32] [--temp 0.7 --top-k 40 --top-p 0.95]
//   llm bench    -m <model_dir> [-n 128] [--prompt-tokens 512] [--ctx 1] [--profile] [--f32]
#include <cstdio>
#include <cstring>
#include <map>
#include <string>

#include "../src/engine.h"
#include "../src/sampler.h"
#include "../src/tokenizer.h"
#include "../src/util.h"

struct Args {
    std::string cmd;
    std::map<std::string, std::string> kv;
    bool flag(const std::string& k) const { return kv.count(k) != 0; }
    std::string str(const std::string& k, const std::string& d = "") const { return kv.count(k) ? kv.at(k) : d; }
    int num(const std::string& k, int d) const { return kv.count(k) ? atoi(kv.at(k).c_str()) : d; }
    double real(const std::string& k, double d) const { return kv.count(k) ? atof(kv.at(k).c_str()) : d; }
};

static Args parse(int argc, char** argv) {
    Args a;
    if (argc > 1) a.cmd = argv[1];
    static const std::map<std::string, std::string> alias = {{"-m", "--model"}, {"-p", "--prompt"}, {"-n", "--max-tokens"}};
    for (int i = 2; i < argc; i++) {
        std::string k = argv[i];
        if (alias.count(k)) k = alias.at(k);
        if (k.rfind("--", 0) != 0) die("unexpected argument %s", argv[i]);
        const bool has_val = i + 1 < argc && (argv[i + 1][0] != '-' || !argv[i + 1][1] || isdigit(argv[i + 1][1]));
        a.kv[k.substr(2)] = has_val ? argv[++i] : "1";
    }
    return a;
}

static std::string chat_prompt(const std::string& user) {
    return "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + user +
           "<|im_end|>\n<|im_start|>assistant\n";
}

static EngineOptions engine_options(const Args& a) {
    EngineOptions o;
    o.format = a.flag("f32") ? WeightFormat::F32 : WeightFormat::Q4;
    o.verbose = true;
    o.profile = a.flag("profile");
    return o;
}

static bool is_eos(const ModelConfig& c, int id) {
    for (int e : c.eos_ids)
        if (e == id) return true;
    return false;
}

static int cmd_generate(const Args& a) {
    const std::string dir = a.str("model");
    Tokenizer tok(dir + "/tokenizer.json");
    Engine eng(dir, engine_options(a));
    SamplerConfig sc;
    sc.temperature = (float)a.real("temp", 0.0);
    sc.top_k = a.num("top-k", 0);
    sc.top_p = (float)a.real("top-p", 1.0);
    sc.seed = (uint64_t)a.num("seed", 0);
    Sampler sampler(sc);

    std::string prompt = a.str("prompt", "Hello");
    if (a.flag("chat")) prompt = chat_prompt(prompt);
    std::vector<int> ids = tok.encode(prompt);
    const int vocab = eng.config().vocab_size, seq = eng.seq_create();

    std::vector<TokenIn> toks;
    for (int id : ids) toks.push_back({id, seq});
    double t0 = now_sec();
    const float* logits = eng.forward(toks, {(int)toks.size() - 1});
    double t_prefill = now_sec() - t0;

    if (!a.flag("chat")) fputs(prompt.c_str(), stdout);
    int n_gen = 0;
    const int max_tokens = a.num("max-tokens", 128);
    t0 = now_sec();
    while (n_gen < max_tokens) {
        const int next = sampler.sample(logits, vocab);
        if (is_eos(eng.config(), next)) break;
        fputs(tok.token_bytes(next).c_str(), stdout);
        fflush(stdout);
        n_gen++;
        logits = eng.forward({{next, seq}}, {0});
    }
    double t_gen = now_sec() - t0;
    fprintf(stderr, "\n\nprefill: %zu tokens, %.1f tok/s | decode: %d tokens, %.1f tok/s\n", ids.size(),
            ids.size() / t_prefill, n_gen, n_gen / t_gen);
    return 0;
}

// Synthetic throughput benchmark with the same shape as `llama-bench -p N -n M`:
// prefill speed on an N-token prompt, and decode speed generating M tokens
// from an empty context. --ctx C measures decode after a C-token prompt.
static int cmd_bench(const Args& a) {
    const std::string dir = a.str("model");
    Engine eng(dir, engine_options(a));
    const int vocab = eng.config().vocab_size;
    const int n_prompt = a.num("prompt-tokens", 512), n_gen = a.num("max-tokens", 128), reps = a.num("reps", 3);
    const int ctx = a.num("ctx", 1);
    std::vector<TokenIn> prompt;
    for (int i = 0; i < std::max(n_prompt, ctx); i++) prompt.push_back({(i * 7919 + 13) % 50000, 0});

    double best_pp = 0, best_tg = 0;
    std::map<std::string, double> prefill_stages;
    for (int r = 0; r < reps + 1; r++) {  // first rep is warm-up
        eng.stage_seconds().clear();
        int seq = eng.seq_create();
        for (auto& t : prompt) t.seq = seq;
        double t0 = now_sec();
        if (n_prompt) eng.forward(std::vector<TokenIn>(prompt.begin(), prompt.begin() + n_prompt), {n_prompt - 1});
        const double pp = n_prompt / (now_sec() - t0);
        eng.seq_free(seq);
        prefill_stages = eng.stage_seconds();

        seq = eng.seq_create();
        int next = argmax(eng.forward(std::vector<TokenIn>(prompt.begin(), prompt.begin() + ctx), {ctx - 1}), vocab);
        if (a.flag("profile")) eng.stage_seconds().clear();
        t0 = now_sec();
        for (int i = 0; i < n_gen; i++) next = argmax(eng.forward({{next, seq}}, {0}), vocab);
        const double tg = n_gen / (now_sec() - t0);
        eng.seq_free(seq);
        if (r > 0) { best_pp = std::max(best_pp, pp); best_tg = std::max(best_tg, tg); }
    }
    printf("weights: %s (%.2f GB)\n", a.flag("f32") ? "fp32" : "int4", eng.weight_bytes() / 1e9);
    printf("prefill (%d tokens): %8.1f tok/s\n", n_prompt, best_pp);
    printf("decode  (%d tokens after %d): %8.1f tok/s\n", n_gen, ctx, best_tg);
    if (a.flag("profile")) {
        auto table = [](const char* title, const std::map<std::string, double>& st, const char* unit, double div) {
            double total = 0;
            for (auto& [k, v] : st) total += v;
            printf("%s GPU time by stage (one command buffer per stage):\n", title);
            for (auto& [k, v] : st) printf("  %-16s %8.3f %s  %5.1f%%\n", k.c_str(), 1e3 * v / div, unit, 100 * v / total);
        };
        if (n_prompt) table("prefill", prefill_stages, "ms", 1);
        table("decode", eng.stage_seconds(), "ms/token", n_gen);
    }
    return 0;
}

int main(int argc, char** argv) {
    try {
        Args a = parse(argc, argv);
        if (a.cmd == "generate") return cmd_generate(a);
        if (a.cmd == "bench") return cmd_bench(a);
        fprintf(stderr, "usage: llm <generate|bench> -m <model_dir> [options]\n");
        return 2;
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
