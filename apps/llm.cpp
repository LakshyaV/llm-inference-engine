// Command-line front end.
//
//   llm generate -m <model_dir> -p "prompt" [-n 128] [--chat] [--f32] [--temp 0.7 --top-k 40 --top-p 0.95]
//   llm bench    -m <model_dir> [-n 128] [--prompt-tokens 512] [--ctx 1] [--profile] [--f32]
//   llm batch    -m <model_dir> [--requests 16] [--max-active 8] [-n 96] [--arrival-ms 0]
//
// Common: --f32 (exact weights), --tensor-min N (token count at which linear
// layers switch to the matrix-unit kernels; default 16).
//   llm eval     -m <model_dir> --file text.txt [--ctx 256] [--max-chunks 8]
//   llm spec     -m <target_dir> --draft <draft_dir> [-p "prompt"] [--gamma 4] [-n 128] [--temp 0]
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <map>
#include <string>

#include "../src/engine.h"
#include "../src/sampler.h"
#include "../src/scheduler.h"
#include "../src/speculative.h"
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
    o.tensor_min_tokens = a.num("tensor-min", o.tensor_min_tokens);
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

static const char* kPrompts[] = {
    "Explain how a hash map works.",
    "Write a short poem about the ocean.",
    "What causes the seasons on Earth?",
    "Give me three tips for writing clean code.",
    "Summarize the plot of Romeo and Juliet in a paragraph.",
    "How does a transformer neural network process text?",
    "What is the difference between TCP and UDP?",
    "Describe how photosynthesis works.",
    "Write a Python function that reverses a linked list.",
    "Why is the sky dark at night?",
    "Explain the rules of chess to a beginner.",
    "What were the main causes of the First World War?",
    "How do vaccines train the immune system?",
    "Explain what a derivative is in calculus.",
    "Describe the water cycle.",
    "What is the purpose of an operating system kernel?",
};

// Continuous batching demo: the same requests served one at a time and then
// concurrently, optionally arriving over time (--arrival-ms).
static int cmd_batch(const Args& a) {
    const std::string dir = a.str("model");
    Tokenizer tok(dir + "/tokenizer.json");
    EngineOptions eo = engine_options(a);
    eo.max_seqs = 64;
    eo.n_pages = 1024;
    Engine eng(dir, eo);
    const int n_req = a.num("requests", 16), max_tokens = a.num("max-tokens", 96);
    const double arrival = a.real("arrival-ms", 0) / 1e3;

    auto serve = [&](int max_active, bool print) {
        SchedulerConfig sc;
        sc.max_active = max_active;
        Scheduler sched(eng, sc);
        const double t0 = now_sec();
        int submitted = 0;
        while (true) {
            // Requests arrive on a fixed schedule, independent of how busy the engine is.
            while (submitted < n_req && now_sec() - t0 >= submitted * arrival) {
                Request r;
                r.prompt = tok.encode(chat_prompt(kPrompts[submitted % (sizeof kPrompts / sizeof *kPrompts)]));
                r.max_tokens = max_tokens;
                sched.submit(std::move(r));
                submitted++;
            }
            if (!sched.step()) {
                if (submitted == n_req) break;
                usleep(200);
            }
        }
        const double wall = now_sec() - t0;
        long gen = 0;
        double ttft = 0, latency = 0;
        for (int i = 0; i < n_req; i++) {
            const RequestResult& r = sched.result(i);
            gen += (long)r.tokens.size();
            ttft += r.t_first - r.t_submit;
            latency += r.t_done - r.t_submit;
        }
        printf("max_active=%-3d %6ld tokens in %6.2fs  %7.1f tok/s | %4d passes | mean TTFT %6.0f ms | mean latency %6.0f ms\n",
               max_active, gen, wall, gen / wall, sched.steps(), 1e3 * ttft / n_req, 1e3 * latency / n_req);
        if (print)
            for (int i = 0; i < std::min(n_req, 2); i++)
                printf("--- request %d ---\n%s\n", i, tok.decode(sched.result(i).tokens).c_str());
    };

    printf("%d requests, up to %d new tokens each, arrival every %.0f ms\n", n_req, max_tokens, arrival * 1e3);
    serve(1, false);  // warm-up
    serve(1, false);
    for (int m : {2, 4, 8, 16, 32})
        if (m <= std::max(2, a.num("max-active", 32))) serve(m, false);
    if (a.flag("print")) serve(a.num("max-active", 8), true);
    return 0;
}

// int4 quality check: perplexity of the fp32 and int4 engines on the same
// text, plus how often they agree on the top token and the mean KL divergence
// between their next-token distributions.
//
// Protocol (the one llama.cpp's perplexity tool uses): the text is cut into
// chunks of --ctx tokens and only the second half of each chunk is scored, so
// every scored token has at least ctx/2 tokens of context.
static int cmd_eval(const Args& a) {
    const std::string dir = a.str("model");
    if (!a.flag("file")) die("eval: --file <text file> is required");
    Tokenizer tok(dir + "/tokenizer.json");
    const std::vector<int> ids = tok.encode(read_file(a.str("file")));
    const int ctx = a.num("ctx", 256), first = ctx / 2;
    const int n_chunks = std::min((int)ids.size() / ctx, a.num("max-chunks", 8));
    if (n_chunks < 1) die("eval: need at least %d tokens of text, got %zu", ctx, ids.size());

    EngineOptions eo = engine_options(a);
    eo.max_logit_rows = ctx - first;
    eo.max_batch_tokens = std::max(eo.max_batch_tokens, ctx);
    eo.format = WeightFormat::F32;
    Engine ref(dir, eo);
    eo.format = WeightFormat::Q4;
    Engine q4(dir, eo);
    const int V = ref.config().vocab_size;

    // log-softmax of one row, in double precision
    auto log_softmax = [V](const float* row, std::vector<double>& out) {
        double mx = row[0];
        for (int i = 1; i < V; i++) mx = std::max(mx, (double)row[i]);
        double sum = 0;
        for (int i = 0; i < V; i++) sum += std::exp(row[i] - mx);
        const double lse = mx + std::log(sum);
        out.resize(V);
        for (int i = 0; i < V; i++) out[i] = row[i] - lse;
    };

    double nll_ref = 0, nll_q4 = 0, kl = 0;
    long scored = 0, agree = 0;
    std::vector<double> lp_ref, lp_q4;
    for (int c = 0; c < n_chunks; c++) {
        std::vector<TokenIn> toks;
        std::vector<int> want;
        for (int t = 0; t < ctx; t++) toks.push_back({ids[(size_t)c * ctx + t], 0});
        for (int t = first; t < ctx - 1; t++) want.push_back(t);  // row for position t predicts token t + 1

        const int s_ref = ref.seq_create(), s_q4 = q4.seq_create();
        for (auto& t : toks) t.seq = s_ref;
        const float* l_ref = ref.forward(toks, want);
        for (auto& t : toks) t.seq = s_q4;
        const float* l_q4 = q4.forward(toks, want);

        for (size_t r = 0; r < want.size(); r++) {
            const int target = ids[(size_t)c * ctx + want[r] + 1];
            log_softmax(l_ref + r * (size_t)V, lp_ref);
            log_softmax(l_q4 + r * (size_t)V, lp_q4);
            nll_ref -= lp_ref[target];
            nll_q4 -= lp_q4[target];
            for (int i = 0; i < V; i++) kl += std::exp(lp_ref[i]) * (lp_ref[i] - lp_q4[i]);
            agree += argmax(l_ref + r * (size_t)V, V) == argmax(l_q4 + r * (size_t)V, V);
            scored++;
        }
        ref.seq_free(s_ref);
        q4.seq_free(s_q4);
    }
    printf("%ld tokens scored (%d chunks of %d, second half of each)\n", scored, n_chunks, ctx);
    printf("perplexity  fp32: %8.4f\n", std::exp(nll_ref / scored));
    printf("perplexity  int4: %8.4f  (%+.2f%%)\n", std::exp(nll_q4 / scored),
           100.0 * (std::exp(nll_q4 / scored) / std::exp(nll_ref / scored) - 1.0));
    printf("top-1 agreement:  %7.2f%%\n", 100.0 * agree / scored);
    printf("mean KL(fp32 || int4): %.5f nats\n", kl / scored);
    return 0;
}

static int cmd_spec(const Args& a) {
    const std::string dir = a.str("model"), ddir = a.str("draft");
    if (ddir.empty()) die("spec: --draft <model_dir> is required");
    Tokenizer tok(dir + "/tokenizer.json");
    Engine target(dir, engine_options(a));
    Engine draft(ddir, engine_options(a));
    SamplerConfig sc;
    sc.temperature = (float)a.real("temp", 0.0);
    sc.top_k = a.num("top-k", 0);
    sc.top_p = (float)a.real("top-p", 1.0);
    sc.seed = (uint64_t)a.num("seed", 0);
    const int max_tokens = a.num("max-tokens", 128), vocab = target.config().vocab_size;

    std::vector<std::string> prompts;
    if (a.flag("prompt")) prompts.push_back(a.str("prompt"));
    else for (int i = 0; i < 8; i++) prompts.push_back(kPrompts[i]);

    auto plain = [&](const std::vector<int>& ids) {
        Sampler sampler(sc);
        const int seq = target.seq_create();
        std::vector<TokenIn> toks;
        for (int id : ids) toks.push_back({id, seq});
        const float* logits = target.forward(toks, {(int)toks.size() - 1});
        std::vector<int> out;
        while ((int)out.size() < max_tokens) {
            const int next = sampler.sample(logits, vocab);
            if (is_eos(target.config(), next)) break;
            out.push_back(next);
            logits = target.forward({{next, seq}}, {0});
        }
        target.seq_free(seq);
        return out;
    };

    for (int gamma : a.flag("gamma") ? std::vector<int>{a.num("gamma", 4)} : std::vector<int>{2, 4, 6, 8}) {
        SpeculativeDecoder spec(target, draft, gamma, sc);
        double t_plain = 0, t_spec = 0;
        long n_plain = 0, n_spec = 0, drafted = 0, accepted = 0, passes = 0;
        int identical = 0;
        for (size_t i = 0; i < prompts.size(); i++) {
            const std::vector<int> ids = tok.encode(chat_prompt(prompts[i]));
            if (i == 0) { plain(ids); spec.generate(ids, 8); }  // warm-up
            double t0 = now_sec();
            const std::vector<int> ref = plain(ids);
            t_plain += now_sec() - t0;
            t0 = now_sec();
            const std::vector<int> out = spec.generate(ids, max_tokens);
            t_spec += now_sec() - t0;
            n_plain += (long)ref.size();
            n_spec += (long)out.size();
            drafted += spec.stats().drafted;
            accepted += spec.stats().accepted;
            passes += spec.stats().target_passes;
            identical += out == ref;
            if (a.flag("print") && i == 0) printf("%s\n", tok.decode(out).c_str());
        }
        printf("gamma=%d  plain %6.1f tok/s | speculative %6.1f tok/s (%.2fx) | acceptance %4.1f%% | %.2f tokens per target pass",
               gamma, n_plain / t_plain, n_spec / t_spec, (n_spec / t_spec) / (n_plain / t_plain),
               100.0 * accepted / std::max(1L, drafted), (double)n_spec / std::max(1L, passes));
        if (sc.temperature <= 0) printf(" | identical to plain greedy: %d/%zu", identical, prompts.size());
        printf("\n");
    }
    return 0;
}

int main(int argc, char** argv) {
    try {
        Args a = parse(argc, argv);
        if (a.cmd == "generate") return cmd_generate(a);
        if (a.cmd == "bench") return cmd_bench(a);
        if (a.cmd == "batch") return cmd_batch(a);
        if (a.cmd == "eval") return cmd_eval(a);
        if (a.cmd == "spec") return cmd_spec(a);
        fprintf(stderr, "usage: llm <generate|bench|batch|eval|spec> -m <model_dir> [options]\n");
        return 2;
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
