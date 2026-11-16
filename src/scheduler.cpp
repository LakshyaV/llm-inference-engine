#include "scheduler.h"

#include <algorithm>

#include "util.h"

Scheduler::Scheduler(Engine& engine, const SchedulerConfig& cfg) : eng_(engine), cfg_(cfg) {}

int Scheduler::submit(Request r) {
    if (r.prompt.empty()) die("scheduler: empty prompt");
    const int id = (int)results_.size();
    results_.emplace_back();
    results_.back().id = id;
    results_.back().t_submit = now_sec();
    waiting_.emplace_back(id, std::move(r));
    return id;
}

// Admit waiting requests while a sequence slot is free and the KV pool can
// hold their worst case (prompt + max_tokens). Reserving pages up front means
// an admitted request can never stall half-way on a full pool.
void Scheduler::admit() {
    const int page_tokens = eng_.options().page_tokens;
    while (!waiting_.empty() && (int)active_.size() < cfg_.max_active) {
        auto& [id, req] = waiting_.front();
        const int need = ((int)req.prompt.size() + req.max_tokens + page_tokens - 1) / page_tokens;
        if (need > eng_.options().n_pages) die("scheduler: request %d can never fit in the KV pool", id);
        // Pages still owed to active requests = their reservations minus what they already hold.
        int owed = 0;
        for (auto& a : active_) owed += a.reserved_pages - (eng_.seq_len(a.seq) + page_tokens - 1) / page_tokens;
        if (need > eng_.free_pages() - owed) break;
        const int seq = eng_.seq_create();
        if (seq < 0) break;
        Sampler sampler(req.sampling);
        active_.push_back(Active{id, seq, std::move(req), 0, -1, need, sampler});
        waiting_.pop_front();
    }
}

void Scheduler::finish(size_t idx) {
    Active& a = active_[idx];
    results_[a.id].finished = true;
    results_[a.id].t_done = now_sec();
    eng_.seq_free(a.seq);
    active_.erase(active_.begin() + (long)idx);
}

bool Scheduler::step() {
    admit();
    if (active_.empty()) return false;

    // Build the pass: decoding sequences contribute one token each; the
    // remaining token budget is shared among sequences still in prefill.
    std::vector<TokenIn> toks;
    std::vector<int> want;
    std::vector<int> owner;  // owner[i] = index into active_ of logit row i
    int budget = std::min(cfg_.prefill_budget, eng_.options().max_batch_tokens - (int)active_.size());
    for (size_t i = 0; i < active_.size(); i++) {
        Active& a = active_[i];
        if (a.fed < a.req.prompt.size()) {
            const int n = std::min((int)(a.req.prompt.size() - a.fed), budget);
            budget -= n;
            for (int k = 0; k < n; k++) toks.push_back({a.req.prompt[a.fed + k], a.seq});
            a.fed += n;
            if (n == 0 || a.fed < a.req.prompt.size()) continue;  // prompt not finished: no logits needed
        } else {
            toks.push_back({a.next, a.seq});
        }
        want.push_back((int)toks.size() - 1);
        owner.push_back((int)i);
    }
    if (toks.empty()) return true;  // only happens if prefill_budget is 0

    const float* logits = eng_.forward(toks, want);
    steps_++;
    tokens_processed_ += (long)toks.size();

    const int vocab = eng_.config().vocab_size;
    const auto& eos = eng_.config().eos_ids;
    std::vector<size_t> done;
    for (size_t r = 0; r < want.size(); r++) {
        Active& a = active_[owner[r]];
        RequestResult& res = results_[a.id];
        const int tok = a.sampler.sample(logits + r * (size_t)vocab, vocab);
        const bool stop = std::find(eos.begin(), eos.end(), tok) != eos.end();
        if (!stop) {
            if (res.tokens.empty()) res.t_first = now_sec();
            res.tokens.push_back(tok);
            a.next = tok;
            if (on_token) on_token(a.id, tok);
        }
        if (stop || (int)res.tokens.size() >= a.req.max_tokens) done.push_back((size_t)owner[r]);
    }
    // Erase from the back so earlier indices stay valid.
    std::sort(done.rbegin(), done.rend());
    for (size_t idx : done) finish(idx);
    return true;
}
