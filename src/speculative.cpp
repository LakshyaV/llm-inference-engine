#include "speculative.h"

#include <algorithm>

#include "util.h"

bool speculative_accept(const std::vector<float>& p, const std::vector<float>& q, int x, Sampler& sampler,
                        int& replacement) {
    if (sampler.uniform() * q[x] < p[x]) return true;  // accept with probability min(1, p/q)
    std::vector<float> resid(p.size());
    double sum = 0;
    for (size_t v = 0; v < p.size(); v++) sum += (resid[v] = std::max(0.0f, p[v] - q[v]));
    if (sum > 0) {
        for (auto& r : resid) r = (float)(r / sum);
        replacement = sampler.draw(resid);
    } else {
        replacement = sampler.draw(p);  // p == q up to rounding
    }
    return false;
}

SpeculativeDecoder::SpeculativeDecoder(Engine& target, Engine& draft, int gamma, const SamplerConfig& sampling)
    : target_(target), draft_(draft), gamma_(gamma), sampler_(sampling) {
    if (target.config().vocab_size != draft.config().vocab_size) die("speculative: models must share a vocabulary");
    if (gamma < 1 || gamma + 1 > target.options().max_logit_rows) die("speculative: bad gamma");
}

std::vector<int> SpeculativeDecoder::generate(const std::vector<int>& prompt, int max_tokens,
                                              const std::function<void(int)>& on_token) {
    stats_ = {};
    const int V = target_.config().vocab_size;
    const auto& eos = target_.config().eos_ids;
    const bool greedy = sampler_.greedy();
    const int tseq = target_.seq_create(), dseq = draft_.seq_create();
    if (tseq < 0 || dseq < 0) die("speculative: no free sequence slot");

    // Invariant between rounds:
    //   target cache = everything emitted except `t_pending` (the newest token)
    //   draft cache  = everything emitted except the tokens in `d_pending`
    std::vector<TokenIn> feed;
    for (size_t i = 0; i + 1 < prompt.size(); i++) feed.push_back({prompt[i], tseq});
    if (!feed.empty()) target_.forward(feed, {});
    for (auto& t : feed) t.seq = dseq;
    if (!feed.empty()) draft_.forward(feed, {});
    int t_pending = prompt.back();
    std::vector<int> d_pending = {prompt.back()};

    std::vector<int> out;
    std::vector<int> x(gamma_);
    std::vector<std::vector<float>> q(greedy ? 0 : gamma_);
    std::vector<float> p;
    bool stop = false;

    auto emit = [&](int tok) {
        if (std::find(eos.begin(), eos.end(), tok) != eos.end()) { stop = true; return; }
        out.push_back(tok);
        stats_.generated++;
        if (on_token) on_token(tok);
        if ((int)out.size() >= max_tokens) stop = true;
    };

    while (!stop) {
        const int gamma = std::min(gamma_, max_tokens - (int)out.size());
        const int t_len = target_.seq_len(tseq), d_len = draft_.seq_len(dseq);

        // --- 1. draft proposes x[0..gamma)
        for (int i = 0; i < gamma; i++) {
            feed.clear();
            if (i == 0) for (int tok : d_pending) feed.push_back({tok, dseq});
            else feed.push_back({x[i - 1], dseq});
            const float* dl = draft_.forward(feed, {(int)feed.size() - 1});
            if (greedy) {
                x[i] = argmax(dl, V);
            } else {
                sampler_.distribution(dl, V, q[i]);
                x[i] = sampler_.draw(q[i]);
            }
        }
        stats_.drafted += gamma;

        // --- 2. target scores the pending token and all proposals in one pass
        feed.assign(1, {t_pending, tseq});
        std::vector<int> want(gamma + 1);
        for (int i = 0; i < gamma; i++) feed.push_back({x[i], tseq});
        for (int i = 0; i <= gamma; i++) want[i] = i;
        const float* tl = target_.forward(feed, want);
        stats_.target_passes++;

        // --- 3. accept / reject, left to right. Row i is the target's
        // distribution for the position x[i] was proposed for.
        int n_acc = 0, next = -1;
        for (; n_acc < gamma; n_acc++) {
            const float* row = tl + (size_t)n_acc * V;
            const int xi = x[n_acc];
            if (greedy) {
                const int best = argmax(row, V);
                if (best != xi) { next = best; break; }
            } else {
                sampler_.distribution(row, V, p);
                if (!speculative_accept(p, q[n_acc], xi, sampler_, next)) break;
            }
        }
        if (next < 0) next = sampler_.sample(tl + (size_t)gamma * V, V);  // all accepted: bonus token
        stats_.accepted += n_acc;

        // --- 4. emit and roll the caches back to the accepted prefix
        for (int i = 0; i < n_acc && !stop; i++) emit(x[i]);
        if (!stop) emit(next);

        target_.seq_truncate(tseq, t_len + 1 + n_acc);          // pending + accepted proposals
        t_pending = next;
        // The draft consumed d_pending and x[0..gamma-1); keep the accepted part.
        const int d_keep = std::min(n_acc, gamma - 1);
        draft_.seq_truncate(dseq, d_len + (int)d_pending.size() + d_keep);
        d_pending.clear();
        if (n_acc == gamma) d_pending.push_back(x[gamma - 1]);  // accepted but never fed to the draft
        d_pending.push_back(next);
    }
    target_.seq_free(tseq);
    draft_.seq_free(dseq);
    return out;
}
