// Continuous batching.
//
// Requests join and leave the running batch at token granularity: every
// step() builds one engine pass from whatever is active right now -- prompt
// chunks for sequences still in prefill, one token for each sequence that is
// decoding -- so a new request never waits for the batch in flight to finish
// and a finished one frees its KV pages immediately.
#pragma once
#include <deque>
#include <functional>
#include <vector>

#include "engine.h"
#include "sampler.h"

struct Request {
    std::vector<int> prompt;
    int max_tokens = 128;
    SamplerConfig sampling;
};

struct RequestResult {
    int id = -1;
    std::vector<int> tokens;        // generated tokens (without the stop token)
    double t_submit = 0, t_first = 0, t_done = 0;
    bool finished = false;
};

struct SchedulerConfig {
    int max_active = 8;             // sequences decoded concurrently
    int prefill_budget = 256;       // prompt tokens admitted per step (keeps decode latency bounded)
};

class Scheduler {
public:
    Scheduler(Engine& engine, const SchedulerConfig& cfg = {});

    int submit(Request r);          // returns request id; the request waits until a slot is free
    bool step();                    // run one batched pass; false when there is nothing to do
    void run() { while (step()) {} }

    const RequestResult& result(int id) const { return results_[id]; }
    int active() const { return (int)active_.size(); }
    int waiting() const { return (int)waiting_.size(); }
    int steps() const { return steps_; }
    long tokens_processed() const { return tokens_processed_; }

    // Called for every generated token as soon as it is sampled.
    std::function<void(int id, int token)> on_token;

private:
    struct Active {
        int id, seq;
        Request req;
        size_t fed = 0;             // prompt tokens already in the KV cache
        int next = -1;              // sampled token to feed on the next step
        int reserved_pages = 0;
        Sampler sampler;
    };
    void admit();
    void finish(size_t idx);

    Engine& eng_;
    SchedulerConfig cfg_;
    std::deque<std::pair<int, Request>> waiting_;
    std::vector<Active> active_;
    std::vector<RequestResult> results_;
    int steps_ = 0;
    long tokens_processed_ = 0;
};
