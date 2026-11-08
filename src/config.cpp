#include "config.h"

#include "json.h"
#include "util.h"

ModelConfig ModelConfig::load(const std::string& dir) {
    json::Value j = json::parse(read_file(dir + "/config.json"));
    ModelConfig c;
    c.vocab_size = (int)j["vocab_size"].as_int();
    c.hidden = (int)j["hidden_size"].as_int();
    c.inter = (int)j["intermediate_size"].as_int();
    c.n_layers = (int)j["num_hidden_layers"].as_int();
    c.n_heads = (int)j["num_attention_heads"].as_int();
    c.n_kv_heads = (int)j.number_or("num_key_value_heads", c.n_heads);
    c.head_dim = c.hidden / c.n_heads;
    c.rms_eps = (float)j.number_or("rms_norm_eps", 1e-6);
    c.rope_theta = (float)j.number_or("rope_theta", 10000.0);
    c.tie_embeddings = j.bool_or("tie_word_embeddings", false);

    auto add_eos = [&](const json::Value& v) {
        if (v.type == json::Value::Number) c.eos_ids.push_back((int)v.as_int());
        else if (v.is_array())
            for (auto& e : v.arr) c.eos_ids.push_back((int)e.as_int());
    };
    if (auto* e = j.find("eos_token_id")) add_eos(*e);
    // generation_config.json usually lists the full stop set (e.g. <|im_end|> and <|endoftext|>).
    if (file_exists(dir + "/generation_config.json")) {
        json::Value g = json::parse(read_file(dir + "/generation_config.json"));
        if (auto* e = g.find("eos_token_id")) {
            c.eos_ids.clear();
            add_eos(*e);
        }
    }
    return c;
}
