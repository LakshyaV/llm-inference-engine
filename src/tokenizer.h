// Byte-level BPE tokenizer compatible with Hugging Face `tokenizer.json`
// files that use the GPT-2/tiktoken-style scheme (Qwen2.5, Llama-3, GPT-2).
//
// Pipeline: split out added/special tokens -> NFC normalization -> regex pre-tokenization
// (hand-written matcher, see pretokenize()) -> byte-level BPE merges.
#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// Unicode NFC normalization of UTF-8 text (malformed input is returned unchanged).
std::string nfc_normalize(std::string_view text);

class Tokenizer {
public:
    explicit Tokenizer(const std::string& tokenizer_json_path);

    std::vector<int> encode(std::string_view text) const;
    std::string decode(const std::vector<int>& ids, bool skip_special = false) const;
    // Raw bytes of one token (may be a partial UTF-8 sequence).
    const std::string& token_bytes(int id) const { return id_to_bytes_.at(id); }

    int vocab_size() const { return (int)id_to_bytes_.size(); }
    int token_id(std::string_view token) const;  // -1 if absent
    bool is_special(int id) const { return id >= 0 && id < (int)special_.size() && special_[id]; }

    // Exposed for tests: split text into pre-tokens exactly as the regex would.
    std::vector<std::string_view> pretokenize(std::string_view text) const;

private:
    struct Added {
        std::string content;
        int id;
    };
    void bpe(std::string_view word, std::vector<int>& out) const;

    std::vector<std::string> id_to_bytes_;
    std::vector<uint8_t> special_;
    std::unordered_map<std::string, int> bytes_to_id_;
    std::unordered_map<uint64_t, std::pair<int, int>> merges_;  // (a,b) -> (rank, merged id)
    std::vector<Added> added_;                                  // sorted longest first
    bool added_first_[256] = {};                                // first bytes of added tokens
    int byte_ids_[256];
    bool nfc_ = false;             // normalizer: NFC
    bool digits_grouped_ = false;  // \p{N}{1,3} (Llama-3) vs \p{N} (Qwen/GPT-2 style)
    mutable std::unordered_map<std::string, std::vector<int>> cache_;
};
