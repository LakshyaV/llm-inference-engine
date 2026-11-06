#include "tokenizer.h"

#include <algorithm>
#include <climits>

#include "json.h"
#include "unicode_tables.h"
#include "util.h"

// ---------------------------------------------------------------- unicode --

static bool in_ranges(const uint32_t (*tbl)[2], size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < tbl[mid][0]) hi = mid;
        else if (cp > tbl[mid][1]) lo = mid + 1;
        else return true;
    }
    return false;
}
static bool is_letter(uint32_t cp) {
    if (cp < 0x80) return (cp | 0x20) >= 'a' && (cp | 0x20) <= 'z';
    return in_ranges(kUnicodeLetterRanges, sizeof kUnicodeLetterRanges / sizeof kUnicodeLetterRanges[0], cp);
}
static bool is_number(uint32_t cp) {
    if (cp < 0x80) return cp >= '0' && cp <= '9';
    return in_ranges(kUnicodeNumberRanges, sizeof kUnicodeNumberRanges / sizeof kUnicodeNumberRanges[0], cp);
}
// Unicode White_Space property (what \s means in the tokenizer's regex engine).
static bool is_space(uint32_t cp) {
    return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F ||
           cp == 0x3000;
}
static bool is_newline(uint32_t cp) { return cp == '\r' || cp == '\n'; }

// Decode one UTF-8 code point at s[i]; invalid bytes decode as U+FFFD-like
// singletons so that arbitrary input still round-trips byte-for-byte.
static uint32_t decode_utf8(std::string_view s, size_t i, size_t& len) {
    uint8_t c = (uint8_t)s[i];
    auto cont = [&](size_t k) { return i + k < s.size() && ((uint8_t)s[i + k] & 0xC0) == 0x80; };
    if (c < 0x80) { len = 1; return c; }
    if ((c & 0xE0) == 0xC0 && cont(1)) { len = 2; return ((c & 0x1F) << 6) | ((uint8_t)s[i + 1] & 0x3F); }
    if ((c & 0xF0) == 0xE0 && cont(1) && cont(2)) {
        len = 3;
        return ((c & 0x0F) << 12) | (((uint8_t)s[i + 1] & 0x3F) << 6) | ((uint8_t)s[i + 2] & 0x3F);
    }
    if ((c & 0xF8) == 0xF0 && cont(1) && cont(2) && cont(3)) {
        len = 4;
        return ((c & 0x07) << 18) | (((uint8_t)s[i + 1] & 0x3F) << 12) | (((uint8_t)s[i + 2] & 0x3F) << 6) |
               ((uint8_t)s[i + 3] & 0x3F);
    }
    len = 1;
    return 0xFFFD;
}

// ------------------------------------------------------------- byte-level --

// GPT-2's byte<->unicode table: printable bytes map to themselves, the rest
// are shifted to U+0100.. so every byte has a visible, JSON-safe character.
static void build_byte_map(uint32_t byte_to_cp[256]) {
    int n = 0;
    for (int b = 0; b < 256; b++) {
        bool printable = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        byte_to_cp[b] = printable ? (uint32_t)b : (uint32_t)(256 + n++);
    }
}

// Convert a vocab entry (byte-level unicode form) back to raw bytes.
static std::string unmap_token(std::string_view tok, const std::unordered_map<uint32_t, uint8_t>& cp_to_byte) {
    std::string out;
    for (size_t i = 0; i < tok.size();) {
        size_t len;
        uint32_t cp = decode_utf8(tok, i, len);
        auto it = cp_to_byte.find(cp);
        if (it == cp_to_byte.end()) return std::string(tok);  // not byte-level (shouldn't happen)
        out += (char)it->second;
        i += len;
    }
    return out;
}

static inline uint64_t pair_key(int a, int b) { return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b; }

// ------------------------------------------------------------------- load --

Tokenizer::Tokenizer(const std::string& path) {
    json::Value root = json::parse(read_file(path));
    const json::Value& model = root["model"];
    if (model["type"].str != "BPE") die("tokenizer: only BPE models are supported");

    uint32_t byte_to_cp[256];
    build_byte_map(byte_to_cp);
    std::unordered_map<uint32_t, uint8_t> cp_to_byte;
    for (int b = 0; b < 256; b++) cp_to_byte[byte_to_cp[b]] = (uint8_t)b;

    for (auto& [tok, idv] : model["vocab"].obj) {
        int id = (int)idv.as_int();
        if (id >= (int)id_to_bytes_.size()) id_to_bytes_.resize(id + 1);
        id_to_bytes_[id] = unmap_token(tok, cp_to_byte);
    }
    for (int id = 0; id < (int)id_to_bytes_.size(); id++) bytes_to_id_[id_to_bytes_[id]] = id;

    for (int b = 0; b < 256; b++) {
        auto it = bytes_to_id_.find(std::string(1, (char)b));
        byte_ids_[b] = it == bytes_to_id_.end() ? -1 : it->second;
    }

    int rank = 0;
    for (auto& m : model["merges"].arr) {
        std::string a, b;
        if (m.is_string()) {  // legacy "left right"
            size_t sp = m.str.find(' ');
            a = m.str.substr(0, sp);
            b = m.str.substr(sp + 1);
        } else {  // ["left", "right"]
            a = m[0].str;
            b = m[1].str;
        }
        a = unmap_token(a, cp_to_byte);
        b = unmap_token(b, cp_to_byte);
        auto ia = bytes_to_id_.find(a), ib = bytes_to_id_.find(b), im = bytes_to_id_.find(a + b);
        if (ia != bytes_to_id_.end() && ib != bytes_to_id_.end() && im != bytes_to_id_.end())
            merges_.emplace(pair_key(ia->second, ib->second), std::make_pair(rank, im->second));
        rank++;
    }

    if (auto* added = root.find("added_tokens")) {
        for (auto& a : added->arr) {
            int id = (int)a["id"].as_int();
            if (id >= (int)id_to_bytes_.size()) id_to_bytes_.resize(id + 1);
            id_to_bytes_[id] = a["content"].str;
            if (id >= (int)special_.size()) special_.resize(id + 1, 0);
            special_[id] = a.bool_or("special", false);
            added_.push_back({a["content"].str, id});
        }
    }
    special_.resize(id_to_bytes_.size(), 0);
    for (auto& a : added_)
        if (!a.content.empty()) added_first_[(unsigned char)a.content[0]] = true;
    std::sort(added_.begin(), added_.end(),
              [](const Added& x, const Added& y) { return x.content.size() > y.content.size(); });

    // Llama-3 groups digits as \p{N}{1,3}; Qwen / GPT-2 emit one digit per pre-token.
    if (auto* pre = root.find("pre_tokenizer")) {
        std::string pattern;
        auto scan = [&](const json::Value& p) {
            if (auto* pat = p.find("pattern"))
                if (auto* re = pat->find("Regex")) pattern = re->str;
        };
        scan(*pre);
        if (auto* seq = pre->find("pretokenizers"))
            for (auto& p : seq->arr) scan(p);
        digits_grouped_ = pattern.find("\\p{N}{1,3}") != std::string::npos;
    }
}

int Tokenizer::token_id(std::string_view token) const {
    for (auto& a : added_)
        if (a.content == token) return a.id;
    auto it = bytes_to_id_.find(std::string(token));
    return it == bytes_to_id_.end() ? -1 : it->second;
}

// ----------------------------------------------------------- pre-tokenize --
//
// Hand-written matcher for the regex
//
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)            contractions
//   | [^\r\n\p{L}\p{N}]?\p{L}+               word with optional leading symbol/space
//   | \p{N}                (or \p{N}{1,3})   digits
//   | ?[^\s\p{L}\p{N}]+[\r\n]*               punctuation run (+ optional leading space)
//   | \s*[\r\n]+                             whitespace ending in newlines
//   | \s+(?!\S)                              whitespace not followed by text
//   | \s+                                    remaining whitespace
//
// Alternatives are tried in order with leftmost-first semantics, mirroring
// the backtracking engine used by `tokenizers`.

std::vector<std::string_view> Tokenizer::pretokenize(std::string_view s) const {
    // Decode once so lookahead is O(1).
    std::vector<uint32_t> cp;
    std::vector<uint32_t> off;
    cp.reserve(s.size());
    off.reserve(s.size() + 1);
    for (size_t i = 0; i < s.size();) {
        size_t len;
        cp.push_back(decode_utf8(s, i, len));
        off.push_back((uint32_t)i);
        i += len;
    }
    off.push_back((uint32_t)s.size());
    const size_t n = cp.size();

    std::vector<std::string_view> out;
    auto lower = [](uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; };
    size_t i = 0;
    while (i < n) {
        size_t end = 0;  // exclusive end of the match; 0 = no match yet
        uint32_t c = cp[i];

        // 1. contractions
        if (c == '\'' && i + 1 < n) {
            uint32_t a = lower(cp[i + 1]);
            uint32_t b = i + 2 < n ? lower(cp[i + 2]) : 0;
            if (a == 's' || a == 't') end = i + 2;
            else if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e')) end = i + 3;
            else if (a == 'm') end = i + 2;
            else if (a == 'l' && b == 'l') end = i + 3;
            else if (a == 'd') end = i + 2;
        }
        // 2. [^\r\n\p{L}\p{N}]?\p{L}+
        if (!end) {
            size_t j = i;
            if (!is_newline(c) && !is_letter(c) && !is_number(c)) j++;
            if (j < n && is_letter(cp[j])) {
                while (j < n && is_letter(cp[j])) j++;
                end = j;
            }
        }
        // 3. \p{N} / \p{N}{1,3}
        if (!end && is_number(c)) {
            size_t j = i + 1;
            if (digits_grouped_)
                while (j < n && j < i + 3 && is_number(cp[j])) j++;
            end = j;
        }
        // 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
        if (!end) {
            auto punct = [&](size_t k) { return k < n && !is_space(cp[k]) && !is_letter(cp[k]) && !is_number(cp[k]); };
            size_t j = i;
            if (c == ' ' && punct(i + 1)) j = i + 1;
            if (punct(j)) {
                while (punct(j)) j++;
                while (j < n && is_newline(cp[j])) j++;
                end = j;
            }
        }
        // 5-7. whitespace
        if (!end && is_space(c)) {
            size_t e = i;
            while (e < n && is_space(cp[e])) e++;
            size_t last_nl = SIZE_MAX;
            for (size_t k = e; k-- > i;)
                if (is_newline(cp[k])) { last_nl = k; break; }
            if (last_nl != SIZE_MAX) end = last_nl + 1;      // \s*[\r\n]+
            else if (e == n) end = e;                        // \s+(?!\S) at end of text
            else if (e - i > 1) end = e - 1;                 // \s+(?!\S): leave one space for the next word
            else end = e;                                    // \s+
        }
        if (!end) end = i + 1;  // unreachable for valid input; never stall
        out.push_back(s.substr(off[i], off[end] - off[i]));
        i = end;
    }
    return out;
}

// -------------------------------------------------------------------- BPE --

void Tokenizer::bpe(std::string_view word, std::vector<int>& out) const {
    auto hit = cache_.find(std::string(word));
    if (hit != cache_.end()) {
        out.insert(out.end(), hit->second.begin(), hit->second.end());
        return;
    }
    std::vector<int> sym;
    sym.reserve(word.size());
    for (unsigned char b : word) sym.push_back(byte_ids_[b]);

    // Greedy lowest-rank merge. Words are short, so the O(n^2) scan is fine.
    while (sym.size() > 1) {
        int best_rank = INT_MAX, best_id = -1;
        for (size_t k = 0; k + 1 < sym.size(); k++) {
            auto it = merges_.find(pair_key(sym[k], sym[k + 1]));
            if (it != merges_.end() && it->second.first < best_rank) {
                best_rank = it->second.first;
                best_id = it->second.second;
            }
        }
        if (best_id < 0) break;
        // Merge every occurrence of the winning pair, left to right.
        std::vector<int> next;
        next.reserve(sym.size());
        for (size_t k = 0; k < sym.size();) {
            if (k + 1 < sym.size()) {
                auto it = merges_.find(pair_key(sym[k], sym[k + 1]));
                if (it != merges_.end() && it->second.first == best_rank) {
                    next.push_back(best_id);
                    k += 2;
                    continue;
                }
            }
            next.push_back(sym[k++]);
        }
        sym.swap(next);
    }
    if (word.size() < 64) cache_.emplace(std::string(word), sym);
    out.insert(out.end(), sym.begin(), sym.end());
}

std::vector<int> Tokenizer::encode(std::string_view text) const {
    std::vector<int> ids;
    auto encode_plain = [&](std::string_view chunk) {
        for (auto w : pretokenize(chunk)) bpe(w, ids);
    };
    size_t start = 0, i = 0;
    while (i < text.size()) {
        const Added* match = nullptr;
        if (added_first_[(unsigned char)text[i]])
            for (auto& a : added_)
                if (text.compare(i, a.content.size(), a.content) == 0) { match = &a; break; }
        if (!match) { i++; continue; }
        if (i > start) encode_plain(text.substr(start, i - start));
        ids.push_back(match->id);
        i += match->content.size();
        start = i;
    }
    if (start < text.size()) encode_plain(text.substr(start));
    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids, bool skip_special) const {
    std::string out;
    for (int id : ids) {
        if (id < 0 || id >= (int)id_to_bytes_.size()) continue;
        if (skip_special && is_special(id)) continue;
        out += id_to_bytes_[id];
    }
    return out;
}
