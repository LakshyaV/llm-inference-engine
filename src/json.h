// Minimal JSON parser: just enough for safetensors headers, config.json and tokenizer.json.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace json {

struct Value {
    enum Type { Null, Bool, Number, String, Array, Object } type = Null;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<Value> arr;
    // Insertion-ordered object: tokenizer vocabularies are large, keep it flat.
    std::vector<std::pair<std::string, Value>> obj;

    bool is_null() const { return type == Null; }
    bool is_string() const { return type == String; }
    bool is_array() const { return type == Array; }
    bool is_object() const { return type == Object; }

    const Value* find(std::string_view key) const {
        for (auto& kv : obj)
            if (kv.first == key) return &kv.second;
        return nullptr;
    }
    const Value& at(std::string_view key) const {
        if (auto* v = find(key)) return *v;
        throw std::runtime_error("json: missing key '" + std::string(key) + "'");
    }
    const Value& operator[](std::string_view key) const { return at(key); }
    const Value& operator[](size_t i) const { return arr.at(i); }
    int64_t as_int() const { return (int64_t)num; }
    double number_or(std::string_view key, double dflt) const {
        auto* v = find(key);
        return (v && v->type == Number) ? v->num : dflt;
    }
    bool bool_or(std::string_view key, bool dflt) const {
        auto* v = find(key);
        return (v && v->type == Bool) ? v->b : dflt;
    }
};

class Parser {
public:
    explicit Parser(std::string_view s) : p_(s.data()), end_(s.data() + s.size()) {}

    Value parse() {
        Value v = value();
        ws();
        if (p_ != end_) fail("trailing characters");
        return v;
    }

private:
    const char* p_;
    const char* end_;

    [[noreturn]] void fail(const char* msg) { throw std::runtime_error(std::string("json: ") + msg); }
    void ws() {
        while (p_ < end_ && (*p_ == ' ' || *p_ == '\n' || *p_ == '\t' || *p_ == '\r')) ++p_;
    }
    bool lit(std::string_view s) {
        if ((size_t)(end_ - p_) >= s.size() && std::string_view(p_, s.size()) == s) {
            p_ += s.size();
            return true;
        }
        return false;
    }
    static void put_utf8(std::string& out, uint32_t cp) {
        if (cp < 0x80) out += (char)cp;
        else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) {
            out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F));
        } else {
            out += (char)(0xF0 | (cp >> 18)); out += (char)(0x80 | ((cp >> 12) & 0x3F));
            out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F));
        }
    }
    uint32_t hex4() {
        if (end_ - p_ < 4) fail("bad \\u escape");
        uint32_t v = 0;
        for (int i = 0; i < 4; i++) {
            char c = *p_++;
            v <<= 4;
            if (c >= '0' && c <= '9') v |= c - '0';
            else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
            else fail("bad hex digit");
        }
        return v;
    }
    std::string string() {
        if (*p_ != '"') fail("expected string");
        ++p_;
        std::string out;
        while (true) {
            if (p_ >= end_) fail("unterminated string");
            char c = *p_++;
            if (c == '"') break;
            if (c != '\\') { out += c; continue; }
            if (p_ >= end_) fail("bad escape");
            char e = *p_++;
            switch (e) {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00 && end_ - p_ >= 6 && p_[0] == '\\' && p_[1] == 'u') {
                        p_ += 2;
                        uint32_t lo = hex4();
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    put_utf8(out, cp);
                    break;
                }
                default: out += e;  // \" \\ \/
            }
        }
        return out;
    }
    Value value() {
        ws();
        if (p_ >= end_) fail("unexpected end");
        Value v;
        char c = *p_;
        if (c == '{') {
            v.type = Value::Object;
            ++p_;
            ws();
            if (p_ < end_ && *p_ == '}') { ++p_; return v; }
            while (true) {
                ws();
                std::string k = string();
                ws();
                if (p_ >= end_ || *p_++ != ':') fail("expected ':'");
                v.obj.emplace_back(std::move(k), value());
                ws();
                if (p_ < end_ && *p_ == ',') { ++p_; continue; }
                if (p_ < end_ && *p_ == '}') { ++p_; break; }
                fail("expected ',' or '}'");
            }
        } else if (c == '[') {
            v.type = Value::Array;
            ++p_;
            ws();
            if (p_ < end_ && *p_ == ']') { ++p_; return v; }
            while (true) {
                v.arr.push_back(value());
                ws();
                if (p_ < end_ && *p_ == ',') { ++p_; continue; }
                if (p_ < end_ && *p_ == ']') { ++p_; break; }
                fail("expected ',' or ']'");
            }
        } else if (c == '"') {
            v.type = Value::String;
            v.str = string();
        } else if (lit("true")) {
            v.type = Value::Bool; v.b = true;
        } else if (lit("false")) {
            v.type = Value::Bool; v.b = false;
        } else if (lit("null")) {
            v.type = Value::Null;
        } else {
            char* e = nullptr;
            v.type = Value::Number;
            v.num = std::strtod(p_, &e);
            if (e == p_) fail("bad value");
            p_ = e;
        }
        return v;
    }
};

inline Value parse(std::string_view s) { return Parser(s).parse(); }

}  // namespace json
