// Unit tests for the safetensors reader (no GPU, no model needed): writes
// small checkpoints to a temp directory and reads them back, including a
// sharded layout, every supported dtype and a corrupt header.
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../src/safetensors.h"

static int failures = 0;
#define CHECK(cond)                                                    \
    do {                                                               \
        if (!(cond)) {                                                 \
            printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);     \
            failures++;                                                \
        }                                                              \
    } while (0)

struct Entry {
    std::string name, dtype;
    std::vector<int> shape;
    std::vector<uint8_t> bytes;
};

static void write_file(const std::string& path, const std::vector<Entry>& entries, long offset_skew = 0) {
    std::string hdr = "{\"__metadata__\":{\"format\":\"pt\"}";
    size_t off = 0;
    for (auto& e : entries) {
        hdr += ",\"" + e.name + "\":{\"dtype\":\"" + e.dtype + "\",\"shape\":[";
        for (size_t i = 0; i < e.shape.size(); i++) hdr += (i ? "," : "") + std::to_string(e.shape[i]);
        hdr += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string((long)(off + e.bytes.size()) + offset_skew) + "]}";
        off += e.bytes.size();
    }
    hdr += "}";
    while (hdr.size() % 8) hdr += ' ';  // the format pads the header with spaces
    FILE* f = fopen(path.c_str(), "wb");
    const uint64_t n = hdr.size();
    fwrite(&n, 8, 1, f);
    fwrite(hdr.data(), 1, hdr.size(), f);
    for (auto& e : entries) fwrite(e.bytes.data(), 1, e.bytes.size(), f);
    fclose(f);
}

template <class T>
static std::vector<uint8_t> raw(const std::vector<T>& v) {
    std::vector<uint8_t> b(v.size() * sizeof(T));
    memcpy(b.data(), v.data(), b.size());
    return b;
}

int main() {
    char tmpl[] = "/tmp/st_test_XXXXXX";
    const std::string dir = mkdtemp(tmpl);
    const std::vector<float> vals = {0.0f, 1.0f, -2.5f, 3.140625f, 1024.0f, -0.0078125f};  // exact in fp16 and bf16

    std::vector<__fp16> f16(vals.begin(), vals.end());
    std::vector<uint16_t> bf16;
    for (float v : vals) {
        uint32_t bits;
        memcpy(&bits, &v, 4);
        bf16.push_back((uint16_t)(bits >> 16));
    }

    // --- sharded checkpoint: tensors spread over two files
    write_file(dir + "/model-00001-of-00002.safetensors", {{"a.f32", "F32", {2, 3}, raw(vals)}, {"b.bf16", "BF16", {6}, raw(bf16)}});
    write_file(dir + "/model-00002-of-00002.safetensors", {{"c.f16", "F16", {3, 2}, raw(f16)}});
    {
        const std::vector<std::string> files = SafeTensors::model_files(dir);
        CHECK(files.size() == 2);
        SafeTensors st(files);
        CHECK(st.tensors().size() == 3);
        CHECK(st.has("a.f32") && st.has("b.bf16") && st.has("c.f16") && !st.has("__metadata__"));
        CHECK(st.get("a.f32").shape == (std::vector<int64_t>{2, 3}));
        CHECK(st.get("c.f16").shape == (std::vector<int64_t>{3, 2}));
        for (const char* name : {"a.f32", "b.bf16", "c.f16"}) {
            std::vector<float> out(6, NAN);
            CHECK(st.get(name).numel() == 6);
            SafeTensors::to_f32(st.get(name), out.data());
            for (int i = 0; i < 6; i++) CHECK(out[i] == vals[i]);
        }
        bool threw = false;
        try { st.get("missing"); } catch (const std::exception&) { threw = true; }
        CHECK(threw);
    }

    // --- a single model.safetensors takes precedence over shards
    write_file(dir + "/model.safetensors", {{"only", "F32", {6}, raw(vals)}});
    {
        const std::vector<std::string> files = SafeTensors::model_files(dir);
        CHECK(files.size() == 1);
        SafeTensors st(files);
        CHECK(st.has("only") && !st.has("a.f32"));
    }

    // --- offsets that run past the end of the file are rejected, not mapped
    write_file(dir + "/bad.safetensors", {{"x", "F32", {6}, raw(vals)}}, 4096);
    {
        bool threw = false;
        try { SafeTensors st(dir + "/bad.safetensors"); } catch (const std::exception&) { threw = true; }
        CHECK(threw);
    }

    for (const char* f : {"/model-00001-of-00002.safetensors", "/model-00002-of-00002.safetensors", "/model.safetensors", "/bad.safetensors"})
        unlink((dir + f).c_str());
    rmdir(dir.c_str());
    printf("safetensors: %s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
