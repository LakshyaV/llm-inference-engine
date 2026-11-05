// Memory-mapped safetensors reader.
//
// File layout: [u64 little-endian header length][JSON header][raw tensor bytes].
// The header maps tensor name -> {dtype, shape, data_offsets:[begin,end)} with
// offsets relative to the end of the header.
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct TensorView {
    std::string dtype;            // "F32", "F16", "BF16", ...
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr;
    size_t nbytes = 0;

    size_t numel() const {
        size_t n = 1;
        for (auto d : shape) n *= (size_t)d;
        return n;
    }
};

class SafeTensors {
public:
    explicit SafeTensors(const std::string& path) : SafeTensors(std::vector<std::string>{path}) {}
    // Several files presented as one checkpoint (sharded models).
    explicit SafeTensors(const std::vector<std::string>& paths);
    // The weight files of a Hugging Face model directory: model.safetensors,
    // or the shards model-00001-of-0000N.safetensors ... in order.
    static std::vector<std::string> model_files(const std::string& model_dir);
    ~SafeTensors();
    SafeTensors(const SafeTensors&) = delete;
    SafeTensors& operator=(const SafeTensors&) = delete;

    bool has(const std::string& name) const { return tensors_.count(name) != 0; }
    const TensorView& get(const std::string& name) const;
    const std::unordered_map<std::string, TensorView>& tensors() const { return tensors_; }

    // Decode a tensor to float32 (supports F32 / F16 / BF16). `out` must hold numel() floats.
    static void to_f32(const TensorView& t, float* out);

private:
    struct Mapping {
        int fd;
        uint8_t* data;
        size_t size;
    };
    void map_file(const std::string& path);
    std::vector<Mapping> maps_;
    std::unordered_map<std::string, TensorView> tensors_;
};
