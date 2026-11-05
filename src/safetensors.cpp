#include "safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>

#include "json.h"
#include "util.h"

SafeTensors::SafeTensors(const std::vector<std::string>& paths) {
    if (paths.empty()) die("safetensors: no weight files");
    for (auto& p : paths) map_file(p);
}

std::vector<std::string> SafeTensors::model_files(const std::string& dir) {
    if (file_exists(dir + "/model.safetensors")) return {dir + "/model.safetensors"};
    std::vector<std::string> shards;
    for (int total = 2; total < 100 && shards.empty(); total++) {
        char name[64];
        snprintf(name, sizeof name, "/model-00001-of-%05d.safetensors", total);
        if (!file_exists(dir + name)) continue;
        for (int i = 1; i <= total; i++) {
            snprintf(name, sizeof name, "/model-%05d-of-%05d.safetensors", i, total);
            shards.push_back(dir + name);
        }
    }
    if (shards.empty()) die("no model.safetensors (or shards) in %s", dir.c_str());
    return shards;
}

void SafeTensors::map_file(const std::string& path) {
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) die("safetensors: cannot open %s", path.c_str());
    struct stat st;
    fstat(fd, &st);
    const size_t size = (size_t)st.st_size;
    uint8_t* map = (uint8_t*)mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) die("safetensors: mmap failed for %s", path.c_str());
    maps_.push_back({fd, map, size});
    if (size < 8) die("safetensors: file too small");

    uint64_t hlen;
    memcpy(&hlen, map, 8);
    if (hlen > size - 8) die("safetensors: corrupt header length");
    json::Value hdr = json::parse(std::string_view((const char*)map + 8, hlen));
    const uint8_t* base = map + 8 + hlen;
    const size_t payload = size - 8 - hlen;

    for (auto& [name, info] : hdr.obj) {
        if (name == "__metadata__") continue;
        TensorView t;
        t.dtype = info["dtype"].str;
        for (auto& d : info["shape"].arr) t.shape.push_back(d.as_int());
        size_t b = (size_t)info["data_offsets"][0].as_int();
        size_t e = (size_t)info["data_offsets"][1].as_int();
        if (b > e || e > payload) die("safetensors: tensor %s out of bounds", name.c_str());
        t.data = base + b;
        t.nbytes = e - b;
        tensors_.emplace(name, std::move(t));
    }
}

SafeTensors::~SafeTensors() {
    for (auto& m : maps_) {
        munmap(m.data, m.size);
        close(m.fd);
    }
}

const TensorView& SafeTensors::get(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) die("safetensors: no tensor named %s", name.c_str());
    return it->second;
}

static inline float f16_to_f32(uint16_t h) {
    __fp16 v;
    memcpy(&v, &h, 2);
    return (float)v;
}

void SafeTensors::to_f32(const TensorView& t, float* out) {
    const size_t n = t.numel();
    if (t.dtype == "F32") {
        memcpy(out, t.data, n * 4);
    } else if (t.dtype == "BF16") {
        // bfloat16 is the top 16 bits of an IEEE float32.
        const uint16_t* src = (const uint16_t*)t.data;
        for (size_t i = 0; i < n; i++) {
            uint32_t bits = (uint32_t)src[i] << 16;
            memcpy(&out[i], &bits, 4);
        }
    } else if (t.dtype == "F16") {
        const uint16_t* src = (const uint16_t*)t.data;
        for (size_t i = 0; i < n; i++) out[i] = f16_to_f32(src[i]);
    } else {
        die("safetensors: unsupported dtype %s", t.dtype.c_str());
    }
}
