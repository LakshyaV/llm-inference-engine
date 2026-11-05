#pragma once
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

inline std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

inline bool file_exists(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return (bool)f;
}

inline double now_sec() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

[[noreturn]] inline void die(const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    throw std::runtime_error(buf);
}
