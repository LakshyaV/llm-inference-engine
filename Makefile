# Build with the Xcode Command Line Tools only (no Xcode, no CMake).
CXX      := clang++
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -Wno-unused-parameter -Wno-deprecated-declarations -fobjc-arc -Ibuild
LDFLAGS  := -framework Metal -framework Foundation -framework Accelerate -framework MetalPerformanceShaders

KERNELS  := $(sort $(wildcard src/kernels/*.metal))
SRC      := $(wildcard src/*.cpp) $(wildcard src/*.mm)
OBJ      := $(patsubst src/%,build/obj/%.o,$(SRC))
TESTS    := $(patsubst tests/%.cpp,build/%,$(wildcard tests/*.cpp)) $(patsubst tests/%.mm,build/%,$(wildcard tests/*.mm))
BENCHES  := $(patsubst bench/%.mm,build/%,$(wildcard bench/*.mm))
APPS     := $(patsubst apps/%.cpp,build/%,$(wildcard apps/*.cpp))

all: $(APPS) $(TESTS) $(BENCHES)

# Metal shaders are compiled at runtime; embed their source in the binary.
build/kernels.inc: $(KERNELS)
	@mkdir -p build
	cat $(KERNELS) | xxd -i > $@

build/obj/metal_ctx.mm.o: build/kernels.inc

build/obj/%.o: src/% $(wildcard src/*.h)
	@mkdir -p build/obj
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/libengine.a: $(OBJ)
	ar rcs $@ $^

build/%: apps/%.cpp build/libengine.a
	$(CXX) $(CXXFLAGS) $< build/libengine.a $(LDFLAGS) -o $@

build/%: tests/%.cpp build/libengine.a
	$(CXX) $(CXXFLAGS) $< build/libengine.a $(LDFLAGS) -o $@

build/%: tests/%.mm build/libengine.a
	$(CXX) $(CXXFLAGS) $< build/libengine.a $(LDFLAGS) -o $@

build/%: bench/%.mm build/libengine.a
	$(CXX) $(CXXFLAGS) $< build/libengine.a $(LDFLAGS) -o $@

# Unit tests need nothing; the model tests need weights and reference data:
#   make test MODEL=models/Qwen2.5-1.5B-Instruct
MODEL ?=
test: all
	build/test_kv_pager
	build/test_quant
	build/test_safetensors
	build/test_speculative
ifneq ($(MODEL),)
	build/test_tokenizer $(MODEL) tests/data/tokenizer_cases.json
	build/test_logits $(MODEL) tests/data/logits_$(notdir $(MODEL)).bin
endif

clean:
	rm -rf build

.PHONY: all test clean
