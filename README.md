# TernixEngine

A bare-metal, dependency-free C++ inference engine specifically optimized for the new generation of 1.58-bit ternary Vision-Language Models. 

By replacing standard FP16 matrix multiplications with AVX2 SIMD-accelerated integer additions, TernixEngine drastically reduces memory footprints while maximizing CPU token throughput.

## Prerequisites
* A CPU supporting AVX2 or AVX-512 instructions.
* CMake (3.20+)
* A C++20 compatible compiler (GCC, Clang, or MSVC)

## Build Instructions
```bash
git clone --recursive [https://github.com/yourusername/TernixEngine.git](https://github.com/yourusername/TernixEngine.git)
cd TernixEngine

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release 
make -j4
```

## Running Inference
1. Download or quantize a compatible ternary model using scripts/quantize.py.

2. Place the model in the models/ directory.

3. Run the chat interface:
```bash
./build/ternix_cli -m ../models/ternary_model.bin -p "Explain quantum computing."
```