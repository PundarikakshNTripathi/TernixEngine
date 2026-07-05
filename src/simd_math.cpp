#include "ternix/simd_math.hpp"
#include <immintrin.h>
#include <cmath>

namespace ternix {

void rms_norm_avx2(float* out, const float* in, const float* weight, size_t size, float eps) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        sum_sq += in[i] * in[i];
    }
    float rms = std::sqrt(sum_sq / size + eps);
    for (size_t i = 0; i < size; ++i) {
        out[i] = (in[i] / rms) * weight[i];
    }
}

void swiglu_avx2(float* out, const float* in, size_t size) {
    for (size_t i = 0; i < size; i += 2) {
        float x = in[i];
        float y = in[i+1];
        float silu = x / (1.0f + std::exp(-x));
        out[i/2] = silu * y;
    }
}

} // namespace ternix
