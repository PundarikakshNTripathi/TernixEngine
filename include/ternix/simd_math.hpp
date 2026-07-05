#pragma once
#include <cstdint>
#include <cstddef>

namespace ternix {
    void rms_norm_avx2(float* out, const float* in, const float* weight, size_t size, float eps);
    void swiglu_avx2(float* out, const float* in, size_t size);
}
