#include <gtest/gtest.h>
#include "ternix/simd_math.hpp"
#include <vector>
#include <cmath>

TEST(SimdMathTest, RmsNorm) {
    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> weight = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> out(4);
    
    ternix::rms_norm_avx2(out.data(), in.data(), weight.data(), 4, 1e-5f);
    
    float expected_rms = std::sqrt((1+4+9+16)/4.0f + 1e-5f);
    EXPECT_NEAR(out[0], 1.0f / expected_rms, 1e-4);
}

TEST(SimdMathTest, Swiglu) {
    std::vector<float> in = {1.0f, 2.0f};
    std::vector<float> out(1);
    
    ternix::swiglu_avx2(out.data(), in.data(), 2);
    
    float silu = 1.0f / (1.0f + std::exp(-1.0f));
    EXPECT_NEAR(out[0], silu * 2.0f, 1e-4);
}
