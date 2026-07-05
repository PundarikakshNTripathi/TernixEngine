#include <gtest/gtest.h>
#include "ternix/bitpack.hpp"
#include <vector>

TEST(BitpackTest, PackAndUnpack) {
    std::vector<int8_t> original = {1, 0, -1, 1, -1, 0, 1, 0};
    std::vector<uint8_t> packed(original.size() / 4);
    
    ternix::pack_ternary_weights(packed.data(), original.data(), original.size());
    
    std::vector<int8_t> unpacked(original.size());
    ternix::unpack_ternary_weights(unpacked.data(), packed.data(), original.size());
    
    EXPECT_EQ(original, unpacked);
}
