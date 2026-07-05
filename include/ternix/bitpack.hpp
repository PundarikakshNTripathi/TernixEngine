#pragma once
#include <cstdint>
#include <cstddef>

namespace ternix {
    void pack_ternary_weights(uint8_t* packed, const int8_t* unpacked, size_t size);
    void unpack_ternary_weights(int8_t* unpacked, const uint8_t* packed, size_t size);
    void interleave_weights_offline(uint8_t* interleaved, const uint8_t* packed, size_t size);
}
