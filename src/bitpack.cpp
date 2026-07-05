#include "ternix/bitpack.hpp"

namespace ternix {

void pack_ternary_weights(uint8_t* packed, const int8_t* unpacked, size_t size) {
    for (size_t i = 0; i < size / 4; ++i) {
        uint8_t b = 0;
        for (int j = 0; j < 4; ++j) {
            int8_t val = unpacked[i * 4 + j];
            uint8_t bits = (val == 1) ? 1 : ((val == -1) ? 3 : 0);
            b |= (bits << (j * 2));
        }
        packed[i] = b;
    }
}

void unpack_ternary_weights(int8_t* unpacked, const uint8_t* packed, size_t size) {
    for (size_t i = 0; i < size / 4; ++i) {
        uint8_t b = packed[i];
        for (int j = 0; j < 4; ++j) {
            uint8_t bits = (b >> (j * 2)) & 0x3;
            if (bits == 1) unpacked[i * 4 + j] = 1;
            else if (bits == 3) unpacked[i * 4 + j] = -1;
            else unpacked[i * 4 + j] = 0;
        }
    }
}

void interleave_weights_offline(uint8_t* interleaved, const uint8_t* packed, size_t size) {
    for (size_t i = 0; i < size / 4; ++i) {
        interleaved[i] = packed[i]; 
    }
}

} // namespace ternix
