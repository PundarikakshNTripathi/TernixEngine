#include "ternix/mul_mat_ternary.h"
#include <memory>

namespace ternix {

void mul_mat_ternary_naive(
    const int8_t* __restrict A,
    const uint8_t* __restrict B_packed,
    int32_t* __restrict C,
    size_t M, size_t K, size_t N) 
{
    // Ensure memory is 32-byte aligned per project directives
    const int8_t* aligned_A = std::assume_aligned<32>(A);
    const uint8_t* aligned_B = std::assume_aligned<32>(B_packed);
    int32_t* aligned_C = std::assume_aligned<32>(C);

    // B is packed with 4 weights per byte.
    // For this naive implementation, we assume a sequential packed layout:
    // Bits 0-1: weight 0, Bits 2-3: weight 1, Bits 4-5: weight 2, Bits 6-7: weight 3
    // Binary mapping:
    // 00 -> 0
    // 01 -> 1
    // 11 -> -1
    // 10 -> (unused/reserved)

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            int32_t sum = 0;
            for (size_t k = 0; k < K; ++k) {
                // Determine the byte index and bit offset for the specific weight
                // Assuming B is stored in row-major format, but packed along K
                size_t k_byte = k / 4;
                size_t bit_offset = (k % 4) * 2;
                
                // The byte offset is (k_byte * N) + j
                uint8_t packed_val = aligned_B[k_byte * N + j];
                uint8_t weight_bits = (packed_val >> bit_offset) & 0x03;
                
                int8_t weight = 0;
                if (weight_bits == 1) {
                    weight = 1;
                } else if (weight_bits == 3) {
                    weight = -1;
                }
                
                sum += aligned_A[i * K + k] * weight;
            }
            aligned_C[i * N + j] = sum;
        }
    }
}

} // namespace ternix
