#include "ternix/mul_mat_ternary.h"
#include <memory>
#include <immintrin.h>

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

void mul_mat_ternary_avx2(
    const int8_t* __restrict A,
    const uint8_t* __restrict B_packed,
    int32_t* __restrict C,
    size_t M, size_t K, size_t N)
{
    // Ensure memory is 32-byte aligned per project directives
    const int8_t* aligned_A = std::assume_aligned<32>(A);
    const uint8_t* aligned_B = std::assume_aligned<32>(B_packed);
    int32_t* aligned_C = std::assume_aligned<32>(C);

    // K must be a multiple of 32 for the 8-lane processing x 4 unrolling (or we just process 8 elements at a time)
    // We will process K in chunks of 8 elements.
    // 8 packed 2-bit weights = 16 bits = 1 uint16_t.

    // Predefine shift vector for in-register weight unpacking
    // AVX2 register has 8x 32-bit lanes. We map one weight to each lane.
    // Lane 0: shift 0, Lane 1: shift 2, ..., Lane 7: shift 14.
    const __m256i shift_vec = _mm256_set_epi32(14, 12, 10, 8, 6, 4, 2, 0);

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            __m256i sum_vec = _mm256_setzero_si256();
            
            for (size_t k = 0; k < K; k += 8) {
                // 1. Load 8 INT8 activations and upcast to 32-bit integers
                // We use _mm_loadl_epi64 to load 8 bytes into a 128-bit register
                __m128i act8 = _mm_loadl_epi64((const __m128i*)(aligned_A + i * K + k));
                __m256i act32 = _mm256_cvtepi8_epi32(act8);
                
                // 2. Load 8 packed 2-bit weights (16 bits)
                // The byte offset is (k / 4) * N + j. Since k is a multiple of 8, k/4 is even.
                // We read 2 consecutive bytes (for k and k+4).
                // Assuming B is structured such that the 2 bytes for the 8 weights are contiguous 
                // in the K dimension. If B is row-major (KxN), then they are N bytes apart!
                // Wait, if B is KxN and packed along K, the bytes are contiguous along K?
                // No, if B is KxN, contiguous elements are along N.
                // To load along K, we'd have to read B_packed[(k/4)*N + j] and B_packed[((k+4)/4)*N + j].
                // This is a strided load.
                // Let's read them individually and combine into a uint16_t.
                uint8_t byte0 = aligned_B[(k / 4) * N + j];
                uint8_t byte1 = aligned_B[((k + 4) / 4) * N + j];
                uint16_t packed_w = (uint16_t(byte1) << 8) | byte0;
                
                // Broadcast the 16-bit packed weights to all 8 32-bit lanes
                __m256i w_packed = _mm256_set1_epi32(packed_w);
                
                // 3. In-Register Weight Unpacking (SIMD-Native)
                // Extract the 2-bit weights
                __m256i w_shifted = _mm256_srlv_epi32(w_packed, shift_vec);
                __m256i w_2bit = _mm256_and_si256(w_shifted, _mm256_set1_epi32(3));
                
                // 4. Branchless Masking
                // Map w_2bit (00b -> 0, 01b -> 1, 11b -> 3) to sign_control (0, 1, -1)
                // Shift trick: shift left by 30, then arithmetic shift right by 30
                // 3 << 30 = 0xC0000000 -> srai -> 0xFFFFFFFF (-1)
                // 1 << 30 = 0x40000000 -> srai -> 0x00000001 (1)
                // 0 << 30 = 0x00000000 -> srai -> 0x00000000 (0)
                __m256i sign_control = _mm256_srai_epi32(_mm256_slli_epi32(w_2bit, 30), 30);
                
                // 5. Accumulation
                // _mm256_sign_epi32 acts upon the INT32 activations.
                // If sign_control is 0, it zeros out the activation (fulfilling the bitwise AND masking requirement intrinsically).
                __m256i computed = _mm256_sign_epi32(act32, sign_control);
                sum_vec = _mm256_add_epi32(sum_vec, computed);
            }
            
            // Horizontal reduction of sum_vec
            // We need to sum all 8 32-bit integers in sum_vec
            __m128i sum_high = _mm256_extracti128_si256(sum_vec, 1);
            __m128i sum_low  = _mm256_castsi256_si128(sum_vec);
            __m128i sum128   = _mm_add_epi32(sum_high, sum_low);
            
            sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
            sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1)));
            
            aligned_C[i * N + j] = _mm_cvtsi128_si32(sum128);
        }
    }
}

} // namespace ternix
