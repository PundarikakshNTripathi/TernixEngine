#include "ternix/tensor.hpp"
#include <immintrin.h>
#include <numeric>
#include <utility>

namespace ternix {

Tensor::Tensor(const std::vector<size_t>& shape_, size_t elem_size) : shape(shape_), data(nullptr) {
    size = std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<size_t>());
    data = _mm_malloc(size * elem_size, 32);
}

Tensor::~Tensor() {
    if (data) {
        _mm_free(data);
    }
}

Tensor::Tensor(Tensor&& other) noexcept : shape(std::move(other.shape)), size(other.size), data(other.data) {
    other.data = nullptr;
    other.size = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        if (data) _mm_free(data);
        shape = std::move(other.shape);
        size = other.size;
        data = other.data;
        other.data = nullptr;
        other.size = 0;
    }
    return *this;
}

} // namespace ternix
