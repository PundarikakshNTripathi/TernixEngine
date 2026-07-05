#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

namespace ternix {

struct Tensor {
    std::vector<size_t> shape;
    size_t size;
    void* data; // 32-byte aligned memory
    
    Tensor(const std::vector<size_t>& shape_, size_t elem_size);
    ~Tensor();
    
    // Disable copy
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    
    // Allow move
    Tensor(Tensor&&) noexcept;
    Tensor& operator=(Tensor&&) noexcept;
};

} // namespace ternix
