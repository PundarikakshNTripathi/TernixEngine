#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "ternix/mul_mat_ternary.h"
#include <stdexcept>

namespace py = pybind11;

void py_mul_mat_ternary_avx2(
    py::array_t<int8_t> A,
    py::array_t<uint8_t> B_packed,
    py::array_t<int32_t> C,
    size_t M, size_t K, size_t N) 
{
    py::buffer_info buf_A = A.request();
    py::buffer_info buf_B = B_packed.request();
    py::buffer_info buf_C = C.request();

    if (buf_A.ndim != 2 || buf_B.ndim != 2 || buf_C.ndim != 2) {
        throw std::runtime_error("Number of dimensions must be 2");
    }

    auto ptr_A = static_cast<int8_t*>(buf_A.ptr);
    auto ptr_B = static_cast<uint8_t*>(buf_B.ptr);
    auto ptr_C = static_cast<int32_t*>(buf_C.ptr);

    ternix::mul_mat_ternary_avx2(ptr_A, ptr_B, ptr_C, M, K, N);
}

PYBIND11_MODULE(ternix_core, m) {
    m.doc() = "TernixEngine: 1.58-bit SIMD-Native Inference Engine Python Bindings";
    m.def("mul_mat_ternary_avx2", &py_mul_mat_ternary_avx2, "AVX2 SIMD-native ternary matrix multiplication",
          py::arg("A"), py::arg("B_packed"), py::arg("C"), py::arg("M"), py::arg("K"), py::arg("N"));
}
