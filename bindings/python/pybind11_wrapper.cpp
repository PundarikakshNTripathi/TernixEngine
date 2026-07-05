#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ternix/model.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ternix_engine, m) {
    m.doc() = "TernixEngine Python Bindings for 1.58-bit LLM Inference";

    py::class_<ternix::Model>(m, "Model")
        .def(py::init<const std::string &>())
        .def("load", &ternix::Model::load)
        .def_readwrite("vocab_size", &ternix::Model::vocab_size)
        .def_readwrite("hidden_size", &ternix::Model::hidden_size)
        .def_readwrite("num_layers", &ternix::Model::num_layers);
}
