#pragma once
#include <string>
#include <vector>
#include <memory>
#include "ternix/tensor.hpp"

namespace ternix {

class Model {
public:
    Model(const std::string& path);
    ~Model();
    
    void load();
    
    size_t vocab_size;
    size_t hidden_size;
    size_t num_layers;

private:
    std::string file_path;
    std::unique_ptr<Tensor> embedding;
};

} // namespace ternix
