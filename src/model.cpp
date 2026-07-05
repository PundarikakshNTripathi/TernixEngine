#include "ternix/model.hpp"

namespace ternix {

Model::Model(const std::string& path) : file_path(path), vocab_size(32000), hidden_size(4096), num_layers(32) {
}

Model::~Model() {}

void Model::load() {
    std::vector<size_t> emb_shape = {vocab_size, hidden_size};
    embedding = std::make_unique<Tensor>(emb_shape, sizeof(float));
}

} // namespace ternix
