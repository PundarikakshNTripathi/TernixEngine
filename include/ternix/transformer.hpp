#pragma once
#include "ternix/tensor.hpp"

namespace ternix {

struct TransformerLayer {
    Tensor* wq;
    Tensor* wk;
    Tensor* wv;
    Tensor* wo;
    Tensor* rms_norm_weight;
    
    void forward(Tensor* out, const Tensor* in);
};

} // namespace ternix
