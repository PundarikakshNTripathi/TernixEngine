#include <iostream>
#include "ternix/model.hpp"

int main(int argc, char** argv) {
    std::cout << "TernixEngine: 1.58-bit SIMD-Native Inference Engine" << std::endl;
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return 1;
    }
    
    ternix::Model model(argv[1]);
    model.load();
    
    std::cout << "Model loaded successfully." << std::endl;
    return 0;
}
