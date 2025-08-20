# // FILE: gpu-image-suite/include/cli.hpp
#pragma once
#include <string>
struct CLI {
    std::string filter = "blur";   // blur|sobel|sharpen|histeq
    std::string impl   = "gpu";    // cpu|gpu
    std::string input;             // path
    std::string output;            // path
    int radius = 3;                // for Gaussian
    double sigma = 1.5;
    static CLI parse(int argc, char** argv);
};
