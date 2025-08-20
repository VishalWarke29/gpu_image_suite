# // FILE: gpu-image-suite/include/timers.hpp
#pragma once
#include <chrono>

// CPU timer (uses std::chrono)
struct CPUTimer {
    std::chrono::high_resolution_clock::time_point t0;
    void begin(){ t0 = std::chrono::high_resolution_clock::now(); }
    double end(){ auto t1=std::chrono::high_resolution_clock::now();
      return std::chrono::duration<double, std::milli>(t1 - t0).count(); }
};

// GPU timer stub (so it compiles even if CUDA headers are missing)
// If you install CUDA headers later, you can implement real cudaEvent timing.
struct GPUTimer {
    void begin() {}
    float end() { return 0.0f; }
};
