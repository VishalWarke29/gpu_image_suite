# // FILE: gpu-image-suite/CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(gpu_image_suite LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Common Colab GPUs (T4=75, A100=80, L4=89)
set(CMAKE_CUDA_ARCHITECTURES 75;80;89)

find_package(OpenCV REQUIRED core imgcodecs imgproc highgui)

add_executable(gpu_image_suite
  src/main.cpp
  src/cpu_filters.cpp
  src/gpu_filters.cu
)

target_include_directories(gpu_image_suite PRIVATE include)
target_link_libraries(gpu_image_suite PRIVATE ${OpenCV_LIBS})

# Per-language flags (prevents leaking CUDA flags to C++)
target_compile_options(gpu_image_suite PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

set_target_properties(gpu_image_suite PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
