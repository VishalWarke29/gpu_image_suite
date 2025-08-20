# GPU Image Suite

A **GPU-accelerated image processing toolkit** that demonstrates common computer vision filters implemented in both **CPU (OpenCV)** and **GPU (CUDA)**.  
It provides side-by-side performance benchmarks and visual outputs to showcase the benefits of GPU parallelism.

---

## ✨ Features

- **Gaussian Blur (separable convolution)**
- **Sobel Edge Detection**
- **Unsharp Mask Sharpening**
- **Histogram Equalization**
- CPU reference implementations (OpenCV)
- GPU implementations (CUDA kernels)
- Built-in timing (CPU vs GPU)

---

## 📂 Project Structure

```
gpu-image-suite/
├── include/           # Header files
│   ├── cli.hpp        # Command-line parser
│   ├── timers.hpp     # CPU/GPU timers
│   ├── cpu_filters.hpp
│   └── gpu_filters.hpp
├── src/               # Implementation files
│   ├── main.cpp       # CLI entry point
│   ├── cpu_filters.cpp
│   └── gpu_filters.cu
├── data/
│   ├── input/         # Input test images
│   └── output/        # Filter results
└── CMakeLists.txt     # Build configuration
```

---

## 🚀 Build & Run

### 📌 Local (Linux with CUDA + OpenCV)

```bash
git clone https://github.com/VishalWarke29/gpu-image-suite.git
cd gpu-image-suite
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
```

Run a filter:

```bash
./gpu_image_suite   --filter blur   --impl gpu   --input ../data/input/test.jpg   --output ../data/output/test_blur.png   --radius 3 --sigma 1.5
```

---

### 📌 Google Colab

1. Open the Colab notebook provided in this repo (or create one).  
2. Run setup cells to install OpenCV, build the project, and download a test image.  
3. Example:

```bash
cd gpu-image-suite/build
./gpu_image_suite   --filter sobel   --impl gpu   --input ../data/input/test.jpg   --output ../data/output/test_sobel.png
```

---

## 📊 Example Output

- **CPU vs GPU runtime (ms)** is printed for each run:
  ```
  CPU_ms, 23.41
  GPU_kernel_ms, 1.92
  ```

- Result images are saved in `data/output/` (and `final_outputs/` in Colab).

---

## 💡 Skills Demonstrated

- CUDA programming (memory transfers, kernels, shared/constant memory)
- Image processing algorithms (blur, edge detection, sharpening, histogram equalization)
- Performance benchmarking (CPU vs GPU)
- Modern CMake + OpenCV integration
- Reproducible builds (local + Colab)

---
