# // FILE: gpu-image-suite/src/main.cpp
#include <iostream>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "cli.hpp"
#include "timers.hpp"
#include "cpu_filters.hpp"
#include "gpu_filters.hpp"
using std::string;

static void usage(){
    std::cerr << "Usage: gpu_image_suite --filter [blur|sobel|sharpen|histeq] --impl [cpu|gpu]\n"
                 "                   --input <path> --output <path> [--radius N --sigma S]\n";
}

CLI CLI::parse(int argc, char** argv){
    CLI c; for(int i=1;i<argc;++i){ string k=argv[i];
        auto need=[&](int j){ if(i+1>=argc) throw std::runtime_error("Missing arg for "+k); return string(argv[++i]); };
        if(k=="--filter") c.filter = need(i);
        else if(k=="--impl") c.impl = need(i);
        else if(k=="--input") c.input = need(i);
        else if(k=="--output") c.output = need(i);
        else if(k=="--radius") c.radius = std::stoi(need(i));
        else if(k=="--sigma") c.sigma = std::stod(need(i));
        else { std::cerr << "Unknown arg: "<<k<<"\n"; }
    }
    if(c.input.empty()||c.output.empty()) throw std::runtime_error("input/output required");
    return c;
}

int main(int argc, char** argv){
    try{
        CLI cli = CLI::parse(argc, argv);
        cv::Mat img = cv::imread(cli.input, cv::IMREAD_COLOR);
        if (img.empty()) { std::cerr << "Failed to read input image\n"; return 1; }

        cv::Mat out_cpu, out_gpu; double cpu_ms=0; float gpu_ms=0;

        if (cli.filter=="blur"){
            CPUTimer t; t.begin(); out_cpu = cpu::gaussian_blur(img, cli.radius, cli.sigma); cpu_ms = t.end();
            GPUTimer g; g.begin(); out_gpu = gpu::gaussian_blur(img, cli.radius, cli.sigma); gpu_ms = g.end();
        } else if (cli.filter=="sobel"){
            CPUTimer t; t.begin(); out_cpu = cpu::sobel_edge(img); cpu_ms = t.end();
            GPUTimer g; g.begin(); out_gpu = gpu::sobel_edge(img); gpu_ms = g.end();
        } else if (cli.filter=="sharpen"){
            CPUTimer t; t.begin(); out_cpu = cpu::sharpen_unsharp(img, cli.radius, cli.sigma); cpu_ms = t.end();
            GPUTimer g; g.begin(); out_gpu = gpu::sharpen_unsharp(img, cli.radius, cli.sigma); gpu_ms = g.end();
        } else if (cli.filter=="histeq"){
            CPUTimer t; t.begin(); out_cpu = cpu::hist_equalize(img); cpu_ms = t.end();
            GPUTimer g; g.begin(); out_gpu = gpu::hist_equalize(img); gpu_ms = g.end();
        } else { usage(); return 2; }

        cv::Mat to_save = (cli.impl=="cpu") ? out_cpu : out_gpu;
        std::filesystem::create_directories(std::filesystem::path(cli.output).parent_path());
        cv::imwrite(cli.output, to_save);
        std::cout << "CPU_ms," << cpu_ms << "\nGPU_kernel_ms," << gpu_ms << "\n";
        return 0;
    } catch(const std::exception& e){
        std::cerr << "Error: " << e.what() << "\n"; usage(); return 1;
    }
}
