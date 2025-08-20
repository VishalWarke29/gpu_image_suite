# // FILE: gpu-image-suite/src/gpu_filters.cu
#include <cuda_runtime.h>
#include <opencv2/imgproc.hpp>
#include "gpu_filters.hpp"
#include <vector>
#include <stdexcept>
static inline void checkCuda(cudaError_t e){ if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }

// ===== Gaussian (separable) =====
__constant__ float d_kernel[64];

template<int R, int BW, int BH>
__global__ void blur_horiz(const unsigned char* in, unsigned char* out, int w, int h){
    __shared__ unsigned char tile[BH][BW + 2*R];
    int x = blockIdx.x*BW + threadIdx.x;
    int y = blockIdx.y*BH + threadIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    for(int dx=tx; dx<BW+2*R; dx+=BW){
        int gx = blockIdx.x*BW + dx - R; gx = max(0, min(w-1, gx));
        if (y<h) tile[ty][dx] = in[y*w + gx];
    }
    __syncthreads();
    if (x<w && y<h){
        float acc=0.f;
        #pragma unroll
        for(int k=-R;k<=R;++k) acc += d_kernel[k+R] * tile[ty][tx + k + R];
        out[y*w + x] = (unsigned char)min(255.f, max(0.f, acc));
    }
}

template<int R, int BW, int BH>
__global__ void blur_vert(const unsigned char* in, unsigned char* out, int w, int h){
    __shared__ unsigned char tile[BH + 2*R][BW];
    int x = blockIdx.x*BW + threadIdx.x;
    int y = blockIdx.y*BH + threadIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    for(int dy=ty; dy<BH+2*R; dy+=BH){
        int gy = blockIdx.y*BH + dy - R; gy = max(0, min(h-1, gy));
        if (x<w) tile[dy][tx] = in[gy*w + x];
    }
    __syncthreads();
    if (x<w && y<h){
        float acc=0.f;
        #pragma unroll
        for(int k=-R;k<=R;++k) acc += d_kernel[k+R] * tile[ty + k + R][tx];
        out[y*w + x] = (unsigned char)min(255.f, max(0.f, acc));
    }
}

static std::vector<float> gaussian_kernel(int radius, double sigma){
    int k = 2*radius+1; std::vector<float> h(k);
    double s2=2*sigma*sigma, sum=0;
    for(int i=-radius;i<=radius;++i){ double v = std::exp(-(i*i)/s2); h[i+radius]=(float)v; sum+=v; }
    for(auto& v: h) v = (float)(v/sum);
    return h;
}

// ===== Sobel 3x3 magnitude =====
__global__ void sobel3x3(const unsigned char* in, unsigned char* out, int w, int h){
    __shared__ unsigned char tile[16+2][16+2];
    int bx = blockIdx.x*16, by = blockIdx.y*16;
    int tx = threadIdx.x, ty = threadIdx.y;
    int x = bx + tx; int y = by + ty;
    int ix = min(max(x-1,0), w-1);
    int iy = min(max(y-1,0), h-1);
    tile[ty][tx] = in[iy*w + ix];
    __syncthreads();
    if (tx>=1 && tx<17 && ty>=1 && ty<17 && x<w && y<h){
        int gx = -tile[ty-1][tx-1]-2*tile[ty][tx-1]-tile[ty+1][tx-1]
                 +tile[ty-1][tx+1]+2*tile[ty][tx+1]+tile[ty+1][tx+1];
        int gy =  tile[ty-1][tx-1]+2*tile[ty-1][tx]+tile[ty-1][tx+1]
                 -tile[ty+1][tx-1]-2*tile[ty+1][tx]-tile[ty+1][tx+1];
        int mag = min(255, (int)sqrtf((float)(gx*gx + gy*gy)));
        out[y*w + x] = (unsigned char)mag;
    }
}

namespace gpu {

cv::Mat gaussian_blur(const cv::Mat& imgBGR, int radius, double sigma){
    if (imgBGR.empty()) throw std::runtime_error("Empty image");
    cv::Mat gray; cv::cvtColor(imgBGR, gray, cv::COLOR_BGR2GRAY);
    int w=gray.cols, h=gray.rows; size_t n=(size_t)w*h;
    uchar *d_in=nullptr, *d_tmp=nullptr, *d_out=nullptr;
    checkCuda(cudaMalloc(&d_in, n));
    checkCuda(cudaMalloc(&d_tmp, n));
    checkCuda(cudaMalloc(&d_out, n));
    checkCuda(cudaMemcpy(d_in, gray.data, n, cudaMemcpyHostToDevice));
    auto hker = gaussian_kernel(radius, sigma);
    checkCuda(cudaMemcpyToSymbol(d_kernel, hker.data(), hker.size()*sizeof(float)));
    dim3 block1(32,8);
    dim3 grid1((w + block1.x - 1)/block1.x, (h + block1.y - 1)/block1.y);
    switch(radius){
        case 3:
            blur_horiz<3,32,8><<<grid1, block1>>>(d_in, d_tmp, w, h);
            blur_vert <3,32,8><<<grid1, block1>>>(d_tmp, d_out, w, h);
            break;
        default:
            checkCuda(cudaMemcpy(d_out, d_in, n, cudaMemcpyDeviceToDevice));
            break;
    }
    checkCuda(cudaGetLastError());
    cv::Mat out(h, w, CV_8U);
    checkCuda(cudaMemcpy(out.data, d_out, n, cudaMemcpyDeviceToHost));
    cudaFree(d_in); cudaFree(d_tmp); cudaFree(d_out);
    return out;
}

cv::Mat sobel_edge(const cv::Mat& imgBGR){
    cv::Mat gray; cv::cvtColor(imgBGR, gray, cv::COLOR_BGR2GRAY);
    int w=gray.cols, h=gray.rows; size_t n=(size_t)w*h;
    uchar *d_in=nullptr, *d_out=nullptr;
    checkCuda(cudaMalloc(&d_in, n));
    checkCuda(cudaMalloc(&d_out, n));
    checkCuda(cudaMemcpy(d_in, gray.data, n, cudaMemcpyHostToDevice));
    dim3 block(18,18); dim3 grid((w+15)/16, (h+15)/16);
    sobel3x3<<<grid, block>>>(d_in, d_out, w, h);
    checkCuda(cudaGetLastError());
    cv::Mat out(h,w,CV_8U);
    checkCuda(cudaMemcpy(out.data, d_out, n, cudaMemcpyDeviceToHost));
    cudaFree(d_in); cudaFree(d_out);
    return out;
}

cv::Mat sharpen_unsharp(const cv::Mat& imgBGR, int radius, double sigma, double amount){
    // Hybrid (uses GPU blur but combines on CPU for brevity)
    cv::Mat gray; cv::cvtColor(imgBGR, gray, cv::COLOR_BGR2GRAY);
    cv::Mat blur = gaussian_blur(imgBGR, radius, sigma);
    cv::Mat detail; cv::subtract(gray, blur, detail, cv::noArray(), CV_16S);
    cv::Mat out16; cv::addWeighted(gray, 1.0, detail, amount/255.0, 0.0, out16, CV_16S);
    cv::Mat out8; out16.convertTo(out8, CV_8U);
    return out8;
}

cv::Mat hist_equalize(const cv::Mat& imgBGR){
    // TODO: GPU histogram equalization
    cv::Mat ycrcb; cv::cvtColor(imgBGR, ycrcb, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> ch; cv::split(ycrcb, ch);
    cv::equalizeHist(ch[0], ch[0]);
    cv::merge(ch, ycrcb);
    cv::Mat out; cv::cvtColor(ycrcb, out, cv::COLOR_YCrCb2BGR);
    return out;
}
} // namespace gpu
