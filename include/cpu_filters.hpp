# // FILE: gpu-image-suite/include/cpu_filters.hpp
#pragma once
#include <opencv2/core.hpp>
namespace cpu {
cv::Mat gaussian_blur(const cv::Mat& img, int radius, double sigma);
cv::Mat sobel_edge(const cv::Mat& img);
cv::Mat sharpen_unsharp(const cv::Mat& img, int radius, double sigma, double amount=1.0);
cv::Mat hist_equalize(const cv::Mat& img);
}
