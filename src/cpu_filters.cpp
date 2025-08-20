# // FILE: gpu-image-suite/src/cpu_filters.cpp
#include "cpu_filters.hpp"
#include <opencv2/imgproc.hpp>
namespace cpu {
cv::Mat gaussian_blur(const cv::Mat& img, int radius, double sigma) {
    int k = 2*radius + 1;
    cv::Mat out; cv::GaussianBlur(img, out, cv::Size(k,k), sigma, sigma, cv::BORDER_REPLICATE);
    return out;
}
cv::Mat sobel_edge(const cv::Mat& img) {
    cv::Mat gray; cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::Mat gx, gy; cv::Sobel(gray, gx, CV_32F, 1, 0, 3); cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
    cv::Mat mag; cv::magnitude(gx, gy, mag);
    double mn, mx; cv::minMaxLoc(mag, &mn, &mx);
    cv::Mat out; mag.convertTo(out, CV_8U, 255.0/(mx+1e-6));
    return out;
}
cv::Mat sharpen_unsharp(const cv::Mat& img, int radius, double sigma, double amount) {
    cv::Mat blur = gaussian_blur(img, radius, sigma);
    cv::Mat detail = img - blur;
    cv::Mat out = img + amount * detail;
    out.convertTo(out, CV_8U);
    return out;
}
cv::Mat hist_equalize(const cv::Mat& img){
    cv::Mat ycrcb; cv::cvtColor(img, ycrcb, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> ch; cv::split(ycrcb, ch);
    cv::equalizeHist(ch[0], ch[0]);
    cv::merge(ch, ycrcb);
    cv::Mat out; cv::cvtColor(ycrcb, out, cv::COLOR_YCrCb2BGR);
    return out;
}
}
