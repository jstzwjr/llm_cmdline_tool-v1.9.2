#pragma once

#include <opencv2/opencv.hpp>

namespace mtk::image_utils {

// BGR format
constexpr float32_t kOpenAICLIPMean[3] = {0.40821073, 0.4578275, 0.48145466};
constexpr float32_t kOpenAICLIPStd[3] = {0.27577711, 0.26130258, 0.26862954};
constexpr int32_t kImgSize = 336;
constexpr int32_t kCropSize[2] = {336, 336};
constexpr float32_t kScale = 0.00392156862745098; // 1 / 255

void normalize(cv::Mat& image, const float* mean, const float* std);

// Resize should be bicubic in LLaVA
void resize(cv::Mat& image, const int size, cv::InterpolationFlags interpolation = cv::INTER_CUBIC);

void center_crop(cv::Mat& image, const int* crop_size);

void rescale(cv::Mat& image, const float scale);

// From image file path
cv::Mat clip_preprocess(const std::string& imgPath, size_t& imageSizeBytes, const int size,
                        const int* crop_size, const float scale,
                        const float* mean = kOpenAICLIPMean, const float* std = kOpenAICLIPStd,
                        cv::InterpolationFlags interpolation = cv::INTER_CUBIC);

// From image buffer
cv::Mat clip_preprocess(const void* imgBuffer, const size_t imgBufferSize, size_t& imageSizeBytes,
                        const int size, const int* crop_size, const float scale,
                        const float* mean = kOpenAICLIPMean, const float* std = kOpenAICLIPStd,
                        cv::InterpolationFlags interpolation = cv::INTER_CUBIC);

// Qwen-VL preprocessing: resize + temporal tile + patch reshape
// resizeMode: "stretch" (direct resize) or "padding" (aspect-ratio preserve + black padding)
// Output shape: [num_patches, patch_features] as float32
cv::Mat qwen_vl_preprocess(const std::string& imgPath, size_t& outputSizeBytes,
                           int imageWidth, int imageHeight,
                           int patchSize, int temporalPatchSize, int mergeSize,
                           const float* mean, const float* std,
                           const std::string& resizeMode = "stretch");

cv::Mat qwen_vl_preprocess(const void* imgBuffer, size_t imgBufferSize, size_t& outputSizeBytes,
                           int imageWidth, int imageHeight,
                           int patchSize, int temporalPatchSize, int mergeSize,
                           const float* mean, const float* std,
                           const std::string& resizeMode = "stretch");

} // namespace mtk::image_utils