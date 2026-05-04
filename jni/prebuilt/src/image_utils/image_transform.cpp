#include "image_transform.h"

#include <opencv2/opencv.hpp>

namespace mtk::image_utils {

void normalize(cv::Mat& image, const float* mean, const float* std) {
    if (image.channels() == 3 && image.type() != 21) {
        image.convertTo(image, CV_32F);
    }
    const int rows = image.rows;
    const int cols = image.cols;
    cv::Mat img_mean(rows, cols, CV_32FC3, cv::Scalar(mean[0], mean[1], mean[2]));
    cv::Mat img_std(rows, cols, CV_32FC3, cv::Scalar(std[0], std[1], std[2]));
    image -= img_mean;
    image /= img_std;
}

void resize(cv::Mat& image, const int size, cv::InterpolationFlags interpolation) {
    // TODO: Might need to support adaptive. Reference to
    // transformers.image_transform.get_resize_output_image_size
    const int rows = image.rows;
    const int cols = image.cols;
    int short_e, long_e;
    if (rows <= cols) {
        short_e = rows;
        long_e = cols;
    } else {
        short_e = cols;
        long_e = rows;
    }
    const int new_short = size;
    const int new_long = size * long_e / short_e;
    int new_cols, new_rows;
    if (rows <= cols) {
        new_cols = new_long;
        new_rows = new_short;
    } else {
        new_cols = new_short;
        new_rows = new_long;
    }
    cv::Size newsize(new_cols, new_rows);
    cv::resize(image, image, newsize, 0, 0, interpolation);
    image.convertTo(image, CV_8U);
}

void center_crop(cv::Mat& image, const int* crop_size) {
    const int rows = image.rows;
    const int cols = image.cols;
    const int crop_rows = crop_size[0];
    const int crop_cols = crop_size[1];
    int top = (rows - crop_rows) / 2;
    int bottom = top + crop_rows;
    int left = (cols - crop_cols) / 2;
    int right = left + crop_cols;

    if (top >= 0 && bottom <= rows && left >= 0 && right <= cols) {
        // If cropped area is within image boundaries
        image = image(cv::Range(top, bottom), cv::Range(left, right));
    } else {
        // If image is too small, pad it with zero...
        int new_rows = std::max(crop_rows, rows);
        int new_cols = std::max(crop_cols, cols);
        int top_pad = (new_rows - rows) / 2;
        int bottom_pad = top_pad + rows;
        int left_pad = (new_cols - cols) / 2;
        int right_pad = left_pad + cols;
        cv::copyMakeBorder(image, image, top_pad, bottom_pad, left_pad, right_pad,
                           cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

        top += top_pad;
        bottom += top_pad;
        left += left_pad;
        right += left_pad;
        image = image(cv::Range(std::max(0, top), std::min(new_rows, bottom)),
                      cv::Range(std::max(0, left), std::min(new_cols, right)));
    }
}

void rescale(cv::Mat& image, const float scale) {
    if (image.channels() == 3 && image.type() != 21) {
        image.convertTo(image, CV_32F);
    }
    image *= scale;
}

// Inplace preprocess
inline void clip_preprocess_impl(cv::Mat& image, size_t& imageSizeBytes, const int size,
                                 const int* crop_size, const float scale, const float* mean,
                                 const float* std, cv::InterpolationFlags interpolation) {
    // Convert to FP32 RGB
    cv::Mat imageRGB;
    cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);
    imageRGB.convertTo(imageRGB, CV_32F);
    image = imageRGB;

    resize(image, size, interpolation);
    center_crop(image, crop_size);
    rescale(image, scale);
    normalize(image, mean, std);
    if (!image.isContinuous()) {
        image = image.clone();
    }
    imageSizeBytes = image.total() * image.elemSize();
}

// From image file path
cv::Mat clip_preprocess(const std::string& imgPath, size_t& imageSizeBytes, const int size,
                        const int* crop_size, const float scale, const float* mean,
                        const float* std, cv::InterpolationFlags interpolation) {
    cv::Mat image = cv::imread(imgPath, cv::IMREAD_COLOR);
    clip_preprocess_impl(image, imageSizeBytes, size, crop_size, scale, mean, std, interpolation);
    return image;
}

// From image buffer
cv::Mat clip_preprocess(const void* imgBuffer, const size_t imgBufferSize, size_t& imageSizeBytes,
                        const int size, const int* crop_size, const float scale, const float* mean,
                        const float* std, cv::InterpolationFlags interpolation) {
    cv::Mat image = cv::imdecode(
        cv::Mat(1, imgBufferSize, CV_8UC1, const_cast<void*>(imgBuffer)), cv::IMREAD_COLOR);
    clip_preprocess_impl(image, imageSizeBytes, size, crop_size, scale, mean, std, interpolation);
    return image;
}



// Qwen-VL preprocessing implementation
static cv::Mat qwen_vl_preprocess_impl(cv::Mat& image, size_t& outputSizeBytes,
                                       int imageWidth, int imageHeight,
                                       int patchSize, int temporalPatchSize, int mergeSize,
                                       const float* mean, const float* std,
                                       const std::string& resizeMode) {
    cv::Mat imageRGB;
    cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);
    imageRGB.convertTo(imageRGB, CV_32F);

    if (resizeMode == "padding") {
        // Aspect-ratio preserving resize + center black padding
        float scale = std::min((float)imageWidth / imageRGB.cols,
                               (float)imageHeight / imageRGB.rows);
        int newW = ((int)(imageRGB.cols * scale) / patchSize) * patchSize;
        int newH = ((int)(imageRGB.rows * scale) / patchSize) * patchSize;
        if (newW <= 0) newW = patchSize;
        if (newH <= 0) newH = patchSize;
        cv::resize(imageRGB, imageRGB, cv::Size(newW, newH), 0, 0, cv::INTER_CUBIC);
        // Center pad to target size with black
        cv::Mat padded(imageHeight, imageWidth, CV_32FC3, cv::Scalar(0, 0, 0));
        int offsetX = (imageWidth - newW) / 2;
        int offsetY = (imageHeight - newH) / 2;
        imageRGB.copyTo(padded(cv::Rect(offsetX, offsetY, newW, newH)));
        imageRGB = padded;
    } else {
        // Direct stretch resize (default)
        cv::resize(imageRGB, imageRGB, cv::Size(imageWidth, imageHeight), 0, 0, cv::INTER_CUBIC);
    }

    imageRGB *= (1.0f / 255.0f);
    const int rows = imageRGB.rows;
    const int cols = imageRGB.cols;
    cv::Mat img_mean(rows, cols, CV_32FC3, cv::Scalar(mean[0], mean[1], mean[2]));
    cv::Mat img_std(rows, cols, CV_32FC3, cv::Scalar(std[0], std[1], std[2]));
    imageRGB -= img_mean;
    imageRGB /= img_std;

    std::vector<cv::Mat> channels(3);
    cv::split(imageRGB, channels);

    const int H = imageHeight, W = imageWidth, C = 3;
    const int T = temporalPatchSize, P = patchSize, M = mergeSize;
    const int grid_h = H / P, grid_w = W / P;
    const int num_patches = grid_h * grid_w;
    const int patch_features = C * T * P * P;

    cv::Mat output(num_patches, patch_features, CV_32F);
    float* out_ptr = (float*)output.data;
    std::vector<float*> ch_data(C);
    for (int c = 0; c < C; c++) ch_data[c] = (float*)channels[c].data;

    int patch_idx = 0;
    for (int gh_blk = 0; gh_blk < grid_h / M; gh_blk++) {
        for (int gw_blk = 0; gw_blk < grid_w / M; gw_blk++) {
            for (int mh = 0; mh < M; mh++) {
                for (int mw = 0; mw < M; mw++) {
                    float* dst = out_ptr + patch_idx * patch_features;
                    int feat_idx = 0;
                    for (int c = 0; c < C; c++) {
                        for (int t = 0; t < T; t++) {
                            for (int ph = 0; ph < P; ph++) {
                                for (int pw = 0; pw < P; pw++) {
                                    int row = (gh_blk * M + mh) * P + ph;
                                    int col = (gw_blk * M + mw) * P + pw;
                                    dst[feat_idx++] = ch_data[c][row * W + col];
                                }
                            }
                        }
                    }
                    patch_idx++;
                }
            }
        }
    }
    outputSizeBytes = num_patches * patch_features * sizeof(float);
    return output;
}

cv::Mat qwen_vl_preprocess(const std::string& imgPath, size_t& outputSizeBytes,
                           int imageWidth, int imageHeight,
                           int patchSize, int temporalPatchSize, int mergeSize,
                           const float* mean, const float* std,
                           const std::string& resizeMode) {
    cv::Mat image = cv::imread(imgPath, cv::IMREAD_COLOR);
    return qwen_vl_preprocess_impl(image, outputSizeBytes,
                                   imageWidth, imageHeight,
                                   patchSize, temporalPatchSize, mergeSize, mean, std,
                                   resizeMode);
}

cv::Mat qwen_vl_preprocess(const void* imgBuffer, size_t imgBufferSize, size_t& outputSizeBytes,
                           int imageWidth, int imageHeight,
                           int patchSize, int temporalPatchSize, int mergeSize,
                           const float* mean, const float* std,
                           const std::string& resizeMode) {
    cv::Mat image = cv::imdecode(
        cv::Mat(1, imgBufferSize, CV_8UC1, const_cast<void*>(imgBuffer)), cv::IMREAD_COLOR);
    return qwen_vl_preprocess_impl(image, outputSizeBytes,
                                   imageWidth, imageHeight,
                                   patchSize, temporalPatchSize, mergeSize, mean, std,
                                   resizeMode);
}

} // namespace mtk::image_utils