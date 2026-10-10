// 预处理:与训练 val / tools/inference/onnx_inf_v2.py 完全一致的"直接拉伸"路径。
//
//   PIL 双线性 resize 到 (w, h)(非 letterbox、无 padding)
//   -> /255
//   -> (x - mean) / std(仅 deimv2_dinov3 需要,dfine 无该步)
//   -> HWC uint8 -> CHW float32,通道序 RGB
#pragma once

#include <cstdint>
#include <vector>

namespace deimv2 {

// 归一化参数。enable=false 时只做 /255(对应 deim_dfine 的 val 增强)。
struct NormParams {
    bool enable = true;
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std[3] = {0.229f, 0.224f, 0.225f};
};

// PIL 风格的双线性重采样系数(与 Pillow ImagingResample 的 precompute_coeffs 同公式):
//   scale        = in_size / out_size
//   filterscale  = max(1, scale)          <- 缩小图时把三角滤波核展宽,即抗锯齿
//   support      = 1.0 * filterscale
//   center       = (i + 0.5) * scale
//   w(j)         = triangle((j + 0.5 - center) / filterscale),再归一化到 sum = 1
//
// 两趟的顺序与中间量化也很关键:Pillow 是"水平 -> 量化 uint8 -> 垂直 -> 量化 uint8",
// 实测这样能与其输出逐像素一致,而先垂直或中间不量化都会差约 1 个灰阶。
struct ResampleCoeffs {
    int ksize = 0;
    std::vector<int> bounds;     // 每个输出位置一对 [begin, end)
    std::vector<float> weights;  // out_size * ksize
};

ResampleCoeffs precomputeCoeffs(int in_size, int out_size);

// src: host 端 RGB 连续内存 [src_h, src_w, 3] uint8
// dst: 填充为 [3, dst_h, dst_w] float32(CHW,RGB)
void preprocess(const uint8_t* src, int src_h, int src_w,
                int dst_h, int dst_w,
                const NormParams& norm,
                std::vector<float>& dst);

}  // namespace deimv2
