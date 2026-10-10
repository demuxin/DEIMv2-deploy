#include "preprocess.h"

#include <algorithm>
#include <cmath>

namespace deimv2 {
namespace {

// 与 Pillow Resample.c 的 clip8 一致:x<=0 -> 0,x>=255 -> 255,否则截断 (x+0.5)
inline uint8_t clip8(float v) {
    if (v <= 0.f) return 0;
    if (v >= 255.f) return 255;
    return static_cast<uint8_t>(v + 0.5f);
}

}  // namespace

ResampleCoeffs precomputeCoeffs(int in_size, int out_size) {
    const double scale = static_cast<double>(in_size) / out_size;
    const double filterscale = std::max(1.0, scale);
    const double support = 1.0 * filterscale;          // 三角滤波器的 support = 1.0
    const double ss = 1.0 / filterscale;

    ResampleCoeffs c;
    c.ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
    c.bounds.assign(static_cast<size_t>(out_size) * 2, 0);
    c.weights.assign(static_cast<size_t>(out_size) * c.ksize, 0.0f);

    for (int i = 0; i < out_size; ++i) {
        const double center = (i + 0.5) * scale;
        int begin = static_cast<int>(center - support + 0.5);
        if (begin < 0) begin = 0;
        int end = static_cast<int>(center + support + 0.5);
        if (end > in_size) end = in_size;
        const int count = end - begin;

        float* w = &c.weights[static_cast<size_t>(i) * c.ksize];
        double sum = 0.0;
        for (int j = 0; j < count; ++j) {
            double t = (j + begin - center + 0.5) * ss;
            t = std::fabs(t);
            const double v = (t < 1.0) ? (1.0 - t) : 0.0;
            w[j] = static_cast<float>(v);
            sum += v;
        }
        if (sum != 0.0) {
            for (int j = 0; j < count; ++j) w[j] = static_cast<float>(w[j] / sum);
        }
        c.bounds[static_cast<size_t>(i) * 2] = begin;
        c.bounds[static_cast<size_t>(i) * 2 + 1] = begin + count;
    }
    return c;
}

void preprocess(const uint8_t* src, int src_h, int src_w,
                int dst_h, int dst_w,
                const NormParams& norm,
                std::vector<float>& dst) {
    const ResampleCoeffs cx = precomputeCoeffs(src_w, dst_w);
    const ResampleCoeffs cy = precomputeCoeffs(src_h, dst_h);

    // 第一趟:水平 src_w -> dst_w。
    // Pillow 的 8bpc 路径会把这一趟的结果直接量化成 uint8 再做垂直,必须跟着量化,
    // 否则与 PIL 的结果会差约 1 个灰阶(实测:不量化 max|d|≈1/255,量化后逐像素一致)。
    std::vector<uint8_t> tmp(static_cast<size_t>(src_h) * dst_w * 3);
    for (int y = 0; y < src_h; ++y) {
        const uint8_t* srow = src + static_cast<size_t>(y) * src_w * 3;
        uint8_t* trow = tmp.data() + static_cast<size_t>(y) * dst_w * 3;
        for (int x = 0; x < dst_w; ++x) {
            const int x0 = cx.bounds[x * 2];
            const int x1 = cx.bounds[x * 2 + 1];
            const float* w = &cx.weights[static_cast<size_t>(x) * cx.ksize];
            float r = 0.f, g = 0.f, b = 0.f;
            for (int i = x0; i < x1; ++i) {
                const uint8_t* p = srow + static_cast<size_t>(i) * 3;
                const float wi = w[i - x0];
                r += wi * p[0];
                g += wi * p[1];
                b += wi * p[2];
            }
            trow[x * 3 + 0] = clip8(r);
            trow[x * 3 + 1] = clip8(g);
            trow[x * 3 + 2] = clip8(b);
        }
    }

    // 第二趟:垂直 src_h -> dst_h,再 /255 + 归一化 + HWC -> CHW
    // 注意用除法而不是乘 1/255:参考实现(numpy)就是除,乘倒数会在 float32 末位引入差异
    dst.assign(static_cast<size_t>(3) * dst_h * dst_w, 0.f);
    for (int y = 0; y < dst_h; ++y) {
        const int y0 = cy.bounds[y * 2];
        const int y1 = cy.bounds[y * 2 + 1];
        const float* w = &cy.weights[static_cast<size_t>(y) * cy.ksize];
        for (int x = 0; x < dst_w; ++x) {
            float r = 0.f, g = 0.f, b = 0.f;
            for (int i = y0; i < y1; ++i) {
                const uint8_t* p = tmp.data() + (static_cast<size_t>(i) * dst_w + x) * 3;
                const float wi = w[i - y0];
                r += wi * p[0];
                g += wi * p[1];
                b += wi * p[2];
            }
            const float v[3] = {static_cast<float>(clip8(r)),
                                static_cast<float>(clip8(g)),
                                static_cast<float>(clip8(b))};
            for (int c = 0; c < 3; ++c) {
                float t = v[c] / 255.0f;
                if (norm.enable) t = (t - norm.mean[c]) / norm.std[c];
                dst[(static_cast<size_t>(c) * dst_h + y) * dst_w + x] = t;
            }
        }
    }
}

}  // namespace deimv2
