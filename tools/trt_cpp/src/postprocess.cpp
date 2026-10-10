#include "postprocess.h"

#include <algorithm>
#include <cmath>

namespace deimv2 {
namespace {

inline float sigmoid(float x) {
    // 分段实现,避免 exp 溢出
    if (x >= 0.f) return 1.f / (1.f + std::exp(-x));
    const float e = std::exp(x);
    return e / (1.f + e);
}

inline float boxArea(const Detection& d) {
    return (d.x2 - d.x1) * (d.y2 - d.y1);
}

inline float iou(const Detection& a, const Detection& b) {
    const float ix1 = std::max(a.x1, b.x1);
    const float iy1 = std::max(a.y1, b.y1);
    const float ix2 = std::min(a.x2, b.x2);
    const float iy2 = std::min(a.y2, b.y2);
    const float iw = ix2 - ix1;
    const float ih = iy2 - iy1;
    if (iw <= 0.f || ih <= 0.f) return 0.f;
    const float inter = iw * ih;
    const float uni = boxArea(a) + boxArea(b) - inter;
    return uni > 0.f ? inter / uni : 0.f;
}

}  // namespace

std::vector<Detection> decodeOutput(const float* pred, int n_queries, int num_classes,
                                    const float scale[4], float score_thresh) {
    std::vector<Detection> dets;
    dets.reserve(static_cast<size_t>(n_queries));

    for (int q = 0; q < n_queries; ++q) {
        const float* p = pred + static_cast<size_t>(q) * (4 + num_classes);

        int best = 0;
        float best_score = -1.f;
        for (int c = 0; c < num_classes; ++c) {
            const float s = sigmoid(p[4 + c]);
            if (s > best_score) {
                best_score = s;
                best = c;
            }
        }
        if (best_score < score_thresh) continue;

        // cxcywh(相对输入图,0~1)→ 原图像素 xyxy
        const float cx = p[0] * scale[0];
        const float cy = p[1] * scale[1];
        const float w = p[2] * scale[2];
        const float h = p[3] * scale[3];

        Detection d;
        d.x1 = cx - 0.5f * w;
        d.y1 = cy - 0.5f * h;
        d.x2 = cx + 0.5f * w;
        d.y2 = cy + 0.5f * h;
        d.score = best_score;
        d.label = best;
        dets.push_back(d);
    }
    return dets;
}

void nmsClassAgnostic(std::vector<Detection>& dets, float iou_thresh) {
    if (iou_thresh <= 0.f || dets.size() <= 1) return;

    std::stable_sort(dets.begin(), dets.end(),
                     [](const Detection& a, const Detection& b) { return a.score > b.score; });

    std::vector<char> suppressed(dets.size(), 0);
    std::vector<Detection> kept;
    kept.reserve(dets.size());
    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        kept.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (suppressed[j]) continue;
            if (iou(dets[i], dets[j]) > iou_thresh) suppressed[j] = 1;
        }
    }
    dets.swap(kept);
}

}  // namespace deimv2
