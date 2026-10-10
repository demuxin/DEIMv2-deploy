// 后处理:与 tools/inference/onnx_inf_v2.py 的 decode_output / class_agnostic_nms 一致。
//
//   output[B,N,4+C] -> sigmoid -> 每个 query 取最高分类别 -> 分数阈值过滤
//                   -> cxcywh(相对输入图 0~1)× 原图 (w,h) -> xyxy
//                   -> 全类别 NMS(忽略类别、只按框重合度过滤)
#pragma once

#include <cstdint>
#include <vector>

namespace deimv2 {

struct Detection {
    float x1 = 0.f, y1 = 0.f, x2 = 0.f, y2 = 0.f;
    float score = 0.f;
    int label = 0;
};

// pred: 单个 batch 的 [n_queries, 4 + num_classes] 连续数据
// scale: {w, h, w, h},把归一化的 cxcywh 还原回原图像素
std::vector<Detection> decodeOutput(const float* pred, int n_queries, int num_classes,
                                    const float scale[4], float score_thresh);

// 全类别 NMS。iou_thresh <= 0 时原样返回(与 Python 侧语义一致)。
void nmsClassAgnostic(std::vector<Detection>& dets, float iou_thresh);

}  // namespace deimv2
