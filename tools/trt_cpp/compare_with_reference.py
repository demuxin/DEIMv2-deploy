"""比较 C++ TensorRT 与 Python 参考(onnx_inf_v2.py 的实现)的结果。

需要 numpy(在训练容器里跑即可)。输入是两边导出的裸 f32:
  --cpp-input / --ref-input    预处理后的输入张量 -> 核对预处理
  --cpp-output / --ref-output  模型原始输出       -> 核对推理 + 后处理

判据:
  1) 输入逐元素一致(预处理与 PIL 对齐)
  2) 输出在 sigmoid 之后的分数差异极小
  3) 两边各自 decode+NMS 得到的检测框一一对应(IoU 高、分数接近)

用法:
  python3 tools/trt_cpp/compare_with_reference.py \
      --cpp-input z_trt_build/cpp_in_l.bin --ref-input z_trt_build/ref_in_l.bin \
      --cpp-output z_trt_build/cpp_out_l.bin --ref-output z_trt_build/ref_out_l.bin \
      --image-size 896 512 --thrh 0.25 --nms-iou 0.9
"""

import argparse
import os

import numpy as np


def sigmoid(x):
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.abs(x))), np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))))


def decode(pred, orig_wh, thrh):
    """pred: [N, 4+C] -> [(x1,y1,x2,y2,score,label)],与 onnx_inf_v2.py 的 decode_output 一致。"""
    w, h = orig_wh
    scores_all = sigmoid(pred[:, 4:])
    labels = scores_all.argmax(axis=1)
    scores = scores_all.max(axis=1)
    keep = scores >= thrh
    b = pred[keep, :4].copy()
    b[:, 0] *= w
    b[:, 1] *= h
    b[:, 2] *= w
    b[:, 3] *= h
    cx, cy, bw, bh = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    out = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    return [(out[i], float(scores[keep][i]), int(labels[keep][i])) for i in range(len(out))]


def nms(dets, iou_thr):
    """全类别 NMS,与 torchvision.ops.nms 语义一致(去掉 IoU > thr 的低分框)。"""
    if iou_thr <= 0 or not dets:
        return dets
    boxes = np.array([d[0] for d in dets], np.float64)
    scores = np.array([d[1] for d in dets], np.float64)
    order = scores.argsort()[::-1]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        iw = np.maximum(0.0, xx2 - xx1)
        ih = np.maximum(0.0, yy2 - yy1)
        inter = iw * ih
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
        order = rest[iou <= iou_thr]
    return [dets[i] for i in keep]


def load_shape(ref_output, fallback, total):
    """形状优先取 <ref_output>.shape 旁文件(verify_reference.py 会写),否则用 --output-shape。"""
    sidecar = ref_output + ".shape"
    shape = None
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            shape = tuple(int(x) for x in f.read().split())
    elif fallback:
        shape = tuple(fallback)
    if shape is None:
        raise SystemExit(f"缺少 {sidecar},请加 --output-shape (如 --output-shape 1 300 85)")
    if int(np.prod(shape)) != total:
        raise SystemExit(f"形状 {shape} 与元素数 {total} 不匹配")
    return shape


def iou_pair(a, b):
    xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
    xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, xx2 - xx1), max(0.0, yy2 - yy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cpp-input", required=True)
    p.add_argument("--ref-input", required=True)
    p.add_argument("--cpp-output", required=True)
    p.add_argument("--ref-output", required=True)
    p.add_argument("--image-size", type=int, nargs=2, required=True, metavar=("W", "H"),
                   help="原图尺寸(宽 高),用于把归一化框还原到像素")
    p.add_argument("--output-shape", type=int, nargs=3, default=None, metavar=("B", "N", "C"),
                   help="输出张量形状(仅当缺少 <ref-output>.shape 旁文件时需要)")
    p.add_argument("--thrh", type=float, default=0.25)
    p.add_argument("--nms-iou", type=float, default=0.9)
    args = p.parse_args()

    ok = True

    # ---- 1. 预处理 ----
    ci = np.fromfile(args.cpp_input, np.float32)
    ri = np.fromfile(args.ref_input, np.float32)
    if ci.size != ri.size:
        print(f"[FAIL] 输入大小不一致: cpp={ci.size} ref={ri.size}")
        return 1
    din = np.abs(ci - ri)
    print(f"[1] 预处理  n={ci.size}  max|d|={din.max():.3e}  mean|d|={din.mean():.3e}  "
          f"不同元素={int((din > 0).sum())}")
    if din.max() > 0:
        ok = False
        print("      ^ 预处理与 PIL 参考不完全一致")

    # ---- 2. 原始输出 ----
    co = np.fromfile(args.cpp_output, np.float32)
    ro = np.fromfile(args.ref_output, np.float32)
    if co.size != ro.size:
        print(f"[FAIL] 输出大小不一致: cpp={co.size} ref={ro.size}")
        return 1
    shape = load_shape(args.ref_output, args.output_shape, co.size)
    cout = co.reshape(shape)
    rout = ro.reshape(shape)
    n_queries, n_cols = shape[1], shape[2]
    num_classes = n_cols - 4

    dsig = np.abs(sigmoid(cout) - sigmoid(rout))
    dbox = np.abs(cout[:, :, :4] - rout[:, :, :4])
    print(f"[2] 推理    shape={shape}  query={n_queries}  类别={num_classes}")
    print(f"           sigmoid 分数  max|d|={dsig.max():.3e}  mean|d|={dsig.mean():.3e}")
    print(f"           框(0~1)      max|d|={dbox.max():.3e}  mean|d|={dbox.mean():.3e}")
    # 分数差异超过 1e-3 就说明不是数值噪声级别
    if dsig.max() > 1e-3:
        ok = False
        print("      ^ sigmoid 分数差异偏大")
    # 只关心高置信度的框:低分 query 的 logits 本来就在噪声里
    top = sigmoid(rout[0]).max(axis=1).argsort()[::-1][:min(30, n_queries)]
    dtop = np.abs(sigmoid(cout[0])[top].max(axis=1) - sigmoid(rout[0])[top].max(axis=1))
    print(f"           前 30 个高分 query 的分数差异 max={dtop.max():.3e}")

    # ---- 3. 端到端检测框 ----
    d_cpp = nms(decode(cout[0], args.image_size, args.thrh), args.nms_iou)
    d_ref = nms(decode(rout[0], args.image_size, args.thrh), args.nms_iou)
    print(f"[3] 检测框  C++ {len(d_cpp)} 个 | 参考 {len(d_ref)} 个  "
          f"(thrh={args.thrh}, NMS IoU={args.nms_iou})")

    matched, used = 0, set()
    worst_iou, worst_dscore = 1.0, 0.0
    for box_c, sc_c, lb_c in sorted(d_cpp, key=lambda d: -d[1]):
        best, best_iou = -1, 0.0
        for j, (box_r, sc_r, lb_r) in enumerate(d_ref):
            if j in used or lb_r != lb_c:
                continue
            v = iou_pair(box_c, box_r)
            if v > best_iou:
                best, best_iou = j, v
        if best >= 0 and best_iou >= 0.5:
            used.add(best)
            matched += 1
            worst_iou = min(worst_iou, best_iou)
            worst_dscore = max(worst_dscore, abs(sc_c - d_ref[best][1]))
    print(f"           成功配对 {matched}/{len(d_cpp)} 个(C++ 侧),"
          f"最低配对 IoU={worst_iou:.4f},最大分数差={worst_dscore:.3e}")
    if matched != len(d_cpp) or len(d_cpp) != len(d_ref):
        ok = False
        print("      ^ 两侧检测框没有完全一一对应")

    print()
    print("结论:" + ("PASS —— C++ TensorRT 与 Python 参考结果一致" if ok
                    else "FAIL —— 存在差异,见上面 ^ 标记"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
