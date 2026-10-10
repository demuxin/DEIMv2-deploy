"""把 `tools/inference/onnx_inf_v2.py`(参考工具)的检测结果与 C++ TensorRT 的逐框比对。

这是最直接的验证:C++ 侧和"已经确认可用"的 Python 工具是不是给出同一批框。

先在两边各自导出文本:
  # Python 参考(训练容器,把 stdout 存下来)
  python3 tools/inference/onnx_inf_v2.py --onnx <model.onnx> \
      --config configs/deimv2/deimv2_dinov3_l_coco.yml \
      --input example.jpg --thrh 0.25 --output /tmp/onnx_l.jpg > ref_l.txt 2>&1

  # C++(nvrt1012_dev)
  ./tools/trt_cpp/build/deimv2_trt_infer --onnx <model.onnx> --image example.jpg \
      --thrh 0.25 --output /tmp/trt_l.jpg --save-txt cpp_l.txt

再比对:
  python3 tools/trt_cpp/compare_with_onnx_inf.py ref_l.txt cpp_l.txt
"""

import re
import sys

# onnx_inf_v2.py 打印的检测行: "  [0] score=0.810 xyxy=(290,60,314,98)"
PY_LINE = re.compile(
    r"\[(\d+)\]\s+score=([\d.]+)\s+xyxy=\((-?[\d.]+),(-?[\d.]+),(-?[\d.]+),(-?[\d.]+)\)")

# C++ --save-txt 的格式: "label score x1 y1 x2 y2"(全精度)
BOX_TOLERANCE = 1.0     # 像素;python 侧打印时把 xyxy 四舍五入到 0 位,所以有 0.5 的固有误差
SCORE_TOLERANCE = 1e-3


def parse_python(path):
    dets = []
    with open(path, errors="replace") as f:
        for line in f:
            m = PY_LINE.search(line)
            if m:
                dets.append((int(m.group(1)), float(m.group(2)),
                             *[float(m.group(i)) for i in range(3, 7)]))
    return dets


def parse_cpp(path):
    dets = []
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split()
            if len(p) == 6:
                dets.append((int(p[0]), float(p[1]), *[float(x) for x in p[2:]]))
    return dets


def iou(a, b):
    xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
    xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, xx2 - xx1), max(0.0, yy2 - yy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main(py_txt, cpp_txt):
    a, b = parse_python(py_txt), parse_cpp(cpp_txt)
    print(f"onnx_inf_v2.py: {len(a)} 个框 | C++ TensorRT: {len(b)} 个框")
    if not a or not b:
        print("  !! 有一侧没解析到检测框,检查输入文件")
        return 1

    # 按 IoU 配对而不是按位置 —— 两侧分数有微小差异时 NMS 的输出顺序可能不同,
    # 按位置比会张冠李戴,报出几百像素的假偏差。
    used = set()
    matched = bad_class = 0
    worst_box = worst_score = 0.0
    worst_iou = 1.0
    for lb, sb, *bb in sorted(b, key=lambda d: -d[1]):
        best, best_iou = -1, 0.0
        for j, (la, sa, *ba) in enumerate(a):
            if j in used:
                continue
            v = iou(bb, ba)
            if v > best_iou:
                best, best_iou = j, v
        if best < 0 or best_iou < 0.5:
            print(f"  C++ 侧有框没配上: label={lb} score={sb:.3f} box={[round(v) for v in bb]}")
            continue
        used.add(best)
        matched += 1
        la, sa, *ba = a[best]
        if la != lb:
            bad_class += 1
            print(f"  类别不一致: py={la} cpp={lb} (IoU={best_iou:.3f})")
        worst_box = max(worst_box, max(abs(x - y) for x, y in zip(ba, bb)))
        worst_score = max(worst_score, abs(sa - sb))
        worst_iou = min(worst_iou, best_iou)

    print(f"  配对: {matched}/{len(b)}(C++ 侧),参考侧共 {len(a)} 个")
    print(f"  类别不一致数: {bad_class}")
    print(f"  最低配对 IoU: {worst_iou:.4f}")
    print(f"  框坐标最大差: {worst_box:.3f} 像素")
    print(f"  分数最大差:   {worst_score:.6f}")
    ok = (matched == len(b) == len(a) and bad_class == 0
          and worst_box <= BOX_TOLERANCE and worst_score <= SCORE_TOLERANCE)
    print("  结论:", "PASS —— 与 onnx_inf_v2.py 完全对齐" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
