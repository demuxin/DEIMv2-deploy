"""
ONNX Runtime 单图推理 —— 配套 tools/deployment/export_onnx_v2.py 的
裸 concat 导出图(images[B,3,H,W] -> output[B,N,4+num_classes])。

预处理与训练 val 完全一致(从配置文件自动推导):
  直接拉伸(非 letterbox、无 padding)到 eval_spatial_size [h, w];
  dfine:仅 /255; deimv2_dinov3:再按 ImageNet mean/std 归一化。
后处理(与部署侧一致):
  sigmoid -> 每 query 取最高分类别 -> 按阈值过滤 ->
  cxcywh(相对输入图,0~1)× 原图(w,h) -> xyxy(拉伸无 padding,直接反算)
  -> 全类别 NMS(过滤高度重合框,IoU 阈值 --nms-iou 默认 0.95,<=0 关闭)。

用法:
  python3 tools/inference/onnx_inf_v2.py \
      --onnx outputs/deim_hgnetv2_x_charging_gun_nc4/best_stg2.onnx \
      --config configs/deim_dfine/deim_hgnetv2_x_coco.yml \
      --input charging_gun.jpg --thrh 0.25

  python3 tools/inference/onnx_inf_v2.py \
      --onnx outputs/deimv2_dinov3_x_charging_gun_nc4/best_stg2.onnx \
      --config configs/deimv2/deimv2_dinov3_x_coco.yml \
      --input charging_gun.jpg --thrh 0.25

结果保存为 --output(默认 onnx_result.jpg)。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import numpy as np
import onnxruntime as ort
import torch
import torchvision
from PIL import Image, ImageDraw

from engine.core import YAMLConfig

_PIL_RESAMPLE = {0: Image.NEAREST, 1: Image.LANCZOS, 2: Image.BILINEAR, 3: Image.BICUBIC}

# 按类别固定配色(类别数超过调色板长度时自动循环)
_CLASS_COLORS = (
    (230, 25, 75),    # 0 红
    (60, 180, 75),    # 1 绿
    (255, 225, 25),   # 2 黄
    (0, 130, 200),    # 3 蓝
    (245, 130, 48),   # 4 橙
    (145, 30, 180),   # 5 紫
    (70, 240, 240),   # 6 青
    (240, 50, 230),   # 7 品红
    (250, 190, 190),  # 8 粉
    (170, 110, 40),   # 9 棕
)


def build_preprocess(cfg_path):
    """从合并配置的 val 增强构造与训练一致的预处理。"""
    cfg = YAMLConfig(cfg_path)
    val_ops = cfg.yaml_cfg["val_dataloader"]["dataset"]["transforms"]["ops"]
    size = list(cfg.yaml_cfg["eval_spatial_size"])          # [h, w]
    interp = 2                                              # PIL BILINEAR
    scale = True
    mean = std = None
    for op in val_ops:
        t = op.get("type")
        if t == "Resize":
            interp = int(op.get("interpolation", 2))
        elif t == "ConvertPILImage":
            scale = bool(op.get("scale", True))
        elif t == "Normalize":
            mean, std = op.get("mean"), op.get("std")
    resample = _PIL_RESAMPLE.get(interp, Image.BILINEAR)

    def preprocess(im_pil):
        im = im_pil.resize((size[1], size[0]), resample)    # 直接拉伸,PIL 参数为 (w, h)
        arr = np.asarray(im, dtype=np.float32)
        if scale:
            arr /= 255.0
        arr = arr.transpose(2, 0, 1)                        # CHW
        if mean is not None and std is not None:
            arr = (arr - np.array(mean, np.float32)[:, None, None]) \
                  / np.array(std, np.float32)[:, None, None]
        return arr[None].astype(np.float32)

    return preprocess


def decode_output(pred, im_pil, thrh):
    """output[B,N,4+C] -> 原图像素 xyxy + (label, score) 列表。"""
    p = pred[0]
    logits = torch.from_numpy(p[:, 4:])
    boxes = torch.from_numpy(p[:, :4])
    scores, labels = logits.sigmoid().max(dim=-1)

    w, h = im_pil.size
    scale = torch.tensor([w, h, w, h], dtype=torch.float32)  # 拉伸无 padding,直接反算原图
    xyxy = torchvision.ops.box_convert(boxes * scale, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    dets = []
    for i, s in enumerate(scores.numpy()):
        if s >= thrh:
            dets.append((float(xyxy[i][0]), float(xyxy[i][1]),
                         float(xyxy[i][2]), float(xyxy[i][3]),
                         float(s), int(labels[i].item())))
    return dets


def class_agnostic_nms(dets, iou_thr):
    """全类别 NMS:忽略类别、只按框重合度(IoU)过滤,保留最高分框。

    iou_thr<=0 时不做 NMS,原样返回。dets: [(x1,y1,x2,y2,score,label), ...]
    """
    if iou_thr <= 0 or len(dets) == 0:
        return dets
    boxes = torch.tensor([d[:4] for d in dets], dtype=torch.float32)
    scores = torch.tensor([d[4] for d in dets], dtype=torch.float32)
    keep = torchvision.ops.nms(boxes, scores, iou_threshold=iou_thr)
    return [dets[i] for i in keep.tolist()]


def main(args):
    sess = ort.InferenceSession(args.onnx,
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    preprocess = build_preprocess(args.config)

    im_pil = Image.open(args.input).convert("RGB")
    feed = preprocess(im_pil)
    pred = sess.run(["output"], {"images": feed})[0]
    dets = decode_output(pred, im_pil, args.thrh)
    n_before = len(dets)
    if args.nms_iou > 0:
        dets = class_agnostic_nms(dets, args.nms_iou)
        print(f"检测到 {len(dets)} 个目标(score >= {args.thrh};全类别 NMS IoU={args.nms_iou} 过滤 {n_before} -> {len(dets)})")
    else:
        print(f"检测到 {len(dets)} 个目标(score >= {args.thrh};NMS 已关闭)")

    draw = ImageDraw.Draw(im_pil)
    for x1, y1, x2, y2, s, lb in dets:
        print(f"  [{lb}] score={s:.3f} xyxy=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
        color = _CLASS_COLORS[lb % len(_CLASS_COLORS)]      # 每个类别固定一种颜色
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"cls{lb}: {s:.2f}"
        tb = draw.textbbox((x1, y1 - 16), label)
        draw.rectangle(tb, fill=color)                      # 文字底色=类别色,更易读
        draw.text((x1, y1 - 16), label, fill=(255, 255, 255))
    im_pil.save(args.output)
    print(f"结果已保存: {args.output}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, required=True, help="export_onnx_v2.py 导出的 onnx")
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="训练对应配置(用于推导预处理,dfine/dinov3 通用)")
    parser.add_argument("--input", type=str, required=True, help="单张图片路径")
    parser.add_argument("--thrh", type=float, default=0.25, help="分数阈值(建议先看 0.25~0.4)")
    parser.add_argument("--nms-iou", type=float, default=0.9,
                        help="全类别 NMS 的 IoU 阈值(默认 0.9,只滤高度重合框;<=0 关闭)")
    parser.add_argument("--output", type=str, default="onnx_result.jpg")
    args = parser.parse_args()
    main(args)
