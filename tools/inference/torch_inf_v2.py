"""
PyTorch 单图推理(训练好的 .pth 直接用,无需导出)——
与 tools/inference/onnx_inf_v2.py 完全相同的预处理/后处理口径:
  预处理:直接拉伸(非 letterbox、无 padding)到 eval_spatial_size [h, w];
          dfine 仅 /255;deimv2_dinov3 再按 ImageNet mean/std 归一化(从配置自动推导)。
  后处理:sigmoid -> 每 query 取最高分类别 -> 分数阈值 ->
          cxcywh(相对输入图,0~1)× 原图(w,h) -> xyxy -> 全类别 NMS(默认 IoU 0.9,<=0 关闭)。
  画框:按类别固定配色(与 onnx_inf_v2 同一调色板),便于两个后端结果直接对照。

用法(容器内,代码目录 /workspace/DEIMv2):
  # DEIM-DFINE
  python3 tools/inference/torch_inf_v2.py \
      -c configs/deim_dfine/deim_hgnetv2_x_coco.yml \
      -r outputs/deim_hgnetv2_x_charging_gun_nc4/best_stg2.pth \
      --input charging_gun.jpg --thrh 0.25

  # DEIMv2-DINOv3(需给 --backbone 指向训练时提取的 backbone 权重)
  python3 tools/inference/torch_inf_v2.py \
      -c configs/deimv2/deimv2_dinov3_x_coco.yml \
      -r outputs/deimv2_dinov3_x_charging_gun_nc4/best_stg2.pth \
      --backbone ckpts/dinov3_vits16plus_from_deimv2_x.pth \
      --input charging_gun.jpg --thrh 0.25

结果保存为 --output(默认 torch_result.jpg)。
"""

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import numpy as np
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


def load_ckpt_state(path):
    """兼容纯 model dict / 训练 solver 状态(含 ema/model 等顶层键)。"""
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "ema" in ckpt and isinstance(ckpt["ema"], dict):
        state = ckpt["ema"].get("module", ckpt["ema"])     # EMA 权重通常更优
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    return state


def infer_num_classes(state):
    """从 decoder 分类头权重行数推断类别数。"""
    for key, val in state.items():
        if re.search(r"decoder\.dec_score_head\.\d+\.weight$", key) and val.dim() == 2:
            return int(val.shape[0])
    return None


def build_preprocess(cfg):
    """从合并配置的 val 增强构造与训练一致的预处理(与 onnx_inf_v2 相同口径)。"""
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
        return torch.from_numpy(arr[None].astype(np.float32))   # [1,3,H,W]

    return preprocess


def decode_output(pred_boxes, pred_logits, im_pil, thrh):
    """pred_boxes[B,N,4] cxcywh(0~1) + pred_logits[B,N,C] -> 原图像素 xyxy 检测列表。"""
    boxes = pred_boxes[0].float().cpu()
    logits = pred_logits[0].float().cpu()
    scores, labels = logits.sigmoid().max(dim=-1)

    w, h = im_pil.size
    scale = torch.tensor([w, h, w, h], dtype=torch.float32)  # 拉伸无 padding,直接反算原图
    xyxy = torchvision.ops.box_convert(boxes * scale, in_fmt="cxcywh", out_fmt="xyxy")

    dets = []
    for i, s in enumerate(scores.tolist()):
        if s >= thrh:
            x1, y1, x2, y2 = xyxy[i].tolist()
            dets.append((float(x1), float(y1), float(x2), float(y2), float(s), int(labels[i].item())))
    return dets


def class_agnostic_nms(dets, iou_thr):
    """全类别 NMS:忽略类别、只按框重合度(IoU)过滤,保留最高分框;<=0 关闭。"""
    if iou_thr <= 0 or len(dets) == 0:
        return dets
    boxes = torch.tensor([d[:4] for d in dets], dtype=torch.float32)
    scores = torch.tensor([d[4] for d in dets], dtype=torch.float32)
    keep = torchvision.ops.nms(boxes, scores, iou_threshold=iou_thr)
    return [dets[i] for i in keep.tolist()]


def main(args):
    # 1) 读 ckpt -> 推断类别数 -> 构建配置 -> 装载权重(与 export_onnx_v2 相同流程)
    state = load_ckpt_state(args.resume)
    num_classes = args.num_classes or infer_num_classes(state)
    if num_classes is None:
        raise RuntimeError("无法从 checkpoint 推断 num_classes,请用 --num-classes 显式指定")

    updates = {"num_classes": num_classes}
    if args.backbone:
        updates["DINOv3STAs.weights_path"] = args.backbone
    cfg = YAMLConfig(args.config, **updates)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    loadable = {k: v for k, v in state.items()
                if not k.endswith(("decoder.anchors", "decoder.valid_mask"))}
    miss, unexp = cfg.model.load_state_dict(loadable, strict=False)
    if miss or unexp:
        print(f"[warn] 未匹配 {len(miss)} 键 / 多余 {len(unexp)} 键")
        if miss:
            print("       示例:", miss[:3])

    device = torch.device(args.device)
    model = cfg.model.deploy().to(device).eval()

    # 2) 推理(裸输出,自行后处理,口径与 onnx_inf_v2 完全一致)
    im_pil = Image.open(args.input).convert("RGB")
    feed = build_preprocess(cfg)(im_pil).to(device)
    with torch.no_grad():
        outputs = model(feed)

    dets = decode_output(outputs["pred_boxes"], outputs["pred_logits"], im_pil, args.thrh)
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
    parser.add_argument("--config", "-c", type=str, required=True, help="训练对应配置(dfine/dinov3 通用)")
    parser.add_argument("--resume", "-r", type=str, required=True, help="训练 checkpoint(.pth)")
    parser.add_argument("--num-classes", type=int, default=None, help="默认从 ckpt 自动推断")
    parser.add_argument("--backbone", type=str, default=None,
                        help="DINOv3STAs.weights_path(仅 deimv2_dinov3 需要,训练脚本提取的那个文件)")
    parser.add_argument("--input", type=str, required=True, help="单张图片路径")
    parser.add_argument("--thrh", type=float, default=0.25, help="分数阈值(建议先看 0.25~0.4)")
    parser.add_argument("--nms-iou", type=float, default=0.9,
                        help="全类别 NMS 的 IoU 阈值(默认 0.9,只滤高度重合框;<=0 关闭)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="torch_result.jpg")
    args = parser.parse_args()
    main(args)
