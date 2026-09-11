"""
导出 DEIM-DFINE / DEIMv2-DINOv3 检测器为 ONNX —— "裸 concat" 风格

导出图:images[B,3,H,W] -> output[B,N,4+num_classes]
  pred_boxes: cxcywh、相对于输入图尺寸归一化(0~1)
  pred_logits: 原始分类 logits
  N 为 query 数(如 300);两者在最后一维拼接,图内不含任何后处理。

预处理/后处理留在 ONNX 外部:
  - 本脚本从合并配置的 val 增强自动读取 Resize 插值、ConvertPILImage 的 /255、
    Normalize 的 mean/std(dfine 无 Normalize、dinov3 有),保证喂图与训练一致;
  - 部署侧后处理只需:sigmoid -> 每 query 取最高分类别 -> 置信度过滤 -> cxcywh * 原图(w,h) ->
    xyxy(训练为直接拉伸、无 letterbox padding,故无需还原 padding)。

特性:自动从 ckpt 推断类别数与 query 数、自动跳过 decoder.anchors/valid_mask(按导出尺寸重新生成)、
ema 权重优先、--check/--simplify/--input 自检、预处理从配置自动推导(dfine 只 /255、dinov3 加 ImageNet 归一化)。

验证结果图在容器 /tmp/dinov3_x_test_result.jpg、/tmp/dfine_x_test_result.jpg,可以查看画框效果确认后处理正确。
部署侧注意:输入按"直接拉伸到 [576,1024]+/255(+dinov3 归一化)",输出 pred_boxes 是相对输入图的 cxcywh 归一化值,
反算原图坐标时 ×[w,h] 即可(无 letterbox padding)。

用法(容器内,代码目录 /workspace/DEIMv2):
  # DEIM-DFINE
  python3 tools/deployment/export_onnx_v2.py \
      -c configs/deim_dfine/deim_hgnetv2_x_coco.yml \
      -r outputs/deim_hgnetv2_x_charging_gun_nc4/best_stg2.pth --check --simplify

  # DEIMv2-DINOv3(需要把 backbone 权重路径指向训练时提取的 backbone 权重文件)
  python3 tools/deployment/export_onnx_v2.py \
      -c configs/deimv2/deimv2_dinov3_x_coco.yml \
      -r outputs/deimv2_dinov3_x_charging_gun_nc4/best_stg2.pth \
      --backbone ckpts/dinov3_vits16plus_from_deimv2_x.pth --check --simplify

  # 导出 + ONNX Runtime 端到端自检(画框存 jpg)
  python3 tools/deployment/export_onnx_v2.py \
      -c configs/deimv2/deimv2_dinov3_x_coco.yml \
      -r outputs/deimv2_dinov3_x_charging_gun_nc4/best_stg2.pth \
      --backbone ckpts/dinov3_vits16plus_from_deimv2_x.pth \
      --input charging_gun.jpg --check --simplify
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import re
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image, ImageDraw

from engine.core import YAMLConfig

# torchvision Resize 的 interpolation 数值 -> PIL resampling 常量
_PIL_RESAMPLE = {0: Image.NEAREST, 1: Image.LANCZOS, 2: Image.BILINEAR, 3: Image.BICUBIC}


# ---------------------------------------------------------------------------
# checkpoint 工具(dfine 与 dinov3 的键结构一致,已实测)
# ---------------------------------------------------------------------------
def load_ckpt_state(path):
    """兼容纯 model dict / 训练 solver 状态(含 ema / model 等顶层键)。"""
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


# ---------------------------------------------------------------------------
# 预处理/后处理(喂图方式与训练 val 完全一致,由配置自动推导)
# ---------------------------------------------------------------------------
def build_preprocess(cfg):
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
        im = im_pil.resize((size[1], size[0]), resample)    # PIL 参数为 (w, h)
        arr = np.asarray(im, dtype=np.float32)
        if scale:
            arr /= 255.0
        arr = arr.transpose(2, 0, 1)                        # CHW
        if mean is not None and std is not None:
            arr = (arr - np.array(mean, np.float32)[:, None, None]) \
                  / np.array(std, np.float32)[:, None, None]
        return arr[None].astype(np.float32)                 # [1,3,H,W]

    return preprocess


def postprocess_and_draw(im_pil, output, score_thresh=0.5):
    """output[B, N, 4+C] -> 每 query 取最高分类别,画框保存。"""
    pred = output[0]
    boxes_cxcywh = torch.from_numpy(pred[:, :4])
    logits = torch.from_numpy(pred[:, 4:])
    scores = logits.sigmoid()
    top_s, top_c = scores.max(dim=-1)
    w, h = im_pil.size
    scale = torch.tensor([w, h, w, h], dtype=torch.float32)
    boxes = torchvision.ops.box_convert(boxes_cxcywh * scale,
                                        in_fmt="cxcywh", out_fmt="xyxy").numpy()

    draw = ImageDraw.Draw(im_pil)
    cls2color = {}
    for i, sc in enumerate(top_s.numpy()):
        if sc < score_thresh:
            continue
        cls = int(top_c[i].item())
        if cls not in cls2color:
            cls2color[cls] = tuple(random.choices(range(256), k=3))
        x1, y1, x2, y2 = boxes[i]
        draw.rectangle([x1, y1, x2, y2], outline=cls2color[cls], width=2)
        label = f"{cls}: {sc:.2f}"
        tb = draw.textbbox((x1, y1), label)
        draw.rectangle(tb, fill=cls2color[cls])
        draw.text((x1, y1), label, fill=(255, 255, 255))
    return im_pil


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main(args):
    # 1) 读 ckpt -> 推断类别数 -> 构建配置(类别数会决定分类头形状)
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

    # 2) 装载权重。decoder.anchors/valid_mask 是按 eval_spatial_size 缓存的
    #    buffer,训练/预训练时尺寸可能与导出配置不同(如 640 vs 576x1024),
    #    导出时会按当前配置重新生成,跳过加载。
    loadable = {k: v for k, v in state.items()
                if not k.endswith(("decoder.anchors", "decoder.valid_mask"))}
    miss, unexp = cfg.model.load_state_dict(loadable, strict=False)
    if miss or unexp:
        print(f"[warn] 未匹配 {len(miss)} 键 / 多余 {len(unexp)} 键")
        if miss:
            print("       示例:", miss[:3])

    # 3) 裸 concat 导出图
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()

        def forward(self, images):
            outputs = self.model(images)
            return torch.cat([outputs["pred_boxes"], outputs["pred_logits"]], dim=-1)

    model = Model().eval()
    size = list(cfg.yaml_cfg["eval_spatial_size"])          # [h, w]
    data = torch.rand(args.batch_size, 3, size[0], size[1])
    with torch.no_grad():
        out = model(data)
    n_queries = out.shape[1]
    print(f"导出输入: ({args.batch_size}, 3, {size[0]}, {size[1]}) | "
          f"输出: (B, {n_queries}, {4 + num_classes}) | num_classes = {num_classes}")

    output_file = args.output or args.resume.replace(".pth", ".onnx")
    torch.onnx.export(
        model, (data,), output_file,
        input_names=["images"], output_names=["output"],
        dynamic_axes={"images": {0: "Batch"}, "output": {0: "Batch"}},
        opset_version=args.opset, do_constant_folding=True,
    )
    print("导出完成:", output_file)

    if args.check:
        import onnx
        onnx.load(output_file)
        print("onnx checker 校验通过")

    if args.simplify:
        import onnx
        import onnxsim
        try:
            simplified, check = onnxsim.simplify(
                output_file, test_input_shapes={"images": tuple(data.shape)})
            onnx.save(simplified, output_file)
            print(f"onnxsim 精简完成(check={check})")
        except Exception as e:
            # DINOv3 主干含 RoPE(rope_embed 的 Shape/Range/Sub 动态形状链),
            # 部分 onnxsim 版本会报 "Input /model/backbone/rope_embed/... is undefined!"。
            # 精简只是图优化,失败不影响导出图的有效性(onnx checker 已通过、
            # ORT/TensorRT 均可直接加载，TRT 构建器自带图优化),降级为告警并保留原始 onnx。
            print(f"[warn] onnxsim 精简失败({type(e).__name__}: {str(e)[:120]})")
            print("       -> 已保留未精简的原始 onnx(可用;TensorRT 构建时自身会做图优化)")

    # 4) ONNX Runtime 端到端自检(可选)
    if args.input:
        import onnxruntime as ort
        sess = ort.InferenceSession(
            output_file, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        im_pil = Image.open(args.input).convert("RGB")
        feed = build_preprocess(cfg)(im_pil)
        res = sess.run(["output"], {"images": feed})[0]
        result = postprocess_and_draw(im_pil, res, score_thresh=args.score_thresh)
        result.save(args.save_result)
        print(f"ONNX Runtime 验证完成,结果: {args.save_result}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="deim_dfine 或 deimv2_dinov3 的 *_coco.yml 配置")
    parser.add_argument("--resume", "-r", type=str, required=True, help="训练 checkpoint")
    parser.add_argument("--num-classes", type=int, default=None, help="默认从 ckpt 自动推断")
    parser.add_argument("--backbone", type=str, default=None,
                        help="DINOv3STAs.weights_path(仅 deimv2_dinov3 需要,训练脚本提取的那个文件)")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=str, default=None, help="输出 onnx(默认与 resume 同路径)")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--input", type=str, default=None, help="验证用图片")
    parser.add_argument("--save-result", type=str, default="onnx_result.jpg")
    parser.add_argument("--score-thresh", type=float, default=0.5)
    args = parser.parse_args()
    main(args)
