"""生成 onnx_inf_v2.py 同款的预处理/推理结果(裸 f32),用于和 C++ 端
--dump-input / --dump-output 逐元素对比,确认 C++ 的预处理和后处理没有跑偏。

前半段直接照抄 tools/inference/onnx_inf_v2.py 的 build_preprocess:
    PIL BILINEAR 直接拉伸到 (w, h) -> /255 -> ImageNet 归一化 -> CHW -> [1,3,H,W]

用法(需要 numpy + pillow,可选 onnxruntime):
  # 只导出预处理结果
  python3 tools/trt_cpp/verify_reference.py --image example.jpg --size 512 896 \
      --dump-input /tmp/ref_in_l.bin

  # 顺便跑 onnxruntime 导出原始输出
  python3 tools/trt_cpp/verify_reference.py --image example.jpg --size 512 896 \
      --onnx z_onnx/xxx.onnx --dump-input /tmp/ref_in_l.bin --dump-output /tmp/ref_out_l.bin
"""

import argparse

import numpy as np
from PIL import Image

# deimv2_dinov3 的 val 增强(dfine 无 Normalize,用 --no-normalize)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_input(image_path, size, normalize=True):
    """size = [h, w]"""
    h, w = size
    im = Image.open(image_path).convert("RGB")
    im = im.resize((w, h), Image.BILINEAR)          # PIL 参数是 (w, h)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)                    # HWC -> CHW
    if normalize:
        mean = np.array(IMAGENET_MEAN, np.float32)[:, None, None]
        std = np.array(IMAGENET_STD, np.float32)[:, None, None]
        arr = (arr - mean) / std
    return np.ascontiguousarray(arr[None].astype(np.float32))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--size", type=int, nargs=2, required=True, metavar=("H", "W"))
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--onnx", default=None, help="给了就跑 onnxruntime 导出原始输出")
    p.add_argument("--dump-input", required=True)
    p.add_argument("--dump-output", default=None)
    args = p.parse_args()

    feed = build_input(args.image, args.size, normalize=not args.no_normalize)
    feed.tofile(args.dump_input)
    print(f"input  {feed.shape} {feed.dtype} -> {args.dump_input}")
    print(f"        min={feed.min():.6f} max={feed.max():.6f} mean={feed.mean():.6f}")

    if args.onnx:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
        pred = sess.run(["output"], {"images": feed})[0]
        print(f"output {pred.shape} {pred.dtype} (onnxruntime)")
        if args.dump_output:
            np.ascontiguousarray(pred, dtype=np.float32).tofile(args.dump_output)
            # 形状写到同名的 .shape 旁文件,compare_with_reference.py 靠它 reshape
            with open(args.dump_output + ".shape", "w") as f:
                f.write(" ".join(str(d) for d in pred.shape))
            print(f"       -> {args.dump_output} (shape {'x'.join(map(str, pred.shape))})")


if __name__ == "__main__":
    main()
