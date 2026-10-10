# DEIMv2 C++ TensorRT 推理验证工具

用 C++ + TensorRT 10.12 跑通 `tools/deployment/export_onnx_v2.py` 导出的"裸 concat"检测模型
(`images[B,3,H,W] -> output[B,N,4+num_classes]`),验证 ONNX 模型能否正确输出检测框。

预处理/后处理与 `tools/inference/onnx_inf_v2.py` 一一对应,实测与 Python 参考的结果一致。

```text
输入图片 ──► 直接拉伸(非 letterbox) ──► /255 ──► ImageNet 归一化 ──► CHW
                                                                    │
                                            TensorRT engine ◄───────┘
                                                    │
   检测框 ◄── 全类别 NMS ◄── cxcywh×原图(宽,高) ◄── 阈值过滤 ◄── sigmoid+取最高分类别
```

## 目录

| 文件                          | 说明                                               |
| ----------------------------- | -------------------------------------------------- |
| `src/main.cpp`              | 命令行入口、画框、结果输出                         |
| `src/trt_engine.{h,cpp}`    | TensorRT 引擎构建/缓存/反序列化 + 推理             |
| `src/preprocess.{h,cpp}`    | PIL 兼容的拉伸 resize + 归一化                     |
| `src/postprocess.{h,cpp}`   | 解码 + 全类别 NMS                                  |
| `run_both.sh`               | 一次跑通 L / X 两个模型                            |
| `verify_reference.py`       | 生成 Python 参考(onnxruntime + PIL)的裸 f32,供比对 |
| `compare_with_reference.py` | 逐元素 / 检测框两两比对,输出 PASS/FAIL             |
| `compare_with_onnx_inf.py`  | 与`onnx_inf_v2.py` 的检测框逐框比对              |

## 构建

都在容器 `nvrt1012_dev` 内操作。宿主机仓库路径 `/media/yin/Data/workspace/DEIMv2`
在容器里挂载为 `/workspace/DEIMv2`:

```bash
docker exec -it nvrt1012_dev bash      # 进入容器

cd /workspace/DEIMv2/tools/trt_cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

TensorRT 路径默认 `/usr/local/tensorrt/TensorRT-10.12.0.36`,换版本加
`-DTENSORRT_ROOT=/path/to/TensorRT-x.y.z`。可执行文件会写死 RPATH,运行时不需要
`export LD_LIBRARY_PATH`。

## 使用

在 `/workspace/DEIMv2` 下执行:

```bash
# L 规模:输入 896x512
./tools/trt_cpp/build/deimv2_trt_infer \
    --onnx z_onnx/PersonCarAnimal_od-v5-2-l-d4-512-896_20260920.onnx \
    --image example.jpg --thrh 0.25

# X 规模:输入 768x416
./tools/trt_cpp/build/deimv2_trt_infer \
    --onnx z_onnx/PersonCarAnimal_od-v5-2-x-d4-416-768_20260918.onnx \
    --image example.jpg --thrh 0.25
```

或直接跑两个模型:

```bash
bash tools/trt_cpp/run_both.sh          # 可用 IMG=xxx.jpg 换图
```

首次运行会从 ONNX 构建 engine(几分钟)并缓存到 `<onnx 路径>.engine`,之后秒开。
换模型、换 TensorRT 版本、换 GPU 后要加 `--rebuild`。

### 常用参数

| 参数                     | 说明                                                                   |
| ------------------------ | ---------------------------------------------------------------------- |
| `--thrh <f>`           | 分数阈值,默认`0.25`(与 `onnx_inf_v2.py` 一致)                      |
| `--nms-iou <f>`        | 全类别 NMS 的 IoU 阈值,默认`0.9`;`<=0` 关闭                        |
| `--output <path>`      | 结果图,默认`trt_result.jpg`                                          |
| `--save-txt <path>`    | 把检测框写成文本,便于与 Python 侧 diff                                 |
| `--dump-input <path>`  | 导出预处理后的输入张量(f32 裸数据),用于核对预处理                      |
| `--dump-output <path>` | 导出模型原始输出(f32 裸数据),用于核对推理                              |
| `--fp16`               | 弱类型 FP16 构建。**精度不可接受**,要用 FP16 请用 `--fp16` 导出的模型 + `--strongly-typed` |
| `--fp16-fp32-norm`     | 同上,但把 norm 相关的层钉回 FP32 |
| `--strongly-typed`     | 强类型构建:精度完全按 ONNX 的 dtype 走,配合 `--fp16` 导出的模型用 |
| `--debug-nan`          | 逐张量找 NaN,打印第一个出现 NaN 的张量名(用 TensorRT debug tensor) |
| `--debug-nan-all`      | 同上,但标记每个张量;会阻止算子融合,engine 更大更慢 |
| `--benchmark <n>`      | 推理 n 次取平均耗时                                                    |
| `--no-normalize`       | 只做 /255,不做 ImageNet 归一化(对应`deim_dfine`,见下)                |
| `--mean` / `--std`   | 自定义归一化,格式为`r,g,b`                                           |
| `--device <n>`         | 用哪块 GPU,默认 `0`(见下"engine 与 GPU 的关系")                       |
| `--workspace <mb>`     | 构建时的 workspace 上限,默认 `2048`。**高分辨率模型要调大**(见下)      |
| `--engine <path>`      | engine 缓存路径                                                        |
| `--rebuild`            | 忽略缓存,强制重新构建                                                  |

## 与 onnx_inf_v2.py 的对应关系

预处理参数不是猜的,按模型对应的训练配置推导:

|                                        | L 模型                                                       | X 模型                                      |
| -------------------------------------- | ------------------------------------------------------------ | ------------------------------------------- |
| 配置                                   | `configs/deimv2/deimv2_dinov3_l_coco.yml`                  | `configs/deimv2/deimv2_dinov3_x_coco.yml` |
| `input_size` / `eval_spatial_size` | `[512, 896]` = H×W                                        | `[416, 768]` = H×W                       |
| 模型输入                               | `[1, 3, 512, 896]`                                         | `[1, 3, 416, 768]`                        |
| Resize                                 | 直接拉伸,**无 letterbox、无 padding**                  | 同左                                        |
| /255                                   | 是                                                           | 是                                          |
| Normalize                              | ImageNet`mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]` | 同左                                        |

两个模型的骨干都是 DINOv3(图里有 `model.backbone.dinov3.patch_embed.proj`),所以带
ImageNet 归一化 —— 这也是 `onnx_inf_v2.py` 里 `build_preprocess()` 从 val 增强中读到的结果。
只有 `deim_dfine`/`deimv2_hgnetv2` 这类没有 `Normalize` op 的配置才用 `--no-normalize`。

因为训练/验证都是直接拉伸、没有 padding,后处理里把归一化的 `cxcywh` 乘 `(原图宽, 原图高)`
再转 `xyxy` 即可,**不需要**反算 letterbox 的 padding。

### 两个实现细节

1. **resize 与 Pillow 逐像素一致。** Pillow 的双线性不是简单的插值:`filterscale = max(1, scale)`
   会把滤波核展宽(缩小时即抗锯齿),而且 8bpc 路径是"水平 → 量化成 uint8 → 垂直 → 量化 uint8"。
   本实现照此复刻,实测与 PIL 输出逐像素相同;若省掉中间量化或调换两趟顺序,会差约 1 个灰阶。
2. **归一化用除法而非乘倒数。** 参考实现是 `x / 255`,乘 `1/255` 会在 float32 末位引入差异,
   导致两边预处理结果不能逐位对齐。

## 验证:与 Python 参考对比

`compare_with_reference.py` 会把两边的结果逐元素比一遍,再各跑一次解码+NMS 对比检测框。

```bash
# 1) C++ 侧导出中间结果(在 nvrt1012_dev 里)
cd /workspace/DEIMv2
B=z_trt_build
./tools/trt_cpp/build/deimv2_trt_infer \
    --onnx z_onnx/PersonCarAnimal_od-v5-2-l-d4-512-896_20260920.onnx \
    --image example.jpg --dump-input $B/cpp_in_l.bin --dump-output $B/cpp_out_l.bin

# 2) Python 参考侧导出(在训练容器里,需要 numpy/pillow/onnxruntime)
python3 tools/trt_cpp/verify_reference.py --image example.jpg --size 512 896 \
    --onnx z_onnx/PersonCarAnimal_od-v5-2-l-d4-512-896_20260920.onnx \
    --dump-input $B/ref_in_l.bin --dump-output $B/ref_out_l.bin

# 3) 比对
python3 tools/trt_cpp/compare_with_reference.py \
    --cpp-input $B/cpp_in_l.bin --ref-input $B/ref_in_l.bin \
    --cpp-output $B/cpp_out_l.bin --ref-output $B/ref_out_l.bin \
    --image-size 896 512 --thrh 0.25 --nms-iou 0.9

# 也可以直接和 onnx_inf_v2.py 的检测框比:两边各存一份文本,再逐框对比
python3 tools/trt_cpp/compare_with_onnx_inf.py ref_l.txt cpp_l.txt
```

### 实测结果(RTX 2070,FP32,example.jpg)

#### ① 与 Python 参考(onnxruntime + PIL)逐元素对比

|                             | L 模型                             | X 模型                            |
| --------------------------- | ---------------------------------- | --------------------------------- |
| 模型输入                    | `[1,3,512,896]`                  | `[1,3,416,768]`                 |
| 模型输出                    | `[1,300,85]`(300 query,81 类)    | `[1,300,85]`                    |
| 预处理 vs PIL               | **逐元素完全一致**(0 个不同) | **逐元素完全一致**          |
| TRT vs ORT 分数(sigmoid 后) | max\|d\| = 3.7e-05                 | max\|d\| = 1.4e-05                |
| TRT vs ORT 框(0~1)          | max\|d\| = 8.3e-06                 | max\|d\| = 1.2e-05                |
| 检测框个数(≥0.25)          | C++ 16 / 参考 16                   | C++ 21 / 参考 21                  |
| 配对情况                    | **16/16,最低 IoU = 1.0000**  | **21/21,最低 IoU = 1.0000** |
| 结论                        | **PASS**                     | **PASS**                    |

#### ② 与 `tools/inference/onnx_inf_v2.py` 直接对比

也就是最初确认模型可用的那个工具:

```bash
$ python3 tools/inference/onnx_inf_v2.py --onnx <L模型> \
      --config configs/deimv2/deimv2_dinov3_l_coco.yml --input example.jpg --thrh 0.25
$ ./tools/trt_cpp/build/deimv2_trt_infer --onnx <L模型> --image example.jpg --thrh 0.25
```

|              | L 模型         | X 模型         |
| ------------ | -------------- | -------------- |
| 检测框个数   | 16 / 16        | 21 / 21        |
| 类别不一致数 | 0              | 0              |
| 框坐标最大差 | 0.5 像素 *     | 0.5 像素 *     |
| 分数最大差   | 4.6e-04        | 4.0e-04        |
| 结论         | **PASS** | **PASS** |

\* Python 侧打印 `xyxy` 时已取整到整数,`.5` 就是四舍五入的半个像素,不是真实偏差。

#### ③ 负对照

去掉 ImageNet 归一化后会明显变差,说明该归一化确实是这两个模型需要的:
L 的框数 16→12;X 在阈值 0.05 下冒出 113 个框,退化成噪声。

### 耗时(RTX 2070,FP32,`--benchmark 100`)

| 模型        | 预处理  | 推理(平均)        | 后处理  |
| ----------- | ------- | ----------------- | ------- |
| L(512×896) | 10.3 ms | **39.4 ms** | 0.13 ms |
| X(416×768) | 8.0 ms  | **32.0 ms** | 0.13 ms |

预处理目前是单线程 CPU 实现(`preprocess.cpp`),图片大时它占比不低,需要提速可以
按行 OpenMP 并行或挪到 CUDA。

## FP16:两个模型都能用

弱类型 FP16(`--fp16` + `BuilderFlag::kFP16`)精度不可接受,但**修正后两个模型都能用 FP16**。

| 方案 | 推理耗时 | 配对数 | 最低 IoU | 最大框差 |
| --- | --- | --- | --- | --- |
| FP32(基准) | L 39.2 / X 32.1 ms | 33/33、50/50 | — | — |
| 纯 FP16(弱类型 `kFP16`) | 9.1 ms | 18/21 | 0.583 | **34.3 px** |
| **`--fp16` 导出 + `--strongly-typed`(推荐)** | **L 24.8 / X 16.1 ms** | **33/33、50/50** | **0.9519 / 0.9872** | **0.820 / 0.591 px** |

(3 张测试图:L 33 个框 / X 50 个框)

**根因**:ViT 注意力的 `q·k` 是逐通道累加,量级远大于 q/k 本身 ——
L 模型 block0 的 `|q·k|·scale` 达到 **2,518,337**,是 FP16 上限 65504 的 **38 倍**,
变 Inf 后 `softmax` 就是 NaN,再顺残差污染整张图(L 修复前输出全 NaN)。
把注意力放 FP32 里算即可(`engine/backbone/dinov3/layers/attention.py`),
实测对训练逐位中性。

### 怎么用

```bash
# 1) 导出混合精度图(I/O 仍是 FP32,部署侧代码不用改)
python3 tools/deployment/export_onnx_v2.py \
    -c configs/deimv2/deimv2_dinov3_x_coco.yml \
    -r outputs/deimv2_dinov3_x_personcaraninal/best_stg2.pth \
    --backbone ckpt/dinov3_vits16plus_from_deimv2_x.pth \
    --fp16 --output z_onnx/xxx_fp16.onnx

# 2) 强类型构建(弱类型会忽略图里的 dtype 分界,精度照样崩)
./tools/trt_cpp/build/deimv2_trt_infer \
    --onnx z_onnx/xxx_fp16.onnx --strongly-typed --image example.jpg --benchmark 100
```

**不方便重导模型时的退路**:直接在原 FP32 ONNX 上 `--fp16-fp32-norm`
(把名字含 `norm` 的层钉回 FP32)。实测 L 20/21、5.7 px,不如上面的方案但不用重导。

### 三个要点

- **必须配 `--strongly-typed`**:弱类型 + `kFP16` 下 TensorRT 有权把任意 FP32 层降成 FP16,
  图里的 dtype 分界会被整个忽略,模型代码里原本的 `x.float()` 保护就成了空操作。
- **不要用 `convert_float_to_float16` 手工转 ONNX 来代替** —— 它会把 DINOv3 RoPE 的
  `Cos`/`Sin` 也降成 FP16(而 `.half()` 导出不会,因为模块里写死了 `self.dtype`),实测差 34 px。
- `--fp16` 导出需要 GPU(CPU 没有 `grid_sample` 的 FP16 实现)。

**副产品**:修正后**半精度模型在 PyTorch 里也能直接前向**(不只是导出),
实测 0 个 NaN,L 33/33 配对(框差 2.107 px)、X 50/50(0.392 px)。

完整排查过程、逐框数据和复现脚本见 [docs/fp16_mixed_precision.md](docs/fp16_mixed_precision.md);
部署实现与面试问答见 [docs/tensorrt_deployment_summary.md](docs/tensorrt_deployment_summary.md)。

## 高分辨率模型:记得调大 `--workspace`

输入越大,某些 tactic 需要的工作内存越多,**默认 2048 MB 可能直接构建失败**。
典型报错:

```text
[TRT][W] UNSUPPORTED_STATE: Skipping tactic 0 due to insufficient memory
         on requested size of 3262730200 detected for tactic 0x0000000000000000.
[TRT][E] IBuilder::buildSerializedNetwork: Error Code 10: Internal Error
         (Could not find any implementation for node {ForeignNode[.../blocks.11/attn/...]}.)
```

这个数字可以自己算:输入 1088×1920 时 ViT 的 patch 数是 68×120 = **8160**,
注意力矩阵 `[1,6,8160,8160]` 在 FP32 下是 **1.6 GB**,加 softmax 输出约 **3.2 GB** ——
正好对上那个 3.26 GB。(注意力走 FP32 是为了修 FP16 溢出,见
[docs/fp16_mixed_precision.md](docs/fp16_mixed_precision.md)。)

```bash
# 1088x1920 这类大输入:加到 5 GB
./tools/trt_cpp/build/deimv2_trt_infer --onnx xxx.onnx --image x.jpg --workspace 5120
```

对比参考:

| 输入 | patch 数 | 注意力矩阵(FP32) | 建议 workspace | 实测耗时(FP32/FP16) |
| --- | --- | --- | --- | --- |
| 416×768 | 1,248 | 37 MB | 2048 够 | 32.1 / 16.1 ms(**2.00×**) |
| 1088×1920 | 8,160 | **1.6 GB** | **5120** | 475.1 / 389.2 ms(**1.22×**) |

**高分辨率下 FP16 的加速比会被摊平** —— 逐层 profiling 显示,注意力在 FP16 engine 里
占 **90%** 的耗时却几乎拿不到收益:单个 ViT softmax 层要搬运 6.39 GB 数据
(`[1,6,8160,8160]` 矩阵两趟读写),除以下 448 GB/s = 理论 14.26 ms,实测 13.8~14.5 ms,
**已经跑在内存带宽上限**。FP16 的算力优势对带宽瓶颈无效,而强制 FP32 让字节数翻倍。

> 测量提示:比较高分辨率下的 FP32/FP16 时**务必交替复测**。顺序测量会让第二个
> 跑在热 GPU 上,把 FP16 测成比 FP32 还慢(实测差 30%+)。

## engine 与 GPU 的关系

三个概念容易混,分清楚:

| | 绑定关系 |
| --- | --- |
| **ONNX 模型** | **与设备无关**。ONNX 描述的是计算图(算子/形状/dtype),不含"在哪个 GPU 上算"。导出脚本里的 `.cuda()` 只是为了能追踪出图(CPU 没有 `grid_sample` 的 FP16 实现),导出的文件里没有任何 device 字段 |
| **TensorRT engine** | 绑定 **GPU 架构**(SM 版本)和 TensorRT 版本,**不绑定设备序号**。同架构的另一块卡可以直接复用同一个 engine(用 `--device N` 指定);换架构/换 TRT 版本必须 `--rebuild` |
| **推理进程** | 用 `--device N` 选卡,默认 0。工具会在启动时打印实际用的设备和 SM 版本 |

```bash
# 在 1 号卡上跑(engine 与卡号无关,同架构可直接复用)
./tools/trt_cpp/build/deimv2_trt_infer --onnx xxx.onnx --image x.jpg --device 1

# 启动日志会显示实际设备,便于确认
# [CUDA] 使用设备 0: NVIDIA GeForce RTX 2070 (sm_75, 7778 MB)
```

## 说明

- 只支持单输入单输出、FP32 输入输出、H/W 静态的网络(本项目的导出图正好如此)。
- batch 维在导出图里是动态的,profile 取 `[1, --max-batch]`、opt 定在 1(本工具一次跑一张图);
  ONNX 若是静态 batch,则三个值都取该静态值。
- 输出的 query 数和类别数是运行时从 engine 形状推出来的(`N = shape[1]`, `C = shape[2] - 4`),
  不写死,换模型不用改代码。
- `--fp16` 走的是 TensorRT 10.12 里已标记 deprecated 的 `BuilderFlag::kFP16`(官方建议改用
  strong typing),但弱类型网络 + kFP16 仍受支持,好处是输入/输出保持 FP32。
