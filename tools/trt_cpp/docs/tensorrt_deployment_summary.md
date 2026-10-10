# DEIMv2 检测模型 TensorRT 部署 与 FP16 精度问题排查

> 本文覆盖:C++ TensorRT 推理工具的实现、与 Python 参考的逐位对齐、以及 X/L 两个模型
> FP16 精度问题的完整排查过程。所有数字都来自本机实测(RTX 2070, TensorRT 10.12.0.36, CUDA 12.8)。

---

## 0. 一页速览(先背这一页)

**任务**:把两个 ONNX 检测模型(L / X)用 C++ + TensorRT 跑起来,确认能正确输出检测框,
并搞清楚 FP16 为什么会掉精度。

**模型**

|      | L                                               | X                    |
| ---- | ----------------------------------------------- | -------------------- |
| ONNX | `PersonCarAnimal_od-v5-2-l-d4-512-896`        | `...-x-d4-416-768` |
| 输入 | `[1, 3, 512, 896]`(H×W)                      | `[1, 3, 416, 768]` |
| 输出 | `[1, 300, 85]` = 300 query × (4 box + 81 类) | 同                   |
| 骨干 | DINOv3,**需要 ImageNet 归一化**           | 同                   |

**三段一致性(3 张图,L 33 个框 / X 50 个框)**

|                          | 最低 IoU            | 最大框差                |
| ------------------------ | ------------------- | ----------------------- |
| ORT FP32 vs PyTorch FP32 | **1.0000**    | **0.000 px**      |
| TRT FP32 vs PyTorch FP32 | **1.0000**    | **0.000 px**      |
| TRT FP16 vs TRT FP32     | L 0.9519 / X 0.9872 | L 0.820 px / X 0.591 px |

**性能(RTX 2070,`--benchmark 100`)**

|   | FP32     | FP16               | 加速   |
| - | -------- | ------------------ | ------ |
| L | 39.21 ms | **24.81 ms** | 1.58× |
| X | 32.06 ms | **16.05 ms** | 2.00× |

(预处理约 8~10 ms,后处理 0.13 ms)

**FP16 精度问题的根因(一句话)**:
ViT 注意力的 `q·k` 是逐通道累加,量级远大于 q/k 本身 —— L 模型 block0 的
`|q·k|·scale` 达到 **2,518,337**,是 FP16 上限 65504 的 **38 倍**,在 FP16 里变成 Inf,
`softmax(Inf)` 就是 NaN,再顺残差污染整张图。**把注意力放 FP32 里算即可**
(顺带让半精度模型在 PyTorch 里也能直接前向)。

**高分辨率补充(1088×1920 的模型,详见 §7.10)**

| | |
| --- | --- |
| FP32 图上加 `--fp16` | **全 NaN**(25500/25500),必须用 `fp16` 导出的图 + `--strongly-typed` |
| 强类型 FP16 vs FP32 | 3 张图 38/39 配对,最低 IoU 0.9839,最大框差 **0.347 px** |
| 显存 | 默认 workspace 2048 MB **构建失败**(tactic 要 3.26 GB),需 `--workspace 5120` |
| 加速比 | 从 2.00×(416×768)降到 **1.19×** —— FP32 注意力成了瓶颈 |

**三个最容易踩的坑**

1. Pillow 的 resize 是"水平 → 量化 uint8 → 垂直 → 量化 uint8",顺序和中间量化都不能改
2. 归一化要用 `x / 255` 而不是 `x * (1/255)`(float32 末位差异)
3. TensorRT 弱类型 + `kFP16` 会忽略图里的 dtype 分界;要用 FP16 必须 **strongly-typed**

---

## 1. 背景

两个 ONNX 检测模型,导出格式是"裸 concat":

```
images[B,3,H,W] -> output[B, N, 4 + num_classes]
```

- `pred_boxes`:cxcywh,相对于输入图尺寸归一化(0~1)
- `pred_logits`:原始分类 logits,**图内不含任何后处理**

要求:用 C++ + TensorRT 实现推理,验证两个模型能正确输出检测框,
预处理/后处理对齐 `tools/inference/onnx_inf_v2.py`。

后来又追加了第二个任务:**搞清 FP16 掉精度的原因并解决**。

### 环境

| 项                          | 值                                                         |
| --------------------------- | ---------------------------------------------------------- |
| 容器                        | `nvrt1012_dev`(代码挂在 `/workspace/DEIMv2`)           |
| TensorRT                    | `/usr/local/tensorrt/TensorRT-10.12.0.36/`               |
| GPU                         | RTX 2070,8GB,**Turing(sm_75)**,支持 FP16 不支持 BF16 |
| CUDA / OpenCV / cmake / g++ | 12.8 / 4.5.4 / 3.22.1 / 11.4                               |

---

## 2. 预处理参数怎么定(不是猜的)

`onnx_inf_v2.py` 是从配置文件的 val 增强里推导预处理参数的,所以 C++ 侧也要按同一来源确定。

**两条证据链**:

1. **模型名里的尺寸与配置对得上**:`l-d4-512-896` ↔ `configs/deimv2/deimv2_dinov3_l_coco.yml`
   的 `input_size: [512, 896]`(格式 `[h, w]`,即 H=512 W=896)。
2. **骨干是 DINOv3**:查 ONNX 图,`images` 的第一个消费者是
   `/model/backbone/patch_embed/proj/Conv`,还有 `/model/backbone/sta/stem/...`,
   说明是 `deimv2_dinov3` 系列 → val 增强里有 `Normalize`,**要带 ImageNet 归一化**。
   (只有 `deim_dfine` / `deimv2_hgnetv2` 这类没有 `Normalize` op 的才只做 /255。)

所以两个模型的预处理都是:

```
直接拉伸(非 letterbox、无 padding)到 (W, H)
-> /255
-> (x - mean) / std,  mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]
-> HWC -> CHW,RGB
```

**因为训练/验证是直接拉伸、没有 padding,后处理把归一化的 cxcywh 乘 `(原图宽, 原图高)`
再转 xyxy 即可,不需要反算 letterbox 的 padding。** 这是选型上的一个关键简化。

---

## 3. C++ TensorRT 实现

### 3.1 工程结构

```
tools/trt_cpp/
├── CMakeLists.txt
├── src/
│   ├── trt_engine.{h,cpp}    引擎构建/缓存/反序列化 + 推理
│   ├── preprocess.{h,cpp}    PIL 兼容的拉伸 resize + 归一化
│   ├── postprocess.{h,cpp}   解码 + 全类别 NMS
│   └── main.cpp              CLI / 画框 / 结果输出
└── docs/
```

### 3.2 CMake 要点

- `find_package(CUDAToolkit REQUIRED)` + `find_package(OpenCV COMPONENTS core imgproc imgcodecs)`
- TensorRT 10.x **不提供 CMake config**,手工 `add_library(... SHARED IMPORTED)`
  导入 `libnvinfer.so`、`libnvonnxparser.so`、`libnvinfer_plugin.so`
  (第三个必须有,`initLibNvInferPlugins` 在里面,不在 libnvinfer)
- 设置 `BUILD_RPATH` / `INSTALL_RPATH` 指向 TensorRT 的 lib,运行时不用 `export LD_LIBRARY_PATH`

### 3.3 引擎构建(TRT 10.x API)

```cpp
auto builder = nvinfer1::createInferBuilder(logger);
auto network = builder->createNetworkV2(flags);   // 0U=弱类型;kSTRONGLY_TYPED=强类型
auto parser  = nvonnxparser::createParser(*network, logger);
parser->parseFromFile(onnx_path, (int)nvinfer1::ILogger::Severity::kWARNING);

auto config = builder->createBuilderConfig();
config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, mb << 20);
if (fp16) config->setFlag(nvinfer1::BuilderFlag::kFP16);

auto* profile = builder->createOptimizationProfile();   // 注意:返回裸指针
profile->setDimensions("images", kMIN/kOPT/kMAX, dims);
config->addOptimizationProfile(profile);

auto serialized = builder->buildSerializedNetwork(*network, *config);
// 存盘 + 反序列化
runtime->deserializeCudaEngine(serialized->data(), serialized->size());
```

**几个必须知道的 API 细节(面试常问)**:

| 点                       | 说明                                                                                                                                             |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| 对象释放                 | TRT 10**用 `delete`**,不再有 `destroy()`。接口有虚析构,可以直接 `unique_ptr` 管理                                                    |
| `IOptimizationProfile` | 析构函数是**protected**,而且文档明确写 "The builder retains ownership... users must not attempt to delete"。**用裸指针,不要 delete** |
| 动态 shape               | 必须建 optimization profile,min/opt/max 三个都要设;opt 定在 1(一次跑一张图)                                                                      |
| 静态 batch 的 ONNX       | profile 三个值必须**都等于**网络声明值,否则 `setDimensions` 校验不过                                                                     |
| 强类型网络               | 各张量精度完全由 ONNX 的 dtype 决定,`kFP16` flag 会被忽略(两者互斥)                                                                            |

### 3.4 推理

```cpp
context->setInputShape("images", Dims4{batch,3,h,w});
auto out_dims = context->getTensorShape("output");
if (hasDynamicDim(out_dims)) context->inferShapes(0, nullptr);  // 输出有动态维时兜底
cudaMalloc(...); context->setTensorAddress(name, ptr);
cudaMemcpyAsync(in, host_in, ..., cudaMemcpyHostToDevice, stream);
context->enqueueV3(stream);
cudaMemcpyAsync(host_out, out, ..., cudaMemcpyDeviceToHost, stream);
cudaStreamSynchronize(stream);
```

- 网络的 query 数和类别数**运行时从 engine 形状推**:`N = shape[1]`, `C = shape[2] - 4`,
  不写死,换模型不用改代码
- TRT 张量是行优先,`[B,N,4+C]` 的显存布局和 numpy 的 `pred` 一致,可以直接 memcpy
- 只支持单输入单输出、FP32 I/O、H/W 静态的网络(本项目的导出图正好满足)

### 3.5 引擎缓存

默认缓存到 `<onnx 路径>.engine`。换 GPU / 换 TensorRT 版本 / 换模型后要 `--rebuild`。
反序列化失败时自动回退到从 ONNX 重建,不会卡死。

---

## 4. 预处理:PIL 逐像素一致(这是最花功夫的一处)

目标:C++ 的 resize 结果要和 `PIL.Image.resize(..., Image.BILINEAR)` **逐像素相同**,
否则后面的精度对比就没有意义。

### 4.1 Pillow 的系数公式

```
scale       = in_size / out_size
filterscale = max(1.0, scale)        // 缩小时把滤波核展宽 = 抗锯齿
support     = 1.0 * filterscale      // 三角滤波器的 support = 1.0
center      = (i + 0.5) * scale
w(j)        = triangle((j + 0.5 - center) / filterscale),再归一化到 sum = 1
```

关键点:**缩小时 `filterscale > 1`,滤波核被展宽**。普通的 `cv::INTER_LINEAR` 不做这一步,
所以直接用它和 PIL 对不上。

### 4.2 真正的坑:两趟的顺序 + 中间量化

我先按"float 水平 → float 垂直"实现,结果和 PIL 差约 **1 个灰阶**(max|d| ≈ 1/255)。
写了个 numpy 探针枚举四种候选:

| 变体                           | max\|d\|        | 逐像素相同   |
| ------------------------------ | --------------- | ------------ |
| A float H → float V           | 0.997           | ✗           |
| B float V → float H           | 0.997           | ✗           |
| C uint8 V → uint8 H           | 1.000           | ✗           |
| **D uint8 H → uint8 V** | **0.000** | **✓** |

**Pillow 是"水平 → 量化成 uint8 → 垂直 → 量化 uint8"**,中间那一次量化必须跟着做。

### 4.3 第二个坑:归一化用除法

参考实现是 `x / 255`。写 `x * (1.0f/255)` 会在 float32 末位产生差异,
导致两边预处理不能逐位对齐。**改成除法后,预处理与 PIL 逐元素完全一致(0 个不同元素)。**

### 4.4 其他

- 图片解码:OpenCV `imread` 得到 BGR,`cvtColor` 转 RGB
- 预处理是单线程 CPU 实现,512×896 约 10 ms;要提速可以 OpenMP 按行并行或挪到 CUDA

---

## 5. 后处理

和 `onnx_inf_v2.py` 的 `decode_output` / `class_agnostic_nms` 一一对应:

```
sigmoid(logits) -> 每个 query 取最高分类别 -> 分数阈值过滤
-> cxcywh(相对输入图 0~1) × 原图 (w, h) -> xyxy
-> 全类别 NMS(忽略类别,只按框重合度过滤,IoU 阈值默认 0.9)
```

**为什么是"全类别" NMS**:DETR 系模型每个 query 出一个框,同一目标通常只有一个 query 响应;
用 class-agnostic 是为了去掉高度重合的重复框,而不是按类别分别抑制。

**sigmod 要做数值稳定实现**:`x >= 0` 用 `1/(1+exp(-x))`,`x < 0` 用 `exp(x)/(1+exp(x))`。

---

## 6. 验证方法论

### 6.1 三级验证

1. **逐元素**:C++ 预处理后的输入张量 vs Python 的 `feed`(`--dump-input` / `--dump-output`)
2. **整链路**:C++ 结果 vs `onnx_inf_v2.py`(用户自己确认过可用的工具)
3. **负对照**:去掉 ImageNet 归一化看是否变差(L 的框数 16→12;X 在阈值 0.05 下冒出
   113 个框退化成噪声)→ 证明归一化确实是这两个模型需要的,而不是"凑出来的"

### 6.2 一个必须修的对比 bug

最初的 `compare_with_onnx_inf.py` 是**按位置逐框对比**。两侧分数有微小差异时,
NMS 的输出顺序会变,于是拿 A 的框去比 B 的框,**报出过 349 像素的假偏差**。
改成按 IoU 贪心配对后才正常。

**教训:凡是"两个实现的输出列表"做对比,都必须按几何/语义配对,不能按位置 zip。**

### 6.3 看指标要看对地方

原始输出的 `max|d|` 经常是 1e-1 甚至更大 —— 但这些差异**几乎全落在低分 query 上**
(分数接近 0 时 sigmoid 对 logits 极其敏感)。真正决定检测结果的是
**配对后的框坐标和分数**。所以对比报告里我把"检测框配对"作为主要指标。

---

## 7. FP16 精度问题:完整排查过程

这一节是整件事最有价值的部分,面试时按"现象 → 假设 → 证伪 → 再假设"的节奏讲。

### 7.1 现象

直接 `--fp16`(弱类型 + `BuilderFlag::kFP16`):

|           | 耗时    | 配对数 | 最低 IoU | 最大框差          |
| --------- | ------- | ------ | -------- | ----------------- |
| FP32 基准 | 31.8 ms | 21/21  | —       | —                |
| 纯 FP16   | 9.1 ms  | 18/21  | 0.583    | **34.3 px** |

3.5 倍加速很香,但精度不可接受。TensorRT 构建时自己就警告:

```
[TRT][W] Detected layernorm nodes in FP16.
[TRT][W] Running layernorm after self-attention with FP16 Reduce or Pow may cause overflow.
```

### 7.2 第一次走弯路:跟着警告找 LayerNorm / RMSNorm

顺着警告定位到 decoder:ONNX **opset 17 没有 RMSNorm 算子**,所以
`engine/deim/deim_decoder.py` 的 `norm1/norm2/norm3` 被拆成 126 个
`Pow → ReduceMean → +eps → Sqrt → Div` 节点。`x²` 在 FP16 里超过 256 就溢出(上限 65504)。

**证伪**:把这些层钉回 FP32(`--fp16-fp32-norm`)后,精度确实改善到 20/21、5.7 px,
**但还不够**;而且后面发现 decoder 的 RMSNorm 根本不是主要矛盾。

### 7.3 第二次走弯路:错误的"等价物"

在拿到真实 checkpoint 之前,我用 `onnxruntime.transformers.float16.convert_float_to_float16`
手工把现成 ONNX 转成混合精度图当作等价物,结果**不管怎么调 blocklist 都是 34 px**,
于是错误地得出"导出侧这条路走不通"的结论。

**后来发现这个等价物是错的**:盲转会把 DINOv3 RoPE 的 `Cos`/`Sin` 也降成 FP16,
位置编码失真,钉什么都不管用。

**真正的 `.half()` 导出不会** —— 因为 DINOv3 作者把 dtype 写死在模块里:

```python
self.dtype = dtype          # Don't rely on self.periods.dtype   ← 这行注释就是答案
register_buffer("periods", torch.empty(..., dtype=dtype))
...
def forward(self, *, H, W):
    dtype = self.dtype                     # Python 属性,.half() 动不了它
    dd = {"device": device, "dtype": dtype}   # 用它构造中间张量 → 强制 FP32
```

而 attention 侧也有配套设计:

```python
# apply_rope:All operations will use the dtype of rope,
#            the output is cast back to the dtype of q and k
```

实测 dtype 对照(`Cos`/`Sin`):

|       | 手工转 ONNX    | `.half()` 导出  |
| ----- | -------------- | ----------------- |
| L / X | **FP16** | **FP32** ✓ |

**教训:用"近似的等价物"去验证方案,比不验证更危险 —— 会得出方向性错误的结论。**

### 7.4 把 monkeypatch 搬进模型代码

`.half()` 导出路上踩到几个 dtype 硬编码的坑。最初写在导出脚本里做 monkeypatch,
后来按用户要求搬进了模型代码(全部在 FP32 下是空操作,不影响训练,
所以**没有用 `torch.onnx.is_in_onnx_export()` 分支**——那样反而会让导出图和训练图不一致):

| 问题                                                                                                                                                                                                                                                           | 修法                                                                                                       | 文件                              |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | --------------------------------- |
| `pos_embed` 用 `setattr` 存成**普通属性**(旁边 `register_buffer` 被注释掉了),`.half()`/`.to(device)` 都不会带上它 → encoder 里 `src + pos_embed` 把激活提升回 FP32 → `expected m1 and m2 to have the same dtype, but got: float != Half` | 改成`register_buffer(..., persistent=False)`(不进 `state_dict`,**不影响任何 checkpoint 的加载**) | `engine/deim/hybrid_encoder.py` |
| deformable attention 的采样网格是`torch.tensor(value_spatial_shapes)` 现算的,不受 `.half()` 影响留在 FP32,而 `value` 已是 FP16 → `grid_sample` 报 dtype 不符                                                                                          | 在`deformable_attention_core_func_v2` 里把网格对齐到 `value.dtype`(网格量级 0~1,FP16 精度足够)         | `engine/deim/utils.py`          |
| **新发现**:`Integral.forward` 里 `project.to(x.device)` **只转 device 不转 dtype**,FP32 的 `project` 和 FP16 的 `x` 做 `F.linear` 直接报错                                                                                               | 改成`.to(device=x.device, dtype=x.dtype)`                                                                | `engine/deim/dfine_decoder.py`  |

这些都是**逐个踩出来的** —— 每修一个,导出就往前走一层再报下一个。
说明这套模型代码当初不是按混合精度推理设计的。

### 7.5 顺手清掉的隐患:`anchors` 常量里的 Inf

`_generate_anchors()` 把 padding 位置的锚点填成 `torch.inf`,这个 inf 会被**导出成 ONNX 常量**
(L 有 1648 个、X 有 1168 个 Inf)。而 RMSNorm 的 `x * rsqrt(mean(x²)+eps)` 遇到 Inf 会得到
`Inf * 0 = NaN` —— 看着非常像元凶。

**修法**:改成有限常量

```python
ANCHOR_INVALID_FILL = 100.0   # sigmoid(100) 在 FP32/FP16 下都精确等于 1.0,与 inf 行为一致
```

改完四个导出的图里**非有限常量从 1648/1168 个降到 0 个**,`anchors` 范围变成 `[-4.30, 100.00]`。
语义等价有依据:下游只有 `sigmoid` 和 `inverse_sigmoid`(后者按 `eps=1e-5` 截断),
`sigmoid(inf)` 和 `sigmoid(100)` 都精确是 1.0,训练时梯度同样饱和为 0。

**但 L 的 FP16 依然全 NaN。所以 Inf 不是根因**,只是把图里一个明确的隐患清掉了。

### 7.6 用 TensorRT IDebugListener 定位到 backbone 第一个 block

工具里加了 `--debug-nan`,底层是 TensorRT 的 debug tensor 机制:

- 构建时:`network->markUnfusedTensorsAsDebugTensors()`(不阻止融合,代价小)
  或 `markDebug()`(逐个标记,会阻止融合)
- 运行时:`context->setUnfusedTensorsDebugState(true)` + `context->setDebugListener(listener)`
- 回调 `processDebugTensor(addr, location, type, shape, name, stream)` 里把张量拷回 host 检查

扫描结果(L FP16,361 个未融合张量):

```
/inner/model/backbone/sta/stem/stem.2/Mul_1_output_0    正常   ← stem 输出
/inner/model/backbone/blocks_0/mlp/fc2/Add_output_0     NaN    ← 第一个 NaN
/inner/model/backbone/blocks_1/mlp/fc2/Add_output_0     NaN
...                                                      (346/361 全是 NaN)
```

**输入是好的,NaN 在 backbone 的第一个 transformer block 内部产生。**

**两个 API 限制(面试可以讲)**:

- `markDebug` 和 `--strongly-typed` 在 TRT 10.12 下**冲突**,TRT 内部断言
  `[genericPwFusion.cpp] Assertion !stronglyTyped failed`;只能在弱类型下做细查
- 这个模型的张量名有重名(带 `.1` 后缀),`markDebug` 会再触发一个
  `shapeContext.cpp::renameDebugTensor` 断言失败

所以在弱类型 FP16 下排查、再回到强类型验证,是这次实际走的路径。

### 7.7 换思路:直接量激活量级

TRT 侧细粒度受限,改在 PyTorch 里量(脚本 `z_trt_build/act_stats.py`):

- L 的激活最大只有 **3280**,远低于 FP16 上限 65504,FP32 下**没有任何 NaN/Inf**
- 但 `q·k` 是**逐通道累加**,量级会比 q/k 本身大几个数量级

直接量 `|q·k|·scale`(12 次注意力调用):

|             | logits 最大         | 中位 | 超 FP16 上限 65504 |
| ----------- | ------------------- | ---- | ------------------ |
| **L** | **2,518,337** | 101  | **超 38 倍** |
| X           | 67,110              | 42   | 超 1.02 倍(擦边)   |

**L 的 block0 注意力 logits 达到 251 万,在 FP16 里就是 Inf,`softmax(Inf)` → NaN**,
NaN 再顺残差一路污染整张图 —— 和"从 blocks_0 开始全是 NaN"完全对上。

**为什么 L 和 X 差这么多**:两个模型的骨干配置本来就不同 ——
`dinov3_vits16`(`ffn_ratio=4, ffn_layer=mlp`)vs `dinov3_vits16plus`(`ffn_ratio=6, ffn_layer=swiglu`),
激活量级自然不一样。X 只是勉强擦边,所以能用。

### 7.8 修法:注意力放 FP32 算

核心修复(只有一个函数):

```python
# engine/backbone/dinov3/layers/attention.py
def scaled_dot_product_attention(q, k, v, **kwargs):
    return F.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), **kwargs).to(v.dtype)
```

思路和 `apply_rope` 的 dtype 处理一致。**实测对训练逐位中性**(见附录二),所以无条件生效。

另外还有 4 处 dtype 硬编码的坑,是导出 FP16 时逐个暴露出来的(每修一个就往前走一层
再报下一个),细节见 [fp16_mixed_precision.md](fp16_mixed_precision.md) §3.3:

| 现象                                                               | 位置                           |
| ------------------------------------------------------------------ | ------------------------------ |
| `expected m1 and m2 ... float != Half`(`pos_embed` 是普通属性) | `hybrid_encoder.py`          |
| `expected scalar type Half but found Float`(采样网格 dtype)      | `deim/utils.py`              |
| 同上(`project` 只转了 device 没转 dtype)                         | `dfine_decoder.py`           |
| `anchors` 里的 `inf` 被导出成 ONNX 常量                        | `deim/utils.py` / 各 decoder |

### 7.9 结果

3 张测试图(L 33 个框 / X 50 个框):

| 方案                                   | 推理耗时                     | 配对数                 | 最低 IoU                  | 最大框差                   |
| -------------------------------------- | ---------------------------- | ---------------------- | ------------------------- | -------------------------- |
| FP32(基准)                             | L 39.2 / X 32.1 ms           | 33/33、50/50           | —                        | —                         |
| 纯 FP16(弱类型`kFP16`)               | 9.1 ms                       | 18/21                  | 0.583                     | **34.3 px**          |
| **修复后(强类型 + FP32 注意力)** | **L 24.8 / X 16.1 ms** | **33/33、50/50** | **0.9519 / 0.9872** | **0.820 / 0.591 px** |

L 从"全 NaN、0 个框"变成 33/33 配对、框差 0.82 px。X 也顺带变准了
(它的注意力 logits 本来就擦边超限)。

**代价**:加速比从纯 FP16 的 3.5× 降到 1.58×~2.00× —— ViT 里注意力占比不小。
但纯 FP16 是"34 px 偏差 + L 全 NaN",没有可比性。

**副产品**:注意力改成无条件之后,**半精度模型在 PyTorch 里也能直接前向**
(不只是导出),实测 0 个 NaN。

### 7.10 案例二:1088×1920 高分辨率

后来又拿到一对 1088×1920 的模型,**同一个根因在高分辨率下更容易触发**,
并且暴露了显存和加速比两个新问题。完整版见
[fp16_mixed_precision.md](fp16_mixed_precision.md) §7。

**两个图的差异**(输入输出相同,内部 dtype 完全不同):

| | `...-fp16s-...onnx`(98 MB) | `...-fp16-...onnx`(195 MB) |
| --- | --- | --- |
| initializer | **516 FP16** + 1 FP32 | 517 FP32 |
| Conv | 62 × FP16 | 62 × FP32 |
| MatMul | 164 FP16 + **24 FP32** | 188 × FP32 |
| **Softmax** | **12 FP32** + 20 FP16 | 32 × FP32 |
| LayerNormalization | 29 × FP16 | 29 × FP32 |
| Pow / ReduceMean / Cos / Sin | **FP32** | FP32 |

`fp16s` 就是 §7.8 那套混合精度导出的产物(后缀 **s** = strong),
12 个 FP32 Softmax + 24 个 FP32 MatMul 正好是 12 层 ViT 的注意力。
`fp16` 是**纯 FP32 图** —— 文件名里的 `fp16` 指"打算用 FP16 推理",不是图本身的精度。

**四种推理方式**(测试图 500×375):

| 变体 | ONNX 图 | 构建方式 | 耗时 | 检测框 | NaN |
| --- | --- | --- | --- | --- | --- |
| **A** | `fp16s` | `--strongly-typed` | 416.5 ms | **14** ✅ | 0 |
| **B** | `fp16` | `--fp16`(弱类型 `kFP16`) | 80.4 ms | **0** ❌ | 25500/25500 |
| C | `fp16` | (FP32 基准) | 496.5 ms | 14 | 0 |
| D | `fp16` | `--fp16-fp32-norm` | 80.2 ms | **0** ❌ | 25500/25500 |

> ⚠️ B/D 的 80 ms 是假象 —— 快是因为算出 NaN 后退化了,不是真加速。

**A vs C** 3 张图:**38/39 配对,最低 IoU 0.9839,最大框差 0.347 px**。

**为什么 B/D 崩**:TRT 的弱类型保护只覆盖 Reduce/Pow(RMSNorm 归约),
**不管矩阵乘和 softmax**。变体 D(把 126 个 norm 层全钉回 FP32)仍然全 NaN,
证明问题不在 norm —— 和 §7.2 在小模型上的结论一致。

**分辨率放大的连锁反应**:

| | 416×768 | 1088×1920 | 倍数 |
| --- | --- | --- | --- |
| ViT patch 数(16×16) | 26×48 = **1,248** | 68×120 = **8,160** | 6.5× |
| 注意力矩阵元素(6 头) | 9.3 M | **400 M** | **43×** |
| 该矩阵 FP32 大小 | 37 MB | **1.6 GB** | 43× |
| FP32 推理耗时 | 32.1 ms | 496.5 ms | 15× |
| **加速比(FP16)** | **2.00×** | **1.19×** | — |

**显存**:默认 workspace 2048 MB 直接构建失败 ——

```text
Skipping tactic 0 due to insufficient memory on requested size of 3262730200
Could not find any implementation for node {ForeignNode[.../blocks.11/attn/...]}
```

3.26 GB 正好对上 **FP32 注意力矩阵 1.6 GB + softmax 输出 1.6 GB**。
加 `--workspace 5120` 后可通过(8 GB 的 RTX 2070 已比较吃紧)。

**加速比缩水 —— 逐层 profiling 的结论**:注意力变成了**内存带宽瓶颈**。
端到端(交替复测取最小,消除降频影响):**fp32 475.1 ms → fp16 389.2 ms = 1.22×**。

| | fp32 | fp16 | 变化 |
| --- | --- | --- | --- |
| **注意力** | 421.9 ms(**73%**) | **375.6 ms(90%)** | −11%(几乎没变) |
| **非注意力** | 153.5 ms(27%) | **43.1 ms(10%)** | **−72%(3.6×)** |

FP16 在非注意力部分生效了(3.6×),但只占 10%;注意力占 90% 却拿不到收益。

**原因可以算出来**:单个 ViT softmax 层处理 `[1,6,8160,8160]`(4 亿元素),
FP32 下 1.598 GB,两趟读写 = **6.39 GB**,除以 RTX 2070 的 448 GB/s =
**理论 14.26 ms**,**实测 14.5 ms(fp32)/ 13.8 ms(fp16)** ——
达到理论带宽的 **~97%**,**已经跑在内存带宽上限,算力提升无效**。
而强制 FP32 还让矩阵从 0.8 GB 变成 1.6 GB,进一步加重带宽压力。

**旁证**:变体 B(整网 FP16,NaN 不可用)只要 80.4 ms → FP32 注意力大约吃掉 300+ ms。

**优化方向**:① 只对真正超限的层用 FP32(`act_stats.py` 逐层量);
② 从分辨率入手 —— patch 16 → 32 可让 token 数降 4 倍、注意力矩阵降 **16 倍**;
③ 若数值允许让 softmax 走 FP16,矩阵减半,理论耗时 14.3 → 7.1 ms。

---

## 8. 最终精度基线

3 张测试图(原图 500×375、缩小 350×262、放大 700×525),
四个环节共用同一套预处理和后处理(阈值 0.25、全类别 NMS 0.9)。

**L 模型(共 33 个检测框)**

| 对照                     | sigmoid 后 max\|d\| | 配对  | 最低 IoU         | 最大框差           | 最大分数差 |
| ------------------------ | ------------------- | ----- | ---------------- | ------------------ | ---------- |
| ORT FP32 vs PyTorch FP32 | 9.2e-02             | 33/33 | **1.0000** | **0.000 px** | 4.7e-06    |
| TRT FP32 vs PyTorch FP32 | 9.7e-06             | 33/33 | **1.0000** | **0.000 px** | 3.6e-06    |
| TRT FP16 vs PyTorch FP32 | 3.1e-01             | 33/33 | 0.9519           | **0.820 px** | 6.0e-03    |
| TRT FP16 vs TRT FP32     | 3.1e-01             | 33/33 | 0.9519           | 0.820 px           | 6.0e-03    |

**X 模型(共 50 个检测框)**

| 对照                     | sigmoid 后 max\|d\| | 配对  | 最低 IoU         | 最大框差           | 最大分数差 |
| ------------------------ | ------------------- | ----- | ---------------- | ------------------ | ---------- |
| ORT FP32 vs PyTorch FP32 | 4.0e-05             | 50/50 | **1.0000** | **0.000 px** | 7.1e-06    |
| TRT FP32 vs PyTorch FP32 | 1.5e-01             | 50/50 | **1.0000** | **0.000 px** | 2.7e-06    |
| TRT FP16 vs PyTorch FP32 | 4.6e-01             | 50/50 | 0.9872           | **0.591 px** | 1.8e-02    |
| TRT FP16 vs TRT FP32     | 4.6e-01             | 50/50 | 0.9872           | 0.591 px           | 1.8e-02    |

**三条结论**:

1. **PyTorch → ORT FP32 → TRT FP32 三段完全一致**:100% 配对、最低 IoU 1.0000、
   框坐标差 **0.000 px**,分数差 1e-6 量级。导出和两个运行时都没引入可测量的偏差。
2. 表里 `sigmoid 后 max|d|` 那些 1e-1 量级的数字**都落在低分 query 上**,对检测结果无影响。
3. `TRT FP16 vs PyTorch` 与 `TRT FP16 vs TRT FP32` 数值相同 → **误差全部来自 FP16 本身**,
   FP32 链路零贡献。

---

## 9. 用到的工具与命令

```bash
# 构建(容器内)
cd /workspace/DEIMv2/tools/trt_cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)

# 推理
./build/deimv2_trt_infer --onnx <model.onnx> --image example.jpg --thrh 0.25

# 导出混合精度图 + 强类型构建(要用 FP16 时的标准流程)
python3 tools/deployment/export_onnx_v2.py -c <cfg> -r <ckpt> --backbone <bb> \
    --fp16 --output z_onnx/xxx_fp16.onnx
./build/deimv2_trt_infer --onnx z_onnx/xxx_fp16.onnx --strongly-typed --image example.jpg

# 找 NaN
./build/deimv2_trt_infer --onnx xxx.onnx --strongly-typed --debug-nan --nan-dump nan.txt ...
./build/deimv2_trt_infer --onnx xxx.onnx --list-tensors "blocks.0" ...   # 只看名字不构建
```

**工具清单**

| 文件                                               | 用途                              |
| -------------------------------------------------- | --------------------------------- |
| `tools/trt_cpp/`                                 | C++ 推理工具主工程                |
| `tools/trt_cpp/verify_reference.py`              | 生成 ORT+PIL 的参考裸 f32         |
| `tools/trt_cpp/compare_with_reference.py`        | 逐元素 + 检测框配对对比           |
| `tools/trt_cpp/compare_with_onnx_inf.py`         | 与`onnx_inf_v2.py` 的检测框对比 |
| `z_trt_build/act_stats.py`                       | 量各模块激活量级 + 注意力 logits  |
| `z_trt_build/acc/{torch_ref,ort_ref,compare}.py` | 四段精度基线对比                  |

---

## 10. 面试问答准备

**Q:为什么用 TensorRT 而不是直接 ONNX Runtime?**
TensorRT 会针对具体 GPU 做算子融合、kernel 自动调优(tactic 选择)和精度选择,
在 NVIDIA 卡上通常比 ORT 快不少;代价是 engine 与 GPU 型号/TRT 版本绑定,需要构建和缓存。
ORT 的优势是跨平台、开箱即用。生产上常见做法是离线构建 engine 随模型一起发布。

**Q:engine 为什么要缓存?构建一次多久?**
构建要对每一层枚举 kernel 并实测计时,这两个模型在 RTX 2070 上单个要几分钟。
缓存成 `<onnx>.engine` 后反序列化只要 1 秒左右。换 GPU/TRT 版本/模型必须 `--rebuild`,
代码里做了反序列化失败自动回退重建。

**Q:动态 shape 怎么处理?**
ONNX 里 batch 维是动态的,所以构建时必须建 optimization profile,给 min/opt/max 三组尺寸。
我把 min=1、opt=1、max=`--max-batch`,opt 定在 1 是因为部署就是一次一张图。
**踩过的坑**:一开始我把三个值都设成 `max_batch`,结果 `--max-batch 4` 的 engine
按单张图推理会直接失败。ONNX 若是静态 batch,三个值必须都等于声明值。

**Q:strongly-typed 和弱类型有什么区别?**
弱类型网络里 `BuilderFlag::kFP16` 让 TensorRT 自由决定每层精度(会插入 reformat);
强类型网络的精度完全由 ONNX 里的张量 dtype 决定,TRT 必须尊重。
**这正是 FP16 精度问题的关键**:模型代码里本来写了 FP32 保护(比如 `x.float()`),
但整个 ONNX 是 FP32,那个 Cast 是空操作;弱类型下 TRT 照样把层降成 FP16。
TRT 10.12 已经把 `kFP16` 标记为 deprecated,官方方向就是强类型。

**Q:FP16 掉精度你怎么排查的?**
按"现象 → 假设 → 证伪"的节奏:

1. 先怀疑 TensorRT 警告指向的 layernorm/RMSNorm —— **钉回 FP32 后不够**,证伪
2. 用 TensorRT 的 debug tensor 逐张量扫 NaN,定位到 **backbone 第一个 block**
3. TRT 侧细粒度受限(`markDebug` 和强类型冲突、张量重名触发断言),
   改在 **PyTorch 里量激活量级**:发现激活本身才 3280(远低于 65504),
   但**注意力 logits 达到 251 万**,是 FP16 上限的 38 倍
4. 根因确认:`q·k` 逐通道累加 → FP16 溢出成 Inf → `softmax(Inf)` = NaN → 顺残差污染全图
5. 修法:注意力放 FP32 算,量级问题消失

**Q:FP32 图直接加 `--fp16` 行不行?**
**不行,会全 NaN。** 实测(1088×1920 的模型,4 个变体):

| ONNX 图 | 构建方式 | 检测框 | NaN |
| --- | --- | --- | --- |
| `fp16s`(混合精度图) | `--strongly-typed` | **14** ✅ | 0 |
| `fp16`(FP32 图) | `--fp16`(弱类型 `kFP16`) | **0** ❌ | 25500/25500 |
| `fp16`(FP32 图) | `--fp16-fp32-norm` | **0** ❌ | 25500/25500 |
| `fp16`(FP32 图) | (FP32 基准) | 14 | 0 |

TensorRT 在警告里说它会把 Reduce/Pow 强制回 FP32 —— **但那只覆盖 RMSNorm 的归约,
不管矩阵乘和 softmax**。把 126 个 norm 层全钉回 FP32(第 3 行)仍然全 NaN,
证明问题不在 norm 而在注意力。

**所以只有一条路:用 `--fp16` 导出的混合精度图 + `--strongly-typed`。**

**Q:为什么不能直接把整个模型用 FP16?**
两个原因:
(a) 有些数值敏感的部分(位置编码的三角函数、归一化的归约)在 FP16 下会失真或溢出;
(b) 更重要的是**累加类算子**:`q·k`、`x²` 这种逐通道累加,输入量级不大但累加结果可能超大,
FP16 的动态范围(上限 65504)扛不住。
所以正确做法是**混合精度**:大部分走 FP16,把累加/归一化/三角函数这些留在 FP32。

**Q:怎么保证 C++ 实现和 Python 参考一致?**
分三层:

1. **预处理逐位对齐**:先把 resize 做到和 PIL 逐像素相同 ——
   关键是复刻 Pillow 的系数公式(缩小时展宽滤波核)和"水平→uint8→垂直→uint8"的两趟结构,
   还有归一化用除法而不是乘倒数。最终实测 **0 个元素不同**。
2. **中间结果可导出**:`--dump-input` / `--dump-output` 把张量存成裸 f32,可以直接和 numpy 对比。
3. **端到端**:和 `onnx_inf_v2.py` 的检测框做配对对比,以及和 PyTorch/ORT 的三段一致性验证。

**Q:对比两个实现的检测结果要注意什么?**
**不能按位置 zip**。两侧分数有微小差异时 NMS 输出顺序会变,按位置比会张冠李戴 ——
我实际踩过,报出过 349 像素的假偏差。要**按 IoU 贪心配对**再比。
另外**指标要看对地方**:原始输出的 max|d| 常是 1e-1 量级,但差异都落在分数接近 0 的
低分 query 上,真正有意义的是配对后的框坐标和分数。

**Q:为什么预处理不用 letterbox?**
因为训练/验证就是**直接拉伸**到固定尺寸,没有 padding。
推理必须和训练一致,否则分布不匹配。好处是后处理可以简化:
归一化的 cxcywh 直接乘 `(原图宽, 原图高)` 就还原了,不需要反算 padding。

**Q:NMS 为什么用 class-agnostic?**
DETR 系每个 query 输出一个框和一组类别分数,同一目标一般只有一个 query 响应。
用全类别 NMS 是为了去掉高度重合的重复框。如果按类别分别抑制,
跨类别的重复框(比如同一目标被分别判成 person 和 car)会留下来。

**Q:换 GPU 或者换 TensorRT 版本会怎样?**
engine 与 GPU 架构和 TRT 版本强绑定,必须重新构建。
代码里反序列化失败会自动回退到从 ONNX 重建。另外 FP16 的加速比依赖 GPU 的
FP16 吞吐(Turing 及以上才明显);`platformHasFastFp16()` 可以探测。

**Q:如果让你继续优化,你会做什么?**

- 预处理目前是单线程 CPU(10 ms,占比不低),可以 OpenMP 按行并行或做成 CUDA kernel
- 注意力只有少数层会溢出,可以只对**真正超限的那几次**用 FP32,其余保持 FP16,
  把加速比从 2.0× 往上提(需要更细的定位)
- 量化感知训练(QAT)才是拿到 FP32 等价精度又保留速度的正路,但需要重训
- 多 batch / 多流并发、以及 `--max-batch` 下的吞吐优化

**Q:`torch.onnx.is_in_onnx_export()` 能把导出和训练分开吗?**
**能,但不是所有地方都能用,而且有的地方用了反而会坏事。** 三个坑:

1. **构造期的代码,守卫永远不会生效。** `_generate_anchors()` / `_reset_parameters()`
   都在 `__init__` 里被调用,而模型在 `torch.onnx.export()` 之前就构造好了 ——
   那时 `is_in_onnx_export()` 必然是 `False`。写成 `if is_in_onnx_export(): register_buffer else: setattr` **等于把修复取消了**,导出时又退回原样。
2. **它漏掉导出前的那次 eager 前向。** `export_onnx_v2.py` 在 trace 之前会跑一次
   `out = model(data)`(用来拿 `out.shape[1]` 打印 query 数),那一步它为 `False`,
   而硬报错正好发生在那里。实测:不加修复 ❌、纯 `is_in_onnx_export()` ❌、
   `is_in_onnx_export() or not is_autocast_enabled()` ✅。
3. **精度类的改动加了它还不够。** 注意力一开始用它,但导出前的 eager 前向、
   以及直接在 PyTorch 里跑半精度模型都保护不到,L 照样 NaN。

**判据最终按改动性质分三类**:

| 改动性质                       | 判据                                                 |
| ------------------------------ | ---------------------------------------------------- |
| 构造期确定                     | **无条件改** + 用 state_dict/逐位对比证明中性  |
| forward 里、**硬报错**   | `is_in_onnx_export() or not is_autocast_enabled()` |
| forward 里、**精度问题** | **无条件改**(前提:实测对训练中性)              |

**Q:怎么保证改动不影响训练?**
**逐位实测,不靠推理。** 三层:

1. **逐位对比**:训练代码里 `scaler is not None` 走 AMP、否则走纯 FP32,
   而训练脚本默认 `USE_AMP="${USE_AMP:-1}"` —— 所以**两种模式都要测**。
   同一份 ckpt、同样输入,改动前后前向输出对比(实测两模型 FP32/AMP 都是 `max|d| = 0.000e+00`)。
2. **state_dict 不变**:改动涉及 buffer 时必须确认键集合一致(实测 463/551 个键完全相同),
   否则旧 ckpt 加载会报错、或者 EMA 同步出问题。
3. **结构性隔离**:能用守卫的地方就用(构造期不行),让训练侧根本不执行新代码。

正是靠逐位对比才发现:第一版里 `utils.py` 那处改动**真的改变了 AMP 结果**(max|d|=1.115)。

**Q:PyTorch 里能直接跑 FP16 推理吗?**
**能**(现在可以)。把注意力改成无条件之后:

```python
model = <模型>.half().eval().cuda()
out = model(feed.half())        # 不需要 autocast
```

实测 3 张图 0 个 NaN,L 33/33 配对(框差 2.107 px)、X 50/50(0.392 px)。
要点是:**`q·k` 的溢出不是导出才有的问题**,只要模型是半精度就会发生,
所以保护不能只在导出路径上做。

**Q:engine 构建报内存不足怎么办?**
高分辨率模型会遇到。1088×1920 时默认 workspace(2048 MB)直接构建失败:

```text
[TRT][W] UNSUPPORTED_STATE: Skipping tactic 0 due to insufficient memory
         on requested size of 3262730200 detected for tactic 0x0000000000000000.
[TRT][E] IBuilder::buildSerializedNetwork: Error Code 10: Internal Error
         (Could not find any implementation for node {ForeignNode[.../blocks.11/attn/...]}.)
```

**根因可以算出来**:ViT patch 数 68×120 = 8160,注意力矩阵 `[1,6,8160,8160]`
在 FP32 下是 **1.6 GB**,加 softmax 输出约 **3.2 GB** —— 正好是那个 3.26 GB。
而它超限是因为**注意力被强制在 FP32**(§7 的修复)。

加 `--workspace 5120` 即可通过。**代价是显存**:8 GB 的 RTX 2070 已比较吃紧。

**Q:分辨率变大会有什么影响?**
不只是"慢一点",注意力是平方复杂度:

| | 416×768 | 1088×1920 | 倍数 |
| --- | --- | --- | --- |
| ViT patch 数 | 1,248 | **8,160** | 6.5× |
| 注意力矩阵元素(6 头) | 9.3 M | **400 M** | **43×** |
| 该矩阵 FP32 大小 | 37 MB | **1.6 GB** | 43× |
| FP32 耗时 | 32.1 ms | 475.1 ms | 15× |
| FP16 耗时 | 16.1 ms | 389.2 ms | 24× |
| **加速比** | **2.00×** | **1.22×** | — |

**两个后果**:(a) 显存吃紧(见上);(b) **FP16 的收益被摊平**。

**(b) 的机理**(逐层 profiling 实测):注意力在 fp16 engine 里占 **90%** 的耗时,
却几乎拿不到 FP16 的收益 ——

| | fp32 | fp16 | 变化 |
| --- | --- | --- | --- |
| 注意力 | 421.9 ms(73%) | 375.6 ms(90%) | −11% |
| 非注意力 | 153.5 ms(27%) | **43.1 ms(10%)** | **−72%(3.6×)** |

**根因是内存带宽**:单个 ViT softmax 层要处理 `[1,6,8160,8160]` 矩阵,
FP32 下 1.598 GB,两趟读写 6.39 GB,除以 448 GB/s = **理论 14.26 ms**,
实测 13.8~14.5 ms —— **达到理论带宽的 ~97%,已经饱和**。
FP16 的优势在算力(2× FLOPS),对带宽瓶颈无效;强制 FP32 反而让字节数翻倍。

**推论**:高分辨率场景下"整层注意力走 FP32"代价极大。方向:
只对真正超限的层保护(先量)、或从分辨率入手(patch 16→32 可让注意力矩阵降 16 倍)。

**Q:TRT 两次构建结果一样吗?**
**不一样。** 实测同一个 ONNX 独立构建两次,输出 `max|d| = 1.612e-04`。
这是 TensorRT 构建时枚举 kernel 并实测计时(tactic 选择)导致的,不保证逐位一致。
但差异分布很关键:logits 上是 1.6e-4,**框坐标只有 ~6e-7**(float32 精度极限),
所以检测结果不受影响。
**这条的启示**:用"逐位相同"做判据时,TRT 侧天然做不到,必须区分
"框架/代码的确定性"和"TRT 构建的确定性";评估检测精度要用 IoU 和框坐标,不是张量 max|d|。

**Q:这次最大的收获是什么?**
**"用近似的等价物去验证方案,比不验证更危险。"**
我一度用 `convert_float_to_float16` 手工转图当作 `.half()` 导出的等价物,
得出了"导出侧这条路走不通"的方向性错误结论。后来发现那个等价物会把 RoPE 的
`Cos/Sin` 也降成 FP16,而真正的 `.half()` 导出不会。
**验证工具的保真度,和验证结论本身一样重要。**

---

## 附录一:改动清单

对应 commit `6a1f0be`「修复 deimv2_dinov3 L 模型 FP16 精度损失」,
**5 个文件、7 处改动**,全部实测对训练逐位中性(证据见附录二)。

**模型代码**

| 文件                                           | 改动                                              | 形式   |
| ---------------------------------------------- | ------------------------------------------------- | ------ |
| `engine/backbone/dinov3/layers/attention.py` | 注意力放 FP32 算(**关键修复**)              | 无条件 |
| `engine/deim/utils.py`                       | 定义`ANCHOR_INVALID_FILL = 100.0`(各解码器共用) | 构造期 |
| `engine/deim/utils.py`                       | deformable attention 采样网格对齐`value.dtype`  | 带守卫 |
| `engine/deim/deim_decoder.py`                | `anchors` 无效填充改用 `ANCHOR_INVALID_FILL`  | 构造期 |
| `engine/deim/dfine_decoder.py`               | 同上(DFINETransformer)                            | 构造期 |
| `engine/deim/dfine_decoder.py`               | `Integral` 的 `project` 补 dtype              | 带守卫 |
| `engine/deim/hybrid_encoder.py`              | `pos_embed` 注册成 buffer(`persistent=False`) | 构造期 |

> `engine/deim/rtdetrv2_decoder.py` 的 `_generate_anchors()` 仍用 `torch.inf` ——
> 当前模型不走这条路径,故未改。

**导出脚本**:`tools/deployment/export_onnx_v2.py` 新增 `--fp16`,导出
"主体 FP16、I/O 保持 FP32"的图(30 行)。

**C++ 工具**:`tools/trt_cpp/`,新增 `--strongly-typed`、`--device`、`--debug-nan`、
`--debug-nan-all`、`--debug-nan-filter`、`--list-tensors`、`--nan-dump`。

---

## 附录二:这些改动会影响训练吗?需要重新训练吗?

**结论:不影响训练,已训练好的模型不需要重新训练。** 下面是实测证据,不是推断。

### 验证方法

`engine/solver/det_engine.py` 里有两条训练路径:

```python
if scaler is not None:                                   # AMP 路径
    with torch.autocast(device_type=str(device), cache_enabled=True):
        outputs = model(samples, targets=targets)
    with torch.autocast(device_type=str(device), enabled=False):
        loss_dict = criterion(outputs, targets, **metas)
else:                                                    # 纯 FP32 路径
    outputs = model(samples, targets=targets)
```

而 `train_deimv2_dinov3.sh` / `train_deim_dfine.sh` 里是:

```bash
USE_AMP="${USE_AMP:-1}"        # ← 默认就开 AMP
```

**所以训练默认走的是 FP16 autocast 路径**,这一点很容易想当然地搞错。
验证脚本 `z_trt_build/acc/dtype_check.py`:同一份 ckpt、同样 3 张图,
分别用 FP32 和 AMP 两种模式跑前向,和改动前逐位对比。

### 结果

| 检查                              | L                                         | X                          |
| --------------------------------- | ----------------------------------------- | -------------------------- |
| FP32 前向:改动前 vs 后            | **逐位相同**(max\|d\| = 0)          | **逐位相同**         |
| **AMP 前向**:改动前 vs 后   | **逐位相同**                        | **逐位相同**         |
| `state_dict()` 键集合           | **完全一致**(463 个)                | **完全一致**(551 个) |
| `load_state_dict` 未匹配/多余键 | 2 / 0(和改动前一样,是 anchors/valid_mask) | 2 / 0                      |

`ModelEMA` 是通过 `state_dict()` 双向同步的,键集合一致 → EMA 行为也不变。
`pos_embed` 现在是 `persistent=False` 的 buffer,**不在 `state_dict` 里**(实测 0 个匹配),
和改动前作为普通属性时的效果完全一样。

**所以:已有的 `.pth` 照常加载,前向逐位一致,不需要重新训练,继续训练也不会因此改变结果。**

### 一个真实教训:"FP32 下是空操作"不能靠推理

第一版改动里,`engine/deim/utils.py` 的采样网格 dtype 对齐**真的改变了 AMP 训练的前向结果**。
逐个文件隔离测试(每次只改一个文件,跑 AMP 前向和改动前逐位对比):

| 只改这个文件                            | AMP 下逐位相同                   |
| --------------------------------------- | -------------------------------- |
| `backbone/dinov3/layers/attention.py` | ✅ True                          |
| **`deim/utils.py`(采样网格)**   | ❌**False,max\|d\|=1.115** |
| `deim/dfine_decoder.py`               | ✅ True                          |
| `deim/hybrid_encoder.py`              | ✅ True                          |
| `deim/deim_decoder.py`                | ✅ True                          |

**原因**:autocast 本来就会把 `grid_sample` 提升到 FP32 执行;而那段代码看到
"网格是 FP32、value 是 FP16"就**把网格降成了 FP16**,反而损失精度。

**教训**:我一开始判断注意力那处改动在 AMP 下也无害(理由:"autocast 反正会把 SDPA
提升到 FP32")—— 这个判断**恰好是对的**;但同一批改动里另一处却实实在在地改变了结果。
**没有逐位对比的话,这个问题会一路带进训练里。**

### 怎么选守卫:按改动性质分三类

| 改动性质                       | 判据                                                 | 例子                                          |
| ------------------------------ | ---------------------------------------------------- | --------------------------------------------- |
| **构造期**确定           | **无条件改**                                   | `pos_embed` buffer、`ANCHOR_INVALID_FILL` |
| forward 里、**硬报错**   | `is_in_onnx_export() or not is_autocast_enabled()` | 采样网格、`Integral.project`                |
| forward 里、**精度问题** | **无条件改**(前提:实测对训练中性)              | 注意力 FP32                                   |

**① 构造期的代码,守卫永远不会生效。**

`_generate_anchors()` 和 `_reset_parameters()` 都是在 `__init__` 里被调用的,
而模型在 `torch.onnx.export()` **之前**就构造好了 —— 那时 `is_in_onnx_export()` 必然是 `False`:

```python
if torch.onnx.is_in_onnx_export():
    self.register_buffer('pos_embed1', pos_embed, persistent=False)   # 导出时
else:
    setattr(self, 'pos_embed1', pos_embed)                            # ← 实际总会走这里
```

**这等于把修复取消了** —— 导出时 `pos_embed` 还是普通属性、还是 FP32,
encoder 里 `src + pos_embed` 把激活提升回 FP32,直接报
`expected mat1 and mat2 to have the same dtype, but got: float != c10::Half`。
`anchors` 同理:buffer 里存回 `inf`,导出图照样带 inf 常量。

→ 这类改动**只能无条件改**,用别的方式证明中性(state_dict 键集合 + 逐位对比)。

**② 硬报错类的判据为什么是"或"。**

"半精度模型被前向"有两个场景,而 `is_in_onnx_export()` 只覆盖第一个:

- 导出时的 trace → **为真**
- **导出脚本在 trace 之前跑的那次 eager 前向**(`export_onnx_v2.py` 用它推 `n_queries`)→ **为假**
- (以及直接在 PyTorch 里跑半精度模型 → 为假)

所以写成 `is_in_onnx_export() or not is_autocast_enabled()`:
前者覆盖 trace,后者覆盖后两种,同时把 AMP 训练排除掉 ——
而在 AMP 下正需要排除(见上面那个真实教训)。

> 探针实测的四种场景(`value`/`grid` 的 dtype):
>
> | 场景                                  | value | grid | 是否匹配 | autocast |
> | ------------------------------------- | ----- | ---- | -------- | -------- |
> | FP32 推理                             | FP32  | FP32 | ✅       | False    |
> | **FP32 + autocast(= AMP 训练)** | FP16  | FP32 | ❌       | True     |
> | FP16`.half()` 无 autocast           | FP16  | FP16 | ✅       | False    |
>
> 规律:**autocast 开着就不用插手**(`grid_sample` 会被自动提升);关着就必须显式对齐。

**③ 精度类的改动,加守卫反而不够。**

注意力一开始用的是 `is_in_onnx_export()`,但那只覆盖导出那一次 ——
**导出前的 eager 前向、以及直接在 PyTorch 里跑半精度模型都保护不到**,L 照样 NaN。
改成无条件后覆盖面才完整,而且实测对训练逐位中性:

- 纯 FP32:`q.float()` 对 float32 张量是空操作;
- AMP:autocast 本来就会把 SDPA 提升到 FP32 执行,显式 `.float()` 只是把这个提升写出来。

**判断标准**:守卫只能用在 **forward 里"当前是不是导出"的分支**上;
构造期生成、或者纯 dtype 幂等的改动,要么无条件改(用 state_dict / 逐位对比证明中性),
要么根本不需要分支。

**顺带**:`ANCHOR_INVALID_FILL` 定义在 `engine/deim/utils.py` 由各解码器共用
(`DEIMTransformer` / `DFINETransformer`),避免同一个 `100.0` 在多个文件里各写一份、改漏。
`RTDETRTransformerv2`(`rtdetrv2_decoder.py`)目前**没改** —— 当前模型不走这条路径。

### 附:PyTorch 里直接跑 FP16 推理

把上面几处改到位之后,**半精度模型在 PyTorch 里也能直接前向**(不只是导出):

```python
model = <模型>.half().eval().cuda()
out = model(feed.half())        # 不需要 autocast
```

实测(L/X,3 张测试图,0 个 NaN):

|   | 检测框 | 最低 IoU | 最大框差 | 最大分数差 |
| - | ------ | -------- | -------- | ---------- |
| L | 33/33  | 0.9512   | 2.107 px | 4.1e-03    |
| X | 50/50  | 0.9909   | 0.392 px | 8.6e-03    |

这是 attention 那处改成**无条件**之后的收益 —— 只写 `is_in_onnx_export()` 的话,
这条路径保护不到,L 照样出 NaN。

### 重要发现:TensorRT 构建本身不可复现

对比"加守卫前"和"加守卫后"的 FP32 输出时,出现了 `max|d| = 2.87e-04`,**不是逐位相同**。
但按代码逻辑这两者应该完全相同 —— FP32 导出时 `is_in_onnx_export()` 为真,
新旧代码执行的是同样三行(`.float()` → SDPA → `.to(v.dtype)`)。

于是做了个对照实验:**同一个 ONNX,独立构建两次 engine**:

```
逐位相同 = False   max|d| = 1.612e-04
```

**量级完全一样 —— 所以那 2.87e-04 是 TensorRT 构建噪声,不是代码改动。**

差异分布(这个非常重要,决定了它是否要紧):

| 部分                                 | 两次构建的 max\|d\|                           |
| ------------------------------------ | --------------------------------------------- |
| **框坐标**(前 4 列,归一化 0~1) | **5.96e-07** ← 约等于 float32 精度极限 |
| logits(后 81 列)                     | 1.61e-04 ← 噪声全在这里                      |
| sigmoid 后分数                       | 2.27e-06                                      |
| 检测框                               | 11 vs 11,配对后框差 ≈ 0.0005 px              |

**结论**:

- TensorRT 在构建时会枚举 kernel 并实测计时(tactic 选择),**同一份 ONNX 两次构建
  得到的 engine 不保证逐位一致**;
- 噪声体现在 logits 上(~1.6e-4),但**框坐标稳定在 float32 精度极限**(~6e-7);
- 对检测结果没有可测量的影响。

**这条也解释了一个方法论问题**:用"逐位相同"作为判据时,TRT 侧的输出**天然做不到**,
必须区分开"框架/代码的确定性"和"TRT 构建的确定性"。
所以本文里 FP32 那段用 **IoU 1.0000 + 框差 0.000 px** 作为判据(而不是逐位相同),
而训练侧的前向对比才是真正的逐位相同(因为那是纯 PyTorch,没有 TRT 参与)。

**第二条经验**:判断"这个修复什么时候该生效"时,要按**改动的性质**分情况,
而不是套一个统一的开关:

| 改动性质             | 判据                                                 |
| -------------------- | ---------------------------------------------------- |
| 构造期确定           | 无条件改(守卫不可能生效)                             |
| forward 里、硬报错   | `is_in_onnx_export() or not is_autocast_enabled()` |
| forward 里、精度问题 | 无条件改(前提:实测对训练中性)                        |

统一的开关看起来"最保守",实际上两头都会漏:
纯 `is_in_onnx_export()` 会漏掉导出前的 eager 前向,
而无脑加守卫又会在构造期那两处**把修复取消掉**。
