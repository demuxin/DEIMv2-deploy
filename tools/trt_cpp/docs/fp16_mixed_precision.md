# FP16 精度问题:调查记录与结论

> 本文是 FP16 问题的**完整调查记录**。部署工具本身的实现、以及面试问答,
> 见 [tensorrt_deployment_summary.md](tensorrt_deployment_summary.md)。
>
> 所有数字来自本机实测(RTX 2070 / TensorRT 10.12.0.36 / CUDA 12.8)。
>
> 覆盖两个场景:
>
> - **§1~§6**:416×768 的模型(`outputs/deimv2_dinov3_{l,x}_personcaraninal/best_stg2.pth`)——
>   根因定位与修复过程
> - **§7**:1088×1920 的模型(`z_onnx/...-x-d1-...`)—— 同一根因在高分辨率下的表现,
>   外加显存和加速比两个新问题
>
> 两处的结论一致:**要用 FP16 就必须用 `--fp16` 导出的混合精度图 + `--strongly-typed`**。

---

## 0. 结论速览

**两个模型现在都能用 FP16 了。** 根因是 **ViT 注意力的 logits 超出 FP16 上限**,
把注意力放回 FP32 计算即可。改动对训练**逐位无影响**。

|                                | L                                                           | X                                         |
| ------------------------------ | ----------------------------------------------------------- | ----------------------------------------- |
| 根因指标`\|q·k\|·scale` 最大 | **2,518,337**(超 FP16 上限 38 倍)                     | 67,110(擦边 1.02 倍)                      |
| 修复前                         | 输出**全 NaN**,0 个框                                 | 能用,但框差最大 5.8 px                    |
| 修复后 FP16 vs FP32            | 33/33 配对,最低 IoU**0.9519**,框差 **0.820 px** | 50/50,**0.9872**,**0.591 px** |
| 速度                           | 39.21 →**24.81 ms**(1.58×)                          | 32.06 →**16.05 ms**(2.00×)        |

**高分辨率(1088×1920)补充结论** —— 见 §7:

- 同一根因更容易触发:FP32 图上加 `--fp16` 会**全 NaN**;必须用 `fp16s` 图 + `--strongly-typed`
- 强类型 FP16 vs FP32:3 张图 **38/39 配对,最低 IoU 0.9839,最大框差 0.347 px**
- **显存**:注意力矩阵在 FP32 下要 1.6 GB,默认 workspace 2048 MB 不够,需要 `--workspace 5120`
- **加速比缩水**:2.00×(416×768)→ **1.19×**(1088×1920),因为 FP32 注意力成了瓶颈

**三条要点**

1. **根因不在 TensorRT 警告说的地方。** 警告指向 layernorm/RMSNorm,
   但把它钉回 FP32 只能改善到 5.7 px —— 真正的瓶颈是注意力。
2. **`q·k` 是逐通道累加,量级远大于 q/k 本身**,这是 ViT 转 FP16 的通用陷阱。
3. **不是所有修复都能用 `torch.onnx.is_in_onnx_export()` 守卫**,要看改动的性质
   (见 §3.2)。

---

## 1. 现象:纯 FP16 精度崩溃

弱类型 FP16(`--fp16`,即 `BuilderFlag::kFP16`)构建:

|           | 耗时    | 配对数 | 最低 IoU | 最大框差          |
| --------- | ------- | ------ | -------- | ----------------- |
| FP32 基准 | 31.8 ms | 21/21  | —       | —                |
| 纯 FP16   | 9.1 ms  | 18/21  | 0.583    | **34.3 px** |

3.5 倍加速很香,但精度不可接受。TensorRT 构建时自己就警告:

```text
[TRT][W] Detected layernorm nodes in FP16.
[TRT][W] Running layernorm after self-attention with FP16 Reduce or Pow may cause overflow.
Forcing Reduce or Pow Layers in FP32 precision, or exporting the model to use
INormalizationLayer (available with ONNX opset >= 17) can help preserving accuracy.
```

更严重的是 **L 模型在 FP16 下输出全是 NaN**(25500/25500 个元素),检测框 0 个。

---

## 2. 排查过程

### 2.1 第一次弯路:跟着 TensorRT 的警告走

顺着警告定位到 decoder:ONNX **opset 17 没有 RMSNorm 算子**,所以
`engine/deim/deim_decoder.py` 的 `norm1/norm2/norm3` 被拆成 126 个
`Pow → ReduceMean → +eps → Sqrt → Div` 节点。`x²` 在 FP16 里超过 256 就会溢出(上限 65504)。

于是按 TensorRT 的建议,把名字含 `norm` 的层钉回 FP32(工具里的 `--fp16-fp32-norm`):

| 方案                  | 耗时   | 配对数 | 最低 IoU | 最大框差         |
| --------------------- | ------ | ------ | -------- | ---------------- |
| 纯 FP16               | 9.1 ms | 18/21  | 0.583    | 34.3 px          |
| + 所有 norm 层钉 FP32 | 9.8 ms | 20/21  | 0.835    | **5.7 px** |

**有改善但不够**,而且 L 依然全 NaN。所以 decoder 的 RMSNorm 不是根因。

### 2.2 第二次弯路:用错了"等价物"

在拿到 checkpoint 之前,我用
`onnxruntime.transformers.float16.convert_float_to_float16` 手工把现成 ONNX
转成混合精度图当作 `.half()` 导出的等价物,结果**不管怎么调 blocklist 都是 34 px**,
于是得出"导出侧这条路走不通"的错误结论。

**后来发现那个等价物是错的**:盲转会把 DINOv3 RoPE 的 `Cos`/`Sin` 也降成 FP16,
位置编码失真,这时钉什么都不管用。

**而真正的 `.half()` 导出不会**,因为 DINOv3 作者把 dtype 写死在模块里:

```python
self.dtype = dtype                    # Don't rely on self.periods.dtype  ← 注释就是答案
self.register_buffer("periods", torch.empty(..., dtype=dtype))
...
def forward(self, *, H, W):
    dtype = self.dtype                # Python 属性,.half() 动不了它
    dd = {"device": device, "dtype": dtype}   # 用它构造中间张量 → 强制 FP32
```

attention 侧也有配套设计:

```python
# apply_rope:All operations will use the dtype of rope,
#            the output is cast back to the dtype of q and k
```

实测 dtype 对照(shape inference 后查真实输入 dtype):

|                         | `convert_float_to_float16` 手工转 | `.half()` 导出  |
| ----------------------- | ----------------------------------- | ----------------- |
| `Cos` / `Sin`(RoPE) | **FP16** ← 位置编码失真      | **FP32** ✓ |
| `LayerNormalization`  | FP16                                | FP16              |

> **教训:用"近似的等价物"去验证方案,比不验证更危险 —— 会得出方向性错误的结论。**

### 2.3 定位:用 IDebugListener 扫到 backbone 第一个 block

工具里加了 `--debug-nan`,底层是 TensorRT 的 debug tensor 机制:

- 构建时:`network->markUnfusedTensorsAsDebugTensors()`(不阻止融合)
  或 `markDebug()`(逐个标记,会阻止融合)
- 运行时:`context->setUnfusedTensorsDebugState(true)` + `context->setDebugListener(listener)`
- 回调 `processDebugTensor(addr, location, type, shape, name, stream)` 把张量拷回 host 检查

扫描结果(L FP16,361 个未融合张量):

```text
/inner/model/backbone/sta/stem/stem.2/Mul_1_output_0    0   <- stem 输出正常
/inner/model/backbone/blocks_0/mlp/fc2/Add_output_0     1   <- 第一个 NaN
/inner/model/backbone/blocks_1/mlp/fc2/Add_output_0     1
...                                                         346/361 全是 NaN
```

**输入是好的,NaN 在 backbone 的第一个 transformer block 内部产生。**

**两个 API 限制(值得记)**:

- `markDebug` 和 `--strongly-typed` 在 TRT 10.12 下**冲突**,TRT 内部断言
  `[genericPwFusion.cpp] Assertion !stronglyTyped failed`;细查只能在弱类型下做
- 这个模型的张量名有重名(带 `.1` 后缀),`markDebug` 会再触发一个
  `shapeContext.cpp::renameDebugTensor` 断言失败

所以在弱类型 FP16 下排查、再回到强类型验证,是这次实际走的路径。

### 2.4 根因确认:直接量激活量级

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
`dinov3_vits16`(`ffn_ratio=4, ffn_layer=mlp`)vs
`dinov3_vits16plus`(`ffn_ratio=6, ffn_layer=swiglu`),激活量级自然不一样。
X 只是勉强擦边,所以能用。

### 2.5 顺手清掉的隐患:`anchors` 常量里的 Inf

`_generate_anchors()` 把 padding 位置的锚点填成 `torch.inf`,这个 inf 会被
**导出成 ONNX 常量**(L 有 1648 个、X 有 1168 个 Inf)。而 RMSNorm 的
`x * rsqrt(mean(x²)+eps)` 遇到 Inf 会得到 `Inf * 0 = NaN` —— 看着非常像元凶。

改成有限常量后,四个导出图里**非有限常量从 1648/1168 个降到 0 个**:

```python
# engine/deim/utils.py,各解码器共用
ANCHOR_INVALID_FILL = 100.0
```

依据:`sigmoid(100)` 在 FP32/FP16 下都**精确等于** 1.0,与 `sigmoid(inf)` 行为完全一致;
下游只有 `sigmoid` 和 `inverse_sigmoid`(后者按 `eps=1e-5` 截断);
训练时两者都落在饱和区,梯度同为 0。

**但 L 的 FP16 依然全 NaN —— 所以 Inf 不是根因**,只是把图里一个明确的隐患清掉了。

---

## 3. 修复

### 3.1 七处改动总览

最终提交(commit `6a1f0be`)共 **5 个文件、7 处改动**:

| 文件                                    | 改动                          | 形式             | 为什么                      |
| --------------------------------------- | ----------------------------- | ---------------- | --------------------------- |
| `backbone/dinov3/layers/attention.py` | 注意力放 FP32 算              | **无条件** | 根因修复;实测对训练逐位中性 |
| `deim/utils.py`                       | 定义`ANCHOR_INVALID_FILL`   | 构造期           | 见 §2.5                    |
| `deim/utils.py`                       | 采样网格对齐`value.dtype`   | **带守卫** | `grid_sample` 硬报错      |
| `deim/deim_decoder.py`                | 用`ANCHOR_INVALID_FILL`     | 构造期           | 同 §2.5                    |
| `deim/dfine_decoder.py`               | 用`ANCHOR_INVALID_FILL`     | 构造期           | 同 §2.5                    |
| `deim/dfine_decoder.py`               | `Integral.project` 补 dtype | **带守卫** | `F.linear` 硬报错         |
| `deim/hybrid_encoder.py`              | `pos_embed` 注册成 buffer   | 构造期           | `.half()` 带不走普通属性  |

**核心修复**(注意力):

```python
def scaled_dot_product_attention(q, k, v, **kwargs):
    return F.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), **kwargs).to(v.dtype)
```

思路和 `apply_rope` 的 dtype 处理一致。

### 3.2 `torch.onnx.is_in_onnx_export()` 的正确用法(以及三个陷阱)

为了不让导出相关的改动影响训练,自然的想法是用 `is_in_onnx_export()` 把两条路径分开。
**但它不是万能的**,这次踩了三个坑:

**陷阱 ① 构造期的代码,守卫永远不会生效。**
`_generate_anchors()` 和 `_reset_parameters()` 都是在 `__init__` 里被调用的,
而模型在 `torch.onnx.export()` **之前**就构造好了 —— 那时 `is_in_onnx_export()` 必然是 `False`。
所以写成

```python
if torch.onnx.is_in_onnx_export():
    self.register_buffer('pos_embed1', pos_embed, persistent=False)   # 导出时
else:
    setattr(self, 'pos_embed1', pos_embed)                            # 实际总会走这里
```

**等于把修复取消了**,导出时 `pos_embed` 还是普通属性、还是 FP32,
encoder 里 `src + pos_embed` 把激活提升回 FP32,直接报:

```text
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::Half
```

→ **构造期的改动只能无条件改**,用别的方式证明中性(见 §4.4)。

**陷阱 ② 导出脚本在 trace 之前还跑一次 eager 前向。**

```python
# tools/deployment/export_onnx_v2.py
with torch.no_grad():
    out = model(data)          # ← 唯一用途:拿 out.shape[1] 打印 query 数
n_queries = out.shape[1]
...
torch.onnx.export(model, (data,), output_file, ...)   # ← 这里 is_in_onnx_export() 才为真
```

这次前向跑在 export 之外,`is_in_onnx_export()` 为 `False`,
但 `--fp16` 时模型已经是 `.half()` 过的 —— 所以**硬报错类的改动只判断它是不够的**。

**陷阱 ③ 精度类的改动可能根本不该加守卫。**
注意力一开始也用了 `is_in_onnx_export()`,但那只覆盖导出那一次;
**导出前的 eager 前向、以及直接在 PyTorch 里跑半精度模型都保护不到**,L 照样 NaN。
改成无条件后覆盖面才完整,而且实测对训练逐位中性(§4.4)。

**最终判据(按改动性质分三类)**

| 改动性质                     | 判据                                                 | 例子                                          |
| ---------------------------- | ---------------------------------------------------- | --------------------------------------------- |
| 构造期确定                   | **无条件改**                                   | `pos_embed` buffer、`ANCHOR_INVALID_FILL` |
| forward 里、**硬报错** | `is_in_onnx_export() or not is_autocast_enabled()` | 采样网格、`Integral.project`                |
| forward 里、精度问题         | **无条件改**(如果实测中性)                     | 注意力 FP32                                   |

第二类的判据为什么是"或":前者覆盖导出 trace,后者覆盖导出前的 eager 前向和
PyTorch 半精度前向,同时把 AMP 训练排除掉(那里交给 autocast 自己提升精度)。

### 3.3 三个 dtype 硬编码的坑(导出 FP16 时逐个暴露)

第一版 `--fp16` 导出是**报错**的,每修一个就往前走一层再报下一个:

| 现象                                                                  | 原因                                                                                                                              | 修法                                           |
| --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| `expected m1 and m2 to have the same dtype, but got: float != Half` | `pos_embed` 用 `setattr` 存成**普通属性**(旁边 `register_buffer` 被注释掉了),`.half()`/`.to(device)` 都不会带上它 | 改成`register_buffer(..., persistent=False)` |
| `expected scalar type Half but found Float`(`grid_sample`)        | 采样网格是`torch.tensor(value_spatial_shapes)` 现算的,不受 `.half()` 影响留在 FP32,而 `value` 已是 FP16                     | 网格对齐到`value.dtype`                      |
| 同上,在`Integral.forward`                                           | `project.to(x.device)` **只转 device 不转 dtype**,FP32 的 `project` 和 FP16 的 `x` 做 `F.linear`                    | 补上`dtype=x.dtype`                          |

另外 **FP16 追踪必须在 GPU 上做** —— CPU 没有 `grid_sample` 的 FP16 实现。

---

## 4. 验证

### 4.1 三段一致性:PyTorch → ORT FP32 → TRT FP32

3 张测试图(原图 500×375、缩小 350×262、放大 700×525),各环节共用同一套预处理
(与 `onnx_inf_v2.py` 一致的 PIL 拉伸 + ImageNet 归一化)和后处理(阈值 0.25、全类别 NMS 0.9)。

**L 模型**(共 33 个检测框)

| 对照                     | sigmoid 后 max\|d\| | 配对  | 最低 IoU         | 最大框差           | 最大分数差 |
| ------------------------ | ------------------- | ----- | ---------------- | ------------------ | ---------- |
| ORT FP32 vs PyTorch FP32 | 9.2e-02             | 33/33 | **1.0000** | **0.000 px** | 4.7e-06    |
| TRT FP32 vs PyTorch FP32 | 9.7e-06             | 33/33 | **1.0000** | **0.000 px** | 3.6e-06    |
| TRT FP16 vs PyTorch FP32 | 3.1e-01             | 33/33 | 0.9519           | **0.820 px** | 6.0e-03    |

**X 模型**(共 50 个检测框)

| 对照                     | sigmoid 后 max\|d\| | 配对  | 最低 IoU         | 最大框差           | 最大分数差 |
| ------------------------ | ------------------- | ----- | ---------------- | ------------------ | ---------- |
| ORT FP32 vs PyTorch FP32 | 4.0e-05             | 50/50 | **1.0000** | **0.000 px** | 7.1e-06    |
| TRT FP32 vs PyTorch FP32 | 1.5e-01             | 50/50 | **1.0000** | **0.000 px** | 2.7e-06    |
| TRT FP16 vs PyTorch FP32 | 4.6e-01             | 50/50 | 0.9872           | **0.591 px** | 1.8e-02    |

**要点**

- **PyTorch → ORT FP32 → TRT FP32 三段完全一致**:100% 配对、最低 IoU 1.0000、
  框坐标差 **0.000 px**,分数差 1e-6 量级。导出和两个运行时都没引入可测量的偏差。
- 表里 `sigmoid 后 max|d|` 那些 1e-1 量级的数字**都落在低分 query 上**(分数接近 0 时
  sigmoid 对 logits 极其敏感),对检测结果没有影响。**看精度要看配对后的框和分数。**
- 速度:**L 39.21 → 24.81 ms(1.58×)**;**X 32.06 → 16.05 ms(2.00×)**。

### 4.2 PyTorch 里直接跑 FP16 推理

把注意力改成无条件之后,**半精度模型在 PyTorch 里也能直接前向**(不只是导出):

```python
model = <模型>.half().eval().cuda()
out = model(feed.half())        # 不需要 autocast
```

实测(3 张测试图,**0 个 NaN**):

|   | 检测框 | 最低 IoU | 最大框差 | 最大分数差 |
| - | ------ | -------- | -------- | ---------- |
| L | 33/33  | 0.9512   | 2.107 px | 4.1e-03    |
| X | 50/50  | 0.9909   | 0.392 px | 8.6e-03    |

### 4.3 代价

加速比从纯 FP16 的 3.5× 降到 1.58×~2.00× —— ViT 里注意力占比不小。
但纯 FP16 是"34 px 偏差 + L 全 NaN",**没有可比性**。

如果只跑 X 且能接受 5.8 px 偏差,去掉注意力这处改动可以换回 3.6×;
但 X 的 logits 只超限 1.02 倍,属于刀尖上,不建议。

### 4.4 训练影响:逐位相同

**训练脚本默认开 AMP**(`USE_AMP="${USE_AMP:-1}"`),所以 FP32 **和** AMP 两种模式都要验。

| 检查                            | L                                        | X                  |
| ------------------------------- | ---------------------------------------- | ------------------ |
| FP32 前向:改动前 vs 后          | **逐位相同**(max\|d\| = 0.000e+00) | **逐位相同** |
| AMP 前向:改动前 vs 后           | **逐位相同**                       | **逐位相同** |
| `state_dict()` 键集合         | 完全一致(463)                            | 完全一致(551)      |
| `load_state_dict` 未匹配/多余 | 2 / 0(与改动前相同)                      | 2 / 0              |

`ModelEMA` 通过 `state_dict()` 双向同步 → 键集合一致则 EMA 行为不变。
`pos_embed` 是 `persistent=False` 的 buffer,**不在 `state_dict` 里**(实测 0 个匹配),
与改动前作为普通属性时效果一致。

**结论:已有 `.pth` 照常加载,前向逐位一致,不需要重新训练。**

> **一条经验**:"FP32 下是空操作"不能靠推理,得靠逐位实测。
> 我一开始判断注意力那处改动在 AMP 下也无害(理由:autocast 反正会把 SDPA 提升到 FP32)——
> 这个判断**恰好是对的**;但同一批改动里另一处却实实在在地改变了 AMP 结果
> (采样网格那次,`max|d| ≈ 1.1`),是逐位对比把它揪出来的。

### 4.5 一个重要发现:TensorRT 构建本身不可复现

对比"加守卫前/后"的 FP32 输出时出现 `max|d| = 2.87e-04`,**不是逐位相同**。
但按代码逻辑这两者应该完全相同。于是做了对照实验 ——
**同一个 ONNX,独立构建两次 engine**:

```text
逐位相同 = False   max|d| = 1.612e-04
```

量级一致 → **是 TensorRT 构建噪声**(kernel tactic 选择),不是代码改动。
差异分布(决定了它是否要紧):

| 部分                                 | 两次构建的 max\|d\|                           |
| ------------------------------------ | --------------------------------------------- |
| **框坐标**(前 4 列,归一化 0~1) | **5.96e-07** ← 约等于 float32 精度极限 |
| logits(后 81 列)                     | 1.61e-04 ← 噪声全在这里                      |
| sigmoid 后分数                       | 2.27e-06                                      |
| 检测框                               | 11 vs 11,框差 ≈ 0.0005 px                    |

**方法论意义**:用"逐位相同"做判据时,TRT 侧**天然做不到**,必须区分
"框架/代码的确定性"和"TRT 构建的确定性"。所以本文里 FP32 那段用
**IoU 1.0000 + 框差 0.000 px** 作判据,而训练侧才是真正的逐位相同(纯 PyTorch,无 TRT)。

---

## 5. 怎么用

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

**`--strongly-typed` 是必须的。** 弱类型网络 + `BuilderFlag::kFP16` 下,
TensorRT 有权把任意 FP32 层降成 FP16,图里的 dtype 分界会被整个忽略 ——
模型代码里原本的 `x.float()` 保护也就成了空操作。

**如果不方便重导模型**,工具里还有一条退路:直接在原 FP32 ONNX 上
`--fp16-fp32-norm`(把名字含 `norm` 的层钉回 FP32)。实测 L 20/21、5.7 px,
不如上面的方案,但不用重导。

---

## 6. 排查工具

`z_trt_build/` 下的一次性脚本,换模型可复用:

| 脚本                                   | 用途                                                               |
| -------------------------------------- | ------------------------------------------------------------------ |
| `opcount.py`                         | 纯 Python 数 ONNX 的算子类型(容器里没有`onnx` 包)                |
| `oppos.py`                           | 打印指定算子的节点名,定位是哪个子模块                              |
| `subgraph.py`                        | 打印某个作用域的子图(查 RMSNorm 被拆成什么)                        |
| `act_stats.py`                       | 量各模块激活量级 +**注意力 logits**(定位 FP16 溢出点最有用)  |
| `mk_fp16_onnx.py`                    | 手工转混合精度图(仅供粗筛,**不等价于 `.half()`**,见 §2.2) |
| `mk_fp16_probe.py`                   | 按名字前缀把某部分固定 FP32,二分定位                               |
| `acc/dtype_check.py`                 | 训练侧逐位回归(FP32 + AMP)+ PyTorch FP16 推理                      |
| `acc/{torch_ref,ort_ref,compare}.py` | 四段精度基线对比                                                   |

C++ 工具侧相关开关:

```bash
--strongly-typed                   # 强类型构建(FP16 必需)
--debug-nan                        # 逐张量找 NaN(标记未融合张量,不掉性能)
--debug-nan-filter "blocks.0/"     # 只标记匹配的张量(会阻止融合,但范围小)
--list-tensors "blocks.0"          # 只列张量名和 dtype,不构建(用于查命名)
--nan-dump nan.txt                 # 导出全部张量的 NaN 状态
```

---

## 7. 案例二:1088×1920 高分辨率模型

前面几节都是 416×768 的模型。后来又拿到一对 1088×1920 的模型(`-x-d1-`),
**同一个根因在高分辨率下更容易触发**,而且暴露了显存和加速比两个新问题。

### 7.1 两个图的差异

两个文件输入输出相同(`[1,3,1088,1920]` FP32 → `[1,300,85]` FP32),
但内部 dtype 完全不同:

|                           | `...-x-d1-fp16s-...onnx`(98 MB) | `...-x-d1-fp16-...onnx`(195 MB) |
| ------------------------- | --------------------------------- | --------------------------------- |
| initializer               | **516 FP16** + 1 FP32       | 517 FP32                          |
| Conv                      | 62 × FP16                        | 62 × FP32                        |
| MatMul                    | 164 FP16 +**24 FP32**       | 188 × FP32                       |
| **Softmax**         | **12 FP32** + 20 FP16       | 32 × FP32                        |
| LayerNormalization        | 29 × FP16                        | 29 × FP32                        |
| Pow / ReduceMean          | **FP32**                    | FP32                              |
| **Cos / Sin**(RoPE) | **FP32**                    | FP32                              |
| GridSample                | 18 × FP16                        | 18 × FP32                        |
| 节点总数                  | 5398                              | 5295                              |

`fp16s` 就是 §3 那套混合精度导出的产物(后缀 **s** = strong)。
里面 **12 个 FP32 Softmax + 24 个 FP32 MatMul** 正好是 **12 层 ViT 的注意力**,
Pow/ReduceMean(RMSNorm 归约)和 Cos/Sin(RoPE)也都在 FP32。

`fp16` 是**纯 FP32 图** —— 文件名里的 `fp16` 指的是"打算用 FP16 推理",不是图本身的精度。

### 7.2 四种推理方式的结果

测试图 500×375,阈值 0.25,全类别 NMS IoU 0.9:

| 变体        | ONNX 图   | 构建方式                     | 耗时     | 检测框          | NaN                   |
| ----------- | --------- | ---------------------------- | -------- | --------------- | --------------------- |
| **A** | `fp16s` | `--strongly-typed`         | 416.5 ms | **14** ✅ | 0                     |
| **B** | `fp16`  | `--fp16`(弱类型 `kFP16`) | 80.4 ms  | **0** ❌  | **25500/25500** |
| C           | `fp16`  | (FP32 基准)                  | 496.5 ms | 14              | 0                     |
| D           | `fp16`  | `--fp16-fp32-norm`         | 80.2 ms  | **0** ❌  | **25500/25500** |

**结论:`fp16s` + `--strongly-typed` 能用;在 FP32 图上加 `--fp16` 直接崩。**

> ⚠️ **B/D 的 80 ms 是假象** —— 它快是因为算出 NaN 后退化了,不是真的加速。

**A vs C(强类型 FP16 vs FP32 基准)**,3 张测试图:

| 测试图         | 检测框            | 配对         | 最低 IoU         | 最大框差           |
| -------------- | ----------------- | ------------ | ---------------- | ------------------ |
| 500×375       | 14 / 14           | 14/14        | 0.9906           | 0.206 px           |
| 350×262       | 10 / 10           | 10/10        | 0.9839           | 0.184 px           |
| 700×525       | 14 / 15           | 14/15        | 0.9887           | 0.347 px           |
| **合计** | **38 / 39** | **38** | **0.9839** | **0.347 px** |

**A vs B:14 个框 vs 0 个框,完全不同。**

### 7.3 为什么 B/D 会崩:TRT 的弱类型保护不覆盖注意力

B 和 C 用的是**同一个** `fp16` 图,唯一差别是构建方式。TensorRT 在 B 的构建日志里说:

```text
[TRT][W] Running layernorm after self-attention with FP16 Reduce or Pow may cause overflow.
Forcing Reduce or Pow Layers in FP32 precision...
```

**TRT 确实做了保护,但只覆盖 Reduce/Pow(RMSNorm 的归约)**,不管矩阵乘和 softmax 的 dtype。
而 `fp16` 图里注意力是 FP32(图层面),弱类型下 TRT 有权把它降成 FP16 ——
`q·k` 一溢出,`softmax(Inf)` 就是 NaN。

**变体 D 是关键证据**:把 126 个 norm 层全部钉回 FP32,**仍然全 NaN**。
说明问题不在 norm —— 和 §2.1 在小模型上的结论完全一致,只是高分辨率下更严重。

对照组一目了然:

|                        | `fp16s`(A 用)             | `fp16`(B/D 用)          |
| ---------------------- | --------------------------- | ------------------------- |
| **Softmax 输入** | **12 FP32** + 20 FP16 | 全 FP32(**图层面**) |
| **MatMul 输入**  | 164 FP16 +**24 FP32** | 全 FP32(**图层面**) |

A 的图里那 12 层注意力被**显式**保护成了 FP32;B/D 的图里没有这个保护, 只能指望 TRT 自己,而它不管这一层。

**所以:要用 FP16,必须"`--fp16` 导出 + `--strongly-typed`",不能在 FP32 图上直接 `--fp16`。**

### 7.4 分辨率放大带来的连锁反应

从 416×768 到 1088×1920,不只是"慢一点":

|                      | 416×768                | 1088×1920               | 倍数           |
| -------------------- | ----------------------- | ------------------------ | -------------- |
| 像素                 | 319,488                 | 2,088,960                | 6.5×          |
| ViT patch 数(16×16) | 26×48 =**1,248** | 68×120 =**8,160** | 6.5×          |
| 注意力矩阵元素(6 头) | 9.3 M                   | **400 M**          | **43×** |
| 该矩阵 FP32 大小     | 37 MB                   | **1.6 GB**         | 43×           |
| FP32 推理耗时        | 32.1 ms                 | 475.1 ms                 | **15×** |
| FP16 推理耗时        | 16.1 ms                 | 389.2 ms                 | 24×           |
| **加速比**     | **2.00×**        | **1.22×**         | —             |

> 1088×1920 那两列用**交替复测 3 轮的最小值**(`trtexec` GPU Compute Time);
> 单次顺序测量会把 fp16 测成偏慢 —— 详见下面第 ② 点。
> 用本文档主工具 `deimv2_trt_infer --benchmark`(墙钟含 H2D/D2H)测得的是 496.5 / 416.5 ms = 1.19×,与 1.22× 基本一致。

**两个后果:**

**① 显存告急。** 第一次三个构建**全部失败**:

```text
[TRT][W] UNSUPPORTED_STATE: Skipping tactic 0 due to insufficient memory
         on requested size of 3262730200 detected for tactic 0x0000000000000000.
[TRT][E] IBuilder::buildSerializedNetwork: Error Code 10: Internal Error
         (Could not find any implementation for node {ForeignNode[.../backbone/blocks.11/attn/...]}.)
```

某个 tactic 需要 **3.26 GB**,而默认 workspace 上限是 2048 MB。这个数字正好对上**FP32 注意力矩阵 1.6 GB + softmax 输出 1.6 GB ≈ 3.2 GB**。

加上 `--workspace 5120` 后三个变体都能构建(RTX 2070 只有 8 GB,已比较吃紧)。

**② 加速比被摊平 —— 因为注意力变成了内存带宽瓶颈。**

这一条我做了逐层 profiling 才敢下结论(`trtexec --dumpProfile`,脚本见 §7.5)。

**端到端**(交替复测 3 轮取最小值,消除降频影响 —— 单次顺序测量会把 fp16 测成比 fp32 还慢,那是 GPU 热了之后的假象):

|                       | fp32     | fp16(强类型) | 加速             | 显卡      |
| --------------------- | -------- | ------------ | ---------------- | --------- |
| GPU Compute Time(min) | 475.1 ms | 389.2 ms     | **1.22×** | RTX 2070  |
| GPU Compute Time(min) | 180.3 ms | 161.3 ms    | 1.12x            | RTX 3090  |

**逐层拆解**(按 `attn` / `MaxrSubExpSumDivMul` 归类为注意力):

|                    | fp32                    | fp16                    | 变化                      |
| ------------------ | ----------------------- | ----------------------- | ------------------------- |
| **注意力**   | 421.9 ms(**73%**) | **375.6 ms(90%)** | **−11%**(几乎没变) |
| **非注意力** | 153.5 ms(27%)           | **43.1 ms(10%)**  | **−72%(3.6×)**    |

**FP16 在非注意力部分确实生效了(3.6×),但代价是它只占 10% 的耗时。注意力占了 90%,却几乎享受不到 FP16 的收益。**

**为什么注意力拿不到收益:它是内存带宽瓶颈,不是算力瓶颈。**

单个 ViT softmax 层(`[1,6,8160,8160]`,约 4 亿个元素):

|                             | 值                                |
| --------------------------- | --------------------------------- |
| 矩阵大小(FP32)              | 1.598 GB                          |
| 两趟 softmax 的读写量       | 2 × (读 + 写) =**6.39 GB** |
| RTX 2070 带宽               | 448 GB/s                          |
| **理论耗时**          | **14.26 ms**                |
| **实测(fp32 engine)** | **14.5 ms**                 |
| **实测(fp16 engine)** | **13.8 ms**                 |

softmax 操作 `[1,6,8160,8160]` 矩阵,FP32 下 1.598 GB,两趟读写 = 6.39 GB,除以 RTX 2070 带宽 448 GB/s = **14.26 ms** ——  **实测 13.8 ms,达到了理论带宽的 97%** 。这一层已经跑在内存带宽上限,任何算力提升都无效。

8160 个 token 时注意力已经变成内存带宽瓶颈,FP16 的优势在算力(2× FLOPS),对带宽瓶颈毫无帮助;而强制 FP32 反而让字节数翻倍。**这 12 层 softmax 完全是内存带宽瓶颈,算力提升对它们毫无意义。**

这就是全部原因:

- FP16 的优势在**算力**(2× FLOPS),对带宽瓶颈没有帮助;
- 强制 FP32 还让字节数翻倍(0.8 GB → 1.6 GB),进一步加重带宽压力;
- 12 层这样的 softmax × 14 ms = **~170 ms**,纯粹是搬运数据的开销,
  在 fp32 和 fp16 两个 engine 里几乎一模一样。

**旁证**:变体 B(FP32 图 + 弱类型 FP16,整网 FP16,结果是 NaN 不可用)只要 **80.4 ms** —— 说明 FP32 注意力这一项大约吃掉了 **300+ ms**。

**推论**:高分辨率场景下 "整层注意力走 FP32" 代价极大。可行的优化方向:

1. **只对真正超限的层用 FP32** —— 先用 `act_stats.py` 逐层量 logits,多数层可能远低于 65504;
2. **从分辨率入手** —— 1080p 输入配 patch 16 得到 8160 token,改用 patch 32 可让token 数降 4 倍、注意力算力和矩阵大小降 **16 倍**,这是数量级上的改善;
3. **让 softmax 用 FP16**(如果数值允许)—— 矩阵减半,理论耗时 14.3 → 7.1 ms。

### 7.5 复现命令

```bash
Z=z_onnx
O=$Z/PersonCarAnimal_od-v5-2-x-d1-fp16s-1088-1920_20261008.onnx   # 强类型 FP16 图
P=$Z/PersonCarAnimal_od-v5-2-x-d1-fp16-1088-1920_20261008.onnx    # FP32 图
B=./tools/trt_cpp/build/deimv2_trt_infer

# A:推荐做法 —— 混合精度图 + 强类型
$B --onnx $O --strongly-typed --workspace 5120 --image x.jpg --thrh 0.25

# B:不推荐 —— FP32 图 + 弱类型 FP16(会全 NaN)
$B --onnx $P --fp16 --workspace 5120 --image x.jpg --thrh 0.25

# C:FP32 基准
$B --onnx $P --workspace 5120 --image x.jpg --thrh 0.25
```

对比脚本:`z_trt_build/cmp1088/compare3.py`。

逐层 profiling(注意**必须交替复测** —— 顺序跑会把第二个测成偏慢):

```bash
TRT=/usr/local/tensorrt/TensorRT-10.12.0.36
$TRT/bin/trtexec --onnx=$O --stronglyTyped --memPoolSize=workspace:5120 \
    --saveEngine=fp16.engine                    # 建一次,后续用 --loadEngine 复测
$TRT/bin/trtexec --loadEngine=fp16.engine --dumpProfile --separateProfileRun \
    --iterations=20 --warmUp=500
```

解析脚本见 `z_trt_build/prof/parse.py`(按类别汇总逐层耗时)。

> 这两个模型的 ONNX 是**静态 shape**(batch 固定 1),`trtexec` 不能加 `--shapes`
> (会报 "Static model does not take explicit shapes")。之前的模型是动态 batch,注意区别。

---

## 8. 遗留

- **`RTDETRTransformerv2`(`engine/deim/rtdetrv2_decoder.py`)的 `_generate_anchors()`
  仍用 `torch.inf`。** 它对应 `configs/base/rtdetrv2_r50vd.yml`,当前两个模型不走这条路径,
  所以没改。若以后要用 `deim_rtdetrv2` 系列导出 FP16,记得同步。
- **L 的注意力 logits 达到 2.5e6 说明该模型数值本身比较激进。**
  FP16 现在是靠"注意力走 FP32"绕过去的,不是模型变稳了 ——
  换更激进的训练配置时要重新量一遍(用 `act_stats.py`)。
