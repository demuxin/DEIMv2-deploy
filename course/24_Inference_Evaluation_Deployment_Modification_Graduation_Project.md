# DEIMv2 源码课程 · 第 24 课

# 推理、评估、部署、改模型与最终毕业项目

> **本课定位**：这是 24 课主课程的收官课。前 23
> 课已经完成视觉基础、DETR、D-FINE、DEIM Dense O2O、MAL、DEIMv2 训练
> Recipe、DINOv3、STA、DEIMTransformer
> 和完整训练系统。本课补齐最后的工程闭环：
>
> **checkpoint → PyTorch 推理 → PostProcessor → COCO 评估 → FLOPs /
> Params → latency / FPS → ONNX → ONNX Runtime → TensorRT → FP16 →
> 数值一致性验证 → 自定义数据集 → 安全修改模型 → ablation →
> 最终毕业项目。**
>
> 本课以官方 DEIMv2 仓库当前 `main` 分支为参考。仓库会持续更新，所以 CLI
> 参数、工具源码和部署兼容性应以你本地 commit 为最高依据。官方当前提供
> PyTorch / ONNX Runtime / TensorRT 推理、ONNX 导出、TensorRT
> 构建、FLOPs/MACs/Params benchmark、TensorRT latency benchmark 和
> FiftyOne 可视化工具；官方还特别提示 TensorRT FP16 推荐使用 **≥
> 10.6**。

---

## 0. 本课最终要回答什么？

学完以后，你应该能够独立回答：

```text
checkpoint 里到底是什么？
为什么推理不需要 matcher 和 criterion？
为什么 model.forward() 的输出还不是最终检测结果？
PostProcessor 怎样恢复像素坐标？
top-k 在哪里发生？
DETR/DEIM 为什么通常不依赖 NMS？

AP / AP50 / AP75 / APS / APM / APL 各说明什么？
Params / MACs / FLOPs / latency / FPS 为什么不能混为一谈？

ONNX 到底导出了什么？
为什么 export 成功不等于部署正确？
TensorRT engine 为什么与环境相关？
为什么 FP16 必须重新做精度验证？

怎样把 COCO 80 类改成自己的 N 类？
改 num_queries、decoder layers、hidden_dim、STA indexes 会影响什么？
怎样设计可信的 ablation？
怎样完成一个真正可复现的毕业项目？
```

本课的目标不是"会执行几个工具命令"，而是形成：

```text
训练
↓
评估
↓
错误分析
↓
结构修改
↓
重新训练
↓
部署
↓
数值验证
↓
速度验证
↓
结论
```

的完整研究与工程闭环。

---

# 第一部分：训练结束后到底得到了什么？

## 1. 输出目录中的文件职责不同

训练后常见：

```text
last.pth
checkpointXXXX.pth
best_stg1.pth
best_stg2.pth
log.txt
eval/
TensorBoard logs
```

其中：

```text
last.pth
→ 最后训练状态

best_stg1 / best_stg2
→ 对应训练阶段的最佳验证模型

log.txt
→ train/test metrics

eval/
→ evaluator 相关结果
```

部署时不要机械选择 `last.pth`，而应明确：

```text
哪个 checkpoint？
对应哪个 config？
对应哪个 commit？
validation AP 是多少？
```

---

## 2. Training checkpoint 不等于部署模型

完整 checkpoint 可能包含：

```text
model
EMA
optimizer
scheduler
GradScaler
last_epoch
...
```

部署通常只关心：

```text
inference model parameters
```

因此概念上是：

```text
training checkpoint
        ↓
选择 raw / EMA weights
        ↓
构造 inference model
        ↓
deploy()
        ↓
export
```

---

# 第二部分：PyTorch 推理完整链条

官方当前提供类似：

```bash
python tools/inference/torch_inf.py \
  -c configs/deimv2/deimv2_dinov3_x_coco.yml \
  -r model.pth \
  --input image.jpg \
  --device cuda:0
```

完整流程：

```text
JPEG / Video Frame
        ↓
decode
        ↓
resize
        ↓
tensor conversion
        ↓
normalize
        ↓
[B,3,H,W]
        ↓
DEIM
        ↓
Backbone
        ↓
Encoder
        ↓
Decoder
        ↓
pred_logits + pred_boxes
        ↓
PostProcessor
        ↓
scores + labels + pixel boxes
        ↓
threshold / visualization
```

---

# 第三部分：推理预处理必须和训练/验证对齐

## 3. 最常见的部署错误之一

训练验证：

```text
RGB
Resize
scale to [0,1]
Normalize(mean,std)
CHW
```

部署却：

```text
OpenCV BGR
直接 /255
没有 mean/std
```

即使权重完全正确，也可能得到明显错误结果。

必须核对：

```text
RGB / BGR
input H,W
resize strategy
aspect ratio
padding / letterbox
pixel scale
mean
std
HWC / CHW
dtype
batch dimension
```

Normalize：

$$
x'_c=\frac{x_c-\mu_c}{\sigma_c}
$$

具体 `mean/std` 必须以当前 config 为准。

---

# 第四部分：`eval()`、`no_grad()`、`deploy()` 不一样

## 4. `model.eval()`

改变：

```text
Dropout
BatchNorm
self.training-dependent branches
```

但它不会自动关闭 autograd。

## 5. `torch.no_grad()` / `inference_mode()`

关闭梯度图：

```python
model.eval()

with torch.inference_mode():
    outputs = model(images)
```

## 6. `model.deploy()`

DEIM 顶层的思路是：

```python
self.eval()

for m in self.modules():
    if hasattr(m, "convert_to_deploy"):
        m.convert_to_deploy()
```

因此：

```text
eval()
≠
deploy()
```

`convert_to_deploy()` 可能执行：

```text
训练态结构 → 推理态结构
Conv/BN 等重参数化
缓存固定量
移除训练辅助行为
```

具体以当前模块实现为准。

---

# 第五部分：为什么推理不需要 Matcher / Criterion / DN？

训练：

```text
prediction
↓
Hungarian matching
↓
GT
↓
criterion
↓
loss
↓
backward
```

推理：

```text
prediction
↓
PostProcessor
↓
detections
```

推理没有 GT，因此：

```text
不需要 HungarianMatcher
不需要 Criterion
不需要 loss
不需要 backward
不需要 GT-based DN queries
```

这是训练图与部署图的重要差异。

---

# 第六部分：模型原始输出

主检测分支概念上：

```text
pred_logits [B,Q,C]
pred_boxes  [B,Q,4]
```

以常见 X / COCO：

```text
B = 1
Q = 300
C = 80
```

则：

```text
pred_logits [1,300,80]
pred_boxes  [1,300,4]
```

`pred_boxes` 通常还是：

```text
normalized cxcywh
```

并不是最终像素框。

---

# 第七部分：PostProcessor------从模型输出到检测结果

## 7. cxcywh → xyxy

$$
x_1=c_x-\frac{w}{2}
$$

$$
y_1=c_y-\frac{h}{2}
$$

$$
x_2=c_x+\frac{w}{2}
$$

$$
y_2=c_y+\frac{h}{2}
$$

例如：

```text
[cx,cy,w,h]
=
[0.50,0.40,0.20,0.10]
```

得到：

```text
[x1,y1,x2,y2]
=
[0.40,0.35,0.60,0.45]
```

---

## 8. normalized → pixel

原图：

```text
W=1920
H=1080
```

则：

$$
x_{pixel}=x_{norm}W
$$

$$
y_{pixel}=y_{norm}H
$$

上面的框变成：

```text
x1=768
y1=378
x2=1152
y2=486
```

所以推理必须保留：

```text
original image size
```

---

## 9. Letterbox 时为什么更复杂？

如果不是直接 resize，而是：

```text
保持比例 resize
+
padding
```

恢复原图时必须逆变换：

```text
减 padding
↓
除 scale
↓
clip to original image
```

不能简单把 normalized box 乘原图宽高。

---

# 第八部分：分类 score 与 Top-K

如果分类采用 sigmoid：

$$
p_{q,c}=\sigma(z_{q,c})
$$

则：

```text
Q=300
C=80

score matrix
[300,80]
```

总 query-class pair：

$$
300\times80=24000
$$

flatten：

```text
[24000]
```

再做 top-k。

若 flatten index：

$$
i=qC+c
$$

则：

$$
q=\lfloor i/C\rfloor
$$

$$
c=i\bmod C
$$

因此 top-k index 可以恢复：

```text
query index
class index
```

再从 `pred_boxes[q]` 取对应框。

---

# 第九部分：Top-K 与 threshold 不是一回事

Top-K：

```text
只保留分数最高 K 个
```

Threshold：

```text
只保留 score >= t
```

通常可以：

```text
top-k
↓
threshold
```

改变 threshold 不是改变模型能力，而是改变：

```text
precision / recall operating point
```

通常：

```text
threshold ↑
→ detection 数 ↓
→ precision 可能 ↑
→ recall 可能 ↓
```

---

# 第十部分：为什么 DETR/DEIM 通常不需要 NMS？

传统 dense detector：

```text
一个 GT
→ 大量相邻 anchors/cells
→ 多个重复框
→ NMS
```

DETR：

```text
set prediction
+
one-to-one matching
```

目标是：

```text
一个 GT
↔
一个主要 query
```

因此天然抑制重复预测。

这就是 end-to-end detection 的重要意义。

工程上仍可以加 NMS，但如果 O2O DETR 必须依赖强 NMS
才能正常工作，应进一步检查：

```text
训练
score calibration
postprocess
deployment pipeline
```

---

# 第十一部分：COCO Evaluation

## 10. IoU

$$
IoU= \frac{|B_p\cap B_g|} {\|B_p\cup B_g\|}
$$

例如：

```text
intersection=6000
union=10000
```

则：

$$
IoU=0.6
$$

在 IoU=0.5 时可能算 TP，在 IoU=0.75 时不满足。

---

## 11. Precision / Recall

$$
Precision=\frac{TP}{TP+FP}
$$

$$
Recall=\frac{TP}{TP+FN}
$$

score threshold 改变会产生不同 precision/recall 点。

---

## 12. AP

AP 概括 Precision-Recall curve 的质量。

COCO 主 AP 通常平均：

```text
IoU 0.50
0.55
...
0.95
```

所以它比单独 AP50 更严格。

---

# 第十二部分：AP50 / AP75 / APS / APM / APL

```text
AP50
→ 较宽松定位要求

AP75
→ 更严格定位要求

APS
→ small objects

APM
→ medium objects

APL
→ large objects
```

如果：

```text
AP50 很高
AP75 明显低
```

可能说明：

```text
目标找到了
但定位不够精细
```

如果改 STA 后：

```text
overall AP +0.2
APS +1.1
APL -0.1
```

这比只看 `+0.2 AP` 更能解释修改到底改变了什么。

---

# 第十三部分：为什么改模型后不能只看 AP？

至少同时看：

```text
AP
AP50
AP75
APS
APM
APL
Params
MACs / GFLOPs
Latency
Peak Memory
```

因为：

```text
AP +0.3
Latency +40%
```

对实时检测器未必是更好方案。

---

# 第十四部分：Params

总参数：

$$
Params=\sum_i \#\theta_i
$$

Linear：

```text
in=256
out=256
```

参数：

$$
256\times256+256=65792
$$

Params 主要影响：

```text
weight storage
weight memory
optimizer state
部分 memory bandwidth
```

但：

```text
Params 少
≠
一定更快
```

---

# 第十五部分：MACs 与 FLOPs

卷积 MACs 近似：

$$
H_oW_oC_o \left( \frac{C_i}{groups}K_hK_w
\right)
$$

不同工具可能：

```text
1 MAC = 1 FLOP
```

或：

```text
1 MAC = 2 FLOPs
```

比较论文数字时必须确认定义。

---

# 第十六部分：为什么 GFLOPs 不等于真实速度？

真实 latency 还受：

```text
memory access
kernel fusion
parallelism
operator implementation
tensor shape
GPU architecture
TensorRT tactics
dynamic shape
attention implementation
```

影响。

两个 50 GFLOPs 模型完全可能有明显不同 latency。

---

# 第十七部分：Latency 与 FPS

若：

$$
T=10ms
$$

理论模型 FPS：

$$
FPS=\frac{1000}{10}=100
$$

但真实应用还有：

```text
decode
preprocess
H2D
model
postprocess
D2H
render
I/O
```

因此必须区分：

```text
model-only latency
end-to-end latency
```

---

# 第十八部分：GPU Benchmark 必须 Warmup

第一次运行可能包含：

```text
CUDA context initialization
memory allocation
kernel loading
cache effects
TensorRT tactic initialization
```

正确 PyTorch timing 示例：

```python
model.eval()

with torch.inference_mode():
    for _ in range(50):
        _ = model(x)

torch.cuda.synchronize()

starter = torch.cuda.Event(enable_timing=True)
ender = torch.cuda.Event(enable_timing=True)

times = []

with torch.inference_mode():
    for _ in range(200):
        starter.record()
        _ = model(x)
        ender.record()

        torch.cuda.synchronize()

        times.append(
            starter.elapsed_time(ender)
        )
```

---

# 第十九部分：为什么必须 `cuda.synchronize()`？

CUDA 默认异步。

CPU 可能只是：

```text
提交 kernel
→ 立即继续
```

如果不同步，测到的可能主要是：

```text
kernel launch time
```

而不是完整 GPU 执行时间。

---

# 第二十部分：一个可信 Benchmark 应报告什么？

至少：

```text
GPU
CPU
CUDA
TensorRT
precision
batch size
input size
warmup iterations
measured iterations
mean latency
median latency
P90 / P95
throughput
peak memory
preprocess included?
postprocess included?
```

否则一句：

```text
2.8 ms
```

几乎不可复现。

---

# 第二十一部分：官方 Benchmark 工具

当前官方 README 提供：

```bash
python tools/benchmark/get_info.py \
  -c configs/deimv2/deimv2_dinov3_x_coco.yml
```

用于：

```text
FLOPs
MACs
Params
```

以及：

```bash
python tools/benchmark/trt_benchmark.py \
  --COCO_dir path/to/COCO2017 \
  --engine_dir model.engine
```

用于 TensorRT latency benchmark。

工具报错时不要直接认为模型有问题。逐层验证：

```text
config create
↓
model create
↓
deploy()
↓
dummy forward
↓
benchmark wrapper
```

---

# 第二十二部分：ONNX 是什么？

概念：

```text
PyTorch Module
↓
export
ONNX Graph
↓
runtime/compiler
ONNX Runtime / TensorRT / OpenVINO ...
```

`.pth` 更接近：

```text
parameters / training state
```

`.onnx` 是：

```text
operators
+
tensor graph
+
weights
+
input/output metadata
```

两者本质不同。

---

# 第二十三部分：官方 ONNX 导出

当前官方 README：

```bash
pip install onnx onnxsim

python tools/deployment/export_onnx.py \
  --check \
  -c configs/deimv2/deimv2_dinov3_x_coco.yml \
  -r model.pth
```

---

# 第二十四部分：为什么 `--check` 很重要？

文件导出成功只证明：

```text
exporter 没立即失败
```

不证明：

```text
graph schema 正确
runtime 可加载
shape 正确
数值一致
最终 detection 一致
AP 一致
```

所以 export 后必须做分层验证。

---

# 第二十五部分：部署 Wrapper

导出脚本通常会构造一个 inference wrapper，把：

```text
model
+
postprocessor
```

组合起来。

部署图可能接受：

```text
images
orig_target_sizes
```

直接输出：

```text
labels
boxes
scores
```

具体 input/output names 与 shape 必须以当前 `export_onnx.py` 为准。

---

# 第二十六部分：PostProcessor 是否应该导入 ONNX？

方案 A：

```text
Model + PostProcessor
全部导出
```

优点：

```text
部署端简单
Python/C++ 后处理差异小
```

缺点：

```text
top-k/gather/shape ops 增加导出复杂度
策略不易动态修改
```

方案 B：

```text
只导出 raw predictions
PostProcessor 在 C++/CUDA 实现
```

优点：

```text
更灵活
更容易针对产品优化
```

缺点：

```text
必须严格验证 Python/C++ parity
```

---

# 第二十七部分：Static vs Dynamic Shape

固定：

```text
[1,3,640,640]
```

优点：

```text
TensorRT 更易优化
memory planning 简单
性能稳定
```

动态：

```text
[B,3,H,W]
```

更灵活，但：

```text
export
optimization profile
memory
latency
operator behavior
```

更复杂。

---

# 第二十八部分：为什么高分辨率导出会突然很重？

ViT patch size 16：

640 输入：

```text
40×40 = 1600 tokens
```

2048 输入：

```text
128×128 = 16384 tokens
```

token 数比例：

$$
16384/1600=10.24
$$

如果某些操作复杂度随 token 数高阶增长，内存/计算压力会更剧烈。

因此：

```text
640 export 成功
```

不能推断：

```text
2048 只多占一点资源
```

---

# 第二十九部分：ONNX Tracer Warning

如果源码：

```python
if tensor.shape[0] > 1:
    ...
```

tracing 可能提示：

```text
Tensor → Python boolean
```

这意味着当前 dummy input 的控制流可能被固定。

所以 dynamic shape 比 static export 更难，需要实际多 shape 验证。

---

# 第三十部分：ONNX 验证金字塔

## Level 1：Graph Check

```python
import onnx

m = onnx.load("model.onnx")
onnx.checker.check_model(m)
```

只说明 graph schema 基本合法。

## Level 2：Shape

核对：

```text
input names
input shape
dtype
output names
output shape
```

## Level 3：Raw Numerical Parity

同一输入：

```text
PyTorch
vs
ONNX Runtime
```

比较：

$$
MAE= \frac1N\sum_i\|y_i^{pt}-y_i^{onnx}\|
$$

以及：

$$
MaxError= \max_i\|y_i^{pt}-y_i^{onnx}\|
$$

## Level 4：Detection Parity

比较：

```text
labels
scores
boxes
```

## Level 5：Subset AP

100\~1000 张验证图。

## Level 6：Full Validation AP

完整验证集。

## Level 7：Latency / Memory

达到部署要求。

---

# 第三十一部分：为什么只比较最终 box 不够？

top-k 是离散排序。

两个 backend：

```text
0.50001
0.50000
```

微小误差就可能交换排序。

因此最好同时比较：

```text
raw logits
raw boxes
top-k indices
final detections
full AP
```

---

# 第三十二部分：ONNX Runtime

官方当前提供：

```bash
python tools/inference/onnx_inf.py \
  --onnx model.onnx \
  --input image.jpg
```

它是 PyTorch 与 TensorRT 之间非常重要的中间验证层。

如果：

```text
PyTorch 正确
ONNX Runtime 错
```

优先查：

```text
export graph
operator
input shape
preprocess
postprocess
```

而不是先查 TensorRT。

---

# 第三十三部分：TensorRT

概念：

```text
ONNX
↓
TensorRT Parser
↓
Graph Optimization
↓
Tactic / Kernel Selection
↓
Precision Selection
↓
Engine
```

官方当前示例：

```bash
trtexec \
  --onnx="model.onnx" \
  --saveEngine="model.engine" \
  --fp16
```

---

# 第三十四部分：为什么 TensorRT Engine 环境相关？

engine 与：

```text
GPU architecture
TensorRT version
CUDA stack
precision
input profile
builder settings
```

高度相关。

更稳妥的生产方式：

```text
保留 ONNX + build command + environment
在目标机器/目标镜像构建 engine
```

而不是把 `.engine` 当成完全通用模型文件。

---

# 第三十五部分：FP32 vs FP16

FP16 通常：

```text
weight/activation memory ↓
bandwidth ↓
Tensor Core utilization ↑
latency ↓
```

但数值精度/范围也降低。

所以：

```text
engine 能运行
≠
FP16 精度正确
```

---

# 第三十六部分：DEIMv2 当前 TensorRT 版本注意事项

本课编写时，官方 README 明确提示：

```text
TensorRT FP16 推荐 >= 10.6
```

并指出：

```text
TensorRT 10.4
```

存在已知错误输出风险。

所以部署实验必须记录：

```text
TensorRT exact version
```

并做：

```text
FP32 reference
+
FP16 parity
```

---

# 第三十七部分：TensorRT 推理

官方当前提供：

```bash
python tools/inference/trt_inf.py \
  --trt model.engine \
  --input image.jpg
```

生产系统还要处理：

```text
CUDA stream
input/output buffer
H2D/D2H
preprocess
postprocess
batching
video pipeline
```

---

# 第三十八部分：部署显存由什么组成？

不是只有 weights：

```text
weights
activations
workspace
engine buffers
input/output buffers
CUDA context
runtime allocations
```

因此：

```text
model file 200 MB
```

不代表 GPU 只需要 200 MB。

---

# 第三十九部分：Batch Size 与实时性

Batch=1：

```text
低单帧 latency
适合摄像头实时检测
```

较大 batch：

```text
throughput 可能更高
单样本等待时间可能更长
```

所以：

```text
实时 streaming
```

和：

```text
离线批处理
```

的最优 profile 不一样。

---

# 第四十部分：真实端到端 FPS

假设：

```text
decode      2.0 ms
preprocess  1.0 ms
H2D         0.5 ms
model       6.0 ms
postprocess 0.7 ms
render      2.0 ms
```

串行总计：

$$
12.2ms
$$

端到端理论 FPS：

$$
1000/12.2\approx81.97
$$

而不是只看模型：

$$
1000/6\approx166.7
$$

---

# 第四十一部分：Pipeline 并行

可以设计：

```text
CPU A:
decode frame n+1

GPU:
infer frame n

CPU B:
postprocess frame n-1
```

因此：

```text
throughput
```

与：

```text
单帧串行 latency
```

也不是完全相同的指标。

---

# 第四十二部分：Visualization 是 Error Analysis 工具

官方提供 FiftyOne：

```bash
python tools/visualization/fiftyone_vis.py \
  -c configs/deimv2/deimv2_dinov3_x_coco.yml \
  -r model.pth
```

必须人工分类失败案例：

```text
False Positive
False Negative
Class Confusion
Poor Localization
Duplicate Prediction
Small-object Miss
Crowded-scene Miss
Occlusion Failure
Scale Failure
Background Confusion
Annotation Error
```

---

# 第四十三部分：为什么 AP 不能替代错误分析？

两个模型：

```text
A: 55.0 AP
B: 55.1 AP
```

可能：

```text
A 小目标强
B 大目标强

A 定位强
B 分类强

A 遮挡差
B 低光差
```

只有：

```text
quantitative metrics
+
qualitative error analysis
```

结合，才能指导下一次修改。

---

# 第四十四部分：自定义数据集------从 COCO 80 类到 N 类

至少检查：

```text
COCO JSON
category IDs
class remapping
num_classes
classification heads
DN class embedding
postprocessor labels
visualization class names
pretrained checkpoint mismatch
```

不能只改：

```yaml
num_classes: N
```

就认为完成。

---

# 第四十五部分：COCO Category ID 的坑

COCO 原始 category ID 并不是简单连续的 0\~79。

仓库 Dataset 会做 training label remap。

自定义数据集必须明确：

```text
JSON category_id
↔
continuous training label
↔
display class name
```

三者关系。

---

# 第四十六部分：自定义数据集训练前审计

至少统计：

```text
images
annotations
categories
objects/image
empty images
invalid boxes
class frequency
box area
box aspect ratio
image resolution
```

Invalid box 检查：

```text
w <= 0
h <= 0
x < 0
y < 0
x+w > W
y+h > H
NaN
Inf
```

---

# 第四十七部分：Class Imbalance

例如：

```text
class A = 100000 boxes
class B = 300 boxes
```

只看 overall AP 不够。

应该同时看：

```text
per-class AP
class frequency
confusion/error cases
```

---

# 第四十八部分：修改模型的黄金原则

每次改动必须先写：

```text
1. 为什么改？
2. 改哪个 tensor / module？
3. 理论预期是什么？
4. 哪些指标会变化？
5. 怎样验证？
```

否则：

```text
改代码
→ AP 变化
```

不能形成可靠知识。

---

# 第四十九部分：实验 1------修改 `num_queries`

例如：

```text
300 → 200
```

先预测：

```text
query tensor ↓
self-attention cost ↓
decoder memory ↓
crowded-scene capacity 可能 ↓
latency 可能 ↓
```

Self-attention pair count：

$$
O(Q^2)
$$

300：

$$
300^2=90000
$$

200：

$$
200^2=40000
$$

比例：

$$
40000/90000\approx0.444
$$

仅 query self-attention 的 pair count 理论下降约 55.6%。

但整个模型 latency 不会下降 55.6%，因为：

```text
DINOv3
STA
HybridEncoder
cross-attention
heads
```

仍然存在。

---

# 第五十部分：`num_queries` Ablation

建议：

```text
Q=100
Q=200
Q=300
Q=500
```

记录：

```text
AP
AP50
AP75
APS/M/L
latency
peak memory
crowded-scene recall
```

---

# 第五十一部分：实验 2------Decoder Layers

例如：

```text
6 → 4
```

预期：

```text
decoder latency ↓
memory ↓
refinement stages ↓
AP 可能下降
```

重点看：

```text
AP75
```

因为减少 refinement layer 可能首先影响精细定位。

---

# 第五十二部分：实验 3------冻结 DINOv3

```text
requires_grad=False
```

训练侧：

```text
backward cost ↓
optimizer states ↓
training memory ↓
domain adaptation ability 可能 ↓
```

推理侧：

```text
Params 基本不变
latency 基本不因 requires_grad 改变
```

因为 inference 本来就不计算梯度。

---

# 第五十三部分：实验 4------STA Interaction Indexes

例如 baseline：

```text
[5,8,11]
```

可比较：

```text
[3,7,11]
[4,8,11]
[5,8,11]
```

它影响：

```text
ViT feature depth
semantic strength
spatial detail
interaction timing
gradient path
```

不要把它理解成简单"取三层 feature"。

重点观察：

```text
APS
APM
APL
latency
memory
```

---

# 第五十四部分：实验 5------hidden_dim

例如：

```text
256 → 192
```

会连锁影响：

```text
STA channels
encoder
decoder
Q/K/V
FFN
heads
projection
```

很多 D×D projection 参数量近似：

$$
Params\propto D^2
$$

比例：

$$
(192/256)^2=0.5625
$$

但总模型变化还取决于 backbone。

---

# 第五十五部分：实验 6------FFN Dimension

Decoder FFN：

```text
D
→ Dff
→ D
```

权重近似：

$$
2DD_{ff}
$$

若：

```text
D=256
Dff=2048
```

则两个主 Linear：

$$
2\times256\times2048 = 1,048,576
$$

每层约百万级权重。

---

# 第五十六部分：实验 7------Deformable Sampling Points

增加采样点：

```text
local evidence ↑
compute ↑
```

减少：

```text
latency ↓
representation capacity 可能 ↓
```

目标不是只追 AP，而是研究：

```text
AP vs latency Pareto
```

---

# 第五十七部分：实验 8------去掉高分辨率 Feature Level

640 输入：

```text
1/8  : 80×80 = 6400
1/16 : 40×40 = 1600
1/32 : 20×20 = 400
```

总位置：

$$
6400+1600+400=8400
$$

去掉 1/8：

$$
1600+400=2000
$$

位置数减少：

$$
1-\frac{2000}{8400}\approx76.2%
$$

但预期：

```text
APS 明显受损
```

---

# 第五十八部分：实验 9------输入分辨率

例如：

```text
512
640
768
```

一般：

```text
resolution ↑
small-object detail ↑
compute ↑
memory ↑
latency ↑
```

640 → 1280 的面积：

$$
\frac{1280^2}{640^2}=4
$$

所以分辨率是非常昂贵的性能旋钮。

---

# 第五十九部分：实验 10------Dense O2O Recipe Ablation

固定 architecture：

```text
same model
same init
same epochs
```

比较：

```text
baseline
+ Mosaic
+ MixUp
+ CopyBlend
+ matcher schedule
+ MAL
full recipe
```

记录：

```text
GT density/image
positive matches/image
AP vs epoch
time to target AP
final AP
training stability
```

---

# 第六十部分：为什么不能只看最终 AP？

DEIM 的核心价值之一是 fast convergence。

两个实验可能：

```text
最终都 52 AP
```

但：

```text
A: epoch 30 达到 50 AP
B: epoch 60 才达到 50 AP
```

训练效率完全不同。

所以必须画：

```text
AP vs Epoch
```

---

# 第六十一部分：公平 Ablation

只改变一个核心因素。

例如：

```text
A:
Q=300

B:
Q=200
```

其余尽量保持：

```text
dataset
seed
epochs
batch
optimizer
LR
augmentation
evaluation
hardware
```

一致。

---

# 第六十二部分：为什么 +0.1 AP 不一定可信？

训练有随机波动。

重要实验最好多 seed：

$$
mean\pm std
$$

如果：

```text
提升 0.1
标准差 0.2
```

就不能轻易声称结构稳定有效。

---

# 第六十三部分：Pareto Frontier

实时检测不是单目标优化。

例如：

```text
A: 55.0 AP / 12 ms
B: 54.8 AP / 7 ms
C: 53.0 AP / 3 ms
```

没有绝对最好。

实际问题：

$$
\max AP
$$

subject to：

$$
Latency\le10ms
$$

$$
Memory\le4GB
$$

这才是部署模型选择。

---

# 第六十四部分：为什么 X 不一定最适合你？

X 偏向高精度。

你的系统可能是：

```text
CPU
iGPU
Jetson
edge accelerator
低功耗
30/60 FPS
```

这时较小 variant 可能更优。

Model Zoo 应同时看：

```text
AP
Params
GFLOPs
Latency
hardware
precision
input resolution
```

---

# 第六十五部分：最终毕业项目

最终项目：

> **在一个自定义 COCO-format 数据集上训练 DEIMv2，建立可靠
> baseline，完成至少 3 组结构/训练 ablation，完成 PyTorch → ONNX →
> TensorRT 部署，并提交可复现的技术报告。**

---

# 第六十六部分：毕业项目 A------数据审计

输出：

```text
train images
val images
classes
annotations
objects/image
class histogram
box area histogram
aspect ratio histogram
image resolution histogram
invalid annotations
```

---

# 第六十七部分：毕业项目 B------GT 可视化

随机至少检查：

```text
100 images
```

确认：

```text
class
box
category mapping
small objects
crowded objects
annotation quality
```

任何明显 dataset bug 都应在训练前解决。

---

# 第六十八部分：毕业项目 C------Baseline

选择：

```text
deimv2_dinov3_s
或
deimv2_dinov3_x
```

根据硬件决定。

必须记录：

```text
config
commit hash
checkpoint source
seed
hardware
batch
epochs
LR
input size
augmentation
```

---

# 第六十九部分：毕业项目 D------Tiny Overfit

先：

```text
4~16 images
```

关闭大部分随机增强，反复训练。

预期：

```text
loss 明显下降
predictions 贴近 GT
```

如果不能 overfit：

```text
不要直接开始完整训练
```

优先排查：

```text
label
box
class mapping
loss
optimizer
postprocess
```

---

# 第七十部分：毕业项目 E------正式训练

记录：

```text
loss curves
AP vs epoch
LR
gradient norm
AMP scale
GT density
GPU memory
training time
```

---

# 第七十一部分：毕业项目 F------Baseline Evaluation

至少报告：

```text
AP
AP50
AP75
APS
APM
APL
per-class AP
Params
GFLOPs
PyTorch latency
peak memory
```

---

# 第七十二部分：毕业项目 G------Error Analysis

收集：

```text
50 FP
50 FN
50 localization errors
50 class confusions
50 difficult small objects
```

总结主要失败模式。

---

# 第七十三部分：毕业项目 H------提出假设

例如：

```text
观察：
small-object miss 很多

H1:
提高 input resolution
可能提升 APS

H2:
加强高分辨率 feature
可能提升 APS

H3:
改变 STA interaction
可能改善 spatial detail
```

先写假设，再改代码。

---

# 第七十四部分：毕业项目 I------至少 3 个 Ablation

建议从：

```text
num_queries
decoder layers
STA interaction indexes
input resolution
Dense O2O augmentation
DINOv3 freeze policy
```

选择至少三个。

---

# 第七十五部分：Ablation 表格模板

---

  Experiment        AP    AP50    AP75     APS     APM     APL   Params   GFLOPs   Latency

---

  Baseline

  Exp A

  Exp B

  Exp C

---

# 第七十六部分：毕业项目 J------选择 Final Model

不要写：

```text
B 的 AP 最高，所以选 B。
```

应该写：

```text
B:
AP +0.4
APS +1.1
latency +0.3 ms
memory +4%

仍满足 10 ms latency budget，
因此选择 B。
```

这才是工程结论。

---

# 第七十七部分：毕业项目 K------PyTorch Reference

固定：

```text
batch=1
input=640×640
precision=FP32
```

分别测：

```text
preprocess
model
postprocess
end-to-end
```

它是 ONNX/TensorRT 的 reference。

---

# 第七十八部分：毕业项目 L------ONNX

```text
best checkpoint
↓
load config
↓
model.deploy()
↓
eval
↓
dummy input
↓
ONNX export
↓
checker
↓
ONNX Runtime
```

---

# 第七十九部分：毕业项目 M------ONNX Parity

100 张图比较：

```text
raw logits
raw boxes
labels
scores
pixel boxes
```

记录：

```text
MAE
max error
detection mismatch
AP delta
```

---

# 第八十部分：毕业项目 N------TensorRT

严格顺序：

```text
ONNX
↓
TensorRT FP32
↓
validate
↓
TensorRT FP16
↓
validate
↓
benchmark
```

不要：

```text
ONNX
→ FP16
→ 看一张图
→ 宣布部署成功
```

---

# 第八十一部分：部署验证矩阵

  Backend        Precision     AP   AP50   Latency   Memory Status

---

  PyTorch        FP32
  ONNX Runtime   FP32
  TensorRT       FP32
  TensorRT       FP16

---

# 第八十二部分：AP Delta 阈值

没有统一标准。

你可以根据项目定义，例如：

```text
TRT FP32:
AP delta <= 0.05

TRT FP16:
AP delta <= 0.10
```

这里只是示例。

真正阈值取决于：

```text
任务风险
业务要求
数据规模
baseline
```

---

# 第八十三部分：毕业项目 O------真实视频

测试：

```text
静态
快速运动
小目标
拥挤
低光
运动模糊
尺度变化
```

记录：

```text
end-to-end FPS
P95 latency
dropped frames
GPU utilization
memory
```

---

# 第八十四部分：为什么视频与 COCO AP 都需要？

COCO：

```text
检测精度
```

视频：

```text
实时性
pipeline 稳定性
I/O
temporal behavior
```

回答的是不同问题。

---

# 第八十五部分：最终报告结构

```text
1. Problem
2. Dataset
3. Baseline
4. Architecture
5. Training Recipe
6. Experimental Setup
7. Ablations
8. Error Analysis
9. Final Model
10. PyTorch Inference
11. ONNX
12. TensorRT
13. Numerical Parity
14. Latency
15. Limitations
16. Future Work
```

---

# 第八十六部分：实验文件组织

建议：

```text
configs/
checkpoints/
logs/
tensorboard/
evaluation/
visualizations/
onnx/
tensorrt/
benchmark/
ablation_results.csv
environment.txt
README.md
```

---

# 第八十七部分：环境记录

至少保存：

```text
OS
Python
PyTorch
CUDA
cuDNN
GPU
driver
ONNX
ONNX Runtime
TensorRT
DEIMv2 commit
```

部署结果尤其依赖环境。

---

# 第八十八部分：为什么 Git Commit 必须记录？

`main` 会更新。

报告必须写：

```text
commit hash
```

而不是：

```text
DEIMv2 main
```

模型修改也建议每个实验单独 commit/branch：

```text
exp/q200
exp/decoder4
exp/sta_4_8_11
```

让：

```text
checkpoint
config
source
```

一一对应。

---

# 第八十九部分：不要覆盖同一个 YAML

错误：

```text
修改 config
训练
再次修改同一个 config
训练
```

最终 checkpoint 很容易失去配置对应关系。

建议：

```text
baseline.yml
q200.yml
decoder4.yml
sta_variant.yml
```

---

# 第九十部分：实验命名

推荐：

```text
dataset_model_change_seed
```

例如：

```text
custom_dinov3x_q200_s0
custom_dinov3x_dec4_s0
custom_dinov3x_sta4811_s0
```

---

# 第九十一部分：部署 Artifact 命名

建议包含：

```text
model
commit
input
batch
precision
backend
runtime version
```

例如：

```text
deimv2_x_abc123_640_b1_fp16_trt10.6.engine
```

---

# 第九十二部分：部署 Bug 排查树

```text
PyTorch 正确？
│
├─ No
│  └─ checkpoint/config/preprocess/model
│
└─ Yes
   ↓
ONNX Runtime 正确？
│
├─ No
│  └─ export/operator/shape/preprocess
│
└─ Yes
   ↓
TensorRT FP32 正确？
│
├─ No
│  └─ parser/profile/tactic/runtime
│
└─ Yes
   ↓
TensorRT FP16 正确？
│
├─ No
│  └─ precision/TRT version/numerical
│
└─ Yes
   ↓
Application 正确？
│
├─ No
│  └─ C++ preprocess/postprocess/buffer
│
└─ Yes
   ↓
Deployment Complete
```

---

# 第九十三部分：最重要的部署原则

> **每次只跨一个边界。**

不要：

```text
PyTorch
↓
直接 C++ TensorRT
↓
结果错
```

而应该：

```text
PyTorch
↓ compare
ONNX Runtime
↓ compare
TensorRT Python
↓ compare
C++ TensorRT
```

这样问题才可定位。

---

# 第九十四部分：Preprocess Parity Test

同一张图保存：

```text
Python/PyTorch preprocess tensor
ONNX app preprocess tensor
C++ preprocess tensor
```

比较：

```text
shape
dtype
channel order
min/max
mean/std
selected pixels
```

很多"部署精度下降"其实发生在模型之前。

---

# 第九十五部分：Postprocess Parity Test

固定一组：

```text
logits
boxes
```

分别送：

```text
Python PostProcessor
C++ PostProcessor
```

比较：

```text
top-k indices
labels
scores
boxes
```

这样把模型与后处理完全解耦。

---

# 第九十六部分：坐标单元测试

构造：

```text
normalized cxcywh
=
[0.5,0.5,0.2,0.4]
```

原图：

```text
W=1000
H=500
```

则：

```text
x1=400
y1=150
x2=600
y2=350
```

所有 backend 必须一致。

---

# 第九十七部分：Class Mapping 单元测试

人为令：

```text
class index=3
```

最高。

确认：

```text
PyTorch
ONNX
TensorRT
UI label
```

都显示同一类。

---

# 第九十八部分：Top-K 单元测试

```text
Q=3
C=2

scores =
[
 [0.1,0.8],
 [0.7,0.2],
 [0.4,0.3]
]
```

flatten：

```text
[0.1,0.8,0.7,0.2,0.4,0.3]
```

top-2：

```text
0.8 → q0,c1
0.7 → q1,c0
```

这是部署端非常值得保留的测试。

---

# 第九十九部分：Shape Contract

任何模块都有契约：

```text
input shape
output shape
dtype
stride
coordinate system
semantic meaning
```

例如 STA：

```text
P3 [B,256,H/8,W/8]
P4 [B,256,H/16,W/16]
P5 [B,256,H/32,W/32]
```

替换模块时不仅要 shape 对，还要：

```text
feature order
stride semantics
spatial correspondence
normalization
```

正确。

---

# 第一百部分：改 Backbone 检查表

```text
return levels
channels
strides
pretraining
normalization
freeze policy
optimizer LR group
feature order
encoder in_channels
encoder feat_strides
```

---

# 第一百零一部分：改 Encoder 检查表

```text
in_channels
hidden_dim
feat_strides
number of levels
top-down
bottom-up
output order
decoder feat_channels
decoder feat_strides
```

---

# 第一百零二部分：改 Decoder 检查表

```text
hidden_dim
num_queries
num_layers
num_levels
num_points
reg_max
reg_scale
class head
bbox head
FDR
DN
aux outputs
criterion compatibility
postprocessor compatibility
```

---

# 第一百零三部分：改 Loss 检查表

```text
matcher cost
target definition
loss weight
normalization
aux layers
DN
distributed normalization
AMP numerical stability
logging
```

---

# 第一百零四部分：改 Training Recipe 检查表

```text
augmentation schedule
GT density
batch size
LR
warmup
EMA
stop_epoch
matcher change epoch
epochs
stage reload
best checkpoint logic
```

---

# 第一百零五部分：为什么结构与 Recipe 应分开实验？

如果同时：

```text
改 STA
改 LR
改 MixUp
改 decoder
```

AP 上升后无法归因。

更好的顺序：

```text
architecture ablation
↓
选择结构
↓
recipe tuning
```

---

# 第一百零六部分：统一 Backend Interface

毕业项目建议自己写：

```python
class Backend:
    def preprocess(self, image):
        raise NotImplementedError

    def infer(self, tensor):
        raise NotImplementedError

    def postprocess(
        self,
        outputs,
        original_size
    ):
        raise NotImplementedError
```

实现：

```text
TorchBackend
ONNXBackend
TensorRTBackend
```

---

# 第一百零七部分：统一 Parity Harness

```python
def compare_detections(a, b):
    # deterministic sorting first
    # compare labels
    # compare scores
    # compare boxes
    pass
```

三个 backend 应尽量共享：

```text
same input image
same preprocessing contract
same postprocessing contract
same evaluation
```

---

# 第一百零八部分：统一 Benchmark Harness

```python
def benchmark(
    backend,
    images,
    warmup=50,
    runs=500
):
    # return:
    # mean / median / P90 / P95 latency
    # FPS
    # peak memory
    pass
```

所有实验用同一套 benchmark 代码，避免测量协议漂移。

---

# 第一百零九部分：统一 Evaluation Harness

理想结构：

```text
Backend
↓
COCO validation images
↓
detections
↓
same evaluator
↓
AP
```

这样：

```text
PyTorch
ONNX
TensorRT
```

得到真正可比较的结果。

---

# 第一百一十部分：最终报告最重要的三张图

## 图 1：AP vs Epoch

比较 baseline 与最佳改动。

## 图 2：AP vs Latency

展示 accuracy-speed trade-off。

## 图 3：Failure Cases

展示典型：

```text
FP
FN
localization
class confusion
small-object failure
```

---

# 第一百一十一部分：最终报告最重要的三张表

## Architecture Table

```text
variant
backbone
STA
hidden_dim
queries
decoder layers
params
GFLOPs
```

## Accuracy Table

```text
AP
AP50
AP75
APS
APM
APL
```

## Deployment Table

```text
backend
precision
latency
FPS
memory
AP delta
```

---

# 第一百一十二部分：不要做的 10 件事

```text
1. 不验证数据就直接训练
2. 只看 train loss
3. 只看 AP 不看失败案例
4. 同时改 5 个变量再称为 ablation
5. latency 只测一次
6. 不 warmup 就报告 GPU latency
7. ONNX 文件生成就认为部署完成
8. FP16 看一张图正常就认为正确
9. 不记录 commit/environment
10. checkpoint/config/source 对不上
```

---

# 第一百一十三部分：24 课知识总图

```text
                    DEIMv2
                       │
     ┌─────────────────┼─────────────────┐
     │                 │                 │
 Architecture        Training          System
     │                 │                 │
 HGNetv2            Hungarian O2O     YAMLConfig
 DINOv3             Dense O2O         Registry
 STA                 MAL              Dataset
 HybridEncoder       Mosaic           Transform
 Query               MixUp            Collate
 DeformAttn          CopyBlend        Solver
 DFINETransformer    FGL/DDF          AMP
 DEIMTransformer     AdamW            DDP
 FDR                 LR               EMA
 GO-LSD              stage policy     Checkpoint
     │                 │                 │
     └─────────────────┼─────────────────┘
                       │
                   Inference
                       │
                 PostProcessor
                       │
                   COCO AP
                       │
              FLOPs / Latency
                       │
                    ONNX
                       │
                 TensorRT
                       │
               Real Application
```

---

# 第一百一十四部分：24 课回顾

```text
01  640×640 图片如何变成 boxes
02  PyTorch Tensor / Shape
03  CNN / Stride / Multi-scale
04  Transformer / QKV / Attention
05  DETR / Object Query
06  Hungarian Matching / O2O
07  Deformable Attention

08  deim_dfine_x config
09  HGNetv2 X
10  HybridEncoder
11  DFINETransformer
12  FDR
13  GO-LSD / Multi-layer Supervision
14  deim_dfine_x Input → Loss

15  Dense O2O
16  MAL
17  Dense O2O v2
18  LR / AdamW / EMA / AMP / DDP

19  ViT / DINOv3
20  STA
21  DEIMTransformer
22  deimv2_dinov3_x Full Tensor Trace
23  Full Training Batch Lifecycle
24  Inference / Evaluation / Deployment / Modification / Graduation
```

---

# 第一百一十五部分：最终综合检查题

## Architecture

1. CNN hierarchical feature 与 ViT token feature 的差异是什么？
2. STA 为什么存在？
3. STA 与 HybridEncoder 的职责为什么不同？
4. 8400 怎样由 640 输入得到？
5. query 与 image token 有什么区别？
6. deformable attention 为什么适合这里？
7. reference box 怎样逐层 refinement？
8. FDR 为什么不是简单回归四个数？
9. DFINETransformer 与 DEIMTransformer 的取舍是什么？

## Training

10. Matching cost 与 training loss 有什么区别？
11. O2O 为什么能 end-to-end？
12. sparse supervision 为什么成为问题？
13. Dense O2O 怎样提高监督密度？
14. Mosaic/MixUp/CopyBlend 为什么不只是 regularization？
15. MAL 为什么需要 matchability？
16. DN query 与 matching query 有什么区别？
17. auxiliary losses 为什么存在？
18. DINOv3 为什么使用更小 LR？
19. EMA 为什么常用于 validation？
20. AMP 的 scale/unscale 顺序？
21. DDP 在哪里同步 gradient？

## Inference

22. 为什么 inference 不需要 matcher？
23. normalized cxcywh 怎样恢复 pixel xyxy？
24. top-k flattened index 怎样恢复 query/class？
25. 为什么 DEIM 通常不需要 NMS？
26. threshold 与 top-k 的区别？

## Evaluation

27. AP50 与 AP75 各强调什么？
28. APS 为什么适合评价 STA/multi-scale 改动？
29. Params 与 latency 为什么不同？
30. FLOPs 相同为什么速度仍可能不同？

## Deployment

31. `.pth` 与 `.onnx` 的本质区别？
32. ONNX export 成功为什么不等于部署成功？
33. static/dynamic shape 的取舍？
34. TensorRT engine 为什么环境相关？
35. FP16 为什么必须做 AP parity？
36. 为什么要记录 TensorRT 版本？
37. benchmark 为什么 warmup？
38. GPU timing 为什么 synchronize？

## Research

39. 修改 num_queries 前应该预测哪些影响？
40. 减少 decoder layers 为什么应重点看 AP75？
41. 冻结 backbone 为什么不减少 inference Params？
42. hidden_dim 为什么是全局结构参数？
43. 怎样设计公平 ablation？
44. 为什么 +0.1 AP 可能不足以证明有效？
45. 什么是 accuracy-latency Pareto frontier？

---

# 第一百一十六部分：最终综合实践

要求独立完成：

```text
Step 1  准备 COCO dataset
Step 2  dataset audit
Step 3  GT visualization
Step 4  修改 num_classes
Step 5  pretrained tuning
Step 6  tiny-set overfit
Step 7  baseline training
Step 8  COCO evaluation
Step 9  error analysis
Step 10 提出 3 个假设
Step 11 3+ ablations
Step 12 选择 final model
Step 13 PyTorch benchmark
Step 14 ONNX export
Step 15 ONNX Runtime parity
Step 16 TensorRT FP32
Step 17 TensorRT FP16
Step 18 full validation AP parity
Step 19 end-to-end latency
Step 20 final technical report
```

---

# 第一百一十七部分：毕业验收清单

```text
[ ] 能从 train.py 追完整训练调用链
[ ] 能追每个核心 tensor
[ ] 能解释 HGNetv2 与 DINOv3 两条路线
[ ] 能解释 STA
[ ] 能解释 HybridEncoder
[ ] 能解释 Object Query
[ ] 能解释 Deformable Attention
[ ] 能解释 FDR / GO-LSD
[ ] 能解释 Hungarian O2O
[ ] 能解释 Dense O2O
[ ] 能解释 MAL
[ ] 能解释 Mosaic/MixUp/CopyBlend 的训练意义
[ ] 能解释 AdamW/EMA/AMP/DDP
[ ] 能调试一个 batch 到 optimizer.step
[ ] 能读 COCO AP
[ ] 能做 Error Analysis
[ ] 能修改 num_classes
[ ] 能修改 num_queries
[ ] 能修改 decoder layers
[ ] 能安全修改 STA
[ ] 能设计公平 ablation
[ ] 能正确测 latency
[ ] 能导出 ONNX
[ ] 能验证 ONNX parity
[ ] 能构建 TensorRT engine
[ ] 能验证 FP16 AP
[ ] 能完成真实视频 inference
[ ] checkpoint/config/commit 一一对应
[ ] 能写可复现技术报告
```

---

# 第一百一十八部分：官方当前工具清单

本课编写时，官方 README 列出的工程工具包括：

```text
tools/deployment/export_onnx.py

tools/inference/
  torch_inf.py
  onnx_inf.py
  trt_inf.py

tools/benchmark/
  get_info.py
  trt_benchmark.py

tools/visualization/
  fiftyone_vis.py

reference/
  safe_training.sh
  convert_weight.py
```

它们形成：

```text
training
→ inference
→ export
→ runtime
→ benchmark
→ visualization
```

的工程闭环。

---

# 第一百一十九部分：本课源码阅读清单

建议亲自打开当前仓库：

```text
README.md

engine/deim/deim.py
engine/deim/*postprocessor*
engine/solver/det_engine.py
engine/data/*coco*

tools/deployment/export_onnx.py

tools/inference/
  torch_inf.py
  onnx_inf.py
  trt_inf.py

tools/benchmark/
  get_info.py
  trt_benchmark.py

tools/visualization/
  fiftyone_vis.py

reference/
  safe_training.sh
  convert_weight.py
```

具体文件名和 CLI 参数以你本地 commit 为准。

---

# 第一百二十部分：版本阅读原则

整个课程最后再强调：

```text
真实 runtime behavior
>
你本地 commit 的源码
>
最终 merged config
>
checkpoint 对应 config
>
官方 README
>
论文
>
课程讲义
```

论文负责解释设计思想。

源码决定你当前实际运行的程序。

二者都重要，但工程调试时不能拿论文图替代运行时事实。

---

# 第一百二十一部分：课程最终总结

24 课从：

```text
Tensor shape 是什么？
```

一路走到：

```text
怎样把 DINOv3-based real-time DETR
从源码训练到 TensorRT，
并通过可复现实验证明自己的结构修改是否真的有效？
```

最重要的是建立五层思维：

```text
第一层：Tensor
shape / dtype / coordinate

第二层：Module
why / input / output / math

第三层：Model
backbone / encoder / decoder / query

第四层：Training System
data / matching / loss / optimizer / distributed

第五层：Engineering & Research
evaluation / ablation / deployment / reproducibility
```

---

# 第一百二十二部分：最后一张完整地图

```text
Image
  │
  ▼
Preprocess
  │
  ▼
DINOv3 / HGNetv2
  │
  ▼
STA / Hierarchical Features
  │
  ▼
HybridEncoder
  │
  ▼
Multi-scale Memory
  │
  ▼
Queries + References
  │
  ▼
DEIM / D-FINE Decoder
  │
  ├──────────── TRAIN ──────────────┐
  │                                 │
  │ Hungarian O2O                   │
  │      ↓                          │
  │ Dense O2O                       │
  │      ↓                          │
  │ MAL + Box + Localization Loss   │
  │      ↓                          │
  │ Backward                        │
  │      ↓                          │
  │ AdamW / AMP / DDP               │
  │      ↓                          │
  │ EMA / LR Schedule               │
  │      ↓                          │
  │ Checkpoint                      │
  │                                 │
  └────────── INFERENCE ────────────┤
                                    │
       pred_logits + pred_boxes     │
                 ↓                  │
            PostProcessor           │
                 ↓                  │
       labels / scores / boxes      │
                 ↓                  │
       ┌─────────┴─────────┐        │
       ▼                   ▼        │
    COCO Eval         Visualization │
       ↓                            │
 AP/AP50/AP75/APS/M/L               │
       ↓                            │
 Architecture / Recipe Ablation     │
       ↓                            │
    Final Model                     │
       ↓                            │
      ONNX                          │
       ↓                            │
 ONNX Runtime                       │
       ↓                            │
   TensorRT                         │
       ↓                            │
 FP32 / FP16                        │
       ↓                            │
 Accuracy Parity + Latency          │
       ↓                            │
 Real Application                   │
       └────────────────────────────┘
```

---

# 第一百二十三部分：本课课后作业

## 作业 1：完整 PyTorch Inference Trace

记录：

```text
original size
preprocessed tensor
backbone outputs
encoder outputs
decoder outputs
pred_logits
pred_boxes
top-k
pixel boxes
visualization
```

## 作业 2：手算 PostProcessor

构造：

```text
Q=3
C=2
```

手算：

```text
sigmoid
flatten
top-k
query/class index
cxcywh→xyxy
pixel scaling
```

再用 PyTorch 验证。

## 作业 3：Latency Benchmark

统一：

```text
B=1
640×640
same GPU
same preprocess
```

比较：

```text
PyTorch
ONNX Runtime
TensorRT FP32
TensorRT FP16
```

## 作业 4：Backend Parity

100 张图输出：

```text
raw tensor error
box error
score error
AP delta
```

## 作业 5：num_queries Ablation

```text
300 vs 200
```

记录：

```text
AP
AP75
latency
memory
```

## 作业 6：Dense O2O Ablation

关闭 MixUp，比较：

```text
GT density
AP vs epoch
final AP
```

## 作业 7：STA Ablation

先写理论预测，再训练验证。

## 作业 8：最终毕业项目

完整完成：

```text
custom dataset
training
evaluation
ablation
deployment
parity
benchmark
report
```

---

# 24 课主课程完结

接下来最有价值的事情，不是继续堆理论，而是把这 24 课变成你自己的能力：

```text
拿真实数据集
→ 建 baseline
→ 找失败模式
→ 提假设
→ 改 DEIMv2
→ 做 ablation
→ 部署
→ 验证
→ 写报告
```

当你能独立完成这个闭环时，DEIMv2 就真正从"一个 GitHub
仓库"变成了你掌握的一套视觉算法、训练方法和工程方法。
