# 第 1 课：一张 640×640 图片究竟怎样在 DEIMv2 中变成 Bounding Boxes？

> **课程定位**：DEIMv2 从零到源码 · 第 1 / 24 课
> **本课目标**：暂时不陷入每个模块的内部算法，而是先建立一张准确的"全模型地图"。学完本课，你应该能解释一张图片从输入到检测框的完整数据流，并能读懂后续课程里最常见的Tensor shape。
> **重点模型**：`deim_dfine_x` 与 `deimv2_dinov3_x`
> **源码基准**：DEIMv2 官方仓库当前 `main` 分支（课程制作时核对）

---

## 0. 先回答本课唯一的大问题

假设我们把一张图片处理成：

```text
shape = [3, 640, 640]
```

如果一次输入 2 张图片，那么神经网络看到的是：

```text
[B, C, H, W]
[2, 3, 640, 640]
```

其中：

- `B`：Batch size，一次处理多少张图片；
- `C`：Channel，RGB 图片为 3；
- `H`：Height；
- `W`：Width。

目标检测模型最终希望回答两个问题：

1. **图片里有什么？** ------ classification；
2. **它在哪里？** ------ bounding-box localization。

因此，可以先把 DEIMv2 极度抽象成：

```text
                     一张 RGB 图片
                  [B, 3, 640, 640]
                           │
                           ▼
                    ┌────────────┐
                    │  Backbone  │
                    └────────────┘
                           │
                    提取视觉特征
                           │
                           ▼
                 多尺度 Feature Maps
                           │
                           ▼
                    ┌────────────┐
                    │  Encoder   │
                    └────────────┘
                           │
                 融合不同尺度的信息
                           │
                           ▼
                    ┌────────────┐
                    │  Decoder   │
                    └────────────┘
                           │
                  Object Queries 检测
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
          class logits             boxes
          [B,N,K]                 [B,N,4]
                │                     │
                └──────────┬──────────┘
                           ▼
                    PostProcessor
                           │
                           ▼
             label + score + pixel box
```

这张图是未来 24 课的"总坐标系"。

---

# 1. 什么叫 Bounding Box？

假设图片大小为 `640×640`，图片中有一只猫。

我们可以用矩形框：

```text
(x1, y1) -----------------
    |                     |
    |         cat         |
    |                     |
    ----------------- (x2, y2)
```

例如：

```text
x1 = 120
y1 = 160
x2 = 420
y2 = 500
```

这是一种 `xyxy` 表示。

因此也可以写成：

```text
(cx, cy, w, h)
= (270, 330, 300, 340)
```

如果除以图片宽高进行归一化：

\[ c_x\^{norm}=270/640=0.421875 \]

\[ c_y\^{norm}=330/640=0.515625 \]

\[ w\^{norm}=300/640=0.46875 \]

\[ h\^{norm}=340/640=0.53125 \]

得到：

```text
[0.421875, 0.515625, 0.46875, 0.53125]
```

这件事非常重要。

**Transformer 检测器内部通常不会一直拿 `(120,160,420,500)`这样的像素坐标工作，而经常使用归一化的 box/reference representation。**

最后 PostProcessor 再把结果恢复到原图像素空间。

---

# 2. DEIMv2 的模型外壳其实非常简单

DEIM 系列的顶层模型可以理解成：

```python
def forward(self, x, targets=None):
    x = self.backbone(x)
    x = self.encoder(x)
    x = self.decoder(x, targets)
    return x
```

这段结构来自官方 DEIM 模型容器；

DEIMv2 README 给出的推理模型也保持同样的三段式思想，并在最后增加 `PostProcessor`：

```python
x = self.backbone(x)
x = self.encoder(x)
x = self.decoder(x)
x = self.postprocessor(x, orig_target_sizes)
```

逐行解释。

第 1 行

```python
x = self.backbone(x)
```

输入：

```text
[B, 3, 640, 640]
```

Backbone 的任务不是"直接画框"。

它首先要回答：

> 这张图片中有哪些有意义的视觉模式？

最初可能是边缘、颜色、纹理；随着网络越来越深，特征会逐渐具有更强的语义信息。

第 2 行

```python
x = self.encoder(x)
```

Encoder 的主要工作可以先粗略理解为：

> 把 Backbone 提取出的多个尺度特征进一步投影、交互和融合，让检测器同时拥有局部空间信息与更强的上下文信息。

DEIM/D-FINE 的重要实现是 `HybridEncoder`。

第 3 行

```python
x = self.decoder(x, targets)
```

Decoder 开始真正围绕"我要找哪些目标？"工作。

DETR 系列引入 **object query**。

可以暂时把一个 query 想象成：

> 一个不断向图片特征询问"这里有没有一个值得我负责的物体？"的检测槽位。

如果 `num_queries = 300`，那么最终通常会有 300 个候选检测槽位，而不是说图片一定有 300 个物体。

第 4 步

推理时：

```python
x = self.postprocessor(x, orig_target_sizes)
```

PostProcessor 把网络内部预测转换成更容易使用的：

```text
labels
boxes
scores
```

并根据原图尺寸把 box 恢复到像素坐标。

---

# 3. 两条重点 X 模型路线到底哪里不同？

这是整个课程最需要提前建立的概念。

## 路线 A：DEIM-D-FINE-X

官方 DEIM 的 X 配置对应：

```text
configs/deim_dfine/
├── deim_hgnetv2_x_coco.yml
└── dfine_hgnetv2_x_coco.yml
```

其核心结构可以先画成：

```text
Image
[B,3,640,640]
       │
       ▼
┌──────────────┐
│   HGNetv2    │
│     B5       │
└──────────────┘
       │
       ▼
multi-scale CNN features
       │
       ▼
┌──────────────┐
│HybridEncoder │
│ hidden=384   │
└──────────────┘
       │
       ▼
P3 / P4 / P5
       │
       ▼
┌──────────────────┐
│DFINETransformer  │
│ feat=384×3       │
└──────────────────┘
       │
       ▼
classification + localization
```

官方 X 基础配置中可以看到关键设置：

```yaml
DEIM:
  backbone: HGNetv2

HGNetv2:
  name: 'B5'
  return_idx: [1, 2, 3]

HybridEncoder:
  hidden_dim: 384
  dim_feedforward: 2048

DFINETransformer:
  feat_channels: [384, 384, 384]
  reg_scale: 8
```

现在不要求记住 `reg_scale=8` 是什么。第 12 课会专门拆 D-FINE 的定位表示。

本课只需要记住：

> **DEIM-D-FINE-X = HGNetv2 Backbone + HybridEncoder + DFINETransformer，再配合 DEIM 的训练策略。**

---

# 4. 路线 B：DEIMv2-DINOv3-X

这是我们后半程的另一条主线。

官方配置：

```text
configs/deimv2/deimv2_dinov3_x_coco.yml
```

核心设置：

```yaml
DEIM:
  backbone: DINOv3STAs

DINOv3STAs:
  name: dinov3_vits16plus
  interaction_indexes: [5, 8, 11]
  conv_inplane: 64
  hidden_dim: 256

HybridEncoder:
  in_channels: [256, 256, 256]
  hidden_dim: 256

DEIMTransformer:
  num_layers: 6
  feat_channels: [256, 256, 256]
  hidden_dim: 256
  dim_feedforward: 2048
```

于是第二条路线变成：

```text
Image
[B,3,640,640]
       │
       ▼
┌─────────────────┐
│ DINOv3 ViT-S+   │
└─────────────────┘
       │
       │ ViT features
       ▼
┌─────────────────┐
│       STA       │
│ Spatial Tuning │
│    Adapter      │
└─────────────────┘
       │
       ▼
1/8, 1/16, 1/32 multi-scale features
       │
       ▼
┌─────────────────┐
│ HybridEncoder   │
│ hidden = 256    │
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ DEIMTransformer │
│ 6 decoder layers│
└─────────────────┘
       │
       ▼
classification + localization
```

所以两条路线最表面的区别是：

```text
DEIM-D-FINE-X

HGNetv2
   ↓
HybridEncoder
   ↓
DFINETransformer


DEIMv2-DINOv3-X

DINOv3 + STA
   ↓
HybridEncoder
   ↓
DEIMTransformer
```

注意一个非常重要的观察：

> **两条路线中间都出现了 HybridEncoder。**

这说明以后学习时不能简单认为：

```text
STA = HybridEncoder
```

它们承担的职责并不相同。这个问题会在第 10、20 课分别彻底解决。

---

# 5. 为什么检测器需要多个空间尺度？

假设输入：

```text
640 × 640
```

典型检测特征可以对应 stride：

```text
stride 8
stride 16
stride 32
```

所谓 stride 8，可以暂时理解成：

> 特征图上的一个格子，在输入图片坐标上对应约 8 个像素的步长。

因此：

\[ 640/8=80 \]

\[ 640/16=40 \]

\[ 640/32=20 ]

所以三层 feature map 的空间尺寸为：

```text
P3: 80 × 80
P4: 40 × 40
P5: 20 × 20
```

如果 channel 都是 256：

```text
P3 = [B, 256, 80, 80]
P4 = [B, 256, 40, 40]
P5 = [B, 256, 20, 20]
```

为什么不只留一个？

直觉上：

```text
80×80
格子多、空间信息细
→ 对较小物体有帮助

40×40
→ 中间尺度

20×20
空间更粗、语义通常更强
→ 对较大物体和全局语义有帮助
```

注意：这只是第一课的直觉版本。

以后我们会发现"浅层=小目标、深层=大目标"虽然方便理解，但真实网络的信息已经经过大量跨尺度融合，不能机械地把每层只分配给某一种目标大小。

---

# 6. 三层 Feature Map 为什么会变成 8400？

这是以后阅读 Transformer 检测源码时非常高频的数字。

有：

```text
80 × 80 = 6400
40 × 40 = 1600
20 × 20 =  400
```

所以：

\[ 6400+1600+400=8400 \]

假设三层都是 256 channel：

```text
[B,256,80,80]
[B,256,40,40]
[B,256,20,20]
```

Transformer 更习惯：

```text
[B, N, C]
```

所以可以把每层空间维度 flatten。

第一层：

```python
x = x.flatten(2)
```

shape：

```text
[B,256,80,80]
→
[B,256,6400]
```

再：

```python
x = x.transpose(1, 2)
```

得到：

```text
[B,6400,256]
```

同理：

```text
P3 → [B,6400,256]
P4 → [B,1600,256]
P5 → [B, 400,256]
```

沿 token 维连接：

```python
memory = torch.cat([p3, p4, p5], dim=1)
```

得到：

```text
[B,8400,256]
```

这就是一个非常重要的思想转换：

```text
CNN 世界：

[B,C,H,W]

       ↓ flatten

Transformer 世界：

[B,N,C]
```

其中：

```text
N = 所有空间位置的数量
C = 每个位置的 feature dimension
```

以后当你在源码中看到：

```text
spatial_shapes
level_start_index
memory
value
```

就要想起这 8400 个位置实际上来自：

```text
[80,80]
[40,40]
[20,20]
```

三个不同 feature levels。

---

# 7. 8400 个位置和 300 个 Object Queries 是一回事吗？

**不是。**

这是初学 DETR 时最容易混淆的问题之一。

假设：

```text
image features:
[B,8400,256]

queries:
[B,300,256]
```

8400 表示：

> 图片特征中所有多尺度空间位置。

300 表示：

> 检测器准备的 300 个候选检测槽位。

可以想象成：

```text
8400 image positions
████████████████████████████

             ↑
             │ query 从图像特征中取信息
             │
      Q1 Q2 Q3 ... Q300
```

Decoder 的任务之一，就是让这些 query 根据 image features 不断更新自身表示。

经过若干 decoder layers：

```text
query 1  → 我可能负责 person
query 2  → 我可能负责 car
query 3  → 没有目标
...
query 300
```

最后每个 query 会产生分类和定位结果。

---

# 8. Decoder 输出为什么通常是 `[B, N, C]`？

假设：

```text
B = 2
N = 300 queries
C = 256 hidden dimension
```

则 decoder representation：

```text
[2,300,256]
```

也就是：

```text
第 1 张图：
  query 1 → 256 numbers
  query 2 → 256 numbers
  ...
  query 300 → 256 numbers

第 2 张图：
  query 1 → 256 numbers
  ...
```

注意：

> 这 256 个数本身还不是最终类别，也不是最终 box。

我们还需要 prediction heads。

---

# 9. Classification Head 怎样把 256 维特征变成类别？

为了建立直觉，先用最简单的线性层理解。

假设：

```python
class_head = nn.Linear(256, 80)
```

输入：

```text
[B,300,256]
```

输出：

```text
[B,300,80]
```

可以理解为：

> 每张图有 300 个 query，每个 query 对 80 个 COCO 类别产生分类分数。

线性层数学形式：

$ y=xW\^T+b$

其中：

\[x`\in`{=tex}`\mathbb{R}`{=tex}\^{256} \]

\[ W`\in`{=tex}`\mathbb{R}`{=tex}\^{80`\times256`{=tex}} \]

因此：

\[ y`\in`{=tex}`\mathbb{R}`{=tex}\^{80} \]

对 300 个 query 批量计算：

\[ \[B,300,256\] `\rightarrow [B,300,80]`{=tex}\]

---

# 10. Box Head 怎样得到 4 个数？

概念上可以先理解为：

```text
query feature
[256]
   ↓
bbox prediction head
   ↓
4 numbers
```

所以：

```text
[B,300,256]
→
[B,300,4]
```

这四个数与 bounding box 位置相关。

但是这里必须特别提醒：

> **D-FINE/DEIMTransformer 的真实 box regression 比简单的 `Linear(256,4)` 更复杂。**

后面我们会学习：

- reference points；
- iterative box refinement；
- distribution-based fine-grained localization；
- `reg_max`；
- `reg_scale`；
- `distance2bbox`；
- decoder layer 间的 box 更新。

所以这里的：

```python
nn.Linear(256, 4)
```

只用于理解"检测 head 的输入输出关系"，**不是在冒充真实 D-FINE 实现。**

这是后续读源码时必须保持的严谨性。

---

# 11. 从归一化 Bounding Box 回到 640×640

假设网络最终给出：

```text
(cx, cy, w, h)

=
(0.50, 0.50, 0.25, 0.50)
```

图片大小：

```text
W = 640
H = 640
```

转换成像素：

\[ c_x=0.5`\times640`{=tex}=320 \]

\[ c_y=0.5`\times640`{=tex}=320 \]

\[ w=0.25`\times640`{=tex}=160 \]

\[ h=0.5`\times640`{=tex}=320 \]

所以：

```text
(cx, cy, w, h)
=
(320,320,160,320)
```

转换成 `xyxy`：

\[ x_1=c_x-`\frac{w}{2}`{=tex}=320-80=240 \]

\[ y_1=c_y-`\frac{h}{2}`{=tex}=320-160=160 \]

\[ x_2=c_x+`\frac{w}{2}`{=tex}=400 \]

\[ y_2=c_y+`\frac{h}{2}`{=tex}=480 \]

最终：

```text
(x1,y1,x2,y2)
=
(240,160,400,480)
```

于是可以真的在图片上画：

```text
(240,160) ┌────────────┐
          │            │
          │   object   │
          │            │
          └────────────┘ (400,480)
```

这就是：

```text
网络内部的数字
      ↓
PostProcessor
      ↓
真实图片上的矩形框
```

---

# 12. 训练阶段和推理阶段不是完全一样的

第一课必须先知道这个区别。

## 推理

```text
image
 ↓
backbone
 ↓
encoder
 ↓
decoder
 ↓
predictions
 ↓
postprocessor
 ↓
boxes
```

推理阶段我们已经有训练好的参数，所以目标只是得到结果。

---

## 训练

训练多了 Ground Truth：

```text
image ----------------------┐
                            │
                            ▼
                        model
                            │
                            ▼
                       predictions
                            │
                            │
GT boxes + GT labels -------┤
                            ▼
                         matcher
                            │
                            ▼
                         losses
                            │
                            ▼
                        backward
                            │
                            ▼
                      optimizer.step()
```

其中一个巨大问题是：

> 有 300 个 queries，但图片可能只有 7 个 GT objects，应该让哪几个 query
> 对应这 7 个 GT？

这就会引出：

```text
Hungarian Matching
        ↓
One-to-One matching
        ↓
sparse positive supervision
        ↓
DEIM Dense O2O
```

这是第 6、15～17 课的核心。

因此请先建立一个很重要的区分：

```text
模型结构
≠
训练策略
```

DEIM 的关键价值并不只是"换一个网络层"。

DEIM 很重要的一部分贡献发生在：

```text
prediction ↔ GT 如何匹配
以及
怎样产生更有效、更密集的训练监督
```

这也是为什么我们后面必须把 architecture 和 training framework 分开学习。

---

# 13. `deim_dfine_x` 与 `deimv2_dinov3_x` 的全流程对照

现在把第一课所有东西拼起来。

## DEIM-D-FINE-X

```text
Image
[B,3,640,640]
       │
       ▼
HGNetv2-B5
       │
       │ CNN hierarchical features
       ▼
multi-scale features
       │
       ▼
HybridEncoder
hidden_dim = 384
       │
       ▼
multi-scale encoded features
       │
       ▼
DFINETransformer
       │
       ├── query representation
       ├── classification prediction
       └── fine-grained bbox localization
       │
       ▼
pred_logits + pred_boxes
       │
       ▼
PostProcessor
       │
       ▼
label + score + pixel bbox
```

---

## DEIMv2-DINOv3-X

```text
Image
[B,3,640,640]
       │
       ▼
DINOv3 ViT-S+
       │
       ▼
STA
interaction indexes = [5,8,11]
       │
       ▼
multi-scale features
1/8, 1/16, 1/32
       │
       ▼
HybridEncoder
hidden_dim = 256
       │
       ▼
encoded multi-scale features
       │
       ▼
DEIMTransformer
6 decoder layers
       │
       ├── object queries
       ├── classification
       └── localization
       │
       ▼
predictions
       │
       ▼
PostProcessor
       │
       ▼
label + score + pixel bbox
```

两条路线虽然内部实现不同，但都可以被我们统一理解为：

\[ `\boxed{ Image \rightarrow Backbone \rightarrow Encoder \rightarrow Decoder \rightarrow Predictions }`{=tex} \]

这就是以后读整个仓库最重要的一级抽象。

---

# 14. 一个容易混淆的问题：DINOv3 为什么还需要 Encoder？

初学者很容易产生这样的想法：

> DINOv3 自己已经是 Transformer 了，那后面为什么还有 `HybridEncoder` 和
> `DEIMTransformer`？

因为"Transformer"只是计算模块类别，并不意味着所有 Transformer
都承担相同职责。

可以暂时这样理解：

```text
DINOv3
负责：
“把图片理解成强大的视觉特征。”

STA
负责：
“让这些 ViT 特征更适合多尺度检测。”

HybridEncoder
负责：
“继续进行检测所需的多尺度特征投影、交互和融合。”

DEIMTransformer
负责：
“让 object queries 根据这些图像特征形成最终检测预测。”
```

所以：

```text
DINOv3 Transformer
≠
DEIMTransformer Decoder
```

名称里都有 Transformer，但它们在整条流水线中的位置和职责完全不同。

第 19～22 课会把这件事彻底拆开。

---

# 15. 第一个小型 PyTorch 实验：亲手模拟整条 shape 流

这段代码**不是 DEIMv2 的实现**。

它只有一个目的：

> 让你亲手感受
> `image → multi-scale features → flatten → queries → classes/boxes` 的
> shape。

```python
import torch
import torch.nn as nn

B = 2
C = 256

# 假装这些是 Backbone + Encoder 输出
p3 = torch.randn(B, C, 80, 80)
p4 = torch.randn(B, C, 40, 40)
p5 = torch.randn(B, C, 20, 20)

print("P3:", p3.shape)
print("P4:", p4.shape)
print("P5:", p5.shape)

# [B,C,H,W] -> [B,HW,C]
p3_tokens = p3.flatten(2).transpose(1, 2)
p4_tokens = p4.flatten(2).transpose(1, 2)
p5_tokens = p5.flatten(2).transpose(1, 2)

print("P3 tokens:", p3_tokens.shape)
print("P4 tokens:", p4_tokens.shape)
print("P5 tokens:", p5_tokens.shape)

# 把三个 feature levels 连起来
memory = torch.cat(
    [p3_tokens, p4_tokens, p5_tokens],
    dim=1
)

print("memory:", memory.shape)

# 模拟 300 object queries
queries = torch.randn(B, 300, C)

print("queries:", queries.shape)

# 仅用于理解输出 shape 的 toy heads
class_head = nn.Linear(C, 80)
box_head = nn.Linear(C, 4)

logits = class_head(queries)
boxes = box_head(queries).sigmoid()

print("logits:", logits.shape)
print("boxes:", boxes.shape)
```

预期：

```text
P3:        [2, 256, 80, 80]
P4:        [2, 256, 40, 40]
P5:        [2, 256, 20, 20]

P3 tokens: [2, 6400, 256]
P4 tokens: [2, 1600, 256]
P5 tokens: [2,  400, 256]

memory:    [2, 8400, 256]

queries:   [2, 300, 256]

logits:    [2, 300, 80]
boxes:     [2, 300, 4]
```

请特别记住：

\[ 8400=80^2+40^2+20\^2 \]

以及：

```text
8400 = image feature positions
300  = object queries
256  = feature/hidden dimension
80   = category dimension（此 toy example）
4    = bbox-related output dimension
```

这五个数字承担完全不同的语义。

---

# 16. 第二个 PyTorch 实验：自己完成 box 坐标转换

```python
import torch

# normalized cxcywh
boxes = torch.tensor([
    [0.50, 0.50, 0.25, 0.50]
])

cx, cy, w, h = boxes.unbind(-1)

x1 = cx - w / 2
y1 = cy - h / 2
x2 = cx + w / 2
y2 = cy + h / 2

boxes_xyxy = torch.stack(
    [x1, y1, x2, y2],
    dim=-1
)

print("normalized xyxy:")
print(boxes_xyxy)

scale = torch.tensor([640, 640, 640, 640])

pixel_boxes = boxes_xyxy * scale

print("pixel xyxy:")
print(pixel_boxes)
```

应该得到近似：

```text
normalized:
[0.375, 0.250, 0.625, 0.750]

pixel:
[240, 160, 400, 480]
```

这就是一个极简版的：

```text
normalized prediction
→ coordinate conversion
→ original-size scaling
```

---

# 17. 第三个实验：为什么 `flatten(2)` 不会丢失信息？

考虑：

```python
x = torch.randn(2, 256, 80, 80)

y = x.flatten(2)
```

shape：

```text
[2,256,80,80]
→
[2,256,6400]
```

它并没有把 6400 个位置"平均"掉。

只是把二维空间索引：

\[ (h,w) \]

重新编号成一维位置：

\[ n \]

例如 row-major 情况下，可以直觉理解：

\[ n=hW+w \]

当：

```text
W = 80
h = 3
w = 5
```

则：

\[ n=3`\times80`{=tex}+5=245 \]

所以：

```text
二维空间坐标
(3,5)

↕

一维 token index
245
```

信息仍然存在，只是数据组织形式改变。

后面学习 deformable attention 时，`spatial_shapes` 等变量就是为了让
Transformer 知道这些 flatten 后的位置原本来自哪些 feature level
和二维空间。

---

# 18. 第一次认识真实源码阅读方法

后续课程看到源码时，不要这样读：

```text
第一行是什么意思？
第二行是什么意思？
第三行是什么意思？
```

这种方法非常容易迷失。

以后我们固定问 6 个问题：

```text
① 这个类为什么存在？
② 输入是什么？
③ 输入 shape 是什么？
④ 它做了什么数学/信息处理？
⑤ 输出 shape 是什么？
⑥ 下一个模块为什么需要这个输出？
```

例如：

```python
x = self.backbone(x)
```

不要满足于：

> "调用 backbone。"

而应该最终能回答：

```text
输入：
[B,3,640,640]

为什么存在：
把像素转换成可用于检测的视觉特征。

输出：
多个 feature levels。

为什么是多个：
检测需要不同空间分辨率的信息。

下一步：
HybridEncoder 进一步投影与融合这些 feature maps。
```

这才叫"读懂源码"。

---

# 19. 本课需要掌握的 Vocabulary

这些词后面会不停出现。

  词                  本课阶段的理解

---

  Image Tensor        `[B,C,H,W]` 的图片数据
  Batch               一次送进模型的一组图片
  Channel             特征维度之一；RGB 输入为 3
  Feature Map         网络从图片中提取出的空间特征
  Backbone            从图片提取基础/高级视觉特征
  Stride              feature map 相对输入图的空间缩小倍率
  Multi-scale         同时使用多个空间分辨率
  Encoder             进一步投影、交互、融合视觉特征
  Token               Transformer 处理的一个特征单元
  Object Query        DETR 中用于形成一个候选目标预测的检测槽位
  Decoder             query 与图像特征交互并形成检测预测
  Logit               激活函数之前的原始预测分数
  Bounding Box        描述目标空间位置的矩形
  `cxcywh`            中心 x、中心 y、宽、高
  `xyxy`              左上 x/y、右下 x/y
  Normalize           将坐标等量转换到统一尺度，如 `[0,1]`
  PostProcessor       把网络预测转换成最终可使用的检测结果
  Ground Truth / GT   人工标注的真实类别与框
  Matcher             训练时决定 prediction 与 GT 如何对应
  Loss                衡量预测与训练目标差异的函数

现在只要求"认识"。

后面每一个都会深入。

---

# 20. 本课最重要的 8 个结论

**结论 1**

目标检测不是直接：

```text
image → rectangle
```

而是：

```text
pixels
→ features
→ multi-scale features
→ query-based detection
→ class + localization
→ pixel boxes
```

**结论 2**

DEIM 系列顶层结构非常清楚：

```text
Backbone → Encoder → Decoder
```

复杂性藏在三个模块内部。

**结论 3**

`deim_dfine_x` 和 `deimv2_dinov3_x` 的 Backbone/Decoder 不同：

```text
HGNetv2 + DFINETransformer

vs

DINOv3+STA + DEIMTransformer
```

**结论 4**

两者都需要检测所需的多尺度特征。

**结论 5**

对于 640 输入和 stride `[8,16,32]`：

```text
80×80
40×40
20×20
```

flatten 后共有：

\[ 8400 \]

个空间位置。

**结论 6**

```text
8400 image positions
```

与：

```text
300 object queries
```

完全不是一个概念。

**结论 7**

Decoder 的 query feature 还不是最终 box。

它还需要 classification/localization heads 和相应的 box
refinement/representation 机制。

**结论 8**

DEIM 的"模型结构"和"训练优化"必须分开理解。

```text
Architecture:
Backbone + Encoder + Decoder

Training:
GT matching + losses + Dense O2O + augmentation + optimizer recipe + ...
```

这会成为后续课程的一条主线。

---

# 21. 本课自测

先不要查答案，试着口头回答。

Q1

`[2,3,640,640]` 中四个数字分别是什么？

Q2

为什么目标检测需要 feature map，而不直接在 RGB pixel 上输出 box？

Q3

640 输入、stride=16 时 feature map 的空间尺寸是多少？

Q4

为什么：

```text
80×80 + 40×40 + 20×20
```

会得到 8400？

Q5

`[B,256,80,80]` 怎样变成 `[B,6400,256]`？

Q6

8400 与 300 queries 有什么本质区别？

Q7

为什么 `[B,300,256]` 不能直接理解成 300 个 bounding boxes？

Q8

`deim_dfine_x` 与 `deimv2_dinov3_x` 最明显的两个结构差异是什么？

Q9

为什么 DINOv3 已经是 Transformer，DEIMv2 后面仍然需要 `HybridEncoder` 和
`DEIMTransformer`？

Q10

训练阶段为什么还需要 GT、Matcher 和 Loss，而推理阶段不需要？

如果这 10 题能用自己的话回答出来，第 1 课就真正学会了。

---

# 22. 课后动手题

请自己修改第一个 PyTorch 实验。

把：

```text
640×640
```

改成：

```text
320×320
```

假设 feature strides 仍为：

```text
[8,16,32]
```

先不要运行代码，自己算：

```text
P3 = ?
P4 = ?
P5 = ?

总 token 数 N = ?
```

然后再用 PyTorch 验证。

再思考：

> 输入从 640 降到 320 后，空间 token 数是变成原来的 1/2，还是 1/4？

为什么？

这个问题会直接连接到后面的：

```text
FLOPs
latency
attention complexity
small-object performance
```

---

# 23. 下一课预告

## 第 2 课：PyTorch Tensor 与 Shape------以后不再害怕 `reshape / flatten / permute / transpose`

下一课会把本课出现的：

```text
[B,C,H,W]
[B,N,C]
[B,head,N,d]
```

全部拆透。

我们会专门用 DEIMv2 风格的数据做实验：

```text
view
reshape
flatten
transpose
permute
unsqueeze
squeeze
cat
stack
split
expand
repeat
contiguous
```

并重点解决一个源码阅读中的常见问题：

> **为什么仅仅交换几个维度，Transformer 就能把二维 feature map 当成
> token sequence？**

之后再进入 CNN、Transformer、DETR，我们就不会被 shape 操作卡住。

---

# 附录 A：本课的 Shape Cheat Sheet

```text
输入图片
[B,3,640,640]

典型三层检测特征
[B,C,80,80]
[B,C,40,40]
[B,C,20,20]

flatten
[B,6400,C]
[B,1600,C]
[B, 400,C]

concat
[B,8400,C]

object queries（概念示例）
[B,300,C]

classification output（以 80 类为直觉示例）
[B,300,80]

box-related prediction（抽象表示）
[B,300,4]
```

其中：

```text
B     = batch size
3     = RGB channels
C     = feature dimension
8400  = spatial positions
300   = object queries
80    = category dimension（示例）
4     = bbox-related coordinates
```

---

# 附录 B：本课源码导航

后续课程会逐个进入这些位置；第 1 课只需要知道它们存在。

```text
DEIMv2/
├── train.py
├── configs/
│   ├── deimv2/
│   │   └── deimv2_dinov3_x_coco.yml
│   └── ...
├── engine/
│   ├── backbone/
│   │   ├── HGNetv2 ...
│   │   └── DINOv3STAs ...
│   └── deim/
│       ├── HybridEncoder / LiteEncoder ...
│       ├── DFINETransformer ...
│       ├── DEIMTransformer ...
│       └── PostProcessor ...
└── tools/
```

DEIM-D-FINE 的配置路线还可以从原始 DEIM 仓库对应目录学习：

```text
configs/deim_dfine/
├── deim_hgnetv2_x_coco.yml
└── dfine_hgnetv2_x_coco.yml
```

我们后面不会靠文件名猜功能，而会沿着真实 `forward()` 调用链阅读。

---

# 附录 C：参考资料

课程优先使用官方/一手资料：

1. DEIMv2 官方仓库https://github.com/Intellindust-AI-Lab/DEIMv2
2. DEIM 官方仓库（用于 DEIM-D-FINE 路线与 Dense O2O 的历史/实现对照）https://github.com/Intellindust-AI-Lab/DEIM
3. DEIMv2 X 配置
   https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/configs/deimv2/deimv2_dinov3_x_coco.yml

> 后续课程涉及某个具体模块时，会继续核对当前官方源码，而不是只依赖论文结构图或二手教程。

---

**第 1 课结束。**

如果你现在能够从头解释：

```text
[B,3,640,640]
→ Backbone
→ multi-scale features
→ HybridEncoder
→ Transformer Decoder
→ object queries
→ class + localization
→ PostProcessor
→ pixel bounding boxes
```

那么我们已经完成了第一件最重要的事：

> **先知道自己在整张 DEIMv2 地图的哪里，再进入每一栋建筑。**
