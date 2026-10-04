# DEIMv2 源码课程 · 第 14 课
# 把 `deim_dfine_x` 从输入到 Loss 完整走一遍

> **本课目标**
>
> 前 9～13 课已经分别拆过 `HGNetv2 → HybridEncoder → DFINETransformer → FDR → GO-LSD`。  
> 这一课不再孤立地看模块，而是把 **`deim_dfine_x` 的一次完整训练 forward** 串成一条闭环：
>
> ```text
> COCO image + targets
>        ↓
> transforms / batch
>        ↓
> HGNetv2-B5
>        ↓
> C3 / C4 / C5
>        ↓
> HybridEncoder
>        ↓
> P3 / P4 / P5
>        ↓
> flatten + concatenate
>        ↓
> encoder proposals
>        ↓
> Top-K 300 queries
>        ↓
> + denoising queries（训练期）
>        ↓
> 6-layer DFINETransformer
>        ↓
> class + box + FDR distributions
>        ↓
> main / aux / pre / encoder / DN outputs
>        ↓
> Hungarian Matching
>        ↓
> Global Union Set
>        ↓
> MAL + L1 + GIoU + FGL + DDF
>        ↓
> weighted total loss
>        ↓
> backward()
> ```
>
> 本课最重要的成果不是背下一组 shape，而是建立一种能力：
>
> **以后你看到 `deim_dfine_x` 中任何一个 Tensor，都能回答它来自哪里、为什么是这个 shape、下一步去哪里、最终怎样影响 loss。**

---

# 0. 本课使用的模型与“640×640”假设

本课主角是 DEIM 仓库中的：

```text
configs/deim_dfine/deim_hgnetv2_x_coco.yml
```

它组合了：

```text
D-FINE X architecture
+
DEIM Dense O2O training configuration
```

官方 X 配置继承：

```yaml
__include__:
  - ./dfine_hgnetv2_x_coco.yml
  - ../base/deim.yml
```

因此你必须始终记住：

> `deim_dfine_x` 不是“一个单独 YAML 里写完所有参数”，而是多个 YAML 合并后的最终配置。

本课为了能逐项算清楚 shape，统一使用一个**教学样例**：

```text
Batch size B = 2
Input H = W = 640
COCO num_classes = 80
num_queries = 300
hidden_dim = 384      # X 的 HybridEncoder / decoder feature width
num_layers = 6
num_levels = 3
reg_max = 32
reg_scale = 8         # X override
```

> **重要：训练时实际输入不一定永远是 640×640。**
>
> DEIM 的 collate / augmentation recipe 可以使用 multi-scale 等策略。  
> 所以下文 `80×80 / 40×40 / 20×20 / 8400` 是在 **当前 batch 输入恰好为 640×640** 时的具体数字。
>
> 一旦输入尺寸变化，空间维度和 token 数也会变化；但所有推导规则完全相同。

---

# 1. 先把 `deim_dfine_x` 的最终结构写出来

最终模型容器非常简单：

```python
class DEIM(nn.Module):
    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)
        return x
```

也就是：

```text
images
  │
  ▼
HGNetv2-B5
  │
  ▼
HybridEncoder
  │
  ▼
DFINETransformer
  │
  ▼
prediction dict
```

训练时真正的 loss 并不在 `DEIM.forward()` 内部计算。

而是：

```text
model(images, targets)
        ↓
outputs
        ↓
criterion(outputs, targets)
        ↓
loss_dict
        ↓
sum(loss_dict.values())
        ↓
backward()
```

这是本课第一条必须记住的边界：

```text
Model:
负责预测

Criterion:
负责 matching + loss
```

---

# 2. X 配置的关键参数

`deim_hgnetv2_x_coco.yml` 的 DEIM 训练配置包含：

```yaml
optimizer:
  type: AdamW

  params:
    - params: '^(?=.*backbone)(?!.*norm|bn).*$'
      lr: 0.000005

    - params: '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$'
      weight_decay: 0.

  lr: 0.0005
  betas: [0.9, 0.999]
  weight_decay: 0.000125

epoches: 58
flat_epoch: 29
no_aug_epoch: 8
```

而基础 D-FINE X architecture 使用：

```text
HGNetv2:
    B5
    return_idx = [1,2,3]

HybridEncoder:
    hidden_dim = 384
    dim_feedforward = 2048

DFINETransformer:
    feat_channels = [384,384,384]
    reg_scale = 8
```

基础 decoder 配置还给出：

```text
num_queries = 300
num_denoising = 100
reg_max = 32
num_layers = 6
num_levels = 3
num_points = [3,6,3]
```

DEIM base 则把 Criterion 改成：

```yaml
DEIMCriterion:
  weight_dict:
    loss_mal: 1
    loss_bbox: 5
    loss_giou: 2
    loss_fgl: 0.15
    loss_ddf: 1.5

  losses:
    - mal
    - boxes
    - local

  gamma: 1.5
```

所以这一课最终要解释的正是：

```text
loss_mal
loss_bbox
loss_giou
loss_fgl
loss_ddf
```

怎样从前面的 Tensor 一步步产生。

---

# 3. 第一个完整 Shape Map

先不管内部细节，直接看 640×640 的主干：

```text
Input
[B,3,640,640]

        │ HGNetv2-B5
        ▼

C3
[B,512,80,80]

C4
[B,1024,40,40]

C5
[B,2048,20,20]

        │ HybridEncoder
        ▼

P3
[B,384,80,80]

P4
[B,384,40,40]

P5
[B,384,20,20]

        │ flatten spatial dimensions
        ▼

P3:
[B,6400,384]

P4:
[B,1600,384]

P5:
[B,400,384]

        │ concatenate
        ▼

memory
[B,8400,384]

        │ encoder proposal scoring + Top-K
        ▼

300 matching queries
[B,300,384]

reference boxes
[B,300,4]

        │ DFINE Decoder × 6
        ▼

pred_logits
[B,300,80]

pred_boxes
[B,300,4]

pred_corners
[B,300,132]

        │ Criterion
        ▼

scalar losses
```

其中：

\[
8400
=
80\times80
+
40\times40
+
20\times20
\]

即：

\[
6400+1600+400=8400
\]

而：

\[
132
=
4\times(32+1)
\]

即：

```text
4 edges
×
33 bins
```

---

# 4. Stage A：DataLoader 最终交给模型什么？

训练一个 batch 时，可以先抽象成：

```python
samples, targets = batch
```

其中：

```text
samples:
[B,3,H,W]
```

而：

```text
targets
```

是长度为 `B` 的 Python list。

例如：

```python
targets = [
    {
        "labels": Tensor[N0],
        "boxes":  Tensor[N0,4],
        ...
    },
    {
        "labels": Tensor[N1],
        "boxes":  Tensor[N1,4],
        ...
    }
]
```

注意：

```text
N0 != N1
```

完全正常。

因为不同图片里的 GT 数量不同。

---

# 5. GT Box 在进入 Criterion 时是什么格式？

DEIM 的 transforms 中包含：

```text
ConvertBoxes
fmt = cxcywh
normalize = True
```

因此 Criterion 期待：

```text
target["boxes"]
=
[cx, cy, w, h]
```

并且坐标归一化到：

```text
[0,1]
```

例如一张 640×640 图片中：

```text
pixel box:
x1=160
y1=128
x2=416
y2=448
```

则：

```text
cx = (160+416)/2 / 640
   = 288/640
   = 0.45

cy = (128+448)/2 / 640
   = 288/640
   = 0.45

w = (416-160)/640
  = 256/640
  = 0.40

h = (448-128)/640
  = 320/640
  = 0.50
```

所以：

```text
GT = [0.45, 0.45, 0.40, 0.50]
```

---

# 6. Stage B：进入 `DEIM.forward()`

模型入口：

```python
def forward(self, x, targets=None):
    x = self.backbone(x)
    x = self.encoder(x)
    x = self.decoder(x, targets)
    return x
```

本课教学 batch：

```text
x
[2,3,640,640]
```

现在开始逐层追踪。

---

# 7. Stage C：HGNetv2-B5 Backbone

第 9 课已经拆过 HGNetv2。

X 使用：

```text
HGNetv2-B5
```

其后三个返回 stage 的通道数：

```text
512
1024
2048
```

stride：

```text
8
16
32
```

所以输入：

```text
[2,3,640,640]
```

得到：

```text
C3:
[2,512,80,80]

C4:
[2,1024,40,40]

C5:
[2,2048,20,20]
```

因为：

\[
640/8=80
\]

\[
640/16=40
\]

\[
640/32=20
\]

---

# 8. Backbone 输出为什么不是直接送 Decoder？

因为三个尺度：

```text
channel 数不同
```

分别是：

```text
512
1024
2048
```

而 Transformer 希望后续使用统一 embedding width。

同时：

```text
C3
C4
C5
```

虽然已有多尺度，但跨尺度信息融合还不充分。

所以中间还需要：

```text
HybridEncoder
```

---

# 9. Stage D：HybridEncoder 输入

输入 list：

```python
feats = [
    C3,
    C4,
    C5
]
```

shape：

```text
[
  [2, 512, 80,80],
  [2,1024, 40,40],
  [2,2048, 20,20]
]
```

HybridEncoder 首先通过 input projection 将 channel 对齐。

目标 width：

```text
384
```

于是：

```text
C3 → [2,384,80,80]
C4 → [2,384,40,40]
C5 → [2,384,20,20]
```

---

# 10. 为什么 X 是 384，而不是我们之前常见的 256？

因为 X 是大模型配置。

这里必须区分：

```text
base default:
hidden_dim = 256

X override:
HybridEncoder hidden_dim = 384
DFINETransformer feat_channels = [384,384,384]
```

因此读 YAML 时绝不能只看：

```text
base/dfine_hgnetv2.yml
```

还必须看：

```text
X-specific override
```

最终 merged config 才是真实模型。

---

# 11. HybridEncoder 的内部路线

第 10 课已经拆过：

```text
input projection
      ↓
intra-scale Transformer
      ↓
top-down FPN
      ↓
bottom-up PAN
```

最终输出：

```text
P3 [2,384,80,80]
P4 [2,384,40,40]
P5 [2,384,20,20]
```

注意：

> HybridEncoder 改善的是 feature representation；它并没有把 feature map 变成 300 queries。

Query selection 是 decoder 入口做的事情。

---

# 12. Stage E：DFINETransformer 接收什么？

Decoder 接收一个 list：

```text
[
  P3,
  P4,
  P5
]
```

即：

```text
[
 [2,384,80,80],
 [2,384,40,40],
 [2,384,20,20]
]
```

`DFINETransformer` 会先准备 encoder input。

核心思想：

```text
每个二维 feature map
↓
flatten spatial dimensions
↓
统一成为 token sequence
↓
三个尺度 concatenate
```

---

# 13. P3 Flatten

原始：

```text
P3
[2,384,80,80]
```

空间位置：

\[
80\times80=6400
\]

flatten 后：

```text
[2,384,6400]
```

transpose：

```text
[2,6400,384]
```

每个 token：

```text
384-dimensional feature
```

一共有：

```text
6400 tokens
```

---

# 14. P4 Flatten

```text
P4
[2,384,40,40]
```

\[
40\times40=1600
\]

得到：

```text
[2,1600,384]
```

---

# 15. P5 Flatten

```text
P5
[2,384,20,20]
```

\[
20\times20=400
\]

得到：

```text
[2,400,384]
```

---

# 16. 三个尺度 Concatenate

沿 token dimension：

```text
[2,6400,384]
+
[2,1600,384]
+
[2, 400,384]
```

得到：

```text
memory
[2,8400,384]
```

其中：

\[
6400+1600+400=8400
\]

这就是以后看到：

```text
8400
```

时必须瞬间想到的来源。

---

# 17. `spatial_shapes` 是什么？

虽然 feature 已 flatten：

```text
[B,8400,384]
```

但 Deformable Attention 必须知道每一段 token 原来属于哪张二维 feature map。

因此保留：

```text
spatial_shapes
```

概念上：

```python
[
    [80,80],
    [40,40],
    [20,20]
]
```

shape：

```text
[3,2]
```

---

# 18. `level_start_index` 是什么？

三层 flatten 后在大 sequence 中的位置：

```text
P3:
0 ... 6399

P4:
6400 ... 7999

P5:
8000 ... 8399
```

所以：

```text
level_start_index
≈ [0, 6400, 8000]
```

这让 Deformable Attention 可以把：

```text
flattened index
```

重新解释成：

```text
level + 2D coordinate
```

---

# 19. Stage F：8400 个位置为什么不能全部成为 Object Query？

这里必须区分：

```text
image feature positions
```

和：

```text
object queries
```

8400 是：

```text
multi-scale spatial positions
```

而：

```text
num_queries = 300
```

是 decoder 要保留的 object hypothesis 数量。

因此需要：

```text
8400 candidates
↓
score / proposal
↓
Top-K
↓
300 initial object queries
```

---

# 20. Encoder Output Proposals

DFINETransformer 会基于 flatten 后的 memory 生成：

```text
encoder proposals
```

可以把每个 spatial position 理解成一个候选：

```text
(position, scale)
→ candidate box
```

然后通过 encoder classification head 得到：

```text
enc_outputs_class
```

概念 shape：

```text
[B,8400,80]
```

并产生：

```text
enc_outputs_coord
```

概念 shape：

```text
[B,8400,4]
```

---

# 21. Top-K Query Selection

默认：

```text
num_queries = 300
```

于是从：

```text
8400 proposals
```

中选择：

```text
300
```

个最有希望的候选。

得到核心 matching queries：

```text
target/query content
[B,300,D]
```

以及：

```text
reference boxes
[B,300,4]
```

对于 X，feature width 与 decoder 的具体内部 projection 要以源码构造为准；从 HybridEncoder 到 decoder 输入的 feature channel 是 384。

---

# 22. Top-K 的比例有多稀疏？

\[
300/8400
\approx0.035714
\]

即：

```text
约 3.57%
```

所以 decoder 不再让 8400 个 spatial positions 全部承担完整的 object-query refinement。

它先进行候选筛选，再让 300 个 query 进入 decoder。

---

# 23. 这 300 个 Query 和 DETR 原始 Learnable Query 有什么区别？

原始 DETR 常从：

```text
learned query embeddings
```

开始。

D-FINE / RT-DETR 系路线更强调：

```text
encoder feature
→ proposal
→ Top-K selection
→ decoder initialization
```

因此 Query 初始状态与图像内容联系更直接。

这也是第 5 课 Object Query 概念在真实源码中的进一步具体化。

---

# 24. Stage G：训练时还会加入 Denoising Queries

训练模式且有 `targets` 时：

```text
GT
↓
加 label noise
加 box noise
↓
construct denoising queries
```

它们会与：

```text
300 matching queries
```

拼接后一起送入 decoder。

所以训练时 decoder 内部的 Query 数：

```text
Q_total
=
Q_dn
+
300
```

但：

```text
Q_dn
```

不是简单地永远等于配置中的 `num_denoising=100`。

真实数量与：

```text
当前 batch GT 数量
denoising group 构造
```

有关。

这是一个非常容易在 shape 推导中写错的地方。

---

# 25. 为什么配置写 `num_denoising=100`，不能直接说 Decoder 一定有 400 Queries？

因为 denoising query generation 会根据：

```text
max number of GT in batch
number of denoising groups
```

组织 DN queries。

因此：

```text
100
```

更接近 denoising budget / target setting，而不是一句：

```text
“每张图固定追加 100 个 query”
```

后面调试真实 batch 时应该直接打印：

```python
dn_embed.shape
dn_bbox_unact.shape
```

确认实际 Q_dn。

---

# 26. 为什么要 Denoising？

普通 DETR 的 decoder 需要从：

```text
比较不稳定的初始 object hypotheses
```

慢慢学会：

```text
类别
+
位置
```

DN 思想给它一部分：

```text
由 GT 扰动得到的 query
```

让 decoder 学习：

```text
如何把 noisy label / noisy box
恢复到正确 object
```

从而改善训练稳定性和收敛。

---

# 27. Matching Queries 与 DN Queries 的角色不同

```text
Matching queries
↓
需要 Hungarian Matching
↓
与 GT 建立 O2O assignment
```

而：

```text
DN queries
↓
构造时已经知道来自哪个 GT
↓
不需要再通过 Hungarian 找身份
```

因此 Criterion 后面会分开处理：

```text
normal matching branch
DN branch
```

---

# 28. Stage H：进入 6 层 DFINE Decoder

X：

```text
num_layers = 6
```

即：

```text
Layer 0
Layer 1
Layer 2
Layer 3
Layer 4
Layer 5
```

每一层核心仍是：

```text
Query Self-Attention
↓
Multi-scale Deformable Cross-Attention
↓
FFN
↓
classification
↓
FDR localization refinement
```

---

# 29. Self-Attention 在谁和谁之间？

Query Self-Attention：

```text
300 matching queries
```

之间彼此交流。

训练时如果拼入 DN queries：

```text
Q_total
```

会配合 attention mask 控制不希望发生的信息泄漏。

对于只看 matching query 的主线：

```text
[B,300,D]
```

经过 self-attention 后：

```text
[B,300,D]
```

shape 不变。

---

# 30. Cross-Attention 在谁和谁之间？

Query：

```text
[B,300,D]
```

去读取：

```text
memory
[B,8400,384]
```

但不是做：

```text
300 × 8400
```

的完整全局 attention。

而是第 7 课学习的：

```text
Multi-Scale Deformable Attention
```

围绕 reference point：

```text
在 P3 / P4 / P5
各取少量 sampling points
```

X/base 配置：

```text
num_points = [3,6,3]
```

每个 head 每个 query 总采样点：

\[
3+6+3=12
\]

---

# 31. 为什么 P4 取 6 个点，而 P3/P5 各 3 个？

配置：

```text
[3,6,3]
```

说明实现允许：

```text
不同 feature level
使用不同 sampling point 数量
```

它不是教科书中永远固定：

```text
P points per level
```

的版本。

本课不再重复第 7 课公式，只记住：

```text
query
+
reference
+
offsets
+
weights
→ sparse multi-scale feature sampling
```

---

# 32. Stage I：每层产生 Classification Logits

对于 COCO：

```text
num_classes = 80
```

matching branch 每层：

```text
logits_l
[B,300,80]
```

6 层训练输出可想成：

```text
[6,B,300,80]
```

最终：

```text
pred_logits
=
out_logits[-1]
```

shape：

```text
[B,300,80]
```

---

# 33. Classification 为什么没有显式 no-object 第 81 类？

这里使用：

```text
sigmoid-style classification
```

而不是原始 DETR 的：

```text
81-way softmax
```

因此：

```text
pred_logits
[B,Q,80]
```

而不是：

```text
[B,Q,81]
```

unmatched query 的目标相当于：

```text
80 个类别全 0
```

---

# 34. Stage J：每层产生 Bounding Box

每个 Query 最终 box：

```text
[cx,cy,w,h]
```

所以：

```text
out_bboxes
[6,B,300,4]
```

最终：

```text
pred_boxes
[B,300,4]
```

通常是 normalized coordinates。

---

# 35. Stage K：FDR Distribution

第 12 课的关键：

```text
reg_max = 32
```

所以每条边：

```text
33 bins
```

四条边：

```text
4 × 33 = 132
```

每层：

```text
pred_corners_l
[B,300,132]
```

所有层概念 shape：

```text
[6,B,300,132]
```

最终层：

```text
pred_corners
[B,300,132]
```

---

# 36. 这 132 个值不是 Box

再次强调：

```text
pred_boxes
[B,300,4]
```

是最终 box representation。

而：

```text
pred_corners
[B,300,132]
```

是 FDR 的：

```text
4 个离散 localization distributions
```

它们通过：

```text
Softmax
+
Integral
+
non-uniform weighting
+
distance2bbox
```

参与 refined box 生成和 localization loss。

---

# 37. Stage L：6 层 Iterative Refinement

你可以把一个 Query 的 box 演化理解成：

```text
initial reference
      ↓
Layer 0
      ↓
box_0
      ↓ detach as next reference
Layer 1
      ↓
box_1
      ↓
...
Layer 5
      ↓
final box
```

同时 distribution：

```text
dist_0
↓
dist_1
↓
...
dist_5
```

最终：

```text
dist_5
```

还会作为 GO-LSD Teacher。

---

# 38. 为什么源码中大量 `.detach()`？

例如：

```text
ref_points_detach
output_detach
teacher_corners.detach()
```

它们不是随便为了省显存。

不同 detach 有不同目的：

```text
reference detach:
控制 iterative refinement 的梯度路径

teacher detach:
保证 self-distillation 是 final → shallow 的单向监督
```

以后看到 `.detach()`，一定问：

> **这里到底是在切断哪一条梯度路径？**

---

# 39. Stage M：最终主输出 Dictionary

训练时主输出至少包含：

```python
{
    "pred_logits": ...,
    "pred_boxes": ...,
    "pred_corners": ...,
    "ref_points": ...,
    "up": ...,
    "reg_scale": ...,
    ...
}
```

matching branch shape：

```text
pred_logits:
[B,300,80]

pred_boxes:
[B,300,4]

pred_corners:
[B,300,132]

ref_points:
[B,300,4]
```

---

# 40. 但训练输出远不止主输出

D-FINE 为深监督还会提供：

```text
aux_outputs
pre_outputs
enc_aux_outputs
dn_outputs
dn_pre_outputs
dn_meta
```

所以完整训练图远比：

```text
pred_logits + pred_boxes
```

复杂。

---

# 41. `aux_outputs`

6 层 decoder：

```text
Layer 5
```

作为 main output。

前 5 层：

```text
Layer 0～4
```

进入：

```text
aux_outputs
```

因此：

```text
len(aux_outputs) = 5
```

每个 auxiliary dictionary 都可以有：

```text
pred_logits
pred_boxes
pred_corners
ref_points
teacher_corners
teacher_logits
```

---

# 42. GO-LSD Teacher

第 13 课已经看到：

```text
teacher_corners
=
final layer corners

teacher_logits
=
final layer logits
```

即：

```text
Layer 5
        │
        ├────────→ Layer 0
        ├────────→ Layer 1
        ├────────→ Layer 2
        ├────────→ Layer 3
        └────────→ Layer 4
```

通过：

```text
DDF
```

蒸馏 localization distribution。

---

# 43. `pre_outputs`

D-FINE 在第一个 decoder layer 还有：

```text
traditional / preliminary head output
```

被包装为：

```text
pre_outputs
```

它让 FDR refinement 前的 preliminary prediction 也能得到直接监督。

所以 decoder 的监督不是简单：

```text
6 个 layer output
```

而是还有：

```text
preliminary prediction
```

这条支线。

---

# 44. `enc_aux_outputs`

Top-K Query selection 前的 encoder proposal 也有：

```text
encoder auxiliary outputs
```

用于直接监督：

```text
encoder candidate proposals
```

这帮助 proposal generation 本身学得更好。

---

# 45. `dn_outputs`

训练时 DN queries 经过每层 decoder 后：

```text
dn logits
dn boxes
dn corners
```

会与 normal matching query 分离。

它们进入：

```text
dn_outputs
```

并拥有自己的：

```text
DN classification
DN bbox
DN GIoU
DN FGL
DN DDF
```

等训练信号。

---

# 46. 现在进入 Criterion

模型：

```python
outputs = model(samples, targets)
```

然后：

```python
loss_dict = criterion(
    outputs,
    targets
)
```

DEIM 使用：

```text
DEIMCriterion
```

这是第 6、12、13、15、16 课之间真正的汇合点。

---

# 47. Criterion 的第一件事：取主输出

源码：

```python
outputs_without_aux = {
    k: v
    for k, v in outputs.items()
    if 'aux' not in k
}
```

然后：

```python
indices = self.matcher(
    outputs_without_aux,
    targets
)['indices']
```

即：

```text
final decoder predictions
↓
Hungarian Matcher
↓
final-layer O2O assignment
```

---

# 48. Hungarian Matching 的输入

主输出：

```text
pred_logits
[B,300,80]

pred_boxes
[B,300,4]
```

targets：

```text
B 个 GT dictionaries
```

Matcher cost：

```yaml
cost_class: 2
cost_bbox: 5
cost_giou: 2
```

所以概念上：

\[
C_{ij}
=
2C_{cls}
+
5C_{L1}
+
2C_{GIoU}
\]

然后求：

\[
\sigma^*
=
\arg\min_\sigma
\sum_i C_{i,\sigma(i)}
\]

---

# 49. 一个具体 Matching 例子

假设：

```text
Batch 0:
7 GT objects

Batch 1:
11 GT objects
```

普通 O2O final-layer matching：

```text
Batch 0:
7 matched queries

Batch 1:
11 matched queries
```

总 matched：

```text
18
```

而总 matching queries：

```text
2 × 300 = 600
```

因此单层直接 positive assignment 极其稀疏：

\[
18/600=3\%
\]

这正是后面 DEIM Dense O2O 要解决的核心背景之一。

---

# 50. 但 DEIMCriterion 并没有停在 Final-Layer Matching

源码继续：

```text
aux_outputs
pre_outputs
enc_aux_outputs
```

分别重新跑 matcher。

于是会得到：

```text
final indices
aux layer indices
pre indices
encoder indices
```

为什么？

因为不同层预测不同：

```text
class score 不同
box 不同
```

所以 Hungarian 最优 assignment 也可能不同。

---

# 51. Global Union Set

Criterion 源码明确写：

```text
Get the matching union set across all decoder layers.
```

然后调用：

```python
indices_go = self._get_go_indices(
    indices,
    indices_aux_list
)
```

这就是第 13 课 GO-LSD 的关键落地点之一。

---

# 52. 为什么 Union Set 对 Localization 有意义？

假设同一个 GT：

```text
Layer 0 → Query 13
Layer 1 → Query 13
Layer 2 → Query 88
Layer 3 → Query 88
Layer 4 → Query 201
Layer 5 → Query 201
```

如果只看最终层：

```text
Query 201
```

但跨层看：

```text
13
88
201
```

都曾经是某阶段的最佳 assignment。

因此 localization 可以利用：

```text
更丰富的候选集合
```

进行训练。

---

# 53. Classification 为什么仍不能随便用 Union Set？

Criterion 里有一个非常关键的条件：

```python
use_uni_set = (
    self.use_uni_set
    and
    (loss in ['boxes', 'local'])
)
```

也就是说：

```text
boxes
local
```

可以使用：

```text
indices_go
```

而 classification/MAL：

```text
不使用 global union
```

这正是：

```text
Localization:
更密集

Classification:
保持 O2O identity
```

的源码体现。

---

# 54. 这一行是理解 GO-LSD 的钥匙

再写一遍：

```text
loss = boxes/local
→ global union indices

loss = mal
→ ordinary layer-specific O2O indices
```

因此：

```text
一个 GT 可以给多个跨层候选提供 localization supervision
```

但不能因此把多个 Query 全部变成同一个 object 的 classification positive。

---

# 55. `num_boxes` 与 `num_boxes_go`

普通 GT normalization：

```text
num_boxes
```

来自：

```text
batch 中 GT 数量
```

而 global union localization：

```text
num_boxes_go
```

来自：

```text
global union matched pair 数量
```

因此 boxes/local loss 的 normalization 也必须跟着 union set 改变。

---

# 56. 分布式训练时为什么还要 All-Reduce？

如果使用多 GPU：

```text
GPU 0:
有 N0 个 GT

GPU 1:
有 N1 个 GT
...
```

为了让 loss normalization 在各 GPU 一致，需要：

```text
all_reduce
```

统计全局平均数量。

所以源码会：

```python
torch.distributed.all_reduce(...)
```

再除：

```text
world_size
```

这是很多 DETR criterion 中容易被初学者忽略的工程细节。

---

# 57. Criterion 的三类主 Loss

配置：

```python
losses = [
    'mal',
    'boxes',
    'local'
]
```

它们映射到：

```text
mal
→ loss_labels_mal

boxes
→ loss_boxes

local
→ loss_local
```

而 `loss_local` 内部又产生：

```text
loss_fgl
loss_ddf
```

所以最终核心五项：

```text
loss_mal
loss_bbox
loss_giou
loss_fgl
loss_ddf
```

---

# 58. Stage N：MAL

DEIM 的 classification loss：

```text
Matchability-Aware Loss
```

源码首先拿 matched prediction 的 IoU：

```text
pred box
↔
matched GT
```

然后把 IoU 变成 classification quality target。

概念：

```text
分类不是只学：
“是不是这个类别”

还学：
“这个 matched prediction 的定位质量有多好”
```

第 16 课会完整推导 MAL。

本课先理解它在整条训练链的位置。

---

# 59. MAL 的输入 Shape

```text
src_logits
[B,300,80]
```

通过 matching indices 找出 matched classes。

构造：

```text
target
[B,300,80]
```

unmatched query：

```text
全 0
```

matched query：

```text
对应类别位置非 0
```

但 target score 会结合：

```text
IoU^gamma
```

体现 matchability / localization quality。

---

# 60. Stage O：L1 Box Loss

对于 localization union 中的 matched pairs：

```text
src_boxes
[N_match,4]

target_boxes
[N_match,4]
```

计算：

\[
L_{bbox}
=
\frac{1}{N}
\sum
\left|
b-\hat b
\right|
\]

源码：

```python
F.l1_loss(
    src_boxes,
    target_boxes,
    reduction='none'
)
```

再：

```text
sum / num_boxes
```

---

# 61. 为什么 L1 不够？

两个 box：

```text
坐标 L1 很接近
```

不代表：

```text
几何重叠一定好
```

所以还需要：

```text
GIoU
```

---

# 62. Stage P：GIoU Loss

先把：

```text
cxcywh
```

转换为：

```text
xyxy
```

再：

\[
L_{GIoU}
=
1-GIoU
\]

源码逻辑：

```python
1 - diag(
    generalized_box_iou(
        pred,
        target
    )
)
```

所以：

```text
L1
```

负责坐标距离；

```text
GIoU
```

负责几何重叠结构。

---

# 63. Stage Q：FGL

第 12 课已经完整推导。

输入：

```text
pred_corners
```

对 matched localization candidates 取：

```text
[N_match,132]
```

reshape：

```text
[N_match*4,33]
```

GT box 相对于 reference point 转成：

```text
continuous distance target
```

再落到相邻两个 bins：

```text
left bin
right bin
```

做 weighted distribution focal supervision。

---

# 64. FGL 的质量权重

源码还计算：

```text
pred box ↔ GT IoU
```

然后：

```text
repeat 4
```

给：

```text
left/top/right/bottom
```

四个 edge distributions 加权。

因此 FGL 不是单纯：

```text
33-way CE
```

而是：

```text
continuous target interpolation
+
IoU weighting
+
distribution classification
```

---

# 65. Stage R：DDF

第 13 课已经完整推导。

对于 auxiliary layer：

```text
Student:
pred_corners

Teacher:
final teacher_corners
```

reshape：

```text
[B,Q,132]
→
[B*Q*4,33]
```

然后：

```text
temperature-scaled KL
```

T：

```text
5
```

---

# 66. DDF 的 Matched / Unmatched Weight

matched：

```text
weight = IoU
```

unmatched：

```text
weight = teacher confidence
```

并做：

```text
sqrt-count balancing
```

所以：

```text
DDF ≠ 普通 KL
```

而是：

```text
KL
+
quality weighting
+
positive/negative group balancing
```

---

# 67. 主输出 Final Layer 的 DDF 为什么为 0？

Final Layer 自己就是 Teacher。

如果：

```text
pred_corners
==
teacher_corners
```

就没有必要：

```text
Teacher 教自己
```

因此主要的 DDF 价值出现在：

```text
auxiliary shallow decoder layers
```

以及相应 DN deep-supervision 路径。

---

# 68. Main Loss 的权重

DEIM base：

```yaml
loss_mal:  1
loss_bbox: 5
loss_giou: 2
loss_fgl:  0.15
loss_ddf:  1.5
```

因此对于一个 loss dictionary：

```text
raw MAL
raw L1
raw GIoU
raw FGL
raw DDF
```

Criterion 会先乘各自权重。

概念：

\[
L
=
1L_{MAL}
+
5L_{bbox}
+
2L_{giou}
+
0.15L_{fgl}
+
1.5L_{ddf}
+\cdots
\]

---

# 69. “+ …” 是什么？

因为不只 final layer。

还有：

```text
aux_0
aux_1
aux_2
aux_3
aux_4
pre
enc
dn_0
dn_1
...
dn_pre
```

每个分支可能产生自己的 loss key。

所以真正训练的：

```text
loss_dict
```

会比 5 个标量大很多。

---

# 70. Auxiliary Loss Keys

Criterion 对每个：

```text
aux_outputs[i]
```

重复计算 loss。

并加 suffix：

```text
_aux_i
```

例如概念上：

```text
loss_mal_aux_0
loss_bbox_aux_0
loss_giou_aux_0
loss_fgl_aux_0
loss_ddf_aux_0
```

然后：

```text
_aux_1
...
_aux_4
```

---

# 71. `pre_outputs` Loss Keys

preliminary prediction：

```text
_pre
```

例如：

```text
loss_mal_pre
loss_bbox_pre
loss_giou_pre
...
```

具体产生哪些 key 取决于该 output dictionary 是否具有相应字段。

---

# 72. Encoder Auxiliary Loss Keys

Encoder proposals：

```text
_enc_i
```

例如：

```text
loss_mal_enc_0
loss_bbox_enc_0
loss_giou_enc_0
```

Criterion 还支持：

```text
class-agnostic encoder targets
```

如果 encoder proposal head 被设为 class-agnostic，会临时把：

```text
num_classes = 1
```

并把 target labels 变为 0。

---

# 73. DN Loss Keys

Denoising branch：

```text
_dn_i
```

例如：

```text
loss_mal_dn_0
loss_bbox_dn_0
loss_giou_dn_0
loss_fgl_dn_0
loss_ddf_dn_0
```

pre-DN：

```text
_dn_pre
```

---

# 74. 为什么 DN 不需要 Hungarian Matcher？

因为 DN query 在构造时已经知道：

```text
它来自哪个 GT
```

Criterion 调用：

```python
get_cdn_matched_indices(
    dn_meta,
    targets
)
```

直接恢复：

```text
DN query index
↔
GT index
```

而不是重新求 Hungarian 最优 assignment。

---

# 75. 一张完整 Loss Tree

```text
DEIMCriterion
│
├── Final Decoder Output
│   ├── MAL
│   ├── L1
│   ├── GIoU
│   ├── FGL
│   └── DDF (self==teacher 时为 0/不贡献)
│
├── Aux Decoder Layer 0
│   ├── MAL
│   ├── L1
│   ├── GIoU
│   ├── FGL
│   └── DDF ← final teacher
│
├── Aux Decoder Layer 1
│   └── ...
│
├── Aux Decoder Layer 2
│   └── ...
│
├── Aux Decoder Layer 3
│   └── ...
│
├── Aux Decoder Layer 4
│   └── ...
│
├── Preliminary Output
│   └── detection losses
│
├── Encoder Auxiliary Output(s)
│   └── proposal supervision
│
├── DN Decoder Outputs
│   ├── MAL
│   ├── L1
│   ├── GIoU
│   ├── FGL
│   └── DDF
│
└── DN Preliminary Output
    └── detection losses
```

---

# 76. Main / Aux 的 Matching 策略并不完全相同

对于 classification/MAL：

```text
Final:
final layer's own Hungarian indices

Aux layer i:
that layer's own Hungarian indices
```

对于：

```text
boxes
local
```

当：

```text
use_uni_set = True
```

时：

```text
global union indices
```

会被使用。

所以一句：

> “所有 loss 都使用同一套 Hungarian matching”

是错误的。

---

# 77. 为什么这个设计非常关键？

因为 DETR 同时有两个目标：

```text
1. 保持 O2O classification identity
2. 获得更密集、更稳定的 localization supervision
```

如果所有东西都用 union：

```text
classification 可能变成多个 positive query
```

如果所有东西都只用单层 O2O：

```text
大量跨层 localization knowledge 被浪费
```

DEIM/D-FINE 的 Criterion 在这里进行了任务解耦。

---

# 78. 从 Prediction 到 Loss 的完整 Tensor Flow

以 final output 为例：

```text
pred_logits
[2,300,80]
          │
          ├──────── Hungarian Matcher ───────┐
          │                                  │
pred_boxes                                  targets
[2,300,4]                                   │
          │                                  │
          └──────────────────────────────────┘
                         │
                         ▼
                      indices
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
      MAL              L1              GIoU
```

而 localization distribution：

```text
pred_corners
[2,300,132]
       │
       ├── matched/global union
       │        ↓
       │       FGL
       │
       └── teacher/student
                ↓
               DDF
```

---

# 79. 从 Feature 到 Query 的完整 Tensor Flow

```text
[2,3,640,640]
        │
        ▼
HGNetv2-B5
        │
        ├── [2,512,80,80]
        ├── [2,1024,40,40]
        └── [2,2048,20,20]
        │
        ▼
HybridEncoder
        │
        ├── [2,384,80,80]
        ├── [2,384,40,40]
        └── [2,384,20,20]
        │
        ▼
Flatten
        │
        ├── [2,6400,384]
        ├── [2,1600,384]
        └── [2,400,384]
        │
        ▼
Concat
        │
        ▼
[2,8400,384]
        │
        ▼
Encoder proposals
        │
        ▼
Top-K 300
        │
        ├── query content
        └── reference boxes
        │
        ▼
6-layer DFINE Decoder
```

---

# 80. 从 Query 到 FDR 的完整 Flow

```text
Query
[B,300,D]
    │
    ▼
Decoder Layer l
    │
    ├── Self Attention
    ├── Deformable Cross Attention
    └── FFN
    │
    ▼
hidden query
    │
    ├──────── class head
    │            ↓
    │       [B,300,80]
    │
    └──────── FDR regression head
                 ↓
           [B,300,132]
                 ↓
          softmax over 33
                 ↓
             Integral
                 ↓
          l/t/r/b distance
                 ↓
          distance2bbox
                 ↓
           [B,300,4]
```

---

# 81. 为什么 Decoder 输出的 Box 仍是 `[B,300,4]`？

因为 FDR 改变的是：

```text
内部定位表示和训练方式
```

而不是最终 detection API。

最终用户仍然只需要：

```text
class score
+
bounding box
```

所以 inference output 仍可保持：

```text
300 × class
300 × box
```

---

# 82. 训练 Forward 与推理 Forward 的最大区别

训练：

```text
targets != None
training=True
```

会出现：

```text
DN queries
aux outputs
teacher outputs
matching
criterion
loss
```

推理：

```text
targets=None
eval()
```

不需要：

```text
Hungarian Matching
FGL
DDF
MAL
DN training groups
```

而 decoder 在 eval 路径还可根据：

```text
eval_idx
```

在指定层输出/停止。

---

# 83. 推理时 PostProcessor 才把结果变成最终像素 Box

训练阶段：

```text
pred_boxes
```

通常是：

```text
normalized cxcywh
```

PostProcessor：

```text
cxcywh
→ xyxy
→ scale by original image size
```

例如：

```text
pred:
[0.5,0.5,0.25,0.5]
```

640×640：

```text
cx=320
cy=320
w=160
h=320
```

所以：

```text
x1=240
y1=160
x2=400
y2=480
```

最终：

```text
[240,160,400,480]
```

---

# 84. 训练时为什么不直接用像素坐标算 Loss？

归一化到：

```text
[0,1]
```

的好处：

```text
不同图像尺寸下尺度更统一
数值范围稳定
multi-scale training 更自然
```

例如同一个相对位置：

```text
320×320
640×640
1280×1280
```

都可以共享类似的：

```text
normalized target
```

---

# 85. `num_boxes` 为什么要 Clamp 到至少 1？

如果某个 batch 恰好：

```text
没有 GT
```

直接：

```text
loss / num_boxes
```

会除 0。

所以：

```text
num_boxes = clamp(..., min=1)
```

是必要的数值稳定处理。

---

# 86. Criterion 最后为什么 `nan_to_num`？

源码最后：

```python
losses = {
    k: torch.nan_to_num(v, nan=0.0)
    for k, v in losses.items()
}
```

这是额外的鲁棒性处理。

但调试时不要因为它存在就忽略 NaN 来源。

如果模型训练异常，应主动检查：

```text
哪一个 loss 最先出现 NaN？
box 是否非法？
augmentation 后 GT 是否异常？
AMP 是否溢出？
```

而不是把：

```text
nan_to_num
```

当成真正解决问题的方法。

---

# 87. Loss Dict 最终怎样变成一个 Scalar？

训练 loop 通常会得到：

```python
loss_dict = criterion(
    outputs,
    targets
)
```

然后：

```text
losses
=
sum(loss_dict.values())
```

由于 Criterion 已经在内部：

```text
乘 weight_dict
```

所以 loss dict 中各项已经是 weighted loss。

最终：

```python
losses.backward()
```

把梯度送回整个模型。

---

# 88. 梯度从 Loss 怎样一路回去？

例如：

```text
loss_bbox
↓
pred_boxes
↓
FDR / box refinement head
↓
decoder hidden state
↓
cross attention
↓
HybridEncoder features
↓
HGNetv2
```

MAL：

```text
loss_mal
↓
classification logits
↓
class head
↓
decoder hidden state
↓
...
```

FGL：

```text
loss_fgl
↓
pred_corners
↓
regression distribution head
↓
decoder
```

DDF：

```text
loss_ddf
↓
student shallow pred_corners
↓
shallow decoder layers
```

Teacher side：

```text
detach
```

因此 DDF 不沿 Teacher target 分支反传。

---

# 89. 为什么 Backbone 的 LR 比 Detector 小很多？

X 的 DEIM 配置：

```text
backbone lr = 5e-6
main detector lr = 5e-4
```

比例：

\[
\frac{5\times10^{-4}}{5\times10^{-6}}
=
100
\]

也就是 detector 的 base LR 是 backbone LR 的：

```text
100 倍
```

原因直觉：

```text
Backbone:
已有预训练视觉表示
→ 小步微调

Encoder/Decoder:
更任务特定
→ 需要更快学习
```

第 18 课会完整拆 optimizer param groups。

---

# 90. 一次 `backward()` 并不等于只训练 Final Layer

因为总 loss 中同时包含：

```text
main
aux
pre
encoder
DN
```

所以不同深度都能收到直接监督。

这正是：

```text
Deep Supervision
```

的工程意义。

---

# 91. 为什么 DETR 需要这么多 Deep Supervision？

Transformer decoder 是：

```text
多阶段 iterative refinement
```

如果只让最终层承担全部任务：

```text
浅层优化信号弱
proposal learning 慢
query identity 不稳定
localization refinement 难
```

于是 D-FINE / DEIM 使用：

```text
encoder supervision
pre supervision
decoder aux supervision
FDR supervision
GO-LSD
DN supervision
```

共同改善训练。

---

# 92. 现在把“架构创新”和“训练创新”分开

`deim_dfine_x`：

## Architecture / detector side

```text
HGNetv2-B5
HybridEncoder
DFINETransformer
Deformable Attention
FDR
LQE
GO-LSD-related decoder outputs
```

## DEIM training side

```text
Dense O2O augmentation
MAL
training schedule
optimizer recipe
```

Criterion 中两者会汇合。

所以：

```text
同一个 loss graph
```

并不意味着所有机制来自同一篇论文。

---

# 93. 本课非常重要的“责任边界图”

```text
HGNetv2
负责：
从像素提取 hierarchical features

HybridEncoder
负责：
统一 channel + 跨尺度融合

DFINETransformer
负责：
proposal → query → iterative object refinement

FDR
负责：
更细粒度 localization representation

GO-LSD
负责：
final localization knowledge → shallow layers

Hungarian Matcher
负责：
prediction ↔ GT 的 O2O assignment

Global Union Set
负责：
扩展 localization supervision candidates

MAL
负责：
classification / matchability-aware learning

L1 + GIoU
负责：
box geometry

FGL
负责：
GT → localization distribution

DDF
负责：
final teacher → shallow localization distribution

Optimizer
负责：
根据总梯度更新参数
```

---

# 94. 小型 PyTorch 实验 1：完整模拟 Backbone → Encoder Shape

```python
import torch

B = 2

c3 = torch.randn(
    B,512,80,80
)

c4 = torch.randn(
    B,1024,40,40
)

c5 = torch.randn(
    B,2048,20,20
)

# 模拟 input projection 后
p3 = torch.randn(
    B,384,80,80
)

p4 = torch.randn(
    B,384,40,40
)

p5 = torch.randn(
    B,384,20,20
)

print(
    c3.shape,
    c4.shape,
    c5.shape
)

print(
    p3.shape,
    p4.shape,
    p5.shape
)
```

---

# 95. 小型实验 2：Flatten 到 8400

```python
def flatten_feature(x):
    return x.flatten(2).transpose(1,2)

f3 = flatten_feature(p3)
f4 = flatten_feature(p4)
f5 = flatten_feature(p5)

print(f3.shape)
print(f4.shape)
print(f5.shape)

memory = torch.cat(
    [f3,f4,f5],
    dim=1
)

print(memory.shape)
```

预期：

```text
[2,6400,384]
[2,1600,384]
[2,400,384]
[2,8400,384]
```

---

# 96. 小型实验 3：Top-K 300

```python
B = 2
N = 8400
C = 80

scores = torch.randn(
    B,N,C
)

# 为教学简化：
# 每个 spatial proposal 取最大类别分数
proposal_score = (
    scores
    .sigmoid()
    .max(dim=-1)
    .values
)

topk = torch.topk(
    proposal_score,
    k=300,
    dim=1
)

indices = topk.indices

print(
    indices.shape
)
```

输出：

```text
[2,300]
```

---

# 97. 小型实验 4：Gather Query Features

```python
D = 384

index = (
    indices
    .unsqueeze(-1)
    .expand(-1,-1,D)
)

queries = torch.gather(
    memory,
    dim=1,
    index=index
)

print(
    queries.shape
)
```

输出：

```text
[2,300,384]
```

真实 D-FINE query preparation 比这个 toy example 更完整，但这个实验能建立：

```text
8400 → Top-K 300
```

的 Tensor 直觉。

---

# 98. 小型实验 5：模拟 6 层 Decoder 输出

```python
L = 6
B = 2
Q = 300
C = 80
REG_MAX = 32

out_logits = torch.randn(
    L,B,Q,C
)

out_boxes = torch.rand(
    L,B,Q,4
)

out_corners = torch.randn(
    L,B,Q,4*(REG_MAX+1)
)

print(
    out_logits.shape
)

print(
    out_boxes.shape
)

print(
    out_corners.shape
)
```

输出：

```text
[6,2,300,80]
[6,2,300,4]
[6,2,300,132]
```

---

# 99. 小型实验 6：构造 Main + Aux Outputs

```python
main = {
    "pred_logits":
        out_logits[-1],

    "pred_boxes":
        out_boxes[-1],

    "pred_corners":
        out_corners[-1],
}

aux = []

for i in range(L-1):

    aux.append({
        "pred_logits":
            out_logits[i],

        "pred_boxes":
            out_boxes[i],

        "pred_corners":
            out_corners[i],

        "teacher_corners":
            out_corners[-1],

        "teacher_logits":
            out_logits[-1],
    })

print(
    main["pred_boxes"].shape
)

print(
    len(aux)
)

print(
    aux[0]["teacher_corners"].shape
)
```

预期：

```text
[2,300,4]
5
[2,300,132]
```

---

# 100. 小型实验 7：模拟一个 Batch 的 GT

```python
targets = [
    {
        "labels":
            torch.tensor(
                [1,5,17,23,42,60,79]
            ),

        "boxes":
            torch.rand(
                7,4
            )
    },

    {
        "labels":
            torch.tensor(
                [2,4,7,9,11,13,20,31,45,52,70]
            ),

        "boxes":
            torch.rand(
                11,4
            )
    }
]

num_boxes = sum(
    len(t["labels"])
    for t in targets
)

print(num_boxes)
```

输出：

```text
18
```

---

# 101. 小型实验 8：O2O Positive Density

```python
B = 2
Q = 300
GT = 18

density = GT / (B*Q)

print(
    density
)
```

输出：

```text
0.03
```

即：

```text
3%
```

这会直接为第 15 课 Dense O2O 做铺垫。

---

# 102. 小型实验 9：L1 Loss

```python
pred = torch.tensor([
    [0.50,0.50,0.40,0.30]
])

gt = torch.tensor([
    [0.45,0.52,0.35,0.28]
])

l1 = torch.abs(
    pred-gt
).sum()

print(l1)
```

手算：

\[
|0.50-0.45|
+
|0.50-0.52|
+
|0.40-0.35|
+
|0.30-0.28|
\]

\[
=
0.05+0.02+0.05+0.02
=
0.14
\]

---

# 103. 小型实验 10：FDR Shape

```python
corners = torch.randn(
    2,300,132
)

dist = corners.reshape(
    2,300,4,33
)

print(
    dist.shape
)
```

输出：

```text
[2,300,4,33]
```

再：

```python
prob = dist.softmax(
    dim=-1
)

print(
    prob.sum(-1)[0,0]
)
```

应该接近：

```text
tensor([1.,1.,1.,1.])
```

即四条边各自都是一个概率分布。

---

# 104. 小型实验 11：GO-LSD Teacher

```python
teacher = (
    out_corners[-1]
    .detach()
)

student = (
    out_corners[0]
)

T = 5.0

import torch.nn.functional as F

loss_ddf_toy = F.kl_div(
    F.log_softmax(
        student.reshape(-1,33)/T,
        dim=-1
    ),
    F.softmax(
        teacher.reshape(-1,33)/T,
        dim=-1
    ),
    reduction="batchmean"
) * T**2

print(
    loss_ddf_toy.item()
)
```

这只是最小 KL。

真实 DDF 还要加：

```text
matched IoU weighting
unmatched confidence weighting
sqrt-count balancing
```

---

# 105. 小型实验 12：Weighted Total Loss

假设 Criterion 已输出：

```python
loss_dict = {
    "loss_mal":
        torch.tensor(1.20),

    "loss_bbox":
        torch.tensor(0.80),

    "loss_giou":
        torch.tensor(0.65),

    "loss_fgl":
        torch.tensor(0.22),

    "loss_ddf_aux_0":
        torch.tensor(0.31),
}
```

如果这些已经是 Criterion 加权后的值：

```python
total = sum(
    loss_dict.values()
)

print(total)
```

不要再额外乘一次：

```text
weight_dict
```

否则会重复加权。

---

# 106. 真实训练调试：最值得加的 Shape Tracer

```python
def hook(name):

    def _hook(
        module,
        inputs,
        outputs
    ):
        def shape_of(x):

            if torch.is_tensor(x):
                return tuple(x.shape)

            if isinstance(
                x,
                (list,tuple)
            ):
                return [
                    shape_of(v)
                    for v in x
                ]

            if isinstance(
                x,
                dict
            ):
                return {
                    k: shape_of(v)
                    for k,v in x.items()
                    if torch.is_tensor(v)
                }

            return type(x).__name__

        print(
            name,
            "IN:",
            shape_of(inputs),
            "OUT:",
            shape_of(outputs)
        )

    return _hook
```

注册：

```python
model.backbone.register_forward_hook(
    hook("backbone")
)

model.encoder.register_forward_hook(
    hook("encoder")
)

model.decoder.register_forward_hook(
    hook("decoder")
)
```

---

# 107. 更推荐的调试方法：只打印关键模块

不要对所有：

```text
Conv
BN
Activation
```

都 hook。

否则 X 模型会输出海量日志。

优先：

```text
DEIM
HGNetv2 stage outputs
HybridEncoder
DFINETransformer
Decoder layers
classification heads
regression heads
Criterion
```

---

# 108. 真实调试脚本：检查最终输出

```python
outputs = model(
    images,
    targets
)

for k,v in outputs.items():

    if torch.is_tensor(v):
        print(
            k,
            tuple(v.shape)
        )

    elif isinstance(v,list):
        print(
            k,
            "list length =",
            len(v)
        )

    else:
        print(
            k,
            type(v)
        )
```

重点确认：

```text
pred_logits
pred_boxes
pred_corners
aux_outputs
enc_aux_outputs
pre_outputs
dn_outputs
```

---

# 109. 真实调试脚本：检查 Loss Dict

```python
loss_dict = criterion(
    outputs,
    targets
)

for k,v in sorted(
    loss_dict.items()
):
    print(
        f"{k:30s}",
        float(v.detach())
    )
```

你应该主动分类：

```text
main
aux
pre
enc
dn
```

而不是只看：

```text
total loss
```

---

# 110. 推荐做一张 Loss 表

例如：

| Group | Loss | Value |
|---|---|---:|
| Main | MAL | ... |
| Main | bbox | ... |
| Main | GIoU | ... |
| Main | FGL | ... |
| Aux 0 | MAL | ... |
| Aux 0 | bbox | ... |
| Aux 0 | DDF | ... |
| ... | ... | ... |
| DN | MAL | ... |

这样你能快速发现：

```text
哪一个 branch 数值异常
```

---

# 111. 推荐调试：打印每层 Query 的 Box Quality

如果你有 GT 和 matching：

```text
Layer 0 mean matched IoU
Layer 1 mean matched IoU
...
Layer 5 mean matched IoU
```

这可以直观看到：

```text
iterative refinement
```

是否真的在改善 localization。

---

# 112. 推荐调试：打印每层 FDR KL 到 Final Teacher

```text
KL(L0 || L5)
KL(L1 || L5)
...
KL(L4 || L5)
```

如果训练正常，通常值得观察：

```text
更深层是否逐渐接近 final distribution
```

这不是强制单调关系，但非常适合作为学习实验。

---

# 113. 推荐调试：统计每层 Hungarian Query IDs

对同一个 GT：

```text
GT 0:
L0 → q17
L1 → q17
L2 → q83
L3 → q83
L4 → q83
L5 → q101
```

你会亲眼看到：

```text
为什么需要 global union set
```

而不是只在论文图里理解它。

---

# 114. 推荐调试：比较普通 O2O 与 Union Localization 数量

打印：

```text
num_boxes
num_boxes_go
```

例如：

```text
num_boxes = 18
num_boxes_go = 31
```

这说明：

```text
classification O2O positives
```

仍然对应 18 个 GT；

但 localization supervision 的 union candidates：

```text
31
```

更密集。

实际数字取决于当前 batch 和模型状态。

---

# 115. 推荐调试：检查 `indices_go`

不要只看数量。

打印：

```text
query index
GT index
```

确认：

```text
同一个 GT
```

是否因为不同层 matching 而对应多个 Query candidates。

这会让 GO-LSD / global optimization 的概念彻底落地。

---

# 116. 推荐调试：检查 DN Query 数

在 decoder 构造 denoising group 后打印：

```text
dn_num_group
dn_positive_idx
Q_dn
```

你会验证：

> `num_denoising=100` 不意味着每次都机械地追加恰好 100 个 query。

---

# 117. 推荐调试：检查梯度

一次 backward 后：

```python
for name,p in model.named_parameters():

    if p.grad is not None:

        if (
            "backbone" in name
            or
            "decoder" in name
        ):
            print(
                name,
                p.grad.norm().item()
            )
```

观察：

```text
backbone
encoder
decoder
classification head
regression head
```

是否都收到梯度。

---

# 118. 为什么有梯度不等于更新幅度相同？

因为 optimizer param groups 的：

```text
learning rate
weight decay
```

不同。

例如 X：

```text
backbone:
5e-6

detector:
5e-4
```

即使梯度 norm 类似，参数实际更新量也会明显不同。

---

# 119. AMP 在哪里介入？

如果：

```text
--use-amp
```

训练 loop 会在 forward 使用 autocast，并通过：

```text
GradScaler
```

缩放 loss / gradient。

概念：

```text
forward
↓ autocast
loss
↓ scale
backward
↓ unscale
gradient clipping
optimizer.step
scaler.update
```

完整 AMP/optimizer 顺序第 18、23 课再拆。

---

# 120. Gradient Clipping

runtime 中通常配置：

```text
clip_max_norm
```

DETR/Transformer 训练可能出现：

```text
gradient spike
```

所以 optimizer step 前会：

```text
clip_grad_norm_
```

限制总梯度范数。

这不是模型 forward 的一部分，而是训练 engine 的一部分。

---

# 121. EMA 在哪里？

optimizer 更新模型参数后：

```text
EMA
```

维护一个：

```text
指数移动平均模型
```

注意：

```text
EMA Teacher
≠
GO-LSD Teacher
```

再强调一次：

```text
GO-LSD Teacher:
同一次 forward 的 final decoder layer

EMA:
整个网络参数的 moving average copy
```

---

# 122. 一次完整训练 Step 的伪代码

```python
images, targets = batch

optimizer.zero_grad()

with autocast():

    outputs = model(
        images,
        targets
    )

    loss_dict = criterion(
        outputs,
        targets
    )

    loss = sum(
        loss_dict.values()
    )

scaler.scale(
    loss
).backward()

scaler.unscale_(
    optimizer
)

clip_grad_norm_(
    model.parameters(),
    max_norm
)

scaler.step(
    optimizer
)

scaler.update()

ema.update(
    model
)
```

真实仓库还包含：

```text
distributed reduction
logging
scheduler
epoch policy
checkpoint
```

但神经网络训练核心就是这条链。

---

# 123. 完整计算图：从像素到 `backward()`

```text
COCO image
[H0,W0,3]
      │
      ▼
Augmentation
      │
      ▼
Resize / normalize
      │
      ▼
[B,3,640,640]
      │
      ▼
HGNetv2-B5
      │
      ├── [B,512,80,80]
      ├── [B,1024,40,40]
      └── [B,2048,20,20]
      │
      ▼
HybridEncoder
      │
      ├── [B,384,80,80]
      ├── [B,384,40,40]
      └── [B,384,20,20]
      │
      ▼
Flatten + concat
      │
      ▼
[B,8400,384]
      │
      ▼
Encoder proposals
      │
      ▼
Top-K 300
      │
      ├── query
      └── reference box
      │
      ├──── + DN queries (training)
      │
      ▼
DFINE Decoder × 6
      │
      ├── class logits
      ├── bbox
      ├── FDR distribution
      └── refined reference
      │
      ▼
main / aux / pre / enc / dn outputs
      │
      ▼
DEIMCriterion
      │
      ├── Hungarian Matching
      │
      ├── per-layer Matching
      │
      ├── Global Union Set
      │
      ├── MAL
      │
      ├── L1
      │
      ├── GIoU
      │
      ├── FGL
      │
      └── DDF
      │
      ▼
weighted loss_dict
      │
      ▼
sum
      │
      ▼
scalar total loss
      │
      ▼
backward()
      │
      ▼
gradients
      │
      ▼
AdamW step
```

---

# 124. 完整 Shape Trace 表

以下假设：

```text
B=2
input=640×640
COCO=80 classes
```

| Stage | Tensor | Shape |
|---|---|---:|
| Input | images | `[2,3,640,640]` |
| HGNetv2 | C3 | `[2,512,80,80]` |
| HGNetv2 | C4 | `[2,1024,40,40]` |
| HGNetv2 | C5 | `[2,2048,20,20]` |
| HybridEncoder | P3 | `[2,384,80,80]` |
| HybridEncoder | P4 | `[2,384,40,40]` |
| HybridEncoder | P5 | `[2,384,20,20]` |
| Flatten | P3 tokens | `[2,6400,384]` |
| Flatten | P4 tokens | `[2,1600,384]` |
| Flatten | P5 tokens | `[2,400,384]` |
| Concat | memory | `[2,8400,384]` |
| Proposal | class | approx `[2,8400,80]` |
| Proposal | box | approx `[2,8400,4]` |
| Top-K | indices | `[2,300]` |
| Matching Query | query/reference | 300 queries |
| Decoder | layer logits | `[2,300,80]` |
| Decoder | layer boxes | `[2,300,4]` |
| FDR | layer corners | `[2,300,132]` |
| FDR reshape | distributions | `[2,300,4,33]` |
| Final | pred_logits | `[2,300,80]` |
| Final | pred_boxes | `[2,300,4]` |
| Final | pred_corners | `[2,300,132]` |
| Criterion | each loss | scalar |

> DN queries 未写进固定 shape 表，因为其实际数量取决于当前 batch 的 GT / denoising group 构造。

---

# 125. 参数流与监督流不是同一张图

参数流：

```text
image
→ backbone
→ encoder
→ decoder
→ predictions
```

监督流：

```text
GT
├→ Hungarian Matcher
├→ MAL
├→ L1
├→ GIoU
├→ FGL
└→ DN construction

Final distribution
└→ DDF teacher
```

训练时真正发生的是：

```text
前向参数流
+
反向监督流
```

同时存在。

---

# 126. 一个 GT 如何影响网络？

假设：

```text
GT #3
```

最终匹配：

```text
Query 81
```

它可能产生：

```text
MAL:
让 q81 的正确类别 score 更合理

L1:
让 q81 box 靠近 GT

GIoU:
让 q81 几何重叠更好

FGL:
让 q81 的 4×33 distributions 对准 GT 边界
```

如果其他 decoder layer 曾将：

```text
GT #3 ↔ q117
```

匹配：

```text
global union localization
```

还可能让 q117 获得 localization supervision。

---

# 127. 一个 Unmatched Query 如何影响训练？

例如：

```text
Query 250
```

没有被当前 classification matching 选中。

它仍然：

```text
MAL:
学习非目标类别 / 低 score

DDF:
可能从 final teacher 获得 localization distribution guidance
```

因此：

```text
unmatched
```

不等于：

```text
完全没有梯度。
```

---

# 128. 为什么 DEIM / D-FINE 的训练图比 YOLO 看起来复杂？

因为 DETR 系列显式解决：

```text
set prediction
assignment
query identity
iterative refinement
deep supervision
denoising
```

这些问题。

YOLO 系列也有：

```text
assignment
classification
box regression
multi-scale features
```

只是组织方式不同。

不要把：

```text
源码复杂
```

误解为：

```text
目标检测任务本身不同。
```

它们最终仍在学习：

```text
what + where
```

---

# 129. 本课最重要的 10 个数字

对于 640×640 的 `deim_dfine_x`：

```text
1. 640
   输入尺寸

2. 8 / 16 / 32
   三个 feature strides

3. 80 / 40 / 20
   三个 feature spatial sizes

4. 512 / 1024 / 2048
   HGNetv2-B5 返回通道

5. 384
   X HybridEncoder feature width

6. 6400 / 1600 / 400
   三层 spatial token 数

7. 8400
   总 spatial positions

8. 300
   matching object queries

9. 6
   decoder layers

10. 132 = 4×33
    FDR localization logits per query
```

---

# 130. 但不要死背这些数字

真正应该记住的是推导：

\[
H_l = H / stride_l
\]

\[
N_l = H_lW_l
\]

\[
N_{memory}
=
\sum_l H_lW_l
\]

\[
D_{FDR}
=
4(reg\_max+1)
\]

所以输入换成：

```text
800×800
```

你应该自己重新算，而不是说：

```text
“DEIM 就是 8400 tokens。”
```

---

# 131. 练习：800×800 时 memory 有多少 positions？

stride：

```text
8
16
32
```

所以：

```text
100×100
50×50
25×25
```

总数：

\[
10000+2500+625
=
13125
\]

因此：

```text
memory
[B,13125,384]
```

但仍然 Top-K：

```text
300 queries
```

---

# 132. 练习：如果 `reg_max=16`

每条边：

```text
17 bins
```

四边：

\[
4\times17=68
\]

所以：

```text
pred_corners
[B,Q,68]
```

而不是：

```text
132
```

---

# 133. 练习：如果类别数变成 5

```text
pred_logits
[B,300,5]
```

但：

```text
pred_boxes
[B,300,4]
```

不变。

```text
pred_corners
[B,300,132]
```

只要：

```text
reg_max=32
```

也不变。

这说明：

```text
classification dimension
```

与：

```text
localization representation dimension
```

是两套独立维度。

---

# 134. 练习：如果 `num_queries=500`

Top-K：

```text
8400 → 500
```

最终：

```text
pred_logits
[B,500,80]

pred_boxes
[B,500,4]

pred_corners
[B,500,132]
```

Hungarian matching 的 candidate 数也从：

```text
300
```

变成：

```text
500
```

但每个 GT 仍然只在 classification O2O 中对应一个 matched query。

---

# 135. 练习：如果输入 320×320

三尺度：

```text
40×40
20×20
10×10
```

总 positions：

\[
1600+400+100
=
2100
\]

所以：

```text
memory
[B,2100,384]
```

Top-K：

```text
300
```

仍可保持。

---

# 136. 练习：为什么 320 输入下 Top-K 300 占比更高？

640：

\[
300/8400
\approx3.57\%
\]

320：

\[
300/2100
\approx14.29\%
\]

所以固定 Query 数在不同输入 resolution 下对应：

```text
不同 spatial candidate coverage ratio
```

这是理解 query budget 的一个有趣视角。

---

# 137. 常见误解 1：8400 就是 Query 数

错误。

```text
8400:
multi-scale spatial positions

300:
object queries
```

---

# 138. 常见误解 2：HybridEncoder 输出 300 Queries

错误。

HybridEncoder 输出：

```text
P3/P4/P5 feature maps
```

Top-K Query selection 在：

```text
DFINETransformer
```

入口阶段完成。

---

# 139. 常见误解 3：`pred_corners` 是四个 corner coordinates

错误。

它是：

```text
4 × 33 distribution logits
```

所以：

```text
132 dims
```

---

# 140. 常见误解 4：Final Layer 只有一个 Loss

错误。

Final prediction至少参与：

```text
classification
L1
GIoU
FGL
```

同时它还作为：

```text
GO-LSD Teacher
```

影响浅层。

---

# 141. 常见误解 5：Auxiliary Loss 只是把 Final Loss 复制 5 次

错误。

每个 auxiliary layer：

```text
有自己的 prediction
可能有自己的 Hungarian matching
```

而 boxes/local 又可能使用：

```text
global union set
```

所以并非简单复制。

---

# 142. 常见误解 6：所有 Loss 都使用 Global Union

错误。

源码明确：

```text
boxes/local:
可以使用 union

MAL/classification:
保持 O2O layer-specific matching
```

---

# 143. 常见误解 7：`num_denoising=100` 就一定有 100 个 DN Queries

错误。

实际 DN query 数与：

```text
batch GT
DN group construction
```

有关。

调试时以实际 Tensor shape 为准。

---

# 144. 常见误解 8：DN Queries 也要 Hungarian Matching

错误。

它们在构造时已经知道 GT identity。

---

# 145. 常见误解 9：GO-LSD Teacher 是 EMA

错误。

```text
GO-LSD Teacher:
final decoder layer

EMA:
model parameter moving average
```

---

# 146. 常见误解 10：Loss Weight 越大，那个 Loss 就一定更重要

错误。

不同 loss 的：

```text
raw magnitude
normalization
gradient scale
```

不同。

`5 × L1` 与 `0.15 × FGL` 不能仅靠系数判断“重要性”。

---

# 147. 常见误解 11：640×640 永远产生 8400 positions

错误。

只有：

```text
输入 = 640×640
feature strides = 8/16/32
```

时成立。

multi-scale training 时会变化。

---

# 148. 常见误解 12：Backbone 输出通道就是 Decoder hidden dim

错误。

HGNetv2：

```text
512 / 1024 / 2048
```

HybridEncoder 对齐为：

```text
384 / 384 / 384
```

---

# 149. 常见误解 13：`loss_bbox` 直接监督 132 个 FDR logits

错误。

```text
loss_bbox:
监督最终 [cx,cy,w,h]

loss_fgl:
监督 FDR distributions
```

两者互补。

---

# 150. 常见误解 14：DDF 用 GT 当 Teacher

错误。

```text
FGL:
GT → distribution

DDF:
final decoder distribution → shallow distribution
```

---

# 151. 常见误解 15：PostProcessor 参与训练 Loss

通常不是。

PostProcessor 用于：

```text
inference / evaluation
```

把 normalized model outputs 转为最终检测结果。

Criterion 直接使用：

```text
raw model outputs
```

和：

```text
targets
```

计算训练 loss。

---

# 152. 本课“源码导航地图”

以后你要从头调试 `deim_dfine_x`，按这个顺序找：

```text
train.py
│
▼
YAMLConfig / workspace
│
▼
solver.fit()
│
▼
training loop
│
├── model(samples, targets)
│      │
│      ▼
│   engine/deim/deim.py
│      │
│      ├── backbone
│      │    └── engine/backbone/hgnetv2.py
│      │
│      ├── encoder
│      │    └── HybridEncoder implementation
│      │
│      └── decoder
│           └── engine/deim/dfine_decoder.py
│
└── criterion(outputs, targets)
       │
       ├── HungarianMatcher
       └── engine/deim/deim_criterion.py
```

这张导航图以后会反复使用。

---

# 153. 一次真实 Debug Session 应该怎么走？

不要从：

```text
train.py
```

一路 F11 进入每个 PyTorch 内部函数。

推荐：

### Step 1

断点：

```text
DEIM.forward
```

检查：

```text
images
targets
```

### Step 2

执行：

```python
x = self.backbone(x)
```

记录三层 shape。

### Step 3

执行：

```python
x = self.encoder(x)
```

记录三层 shape。

### Step 4

进入：

```text
DFINETransformer.forward
```

记录：

```text
memory
spatial_shapes
topk indices
reference boxes
```

### Step 5

进入：

```text
TransformerDecoder.forward
```

只跟踪一个 layer。

### Step 6

看：

```text
out_logits
out_bboxes
out_corners
```

### Step 7

退出 model，进入：

```text
DEIMCriterion.forward
```

### Step 8

打印：

```text
indices
cached_indices
indices_go
```

### Step 9

分别进入：

```text
loss_labels_mal
loss_boxes
loss_local
```

### Step 10

看最终：

```text
loss_dict
```

---

# 154. 为什么这种 Debug 顺序最好？

因为它遵循：

```text
数据形状变化的自然方向
```

而不是：

```text
文件目录顺序
```

源码学习最有效的单位不是：

```text
“今天读完这个 .py”
```

而是：

```text
“今天追完一个 Tensor 的生命周期。”
```

本课就是第一次完整追：

```text
image → loss
```

的生命周期。

---

# 155. 本课最终闭环：一个 Query 的一生

假设：

```text
Query #81
```

它最初来自：

```text
8400 encoder proposals
```

中的某个 Top-K spatial candidate。

然后：

```text
Layer 0:
读取 multi-scale features
→ class
→ box
→ FDR distribution

Layer 1:
继续 refinement

...

Layer 5:
final class
final box
final distribution
```

如果它最终匹配某个 GT：

```text
MAL
L1
GIoU
FGL
```

会直接训练它。

它的 final distribution 还可能：

```text
作为 Teacher
```

去指导浅层。

如果其他层对同一 GT 选中了不同 Query：

```text
global union
```

又会把 localization supervision 扩展给这些候选。

这就是：

> **一个 object query 在 `deim_dfine_x` 中从候选 proposal 到训练信号的完整生命周期。**

---

# 156. 本课最终闭环：一个 GT 的一生

一个 GT：

```text
label + normalized box
```

首先参与：

```text
DN query construction
```

然后 final / auxiliary / pre / encoder predictions 各自：

```text
Hungarian Matching
```

产生 assignment。

classification：

```text
保持 O2O
```

localization：

```text
可利用 global union
```

随后 GT 影响：

```text
MAL quality target
L1 target
GIoU target
FGL distribution target
```

同时 final prediction 又成为：

```text
DDF Teacher
```

影响 shallow distributions。

所以 GT 的作用远不止：

```text
“和最终 box 算一下 IoU。”
```

---

# 157. 本课最终闭环：一张图片的训练信号

```text
Image
↓
features
↓
8400 spatial candidates
↓
300 object queries
↓
6-stage refinement
↓
multiple prediction branches

GT
↓
multi-layer matching
↓
O2O classification assignments
+
global localization union
+
DN known assignments

Prediction + Target
↓
MAL
L1
GIoU
FGL
DDF
↓
weighted total loss
↓
backward
↓
update:
HGNetv2
HybridEncoder
DFINETransformer
heads
```

这就是第 14 课最核心的完整图。

---

# 158. 本课检查题

## 基础 Shape

1. 640×640 输入为什么产生 80/40/20 三个检测尺度？
2. 为什么 HGNetv2-B5 返回 512/1024/2048 channels？
3. HybridEncoder 为什么把它们统一到 384？
4. 8400 怎样算出来？
5. 8400 与 300 分别代表什么？
6. `pred_logits` 为什么是 `[B,300,80]`？
7. `pred_boxes` 为什么是 `[B,300,4]`？
8. `pred_corners` 为什么是 `[B,300,132]`？

## Decoder

9. Top-K 发生在 HybridEncoder 还是 DFINETransformer？
10. reference box 是干什么的？
11. 为什么 Deformable Attention 不对 8400 positions 做完整全局 attention？
12. `[3,6,3]` 表示什么？
13. 为什么每个 Decoder layer 都产生预测？
14. 为什么 Layer 5 可以当 GO-LSD Teacher？

## Matching / Loss

15. Hungarian Matcher 使用哪三类 cost？
16. 为什么 MAL 不使用 global union classification positives？
17. 为什么 boxes/local 可以使用 global union？
18. `num_boxes` 与 `num_boxes_go` 有什么不同？
19. L1 与 GIoU 分别解决什么？
20. FGL 监督什么？
21. DDF 监督什么？
22. FGL 和 DDF 的 Teacher 分别是谁？
23. 为什么 DDF Teacher 要 detach？
24. 为什么 DN query 不需要 Hungarian Matching？

## Training System

25. 为什么 Backbone LR 比 detector LR 小？
26. Auxiliary Loss 如何帮助浅层？
27. EMA Teacher 与 GO-LSD Teacher 有什么区别？
28. 为什么推理时不需要 Hungarian Matcher？
29. 为什么 PostProcessor 不参与训练 loss？
30. `losses.backward()` 后哪些模块能收到梯度？

---

# 159. 检查题答案速记

```text
1.
stride 8/16/32

3.
统一 Transformer feature width + feature fusion

4.
80² + 40² + 20²
= 6400 + 1600 + 400
= 8400

5.
8400 = spatial feature positions
300 = object queries

8.
4 × (reg_max+1)
= 4 × 33
= 132

9.
DFINETransformer

12.
三个 feature levels 的 sampling point counts

14.
它经历最多 refinement，并提供最终 refined distribution

15.
classification + L1 + GIoU

16.
避免破坏 O2O classification identity

17.
利用跨层更丰富的 localization candidates

18.
普通 GT normalization vs global-union localization normalization

20.
GT → FDR distribution

21.
final refined distribution → shallow distributions

22.
FGL teacher = GT geometry
DDF teacher = final decoder distribution

24.
DN 构造时已知 GT identity

25.
Backbone 已预训练，微调步幅更小

27.
GO-LSD = final decoder layer
EMA = whole-model moving average

28.
matching 只用于训练 assignment

29.
它用于 inference/evaluation output conversion

30.
由计算图决定；通常 backbone/encoder/decoder/heads 都可收到来自相应 loss 的梯度
```

---

# 160. 你现在应该形成的 `deim_dfine_x` 心智模型

不要再把它记成：

```text
一个很复杂的 DETR
```

而应该记成：

```text
                     deim_dfine_x
                           │
         ┌─────────────────┼──────────────────┐
         │                 │                  │
      Features          Queries             Training
         │                 │                  │
    HGNetv2-B5         8400 proposals      Hungarian
         │                 │               per layer
         ▼                 ▼                  │
 HybridEncoder         Top-K 300              ▼
         │                 │             Global Union
         ▼                 ▼                  │
   P3/P4/P5       DFINE Decoder ×6            │
                           │                  │
                           ├─ class ───────── MAL
                           │
                           ├─ box ─────────── L1/GIoU
                           │
                           └─ FDR ─────────── FGL
                               │
                               └───────────── DDF
                                      ▲
                                      │
                                Final Teacher
```

---

# 161. 第 9～14 课至此形成第一条完整主线

你已经依次学习：

```text
第 9 课
HGNetv2-B5
↓
图像怎样变成多尺度 CNN features

第 10 课
HybridEncoder
↓
多尺度 features 怎样被统一与融合

第 11 课
DFINETransformer
↓
8400 spatial positions 怎样变成 300 object queries

第 12 课
FDR
↓
box regression 怎样变成 fine-grained distribution refinement

第 13 课
GO-LSD
↓
final localization knowledge 怎样反向指导 shallow layers

第 14 课
Full Pipeline
↓
所有模块怎样汇入 Hungarian Matching 和最终 Loss
```

到这里：

> **`deim_dfine_x` 的“模型结构主线”已经闭环。**

下一阶段我们不再主要问：

```text
模型里面有什么层？
```

而要问：

> **DEIM 到底怎样改变训练，让 O2O DETR 收敛更快？**

---

# 162. 下一课预告

下一课：

# 第 15 课《DEIM Dense O2O——训练框架真正的核心》

我们会从本课的这个数字开始：

```text
B=2
Q=300
GT=18

single-layer positive density
=
18 / 600
=
3%
```

然后追问：

```text
为什么 O2O supervision 这么稀疏？

为什么 DETR 的强数据增强不能只理解成“防过拟合”？

Mosaic 为什么会增加 GT density？

更多 GT 为什么能产生更多 O2O positive matches？

Dense O2O 到底“Dense”在哪里？

它有没有把 O2O 改成 O2M？

DEIM 为什么推理时仍然无需 NMS？
```

并进入：

```text
configs/base/deim.yml
```

真正逐行分析：

```text
Mosaic
RandomPhotometricDistort
RandomZoomOut
RandomIoUCrop
MixUp
epoch policy
stop_epoch
```

这会正式进入本课程第二个重点：

> **DEIM / DEIMv2 的训练优化。**

---

# 163. 本课源码核对清单

本课关键事实按当前公开 DEIM / D-FINE 代码核对：

## DEIM model container

```text
engine/deim/deim.py
```

主路径：

```python
x = self.backbone(x)
x = self.encoder(x)
x = self.decoder(x, targets)
```

## HGNetv2-B5

```text
engine/backbone/hgnetv2.py
```

B5 stage channels 包含：

```text
stage2 → 512
stage3 → 1024
stage4 → 2048
```

## Base D-FINE detector config

```text
configs/base/dfine_hgnetv2.yml
```

基础设置包括：

```text
strides = [8,16,32]
num_queries = 300
num_layers = 6
num_denoising = 100
reg_max = 32
num_points = [3,6,3]
```

## X override

`deim_dfine_x` 继承 D-FINE X architecture：

```text
HGNetv2-B5
return_idx = [1,2,3]
HybridEncoder hidden_dim = 384
DFINETransformer feat_channels = [384,384,384]
reg_scale = 8
```

## DEIM X training config

```text
configs/deim_dfine/deim_hgnetv2_x_coco.yml
```

包括：

```text
backbone LR = 5e-6
detector LR = 5e-4
epochs = 58
flat_epoch = 29
no_aug_epoch = 8
```

## DEIM base

```text
configs/base/deim.yml
```

Criterion：

```text
loss_mal  = 1
loss_bbox = 5
loss_giou = 2
loss_fgl  = 0.15
loss_ddf  = 1.5
```

## Criterion

```text
engine/deim/deim_criterion.py
```

当前源码明确实现：

```text
final-layer Hungarian matching
per-aux-layer matching
global union indices
MAL
L1
GIoU
FGL
DDF
encoder aux loss
pre loss
DN loss
distributed normalization
```

并且：

```text
global union
```

只用于指定的：

```text
boxes/local
```

路径，而 classification/MAL 保持 O2O matching 逻辑。

---

# 164. 参考资料

1. DEIM official repository  
   https://github.com/Intellindust-AI-Lab/DEIM

2. DEIM model container  
   https://github.com/Intellindust-AI-Lab/DEIM/blob/main/engine/deim/deim.py

3. DEIM `DFINETransformer`  
   https://github.com/Intellindust-AI-Lab/DEIM/blob/main/engine/deim/dfine_decoder.py

4. DEIM Criterion  
   https://github.com/Intellindust-AI-Lab/DEIM/blob/main/engine/deim/deim_criterion.py

5. HGNetv2 implementation  
   https://github.com/Intellindust-AI-Lab/DEIM/blob/main/engine/backbone/hgnetv2.py

6. Base D-FINE HGNetv2 config  
   https://github.com/Intellindust-AI-Lab/DEIM/blob/main/configs/base/dfine_hgnetv2.yml

7. DEIM base training config  
   https://github.com/Intellindust-AI-Lab/DEIM/blob/main/configs/base/deim.yml

8. DEIM HGNetv2-X COCO config  
   https://github.com/Intellindust-AI-Lab/DEIM/blob/main/configs/deim_dfine/deim_hgnetv2_x_coco.yml

9. D-FINE official repository  
   https://github.com/Peterande/D-FINE

10. D-FINE X reference config  
    https://github.com/Peterande/D-FINE/blob/master/configs/dfine/dfine_hgnetv2_x_coco.yml

> **版本说明**
>
> DEIM / DEIMv2 仓库仍会更新。课程中的固定数字均说明了对应假设。  
> 当你的本地 checkout 与这里不同，应以：
>
> ```text
> 1. 本地最终 merged YAML
> 2. 当前源码
> 3. debugger / forward hook 的真实 Tensor shape
> ```
>
> 为最高依据，而不要强行让运行结果去匹配讲义中的某个历史版本数字。
