# DEIMv2 源码课程 · 第 21 课

# DEIMTransformer------DEIMv2 为什么不继续照搬 DFINETransformer

> 本课目标：沿 `deimv2_dinov3_x` 的真实数据流，把第 20
> 课得到的三尺度特征一路追到 300 个 object queries，再进入 6 层
> decoder；同时准确比较 `DEIMTransformer` 与 `DFINETransformer`：哪些
> D-FINE 核心被保留，哪些 decoder block
> 被重新设计，以及这些变化为什么适合 DEIMv2 从超轻量到 X 的模型族。

本课以当前官方仓库 `main` 分支为依据。DEIMv2
仍在更新，因此你本地学习时始终以 **当前 commit → 实际 YAML → runtime
tensor shape** 为最高依据。

------------------------------------------------------------------------

# 1. 先纠正标题可能造成的误解

"DEIMv2 为什么不继续照搬 DFINETransformer"并不等于：

``` text
D-FINE decoder 不好
→ DEIMv2 全部删除
→ 换成普通 DETR decoder
```

当前 `engine/deim/deim_decoder.py` 直接复用了 D-FINE 的关键组件：

``` python
from .dfine_decoder import MSDeformableAttention, LQE, Integral
from .dfine_utils import weighting_function, distance2bbox
```

因此更准确的关系是：

``` text
D-FINE 的精细定位核心
    ├─ MSDeformableAttention
    ├─ FDR
    ├─ Integral
    ├─ LQE
    └─ distance2bbox
              │
              ▼
        继续保留在 DEIMTransformer
              │
              +
              ▼
    decoder block 重新设计
    ├─ RMSNorm
    ├─ SwiGLUFFN
    ├─ optional Gateway
    ├─ optional head sharing
    └─ 更灵活的模型缩放
```

所以本课真正要回答的是：

> **为什么 DEIMv2 选择"保留 D-FINE 精细定位能力 + 改造 decoder
> block"，而不是把 DFINETransformer 原样照搬？**

------------------------------------------------------------------------

# 2. 第 20 课的输出就是本课输入

`deimv2_dinov3_x`，输入 640×640：

``` text
Image
[B,3,640,640]

↓ DINOv3 + STA
↓ HybridEncoder

P3 [B,256,80,80]
P4 [B,256,40,40]
P5 [B,256,20,20]

↓
DEIMTransformer
```

三尺度总位置：

\[ 80^2+40^2+20\^2 = 6400+1600+400 = 8400 \]

所以本课最核心的数字链是：

``` text
8400 spatial positions
        ↓
top-k
        ↓
300 object queries
        ↓
6 decoder layers
        ↓
[B,300,80] class logits
[B,300,4] boxes
```

------------------------------------------------------------------------

# 3. X 配置的真实关键参数

`deimv2_dinov3_x_coco.yml`：

``` yaml
DEIMTransformer:
  num_layers: 6
  eval_idx: -1
  feat_channels: [256, 256, 256]
  hidden_dim: 256
  dim_feedforward: 2048
```

它继承 `configs/base/deimv2.yml` 中的：

``` yaml
DEIMTransformer:
  feat_strides: [8, 16, 32]
  num_levels: 3
  num_queries: 300

  num_denoising: 100
  label_noise_ratio: 0.5
  box_noise_scale: 1.0

  reg_max: 32
  reg_scale: 4
  layer_scale: 1

  num_points: [3, 6, 3]
  cross_attn_method: default
  query_select_method: default

  activation: silu
  mlp_act: silu
```

因此 X 可以先记成：

``` text
hidden_dim       = 256
memory positions = 8400
queries          = 300
decoder layers   = 6
heads            = 8（constructor 默认）
levels           = 3
points/level     = [3,6,3]
reg_max          = 32
classes(COCO)    = 80
```

------------------------------------------------------------------------

# 4. 第一步：三个 feature maps 变成 memory

真实 `_get_encoder_input()` 的核心逻辑：

``` python
proj_feats = [
    self.input_proj[i](feat)
    for i, feat in enumerate(feats)
]

feat_flatten = []
spatial_shapes = []

for feat in proj_feats:
    _, _, h, w = feat.shape

    feat_flatten.append(
        feat.flatten(2).permute(0, 2, 1)
    )

    spatial_shapes.append([h, w])

feat_flatten = torch.concat(
    feat_flatten,
    1
)
```

shape：

``` text
P3 [B,256,80,80]
→ [B,6400,256]

P4 [B,256,40,40]
→ [B,1600,256]

P5 [B,256,20,20]
→ [B, 400,256]

cat
→ memory [B,8400,256]
```

同时保留：

``` text
spatial_shapes =
[
  [80,80],
  [40,40],
  [20,20]
]
```

这是后续 Multi-Scale Deformable Attention 恢复二维几何的依据。

------------------------------------------------------------------------

# 5. X 的 `input_proj` 为什么基本是 Identity？

源码：

``` python
for in_channels in feat_channels:
    if in_channels == self.hidden_dim:
        self.input_proj.append(nn.Identity())
```

X：

``` text
feat_channels = [256,256,256]
hidden_dim = 256
```

所以三个输入 level 都已经与 decoder hidden dimension 对齐：

``` text
input_proj[0] = Identity
input_proj[1] = Identity
input_proj[2] = Identity
```

这正是 STA + HybridEncoder 统一 detector interface 的好处。

------------------------------------------------------------------------

# 6. 第二步：8400 个位置怎样变成 300 个 queries？

这是 DETR 初学者最容易混淆的地方。

最原始 DETR 常用：

``` text
固定 learnable object queries
```

但 DEIMTransformer 默认：

``` python
learn_query_content=False
```

因此 query content 默认来自：

``` text
8400 个 image-dependent memory positions
→ encoder score
→ top-k=300
→ gather top-k memory
```

也就是说：

> **300 个 query
> 的数量是固定超参数，但它们的内容和初始位置是随图像变化的。**

------------------------------------------------------------------------

# 7. Encoder Score Head

默认：

``` python
self.enc_score_head = nn.Linear(
    hidden_dim,
    num_classes
)
```

COCO X：

``` text
[B,8400,256]
↓ Linear(256,80)
[B,8400,80]
```

每个 spatial position 都先得到一个粗分类预测。

------------------------------------------------------------------------

# 8. Default Top-K Query Selection

源码：

``` python
_, topk_ind = torch.topk(
    outputs_logits.max(-1).values,
    topk,
    dim=-1
)
```

shape：

``` text
[B,8400,80]
↓ max over classes
[B,8400]
↓ topk(300)
[B,300]
```

然后 gather：

``` text
topk_memory  [B,300,256]
topk_anchor  [B,300,4]
topk_logits  [B,300,80]  # training
```

所以：

``` text
8400 candidate positions
→ 300 promising candidates
```

------------------------------------------------------------------------

# 9. PyTorch 最小实验：Top-K

``` python
import torch
import torch.nn as nn

B, N, C = 2, 8400, 256
K = 300
num_classes = 80

memory = torch.randn(B, N, C)

score_head = nn.Linear(
    C,
    num_classes
)

logits = score_head(memory)
scores = logits.max(-1).values

_, topk_idx = torch.topk(
    scores,
    K,
    dim=-1
)

topk_memory = memory.gather(
    1,
    topk_idx[..., None].expand(-1, -1, C)
)

print(topk_memory.shape)
# [2,300,256]
```

------------------------------------------------------------------------

# 10. Query Content 与 Reference Box 必须分开理解

一个 decoder query 至少包含两类状态：

``` text
content
[B,300,256]

reference box
[B,300,4]
```

可以把它们理解成：

``` text
content:
“我当前知道什么？”

reference:
“我当前认为目标在哪里、多大？”
```

Decoder 后面会同时更新视觉表示和 box hypothesis。

------------------------------------------------------------------------

# 11. Anchor 怎样生成？

每个 feature cell 都先有一个归一化 anchor center。

核心：

``` python
grid_y, grid_x = torch.meshgrid(
    torch.arange(h),
    torch.arange(w),
    indexing='ij'
)

grid_xy = torch.stack(
    [grid_x, grid_y],
    dim=-1
)

grid_xy = (
    grid_xy.unsqueeze(0) + 0.5
) / torch.tensor([w, h])
```

所以一个 cell 的中心坐标被归一化到：

``` text
[0,1] × [0,1]
```

------------------------------------------------------------------------

# 12. 三个 Level 的 Anchor 宽高

源码：

``` python
wh = torch.ones_like(grid_xy) \
     * grid_size \
     * (2.0 ** lvl)
```

默认：

``` text
grid_size = 0.05
```

于是：

``` text
level 0: w=h=0.05
level 1: w=h=0.10
level 2: w=h=0.20
```

若映射到 640 pixels：

``` text
32 px
64 px
128 px
```

它们只是 coarse reference geometry，不是最终 box。

------------------------------------------------------------------------

# 13. 为什么 Anchor 要变到 Logit Space？

源码：

``` python
anchors = torch.log(
    anchors / (1 - anchors)
)
```

即：

\[ logit(p)=`\ln`{=tex}`\frac{p}{1-p}`{=tex} \]

这样 bbox head 可以预测 unconstrained residual：

\[ z'=z+`\Delta `{=tex}z \]

再：

\[ p'=`\sigma`{=tex}(z') \]

把结果映射回合法 `[0,1]`。

------------------------------------------------------------------------

# 14. Encoder 先对 Top-K Anchor 做一次修正

``` python
enc_topk_bbox_unact = (
    self.enc_bbox_head(enc_topk_memory)
    + enc_topk_anchors
)
```

shape：

``` text
enc_topk_memory
[B,300,256]

↓ bbox MLP

[B,300,4]

+ topk anchors
[B,300,4]

=
[B,300,4]
```

训练时：

``` python
enc_topk_bboxes = F.sigmoid(
    enc_topk_bbox_unact
)
```

所以 decoder 的 reference 已经是 image-conditioned coarse proposal。

------------------------------------------------------------------------

# 15. Query Content 默认也来自 Top-K Memory

源码：

``` python
if self.learn_query_content:
    content = self.tgt_embed.weight...
else:
    content = enc_topk_memory.detach()
```

默认：

``` text
learn_query_content=False
```

所以：

``` text
query content = image feature
query reference = image-conditioned proposal
```

这与"300 个完全固定的 learnable queries"有明显区别。

------------------------------------------------------------------------

# 16. 为什么 `.detach()`？

当前实现：

``` python
content = enc_topk_memory.detach()
enc_topk_bbox_unact = enc_topk_bbox_unact.detach()
```

含义是：

``` text
decoder 后续梯度
```

不会沿这条初始化路径直接回传到 query-selection feature / proposal。

但 encoder proposal 分支仍然可以通过自己的 auxiliary supervision
获得梯度。

所以：

``` text
detach
≠
encoder 永远不训练
```

------------------------------------------------------------------------

# 17. Query Position Embedding 来自当前 Box

DEIMTransformer：

``` python
self.query_pos_head = MLP(
    4,
    hidden_dim,
    hidden_dim,
    3
)
```

Decoder：

``` python
ref_points_detach = F.sigmoid(
    ref_points_unact
)

query_pos_embed = query_pos_head(
    ref_points_detach
)
```

shape：

``` text
[B,300,4]
→ MLP
[B,300,256]
```

所以 query position encoding 是 box-conditioned 的。

------------------------------------------------------------------------

# 18. 训练时还有 Denoising Queries

base：

``` text
num_denoising = 100
label_noise_ratio = 0.5
box_noise_scale = 1.0
```

训练时：

``` text
GT labels / boxes
↓ add noise
DN queries
↓
与普通 matching queries 一起进入 decoder
```

推理时没有 DN queries。

------------------------------------------------------------------------

# 19. Denoising 与 Dense O2O 不是同一个东西

``` text
Denoising:
GT → noisy queries → 学习恢复正确目标

Dense O2O:
通过训练数据/匹配机制增加有效 O2O supervision
```

二者可以同时存在，但职责不同。

------------------------------------------------------------------------

# 20. 一个 DEIMTransformer Decoder Layer 仍有三大阶段

``` text
1. Query Self-Attention
2. Multi-Scale Deformable Cross-Attention
3. Feed-Forward Network
```

DEIMv2 没有推翻 DETR decoder 的基本思想。

真正变化发生在：

``` text
Norm
FFN
Cross-attention fusion
Head sharing
模型缩放
```

------------------------------------------------------------------------

# 21. 真实 Decoder Layer 的关键结构

当前源码：

``` python
self.self_attn = nn.MultiheadAttention(
    d_model,
    n_head,
    batch_first=True
)

self.norm1 = RMSNorm(d_model)

self.cross_attn = MSDeformableAttention(
    d_model,
    n_head,
    n_levels,
    n_points
)

if use_gateway:
    self.gateway = Gate(
        d_model,
        use_rmsnorm=True
    )
else:
    self.norm2 = RMSNorm(d_model)

self.swish_ffn = SwiGLUFFN(
    d_model,
    dim_feedforward // 2,
    d_model
)

self.norm3 = RMSNorm(d_model)
```

这里已经能直接看出它不再是旧 DFINE decoder layer 的原样复制。

------------------------------------------------------------------------

# 22. 改动一：LayerNorm → RMSNorm

旧 DFINE layer 使用 LayerNorm。

DEIMTransformer 当前使用：

``` text
RMSNorm
```

LayerNorm：

\[ `\hat{x}`{=tex} = `\frac{x-\mu}`{=tex}
{`\sqrt{\sigma^2+\epsilon}`{=tex}} \]

RMSNorm 的核心：

\[ `\hat{x}`{=tex} = `\frac{x}`{=tex}
{`\sqrt{\frac{1}{d}\sum_i x_i^2+\epsilon}`{=tex}} \]

区别：

``` text
LayerNorm:
中心化 + 缩放

RMSNorm:
主要做 RMS scaling
```

不要机械记成"RMSNorm 一定更快、更准"；正确结论是 DEIMTransformer 把
decoder block 更新成更现代的 RMSNorm 风格。

------------------------------------------------------------------------

# 23. 改动二：普通 FFN → SwiGLUFFN

旧式 FFN：

``` text
Linear
→ activation
→ Linear
```

DEIMTransformer：

``` python
self.swish_ffn = SwiGLUFFN(...)
```

SwiGLU 的核心直觉：

\[ SwiGLU(x) = SiLU(xW_1)`\odot`{=tex}(xW_2) \]

然后再输出投影。

也就是：

``` text
value branch
×
learned gated branch
```

比单一路径 activation 多了一种数据依赖 gating。

------------------------------------------------------------------------

# 24. X 的 FFN 数字

X：

``` text
d_model = 256
dim_feedforward = 2048
```

当前构造：

``` python
SwiGLUFFN(
    256,
    2048 // 2,
    256
)
```

即内部 hidden 参数传入：

``` text
1024
```

具体参数量必须继续按 `deim_utils.py` 中 `SwiGLUFFN` 的实际 Linear
定义计算，不能直接按标准 FFN 猜。

------------------------------------------------------------------------

# 25. 改动三：Optional Gateway

constructor 当前默认：

``` text
use_gateway=True
```

cross-attention 后：

``` python
if self.use_gateway:
    target = self.gateway(
        target,
        self.dropout2(target2)
    )
else:
    target = target + self.dropout2(target2)
    target = self.norm2(target)
```

普通 residual：

\[ x'=x+`\Delta `{=tex}x \]

Gate 风格则更接近：

\[ x'=G(x,`\Delta `{=tex}x) \]

让模型学习：

``` text
多少旧 query state 应保留
多少 cross-attention 新信息应注入
```

具体 Gate 公式以 `deim_utils.Gate` 为准。

------------------------------------------------------------------------

# 26. 小模型甚至会关闭 Gateway

Femto / Pico 配置显式：

``` yaml
use_gateway: False
```

说明 DEIMv2 的 decoder 不是：

``` text
所有型号完全相同
```

而是允许针对 latency / parameter budget 裁剪。

------------------------------------------------------------------------

# 27. 改动四：Prediction Head Sharing

DEIMTransformer 支持：

``` text
share_bbox_head
share_score_head
```

例如 bbox head：

``` python
dec_bbox_head = MLP(...)

self.dec_bbox_head = nn.ModuleList([
    dec_bbox_head
    if share_bbox_head
    else copy.deepcopy(dec_bbox_head)
    ...
])
```

共享意味着：

``` text
多个 decoder layers
使用同一组 bbox-head 参数
```

首先减少参数量。

------------------------------------------------------------------------

# 28. 小模型为什么特别适合 Head Sharing？

Femto / Pico：

``` yaml
share_bbox_head: True
```

如果 3 层分别拥有独立 bbox head：

``` text
3 × head params
```

共享：

``` text
1 × head params
```

对几十万到几百万参数级的小模型非常有意义。

X 当前配置没有覆盖这一项，因此按 constructor 默认：

``` text
share_bbox_head=False
share_score_head=False
```

保持更高容量。

------------------------------------------------------------------------

# 29. Self-Attention 的 Shape

忽略 DN：

``` text
target
[B,300,256]

query_pos
[B,300,256]
```

源码：

``` python
q = k = target + query_pos

target2, _ = self.self_attn(
    q,
    k,
    value=target
)
```

每 head 的 attention score：

``` text
[B,8,300,300]
```

Self-Attention 让 queries 相互通信。

------------------------------------------------------------------------

# 30. 为什么 Queries 需要相互通信？

Set prediction 希望 queries：

``` text
协同分工
减少重复解释同一目标
共享全局目标上下文
```

Self-Attention 提供这种 query-to-query interaction。

------------------------------------------------------------------------

# 31. Cross-Attention 的输入

``` python
target2 = self.cross_attn(
    target + query_pos_embed,
    reference_points,
    value,
    spatial_shapes
)
```

概念 shape：

``` text
query:
[B,300,256]

reference:
[B,300,1,4]

memory:
[B,8400,256]

spatial_shapes:
[[80,80],[40,40],[20,20]]
```

但 cross-attention 并不会对 8400 个位置做完整 dense softmax。

------------------------------------------------------------------------

# 32. `num_points=[3,6,3]`

三个 feature levels：

``` text
stride 8  → 3 points
stride 16 → 6 points
stride 32 → 3 points
```

所以每个 head、每个 query：

\[ 3+6+3=12 \]

个 sampling points。

X 默认 8 heads：

\[ 300`\times8`{=tex}`\times12`{=tex} = 28,800 \]

个 query-head-point 采样单位/层。

而 dense query-memory pair：

\[ 300`\times8400`{=tex} = 2,520,000 \]

这就是 Multi-Scale Deformable Attention 对实时 DETR 的关键价值。

------------------------------------------------------------------------

# 33. 为什么 stride-16 给 6 个点？

配置确实是：

``` text
[3,6,3]
```

可以合理理解为中间尺度获得更高 sampling budget；对 DINOv3 patch16
路线，stride-16 还是 ViT patch grid 的天然对齐尺度。

但不要把这种直觉当成论文已经证明的唯一原因。`3/6/3`
为什么优于其他分配，最终要靠 ablation 验证。

------------------------------------------------------------------------

# 34. Reference Box 怎样控制采样区域？

当 reference 最后一维为 4：

``` text
(cx,cy,w,h)
```

sampling offsets 会相对当前 box 的宽高进行缩放。

直觉：

``` text
小 reference box
→ 采样更集中

大 reference box
→ 采样覆盖范围更大
```

所以 decoder 的视觉搜索区域会随着当前 box hypothesis 改变。

------------------------------------------------------------------------

# 35. 第一个 Decoder Layer 后的 `pre_bbox_head`

源码核心：

``` python
if i == 0:
    pre_bboxes = F.sigmoid(
        pre_bbox_head(output)
        + inverse_sigmoid(ref_points_detach)
    )

    pre_scores = score_head[0](output)

    ref_points_initial = pre_bboxes.detach()
```

公式：

\[ B\_{pre} = `\sigma`{=tex} `\left`{=tex}( f\_{pre}(Q_0) +
`\sigma`{=tex}\^{-1}(R_0) `\right`{=tex}) \]

这是进入 FDR 精细分布回归前的一次普通 4D refinement。

------------------------------------------------------------------------

# 36. 最重要的事实：FDR 仍然存在

当前 `TransformerDecoder` 的 docstring 仍明确描述：

``` text
Fine-grained Distribution Refinement (FDR)
```

并且实际代码：

``` python
pred_corners = (
    bbox_head[i](...)
    + pred_corners_undetach
)

inter_ref_bbox = distance2bbox(
    ref_points_initial,
    integral(pred_corners, project),
    reg_scale
)
```

因此：

> **DEIMv2 没有把 D-FINE 的 FDR 删除。**

------------------------------------------------------------------------

# 37. `reg_max=32` 的真实输出维度

四条边：

``` text
left
top
right
bottom
```

每条：

``` text
reg_max + 1 = 33 bins
```

所以：

\[ 4`\times33`{=tex}=132 \]

bbox distribution head：

``` text
[B,300,256]
→
[B,300,132]
```

不是 `[B,300,4]`。

------------------------------------------------------------------------

# 38. PyTorch 验证 FDR Shape

``` python
import torch
import torch.nn as nn

reg_max = 32

bbox_head = nn.Linear(
    256,
    4 * (reg_max + 1)
)

x = torch.randn(
    2,300,256
)

corners = bbox_head(x)

print(corners.shape)
# [2,300,132]

dist = corners.reshape(
    2,300,4,33
)

prob = dist.softmax(-1)

print(prob.shape)
# [2,300,4,33]
```

------------------------------------------------------------------------

# 39. Integral 的最小直觉

教学版：

``` python
bins = torch.arange(
    33,
    dtype=prob.dtype
)

distance = (
    prob * bins
).sum(-1)

print(distance.shape)
# [B,300,4]
```

真实 D-FINE/DEIMTransformer 的 `Integral` 还结合：

``` text
weighting_function
reg_scale
up
```

所以这个实验只是帮助理解"distribution → continuous distance"。

------------------------------------------------------------------------

# 40. FDR 的迭代累积

源码：

``` python
pred_corners = (
    bbox_head[i](
        output + output_detach
    )
    + pred_corners_undetach
)

pred_corners_undetach = pred_corners
```

因此后层不是每次从零预测 distribution，而是在上一层 distribution
prediction 基础上继续 refine。

这就是：

``` text
iterative distribution refinement
```

------------------------------------------------------------------------

# 41. LQE 也被保留

``` python
scores = score_head[i](output)

scores = self.lqe_layers[i](
    scores,
    pred_corners
)
```

LQE：

``` text
Localization Quality Estimation
```

利用定位 distribution 的统计信息修正 classification
score，使分类排序更能反映定位质量。

所以当前 DEIMTransformer 仍然明显继承 D-FINE 的精细定位思想。

------------------------------------------------------------------------

# 42. DFINETransformer 与 DEIMTransformer 的共同点

两者都保留了核心链：

``` text
multi-scale feature flatten
anchor generation
encoder proposal scoring
top-k query selection
denoising
query self-attention
MSDeformableAttention
iterative decoder
FDR distribution regression
Integral
distance2bbox
LQE
auxiliary decoder outputs
encoder auxiliary outputs
```

因此它们不是两种完全无关的 detector。

------------------------------------------------------------------------

# 43. 关键差异表

  -----------------------------------------------------------------------
  部分                    DFINETransformer 路线   DEIMTransformer
                                                  当前实现
  ----------------------- ----------------------- -----------------------
  Normalization           LayerNorm               RMSNorm

  FFN                     Linear → activation →   SwiGLUFFN
                          Linear                  

  Cross-attn 融合         residual + norm         optional Gateway；否则
                                                  residual + RMSNorm

  Query-pos head          D-FINE 原实现           4 → C → C → C 的 MLP

  FDR                     有                      保留

  LQE                     有                      保留

  Integral                有                      保留

  MSDeformableAttention   有                      保留

  Head sharing            以独立 head 为主        支持 bbox / score
                                                  sharing

  Tiny-model 裁剪         相对固定                可减 query/layer、关
                                                  gateway、共享 head

  目标                    D-FINE detector         DEIMv2 从超轻量到 X
                                                  的统一 decoder family
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 44. `eval_idx=-1` 的含义

源码：

``` python
self.eval_idx = (
    eval_idx
    if eval_idx >= 0
    else num_layers + eval_idx
)
```

X：

``` text
num_layers = 6
eval_idx = -1
```

所以：

``` text
eval_idx = 5
```

也就是使用第 6 层作为推理输出层。

------------------------------------------------------------------------

# 45. 六层 Decoder 的训练输出 Shape

忽略 DN：

``` text
out_logits
[6,B,300,80]

out_bboxes
[6,B,300,4]

out_corners
[6,B,300,132]

out_refs
[6,B,300,4]
```

最终主输出：

``` python
pred_logits = out_logits[-1]
pred_boxes  = out_bboxes[-1]
```

即：

``` text
[B,300,80]
[B,300,4]
```

------------------------------------------------------------------------

# 46. Training Output 比 Inference Output 丰富

训练：

``` text
pred_logits
pred_boxes
pred_corners
ref_points
up
reg_scale
aux_outputs
enc_aux_outputs
pre_outputs
dn_outputs
...
```

推理只需要核心：

``` text
pred_logits
pred_boxes
```

原因是训练阶段还要支持：

``` text
FDR loss
多层 auxiliary supervision
encoder proposal supervision
denoising supervision
GO-LSD / layer distillation information
```

------------------------------------------------------------------------

# 47. Auxiliary Outputs 为什么重要？

现代 DETR 不只监督最后一层。

概念：

``` text
encoder proposal
      ↓ loss

decoder layer 1
      ↓ loss

decoder layer 2
      ↓ loss

...

decoder layer 6
      ↓ main loss
```

这能让中间层也获得直接学习信号，改善 optimization。

------------------------------------------------------------------------

# 48. `pre_outputs`

当前实现还输出：

``` python
out['pre_outputs'] = {
    'pred_logits': pre_logits,
    'pred_boxes': pre_bboxes
}
```

对应第一个 decoder layer 后、FDR 精细 distribution refinement 链中的
preliminary predictions。

它与第 12～13 课学过的 D-FINE 精细定位训练机制直接相连。

------------------------------------------------------------------------

# 49. Denoising Outputs 会被拆开

训练时 DN queries 与 normal matching queries 一起跑 decoder，但最后根据
`dn_meta`：

``` text
split
```

成两套输出。

因为：

``` text
普通 query
→ 需要 set matching

DN query
→ 来源于已知 GT + noise
```

监督语义不同。

------------------------------------------------------------------------

# 50. 一次完整 X Shape Trace

忽略 DN，B=2：

``` text
P3
[2,256,80,80]

P4
[2,256,40,40]

P5
[2,256,20,20]

↓ flatten + concat

memory
[2,8400,256]

↓ enc_score_head

encoder logits
[2,8400,80]

↓ top-k=300

topk_memory
[2,300,256]

topk_anchor
[2,300,4]

↓ enc_bbox_head

initial reference logits
[2,300,4]

↓ sigmoid

reference boxes
[2,300,4]

↓ query_pos_head

query position
[2,300,256]

↓ decoder layer × 6

hidden
[2,300,256]

↓ score head

class logits
[2,300,80]

↓ FDR bbox head

corner distributions
[2,300,132]

↓ Integral + distance2bbox

boxes
[2,300,4]
```

------------------------------------------------------------------------

# 51. 为什么最终仍是 300 Boxes？

因为 architecture 定义：

``` text
num_queries = 300
```

每个 query 最终对应：

``` text
一个 classification vector
一个 bounding box
```

所以 detector 输出集合大小固定为 300 个 query predictions，再由
PostProcessor 处理。

------------------------------------------------------------------------

# 52. One-to-One 与 NMS

DEIMv2 的最终目标仍是：

``` text
one-to-one set prediction
```

希望一个 GT 主要对应一个 prediction。

因此推理设计不依赖传统 dense detector 那种：

``` text
大量重复框
→ NMS 去重
```

Dense O2O 是训练优化，不改变最终 O2O detector 的性质。

------------------------------------------------------------------------

# 53. 一个 Decoder Layer 的数学抽象

设：

\[
Q_l`\in`{=tex}`\mathbb{R}`{=tex}\^{B`\times `{=tex}N_q`\times `{=tex}C}
\]

reference boxes：

\[
R_l`\in`{=tex}`\mathbb{R}`{=tex}\^{B`\times `{=tex}N_q`\times4`{=tex}}
\]

位置 embedding：

\[ P_l=f\_{pos}(R_l) \]

Self-Attention：

\[ `\tilde `{=tex}Q_l = RMSNorm `\left`{=tex}( Q_l+ SA(Q_l+P_l)
`\right`{=tex}) \]

Cross-Attention：

\[ `\hat `{=tex}Q_l = Fuse `\left`{=tex}( `\tilde `{=tex}Q_l,
DA(`\tilde `{=tex}Q_l+P_l,R_l,M) `\right`{=tex}) \]

FFN：

\[ Q\_{l+1} = RMSNorm `\left`{=tex}( `\hat `{=tex}Q_l+
SwiGLU(`\hat `{=tex}Q_l) `\right`{=tex}) \]

其中：

``` text
M = multi-scale memory
Fuse = Gateway 或 residual + RMSNorm
DA = MSDeformableAttention
```

------------------------------------------------------------------------

# 54. Box Refinement 的数学主线

先：

\[ B\_{pre} = `\sigma`{=tex} `\left`{=tex}( f\_{pre}(Q_0) +
`\sigma`{=tex}\^{-1}(R_0) `\right`{=tex}) \]

再做 distribution refinement：

\[ D_l = f\_{bbox}\^{(l)}(Q_l) + D\_{l-1} \]

Integral：

\[ d_l=Integral(D_l) \]

最后：

\[ B_l = distance2bbox( B\_{pre}, d_l ) \]

这就是 DEIMTransformer 当前仍保留的 D-FINE/FDR 主干。

------------------------------------------------------------------------

# 55. 两种 Reference 要区分

源码中你会同时看到：

``` text
ref_points_initial
ref_points_detach
```

可以先记：

``` text
ref_points_initial
→ FDR distance decoding 的基准

ref_points_detach
→ 当前 decoder layer cross-attention 的 sampling reference
```

每层预测新 box 后：

``` python
ref_points_detach = inter_ref_bbox.detach()
```

下一层就围绕更新后的 box 再采样。

------------------------------------------------------------------------

# 56. Iterative Refinement 的直觉

``` text
Layer 1:
粗 reference
→ 看附近 multi-scale feature
→ refine box

Layer 2:
围绕更好的 box
→ 再采样
→ 再 refine

...

Layer 6:
得到最终 prediction
```

因此 box 既是：

``` text
输出
```

也是：

``` text
下一层 attention 的输入条件
```

------------------------------------------------------------------------

# 57. 为什么 Layer 数量影响速度？

每增加一层都会增加：

``` text
Self-Attention
MSDeformableAttention
SwiGLU FFN
classification head
FDR bbox head
```

所以：

``` text
更多层
→ 更多 refinement capacity
→ 更多 latency / FLOPs
```

X 用 6 层是 accuracy-capacity 选择。

------------------------------------------------------------------------

# 58. DEIMv2 小模型会缩减 Decoder

公开配置例如：

``` text
Femto:
num_queries = 150
num_layers  = 3
share_bbox_head = True
use_gateway = False

Pico:
num_queries = 200
num_layers  = 3
share_bbox_head = True
use_gateway = False
```

而 X：

``` text
num_queries = 300
num_layers  = 6
hidden_dim  = 256
```

这说明 DEIMTransformer 是可缩放的 decoder family。

------------------------------------------------------------------------

# 59. Query 数为什么是重要效率旋钮？

Self-Attention 近似：

\[ O(N_q\^2) \]

只看 query-pair 数：

``` text
300² = 90,000
200² = 40,000
150² = 22,500
```

Deformable Cross-Attention 又近似随 `N_q` 线性增长。

所以减少 query 会同时降低多个 decoder 子模块的成本。

------------------------------------------------------------------------

# 60. Query 数太少有什么风险？

如果图像非常 crowded：

``` text
真实对象数量很多
```

而：

``` text
num_queries 太少
```

理论上会限制 detector 可以表达的 prediction slots，并可能损害 recall。

所以：

``` text
num_queries
```

是速度与检测容量之间的 trade-off。

------------------------------------------------------------------------

# 61. Query Selection 的三种模式

当前源码支持：

``` text
default
one2many
agnostic
```

default：

``` text
每个 spatial position
取最大 class logit
→ top-k positions
```

one2many：

``` text
position × class flatten
→ top-k class-position pairs
→ 再映射回 position
```

agnostic：

``` text
encoder score head 只输出 1 个 objectness-like score
```

X 当前继承：

``` text
default
```

------------------------------------------------------------------------

# 62. One-to-Many Selection 的源码直觉

``` python
topk_ind = torch.topk(
    outputs_logits.flatten(1),
    topk,
    dim=-1
)[1]

topk_ind = (
    topk_ind
    // self.num_classes
)
```

这允许同一个 spatial position 因多个类别得分而进入候选。

注意它只是 query selection 方法名，不要和训练阶段 Dense O2O / O2M
概念机械混为一谈。

------------------------------------------------------------------------

# 63. Agnostic Selection

此时：

``` python
self.enc_score_head = nn.Linear(
    hidden_dim,
    1
)
```

shape：

``` text
[B,8400,1]
```

只判断：

``` text
这个位置是否值得成为 decoder query
```

而不是先决定类别。

------------------------------------------------------------------------

# 64. `cross_attn_method`

当前支持：

``` text
default
discrete
```

X 使用：

``` text
default
```

这一选项影响 MSDeformableAttention 的具体 sampling 参数化。第 7
课已经为这部分建立了基础。

------------------------------------------------------------------------

# 65. 为什么 8400 Memory 不等于 8400 Queries？

严格区分：

``` text
Memory:
8400 image positions

Queries:
300 object slots
```

Cross-Attention 是：

``` text
300 queries
→ 稀疏读取
8400-position multi-scale memory
```

这是整个 DETR decoder 最重要的 shape 区分之一。

------------------------------------------------------------------------

# 66. Two-Stage-Like ≠ Faster R-CNN Two-Stage

DEIMTransformer 有：

``` text
encoder proposals
→ top-k
→ decoder
```

因此具有 two-stage-like query initialization。

但它不是：

``` text
RPN
→ RoIAlign
→ per-RoI head
```

它仍然是：

``` text
Transformer decoder
set prediction
Hungarian O2O matching
```

------------------------------------------------------------------------

# 67. 为什么 Top-K Initialization 有利于收敛？

如果 query 从完全无图像信息的随机 state 开始：

``` text
先学“去哪里找”
再学“是什么”
再学“框多大”
```

Top-k memory 已经给出：

``` text
疑似目标 feature
+
coarse spatial proposal
```

所以 decoder 初始条件更好。

------------------------------------------------------------------------

# 68. 但为什么还需要 6 层 Decoder？

因为 top-k 只是粗候选。

Decoder 仍需要：

``` text
query-to-query coordination
multi-scale feature sampling
classification refinement
fine localization
iterative box update
```

所以 query selection 不能替代 decoder。

------------------------------------------------------------------------

# 69. 从一个 Query 的视角走完整流程

``` text
8400 memory 中某个高分位置
        ↓
被选进 top-300
        ↓
得到 content feature
        ↓
得到 coarse anchor / proposal
        ↓
生成 box-conditioned query position
        ↓
Layer 1 self-attention
        ↓
围绕 box 去 P3/P4/P5 采样
        ↓
更新 feature
        ↓
FDR refine box
        ↓
Layer 2 围绕新 box 再采样
        ↓
...
        ↓
Layer 6
        ↓
class + final box
```

------------------------------------------------------------------------

# 70. 从 Box 的视角走完整流程

``` text
grid anchor
↓
encoder bbox correction
↓
decoder initial reference
↓
pre_bbox refinement
↓
FDR layer 1
↓
FDR layer 2
↓
...
↓
FDR layer 6
↓
final bbox
```

所以"decoder 输出 box"背后其实是一整条 refinement chain。

------------------------------------------------------------------------

# 71. 从 Loss 的视角走完整流程

训练并不是只监督：

``` text
最后一层 300 predictions
```

还包括：

``` text
encoder proposal outputs
decoder intermediate outputs
pre outputs
denoising outputs
FDR/localization outputs
```

再结合：

``` text
Hungarian Matching
MAL
Dense O2O
```

形成完整训练系统。

------------------------------------------------------------------------

# 72. Mixed Precision 的一个工程细节

当前 decoder FFN 后有：

``` python
target = self.norm3(
    target.clamp(
        min=-65504,
        max=65504
    )
)
```

65504 接近 FP16 最大有限值。

这说明：

``` text
AMP 并不只是 trainer 外面包 autocast
```

高性能模型内部也可能加入数值稳定性处理。

这与第 18 课 AMP 直接连接起来。

------------------------------------------------------------------------

# 73. Anchor `valid_mask`

inverse sigmoid 在 0/1 附近会趋向无穷：

\[ logit(0)=-`\infty`{=tex} \]

\[ logit(1)=+`\infty`{=tex} \]

所以源码会检查：

``` text
eps < anchor < 1-eps
```

无效 anchor 被 mask，避免数值问题。

------------------------------------------------------------------------

# 74. 为什么 BBox Head 最后一层常初始化为 0？

当前源码对多个 bbox head 最后一层做 zero initialization。

直觉：

``` text
初始 delta ≈ 0
```

因此模型一开始更接近：

``` text
沿用 anchor/reference
```

而不是随机产生巨大 box 偏移。

这是 iterative refinement 常见的稳定初始化思路。

------------------------------------------------------------------------

# 75. Classification Bias Initialization

源码使用：

``` text
bias_init_with_prob(0.01)
```

让初始 object/class probability 较低。

原因：

``` text
8400 spatial positions
```

绝大多数不是目标中心。

合理的低 positive prior 有助于稳定训练早期。

------------------------------------------------------------------------

# 76. `convert_to_deploy()`

部署模式会裁掉：

``` text
推理不需要的 decoder heads / layers
```

例如只保留 `eval_idx` 需要的结构，把部分 head 替换为 `Identity`。

这体现：

``` text
training graph
```

和：

``` text
deployment graph
```

可以不同。

------------------------------------------------------------------------

# 77. 为什么不能用"DEIMTransformer 参数更少"一句话总结？

因为 X 仍然有：

``` text
6 decoder layers
256 hidden dim
2048 FFN setting
300 queries
FDR
LQE
MSDeformableAttention
```

它不是单纯"砍掉功能"。

更准确的关键词：

``` text
modernized
configurable
scalable
deployment-aware
```

------------------------------------------------------------------------

# 78. 标题问题的五层答案

第一层：

``` text
DEIMv2 的任务已经从单一 D-FINE detector
扩展为从 ultra-light 到 DINOv3-X 的模型族。
```

第二层：

``` text
D-FINE 最有价值的精细定位机制被保留，
没有必要全部重写。
```

第三层：

``` text
decoder block 本体升级为 RMSNorm + SwiGLU，
并支持 Gateway。
```

第四层：

``` text
小模型可以减少 queries/layers，
共享 bbox head，
关闭 gateway。
```

第五层：

``` text
DEIMTransformer 只是系统的一部分；
最终性能还依赖 STA、HybridEncoder、
Dense O2O v2、MAL、augmentation 和 optimizer recipe。
```

------------------------------------------------------------------------

# 79. 推荐的真实源码阅读顺序

不要机械从文件第一行读到最后一行。

按数据流：

``` text
1. DEIMTransformer.__init__
2. DEIMTransformer.forward
3. _get_encoder_input
4. _generate_anchors
5. _get_decoder_input
6. _select_topk
7. TransformerDecoder.forward
8. TransformerDecoderLayer.forward
9. LQE / Integral
10. denoising
11. output packaging
```

这样每一段代码都有"它为什么存在"的上下文。

------------------------------------------------------------------------

# 80. ShapeTracer 应重点 Hook 哪些模块？

``` text
decoder.enc_score_head
decoder.enc_bbox_head
decoder.query_pos_head

decoder.decoder.layers.0
...
decoder.decoder.layers.5

decoder.dec_score_head.*
decoder.dec_bbox_head.*
```

建议记录：

``` text
module name
input shape
output shape
dtype
requires_grad
parameter count
```

第 22 课会把它做成完整脚本。

------------------------------------------------------------------------

# 81. 很值得做的实验：Top-K Query 来自哪个 Level？

8400 index 边界：

``` text
P3:
0 ... 6399

P4:
6400 ... 7999

P5:
8000 ... 8399
```

所以：

``` python
p3 = topk_idx < 6400

p4 = (
    (topk_idx >= 6400)
    & (topk_idx < 8000)
)

p5 = topk_idx >= 8000

print(
    p3.sum(),
    p4.sum(),
    p5.sum()
)
```

在 100 张图上统计，可以直观看到 decoder query initialization
对不同尺度的偏好。

------------------------------------------------------------------------

# 82. 实验：观察六层 Box Refinement

对一个最终匹配成功的 query，保存：

``` text
Layer 1 bbox
Layer 2 bbox
...
Layer 6 bbox
GT bbox
```

把六个框画在同一张图上。

这会让：

``` text
iterative refinement
```

从抽象概念变成肉眼可见的轨迹。

------------------------------------------------------------------------

# 83. 实验：LQE 前后 Score

Hook：

``` text
score_head[i](output)
```

以及：

``` text
lqe_layers[i](score, pred_corners)
```

比较：

``` text
定位很准的 query
定位很差的 query
```

分类 score 如何变化。

------------------------------------------------------------------------

# 84. Ablation 1：Decoder Layers

``` text
6 → 4 → 3
```

记录：

``` text
AP
AP50
AP75
GFLOPs
latency
memory
```

合理预期：

``` text
层数减少 → latency 下降
```

但 AP 具体变化必须实测。

------------------------------------------------------------------------

# 85. Ablation 2：Query 数

``` text
300
200
150
```

记录：

``` text
AP
AR
crowded-scene recall
latency
```

这是理解"小模型为什么减少 queries"的最好实验。

------------------------------------------------------------------------

# 86. Ablation 3：Sampling Points

比较：

``` text
[3,6,3]
[4,4,4]
[2,4,2]
```

记录：

``` text
APs
APm
APl
latency
```

不要只看总 AP。

------------------------------------------------------------------------

# 87. Ablation 4：Gateway

``` text
use_gateway=True
use_gateway=False
```

比较：

``` text
parameters
latency
training stability
AP
```

不能因为 Femto/Pico 关闭 gateway 就断言 X 也不需要。

------------------------------------------------------------------------

# 88. Ablation 5：BBox Head Sharing

``` text
share_bbox_head=False
share_bbox_head=True
```

先比较：

``` text
parameter count
checkpoint size
```

再训练比较 AP。

参数减少是确定的；latency 是否显著变化要实测。

------------------------------------------------------------------------

# 89. Ablation 6：RMSNorm vs LayerNorm

替换时必须控制：

``` text
seed
batch size
optimizer
LR
schedule
model width/depth
```

否则无法把差异归因于 normalization。

------------------------------------------------------------------------

# 90. Ablation 7：SwiGLU vs Standard FFN

尽量匹配参数量和计算量。

否则：

``` text
SwiGLU 更好
```

可能只是因为：

``` text
容量不同
```

而不是 gating 本身。

------------------------------------------------------------------------

# 91. 最常见的误解

``` text
误解 1:
DEIMTransformer 完全删除 D-FINE。
→ 错。

误解 2:
DEIMTransformer 不再使用 FDR。
→ 错。

误解 3:
bbox head 只输出 4 个数。
→ 错，FDR 下是 4×(reg_max+1)。

误解 4:
8400 memory positions = 8400 queries。
→ 错。

误解 5:
300 queries 默认都是固定 learnable embeddings。
→ 错，默认 content 来自 top-k memory。

误解 6:
num_points=[3,6,3] 表示总共 3 个 sampling points。
→ 错，总共 12 points/head/query。

误解 7:
cross-attention 对 8400 positions 做完整 softmax。
→ 错，使用 MSDeformableAttention。

误解 8:
SwiGLU 就是 SiLU。
→ 错，SwiGLU 是 gated FFN。

误解 9:
Gateway 就是 residual add。
→ 错。

误解 10:
所有 DEIMv2 型号都开启 Gateway。
→ 错。

误解 11:
所有型号都是 300 queries。
→ 错。

误解 12:
所有型号都是 6 decoder layers。
→ 错。

误解 13:
Denoising = Dense O2O。
→ 错。

误解 14:
只监督最后一层 decoder。
→ 错。

误解 15:
DEIMv2 提升全部来自 DEIMTransformer。
→ 错。
```

------------------------------------------------------------------------

# 92. X Shape 总表

  阶段                   Shape
  ---------------------- -----------------
  P3                     `[B,256,80,80]`
  P4                     `[B,256,40,40]`
  P5                     `[B,256,20,20]`
  Memory                 `[B,8400,256]`
  Encoder logits         `[B,8400,80]`
  Top-k indices          `[B,300]`
  Query content          `[B,300,256]`
  Initial reference      `[B,300,4]`
  Query position         `[B,300,256]`
  Decoder hidden         `[B,300,256]`
  Class logits           `[B,300,80]`
  FDR corners            `[B,300,132]`
  Box                    `[B,300,4]`
  6-layer logits stack   `[6,B,300,80]`
  6-layer boxes stack    `[6,B,300,4]`

训练时加入 DN queries 后，query 维度会临时增大，输出包装阶段再 split。

------------------------------------------------------------------------

# 93. 最终数据流图

``` text
HybridEncoder
│
├─ P3 [B,256,80,80]
├─ P4 [B,256,40,40]
└─ P5 [B,256,20,20]
        │
        ▼
flatten + concat
        │
        ▼
memory [B,8400,256]
        │
        ├──────────────┐
        ▼              ▼
enc_score_head    generate anchors
[B,8400,80]       [1,8400,4]
        │              │
        └──────┬───────┘
               ▼
           top-k=300
               │
      ┌────────┴─────────┐
      ▼                  ▼
query content       reference box
[B,300,256]         [B,300,4]
      │                  │
      └────────┬─────────┘
               ▼
         query_pos_head
               │
               ▼
         [B,300,256]
               │
               ▼
   ┌───────────────────────────┐
   │ Decoder Layer × 6         │
   │ Self-Attention            │
   │ ↓                         │
   │ MSDeformableAttention     │
   │ [3,6,3] sampling points   │
   │ ↓                         │
   │ Gateway / RMSNorm         │
   │ ↓                         │
   │ SwiGLUFFN                 │
   │ ↓                         │
   │ FDR + Integral + LQE      │
   │ ↓                         │
   │ update reference box      │
   └────────────┬──────────────┘
                ▼
       pred_logits [B,300,80]
       pred_boxes  [B,300,4]
```

------------------------------------------------------------------------

# 94. DFINE → DEIMTransformer 记忆图

``` text
DFINETransformer
       │
       ├─ LayerNorm
       ├─ standard FFN
       ├─ residual + norm
       │
       └──────────────┐
                      │
              D-FINE localization core
              FDR / LQE / Integral
                      │
                      ▼
DEIMTransformer
       │
       ├─ RMSNorm
       ├─ SwiGLUFFN
       ├─ optional Gateway
       ├─ optional head sharing
       ├─ scalable queries/layers
       │
       └─ STILL keeps:
          MSDeformableAttention
          FDR
          LQE
          Integral
          distance2bbox
```

------------------------------------------------------------------------

# 95. 检查题

1.  8400 从哪里来？
2.  X 有多少 queries？
3.  默认 query content 是否为固定 learnable embedding？
4.  默认 query content 从哪里来？
5.  encoder score head 的 shape 是什么？
6.  default top-k 按什么排序？
7.  anchor 用哪四个值表示？
8.  三个 level 默认 anchor wh 分别是多少？
9.  为什么使用 inverse sigmoid？
10. query position embedding 从什么生成？
11. DN 与 Dense O2O 是否相同？
12. 一个 decoder layer 的三大阶段是什么？
13. DEIMTransformer 当前使用什么 Norm？
14. FFN 使用什么结构？
15. Gateway 位于哪里？
16. `[3,6,3]` 是什么意思？
17. 每 head/query 共采多少点？
18. `reg_max=32` 时 FDR head 输出多少维？
19. DEIMTransformer 是否保留 FDR？
20. 是否保留 LQE？
21. 是否保留 Integral？
22. `eval_idx=-1`、6 层时对应哪层？
23. X 默认共享 bbox head 吗？
24. 为什么 Femto/Pico 会共享 bbox head？
25. 为什么小模型减少 query 数？
26. 8400 memory 与 300 queries 的区别是什么？

------------------------------------------------------------------------

# 96. 检查题答案

``` text
1. 80²+40²+20²。
2. 300。
3. 默认不是。
4. 从 8400 memory 中按 encoder score 选出的 top-k features。
5. [B,8400,80]。
6. 每个 position 的最大 class logit。
7. cx,cy,w,h。
8. 0.05、0.10、0.20。
9. 为了在 unconstrained logit space 做 residual refinement。
10. 当前 reference box 经 query_pos_head MLP。
11. 不是。
12. Self-Attention → MSDeformable Cross-Attention → FFN。
13. RMSNorm。
14. SwiGLUFFN。
15. Cross-attention 后，用于融合旧 target 与新 cross-attention feature。
16. 三个 feature levels 分别采 3、6、3 个点。
17. 12。
18. 4×33=132。
19. 保留。
20. 保留。
21. 保留。
22. index 5，即第 6 层。
23. 当前默认不共享。
24. 主要为了降低小模型参数量/容量成本。
25. 降低 self-attention、cross-attention 和 prediction heads 的计算。
26. 8400 是视觉 memory positions；300 是 object prediction slots。
```

------------------------------------------------------------------------

# 97. 课后实验

完成以下实验：

``` text
A. 自己实现 P3/P4/P5 → [B,8400,256]
B. 自己实现三尺度 anchor generation
C. 自己实现 default top-k query selection
D. 实现 one2many / agnostic selection 并比较
E. 统计真实 X 的 top-300 queries 来自 P3/P4/P5 的比例
F. 保存六层 bbox，画 iterative refinement trajectory
G. Hook LQE 前后 classification score
H. 比较 DFINETransformer / DEIMTransformer 模块参数量
I. 6→4→3 decoder layers ablation
J. 300→200→150 queries ablation
K. [3,6,3] vs [4,4,4] sampling ablation
L. Gateway on/off
M. bbox head shared/unshared
```

------------------------------------------------------------------------

# 98. 你必须能口述的答案

学完本课，应能不看讲义解释：

``` text
DEIMTransformer 先把 HybridEncoder 的 80×80、40×40、20×20
三尺度 feature flatten 成 8400 个 memory positions。

它通过 encoder classification score 从 8400 个位置选择 top-300，
把对应 memory feature 作为默认 query content，
同时根据 multi-scale grid anchors 和 encoder bbox head
得到 300 个 coarse reference boxes。

每个 decoder layer 先让 300 queries 做 self-attention，
再围绕当前 reference box 使用 multi-scale deformable attention
从三个 feature levels 稀疏采样，X 的 sampling budget 是 [3,6,3]。

DEIMTransformer 没有删除 D-FINE 的 FDR。
它仍用 reg_max=32 的 4×33 distribution、
Integral、distance2bbox 和 LQE 做精细定位；
真正明显改变的是 decoder block：
LayerNorm 改成 RMSNorm，普通 FFN 改成 SwiGLUFFN，
cross-attention 后可以使用 Gateway，
并增加 prediction-head sharing 等模型缩放能力。

因此 DEIMTransformer 不是抛弃 D-FINE，
而是保留 D-FINE 精细定位核心，
同时把 decoder 改造成更适合 DEIMv2 从超轻量到 X
统一扩展的实时检测 decoder family。
```

------------------------------------------------------------------------

# 99. 下一课预告

第 22 课：

# 《完整追踪 deimv2_dinov3_x------640×640 的每一个 Tensor》

下一课第一次把前 21 课全部合流：

``` text
[B,3,640,640]
↓
DINOv3 patch embedding
↓
ViT blocks
↓
STA
↓
HybridEncoder
↓
8400 memory
↓
top-300 queries
↓
DEIMTransformer layer 1~6
↓
FDR
↓
pred_logits / pred_boxes
↓
PostProcessor
↓
最终像素坐标 boxes
```

并给出可直接用于本地仓库的 `ShapeTracer` / forward-hook 调试脚本。

------------------------------------------------------------------------

# 100. 源码核对清单

本课重点对应：

``` text
configs/base/deimv2.yml
configs/deimv2/deimv2_dinov3_x_coco.yml
engine/deim/deim_decoder.py
engine/deim/dfine_decoder.py
engine/deim/deim_utils.py
engine/deim/denoising.py
```

当前官方源码可直接确认：

``` text
DEIMTransformer 使用 RMSNorm
DEIMTransformer 使用 SwiGLUFFN
支持 optional Gateway
支持 share_bbox_head / share_score_head
复用 D-FINE 的 MSDeformableAttention / LQE / Integral
使用 weighting_function / distance2bbox
保留 FDR-style distribution refinement
base num_queries=300
base num_denoising=100
base reg_max=32
base num_points=[3,6,3]
X num_layers=6
X hidden_dim=256
X dim_feedforward=2048
```

------------------------------------------------------------------------

# 101. 版本阅读原则

``` text
runtime tensor.shape
>
当前 commit 的 Python source
>
实际加载的 YAML
>
checkpoint 对应 config
>
论文 / README
>
本课程讲义
```

特别要区分：

``` text
Python constructor default
```

和：

``` text
具体模型 YAML override
```

------------------------------------------------------------------------

# 102. 参考资料

DEIMv2 official repository:

``` text
https://github.com/Intellindust-AI-Lab/DEIMv2
```

DEIMv2 base config:

``` text
https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/configs/base/deimv2.yml
```

DEIMv2 DINOv3-X config:

``` text
https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/configs/deimv2/deimv2_dinov3_x_coco.yml
```

DEIMTransformer source:

``` text
https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/engine/deim/deim_decoder.py
```

DEIM / D-FINE-derived decoder for comparison:

``` text
https://github.com/Intellindust-AI-Lab/DEIM/blob/main/engine/deim/dfine_decoder.py
```

------------------------------------------------------------------------

# 103. 本课一句话总结

``` text
DEIMTransformer 并不是把 DFINETransformer 的精细定位能力删掉后
换成一个更普通的 DETR decoder；

它保留了 MSDeformableAttention、FDR、Integral、LQE、
distance2bbox 和 iterative refinement，
同时用 RMSNorm、SwiGLU、可选 Gateway、可共享 prediction heads
以及可调 query/layer/hidden dimension，
把 decoder 改造成适合 DEIMv2 从超轻量模型一直扩展到 DINOv3-X
的统一实时检测 decoder family。
```
