# DEIMv2 源码课程 · 第 22 课

# 完整追踪 `deimv2_dinov3_x`------640×640 的每一个 Tensor

> **本课定位**：这是前 21 课第一次真正"合流"。我们不再单独学习
> CNN、ViT、STA、HybridEncoder、Query、Deformable Attention 或
> FDR，而是把一张 `640×640` 图像从模型入口一路追踪到最终像素坐标检测框。
>
> 本课以官方 DEIMv2 当前 `main` 分支和 `deimv2_dinov3_x_coco.yml`
> 为依据。对于 **STA 内部的临时张量**，不同 commit / DINOv3
> 实现可能略有变化，因此本课严格区分：
>
> -   **配置与模块接口可以确定的 shape**
> -   **由 640、stride、hidden_dim 可以数学确定的 shape**
> -   **必须通过你本地当前 commit 的 forward hook 才能最终确认的内部临时
>     shape**
>
> 这是源码学习中非常重要的习惯：**不为了"看起来完整"而猜 shape。**

------------------------------------------------------------------------

# 0. 本课最终要得到什么？

输入：

``` text
image
[B, 3, 640, 640]
```

最终：

``` text
labels
[B, 300]

boxes
[B, 300, 4]   # xyxy，原图像素坐标

scores
[B, 300]
```

中间主干：

``` text
[B,3,640,640]
        │
        ▼
DINOv3 + STA
        │
        ├── [B,256,80,80]   stride 8
        ├── [B,256,40,40]   stride 16
        └── [B,256,20,20]   stride 32
        │
        ▼
HybridEncoder
        │
        ├── [B,256,80,80]
        ├── [B,256,40,40]
        └── [B,256,20,20]
        │
        ▼
flatten + concat
        │
        ▼
[B,8400,256]
        │
        ▼
encoder proposal scoring
        │
        ▼
top-k = 300
        │
        ├── query content [B,300,256]
        └── reference box [B,300,4]
        │
        ▼
DEIMTransformer × 6
        │
        ├── logits [B,300,80]
        ├── FDR distribution [B,300,132]
        └── boxes [B,300,4]
        │
        ▼
PostProcessor
        │
        ▼
300 detections / image
```

本课最重要的目标不是背这张图，而是你能够回答：

> **每一个数字为什么是这个数字？**

------------------------------------------------------------------------

# 1. 先锁定我们研究的模型

官方 X 配置的关键部分：

``` yaml
DEIM:
  backbone: DINOv3STAs

DINOv3STAs:
  name: dinov3_vits16plus
  interaction_indexes: [5, 8, 11]
  finetune: True
  conv_inplane: 64
  hidden_dim: 256

HybridEncoder:
  in_channels: [256, 256, 256]
  hidden_dim: 256
  dim_feedforward: 1024
  expansion: 1.25
  depth_mult: 1.37

DEIMTransformer:
  num_layers: 6
  eval_idx: -1
  feat_channels: [256, 256, 256]
  hidden_dim: 256
  dim_feedforward: 2048
```

base config 继续提供：

``` text
feat_strides       = [8,16,32]
num_levels         = 3
num_queries        = 300
num_denoising      = 100
reg_max            = 32
reg_scale          = 4
num_points         = [3,6,3]
query_select_method= default
cross_attn_method  = default
```

COCO：

``` text
num_classes = 80
```

------------------------------------------------------------------------

# 2. 模型顶层 forward 其实非常短

从模型容器的角度，整个推理过程可以压缩成：

``` python
x = self.backbone(x)
x = self.encoder(x)
x = self.decoder(x)
x = self.postprocessor(x, orig_target_sizes)
```

也就是：

``` text
Image
↓
DINOv3STAs
↓
HybridEncoder
↓
DEIMTransformer
↓
PostProcessor
```

真正困难的是：

> `x` 在每一步究竟是什么？

本课就是把这个 `x` 完全展开。

------------------------------------------------------------------------

# 3. Shape 记号统一

后面统一使用：

``` text
B   = batch size
C   = channel / hidden dimension
H,W = feature-map spatial size
N   = token / spatial-position count
Q   = query count
L   = feature level count
K   = class count
```

X 模型：

``` text
C = 256
Q = 300
L = 3
K = 80
```

------------------------------------------------------------------------

# 4. 输入 Tensor

假设：

``` python
images.shape
```

为：

``` text
[B,3,640,640]
```

例如：

``` text
B = 2

images:
[2,3,640,640]
```

元素数量：

\[ 2`\times3`{=tex}`\times640`{=tex}`\times640`{=tex} = 2,457,600 \]

FP32 原始 tensor 约：

\[ 2,457,600`\times4`{=tex} `\approx9.83`{=tex}`\text{ MB}`{=tex} \]

FP16 约一半。

注意这只是输入 tensor，不代表整个模型显存。

------------------------------------------------------------------------

# 5. 640×640 进入 patch-size-16 ViT

X 使用：

``` text
dinov3_vits16plus
```

名字中的：

``` text
16
```

对应 patch size 16。

空间 patch grid：

\[ 640/16=40 \]

所以：

``` text
640×640
↓ patchify
40×40
```

patch 数：

\[ 40`\times40`{=tex}=1600 \]

因此最重要的 ViT 空间 token 数：

``` text
Npatch = 1600
```

------------------------------------------------------------------------

# 6. 一个 Patch 包含多少原始数值？

一个 RGB patch：

``` text
3 × 16 × 16
```

因此：

\[ 3`\times16`{=tex}`\times16`{=tex}=768 \]

个输入数值。

Patch embedding 可以直觉理解为：

``` text
768-dimensional raw patch
→ learned projection
→ DINOv3 token embedding
```

------------------------------------------------------------------------

# 7. 不要把"1600 patch tokens"与"ViT 序列长度"永远画等号

DINO 系列实现可能包含额外 special/register tokens。

因此：

``` text
spatial patch tokens = 1600
```

是由输入和 patch size **确定**的。

但某个具体 DINOv3 block 内：

``` text
sequence_length
```

是否严格等于 1600，必须以当前 DINOv3 实现为准。

这就是本课第一个源码阅读纪律：

``` text
可以由数学确定的 → 直接推导
依赖实现细节的 → runtime 验证
```

------------------------------------------------------------------------

# 8. DINOv3 block 5 / 8 / 11

X：

``` yaml
interaction_indexes: [5,8,11]
```

说明 STA 会在 DINOv3 的多个深度阶段进行交互。

概念：

``` text
Patch embedding
    ↓
ViT blocks
    ↓
block 5  ← interaction
    ↓
block 8  ← interaction
    ↓
block 11 ← interaction
```

为什么不是只取最后一层？

因为检测既需要：

``` text
深层语义
```

也需要：

``` text
细粒度空间结构
```

STA 的作用就是让 ViT representation 与检测所需的 spatial hierarchy
建立联系。

------------------------------------------------------------------------

# 9. 这里最容易犯的错误

错误写法：

``` text
block 5  = stride 8
block 8  = stride 16
block 11 = stride 32
```

这会把两个不同概念混在一起。

ViT 的普通 transformer block：

``` text
通常不会像 CNN stage 一样
每过几层就把 40×40 下采样一次
```

`interaction_indexes=[5,8,11]` 表示：

``` text
在哪些 ViT 深度进行 interaction
```

而：

``` text
1/8、1/16、1/32
```

是 STA 最终提供给 detector 的多尺度空间输出。

------------------------------------------------------------------------

# 10. STA 输出：本课第一个真正关键接口

官方 X 配置明确让：

``` text
DINOv3STAs
```

输出三个 detector feature levels，供：

``` yaml
HybridEncoder:
  in_channels: [256,256,256]
```

接收。

对于 640×640：

``` text
stride 8:
640 / 8 = 80

stride 16:
640 / 16 = 40

stride 32:
640 / 32 = 20
```

因此 backbone → encoder 的接口：

``` text
S3 [B,256,80,80]
S4 [B,256,40,40]
S5 [B,256,20,20]
```

这三个 shape 是本课必须牢牢记住的。

------------------------------------------------------------------------

# 11. 三个 STA 输出分别有多少 spatial positions？

``` text
80×80 = 6400
40×40 = 1600
20×20 =  400
```

总计：

\[ 6400+1600+400=8400 \]

这个：

``` text
8400
```

会在 decoder 入口再次出现。

但现在不要急着 flatten。

因为中间还有：

``` text
HybridEncoder
```

------------------------------------------------------------------------

# 12. STA 的核心意义再总结一次

原始 ViT patch grid：

``` text
40×40
```

检测器希望：

``` text
80×80
40×40
20×20
```

所以：

``` text
DINOv3
提供强语义 representation

STA
把它改造成检测友好的 spatial multi-scale representation
```

这就是：

``` text
Foundation-model feature
→ detection feature pyramid
```

之间的桥梁。

------------------------------------------------------------------------

# 13. 为什么不能把 40×40 token 直接交给 decoder？

如果只有：

``` text
40×40
```

则小目标所需的高分辨率 feature 不充分。

检测通常希望：

``` text
stride 8
→ 更细 spatial grid

stride 16
→ 中尺度

stride 32
→ 更大 receptive context
```

所以 STA 不是一个"为了多加模块"的 adapter。

它在解决：

> **ViT 单尺度 representation 与多尺度 object detection
> 之间的结构错配。**

------------------------------------------------------------------------

# 14. STA 内部 Shape 应怎样真正验证？

不要只 hook：

``` text
DINOv3STAs
```

还应该 hook：

``` text
patch_embed
selected ViT blocks
spatial-prior branch
interaction blocks
STA output levels
```

但模块名可能随 commit 改动。

因此后面提供的 ShapeTracer 支持：

``` text
先打印 named_modules()
→ 再选择模块
```

而不是把可能过期的内部路径写死。

------------------------------------------------------------------------

# 15. HybridEncoder 收到什么？

输入：

``` text
feats = [
  [B,256,80,80],
  [B,256,40,40],
  [B,256,20,20]
]
```

源码第一步：

``` python
proj_feats = [
    self.input_proj[i](feat)
    for i, feat in enumerate(feats)
]
```

X：

``` text
in_channels = [256,256,256]
hidden_dim  = 256
```

所以 `input_proj` 不需要改 channel。

------------------------------------------------------------------------

# 16. HybridEncoder 的 input projection

源码逻辑：

``` python
if in_channel != hidden_dim:
    Conv2d(in_channel, hidden_dim, 1)
else:
    Identity()
```

X：

``` text
256 == 256
```

所以：

``` text
P3 projection: Identity
P4 projection: Identity
P5 projection: Identity
```

shape 不变：

``` text
[B,256,80,80]
[B,256,40,40]
[B,256,20,20]
```

------------------------------------------------------------------------

# 17. HybridEncoder 只对指定 level 做 intra-scale Transformer

base config：

``` text
use_encoder_idx = [2]
num_encoder_layers = 1
```

index 2 对应第三个 level：

``` text
20×20
```

所以：

``` text
[B,256,20,20]
```

先 flatten：

``` python
src_flatten = (
    proj_feats[2]
    .flatten(2)
    .permute(0,2,1)
)
```

得到：

``` text
[B,400,256]
```

因为：

\[ 20`\times20`{=tex}=400 \]

------------------------------------------------------------------------

# 18. HybridEncoder Transformer 的 Q/K/V 规模

输入：

``` text
[B,400,256]
```

nhead：

``` text
8
```

每 head dimension：

\[ 256/8=32 \]

所以概念上：

``` text
Q,K,V
[B,8,400,32]
```

attention matrix：

``` text
[B,8,400,400]
```

这就是为什么 HybridEncoder 不需要对：

``` text
8400 × 8400
```

做全尺度 self-attention。

------------------------------------------------------------------------

# 19. Encoder 后恢复 feature map

源码：

``` python
proj_feats[enc_ind] = (
    memory
    .permute(0,2,1)
    .reshape(
        -1,
        self.hidden_dim,
        h,
        w
    )
    .contiguous()
)
```

所以：

``` text
[B,400,256]
→ [B,256,20,20]
```

shape 回到 CNN-style feature map。

------------------------------------------------------------------------

# 20. Top-Down FPN：20×20 → 40×40

先：

``` text
high:
[B,256,20,20]
```

经过 lateral conv：

``` text
[B,256,20,20]
```

nearest interpolate ×2：

``` text
[B,256,40,40]
```

与原本：

``` text
[B,256,40,40]
```

融合。

当前 DEIMv2 base 使用：

``` text
fuse_op = sum
```

所以：

``` text
[B,256,40,40]
+
[B,256,40,40]
=
[B,256,40,40]
```

而不是 channel concat 后变成 512。

------------------------------------------------------------------------

# 21. Top-Down FPN：40×40 → 80×80

同理：

``` text
[B,256,40,40]
↓ lateral
[B,256,40,40]
↓ interpolate ×2
[B,256,80,80]

+
[B,256,80,80]

→ fusion block

[B,256,80,80]
```

得到高分辨率 FPN feature。

------------------------------------------------------------------------

# 22. Bottom-Up PAN：80×80 → 40×40

现在再反向传播 spatial information：

``` text
[B,256,80,80]
↓ stride-2 downsample
[B,256,40,40]

+
top-down 40×40 feature

↓ PAN block

[B,256,40,40]
```

------------------------------------------------------------------------

# 23. Bottom-Up PAN：40×40 → 20×20

继续：

``` text
[B,256,40,40]
↓ stride-2
[B,256,20,20]

+
top-down 20×20 feature

↓ PAN block

[B,256,20,20]
```

最终 HybridEncoder 输出：

``` text
E3 [B,256,80,80]
E4 [B,256,40,40]
E5 [B,256,20,20]
```

------------------------------------------------------------------------

# 24. STA 与 HybridEncoder 为什么不是重复工作？

STA：

``` text
DINOv3 representation
→ detector-compatible multi-scale spatial features
```

HybridEncoder：

``` text
已有多尺度 features
→ 跨尺度融合
→ detector decoder 更容易消费的 features
```

所以：

``` text
STA
解决 representation adaptation

HybridEncoder
解决 feature interaction / fusion
```

这是两个不同层次的问题。

------------------------------------------------------------------------

# 25. 到这里，feature map 仍然是 4D

``` text
[B,C,H,W]
```

而 Transformer decoder 更喜欢：

``` text
[B,N,C]
```

所以接下来进入：

``` text
DEIMTransformer._get_encoder_input()
```

------------------------------------------------------------------------

# 26. Decoder 的 input projection

X：

``` text
feat_channels=[256,256,256]
hidden_dim=256
```

源码：

``` python
if in_channels == self.hidden_dim:
    self.input_proj.append(
        nn.Identity()
    )
```

因此 decoder 入口同样无需 channel projection。

------------------------------------------------------------------------

# 27. 第一个 level flatten

``` text
[B,256,80,80]
```

执行：

``` python
feat.flatten(2)
```

得到：

``` text
[B,256,6400]
```

再：

``` python
permute(0,2,1)
```

得到：

``` text
[B,6400,256]
```

------------------------------------------------------------------------

# 28. 第二个 level flatten

``` text
[B,256,40,40]
→ [B,256,1600]
→ [B,1600,256]
```

------------------------------------------------------------------------

# 29. 第三个 level flatten

``` text
[B,256,20,20]
→ [B,256,400]
→ [B,400,256]
```

------------------------------------------------------------------------

# 30. Concatenate：8400 正式出现

``` python
feat_flatten = torch.concat(
    feat_flatten,
    dim=1
)
```

于是：

``` text
[B,6400,256]
[B,1600,256]
[B, 400,256]
        │
        ▼
[B,8400,256]
```

因为：

\[ 6400+1600+400=8400 \]

这个 tensor 后面通常可以称为：

``` text
memory
```

------------------------------------------------------------------------

# 31. `spatial_shapes`

与此同时保存：

``` text
[
 [80,80],
 [40,40],
 [20,20]
]
```

为什么 memory 已经 flatten 了还要保留这个？

因为：

``` text
MSDeformableAttention
```

必须知道：

``` text
8400 个位置中
哪一段属于哪个 feature level
以及该 level 的 H/W
```

------------------------------------------------------------------------

# 32. Memory 的索引区间

``` text
P3:
0 ... 6399

P4:
6400 ... 7999

P5:
8000 ... 8399
```

这是后面统计 top-k query 来源时非常有用的知识。

------------------------------------------------------------------------

# 33. Anchor Generation

每个 spatial position 都有一个 coarse anchor：

``` text
[cx,cy,w,h]
```

总 shape：

``` text
[1,8400,4]
```

batch\>1 时 repeat：

``` text
[B,8400,4]
```

------------------------------------------------------------------------

# 34. Anchor center

对于 level `(H,W)`：

``` python
grid_xy = (
    grid_xy.unsqueeze(0) + 0.5
) / [W,H]
```

例如 stride-16 的：

``` text
40×40
```

左上第一个 cell center：

\[ c_x=`\frac{0.5}{40}`{=tex}=0.0125 \]

\[ c_y=`\frac{0.5}{40}`{=tex}=0.0125 \]

------------------------------------------------------------------------

# 35. Anchor width / height

默认：

``` text
grid_size = 0.05
```

源码：

``` python
wh = grid_size * 2**level
```

三个 level：

``` text
level 0: 0.05
level 1: 0.10
level 2: 0.20
```

映射到 640：

``` text
32 px
64 px
128 px
```

只是 coarse initial geometry。

------------------------------------------------------------------------

# 36. Anchor 转到 logit space

``` python
anchors = torch.log(
    anchors / (1 - anchors)
)
```

即：

\[ logit(p) = `\ln`{=tex}`\frac{p}{1-p}`{=tex} \]

所以：

``` text
[B,8400,4]
```

shape 不变，但数值空间从：

``` text
0~1 normalized box
```

变成：

``` text
unbounded logit space
```

------------------------------------------------------------------------

# 37. Valid Mask

源码会检查：

``` text
eps < anchor < 1-eps
```

得到：

``` text
valid_mask
[1,8400,1]
```

或 broadcast 后：

``` text
[B,8400,1]
```

memory：

``` python
memory = valid_mask * memory
```

------------------------------------------------------------------------

# 38. Encoder Classification Head

``` python
enc_outputs_logits =
    self.enc_score_head(memory)
```

X / COCO：

``` text
[B,8400,256]
↓ Linear(256,80)
[B,8400,80]
```

也就是说：

> decoder 之前，8400 个 spatial candidates 已经先获得一轮 coarse class
> score。

------------------------------------------------------------------------

# 39. Default Top-K Selection

源码核心：

``` python
scores = outputs_logits.max(-1).values

_, topk_ind = torch.topk(
    scores,
    300,
    dim=-1
)
```

shape：

``` text
[B,8400,80]
↓ max over 80 classes
[B,8400]
↓ top-k
[B,300]
```

------------------------------------------------------------------------

# 40. Gather Top-K Memory

根据：

``` text
topk_ind [B,300]
```

从：

``` text
memory [B,8400,256]
```

取出：

``` text
enc_topk_memory
[B,300,256]
```

这就是默认 query content 的来源。

------------------------------------------------------------------------

# 41. Gather Top-K Anchors

从：

``` text
anchors
[B,8400,4]
```

得到：

``` text
enc_topk_anchors
[B,300,4]
```

------------------------------------------------------------------------

# 42. Gather Top-K Logits

训练时还会保存：

``` text
enc_topk_logits
[B,300,80]
```

用于 encoder auxiliary supervision。

------------------------------------------------------------------------

# 43. Encoder BBox Head

``` python
enc_topk_bbox_unact =
    self.enc_bbox_head(
        enc_topk_memory
    ) + enc_topk_anchors
```

shape：

``` text
[B,300,256]
↓ MLP
[B,300,4]

+
[B,300,4]

=
[B,300,4]
```

------------------------------------------------------------------------

# 44. Decoder 的两个核心输入

默认：

``` text
learn_query_content = False
```

因此：

``` python
content =
    enc_topk_memory.detach()
```

得到：

``` text
query content
[B,300,256]
```

同时：

``` text
reference logits
[B,300,4]
```

所以：

``` text
300 queries
```

不是简单的 300 个固定编号。

每张图片会从自身 memory 中选出 top-300 candidates。

------------------------------------------------------------------------

# 45. 训练时 Query 数为什么可能不是 300？

因为：

``` text
num_denoising = 100
```

训练阶段还会创建 denoising queries。

因此 decoder 输入可能是：

``` text
DN queries
+
300 matching queries
```

但：

> DN query 的实际数量不是永远机械等于 100。

它还与：

``` text
当前 batch 的 GT 数量
group construction
```

有关。

所以训练态 decoder 的 query 维度应：

``` text
runtime print
```

而不是死背。

推理态则明确：

``` text
Q = 300
```

------------------------------------------------------------------------

# 46. 本课后续先按推理态追踪

为了得到最干净的数据流：

``` text
model.eval()
targets=None
```

所以：

``` text
query content:
[B,300,256]

reference:
[B,300,4]
```

------------------------------------------------------------------------

# 47. Reference Box → Query Position Embedding

先：

``` python
ref_points_detach =
    sigmoid(ref_points_unact)
```

shape：

``` text
[B,300,4]
```

再：

``` python
query_pos_embed =
    query_pos_head(
        ref_points_detach
    )
```

得到：

``` text
[B,300,256]
```

------------------------------------------------------------------------

# 48. Decoder Layer 1：输入

Layer 1：

``` text
target:
[B,300,256]

query_pos:
[B,300,256]

reference:
[B,300,4]

memory:
[B,8400,256]
```

进入：

``` text
Self-Attention
→ MSDeformable Cross-Attention
→ SwiGLUFFN
```

------------------------------------------------------------------------

# 49. Layer 1 Self-Attention

源码：

``` python
q = k = target + query_pos_embed

target2, _ = self.self_attn(
    q,
    k,
    value=target
)
```

shape 都保持：

``` text
[B,300,256]
```

8 heads：

``` text
head_dim = 256/8 = 32
```

概念 Q/K/V：

``` text
[B,8,300,32]
```

attention score：

``` text
[B,8,300,300]
```

------------------------------------------------------------------------

# 50. Layer 1 Cross-Attention

输入：

``` text
300 queries
8400 memory positions
3 feature levels
```

但不是：

``` text
300×8400 dense attention
```

而是：

``` text
MSDeformableAttention
```

X：

``` text
num_points=[3,6,3]
```

所以每个 query / head：

\[ 3+6+3=12 \]

个采样点。

------------------------------------------------------------------------

# 51. 一层的 sampling budget

8 heads：

\[ 300`\times8`{=tex}`\times12`{=tex} = 28,800 \]

个 query-head-sampling units。

如果 dense query-memory：

\[ 300`\times8400`{=tex} = 2,520,000 \]

pair。

这就是第 7 课所学 Deformable Attention 在完整模型中的真实位置。

------------------------------------------------------------------------

# 52. Cross-Attention 输出

无论内部采样如何：

``` text
target2
[B,300,256]
```

然后：

``` text
Gateway
```

或 residual fusion。

X 当前 constructor 默认：

``` text
use_gateway=True
```

具体实际值应以你加载后的：

``` python
model.decoder.decoder.layers[0].use_gateway
```

打印结果为最终依据。

------------------------------------------------------------------------

# 53. FFN

当前 DEIMTransformer 使用：

``` text
SwiGLUFFN
```

输入：

``` text
[B,300,256]
```

输出仍：

``` text
[B,300,256]
```

所以一个 decoder layer 不改变：

``` text
Q = 300
C = 256
```

------------------------------------------------------------------------

# 54. Layer 1 preliminary bbox

第一层有：

``` python
pre_bboxes = sigmoid(
    pre_bbox_head(output)
    + inverse_sigmoid(
        ref_points_detach
    )
)
```

shape：

``` text
pre_bbox_head:
[B,300,256]
→ [B,300,4]

pre_bboxes:
[B,300,4]
```

------------------------------------------------------------------------

# 55. FDR Head

`reg_max=32`

每条边：

``` text
33 bins
```

4 条边：

\[ 4`\times33`{=tex}=132 \]

所以：

``` text
[B,300,256]
↓ bbox distribution head
[B,300,132]
```

reshape 概念：

``` text
[B,300,4,33]
```

------------------------------------------------------------------------

# 56. Integral

distribution：

``` text
[B,300,4,33]
```

经过 Integral：

``` text
continuous distances
[B,300,4]
```

再：

``` text
distance2bbox
```

得到：

``` text
inter_ref_bbox
[B,300,4]
```

------------------------------------------------------------------------

# 57. Classification Head

``` text
[B,300,256]
↓ Linear
[B,300,80]
```

再经过 LQE：

``` text
[B,300,80]
```

shape 不变。

------------------------------------------------------------------------

# 58. Layer 1 输出

所以 Layer 1 结束：

``` text
hidden:
[B,300,256]

logits:
[B,300,80]

FDR corners:
[B,300,132]

box:
[B,300,4]
```

新的 box 会：

``` python
ref_points_detach =
    inter_ref_bbox.detach()
```

成为下一层 attention reference。

------------------------------------------------------------------------

# 59. Layer 2

输入 shape 仍：

``` text
hidden:
[B,300,256]

reference:
[B,300,4]

query_pos:
[B,300,256]
```

然后重复：

``` text
Self-Attention
Cross-Attention
SwiGLU
FDR
LQE
```

输出：

``` text
[B,300,256]
[B,300,80]
[B,300,132]
[B,300,4]
```

------------------------------------------------------------------------

# 60. Layer 3～6

shape 同样保持。

因此训练态完整 stack：

``` text
decoder logits:
[6,B,300,80]

decoder boxes:
[6,B,300,4]

decoder FDR corners:
[6,B,300,132]

decoder refs:
[6,B,300,4]
```

------------------------------------------------------------------------

# 61. 推理态为什么不一定保存六层输出？

源码中：

``` python
if self.training or i == self.eval_idx:
    ...
    if not self.training:
        break
```

X：

``` text
num_layers=6
eval_idx=-1
```

转换为：

``` text
eval_idx=5
```

所以推理最终使用第 6 层。

训练时才需要中间层 auxiliary outputs。

------------------------------------------------------------------------

# 62. 最终 Decoder 主输出

``` text
pred_logits
[B,300,80]

pred_boxes
[B,300,4]
```

其中 box 格式进入 PostProcessor 前是：

``` text
cxcywh
```

并且是归一化坐标。

------------------------------------------------------------------------

# 63. 一个归一化 Box 示例

假设某 query：

``` text
cx = 0.50
cy = 0.40
w  = 0.20
h  = 0.30
```

即：

``` text
[0.50,0.40,0.20,0.30]
```

转换为 xyxy：

\[ x_1=c_x-w/2 \]

\[ y_1=c_y-h/2 \]

\[ x_2=c_x+w/2 \]

\[ y_2=c_y+h/2 \]

得到：

``` text
[0.40,0.25,0.60,0.55]
```

------------------------------------------------------------------------

# 64. PostProcessor：cxcywh → xyxy

真实代码使用：

``` python
torchvision.ops.box_convert(
    boxes,
    in_fmt='cxcywh',
    out_fmt='xyxy'
)
```

shape：

``` text
[B,300,4]
→
[B,300,4]
```

只是坐标语义改变。

------------------------------------------------------------------------

# 65. 归一化坐标 → 原图像素坐标

源码：

``` python
bbox_pred *= (
    orig_target_sizes
    .repeat(1,2)
    .unsqueeze(1)
)
```

注意：

``` text
orig_target_sizes
```

对应的维度顺序必须以数据管线实际定义为准；官方 postprocessor 直接按该
tensor 的既定约定缩放 xyxy。

如果原图本身就是：

``` text
640×640
```

则示例：

``` text
[0.40,0.25,0.60,0.55]
× 640
```

得到：

``` text
[256,160,384,352]
```

------------------------------------------------------------------------

# 66. Classification Score

DEIMv2 默认 focal-loss 风格：

``` python
scores = sigmoid(logits)
```

所以：

``` text
[B,300,80]
```

仍然是：

``` text
[B,300,80]
```

每个 query-class pair 都有一个 score。

------------------------------------------------------------------------

# 67. 一个非常重要的 PostProcessor 细节

PostProcessor 不是简单：

``` text
每个 query 只取最大 class
→ 300 queries
```

在 focal-loss 路径中，它执行：

``` python
scores.flatten(1)
```

所以：

``` text
[B,300,80]
→ [B,24000]
```

因为：

\[ 300`\times80`{=tex}=24000 \]

------------------------------------------------------------------------

# 68. Final Top-300 是从 24,000 个 query-class pairs 里选

``` python
scores, index = torch.topk(
    scores.flatten(1),
    300,
    dim=-1
)
```

所以：

``` text
scores:
[B,300]

index:
[B,300]
```

------------------------------------------------------------------------

# 69. 从 flattened index 恢复 class 与 query

类别：

``` python
label =
    index % num_classes
```

即：

``` text
[B,300]
```

query index：

``` python
query_index =
    index // num_classes
```

即：

``` text
[B,300]
```

然后从：

``` text
bbox_pred [B,300,4]
```

gather：

``` text
boxes [B,300,4]
```

------------------------------------------------------------------------

# 70. 这意味着同一个 Query 理论上可能贡献多个类别候选

因为 top-k 是在：

``` text
query × class
```

的 flattened score 空间里选。

这是很多初学者第一次读 PostProcessor 时会漏掉的细节。

------------------------------------------------------------------------

# 71. 最终输出

非 deploy mode：

``` python
[
  {
    "labels": ...,
    "boxes": ...,
    "scores": ...
  },
  ...
]
```

每张图：

``` text
labels [300]
boxes  [300,4]
scores [300]
```

------------------------------------------------------------------------

# 72. 从输入到输出的完整 Shape 表

  阶段                        Tensor            Shape
  --------------------------- ----------------- -----------------
  Input                       image             `[B,3,640,640]`
  ViT patch grid              spatial patches   `40×40 = 1600`
  STA output 1                stride-8          `[B,256,80,80]`
  STA output 2                stride-16         `[B,256,40,40]`
  STA output 3                stride-32         `[B,256,20,20]`
  HybridEncoder input         P3                `[B,256,80,80]`
  HybridEncoder input         P4                `[B,256,40,40]`
  HybridEncoder Transformer   P5 flattened      `[B,400,256]`
  HybridEncoder output        E3                `[B,256,80,80]`
  HybridEncoder output        E4                `[B,256,40,40]`
  HybridEncoder output        E5                `[B,256,20,20]`
  Decoder flatten             level 0           `[B,6400,256]`
  Decoder flatten             level 1           `[B,1600,256]`
  Decoder flatten             level 2           `[B,400,256]`
  Decoder memory              concat            `[B,8400,256]`
  Anchor                      all positions     `[B,8400,4]`
  Encoder logits              all candidates    `[B,8400,80]`
  Top-k index                 selected          `[B,300]`
  Query content               top-k memory      `[B,300,256]`
  Initial reference           bbox              `[B,300,4]`
  Query pos                   MLP               `[B,300,256]`
  Decoder hidden              each layer        `[B,300,256]`
  Decoder class               each layer        `[B,300,80]`
  FDR distribution            each layer        `[B,300,132]`
  Decoder bbox                each layer        `[B,300,4]`
  Main logits                 final             `[B,300,80]`
  Main boxes                  final             `[B,300,4]`
  Post sigmoid                scores            `[B,300,80]`
  Flatten class-query         scores            `[B,24000]`
  Final top-k                 scores            `[B,300]`
  Final labels                labels            `[B,300]`
  Final boxes                 xyxy pixels       `[B,300,4]`

------------------------------------------------------------------------

# 73. 一张图的数字总链

对于：

``` text
B = 1
```

可以背成：

``` text
[1,3,640,640]

→ patch grid 40×40 = 1600

→ STA
[1,256,80,80]
[1,256,40,40]
[1,256,20,20]

→ HybridEncoder
[1,256,80,80]
[1,256,40,40]
[1,256,20,20]

→ flatten
[1,6400,256]
[1,1600,256]
[1, 400,256]

→ concat
[1,8400,256]

→ score
[1,8400,80]

→ top-k
[1,300]

→ query
[1,300,256]

→ reference
[1,300,4]

→ decoder ×6

→ logits
[1,300,80]

→ FDR
[1,300,132]

→ box
[1,300,4]

→ score flatten
[1,24000]

→ final top-300

labels [1,300]
boxes  [1,300,4]
scores [1,300]
```

------------------------------------------------------------------------

# 74. 三个最重要的"数量"

整套 X 模型里反复出现三个数量：

``` text
1600
8400
300
```

分别是什么？

``` text
1600:
40×40 ViT spatial patch grid

8400:
80²+40²+20²
decoder multi-scale memory positions

300:
object query / final query slots
```

千万不要混淆。

------------------------------------------------------------------------

# 75. 1600 → 8400 不是"复制 token"

这是一个非常重要的概念。

不能理解成：

``` text
1600 DINO tokens
简单复制几份
→ 8400
```

真正过程是：

``` text
DINOv3 representation
+
STA spatial adaptation
→ 3 个 multi-scale feature maps

HybridEncoder
→ cross-scale fusion

最后 flatten
→ 8400 memory positions
```

所以 8400 是 detector feature pyramid 的空间位置总数。

------------------------------------------------------------------------

# 76. 8400 → 300 也不是池化

这是：

``` text
encoder score
→ top-k candidate selection
```

即：

``` text
8400 candidate positions
→ 选 300 个最有希望成为 object query 的位置
```

------------------------------------------------------------------------

# 77. 300 → 300 为什么没有继续减少？

因为 decoder 的任务不是继续筛掉 query。

它的任务是：

``` text
refine representation
refine class
refine box
coordinate queries
```

最终 PostProcessor 再根据 score 选择输出。

------------------------------------------------------------------------

# 78. `300×80=24000` 又是什么？

它是：

``` text
300 query predictions
×
80 classes
```

形成的 query-class score pairs。

它不是：

``` text
24000 boxes
```

真实 bbox tensor仍只有：

``` text
300 boxes
```

------------------------------------------------------------------------

# 79. Memory 量级

``` text
memory:
[B,8400,256]
```

B=1：

\[ 8400`\times256`{=tex} = 2,150,400 \]

elements。

FP32：

\[ 2,150,400`\times4`{=tex} `\approx8.60`{=tex}`\text{ MB}`{=tex} \]

FP16：

\[ `\approx4.30`{=tex}`\text{ MB}`{=tex} \]

仅是 memory 本体，不包含 autograd、中间 attention、optimizer states。

------------------------------------------------------------------------

# 80. Query Tensor 量级

``` text
[B,300,256]
```

B=1：

\[ 300`\times256`{=tex}=76,800 \]

elements。

FP32：

``` text
~0.31 MB
```

所以：

``` text
memory positions 多
queries 少
```

是 DETR decoder 的一个关键结构特征。

------------------------------------------------------------------------

# 81. 为什么 Deformable Cross-Attention 很关键？

如果每个 query 对 8400 memory 做 dense attention：

\[ 300`\times8400`{=tex} = 2.52M \]

query-memory pairs / head-like interaction scale。

而当前：

``` text
3+6+3=12 samples
```

所以：

\[ 300`\times12`{=tex}=3600 \]

sampling positions / head。

这使多尺度高分辨率 memory 在实时 detector 中可行。

------------------------------------------------------------------------

# 82. Training Mode 与 Eval Mode 的 Shape 差异

推理：

``` text
Q = 300
```

训练：

``` text
Q_total =
Q_dn + 300
```

并且训练会保存：

``` text
6 层 logits
6 层 boxes
6 层 FDR distributions
encoder auxiliary outputs
pre outputs
DN outputs
```

因此训练显存明显高于推理。

------------------------------------------------------------------------

# 83. Training 输出结构

概念上：

``` text
pred_logits
pred_boxes

aux_outputs
enc_aux_outputs
pre_outputs

pred_corners
ref_points

dn_aux_outputs
dn_pre_outputs
dn_meta
...
```

具体 keys 以当前 commit 的 `DEIMTransformer.forward()` 为准。

------------------------------------------------------------------------

# 84. 为什么训练时必须保存中间层？

因为：

``` text
auxiliary loss
FDR loss
localization distillation
denoising loss
encoder proposal supervision
```

都可能需要这些中间结果。

所以：

``` text
训练 graph
```

比：

``` text
最终推理 graph
```

复杂得多。

------------------------------------------------------------------------

# 85. `eval_idx=-1`

X：

``` text
num_layers = 6
eval_idx = -1
```

源码：

``` python
eval_idx =
    num_layers + eval_idx
```

得到：

``` text
5
```

0-based index：

``` text
layer 5
```

也就是：

``` text
第 6 个 decoder layer
```

------------------------------------------------------------------------

# 86. PostProcessor 不做传统 NMS

整个流程没有：

``` text
IoU threshold
→ suppress duplicate boxes
```

这样的传统 NMS 主步骤。

这与：

``` text
O2O set prediction
```

目标一致。

Dense O2O 是训练阶段提高 supervision density
的方法，不意味着推理重新变回 dense+NMS detector。

------------------------------------------------------------------------

# 87. 从"空间尺度"视角重新走一次

``` text
640×640 image

↓ ViT patch size 16

40×40 semantic token grid

↓ STA

80×80
40×40
20×20

↓ HybridEncoder

80×80
40×40
20×20

↓ flatten

6400
1600
400

↓ concat

8400 positions

↓ top-k

300 object slots

↓ decoder

300 refined object slots

↓ postprocess

300 scored detections
```

------------------------------------------------------------------------

# 88. 从"channel"视角重新走一次

输入：

``` text
RGB = 3 channels
```

进入 DINOv3：

``` text
ViT internal embedding
```

经过 STA detector interface：

``` text
256 channels
```

之后：

``` text
HybridEncoder:
256

DEIMTransformer memory:
256

Query:
256

Decoder hidden:
256
```

最终 head：

``` text
classification:
256 → 80

FDR:
256 → 132

box after integral:
132 → 4
```

------------------------------------------------------------------------

# 89. 从"box"视角重新走一次

``` text
multi-scale grid anchor
[B,8400,4]

↓ top-k

[B,300,4]

↓ encoder bbox correction

[B,300,4]

↓ decoder pre_bbox refinement

[B,300,4]

↓ FDR distribution
[B,300,132]

↓ Integral
[B,300,4] distances

↓ distance2bbox

[B,300,4]

↓ repeated refinement ×6

[B,300,4]

↓ cxcywh → xyxy

[B,300,4]

↓ normalized → pixel

[B,300,4]
```

------------------------------------------------------------------------

# 90. 从"classification"视角重新走一次

``` text
memory
[B,8400,256]

↓ encoder score head

[B,8400,80]

↓ max class / position

[B,8400]

↓ top-300

300 candidate positions

↓ decoder

[B,300,256]

↓ class head + LQE

[B,300,80]

↓ sigmoid

[B,300,80]

↓ flatten

[B,24000]

↓ final top-300

scores [B,300]
labels [B,300]
```

------------------------------------------------------------------------

# 91. 从"attention"视角重新走一次

``` text
DINOv3:
ViT self-attention over token sequence

HybridEncoder:
self-attention on selected 20×20 level
400 tokens

DEIMTransformer:
query self-attention
300×300

+
multi-scale deformable cross-attention
300 queries
→ sparse samples from 8400 memory positions
```

所以模型里不是只有"一种 Attention"。

------------------------------------------------------------------------

# 92. 从"stride"视角重新走一次

``` text
Input:
stride 1
640×640

ViT patch grid:
stride 16
40×40

STA output:
stride 8  → 80×80
stride 16 → 40×40
stride 32 → 20×20

HybridEncoder:
保持这三个 detector strides

Decoder:
flatten 后不再用 H/W tensor layout，
但 spatial_shapes 仍保存 stride 对应的二维结构
```

------------------------------------------------------------------------

# 93. ShapeTracer：为什么要自己跑？

本课程可以告诉你：

``` text
模型设计上应该是什么 shape
```

但真正做算法工程时，你必须能自己验证：

``` text
当前 commit
当前 config
当前 batch
当前 input size
当前 train/eval mode
```

的实际 shape。

所以本课最重要的实践工具是：

``` text
ShapeTracer
```

------------------------------------------------------------------------

# 94. 一个通用 Shape 格式化器

``` python
def shape_of(x):
    import torch

    if isinstance(x, torch.Tensor):
        return {
            "shape": tuple(x.shape),
            "dtype": str(x.dtype),
            "device": str(x.device),
            "requires_grad": x.requires_grad,
        }

    if isinstance(x, (list, tuple)):
        return [
            shape_of(v)
            for v in x
        ]

    if isinstance(x, dict):
        return {
            k: shape_of(v)
            for k, v in x.items()
        }

    return type(x).__name__
```

------------------------------------------------------------------------

# 95. 通用 Forward Hook

``` python
def make_hook(name):
    def hook(module, inputs, output):
        print(
            f"\n[{name}]"
        )

        print(
            "  input :",
            shape_of(inputs)
        )

        print(
            "  output:",
            shape_of(output)
        )

    return hook
```

------------------------------------------------------------------------

# 96. 列出所有模块名

在你不知道 STA 内部模块名时：

``` python
for name, module in model.named_modules():
    print(
        name,
        "->",
        module.__class__.__name__
    )
```

然后搜索：

``` text
dinov3
sta
interaction
encoder
decoder
```

这是比猜模块路径可靠得多的方法。

------------------------------------------------------------------------

# 97. 顶层四模块 Hook

假设模型已经实例化：

``` python
handles = []

for name in [
    "backbone",
    "encoder",
    "decoder",
    "postprocessor",
]:
    module = getattr(
        model,
        name
    )

    handles.append(
        module.register_forward_hook(
            make_hook(name)
        )
    )
```

------------------------------------------------------------------------

# 98. 运行一次推理

``` python
import torch

model.eval().cuda()

x = torch.randn(
    1,
    3,
    640,
    640,
    device="cuda"
)

orig_target_sizes = torch.tensor(
    [[640, 640]],
    device="cuda"
)

with torch.no_grad():
    y = model(
        x,
        orig_target_sizes
    )
```

注意：

> 你仓库中的顶层 model forward 签名可能与 Hugging Face 示例 / solver
> 内部调用略有区别。

因此优先查看：

``` python
print(
    model.forward.__doc__
)
```

以及源码定义。

------------------------------------------------------------------------

# 99. Hook HybridEncoder 的 Transformer

可以先查：

``` python
print(
    model.encoder
)
```

通常重点 hook：

``` text
encoder.input_proj
encoder.encoder
encoder.lateral_convs
encoder.fpn_blocks
encoder.downsample_convs
encoder.pan_blocks
```

------------------------------------------------------------------------

# 100. HybridEncoder Hook 示例

``` python
for name, module in model.encoder.named_modules():

    interesting = (
        "input_proj" in name
        or "encoder" in name
        or "fpn_blocks" in name
        or "pan_blocks" in name
        or "downsample_convs" in name
    )

    if interesting:
        handles.append(
            module.register_forward_hook(
                make_hook(
                    "encoder." + name
                )
            )
        )
```

------------------------------------------------------------------------

# 101. Decoder Hook 示例

``` python
for name, module in model.decoder.named_modules():

    interesting = (
        "enc_score_head" in name
        or "enc_bbox_head" in name
        or "query_pos_head" in name
        or "decoder.layers" in name
        or "dec_score_head" in name
        or "dec_bbox_head" in name
    )

    if interesting:
        handles.append(
            module.register_forward_hook(
                make_hook(
                    "decoder." + name
                )
            )
        )
```

------------------------------------------------------------------------

# 102. Hook 完记得删除

``` python
for h in handles:
    h.remove()
```

否则重复运行 notebook 时：

``` text
hook 会越来越多
```

然后你会看到同一模块打印很多次。

------------------------------------------------------------------------

# 103. 更实用的 ShapeTracer 类

``` python
class ShapeTracer:
    def __init__(self):
        self.handles = []

    def _shape(self, x):
        import torch

        if isinstance(x, torch.Tensor):
            return tuple(x.shape)

        if isinstance(x, (list, tuple)):
            return [
                self._shape(v)
                for v in x
            ]

        if isinstance(x, dict):
            return {
                k: self._shape(v)
                for k, v in x.items()
            }

        return type(x).__name__

    def hook(self, name):
        def _fn(module, inputs, output):
            print(
                f"{name:60s}",
                "IN =",
                self._shape(inputs),
                "OUT =",
                self._shape(output)
            )

        return _fn

    def add(self, name, module):
        self.handles.append(
            module.register_forward_hook(
                self.hook(name)
            )
        )

    def clear(self):
        for h in self.handles:
            h.remove()

        self.handles.clear()
```

------------------------------------------------------------------------

# 104. 自动 Hook Leaf Modules

``` python
def add_leaf_hooks(
    model,
    tracer,
    prefix=""
):
    for name, module in model.named_modules():

        if len(
            list(module.children())
        ) == 0:

            full_name = (
                f"{prefix}.{name}"
                if prefix
                else name
            )

            tracer.add(
                full_name,
                module
            )
```

注意：

``` text
全部 leaf hook
```

输出会非常长。

建议先写入文件。

------------------------------------------------------------------------

# 105. 把 Shape Trace 写入日志

``` python
import contextlib

with open(
    "deimv2_x_shape_trace.txt",
    "w",
    encoding="utf-8"
) as f:

    with contextlib.redirect_stdout(f):

        with torch.no_grad():
            output = model(
                x,
                orig_target_sizes
            )
```

这样你可以全文搜索：

``` text
80, 80
40, 40
20, 20
8400
300
132
```

------------------------------------------------------------------------

# 106. 建议的断点位置

如果使用 VS Code / PyCharm debugger：

``` text
DINOv3STAs.forward
HybridEncoder.forward
DEIMTransformer.forward
DEIMTransformer._get_encoder_input
DEIMTransformer._generate_anchors
DEIMTransformer._get_decoder_input
DEIMTransformer._select_topk
TransformerDecoder.forward
TransformerDecoderLayer.forward
PostProcessor.forward
```

------------------------------------------------------------------------

# 107. 每个断点要检查什么？

不要只看代码。

固定检查：

``` python
tensor.shape
tensor.dtype
tensor.device
tensor.min()
tensor.max()
tensor.mean()
tensor.requires_grad
```

对于 box：

``` python
print(
    boxes[0,:5]
)
```

对于 score：

``` python
print(
    scores[0,:5]
)
```

------------------------------------------------------------------------

# 108. 推荐一个调试函数

``` python
def inspect_tensor(
    name,
    x,
    n=3
):
    import torch

    if not isinstance(
        x,
        torch.Tensor
    ):
        print(
            name,
            type(x)
        )
        return

    print(
        f"{name}:"
    )

    print(
        " shape =",
        tuple(x.shape)
    )

    print(
        " dtype =",
        x.dtype
    )

    print(
        " device =",
        x.device
    )

    print(
        " requires_grad =",
        x.requires_grad
    )

    if x.numel():
        y = x.detach().float()

        print(
            " min/max/mean =",
            y.min().item(),
            y.max().item(),
            y.mean().item()
        )

        print(
            " sample =",
            y.flatten()[:n].tolist()
        )
```

------------------------------------------------------------------------

# 109. 为什么不仅要看 Shape？

两个 tensor 都可能是：

``` text
[B,300,4]
```

但含义完全不同：

``` text
anchor logits
normalized cxcywh
FDR distances
xyxy normalized
xyxy pixel
```

所以完整 Tensor Trace 应至少记录：

``` text
shape
dtype
semantic meaning
coordinate system
range
```

------------------------------------------------------------------------

# 110. 建议的 Tensor Trace 表

你自己调试时建立 CSV / Markdown：

``` text
name
module
shape
dtype
range
semantic
coordinate_format
notes
```

例如：

``` text
decoder pred_boxes
[B,300,4]
fp16
0~1
final normalized boxes
cxcywh
before postprocessor
```

------------------------------------------------------------------------

# 111. 一个极易踩坑的地方：`train()` vs `eval()`

如果你：

``` python
model.train()
```

decoder 会创建 DN queries，并保存多层 auxiliary outputs。

如果：

``` python
model.eval()
```

则没有 DN queries，并按 eval_idx 输出。

所以同一个 hook：

``` text
query dimension
```

可能不同。

这不是 bug。

------------------------------------------------------------------------

# 112. 第二个坑：输入尺寸不是 640

如果输入：

``` text
800×800
```

则：

``` text
stride 8  → 100×100
stride 16 → 50×50
stride 32 → 25×25
```

memory positions：

\[ 10000+2500+625=13125 \]

不再是 8400。

所以：

``` text
8400
```

不是模型写死的神秘常数。

它来自：

``` text
input size + feature strides
```

------------------------------------------------------------------------

# 113. 第三个坑：非正方形输入

例如：

``` text
640×960
```

则：

``` text
stride 8:
80×120

stride 16:
40×60

stride 32:
20×30
```

memory：

\[ 80`\times120`{=tex}+ 40`\times60`{=tex}+ 20`\times30`{=tex} \]

\[ =9600+2400+600 =12600 \]

所以所有 shape 推导都应从：

``` text
H,W
```

开始，而不是只记 640。

------------------------------------------------------------------------

# 114. 第四个坑：DINOv3 special tokens

不要把：

``` text
ViT block sequence length
```

直接写死成：

``` text
1600
```

除非你已经确认当前 implementation 是否保留：

``` text
class/register/special tokens
```

但 spatial patch grid：

``` text
40×40
```

仍然成立。

------------------------------------------------------------------------

# 115. 第五个坑：`hidden_dim=256` 不代表 DINOv3 每个内部 tensor 都是 256

这里尤其重要。

X config 中：

``` text
DINOv3STAs.hidden_dim=256
HybridEncoder.hidden_dim=256
DEIMTransformer.hidden_dim=256
```

说明 detector interface 是 256。

但 DINOv3 backbone 内部原生 representation 的具体 embedding width：

``` text
必须以当前实现 / 实例化模型打印结果为准
```

不要因为 detector hidden dim 是 256，就把所有 ViT 内部 tensor 强行标成
256。

------------------------------------------------------------------------

# 116. 第六个坑：`pred_boxes` 的 4 不是 xyxy 一直到底

decoder 主输出：

``` text
normalized cxcywh
```

PostProcessor 后：

``` text
pixel xyxy
```

同样 `[B,300,4]`，语义已经改变。

------------------------------------------------------------------------

# 117. 第七个坑：FDR 的 132 不是 132 个 boxes

``` text
132 =
4 × (32+1)
```

表示：

``` text
4 个边距离
×
33 个 distribution bins
```

最终 Integral 后仍然回到：

``` text
4 个连续距离
```

------------------------------------------------------------------------

# 118. 第八个坑：最终 top-300 不等于 decoder 的 top-300 query selection

有两个不同的 top-k：

第一处：

``` text
8400 spatial candidates
→ top-300 queries
```

第二处：

``` text
300×80 class-query scores
→ final top-300 detections
```

二者目的完全不同。

------------------------------------------------------------------------

# 119. 第九个坑：HybridEncoder 的 400-token Transformer 与 Decoder Memory 不一样

HybridEncoder self-attention：

``` text
[B,400,256]
```

对应：

``` text
20×20 level
```

Decoder memory：

``` text
[B,8400,256]
```

对应：

``` text
三个 level flatten + concat
```

不要把两个"encoder/memory"概念混起来。

------------------------------------------------------------------------

# 120. 第十个坑：`encoder` 这个词在仓库里有多层含义

你可能同时看到：

``` text
HybridEncoder
TransformerEncoder inside HybridEncoder
encoder proposal outputs inside decoder
```

所以读源码时永远问：

> 这里的 encoder 到底指哪一层抽象？

------------------------------------------------------------------------

# 121. 小型 PyTorch 实验：模拟整个主 shape 流

``` python
import torch

B = 2
C = 256

p3 = torch.randn(
    B,C,80,80
)

p4 = torch.randn(
    B,C,40,40
)

p5 = torch.randn(
    B,C,20,20
)

flat = []

for x in [p3,p4,p5]:
    y = (
        x.flatten(2)
         .permute(0,2,1)
    )

    print(
        x.shape,
        "->",
        y.shape
    )

    flat.append(y)

memory = torch.cat(
    flat,
    dim=1
)

print(
    "memory:",
    memory.shape
)

assert memory.shape == (
    B,8400,256
)
```

------------------------------------------------------------------------

# 122. 模拟 Encoder Score + Top-K

``` python
import torch.nn as nn

score_head = nn.Linear(
    256,
    80
)

logits = score_head(
    memory
)

print(
    logits.shape
)
# [B,8400,80]

position_score = (
    logits
    .max(-1)
    .values
)

topk_score, topk_idx = (
    torch.topk(
        position_score,
        300,
        dim=-1
    )
)

print(
    topk_idx.shape
)
# [B,300]
```

------------------------------------------------------------------------

# 123. Gather Query Content

``` python
query = memory.gather(
    dim=1,
    index=topk_idx[
        ...,None
    ].expand(
        -1,-1,256
    )
)

print(
    query.shape
)
# [B,300,256]
```

这就是：

``` text
8400 → 300
```

最小可运行版本。

------------------------------------------------------------------------

# 124. 模拟 Classification Head

``` python
decoder_cls = nn.Linear(
    256,
    80
)

pred_logits = decoder_cls(
    query
)

print(
    pred_logits.shape
)
# [B,300,80]
```

------------------------------------------------------------------------

# 125. 模拟 FDR Head

``` python
reg_max = 32

decoder_reg = nn.Linear(
    256,
    4 * (
        reg_max + 1
    )
)

pred_corners = decoder_reg(
    query
)

print(
    pred_corners.shape
)
# [B,300,132]
```

------------------------------------------------------------------------

# 126. 模拟 PostProcessor Score Flatten

``` python
scores = pred_logits.sigmoid()

print(
    scores.shape
)
# [B,300,80]

flat_scores = scores.flatten(1)

print(
    flat_scores.shape
)
# [B,24000]

top_scores, index = torch.topk(
    flat_scores,
    300,
    dim=-1
)

labels = (
    index % 80
)

query_idx = (
    index // 80
)

print(
    top_scores.shape,
    labels.shape,
    query_idx.shape
)
```

输出：

``` text
[B,300]
[B,300]
[B,300]
```

------------------------------------------------------------------------

# 127. 端到端 Shape Assertion 思路

当你能实例化真实模型后，建议给关键接口加 assertion。

例如：

``` python
with torch.no_grad():
    feats = model.backbone(x)

assert len(feats) == 3

assert feats[0].shape[-2:] == (
    80,80
)

assert feats[1].shape[-2:] == (
    40,40
)

assert feats[2].shape[-2:] == (
    20,20
)
```

------------------------------------------------------------------------

# 128. Encoder Assertion

``` python
with torch.no_grad():
    enc = model.encoder(
        feats
    )

expected = [
    (80,80),
    (40,40),
    (20,20),
]

for feat, hw in zip(
    enc,
    expected
):
    assert (
        feat.shape[1] == 256
    )

    assert (
        feat.shape[-2:] == hw
    )
```

------------------------------------------------------------------------

# 129. Decoder Assertion

``` python
with torch.no_grad():
    out = model.decoder(
        enc
    )

assert out[
    "pred_logits"
].shape == (
    1,300,80
)

assert out[
    "pred_boxes"
].shape == (
    1,300,4
)
```

这是非常好的回归测试。

------------------------------------------------------------------------

# 130. 如果 Assertion 失败怎么办？

不要立刻认为模型错了。

依次检查：

``` text
1. 当前输入真的是 640×640 吗？
2. 当前 config 真的是 X 吗？
3. num_classes 真的是 80 吗？
4. 是否处于 train mode？
5. 是否修改过 num_queries？
6. 是否是不同 commit？
7. 是否加载了自定义 dataset config？
8. 是否修改了 feat_strides？
```

------------------------------------------------------------------------

# 131. 你应该学会打印"最终解析后的配置"

因为 YAML 有：

``` text
__include__
```

最终生效配置并不等于你正在看的一个文件。

所以调试时应该检查：

``` text
最终 merged config
```

而不是只看：

``` text
deimv2_dinov3_x_coco.yml
```

------------------------------------------------------------------------

# 132. 为什么这一课必须在第 22 课才做？

因为如果第 1 课就把这些数字全部告诉你：

``` text
1600
8400
300
132
24000
```

你只能背。

现在你已经知道：

``` text
1600
来自 ViT patch grid

8400
来自 multi-scale feature pyramid

300
来自 object query budget

132
来自 FDR distribution

24000
来自 query×class postprocess score space
```

这才叫真正读懂 shape。

------------------------------------------------------------------------

# 133. 五条主数据流必须同时看到

完整 detector 实际上有五条并行逻辑：

``` text
视觉 feature 流
空间 geometry 流
query 流
classification 流
bbox refinement 流
```

它们在 decoder 中汇合。

------------------------------------------------------------------------

# 134. 视觉 Feature 流

``` text
Image
→ DINOv3
→ STA
→ HybridEncoder
→ memory
→ Deformable Cross-Attention
→ query hidden state
```

------------------------------------------------------------------------

# 135. 空间 Geometry 流

``` text
640×640
→ patch grid
→ stride 8/16/32
→ spatial_shapes
→ anchors
→ reference boxes
→ sampling locations
```

------------------------------------------------------------------------

# 136. Query 流

``` text
8400 memory candidates
→ encoder scores
→ top-300
→ query content
→ 6 decoder layers
```

------------------------------------------------------------------------

# 137. Classification 流

``` text
8400 candidate class scores
→ top-k query selection

then

300 decoder query features
→ 80-class logits
→ LQE
→ sigmoid
→ final top-k
```

------------------------------------------------------------------------

# 138. BBox 流

``` text
grid anchors
→ encoder bbox residual
→ initial reference
→ pre_bbox
→ FDR distribution
→ Integral
→ distance2bbox
→ iterative refinement
→ normalized cxcywh
→ xyxy
→ pixel coordinates
```

------------------------------------------------------------------------

# 139. 一个真正重要的源码阅读能力

看到：

``` python
feat.flatten(2).permute(0,2,1)
```

你应该立即在脑中得到：

``` text
[B,C,H,W]
→ [B,C,HW]
→ [B,HW,C]
```

看到：

``` python
torch.concat(feat_flatten, 1)
```

你应该立即想到：

``` text
6400+1600+400=8400
```

看到：

``` python
4*(reg_max+1)
```

立即想到：

``` text
4×33=132
```

看到：

``` python
scores.flatten(1)
```

立即想到：

``` text
300×80=24000
```

这就是本课程训练 shape intuition 的目的。

------------------------------------------------------------------------

# 140. 建议你实际制作一张 Shape Map

格式：

``` text
MODULE
INPUT
OPERATION
OUTPUT
WHY
```

例如：

``` text
DEIMTransformer._get_encoder_input

INPUT:
[B,256,80,80]
[B,256,40,40]
[B,256,20,20]

OP:
flatten + permute + concat

OUTPUT:
[B,8400,256]

WHY:
把 multi-scale spatial feature
转换成 decoder memory sequence
```

每个模块都这样写一次。

------------------------------------------------------------------------

# 141. 推荐的调试顺序

第一次不要 hook 100 个模块。

先：

``` text
Round 1:
backbone
encoder
decoder
postprocessor
```

确认大接口。

第二轮：

``` text
DINOv3 selected blocks
STA interactions
```

第三轮：

``` text
HybridEncoder FPN/PAN
```

第四轮：

``` text
decoder top-k / layers / heads
```

这样不会被日志淹没。

------------------------------------------------------------------------

# 142. 第一次真实运行建议用 B=1

``` text
B=1
640×640
eval()
no_grad()
```

原因：

``` text
shape 最直观
显存最小
没有 DN queries
没有 backward graph
日志最容易读
```

确认推理链后，再切：

``` text
train()
B=2
targets
```

观察训练态差异。

------------------------------------------------------------------------

# 143. 然后再追训练态

训练态重点观察：

``` text
DN query 数
decoder total query count
6-layer auxiliary outputs
encoder auxiliary outputs
pred_corners
ref_points
loss input
```

这会直接衔接第 23 课：

``` text
一个 batch 到底怎样完成 backward
```

------------------------------------------------------------------------

# 144. 本课最终总图

``` text
IMAGE
[B,3,640,640]
│
│ patch size = 16
▼
DINOv3 spatial patch grid
40×40 = 1600 patches
│
│ interaction at blocks 5 / 8 / 11
▼
STA
├── [B,256,80,80]  stride 8
├── [B,256,40,40]  stride 16
└── [B,256,20,20]  stride 32
│
▼
HYBRID ENCODER
│
├── deepest level:
│   [B,256,20,20]
│   → [B,400,256]
│   → Transformer
│   → [B,256,20,20]
│
├── top-down FPN
└── bottom-up PAN
│
▼
[B,256,80,80]
[B,256,40,40]
[B,256,20,20]
│
▼
DEIMTransformer input flatten
│
├── [B,6400,256]
├── [B,1600,256]
└── [B, 400,256]
│
▼
MEMORY
[B,8400,256]
│
├───────────────┐
│               │
▼               ▼
anchors         enc_score_head
[B,8400,4]      [B,8400,80]
│               │
└───────┬───────┘
        ▼
     TOP-K 300
        │
 ┌──────┴─────────┐
 ▼                ▼
query content     reference
[B,300,256]       [B,300,4]
 │                │
 └──────┬─────────┘
        ▼
query_pos_head
[B,300,256]
        │
        ▼
DEIMTransformer Decoder ×6
        │
        ├─ Self Attention
        │  [B,300,256]
        │
        ├─ MS Deformable Attention
        │  memory = 8400 positions
        │  points = [3,6,3]
        │
        ├─ Gateway
        │
        ├─ SwiGLUFFN
        │
        ├─ Class Head
        │  [B,300,80]
        │
        ├─ FDR Head
        │  [B,300,132]
        │
        ├─ Integral
        │  [B,300,4]
        │
        └─ refined bbox
           [B,300,4]
        │
        ▼
FINAL DECODER OUTPUT
pred_logits [B,300,80]
pred_boxes  [B,300,4]
        │
        ▼
POSTPROCESSOR
        │
        ├─ cxcywh → xyxy
        ├─ normalized → pixel
        ├─ sigmoid logits
        ├─ [B,300,80]
        ├─ flatten → [B,24000]
        └─ top-300 query-class pairs
        │
        ▼
FINAL
labels [B,300]
boxes  [B,300,4]
scores [B,300]
```

------------------------------------------------------------------------

# 145. 本课必须记住的 10 个 Shape

``` text
1.
[B,3,640,640]

2.
40×40 = 1600 patches

3.
[B,256,80,80]

4.
[B,256,40,40]

5.
[B,256,20,20]

6.
[B,8400,256]

7.
[B,300,256]

8.
[B,300,80]

9.
[B,300,132]

10.
[B,300,4]
```

以及：

``` text
PostProcessor score space:
[B,24000]
```

------------------------------------------------------------------------

# 146. 本课必须记住的 8 个公式

Patch grid：

\[ H_p=H/16,`\quad `{=tex}W_p=W/16 \]

640：

\[ 40`\times40`{=tex}=1600 \]

Multi-scale memory：

\[ N= `\frac{H}{8}`{=tex}`\frac{W}{8}`{=tex} +
`\frac{H}{16}`{=tex}`\frac{W}{16}`{=tex} +
`\frac{H}{32}`{=tex}`\frac{W}{32}`{=tex} \]

640：

\[ N=6400+1600+400=8400 \]

Head dimension：

\[ d_h=256/8=32 \]

FDR dimension：

\[ 4(reg\_{max}+1) = 4(32+1) = 132 \]

Postprocess score count：

\[ QK=300`\times80`{=tex}=24000 \]

Normalized box → pixel：

\[ x\_{pixel}=x\_{norm}W \]

\[ y\_{pixel}=y\_{norm}H \]

------------------------------------------------------------------------

# 147. 检查题

1.  640×640、patch size 16 时有多少 spatial patches？
2.  为什么不能直接断言 ViT block sequence length 一定就是 1600？
3.  `interaction_indexes=[5,8,11]` 是否意味着三个 block 天然分别是
    stride 8/16/32？
4.  STA 给 HybridEncoder 的三个 shape 是什么？
5.  HybridEncoder 为什么还需要存在？
6.  `use_encoder_idx=[2]` 对哪个 feature level做 Transformer？
7.  为什么该 level flatten 后是 `[B,400,256]`？
8.  FPN/PAN 后三个空间尺寸是否改变？
9.  8400 怎么得到？
10. `spatial_shapes` 为什么不能丢？
11. anchor shape 是什么？
12. encoder class logits shape 是什么？
13. top-k 后 query content shape 是什么？
14. 默认 query content 从哪里来？
15. inference 时有多少 queries？
16. training 时为什么可能多于 300？
17. query position embedding shape 是什么？
18. 8 heads 时每 head dimension 是多少？
19. `[3,6,3]` 表示什么？
20. FDR 为什么输出 132？
21. decoder 最终 logits shape 是什么？
22. decoder 最终 box shape 是什么？
23. decoder box 在 PostProcessor 前是什么坐标格式？
24. PostProcessor 为什么出现 24000？
25. 最终是否需要传统 NMS？
26. 1600、8400、300 分别表示什么？

------------------------------------------------------------------------

# 148. 检查题答案

``` text
1. 40×40=1600。
2. DINOv3 实现可能还有 special/register tokens。
3. 不是；它们是 ViT interaction depth，1/8/1/16/1/32 是 STA detector outputs。
4. [B,256,80,80]、[B,256,40,40]、[B,256,20,20]。
5. STA 做 representation adaptation；HybridEncoder 继续做 multi-scale fusion。
6. index 2，即 20×20 level。
7. 20×20=400。
8. 最终仍为 80×80、40×40、20×20。
9. 80²+40²+20²=8400。
10. Deformable Attention 需要恢复每个 level 的二维几何。
11. [B,8400,4]（broadcast/repeat 后）。
12. [B,8400,80]。
13. [B,300,256]。
14. 默认来自 top-k encoder memory。
15. 300。
16. 训练加入 denoising queries。
17. [B,300,256]。
18. 32。
19. 三个 feature levels 每 head/query 分别采 3、6、3 点。
20. 4×(32+1)=132。
21. [B,300,80]。
22. [B,300,4]。
23. normalized cxcywh。
24. 300 queries × 80 classes。
25. 主流程不依赖传统 NMS。
26. ViT spatial patches、multi-scale memory positions、object query slots。
```

------------------------------------------------------------------------

# 149. 课后实验：必须真正运行

## 实验 A：顶层 Shape Trace

打印：

``` text
backbone output
encoder output
decoder output
postprocessor output
```

------------------------------------------------------------------------

## 实验 B：STA 内部 Trace

打印：

``` text
patch embedding
selected ViT blocks
spatial prior features
interaction outputs
final 1/8,1/16,1/32
```

并特别确认：

``` text
DINOv3 内部真实 embedding width
special/register token 数量
```

------------------------------------------------------------------------

## 实验 C：HybridEncoder Trace

验证：

``` text
20×20
→ [B,400,256]
→ Transformer
→ 20×20

top-down:
20→40→80

bottom-up:
80→40→20
```

------------------------------------------------------------------------

## 实验 D：Top-K 来源统计

``` text
P3:
index < 6400

P4:
6400 <= index < 8000

P5:
index >= 8000
```

统计 100 张 COCO 图片的 top-300 query 来源比例。

------------------------------------------------------------------------

## 实验 E：六层 Box Refinement

保存：

``` text
layer 1 box
layer 2 box
...
layer 6 box
```

对一个 matched object 可视化六个框。

------------------------------------------------------------------------

## 实验 F：FDR Distribution

选一个 query：

``` text
[B,Q,4,33]
```

画四条边的 33-bin probability distribution。

观察 decoder layer 越深时 distribution 是否更集中。

------------------------------------------------------------------------

## 实验 G：PostProcessor

手动复现：

``` text
[B,300,80]
→ sigmoid
→ [B,24000]
→ topk
→ labels
→ query indices
→ gather boxes
```

与官方 PostProcessor 输出逐项比较。

------------------------------------------------------------------------

# 150. 一个很有价值的毕业式调试任务

请你最终生成一个：

``` text
deimv2_dinov3_x_tensor_trace.md
```

里面自动记录：

``` text
Module
Input Shape
Output Shape
DType
Parameter Count
Semantic Meaning
```

目标不是手写。

而是：

``` text
模型跑一次
→ 自动生成
```

这会成为你以后修改 DEIMv2 最有价值的调试工具之一。

------------------------------------------------------------------------

# 151. 本课最终口述答案

学完本课后，你应该能不看讲义说出：

``` text
一张 640×640 RGB 图像首先进入 DINOv3-S+。
patch size 是 16，所以空间 patch grid 是 40×40，也就是 1600 个 spatial patches。

DEIMv2 不直接把这个单尺度 ViT feature 交给检测 decoder，
而是通过 STA，在多个 ViT 深度进行 interaction，
最终形成 stride 8、16、32 的三个 detector features：
[B,256,80,80]、[B,256,40,40]、[B,256,20,20]。

HybridEncoder 对这些 feature 继续做 intra-scale Transformer
以及 top-down FPN、bottom-up PAN 融合，
输出仍然是三个 256-channel 的多尺度 feature maps。

DEIMTransformer 把它们分别 flatten 成
6400、1600、400 个 spatial positions，
拼接得到 [B,8400,256] 的 memory。

8400 个位置先通过 encoder score head 得到 [B,8400,80]，
再选 top-300 candidates。
默认情况下，这 300 个 top-k memory features
直接成为 [B,300,256] 的 query content，
相应 anchors 经 bbox head 修正后成为 [B,300,4] 的初始 reference boxes。

300 个 queries 进入 6 层 DEIMTransformer decoder。
每层先做 query self-attention，
再围绕当前 reference box 使用多尺度 deformable attention
从三个 feature levels 稀疏采样；
X 的 sampling points 是 [3,6,3]。
之后经过 Gateway、SwiGLUFFN、classification head 和 FDR。

reg_max=32，因此 FDR 每个 query 输出
4×33=132 个 distribution logits，
Integral 再把 distribution 转成连续边距离，
distance2bbox 得到 [B,300,4] 的 refined boxes。
新的 box 又成为下一层 cross-attention 的 reference。

最终 decoder 输出 [B,300,80] logits
和 [B,300,4] normalized cxcywh boxes。

PostProcessor 把 box 转成 xyxy 并缩放到原图像素坐标，
同时把 300×80=24000 个 query-class scores 展平，
取 top-300，恢复 label 与 query index，
最终得到每张图的 labels、boxes 和 scores。
```

如果这段话你能够真正理解，而不是背下来，那么：

> **`deimv2_dinov3_x` 的完整 forward 主干，你已经贯通了。**

------------------------------------------------------------------------

# 152. 下一课预告

第 23 课：

# 《训练系统完整解剖------一条 batch 到底经历了什么》

第 22 课追的是：

``` text
image
→ prediction
```

第 23 课将继续追：

``` text
dataset
→ augmentation
→ collate
→ batch
→ model
→ matcher
→ criterion
→ loss
→ backward
→ optimizer.step
→ EMA
→ scheduler
→ checkpoint
→ evaluator
```

也就是说，我们会第一次真正回答：

> 执行 `python train.py -c configs/deimv2/deimv2_dinov3_x_coco.yml`
> 以后，DEIMv2 的一次训练 iteration 究竟发生了什么？

------------------------------------------------------------------------

# 153. 本课源码核对清单

重点对应官方仓库：

``` text
configs/deimv2/deimv2_dinov3_x_coco.yml
configs/base/deimv2.yml

engine/backbone/...
  DINOv3STAs
  DINOv3 / ViT
  STA / interaction modules

engine/deim/hybrid_encoder.py

engine/deim/deim_decoder.py

engine/deim/dfine_decoder.py

engine/deim/dfine_utils.py

engine/deim/deim_utils.py

engine/deim/postprocessor.py
```

当前官方配置 / 源码可以直接确认的关键数字：

``` text
X backbone:
dinov3_vits16plus

interaction_indexes:
[5,8,11]

STA hidden_dim:
256

HybridEncoder:
in_channels=[256,256,256]
hidden_dim=256

feature strides:
[8,16,32]

DEIMTransformer:
hidden_dim=256
num_layers=6
num_queries=300
num_levels=3
num_points=[3,6,3]
reg_max=32
num_denoising=100

COCO:
num_classes=80

PostProcessor:
num_top_queries=300
```

------------------------------------------------------------------------

# 154. 版本阅读原则

以后所有源码学习都遵循：

``` text
runtime tensor.shape
>
当前 commit 的 Python source
>
最终解析后的 config
>
checkpoint 对应 config
>
论文 / README
>
课程讲义
```

原因很简单：

``` text
仓库会更新
实现会重构
默认值会变化
YAML 会 override constructor defaults
```

而 runtime tensor 不会撒谎。

------------------------------------------------------------------------

# 155. 本课一句话总结

``` text
deimv2_dinov3_x 的核心 Tensor 主线是：

640×640 图像
→ 40×40 ViT patch grid
→ STA 构造 80×80 / 40×40 / 20×20 多尺度特征
→ HybridEncoder 融合
→ flatten 为 8400 个 memory positions
→ top-k 选 300 个 image-conditioned object queries
→ 6 层 DEIMTransformer 通过 self-attention、
多尺度 deformable cross-attention 和 FDR 反复精炼
→ [B,300,80] 分类 + [B,300,4] box
→ PostProcessor 从 24000 个 query-class scores 中取最终 top-300，
并把归一化 cxcywh 转成原图像素 xyxy。
```

------------------------------------------------------------------------

# 参考资料

本课按官方仓库当前公开实现核对：

``` text
DEIMv2 repository
https://github.com/Intellindust-AI-Lab/DEIMv2

X config
https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/configs/deimv2/deimv2_dinov3_x_coco.yml

HybridEncoder
https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/engine/deim/hybrid_encoder.py

DEIMTransformer
https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/engine/deim/deim_decoder.py

PostProcessor
https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/engine/deim/postprocessor.py
```

> **建议**：正式做本课实验时，先记录你本地仓库的 commit
> hash。后续课程中的所有 shape trace、实验结果和源码行号都与这个 commit
> 绑定，避免仓库更新后出现"讲义和代码怎么不一样"的困惑。
