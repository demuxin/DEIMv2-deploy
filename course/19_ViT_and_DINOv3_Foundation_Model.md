# DEIMv2 源码课程 · 第 19 课

# ViT 与 DINOv3——从 CNN 世界切换到 Foundation Model

> 本课目标：从 CNN 的 `[B,C,H,W]` 思维切换到 ViT / DINOv3 的 token 思维，并为下一课 **STA（Spatial Tuning Adapter）** 做完整铺垫。
>
> 本课重点模型：`deimv2_dinov3_x`。
>
> **一个必须先纠正的关键 shape：** `deimv2_dinov3_x` 使用的 `dinov3_vits16plus` 原生 embedding dimension 是 **384**，不是 256。配置里的 `hidden_dim: 256` 是 DEIMv2 的 STA / detection pipeline 使用的通道维度。也就是说：
>
> ```text
> DINOv3-S+ native feature: 384 channels
>                  ↓ STA / adaptation
> DEIMv2 detector feature: 256 channels
> ```
>
> 这个区别是后面读 `DINOv3STAs` 源码时最容易混淆的地方之一。

---

# 1. 为什么第 19 课必须先学 ViT，而不是直接读 STA？

到第 18 课为止，我们已经理解了传统检测路线：

```text
Image
  ↓
CNN Backbone
  ↓
C3 / C4 / C5
  ↓
HybridEncoder
  ↓
Transformer Decoder
  ↓
Detection
```

例如 HGNetv2 天然会随着 stage 加深产生：

```text
stride 8
stride 16
stride 32
```

因此检测器天然可以拿到多尺度 feature maps。

但是 `deimv2_dinov3_x` 的 backbone 换成了：

```text
DINOv3 ViT-S+
```

ViT 的基本世界不是：

```text
[B,C,H,W]
```

而是：

```text
[B,N,D]
```

其中：

```text
B = batch size
N = token 数
D = embedding dimension
```

这会带来一个根本矛盾：

```text
检测器喜欢：
P3 = 1/8
P4 = 1/16
P5 = 1/32

DINOv3 ViT 原生 patch grid：
1/16
```

所以，在理解 STA 之前，必须先回答：

1. 一张图片为什么会变成 token？
2. patch embedding 到底是不是卷积？
3. token 为什么能保留空间位置？
4. ViT block 为什么不会像 CNN 那样自动产生 1/8、1/16、1/32？
5. DINOv3 为什么值得替代传统 CNN backbone？
6. DEIMv2 为什么从 DINOv3 的第 5、8、11 个 block 取特征？
7. 为什么三个 block 的输出仍然不能直接当作 P3/P4/P5？
8. `384 → 256` 到底发生在哪里？

本课就是解决这些问题。

---

# 2. 先回忆 CNN 的世界

假设输入：

```text
x = [B,3,640,640]
```

CNN backbone 可能：

```text
[B,3,640,640]
       ↓ stride 2
[B,64,320,320]
       ↓ stride 2
[B,128,160,160]
       ↓ stride 2
[B,256,80,80]
       ↓ stride 2
[B,512,40,40]
       ↓ stride 2
[B,1024,20,20]
```

于是天然产生：

```text
80×80 → stride 8
40×40 → stride 16
20×20 → stride 32
```

这叫：

```text
hierarchical spatial representation
```

也就是空间分辨率随着网络深度逐渐下降。

---

# 3. CNN 的层级结构为什么特别适合目标检测？

目标检测同时需要：

```text
WHAT?
这是什么物体？

WHERE?
物体在哪里？
```

浅层高分辨率 feature：

```text
80×80
```

更容易保留：

```text
edge
corner
texture
precise position
small object
```

深层低分辨率 feature：

```text
20×20
```

通常语义更强：

```text
object category
global context
large receptive field
```

所以检测器喜欢：

```text
high resolution + low semantics
            +
low resolution + high semantics
```

然后用 FPN / PAN / HybridEncoder 融合。

---

# 4. ViT 做了一件完全不同的事情

Vision Transformer 的核心思想：

```text
Image
  ↓
split into patches
  ↓
each patch → vector
  ↓
patch vectors become tokens
  ↓
Transformer blocks
```

也就是说：

```text
image pixels
```

先被转换成：

```text
token sequence
```

再进入 Transformer。

---

# 5. 640×640 图片怎样切成 patch？

`dinov3_vits16plus`：

```text
patch_size = 16
```

所以：

```text
640 / 16 = 40
```

水平方向 40 个 patch。

垂直方向也是：

```text
640 / 16 = 40
```

因此 patch grid：

```text
40 × 40
```

patch token 数：

$$
N_p = 40\times40=1600
$$

所以：

```text
[B,3,640,640]
        ↓ patch embedding
[B,1600,D]
```

对于 DINOv3-S+：

```text
D = 384
```

于是 patch tokens：

```text
[B,1600,384]
```

---

# 6. 一个 patch 里面到底有多少数字？

一个 patch：

```text
3 × 16 × 16
```

总元素：

$$
3\times16\times16=768
$$

最直观的理解是：

```text
[3,16,16]
    ↓ flatten
[768]
    ↓ linear projection
[384]
```

因此每个 patch：

```text
768 raw pixel values
        ↓
384-dimensional embedding
```

---

# 7. Patch Embedding 真的是逐 patch `Linear(768,384)` 吗？

数学上可以这样理解。

但工程实现通常使用：

```python
nn.Conv2d(
    in_channels=3,
    out_channels=384,
    kernel_size=16,
    stride=16
)
```

为什么？

因为：

```text
kernel = patch size
stride = patch size
```

卷积窗口不会重叠。

每一个输出 spatial position：

```text
正好对应一个 16×16 patch
```

---

# 8. DINOv3 官方 PatchEmbed 的核心结构

DINOv3 官方实现的核心逻辑可以概括为：

```python
self.proj = nn.Conv2d(
    in_chans,
    embed_dim,
    kernel_size=patch_size,
    stride=patch_size
)
```

forward：

```python
x = self.proj(x)
x = x.flatten(2).transpose(1, 2)
x = self.norm(x)
```

注意：

```text
Conv2d
→ flatten
→ transpose
```

正好就是第 2 课学过的 shape 操作。

---

# 9. 逐行推导 PatchEmbed

输入：

```text
x
[B,3,640,640]
```

执行：

```python
x = self.proj(x)
```

因为：

```text
kernel = 16
stride = 16
out_channels = 384
```

输出：

```text
[B,384,40,40]
```

然后：

```python
x = x.flatten(2)
```

把：

```text
40×40
```

压成：

```text
1600
```

得到：

```text
[B,384,1600]
```

然后：

```python
x = x.transpose(1, 2)
```

得到：

```text
[B,1600,384]
```

最终：

```text
1600 spatial patches
×
384-dimensional embedding
```

---

# 10. 为什么 Transformer 喜欢 `[B,N,D]`？

Transformer 原本处理：

```text
word token sequence
```

例如 NLP：

```text
I
love
computer
vision
```

转换成：

```text
[token1, token2, token3, token4]
```

Vision Transformer 做类似的事情：

```text
patch(0,0)
patch(0,1)
patch(0,2)
...
patch(39,39)
```

每个 patch 就相当于一个“视觉单词”。

于是：

```text
N = sequence length
D = embedding dimension
```

---

# 11. Patch Token 并不是 Object Query

这是一个非常重要的区别。

Patch token：

```text
来自图片
```

表示：

```text
某个图像区域的视觉特征
```

Object query：

```text
属于 detector decoder
```

用于：

```text
主动寻找一个目标
```

所以：

```text
patch token ≠ object query
```

在 DEIMv2 中：

```text
DINOv3 patch tokens
        ↓
STA
        ↓
HybridEncoder
        ↓
image memory
        ↓
DEIMTransformer
        ↑
object queries
```

两者最终在 decoder cross-attention 中发生交互。

---

# 12. 1600 token 的空间顺序是什么？

假设：

```text
40×40 patch grid
```

flatten 后通常类似：

```text
(0,0)
(0,1)
...
(0,39)
(1,0)
...
(39,39)
```

即：

```text
2D grid
 ↓ flatten
1D sequence
```

但只保留 sequence index 本身还不够。

为什么？

因为 Transformer 的 self-attention 本身并不知道：

```text
token 37 在 token 38 左边
```

还是：

```text
token 37 在 token 38 上面
```

因此需要：

```text
position information
```

---

# 13. 位置编码为什么必要？

假设两个 patch：

```text
patch A = 天空
patch B = 汽车
```

视觉内容告诉网络：

```text
A 是天空
B 是汽车
```

但检测还需要知道：

```text
天空在哪里？
汽车在哪里？
```

因此 token 必须同时携带：

```text
content
+
position
```

---

# 14. 经典 ViT 的位置编码

经典 ViT 常见：

```text
x_i = patch_embedding_i + position_embedding_i
```

形式：

$$
z_i = e_i + p_i
$$

其中：

```text
e_i = patch content embedding
p_i = position embedding
```

---

# 15. DINOv3 的位置机制：RoPE

当前 DINOv3 ViT 实现使用：

```text
RoPEPositionEmbedding
```

即 Rotary Position Embedding 思路。

DINOv3 模型中会根据：

```text
H
W
```

生成二维空间相关的 RoPE 信息，并在 attention block 中使用。

因此不要把 DINOv3 简化成：

```text
普通 ViT + learned absolute position embedding
```

它的当前实现更接近：

```text
patch tokens
+
2D RoPE spatial information
```

---

# 16. 为什么 RoPE 对不同图像尺寸很重要？

DEIMv2 目标检测不会永远只看：

```text
224×224
```

而可能训练：

```text
640×640
```

甚至多尺度：

```text
480
512
...
800
```

如果位置机制过度绑定固定 token grid：

```text
14×14
```

迁移到：

```text
40×40
```

会比较麻烦。

RoPE 可以根据当前：

```text
H × W
```

构造位置关系，因此更自然地支持不同 spatial grid。

---

# 17. DINOv3-S+ 的真实结构参数

DEIMv2 X 配置：

```yaml
DINOv3STAs:
  name: dinov3_vits16plus
```

DINOv3 官方 `dinov3_vits16plus` 当前定义的关键参数：

```text
patch_size       = 16
embed_dim        = 384
depth            = 12
num_heads        = 6
ffn_ratio        = 6
ffn_layer        = SwiGLU
n_storage_tokens = 4
```

所以它不是：

```text
hidden_dim = 256
```

原生 ViT token dimension 是：

```text
384
```

---

# 18. 每个 attention head 的 dimension

总 embedding：

```text
384
```

heads：

```text
6
```

所以：

$$
d_h=\frac{384}{6}=64
$$

每个 head：

```text
64 dimensions
```

因此 Q/K/V reshape 时：

```text
[B,N,384]
↓
[B,N,6,64]
↓ permute
[B,6,N,64]
```

---

# 19. 重新连接第 4 课的 Q/K/V

输入 token：

$$
X\in\mathbb{R}^{B\times N\times384}
$$

线性投影：

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

拆 6 heads：

```text
Q,K,V
[B,6,N,64]
```

单个 head：

$$
\mathrm{Attention}(Q,K,V)
=
\mathrm{softmax}\left(
\frac{QK^T}{\sqrt{64}}
\right)V
$$

其中：

$$
\sqrt{64}=8
$$

---

# 20. 如果只有 1600 patch tokens，全局 attention matrix 多大？

每个 head：

```text
Q:
[1600,64]

Kᵀ:
[64,1600]
```

于是：

```text
QKᵀ
=
[1600,1600]
```

元素数量：

$$
1600^2=2,560,000
$$

6 heads：

$$
6\times2,560,000
=
15,360,000
$$

这还只是一个 batch sample、一个 block 中 attention score 的数量级直觉。

因此 ViT 的全局 attention 在高分辨率视觉任务中计算并不便宜。

---

# 21. 但是 DINOv3-S+ 实际 token 数不只是 1600

这是本课一个非常重要的源码细节。

DINOv3-S+ 当前配置：

```text
n_storage_tokens = 4
```

另外还有：

```text
1 CLS token
```

所以进入 Transformer block 的 sequence：

```text
1 CLS
+
4 storage tokens
+
1600 patch tokens
```

总计：

$$
N=1605
$$

所以内部 token tensor 是：

```text
[B,1605,384]
```

而不是简单的：

```text
[B,1600,384]
```

---

# 22. CLS Token 是什么？

CLS：

```text
classification token
```

它是一个 learnable token。

初始化类似：

```text
[1,1,384]
```

batch 展开：

```text
[B,1,384]
```

然后和 patch tokens 拼接：

```text
CLS
+
patch tokens
```

经过 self-attention 后，CLS 可以从整张图的 patch tokens 聚合全局信息。

---

# 23. 目标检测真的需要 CLS Token 吗？

检测需要的是：

```text
dense spatial features
```

所以通常最重要的是：

```text
patch tokens
```

而不是只取：

```text
CLS token
```

DINOv3 官方 `get_intermediate_layers()` 会把：

```text
CLS
extra/storage tokens
patch tokens
```

区分开。

DEIMv2 后续 STA 关注的是适合空间检测的 feature representation。

---

# 24. Storage Tokens 是什么？

DINOv3 当前 ViT 中还有：

```text
storage tokens
```

S+ 使用 4 个。

你可以先把它们理解为：

```text
额外的 learnable global tokens
```

它们不是图像 patch。

因此：

```text
token 0
→ CLS

token 1..4
→ storage tokens

token 5..
→ patch tokens
```

对于 640×640：

```text
[B,1605,384]
```

其中 patch 部分：

```text
[B,1600,384]
```

---

# 25. 为什么这件事对源码阅读很重要？

如果你看到：

```python
x[:, self.n_storage_tokens + 1:]
```

不要疑惑。

这里：

```text
+1
```

是跳过 CLS。

```text
n_storage_tokens
```

是跳过 storage tokens。

剩下的才是：

```text
patch tokens
```

---

# 26. DINOv3 官方 prepare_tokens 的逻辑

当前源码核心逻辑可以概括成：

```python
x = self.patch_embed(x)

B, H, W, _ = x.shape

x = x.flatten(1, 2)

x = torch.cat(
    [
        cls_token,
        storage_tokens,
        x,
    ],
    dim=1,
)
```

对于 640×640 S+：

```text
patch_embed
[B,40,40,384]

flatten
[B,1600,384]

CLS
[B,1,384]

storage
[B,4,384]

cat
[B,1605,384]
```

---

# 27. 为什么官方 PatchEmbed 有时返回 `[B,H,W,C]`？

DINOv3 的 `DinoVisionTransformer` 构造 PatchEmbed 时当前设置：

```text
flatten_embedding=False
```

所以 patch embed 可以先保留：

```text
[B,H,W,C]
```

例如：

```text
[B,40,40,384]
```

随后 `prepare_tokens_with_masks()` 再：

```python
x = x.flatten(1, 2)
```

得到：

```text
[B,1600,384]
```

这和直接输出 `[B,N,D]` 数学上等价，只是源码组织不同。

---

# 28. 一个 ViT Block 里面有什么？

最核心：

```text
x
│
├─ Norm
│
├─ Self-Attention
│
└─ Residual
     │
     ├─ Norm
     ├─ FFN / SwiGLU
     └─ Residual
```

抽象公式：

$$
X' = X + Attention(Norm(X))
$$

$$
X'' = X' + FFN(Norm(X'))
$$

DINOv3 当前 block 还包含 LayerScale 等工程细节，但主干思想就是：

```text
attention
+
FFN
+
residual
+
normalization
```

---

# 29. Self-Attention 在视觉里意味着什么？

CNN 的卷积：

```text
一个位置主要和局部邻域交互
```

例如 3×3：

```text
只看周围 9 个位置
```

全局 self-attention：

```text
一个 patch token
```

可以直接与：

```text
所有 patch tokens
```

交互。

所以一个汽车 patch：

```text
可以直接关注
道路
轮胎
其他车辆
天空
建筑
```

即使它们空间距离很远。

---

# 30. CNN 与 ViT 的 inductive bias

CNN 天生假设：

```text
locality
translation equivariance
hierarchy
```

ViT 的结构先验相对更弱。

它更依赖：

```text
large-scale data
pretraining
self-supervised learning
```

来学出视觉规律。

这正是 foundation model 思路的重要背景。

---

# 31. 什么叫 Foundation Model？

不要把它理解成：

```text
“参数很大的模型”
```

更核心的是：

```text
在大规模、多样化数据上预训练
        ↓
学习通用 representation
        ↓
迁移到许多下游任务
```

例如：

```text
classification
detection
segmentation
depth
retrieval
```

DINOv3 的价值就在这里。

---

# 32. 什么叫 Self-Supervised Learning？

传统 supervised：

```text
image
+
human label
```

例如：

```text
dog
car
person
```

Self-supervised 的核心是：

```text
从数据本身构造学习信号
```

不要求每张图片都有人手工提供类别标签。

DINO 系列通过 teacher/student、自蒸馏以及一系列表征学习目标，让模型学到强视觉特征。

---

# 33. 为什么 self-supervised feature 对检测有价值？

分类监督容易鼓励模型回答：

```text
“整张图是什么？”
```

但目标检测需要：

```text
哪里是物体？
边界在哪里？
不同区域是什么？
```

强大的自监督视觉 representation 往往可以学到：

```text
object structure
semantic similarity
spatial organization
part-whole relation
```

这些特征非常适合迁移到 dense prediction。

---

# 34. DINOv3 在 DEIMv2 中扮演什么角色？

不是：

```text
完整 detector
```

而是：

```text
foundation visual backbone
```

整体：

```text
Image
  ↓
DINOv3
  ↓
strong visual representation
  ↓
STA
  ↓
detection-friendly multi-scale features
  ↓
HybridEncoder
  ↓
DEIMTransformer
```

所以：

```text
DINOv3 ≠ DEIMv2
```

DINOv3 只是 DEIMv2 的 backbone 核心。

---

# 35. 为什么不能把 DINOv3 输出直接塞给 HybridEncoder？

HybridEncoder 期待类似：

```text
P3 [B,C,H/8,W/8]
P4 [B,C,H/16,W/16]
P5 [B,C,H/32,W/32]
```

而 DINOv3-S+ patch size：

```text
16
```

原生 patch grid：

```text
H/16 × W/16
```

640 输入：

```text
40×40
```

所以原生空间尺度只有：

```text
stride 16
```

这就是矛盾。

---

# 36. ViT 深度增加，不代表空间分辨率下降

这是 CNN 初学者切换到 ViT 时最容易犯的错误。

CNN：

```text
stage 1 → 80×80
stage 2 → 40×40
stage 3 → 20×20
```

ViT：

```text
block 0  → 40×40 tokens
block 1  → 40×40 tokens
block 2  → 40×40 tokens
...
block 11 → 40×40 tokens
```

空间 token 数通常保持：

```text
1600
```

变化的是：

```text
representation depth / semantics
```

不是 spatial resolution。

---

# 37. 所以 block 5、8、11 不是天然的 P3/P4/P5

这是必须彻底理解的结论。

DEIMv2 X：

```yaml
interaction_indexes: [5, 8, 11]
```

很多初学者会误解：

```text
block 5  → stride 8
block 8  → stride 16
block 11 → stride 32
```

这是错的。

原生 DINOv3 ViT 中，这三个 block 的 patch feature 都仍然基于：

```text
40×40
```

即：

```text
stride 16
```

只是语义深度不同。

---

# 38. 那配置注释为什么写 `[1/8,1/16,1/32]`？

X 配置：

```yaml
interaction_indexes: [5,8,11]
# only need the [1/8, 1/16, 1/32]
```

这里要结合：

```text
DINOv3STAs
```

理解。

不是说：

```text
DINO block 5 自己就是 1/8
```

而是这些 interaction stages 被 STA 用来构造/交互最终检测所需的：

```text
1/8
1/16
1/32
```

多尺度输出。

**真正完成尺度转换的是 STA 系统，而不是 ViT block 深度本身。**

下一课会逐行拆。

---

# 39. 为什么选择中间层？

假设只取最后一层：

```text
block 11
```

虽然语义很强，但会丢掉：

```text
不同深度 representation
```

ViT 不同层会形成不同层次的视觉表示：

```text
earlier
→ lower/mid-level visual structures

middle
→ richer object/part relations

late
→ stronger high-level semantics
```

因此中间层特征对于 dense prediction 很有价值。

---

# 40. DINOv3 官方也提供 `get_intermediate_layers`

当前官方 API 可以：

```python
outputs = backbone.get_intermediate_layers(
    x,
    n=[...],
    reshape=True,
)
```

它可以从指定 block 抽取 feature。

当：

```text
reshape=True
```

patch tokens 可以恢复为：

```text
[B,C,h,w]
```

---

# 41. token 怎样恢复成 feature map？

patch tokens：

```text
[B,1600,384]
```

已知：

```text
1600 = 40×40
```

先：

```text
[B,40,40,384]
```

再 permute：

```text
[B,384,40,40]
```

即：

```python
x = x.reshape(
    B, 40, 40, 384
)

x = x.permute(
    0, 3, 1, 2
)
```

得到 CNN 风格 feature map。

---

# 42. 这意味着 ViT token 并没有“失去二维空间”

flatten：

```text
40×40
→
1600
```

只是改变 tensor 表示方式。

只要知道：

```text
H=40
W=40
```

就可以恢复：

```text
[B,1600,384]
↔
[B,384,40,40]
```

所以：

```text
token representation
```

和：

```text
feature-map representation
```

可以互相转换。

---

# 43. 但是 token reshape 回 feature map 后就等于 CNN feature 吗？

不等于。

虽然 shape 一样：

```text
[B,C,H,W]
```

但 feature 的生成机制不同。

CNN：

```text
local convolution
hierarchical downsampling
```

ViT：

```text
patch projection
global token interaction
Transformer blocks
```

所以：

```text
same tensor shape
≠
same representation
```

---

# 44. 640×640 下 DINOv3-S+ 的完整 shape 起点

现在完整写一次：

```text
Input
[B,3,640,640]

        ↓ PatchEmbed
        Conv2d(3→384,k=16,s=16)

[B,384,40,40]

        ↓ internal layout

[B,40,40,384]

        ↓ flatten patches

[B,1600,384]

        ↓ prepend
        CLS + 4 storage tokens

[B,1605,384]

        ↓ ViT block × 12

[B,1605,384]

        ↓ remove CLS/storage

[B,1600,384]

        ↓ reshape

[B,384,40,40]
```

这张 shape 图必须记住。

---

# 45. Block 5 / 8 / 11 的 shape

忽略 adapter 交互，仅从原生 ViT 主干看：

```text
block 5:
[B,1605,384]
patch part → [B,1600,384]

block 8:
[B,1605,384]
patch part → [B,1600,384]

block 11:
[B,1605,384]
patch part → [B,1600,384]
```

reshape：

```text
block 5  → [B,384,40,40]
block 8  → [B,384,40,40]
block 11 → [B,384,40,40]
```

再次强调：

```text
空间尺寸完全相同
```

---

# 46. 那么 DEIMv2 最终为什么能得到 80×80 / 40×40 / 20×20？

因为：

```text
STA
```

承担了：

```text
single-scale ViT
        ↓
multi-scale detector features
```

的转换。

目标是：

```text
[B,256,80,80]
[B,256,40,40]
[B,256,20,20]
```

这正是下一课的核心。

---

# 47. 384 → 256 也不是 PatchEmbed 做的

再次区分：

DINOv3：

```text
embed_dim = 384
```

DEIMv2 X：

```yaml
DINOv3STAs:
  hidden_dim: 256
```

HybridEncoder：

```yaml
in_channels:
  [256,256,256]
```

所以接口目标是：

```text
DINO representation
384 channels
        ↓
STA/adaptation
256-channel multi-scale features
        ↓
HybridEncoder
```

这是一条非常重要的 channel bridge。

---

# 48. 为什么 detector 喜欢统一成 256 channels？

如果三个尺度：

```text
P3 C=256
P4 C=256
P5 C=256
```

那么后续：

```text
feature fusion
flatten
positional encoding
deformable attention
decoder projection
```

都更统一。

Transformer detector 中：

```text
hidden_dim = 256
```

也是非常常见的设计。

---

# 49. DINOv3-S+ 为什么叫 S+？

不要把：

```text
S+
```

理解成：

```text
只是 S 多几个 block
```

当前官方定义：

`dinov3_vits16`：

```text
embed_dim = 384
depth = 12
heads = 6
ffn_ratio = 4
FFN = MLP
```

`dinov3_vits16plus`：

```text
embed_dim = 384
depth = 12
heads = 6
ffn_ratio = 6
FFN = SwiGLU
```

也就是说 S+ 的宽度和深度仍然是：

```text
384
12 blocks
```

但 FFN 结构/容量增强。

---

# 50. FFN 是干什么的？

Attention：

```text
让不同 tokens 交换信息
```

FFN：

```text
对每个 token 的 channel representation
进行非线性变换
```

经典形式：

$$
FFN(x)
=
W_2\sigma(W_1x)
$$

如果：

```text
D = 384
ffn_ratio = 6
```

中间维度的量级：

$$
384\times6=2304
$$

所以可以粗略理解：

```text
384
 ↓ expand
2304
 ↓ nonlinear/gating
384
```

具体 SwiGLU 的内部投影方式以实现为准。

---

# 51. 什么是 SwiGLU？

普通 MLP：

```text
Linear
→ GELU
→ Linear
```

SwiGLU 属于 gated FFN：

```text
一支生成内容
一支生成 gate
      ↓
element-wise modulation
```

典型思想：

$$
SwiGLU(x)
=
Swish(xW_a)\odot(xW_b)
$$

然后再投影回 embedding dimension。

你现在不需要记住所有实现细节，只需知道：

```text
S+ 使用更强的 FFN 结构
```

即可。

---

# 52. LayerNorm 与 CNN BatchNorm 的区别

BatchNorm：

```text
统计 batch / spatial dimension
```

LayerNorm：

```text
对单个 token 的 feature dimension 归一化
```

对于：

```text
x[b,n,:]
```

LayerNorm 在：

```text
384-dimensional vector
```

上做 normalization。

所以它不依赖：

```text
batch size
```

这也是 ViT 很常使用 LayerNorm 的原因。

---

# 53. LayerNorm 数学

对于 token：

$$
x\in\mathbb{R}^{384}
$$

均值：

$$
\mu=\frac1D\sum_{i=1}^D x_i
$$

方差：

$$
\sigma^2
=
\frac1D
\sum_{i=1}^D
(x_i-\mu)^2
$$

归一化：

$$
\hat x_i
=
\frac{x_i-\mu}
{\sqrt{\sigma^2+\epsilon}}
$$

再：

$$
y_i=\gamma_i\hat x_i+\beta_i
$$

---

# 54. Residual Connection 为什么重要？

如果每个 block：

```text
x → complicated transform
```

直接替换 x，深层训练会困难。

Residual：

$$
y=x+F(x)
$$

让 block 学：

```text
对当前 representation 的增量修正
```

而不是每层都从头重建。

12 个 ViT blocks 因此可以逐步 refine representation。

---

# 55. LayerScale 又是什么？

DINOv3-S+ 当前定义：

```text
layerscale_init = 1e-5
```

直觉上可以理解为：

$$
y=x+\gamma F(x)
$$

其中：

```text
γ
```

初始很小。

这样深层 Transformer 训练初期：

```text
每个 residual branch 的扰动较小
```

有利于稳定优化。

---

# 56. 为什么第 18 课 DINOv3 要用更小 LR？

现在可以从结构角度理解了。

DINOv3 已经有：

```text
12 blocks
384-d representations
pretrained attention
pretrained FFN
pretrained spatial/semantic organization
```

DEIMv2 不希望 COCO detection 一开始就大幅破坏这些 representation。

所以 X：

```text
DINOv3 LR = 1e-5
detector LR = 5e-4
```

正是：

```text
foundation backbone
      ↓ gentle finetuning

new detection modules
      ↓ aggressive adaptation
```

---

# 57. Foundation Model 与传统 ImageNet Backbone 的差别

传统 detector 常见：

```text
ResNet
HGNetv2
ConvNeXt
```

可能用 ImageNet classification pretraining。

Foundation vision model 更强调：

```text
大规模数据
自监督 / 通用 representation
多任务迁移能力
```

因此下游 detector 的设计问题变成：

```text
如何最大限度利用已经很强的 representation？
```

而不是：

```text
从头训练一个 detection backbone
```

---

# 58. 这会改变 detector architecture 的思考方式

传统：

```text
设计一个更强 CNN backbone
```

Foundation era：

```text
已有一个强 pretrained ViT
        ↓
怎样把它适配到 detection？
```

于是新的关键问题是：

```text
adapter
feature extraction
multi-scale reconstruction
spatial detail
efficient finetuning
```

STA 就是在这个背景下出现的。

---

# 59. 为什么“语义强”不等于“检测友好”？

假设 DINOv3 非常擅长判断：

```text
这块区域像一辆车
```

但检测还需要：

```text
车的左边界在哪？
右边界在哪？
小车占几个像素？
```

因此需要：

```text
fine-grained spatial information
```

这就是：

```text
semantic quality
```

和：

```text
spatial localization quality
```

之间的区别。

---

# 60. Patch Size 16 对小目标有什么潜在问题？

输入：

```text
640×640
```

一个 patch：

```text
16×16 pixels
```

如果一个目标只有：

```text
10×10 pixels
```

它甚至小于一个 patch。

CNN stride-8 feature：

```text
一个 cell 对应约 8×8 input region
```

比 stride-16 更细。

因此 detection 希望恢复：

```text
1/8
```

的高分辨率特征。

---

# 61. 为什么不直接把 patch size 改成 8？

理论上可以设计 patch-8 ViT。

但 token 数：

patch 16：

$$
40\times40=1600
$$

patch 8：

$$
80\times80=6400
$$

全局 attention complexity：

$$
O(N^2)
$$

比值：

$$
\left(
\frac{6400}{1600}
\right)^2
=
16
$$

attention score 数量级会放大约：

```text
16×
```

所以直接把 patch size 减半非常昂贵。

---

# 62. 这就是 STA 思路的价值之一

我们希望：

```text
保留 DINOv3 patch-16 的强 representation
```

同时：

```text
恢复 detection 所需的细粒度、多尺度空间 feature
```

而不是把整个 ViT 都运行在：

```text
80×80 = 6400 tokens
```

上。

---

# 63. CNN 与 ViT 的 receptive field 思维差异

CNN：

```text
3×3
→ local
```

随着网络加深：

```text
effective receptive field gradually grows
```

ViT global self-attention：

```text
第一层开始
一个 token 就能和所有 token 交互
```

因此理论 receptive field：

```text
global
```

但：

```text
global context
≠
precise multi-scale spatial hierarchy
```

这两个概念不能混淆。

---

# 64. 一个很重要的结论

ViT 的优势：

```text
global semantic interaction
```

CNN hierarchy 的优势：

```text
natural multi-scale spatial structure
```

DEIMv2 的设计目标可以粗略理解为：

```text
DINOv3 semantics
        +
CNN-like detection pyramid
```

STA 就是连接这两个世界的桥。

---

# 65. PyTorch 实验 1：手写 Patch Embedding

```python
import torch
import torch.nn as nn

B = 2

x = torch.randn(
    B, 3, 640, 640
)

patch_embed = nn.Conv2d(
    3,
    384,
    kernel_size=16,
    stride=16,
)

y = patch_embed(x)

print(y.shape)
```

输出：

```text
torch.Size([2,384,40,40])
```

---

# 66. 实验 2：Feature Map → Tokens

```python
tokens = (
    y
    .flatten(2)
    .transpose(1, 2)
)

print(tokens.shape)
```

输出：

```text
[2,1600,384]
```

验证：

```python
assert 1600 == 40 * 40
```

---

# 67. 实验 3：Tokens → Feature Map

```python
B, N, D = tokens.shape

feature = (
    tokens
    .reshape(B, 40, 40, D)
    .permute(0, 3, 1, 2)
    .contiguous()
)

print(feature.shape)
```

输出：

```text
[2,384,40,40]
```

所以：

```text
[B,C,H,W]
↔
[B,N,C]
```

只是 representation layout 的变化。

---

# 68. 实验 4：验证 patch projection 与 Linear 的关系

取一个 patch：

```python
patch = x[
    :,
    :,
    0:16,
    0:16
]

print(patch.shape)
```

得到：

```text
[B,3,16,16]
```

flatten：

```python
flat = patch.flatten(1)

print(flat.shape)
```

得到：

```text
[B,768]
```

Conv2d 一个 output channel 的 kernel 本质上也是：

```text
[3,16,16]
```

所以它与对 patch 做线性投影是同一个数学结构。

---

# 69. 实验 5：加入 CLS + Storage Tokens

```python
B = 2
D = 384

patch_tokens = torch.randn(
    B, 1600, D
)

cls_token = nn.Parameter(
    torch.randn(1, 1, D)
)

storage_tokens = nn.Parameter(
    torch.randn(1, 4, D)
)

tokens = torch.cat(
    [
        cls_token.expand(
            B, -1, -1
        ),
        storage_tokens.expand(
            B, -1, -1
        ),
        patch_tokens,
    ],
    dim=1,
)

print(tokens.shape)
```

输出：

```text
[B,1605,384]
```

---

# 70. 实验 6：重新取回 Patch Tokens

```python
n_storage = 4

patch_only = tokens[
    :,
    n_storage + 1:
]

print(patch_only.shape)
```

输出：

```text
[B,1600,384]
```

这正对应 DINOv3 中：

```text
跳过 CLS + storage tokens
```

的逻辑。

---

# 71. 实验 7：手写 Multi-Head QKV shape

```python
B = 2
N = 1605
D = 384
H = 6
Dh = D // H

x = torch.randn(
    B, N, D
)

qkv = nn.Linear(
    D,
    3 * D
)(x)

print(qkv.shape)
```

得到：

```text
[B,1605,1152]
```

然后：

```python
qkv = qkv.reshape(
    B,
    N,
    3,
    H,
    Dh,
)

qkv = qkv.permute(
    2, 0, 3, 1, 4
)

q, k, v = qkv.unbind(0)

print(q.shape)
print(k.shape)
print(v.shape)
```

得到：

```text
[B,6,1605,64]
```

---

# 72. 实验 8：Attention Matrix Shape

```python
scores = (
    q @ k.transpose(-2, -1)
)

print(scores.shape)
```

得到：

```text
[B,6,1605,1605]
```

这就是全局 attention 的主要二次复杂度来源。

---

# 73. 实验 9：计算 token 数对 attention 的影响

```python
def attention_elements(
    h,
    w,
    patch,
    heads=6,
):
    n = (
        h // patch
    ) * (
        w // patch
    )

    return (
        n,
        heads * n * n
    )

for p in [32, 16, 8]:
    n, elements = (
        attention_elements(
            640, 640, p
        )
    )

    print(
        "patch",
        p,
        "tokens",
        n,
        "attn elements",
        elements
    )
```

你会看到：

```text
patch 越小
token 数快速增加
attention 二次增长
```

---

# 74. 实验 10：为什么 block 深度不会改变 spatial size？

做一个 toy ViT：

```python
import torch
import torch.nn as nn

layer = nn.TransformerEncoderLayer(
    d_model=384,
    nhead=6,
    dim_feedforward=2304,
    batch_first=True,
)

x = torch.randn(
    2, 1600, 384
)

for i in range(12):
    x = layer(x)
    print(
        i,
        x.shape
    )
```

每一层仍然：

```text
[2,1600,384]
```

所以：

```text
deeper block
≠
lower spatial resolution
```

---

# 75. 实验 11：模拟 block 5 / 8 / 11

```python
features = {}

x = torch.randn(
    2, 1600, 384
)

for i in range(12):
    x = layer(x)

    if i in [5, 8, 11]:
        features[i] = x

for k, v in features.items():
    print(
        k,
        v.shape
    )
```

结果：

```text
5  [2,1600,384]
8  [2,1600,384]
11 [2,1600,384]
```

这就是为什么：

```text
interaction_indexes
```

不是：

```text
spatial stride indexes
```

---

# 76. 实验 12：恢复三个同尺度 feature maps

```python
for idx, t in features.items():

    f = (
        t.reshape(
            2,
            40,
            40,
            384
        )
        .permute(
            0, 3, 1, 2
        )
        .contiguous()
    )

    print(
        idx,
        f.shape
    )
```

得到：

```text
5  [2,384,40,40]
8  [2,384,40,40]
11 [2,384,40,40]
```

这就是下一课 STA 的输入矛盾。

---

# 77. 实验 13：最粗糙的“伪多尺度”

为了理解目标，不代表 DEIMv2 真实实现：

```python
import torch.nn.functional as F

f16 = torch.randn(
    2, 384, 40, 40
)

f8 = F.interpolate(
    f16,
    size=(80, 80),
    mode="bilinear",
    align_corners=False,
)

f32 = F.avg_pool2d(
    f16,
    kernel_size=2,
    stride=2,
)

print(f8.shape)
print(f16.shape)
print(f32.shape)
```

得到：

```text
[B,384,80,80]
[B,384,40,40]
[B,384,20,20]
```

但这只是：

```text
resize
```

并没有真正补回丰富的细粒度 spatial information。

STA 的价值远不只是 `interpolate + pooling`。

---

# 78. 为什么简单 resize 不够？

如果原始 stride-16 token 已经把：

```text
16×16 pixel region
```

压缩进一个 vector，

单纯把：

```text
40×40
```

插值成：

```text
80×80
```

并不会凭空创造：

```text
真实的 8×8 spatial detail
```

它只是：

```text
对已有 coarse feature 做插值
```

所以 STA 需要额外的 spatial pathway / interaction 机制。

---

# 79. 这也是“Adapter”一词的真正含义

Adapter 不是：

```text
随便加一层 Linear
```

在这里更重要的是：

```text
把 foundation representation
适配成 downstream task 所需要的 representation
```

对于 detection：

```text
DINOv3 representation
        ↓
spatial adaptation
        ↓
multi-scale detection features
```

---

# 80. DINOv3 为什么不直接重新训练成 hierarchical ViT？

因为 DEIMv2 想利用：

```text
已经预训练好的 DINOv3
```

如果彻底改变 backbone architecture：

```text
原预训练权重可能无法直接利用
```

Adapter 的优势是：

```text
保留 foundation backbone
+
增加 task-specific structure
```

---

# 81. Finetune 与 Frozen Backbone

X 配置：

```yaml
finetune: True
```

这意味着 DINOv3 不是完全冻结。

结合第 18 课：

```text
DINOv3
lr = 1e-5
```

说明策略是：

```text
保留预训练知识
但允许小幅适应 detection
```

即：

```text
gentle finetuning
```

---

# 82. 如果 `finetune=False` 会怎样？

概念上：

```text
DINOv3 parameters frozen
```

梯度不会更新 backbone。

优点：

```text
显存更低
训练更快
避免破坏 pretrained feature
```

缺点：

```text
无法适应 detection/domain-specific feature
```

DEIMv2 X 官方配置选择：

```text
finetune=True
```

说明作者希望 foundation feature 也参与任务适配。

---

# 83. 为什么中间 block feature 对 Adapter 很重要？

如果只拿最后一层：

```text
STA 只能看到 final representation
```

使用：

```text
5
8
11
```

则 adapter 可以在多个深度阶段与 ViT 表示发生交互。

可以把它想成：

```text
spatial branch
     ↕
ViT block 5
     ↕
ViT block 8
     ↕
ViT block 11
```

而不是：

```text
DINOv3 全跑完
↓
最后才接一个 FPN
```

下一课会具体验证源码到底怎样交互。

---

# 84. 为什么 5 / 8 / 11 很自然？

S+：

```text
12 blocks
index 0..11
```

选择：

```text
5
8
11
```

大致覆盖：

```text
middle
later-middle
final
```

同时最后一个：

```text
11
```

正好是最后 block。

但不要把这个解释成唯一数学最优解。

它是：

```text
architecture design choice
```

应通过论文/消融和源码理解，而不是赋予不存在的严格公式。

---

# 85. 一个非常重要的编号问题

Python block index：

```text
0,1,2,...,11
```

所以：

```text
index 5
```

实际上是第：

```text
6 个 block
```

同理：

```text
8 → 第 9 个 block
11 → 第 12 个 block
```

源码阅读时一定要区分：

```text
index
```

与：

```text
自然语言“第几层”
```

---

# 86. 从 CNN 思维切换到 ViT 思维的 8 个变化

CNN：

```text
channel C
spatial H×W
local convolution
stage hierarchy
stride
receptive field
BatchNorm
feature pyramid
```

ViT：

```text
embedding D
token N
self-attention
block depth
patch size
global interaction
LayerNorm
single-scale token grid
```

但 detection 最终又需要：

```text
feature pyramid
```

所以 DEIMv2 必须把两个世界重新连接。

---

# 87. CNN / ViT 对照表

| CNN 概念              | ViT 对应理解                   |
| --------------------- | ------------------------------ |
| pixel/feature cell    | patch token                    |
| channels C            | embedding dim D                |
| H×W                  | token grid                     |
| stride                | patch size / adaptation stride |
| 3×3 conv             | token mixing via attention     |
| stage                 | transformer block depth        |
| BatchNorm             | LayerNorm 常见                 |
| hierarchical pyramid  | 原生 plain ViT 通常没有        |
| local receptive field | global attention               |
| C3/C4/C5              | 需要 adapter 构造              |

---

# 88. DINOv3 / DEIMv2 关键数字表

以 X、640×640 为例：

| 项目                    |      数值 |
| ----------------------- | --------: |
| input                   |  640×640 |
| patch size              |        16 |
| patch grid              |    40×40 |
| patch tokens            |      1600 |
| CLS tokens              |         1 |
| storage tokens          |         4 |
| Transformer sequence    |      1605 |
| DINO embed dim          |       384 |
| ViT depth               |        12 |
| attention heads         |         6 |
| head dim                |        64 |
| FFN ratio               |         6 |
| interaction indexes     |  5, 8, 11 |
| detector hidden dim     |       256 |
| target detector strides | 8, 16, 32 |

---

# 89. 1600 与 8400 不要混淆

第 7 / 11 / 14 课我们经常看到：

$$
8400
=
80\times80
+
40\times40
+
20\times20
$$

这是 detector 三尺度 feature flatten 后：

```text
6400 + 1600 + 400
```

而 DINOv3 原生 patch tokens：

```text
40×40 = 1600
```

所以：

```text
1600
→ DINOv3 single-scale patch positions

8400
→ detection pyramid total positions
```

STA + HybridEncoder 正是在这两个空间之间建立桥梁。

---

# 90. 最关键的 shape 链

请把这条链背到能随口说出：

```text
Image
[B,3,640,640]

↓ DINOv3 PatchEmbed

Patch Grid
[B,384,40,40]

↓ flatten

Patch Tokens
[B,1600,384]

↓ CLS + 4 storage

ViT Sequence
[B,1605,384]

↓ 12 Transformer Blocks

Intermediate ViT Representations
block 5 / 8 / 11
[B,1605,384]

↓ keep spatial patch representation
↓ STA interaction/adaptation

Multi-scale Detection Features
[B,256,80,80]
[B,256,40,40]
[B,256,20,20]

↓ HybridEncoder

P3/P4/P5
256 channels

↓ flatten for detector

80² + 40² + 20²
= 8400 positions

↓ DEIMTransformer

300 object queries

↓ heads

class + bbox
```

其中 STA 内部的精确 shape 变化留到第 20 课逐行验证。

---

# 91. 为什么 `hidden_dim=256` 容易让人误判 DINO embedding？

因为 X YAML 同时出现：

```yaml
DINOv3STAs:
  hidden_dim: 256
```

和：

```yaml
HybridEncoder:
  hidden_dim: 256
```

以及：

```yaml
DEIMTransformer:
  hidden_dim: 256
```

很容易误以为：

```text
DINOv3 token = 256
```

实际上：

```text
dinov3_vits16plus native D = 384
```

256 是 DEIMv2 detection side 的统一 hidden dimension。

---

# 92. 如何避免以后再犯这种 shape 错误？

永远分三层检查：

```text
① YAML
配置想让 adapter 输出什么？

② backbone constructor
预训练模型原生 embed_dim 是多少？

③ forward runtime
真实 tensor.shape 是什么？
```

最终真相优先级：

```text
runtime tensor.shape
>
current source code
>
merged config
>
论文结构图
>
我们的记忆
```

---

# 93. 推荐的 Runtime Shape Hook

后面拿到本地仓库时，可以：

```python
def hook(name):
    def _hook(
        module,
        inputs,
        outputs
    ):
        def shape_of(x):
            if hasattr(x, "shape"):
                return tuple(x.shape)
            return type(x).__name__

        print(
            name,
            "in:",
            [shape_of(x)
             for x in inputs],
            "out:",
            shape_of(outputs)
        )

    return _hook
```

但 ViT block 输出和 STA 可能包含：

```text
tuple
list
dict
```

后面第 20 / 22 课会写更完整的递归 `ShapeTracer`。

---

# 94. 一个适合初学者的 MiniViT

```python
import torch
import torch.nn as nn


class MiniViT(nn.Module):

    def __init__(
        self,
        patch_size=16,
        embed_dim=384,
        depth=12,
        heads=6,
    ):
        super().__init__()

        self.patch = nn.Conv2d(
            3,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.cls = nn.Parameter(
            torch.zeros(
                1, 1, embed_dim
            )
        )

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=heads,
                dim_feedforward=2304,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ])

    def forward(self, x):

        x = self.patch(x)

        B, C, H, W = x.shape

        x = (
            x.flatten(2)
             .transpose(1, 2)
        )

        cls = self.cls.expand(
            B, -1, -1
        )

        x = torch.cat(
            [cls, x],
            dim=1
        )

        outs = {}

        for i, block in enumerate(
            self.blocks
        ):
            x = block(x)

            if i in [5, 8, 11]:
                outs[i] = x

        return outs
```

运行：

```python
model = MiniViT()

x = torch.randn(
    2,3,640,640
)

outs = model(x)

for k,v in outs.items():
    print(k, v.shape)
```

你应该看到：

```text
5  [2,1601,384]
8  [2,1601,384]
11 [2,1601,384]
```

这里没有实现 DINOv3 的 4 storage tokens，所以是：

```text
1601
```

而不是真实 S+ 的：

```text
1605
```

---

# 95. 为什么 MiniViT 只是教学模型？

它没有实现：

```text
DINOv3 self-supervised pretraining
RoPE
storage tokens
LayerScale
SwiGLU
mask token
DINOv3 normalization details
optimized attention
STA interaction
```

它的用途只有一个：

```text
让你彻底理解 shape 与主干数据流
```

不要把它当成 DINOv3 复刻。

---

# 96. 本课需要掌握的公式

## Patch 数量

$$
H_p=\frac{H}{P}
$$

$$
W_p=\frac{W}{P}
$$

$$
N=H_pW_p
$$

640 / patch16：

$$
N=40\times40=1600
$$

---

## Patch raw dimension

$$
D_{raw}
=
C P^2
$$

所以：

$$
3\times16^2
=
768
$$

---

## Multi-head dimension

$$
d_h=\frac{D}{h}
$$

S+：

$$
\frac{384}{6}=64
$$

---

## Attention

$$
\mathrm{Attention}(Q,K,V)
=
\mathrm{softmax}
\left(
\frac{QK^T}
{\sqrt{d_h}}
\right)V
$$

---

## Attention complexity

主要 token mixing 项：

$$
O(N^2D)
$$

所以 token 数增加非常昂贵。

---

## LayerNorm

$$
\hat x_i
=
\frac{x_i-\mu}
{\sqrt{\sigma^2+\epsilon}}
$$

$$
y_i
=
\gamma_i\hat x_i+\beta_i
$$

---

## Residual

$$
y=x+F(x)
$$

---

# 97. 本课最容易犯的 15 个错误

错误 1

```text
DINOv3-S+ hidden dim = 256
```

错。

原生：

```text
384
```

256 是 DEIMv2 detection side hidden dim。

---

错误 2

```text
640×640 / patch16
= 1600×1600 tokens
```

错。

是：

```text
40×40 = 1600 tokens
```

---

错误 3

```text
block 5 → stride 8
block 8 → stride 16
block 11 → stride 32
```

错。

plain ViT 三层原生 patch grid 仍然是：

```text
40×40
```

---

错误 4

```text
token 经过 Transformer 后数量逐层下降
```

通常不对。

DINOv3 plain ViT block 保持 sequence length。

---

错误 5

```text
patch token = object query
```

完全不同。

---

错误 6

```text
reshape 回 [B,C,H,W]
就变成 CNN feature 了
```

shape 相同，representation mechanism 不同。

---

错误 7

```text
ViT 有 global attention，
所以天然适合所有检测尺度
```

global semantics 不等于 multi-scale localization。

---

错误 8

```text
把 40×40 interpolate 成 80×80
就恢复了真实 stride-8 detail
```

不成立。

插值不会凭空恢复已丢失的细粒度信息。

---

错误 9

```text
CLS 是每个 patch 的类别
```

错。

CLS 是一个全局 learnable token。

---

错误 10

```text
storage token 是图像 patch
```

错。

它们是额外 learnable tokens。

---

错误 11

```text
ViT block 越深，H/W 越小
```

plain ViT 不成立。

---

错误 12

```text
patch embedding 必须写成 Linear
```

不需要。

Conv2d(k=P,s=P) 是非常自然的等价实现。

---

错误 13

```text
DINOv3 就是 DEIMv2 detector
```

错。

它是 backbone / foundation representation source。

---

错误 14

```text
Foundation Model 只是更大的模型
```

不准确。

关键是大规模预训练和通用迁移能力。

---

错误 15

```text
finetune=True 就意味着使用 detector LR=5e-4
```

错。

第 18 课已经看到 DINOv3 有独立小 LR：

```text
1e-5
```

---

# 98. 本课检查题

1. 640×640、patch size 16 时，patch grid 是多少？
2. patch token 数是多少？
3. 每个 raw RGB patch 有多少像素值？
4. DINOv3-S+ 的原生 embed dim 是多少？
5. 为什么不是 256？
6. S+ 有多少 Transformer blocks？
7. 有多少 attention heads？
8. 每个 head 的 dimension 是多少？
9. 为什么内部 sequence 可能是 1605 而不是 1600？
10. CLS token 的作用是什么？
11. storage tokens 是 patch tokens 吗？
12. block 5/8/11 的原生 spatial grid 是否不同？
13. 为什么 block 11 不天然等于 stride 32？
14. `get_intermediate_layers(..., reshape=True)` 做了什么？
15. 为什么 ViT 的 global attention 不等于 multi-scale feature？
16. 为什么 patch size 8 会显著增加计算量？
17. patch16 改 patch8 后 token 数增加多少倍？
18. attention matrix 元素数量大约增加多少倍？
19. DINOv3 为什么适合作为 foundation backbone？
20. STA 为什么是必要的？

---

# 99. 检查题答案

```text
1.
40×40。

2.
1600。

3.
3×16×16 = 768。

4.
384。

5.
256 是 DEIMv2 adapter/detector hidden dimension，
不是 dinov3_vits16plus 原生 embedding dimension。

6.
12。

7.
6。

8.
384 / 6 = 64。

9.
1600 patch
+ 1 CLS
+ 4 storage
= 1605。

10.
学习全局聚合 representation。

11.
不是。

12.
不是；plain ViT 中仍基于同一 40×40 patch grid。

13.
Transformer block depth 不执行 CNN 式 spatial downsampling。

14.
把 patch-token sequence 恢复成
[B,C,Hpatch,Wpatch] feature map。

15.
global interaction 解决信息交换，
不自动产生 1/8、1/16、1/32 pyramid。

16.
token 数显著增加，而 attention 是二次复杂度。

17.
4×：
1600 → 6400。

18.
约 16×。

19.
它通过大规模自监督预训练提供强、可迁移的视觉 representation。

20.
把 DINOv3 的强但原生单尺度表示，
适配为 detection-friendly 的多尺度空间特征。
```

---

# 100. 课后实验 A：完整打印 shape

自己实现：

```text
input
patch feature
patch tokens
CLS/storage sequence
Q/K/V
attention scores
block outputs
patch-only outputs
reshape outputs
```

目标输出类似：

```text
input:
[2,3,640,640]

patch map:
[2,384,40,40]

patch tokens:
[2,1600,384]

with cls/storage:
[2,1605,384]

q:
[2,6,1605,64]

attn:
[2,6,1605,1605]

patch only:
[2,1600,384]

reshape:
[2,384,40,40]
```

---

# 101. 课后实验 B：计算不同输入尺寸

分别计算：

```text
320×320
480×480
640×640
800×800
```

patch16 下 token 数。

答案：

```text
320:
20×20 = 400

480:
30×30 = 900

640:
40×40 = 1600

800:
50×50 = 2500
```

然后计算：

$$
N^2
$$

观察全局 attention 复杂度增长。

---

# 102. 课后实验 C：比较 CNN 与 ViT shape

写：

```text
Tiny CNN
```

产生：

```text
80×80
40×40
20×20
```

再写：

```text
MiniViT
```

输出：

```text
40×40
40×40
40×40
```

分别记录：

```text
layer
shape
stride
channel
```

这是理解 STA 动机最有效的小实验之一。

---

# 103. 课后实验 D：冻结与微调

建立 MiniViT：

```python
for p in backbone.parameters():
    p.requires_grad = False
```

观察：

```text
trainable params
optimizer param count
GPU memory
```

再恢复：

```python
for p in backbone.parameters():
    p.requires_grad = True
```

并设置：

```text
backbone LR = 1e-5
head LR = 5e-4
```

连接第 18 课知识。

---

# 104. 你现在应该怎样理解 `deimv2_dinov3_x`？

不是：

```text
“把 HGNetv2 换成 DINOv3”
```

这么简单。

真正变化是：

```text
CNN hierarchical backbone paradigm
             ↓
foundation ViT representation paradigm
             ↓
需要新的 spatial adaptation
```

因此 architecture 从：

```text
HGNetv2
↓
C3/C4/C5
```

变成：

```text
DINOv3-S+
↓
single-scale token representation
↓
STA
↓
multi-scale spatial representation
```

STA 不是装饰模块。

它是：

```text
Foundation Model
与
Real-Time Detector
之间的关键接口
```

---

# 105. 第 19 课最终记忆图

```text
                     IMAGE
               [B,3,640,640]
                      │
                      ▼
             Patch Embedding
          Conv(k=16, stride=16)
                      │
                      ▼
             [B,384,40,40]
                      │
                  flatten
                      ▼
             [B,1600,384]
                      │
       + CLS + 4 storage tokens
                      ▼
             [B,1605,384]
                      │
          ┌───────────┴───────────┐
          │    DINOv3 ViT-S+      │
          │                       │
          │ embed dim = 384       │
          │ depth = 12            │
          │ heads = 6             │
          │ head dim = 64         │
          │ FFN ratio = 6         │
          │ SwiGLU                │
          │ RoPE                  │
          └───────────┬───────────┘
                      │
             block 5 / 8 / 11
                      │
                      ▼
        strong semantic representations
                      │
        BUT native grid still 40×40
                      │
                      ▼
                    STA
            [NEXT LESSON]
                      │
           ┌──────────┼──────────┐
           ▼          ▼          ▼
        80×80      40×40      20×20
        1/8        1/16       1/32
           │          │          │
           └──────────┼──────────┘
                      ▼
                256 channels
                      │
                      ▼
              HybridEncoder
                      │
                      ▼
              DEIMTransformer
                      │
                      ▼
                class + bbox
```

---

# 106. 下一课：整个 DEIMv2 Architecture 的核心

第 20 课：

# 《STA——DEIMv2 最核心的新结构》

下一课我们不再停留在：

```text
“STA 把单尺度变成多尺度”
```

这种一句话解释。

而会真正追：

```text
DINOv3STAs.__init__
DINOv3STAs.forward
SpatialPriorModule
InteractionBlock
Injector
Extractor
deformable attention
level embedding
feature reshape
upsample/downsample
interaction_indexes
```

并逐步验证：

```text
[B,3,640,640]
        ↓
DINOv3 patch representation
        ↓
spatial prior
        ↓
ViT ↔ spatial interaction
        ↓
1/8  [B,256,80,80]
1/16 [B,256,40,40]
1/32 [B,256,20,20]
```

最重要的是回答：

```text
STA 到底从哪里获得 stride-8 的空间细节？

为什么不是简单 interpolate？

DINO token 怎样和 CNN-like spatial features 交换信息？

interaction_indexes=[5,8,11]
在 forward 中究竟控制什么？

为什么 STA 后还需要 HybridEncoder？
```

学完第 20 课，`deimv2_dinov3_x` 的 backbone 部分才算真正打通。

---

# 107. 源码核对与版本说明

本讲义针对当前公开版本进行核对，重点依据：

1. DEIMv2 官方 X 配置`configs/deimv2/deimv2_dinov3_x_coco.yml`
2. DEIMv2 官方 README
3. DINOv3 官方 `hub/backbones.py`
4. DINOv3 官方 `layers/patch_embed.py`
5. DINOv3 官方 `models/vision_transformer.py`

特别需要注意：

- `dinov3_vits16plus` 当前官方定义为 `patch_size=16, embed_dim=384, depth=12, num_heads=6, ffn_ratio=6, n_storage_tokens=4`。
- DEIMv2 X 配置的 `hidden_dim=256` 不应误认为 DINOv3-S+ 的原生 embedding dimension。
- `interaction_indexes=[5,8,11]` 是 DEIMv2 STA 的交互层选择；不能直接解释为三个 ViT block 天然具有 1/8、1/16、1/32 resolution。
- 本课刻意没有猜测 STA 内部所有具体 tensor，因为这些将在第 20 课根据 `DINOv3STAs` / adapter 源码逐行验证。
- DINOv3 和 DEIMv2 都可能继续更新；学习时以你实际使用 commit 的 runtime shape 为最高依据。

---

# 108. 参考资料

- DEIMv2 official repositoryhttps://github.com/Intellindust-AI-Lab/DEIMv2
- DEIMv2 X confighttps://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/configs/deimv2/deimv2_dinov3_x_coco.yml
- DINOv3 official repositoryhttps://github.com/facebookresearch/dinov3
- DINOv3 PatchEmbedhttps://github.com/facebookresearch/dinov3/blob/main/dinov3/layers/patch_embed.py
- DINOv3 ViT implementationhttps://github.com/facebookresearch/dinov3/blob/main/dinov3/models/vision_transformer.py
- DINOv3 backbone constructors
  https://github.com/facebookresearch/dinov3/blob/main/dinov3/hub/backbones.py

---

## 本课一句话总结

`deimv2_dinov3_x` 的核心起点不是“DINOv3 自动给出三层检测特征”，而是：

```text
640×640 image
→ patch16
→ 40×40 = 1600 patch tokens
→ DINOv3-S+ 在 384 维 token space 中学习强视觉 representation
→ ViT block 5/8/11 提供不同深度但原生同尺度的表示
→ STA 再把 foundation representation 转换成 detector 所需要的
  1/8、1/16、1/32 多尺度 256-channel spatial features
```

理解这一点，下一课读 STA 源码时就不会把“Transformer 深度”和“空间尺度”混为一谈。
