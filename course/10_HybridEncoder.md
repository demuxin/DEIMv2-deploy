# DEIMv2 源码课程 · 第 10 课
# HybridEncoder——DEIM 系列最容易被低估的模块

> **本课主线**
>
> 第 9 课结束时，`deim_dfine_x` 的 HGNetv2-B5 已经产生三层特征：
>
> ```text
> C3 = [B,  512, 80, 80]   stride 8
> C4 = [B, 1024, 40, 40]   stride 16
> C5 = [B, 2048, 20, 20]   stride 32
> ```
>
> 但它们还不能直接成为一个高质量 DETR decoder 的输入。
>
> `HybridEncoder` 的工作可以压缩成三步：
>
> ```text
> ① channel projection
>    512/1024/2048 → 384/384/384
>
> ② intra-scale interaction
>    在选定尺度上做 Transformer Encoder
>
> ③ cross-scale fusion
>    top-down FPN + bottom-up PAN
> ```
>
> 最终输出：
>
> ```text
> P3 = [B,384,80,80]
> P4 = [B,384,40,40]
> P5 = [B,384,20,20]
> ```
>
> 这三个特征随后进入 `DFINETransformer`。

---

# 0. 为什么说 HybridEncoder 很容易被低估？

很多初学者第一次看 DETR 系列，会把注意力全部放在：

```text
Transformer Decoder
Object Query
Deformable Attention
Hungarian Matching
```

于是很容易形成一个错误印象：

```text
Backbone
   ↓
Decoder
```

但 `deim_dfine_x` 实际上是：

```text
Image
  ↓
HGNetv2-B5
  ↓
C3 / C4 / C5
  ↓
HybridEncoder          ← 本课
  ↓
P3 / P4 / P5
  ↓
DFINETransformer
```

HybridEncoder 处于：

```text
CNN backbone
```

和：

```text
DETR decoder
```

之间。

它同时承担：

```text
channel alignment
semantic interaction
multi-scale fusion
feature pyramid reconstruction
```

所以它更接近传统检测器里：

```text
Neck
```

与 Transformer encoder 的混合体。

这也是名字：

```text
HybridEncoder
```

中 `Hybrid` 的真正含义之一。

---

# 1. 本课先固定研究对象：`deim_dfine_x`

本课程的第一条重点路线是：

```text
deim_dfine_x
```

它继承基础配置后，X 级模型把 HybridEncoder 的关键维度提高到：

```yaml
HybridEncoder:
    hidden_dim: 384
    dim_feedforward: 2048
```

基础配置则提供：

```yaml
in_channels: [512, 1024, 2048]
feat_strides: [8, 16, 32]

use_encoder_idx: [2]
num_encoder_layers: 1
nhead: 8

expansion: 1.0
depth_mult: 1
act: silu
```

因此这一课所有主要 shape 都按：

```text
hidden_dim = 384
```

推导。

这点非常重要。

如果你看到网上其他 D-FINE / DEIM 示例使用：

```text
256
128
192
```

并不矛盾。

那通常是其他模型尺寸。

---

# 2. 第 9 课与第 10 课的接口

HGNetv2-B5 返回：

```python
feats = [
    C3,
    C4,
    C5,
]
```

其中：

```text
C3
[B,512,80,80]

C4
[B,1024,40,40]

C5
[B,2048,20,20]
```

HybridEncoder 的 `forward` 接收到的：

```text
feats
```

就是这个 Python list。

所以第一条源码检查逻辑本质上是在确认：

```text
len(feats)
==
len(in_channels)
```

这里：

```text
3 == 3
```

如果 backbone 少返回一层，encoder 会立刻发现接口不一致。

---

# 3. 为什么不能直接把 C3/C4/C5 送进 decoder？

有三个核心问题。

## 问题 A：channel 不统一

```text
C3 = 512
C4 = 1024
C5 = 2048
```

而后面的 detector 希望使用统一 embedding dimension：

```text
384
```

---

## 问题 B：三个尺度彼此还缺乏充分交流

Backbone 的：

```text
C3
C4
C5
```

来自不同深度。

虽然深层特征是由浅层逐步计算而来，但这不等于：

```text
三个最终尺度已经充分双向融合
```

尤其：

```text
C3
```

空间细节丰富但语义较浅；

```text
C5
```

语义强但空间较粗。

检测需要把二者结合。

---

## 问题 C：全尺度全局 Attention 太贵

如果把三层全部 flatten：

```text
80×80 = 6400
40×40 = 1600
20×20 = 400
```

总 token：

\[
N=8400
\]

普通 self-attention 的 attention matrix 大小：

\[
N^2
=
8400^2
=
70,560,000
\]

每个 head、每张图就有超过：

```text
70 million
```

个 attention score。

实时检测器不能随意这样做。

HybridEncoder 的核心思想就是：

> **只把昂贵的全局建模放在最划算的位置，再用高效卷积式路径把信息传播到其他尺度。**

这句话值得记住。

---

# 4. HybridEncoder 的整体结构

先不要看代码。

先把算法地图记住：

```text
Backbone

C3                 C4                 C5
512×80×80          1024×40×40         2048×20×20
 │                  │                  │
 ▼                  ▼                  ▼

1×1 projection     1×1 projection     1×1 projection

 │                  │                  │
 ▼                  ▼                  ▼

384×80×80          384×40×40          384×20×20
                                         │
                                         ▼
                                  Transformer Encoder
                                         │
                                         ▼
                                   384×20×20

                TOP-DOWN FPN
        ◄────────────────────────

384×80×80    ◄── 384×40×40 ◄── 384×20×20

                BOTTOM-UP PAN
        ────────────────────────►

384×80×80    ──► 384×40×40 ──► 384×20×20

 │                  │                  │
 ▼                  ▼                  ▼

P3                 P4                 P5

[B,384,80,80]      [B,384,40,40]      [B,384,20,20]
```

整个 HybridEncoder 可以理解成：

```text
统一维度
+
局部全局建模
+
双向多尺度融合
```

---

# 5. 第一部分：Input Projection

源码在构造阶段会为每个输入尺度建立一个：

```text
input_proj
```

逻辑可以简化理解为：

```python
if in_channel != hidden_dim:
    use 1x1 Conv + BN
else:
    use Identity
```

对于 X：

```text
512  != 384
1024 != 384
2048 != 384
```

所以三层都需要投影。

---

# 6. C3 的投影

输入：

```text
[B,512,80,80]
```

使用：

```text
1×1 Conv
512 → 384
stride 1
```

得到：

```text
[B,384,80,80]
```

空间尺寸完全不变。

为什么？

卷积输出公式：

\[
H_{out}
=
\left\lfloor
\frac{H+2P-K}{S}
\right\rfloor+1
\]

这里：

```text
K=1
P=0
S=1
```

所以：

\[
H_{out}=H
\]

同理：

\[
W_{out}=W
\]

---

# 7. C4 / C5 的投影

同理：

```text
C4
[B,1024,40,40]

1×1 Conv
1024 → 384

↓

[B,384,40,40]
```

以及：

```text
C5
[B,2048,20,20]

1×1 Conv
2048 → 384

↓

[B,384,20,20]
```

所以：

```text
proj_feats[0]
[B,384,80,80]

proj_feats[1]
[B,384,40,40]

proj_feats[2]
[B,384,20,20]
```

---

# 8. 为什么统一到 384 channels？

因为后续：

```text
Transformer
FPN
PAN
DFINETransformer
```

都需要更规则的接口。

如果保留：

```text
512
1024
2048
```

那么每一次跨尺度融合都要额外解决 channel mismatch。

统一到：

```text
384
```

以后：

```text
P3
P4
P5
```

都处于同一个 feature dimension。

后面的 decoder 配置也对应：

```text
feat_channels = [384,384,384]
```

这说明：

> HybridEncoder 的输出接口和 DFINETransformer 的输入接口是严格对齐的。

---

# 9. 为什么使用 1×1 Conv？

1×1 Conv 最重要的作用之一：

```text
channel projection
```

对于某个像素位置：

```text
x ∈ R^512
```

1×1 Conv 本质上学习：

\[
y=Wx
\]

其中：

\[
W\in R^{384\times512}
\]

所以它可以把：

```text
512-dimensional channel vector
```

映射成：

```text
384-dimensional vector
```

但不需要在空间邻域上移动。

---

# 10. 1×1 Conv 参数量

C3 projection：

\[
512\times384
=
196,608
\]

C4：

\[
1024\times384
=
393,216
\]

C5：

\[
2048\times384
=
786,432
\]

仅卷积权重合计：

\[
1,376,256
\]

约：

```text
1.38M
```

参数。

注意：

> 参数量不等于计算量。

C3 虽然 channel 较少，但：

```text
80×80
```

空间最大，因此实际 MACs 也非常可观。

---

# 11. 为什么 projection 后还有 BatchNorm？

投影不仅是：

```text
change channels
```

还要让不同 backbone stage 的 feature 进入相近的表示空间。

BN 可以帮助：

```text
feature scale normalization
optimization stability
```

所以：

```text
Conv1×1 + BN
```

是检测 neck 中非常常见的 channel alignment 单元。

---

# 12. 第二部分：Intra-scale Interaction

投影后：

```text
P3?  [B,384,80,80]
P4?  [B,384,40,40]
P5?  [B,384,20,20]
```

注意这里暂时还不要叫最终 P3/P4/P5。

它们只是：

```text
projected backbone features
```

接下来 HybridEncoder 会对：

```text
use_encoder_idx
```

指定的 feature 做 Transformer Encoder。

X 的基础配置：

```text
use_encoder_idx = [2]
```

所以只处理：

```text
proj_feats[2]
```

也就是：

```text
20×20
```

这一层。

---

# 13. 为什么 index 2 是 C5？

Python index：

```text
0 → C3
1 → C4
2 → C5
```

所以：

```text
use_encoder_idx=[2]
```

意味着：

```text
只在 stride-32 feature 上做 Transformer self-attention
```

不是：

```text
做 2 层 Transformer
```

这是一个很容易误读的配置。

---

# 14. 为什么只在 C5 上做 Attention？

先比较 token 数：

```text
C3:
80×80 = 6400

C4:
40×40 = 1600

C5:
20×20 = 400
```

普通 self-attention 复杂度主要包含：

\[
O(N^2D)
\]

所以 attention score 数量：

### C3

\[
6400^2
=
40,960,000
\]

### C4

\[
1600^2
=
2,560,000
\]

### C5

\[
400^2
=
160,000
\]

比例：

```text
C3 : C4 : C5
=
256 : 16 : 1
```

只看 attention matrix 大小，C3 比 C5 大：

```text
256 倍
```

所以：

> 把 global self-attention 放到 20×20 是一个非常关键的效率设计。

---

# 15. 这也是 RT-DETR 系列的重要思想

传统 Transformer encoder 可能对所有 multi-scale tokens 做大规模全局交互。

实时 DETR 的一个重要方向是：

```text
不要把昂贵 Attention 到处使用
```

而是：

```text
高层低分辨率 feature
→ global semantic interaction

多尺度传播
→ convolutional fusion
```

这就是 HybridEncoder 的：

```text
hybrid
```

之一：

```text
Transformer
+
CNN-style feature pyramid
```

---

# 16. C5 如何变成 Transformer Token？

当前：

```text
C5_projected
=
[B,384,20,20]
```

Transformer 需要：

```text
[B,N,D]
```

所以源码执行概念上：

```text
flatten spatial dimensions
+
permute
```

第一步：

```text
[B,384,20,20]
```

flatten H/W：

```text
[B,384,400]
```

因为：

\[
20\times20=400
\]

再交换维度：

```text
[B,400,384]
```

于是：

```text
N = 400
D = 384
```

---

# 17. 把 shape 写成数学形式

输入 feature：

\[
X\in R^{B\times C\times H\times W}
\]

这里：

\[
C=384,\quad H=W=20
\]

flatten：

\[
X'
\in
R^{B\times C\times HW}
\]

permute：

\[
X''
\in
R^{B\times HW\times C}
\]

所以：

\[
X''
\in
R^{B\times400\times384}
\]

这就是 Transformer Encoder 的输入。

---

# 18. 为什么需要 Positional Encoding？

Self-Attention 本身主要处理 token 间内容关系。

如果只把：

```text
20×20
```

flatten 成：

```text
400 tokens
```

Transformer 并不会天然知道：

```text
token 0 在左上角
token 399 在右下角
```

因此需要把二维位置编码加入：

```text
Q/K
```

让 attention 感知空间位置。

HybridEncoder 使用：

```text
2D sine-cosine positional embedding
```

---

# 19. 2D Sin-Cos Position Embedding

设 embedding dimension：

\[
D=384
\]

源码要求：

\[
D \bmod 4=0
\]

因为二维位置会分成：

```text
x-sin
x-cos
y-sin
y-cos
```

四部分。

每部分维度：

\[
D/4=96
\]

最后拼起来：

\[
96+96+96+96=384
\]

所以位置编码 shape：

```text
[1,400,384]
```

---

# 20. 为什么 position embedding batch dimension 是 1？

因为同一个：

```text
20×20 grid
```

在 batch 中所有图片的位置坐标结构一样。

所以：

```text
pos_embed
[1,400,384]
```

可以通过 broadcasting 与：

```text
[B,400,384]
```

相加。

无需复制 B 份参数。

---

# 21. Position Encoding 的频率公式

一类标准 sin-cos 编码会构造不同频率：

\[
\omega_i
=
\frac{1}
{T^{i/d}}
\]

其中：

```text
T = temperature
```

这里通常：

```text
10000
```

然后对于位置：

\[
p
\]

构造：

\[
\sin(p\omega_i)
\]

和：

\[
\cos(p\omega_i)
\]

二维情况分别对：

```text
x coordinate
y coordinate
```

编码，再拼接。

---

# 22. 为什么用不同频率？

如果只有：

```text
sin(x)
```

一种频率，表达能力有限。

不同频率让 embedding 同时包含：

```text
低频
→ 大尺度位置变化

高频
→ 更细位置差异
```

于是 Transformer 可以利用多个空间尺度的位置模式。

---

# 23. TransformerEncoderLayer 的内部结构

这一课不重新从零讲 Attention；第 4 课已经推导过。

这里直接映射源码：

```text
input
[B,400,384]

  │
  ▼

Q = X + Pos
K = X + Pos
V = X

  │
  ▼

Multi-Head Self-Attention

  │
  ▼

Residual + LayerNorm

  │
  ▼

FFN
384 → 2048 → 384

  │
  ▼

Residual + LayerNorm

  │
  ▼

[B,400,384]
```

---

# 24. 8 个 Attention Heads 如何分维度？

配置：

```text
hidden_dim = 384
nhead = 8
```

每个 head：

\[
d_h
=
384/8
=
48
\]

所以：

```text
Q
[B,8,400,48]

K
[B,8,400,48]

V
[B,8,400,48]
```

每个 head 的 attention matrix：

```text
[B,8,400,400]
```

概念上是这样理解；PyTorch 内部具体存储布局可能不同。

---

# 25. Attention score 有多少个？

每张图片：

\[
8\times400\times400
=
1,280,000
\]

即：

```text
1.28 million scores
```

如果在 80×80 上：

\[
8\times6400^2
=
327,680,000
\]

约：

```text
327.68 million
```

仅这一项就能看出为什么只选择低分辨率层。

---

# 26. FFN 为什么是 384 → 2048 → 384？

X 配置：

```text
dim_feedforward = 2048
```

所以：

\[
X
\in R^{384}
\]

先：

\[
384\rightarrow2048
\]

经过激活：

```text
GELU
```

再：

\[
2048\rightarrow384
\]

FFN 不是做 token-to-token mixing。

它主要对：

```text
每个 token 的 channel representation
```

独立进行非线性变换。

---

# 27. Attention 与 FFN 的分工

可以记成：

```text
Attention
→ token 之间交流

FFN
→ 一个 token 内部的 channel 变换
```

所以 Transformer layer：

```text
spatial/token interaction
+
channel transformation
```

两者缺一不可。

---

# 28. `num_encoder_layers=1`

基础配置：

```text
num_encoder_layers = 1
```

所以 selected C5 feature 只经过：

```text
1 个 Transformer Encoder Layer
```

而不是 6 层、12 层。

这是实时设计的另一个关键点。

HybridEncoder 不是：

```text
“把 ViT encoder 搬进 detector”
```

而是：

```text
只使用少量 Transformer 计算增强高层语义。
```

---

# 29. Transformer 后怎样恢复 feature map？

Transformer 输出：

```text
memory
[B,400,384]
```

先：

```text
permute
```

变成：

```text
[B,384,400]
```

然后 reshape：

```text
[B,384,20,20]
```

所以：

```text
sequence
```

重新变成：

```text
2D feature map
```

接下来才能进入 FPN/PAN 的卷积路径。

---

# 30. 为什么这里常出现 `.contiguous()`？

经过：

```python
permute(...)
```

Tensor 的逻辑维度顺序改变了，但底层 memory layout 不一定连续。

后面：

```text
reshape / convolution
```

通常希望得到连续布局。

所以代码常见：

```text
permute
→ reshape
→ contiguous
```

这正是第 2 课：

```text
shape 操作 vs memory layout
```

在真实源码里的应用。

---

# 31. 到这里发生了什么？

现在三个 projected feature：

```text
F3
[B,384,80,80]

F4
[B,384,40,40]

F5
[B,384,20,20]
```

其中：

```text
F5
```

已经额外经过：

```text
global self-attention
```

因此它包含更强的：

```text
global semantic context
```

接下来真正关键的问题是：

> 怎么把 F5 的强语义传播给高分辨率 F4、F3？

答案：

```text
Top-Down FPN
```

---

# 32. 第三部分：Top-Down FPN

FPN 的直觉：

```text
深层 feature
语义强，分辨率低

        ↓ upsample

与浅层高分辨率 feature 融合

        ↓

高分辨率 feature
同时获得更强语义
```

在本模型中：

```text
20×20
↓
40×40
↓
80×80
```

所以信息方向：

```text
C5 → C4 → C3
```

---

# 33. Top-Down 第一步：从 20×20 到 40×40

起点：

```text
feat_high
=
[B,384,20,20]
```

先经过 lateral conv：

```text
1×1 Conv
384 → 384
```

shape 不变：

```text
[B,384,20,20]
```

然后 nearest-neighbor upsample：

```text
scale_factor = 2
```

得到：

```text
[B,384,40,40]
```

---

# 34. 为什么叫 `feat_high`？

这里：

```text
high
```

不是：

```text
高分辨率
```

而是：

```text
高层级 / high-level semantic feature
```

所以：

```text
feat_high
20×20
```

反而空间更小。

而：

```text
feat_low
40×40
```

表示：

```text
较低层 backbone feature
```

它的空间更大。

这是源码变量命名中非常容易误解的一点。

---

# 35. 与 C4 projected feature 融合

浅层：

```text
feat_low
=
[B,384,40,40]
```

上采样后的高层：

```text
upsample_feat
=
[B,384,40,40]
```

`deim_dfine_x` / D-FINE 风格默认：

```text
fuse_op = cat
```

所以：

```python
concat along channel
```

得到：

```text
[B,768,40,40]
```

因为：

\[
384+384=768
\]

---

# 36. 为什么不是直接相加？

两种方式都合理。

## Cat

```text
[A,B]
```

保留两份 feature 的独立 channel 信息。

输出 channel：

\[
2D
\]

---

## Sum

```text
A+B
```

直接融合。

输出 channel：

\[
D
\]

Sum：

```text
更省计算/显存
```

Cat：

```text
保留更丰富的独立表示
```

DEIMv2 后续某些配置引入：

```text
fuse_op = sum
```

正是在效率与表示能力之间重新取舍。

但当前 `deim_dfine_x` 主线按：

```text
cat
```

理解。

---

# 37. FPN Fuse Block 不是简单一层 Conv

cat 后：

```text
[B,768,40,40]
```

还要经过：

```text
Fuse_Block
```

在当前 `version='dfine'` 路径中，源码选择：

```text
RepNCSPELAN4
```

它本身又由：

```text
1×1 projection
split
CSP-style blocks
3×3 conv
concat
1×1 fusion
```

等结构组成。

所以：

> FPN 的“融合”不是 `cat` 完就结束。

真正学习如何组合两个尺度的信息的是后面的 Fuse Block。

---

# 38. `RepNCSPELAN4` 为什么存在？

如果只是：

```text
upsample
+
concat
```

那么两个来源只是被机械地摆在 channel 维度上。

网络还需要学习：

```text
哪些高层语义有用？
哪些低层细节有用？
如何组合？
如何过滤冗余？
```

Fuse Block 就负责：

```text
learned feature fusion
```

---

# 39. X 配置下 Fuse Block 的关键通道

当前：

```text
hidden_dim = 384
fuse_op = cat
```

所以 Fuse Block 输入：

\[
c_1
=
2\times384
=
768
\]

源码构造逻辑还会得到：

```text
c2 = 384
c3 = 768
```

而：

```text
c4
```

由：

\[
round(
expansion
\times hidden\_dim / 2
)
\]

决定。

基础配置：

```text
expansion = 1.0
```

所以：

\[
c4
=
round(1.0\times384/2)
=
192
\]

因此可以记：

```text
RepNCSPELAN4
c1 = 768
c2 = 384
c3 = 768
c4 = 192
```

---

# 40. Fuse Block 深度是多少？

源码：

\[
num\_blocks
=
round(3\times depth\_mult)
\]

基础配置：

```text
depth_mult = 1
```

所以：

\[
num\_blocks
=
3
\]

因此 X 的融合 block 并不是极浅的一次卷积。

它内部仍有多层 CSP/VGG-style feature transformation。

---

# 41. Top-Down 第一次融合输出

输入：

```text
upsampled F5
[B,384,40,40]

projected F4
[B,384,40,40]
```

concat：

```text
[B,768,40,40]
```

Fuse Block：

```text
768 → 384
```

输出：

```text
I4
[B,384,40,40]
```

这里暂时叫：

```text
I4
```

表示 top-down inner feature。

---

# 42. Top-Down 第二步：40×40 → 80×80

现在：

```text
feat_high = I4
[B,384,40,40]
```

lateral：

```text
[B,384,40,40]
```

upsample ×2：

```text
[B,384,80,80]
```

与 projected C3：

```text
[B,384,80,80]
```

concat：

```text
[B,768,80,80]
```

Fuse Block：

```text
[B,384,80,80]
```

得到：

```text
I3
[B,384,80,80]
```

---

# 43. Top-Down 结束后的三层

此时：

```text
inner_outs[0]
=
I3
[B,384,80,80]

inner_outs[1]
=
I4
[B,384,40,40]

inner_outs[2]
=
I5
[B,384,20,20]
```

它们已经完成：

```text
深层 → 浅层
```

的信息传播。

---

# 44. 为什么 Top-Down 还不够？

Top-down 的效果是：

```text
把深层语义送到浅层
```

但检测不仅需要：

```text
高分辨率 feature 获得语义
```

还希望：

```text
浅层精细定位信息
```

能够重新向深层传播。

所以 HybridEncoder 再走一次：

```text
Bottom-Up PAN
```

方向：

```text
80×80
↓
40×40
↓
20×20
```

---

# 45. FPN 与 PAN 的信息方向

一定要记住：

```text
FPN:
20 → 40 → 80

PAN:
80 → 40 → 20
```

所以合起来：

```text
20 ⇄ 40 ⇄ 80
```

三个尺度都能间接获得其他尺度的信息。

这就是：

```text
bidirectional multi-scale fusion
```

---

# 46. Bottom-Up 第一步

起点：

```text
I3
[B,384,80,80]
```

需要 downsample 到：

```text
40×40
```

HybridEncoder 当前实现对 D-FINE/DEIM 路径使用一个高效下采样模块：

```text
SCDown
```

核心思想：

```text
1×1 Conv
↓
depthwise 3×3 stride 2
```

---

# 47. SCDown 的 shape

输入：

```text
[B,384,80,80]
```

1×1：

```text
384 → 384
```

得到：

```text
[B,384,80,80]
```

再 depthwise：

```text
3×3
stride=2
groups=384
```

得到：

```text
[B,384,40,40]
```

所以：

```text
channel 不变
H/W 减半
```

---

# 48. 为什么下采样用 SCDown？

如果使用普通：

```text
384→384
3×3 stride2
```

卷积核参数：

\[
384^2\times9
=
1,327,104
\]

而 depthwise 3×3：

\[
384\times9
=
3456
\]

当然 SCDown 还包含 1×1 Conv：

\[
384^2
=
147,456
\]

合计仍远低于普通 full 3×3。

所以它适合：

```text
实时 detector
```

---

# 49. Bottom-Up 与 I4 融合

downsample：

```text
[B,384,40,40]
```

top-down 的 I4：

```text
[B,384,40,40]
```

cat：

```text
[B,768,40,40]
```

再经过 PAN Fuse Block：

```text
[B,384,40,40]
```

得到最终：

```text
P4
[B,384,40,40]
```

---

# 50. Bottom-Up 第二步

当前：

```text
P4
[B,384,40,40]
```

SCDown：

```text
[B,384,20,20]
```

与：

```text
I5
[B,384,20,20]
```

concat：

```text
[B,768,20,20]
```

Fuse Block：

```text
[B,384,20,20]
```

得到：

```text
P5
[B,384,20,20]
```

而最开始的：

```text
I3
```

直接作为：

```text
P3
[B,384,80,80]
```

---

# 51. HybridEncoder 最终输出

最终：

```python
outs = [
    P3,
    P4,
    P5,
]
```

shape：

```text
P3
[B,384,80,80]

P4
[B,384,40,40]

P5
[B,384,20,20]
```

stride：

```text
8
16
32
```

这正好匹配下一层：

```text
DFINETransformer
feat_channels = [384,384,384]
feat_strides  = [8,16,32]
```

---

# 52. 完整 Shape Trace

现在把整个 forward 一口气写完。

## Backbone 输入

```text
C3 [B, 512,80,80]
C4 [B,1024,40,40]
C5 [B,2048,20,20]
```

## Input Projection

```text
F3 [B,384,80,80]
F4 [B,384,40,40]
F5 [B,384,20,20]
```

## Transformer on index 2

```text
F5
[B,384,20,20]

flatten
→ [B,384,400]

permute
→ [B,400,384]

Transformer
→ [B,400,384]

reshape
→ [B,384,20,20]
```

## Top-Down

```text
F5
[B,384,20,20]

upsample
→ [B,384,40,40]

cat F4
→ [B,768,40,40]

Fuse
→ I4 [B,384,40,40]

upsample
→ [B,384,80,80]

cat F3
→ [B,768,80,80]

Fuse
→ I3 [B,384,80,80]
```

## Bottom-Up

```text
I3
[B,384,80,80]

SCDown
→ [B,384,40,40]

cat I4
→ [B,768,40,40]

Fuse
→ P4 [B,384,40,40]

SCDown
→ [B,384,20,20]

cat I5
→ [B,768,20,20]

Fuse
→ P5 [B,384,20,20]
```

最终：

```text
P3 [B,384,80,80]
P4 [B,384,40,40]
P5 [B,384,20,20]
```

---

# 53. 一张结构图彻底记住 HybridEncoder

```text
C3 512×80×80
      │
      ▼
   1×1 proj
      │
      ▼
F3 384×80×80
      │                       ▲
      │                       │ upsample
      │                       │
      ├────── cat ◄──── I4 384×40×40
      │
      ▼
 Fuse Block
      │
      ▼
I3 384×80×80
      │
      │ SCDown
      ▼
384×40×40 ── cat ── I4
      │
      ▼
 Fuse Block
      │
      ▼
P4 384×40×40
      │
      │ SCDown
      ▼
384×20×20 ── cat ── I5
      │
      ▼
 Fuse Block
      │
      ▼
P5 384×20×20


另一条 top-down 主线：

C5 2048×20×20
      │
      ▼
   1×1 proj
      │
      ▼
F5 384×20×20
      │
      ▼
Transformer Encoder
      │
      ▼
I5 384×20×20
      │
      ▼ upsample
384×40×40
      │
      ├── cat ── F4
      ▼
I4 384×40×40
      │
      ▼ upsample
384×80×80
      │
      ├── cat ── F3
      ▼
I3 384×80×80
```

---

# 54. `inner_outs.insert(0, ...)` 为什么值得注意？

Top-down 是从：

```text
highest level
```

往低层走。

但最终希望 list 顺序仍然是：

```text
80×80
40×40
20×20
```

所以每产生一个更高分辨率 feature，就插到 list 开头。

概念过程：

```text
开始:
[20]

第一次:
[40,20]

第二次:
[80,40,20]
```

这就是：

```text
insert(0, ...)
```

存在的原因。

---

# 55. Bottom-Up 为什么用 `append`？

PAN 正好反方向：

```text
80
→ 40
→ 20
```

所以：

```text
outs = [80]
```

然后：

```text
append(40)
append(20)
```

自然得到：

```text
[80,40,20]
```

这两个 list 操作其实直接反映了：

```text
FPN/PAN 的传播方向
```

---

# 56. `lateral_convs` 有几个？

输入 level 数：

```text
L=3
```

top-down 需要连接：

```text
C5 → C4
C4 → C3
```

所以：

\[
L-1=2
\]

个 lateral conv。

---

# 57. `fpn_blocks` 有几个？

每一次跨尺度 top-down fusion 后需要一个 Fuse Block：

```text
20→40
40→80
```

所以：

```text
2 个
```

---

# 58. `downsample_convs` 有几个？

PAN：

```text
80→40
40→20
```

也是：

```text
2 个
```

---

# 59. `pan_blocks` 有几个？

每次 bottom-up fusion：

```text
40 level
20 level
```

各需要一个 Fuse Block。

所以：

```text
2 个
```

因此三尺度 HybridEncoder 的跨尺度主体是：

```text
2 lateral
2 FPN fuse
2 downsample
2 PAN fuse
```

---

# 60. 如果只有两个 feature levels 呢？

例如轻量模型可能只有：

```text
stride16
stride32
```

那么：

\[
L=2
\]

所以：

```text
top-down fusion = 1 次
bottom-up fusion = 1 次
```

HybridEncoder 的构造逻辑天然支持不同 level 数。

这就是为什么源码大量使用：

```text
len(in_channels)
```

而不是写死：

```text
3
```

---

# 61. `version='dfine'` 到底控制什么？

当前 HybridEncoder 源码支持不同融合 block 路线。

大致：

```text
version = dfine
→ RepNCSPELAN4

version = deim
→ RepNCSPELAN5

其他/RT-DETR 风格
→ CSP-style block
```

所以：

```text
HybridEncoder
```

不是只有一个固定 neck 实现。

它把：

```text
整体拓扑
```

固定为：

```text
projection
→ selected Transformer
→ FPN
→ PAN
```

但：

```text
Fuse Block 的内部结构
```

可以按版本变化。

---

# 62. 为什么课程当前重点讲 `version='dfine'`？

因为我们正在走：

```text
deim_dfine_x
```

这一条路线。

当前最重要的是搞清：

```text
DEIM training framework
+
D-FINE detector architecture
```

之间的关系。

所以此处重点：

```text
DFINE-style HybridEncoder
```

后面学：

```text
deimv2_dinov3_x
```

时，我们会重新比较 DEIMv2 版本中的：

```text
version
csp_type
fuse_op
```

等设计。

---

# 63. RepNCSPELAN4：先理解大图，不要陷入名字

这个名字很长：

```text
RepNCSPELAN4
```

初学者很容易被吓住。

读它时只问：

```text
输入 channel？
输出 channel？
是否改变 H/W？
内部如何分支？
哪里 concat？
```

对当前 X：

```text
input
[B,768,H,W]

↓

1×1 projection / split

↓

多条 CSP-style transformations

↓

concat

↓

1×1 fusion

↓

output
[B,384,H,W]
```

它最重要的外部契约就是：

```text
768 → 384
H/W 不变
```

先掌握接口，再学内部。

---

# 64. `VGGBlock` 为什么又出现了？

Fuse Block 内部使用的 CSP 模块中有一种 VGG-style reparameterizable block。

训练时可以存在：

```text
3×3 branch
+
1×1 branch
```

输出相加。

部署时可以把两条线性卷积+BN路径：

```text
融合
```

成一个等价：

```text
3×3 Conv
```

这叫：

```text
structural re-parameterization
```

---

# 65. 为什么训练和部署可以结构不同？

假设：

\[
y
=
Conv_{3\times3}(x)
+
Conv_{1\times1}(x)
\]

如果两条路径都是线性变换，并把：

```text
1×1 kernel
```

padding 成：

```text
3×3 kernel
```

就可以构造：

\[
K_{eq}
=
K_{3\times3}
+
pad(K_{1\times1})
\]

bias 也相加：

\[
b_{eq}=b_3+b_1
\]

于是：

\[
y
=
Conv_{eq}(x)
\]

训练时多分支有利于优化；

部署时单分支更高效。

---

# 66. Conv + BN 也可以融合

推理时 BN：

\[
y
=
\gamma
\frac{Wx-\mu}
{\sqrt{\sigma^2+\epsilon}}
+\beta
\]

可以整理成：

\[
y
=
W'x+b'
\]

其中：

\[
W'
=
W
\frac{\gamma}
{\sqrt{\sigma^2+\epsilon}}
\]

\[
b'
=
\beta
-
\mu
\frac{\gamma}
{\sqrt{\sigma^2+\epsilon}}
\]

如果原 Conv 有 bias，还要把对应项一起并入。

因此：

```text
Conv + BN
```

可以部署成：

```text
single Conv with bias
```

这也是 HybridEncoder 代码里出现：

```text
convert_to_deploy
```

一类逻辑的原因。

---

# 67. 为什么实时模型很在意这种 fusion？

训练时：

```text
module richness
```

重要。

推理时：

```text
kernel launch
memory access
operator count
latency
```

也非常重要。

把：

```text
Conv
BN
multiple branches
```

折叠为更少算子，可以减少部署开销。

所以：

```text
训练结构
```

和：

```text
部署结构
```

不必完全一样，只要数学上等价。

---

# 68. FPN 为什么用 nearest upsampling？

源码 top-down 使用：

```text
nearest
```

而不是复杂的：

```text
bilinear
deconvolution
```

主要优点：

```text
简单
快
无额外参数
```

真正的 feature refinement 交给后面的：

```text
Fuse Block
```

所以 upsample 本身不必承担太多学习任务。

---

# 69. Nearest Upsample 在做什么？

假设一个 2×2 feature：

```text
a b
c d
```

放大 2 倍：

```text
a a b b
a a b b
c c d d
c c d d
```

它没有创造新的 learned parameter。

只是把 feature 对齐到：

```text
下一层的 H/W
```

以便：

```text
cat / sum
```

---

# 70. 为什么跨尺度融合前必须空间对齐？

你不能直接：

```text
[B,384,20,20]
```

和：

```text
[B,384,40,40]
```

做 channel concat。

因为除 concat 维度外，其他维度必须一致。

所以：

```text
20×20
```

必须先：

```text
upsample → 40×40
```

才能：

```text
cat(dim=1)
```

同理 PAN：

```text
80×80
```

必须先：

```text
downsample → 40×40
```

才能融合。

---

# 71. 为什么不把所有 feature 都 resize 到 80×80？

如果全部：

```text
20→80
40→80
```

然后一次融合，看起来更简单。

但会有两个问题：

```text
① 计算量暴涨
② 失去真正的 multi-scale hierarchy
```

目标检测希望保留：

```text
P3 / P4 / P5
```

不同 stride。

因为 decoder 的 deformable attention 后面还会利用：

```text
multi-level spatial structure
```

所以不是把所有尺度强行变成一个尺度。

---

# 72. 为什么也不全部压成 20×20？

这样虽然便宜，但：

```text
small object spatial detail
```

损失太大。

因此：

```text
80
40
20
```

三个尺度同时存在，是：

```text
定位精度
语义能力
计算成本
```

之间的折中。

---

# 73. HybridEncoder 的真正作用：不是“制造多尺度”

这是一个关键纠正。

HGNetv2 已经产生：

```text
80
40
20
```

所以 HybridEncoder 并不是第一次制造多尺度。

更准确地说，它：

```text
对已有多尺度 feature
进行 channel alignment
+
semantic enhancement
+
cross-scale fusion
```

这是：

```text
multi-scale refinement
```

而不是简单：

```text
multi-scale generation
```

---

# 74. 与后面 DINOv3 + STA 的区别

这一点先提前建立概念，但不深入。

HGNetv2：

```text
天然 hierarchical CNN
→ 本来就有多个尺度
```

DINOv3 ViT：

```text
天然更接近 single-scale token representation
```

所以 DEIMv2 的 STA 首先要解决：

```text
如何把强 ViT representation
变成 detection-friendly spatial multi-scale features
```

然后：

```text
HybridEncoder
```

仍可继续做：

```text
跨尺度融合
```

所以：

```text
STA
≠
HybridEncoder
```

后面第 20 课会彻底拆开二者职责。

---

# 75. 一个极其重要的概念：Intra-scale vs Cross-scale

HybridEncoder 的结构可以按这两个词记。

## Intra-scale

同一个尺度内部：

```text
20×20 token
↔
20×20 token
```

使用：

```text
Transformer self-attention
```

目的：

```text
global interaction within one level
```

---

## Cross-scale

不同尺度之间：

```text
20 ↔ 40
40 ↔ 80
```

使用：

```text
FPN/PAN + convolutional fuse blocks
```

目的：

```text
multi-scale information exchange
```

所以：

```text
HybridEncoder
=
intra-scale interaction
+
cross-scale fusion
```

这句话非常重要。

---

# 76. 为什么不用 Attention 做所有 Cross-scale Fusion？

理论上可以。

但实时 detector 最关心：

```text
accuracy / latency trade-off
```

卷积式：

```text
upsample
downsample
cat
local fusion
```

在规则 feature map 上非常高效。

而全局 attention 的：

\[
N^2
\]

成本在高分辨率尺度上增长极快。

所以 HybridEncoder 使用：

```text
Attention
放在最值钱的地方

CNN fusion
负责大规模多尺度传播
```

这是非常典型的工程设计思路。

---

# 77. 具体数字：如果三个尺度全部做独立 Self-Attention

只算每层 attention matrix 元素：

```text
C3:
6400² = 40,960,000

C4:
1600² = 2,560,000

C5:
400² = 160,000
```

总：

\[
43,680,000
\]

而只做 C5：

\[
160,000
\]

比例：

\[
43,680,000 / 160,000
=
273
\]

也就是说：

> 仅比较单 head attention matrix 元素数量，三个尺度全做约是只做 C5 的 273 倍。

实际完整 FLOPs 还包括 QKV projection、FFN 等，所以这只是直觉估算，不是完整模型 FLOPs。

---

# 78. 如果把三个尺度 concat 后做一次全局 Self-Attention

总：

\[
N=8400
\]

attention matrix：

\[
8400^2=70,560,000
\]

只做 C5：

\[
400^2=160,000
\]

比例：

\[
441
\]

这更加说明：

```text
HybridEncoder
```

为什么要“混合”不同计算范式。

---

# 79. PyTorch 实验 1：Input Projection

```python
import torch
import torch.nn as nn

B = 2

c3 = torch.randn(B, 512, 80, 80)
c4 = torch.randn(B, 1024, 40, 40)
c5 = torch.randn(B, 2048, 20, 20)

proj3 = nn.Conv2d(512, 384, 1)
proj4 = nn.Conv2d(1024, 384, 1)
proj5 = nn.Conv2d(2048, 384, 1)

f3 = proj3(c3)
f4 = proj4(c4)
f5 = proj5(c5)

print(f3.shape)
print(f4.shape)
print(f5.shape)
```

预期：

```text
[2,384,80,80]
[2,384,40,40]
[2,384,20,20]
```

---

# 80. PyTorch 实验 2：Feature Map → Tokens → Feature Map

```python
B = 2

x = torch.randn(
    B, 384, 20, 20
)

h, w = x.shape[-2:]

tokens = (
    x
    .flatten(2)
    .permute(0, 2, 1)
)

print(tokens.shape)
# [2,400,384]

feature = (
    tokens
    .permute(0, 2, 1)
    .reshape(B,384,h,w)
    .contiguous()
)

print(feature.shape)
# [2,384,20,20]
```

这段实验必须自己跑一次。

---

# 81. PyTorch 实验 3：最小 Transformer Encoder

```python
layer = nn.TransformerEncoderLayer(
    d_model=384,
    nhead=8,
    dim_feedforward=2048,
    dropout=0.0,
    activation="gelu",
    batch_first=True
)

x = torch.randn(
    2,400,384
)

y = layer(x)

print(y.shape)
```

输出：

```text
[2,400,384]
```

Transformer 不改变：

```text
N
D
```

只改变 feature 内容。

---

# 82. PyTorch 实验 4：Top-Down 一步

```python
import torch.nn.functional as F

high = torch.randn(
    2,384,20,20
)

low = torch.randn(
    2,384,40,40
)

up = F.interpolate(
    high,
    scale_factor=2,
    mode="nearest"
)

print(up.shape)

fused = torch.cat(
    [up, low],
    dim=1
)

print(fused.shape)
```

输出：

```text
[2,384,40,40]
[2,768,40,40]
```

---

# 83. PyTorch 实验 5：Bottom-Up 一步

```python
down = nn.Conv2d(
    384,
    384,
    kernel_size=3,
    stride=2,
    padding=1,
    groups=384
)

low_res = down(
    torch.randn(
        2,384,80,80
    )
)

print(low_res.shape)
```

得到：

```text
[2,384,40,40]
```

再与：

```text
[B,384,40,40]
```

cat：

```text
[B,768,40,40]
```

---

# 84. PyTorch 实验 6：手写 Tiny HybridEncoder

下面不追求复刻 Fuse Block，只复现主拓扑：

```python
class TinyHybridEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.proj3 = nn.Conv2d(
            512,384,1
        )

        self.proj4 = nn.Conv2d(
            1024,384,1
        )

        self.proj5 = nn.Conv2d(
            2048,384,1
        )

        self.encoder = (
            nn.TransformerEncoderLayer(
                d_model=384,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True
            )
        )

        self.fuse4 = nn.Conv2d(
            768,384,3,padding=1
        )

        self.fuse3 = nn.Conv2d(
            768,384,3,padding=1
        )

        self.down3 = nn.Conv2d(
            384,384,3,2,1
        )

        self.pan4 = nn.Conv2d(
            768,384,3,padding=1
        )

        self.down4 = nn.Conv2d(
            384,384,3,2,1
        )

        self.pan5 = nn.Conv2d(
            768,384,3,padding=1
        )

    def forward(self, feats):

        c3, c4, c5 = feats

        f3 = self.proj3(c3)
        f4 = self.proj4(c4)
        f5 = self.proj5(c5)

        B,C,H,W = f5.shape

        t5 = (
            f5.flatten(2)
              .permute(0,2,1)
        )

        t5 = self.encoder(t5)

        f5 = (
            t5.permute(0,2,1)
              .reshape(B,C,H,W)
        )

        up5 = F.interpolate(
            f5,
            scale_factor=2,
            mode="nearest"
        )

        i4 = self.fuse4(
            torch.cat(
                [up5,f4],
                dim=1
            )
        )

        up4 = F.interpolate(
            i4,
            scale_factor=2,
            mode="nearest"
        )

        i3 = self.fuse3(
            torch.cat(
                [up4,f3],
                dim=1
            )
        )

        p3 = i3

        p4 = self.pan4(
            torch.cat(
                [self.down3(p3),i4],
                dim=1
            )
        )

        p5 = self.pan5(
            torch.cat(
                [self.down4(p4),f5],
                dim=1
            )
        )

        return [p3,p4,p5]
```

测试：

```python
model = TinyHybridEncoder()

ys = model([
    torch.randn(1,512,80,80),
    torch.randn(1,1024,40,40),
    torch.randn(1,2048,20,20),
])

for y in ys:
    print(y.shape)
```

预期：

```text
[1,384,80,80]
[1,384,40,40]
[1,384,20,20]
```

---

# 85. TinyHybridEncoder 与真实源码有什么不同？

Toy 版本只是帮助理解主拓扑。

真实实现还包含：

```text
BatchNorm
activation
2D sin-cos position encoding
RepNCSPELAN4
CSP sub-blocks
VGG-style reparameterization
SCDown
deploy-time fusion
configurable version
configurable fuse_op
eval positional embedding cache
```

所以不要把 toy 实验当成官方模型。

它只是：

```text
结构骨架实验
```

---

# 86. PyTorch 实验 7：真实模型 Hook

如果已经成功创建：

```text
model.encoder
```

可以：

```python
def shape_hook(name):

    def fn(module, inp, out):

        def fmt(x):
            if torch.is_tensor(x):
                return tuple(x.shape)
            return type(x).__name__

        if isinstance(out, (list,tuple)):
            out_info = [
                fmt(x) for x in out
            ]
        else:
            out_info = fmt(out)

        print(
            name,
            "->",
            out_info
        )

    return fn


for name, module in model.encoder.named_modules():

    if (
        "input_proj" in name
        or "encoder" in name
        or "fpn_blocks" in name
        or "pan_blocks" in name
        or "downsample_convs" in name
    ):
        module.register_forward_hook(
            shape_hook(name)
        )
```

这样可以真正验证本课所有 shape。

---

# 87. 推荐的断点位置

如果用 VS Code / PyCharm debug，最值得打断点的位置：

```text
HybridEncoder.forward 开头

input projection 后

flatten 前

Transformer 后

top-down 每次 upsample 后

top-down fuse 后

bottom-up downsample 后

bottom-up fuse 后

return outs 前
```

然后每次观察：

```python
tensor.shape
tensor.dtype
tensor.device
tensor.requires_grad
```

---

# 88. 不要只看 Shape，还要看数值统计

调试时可以：

```python
def stats(x):
    return {
        "shape": tuple(x.shape),
        "mean": x.mean().item(),
        "std": x.std().item(),
        "min": x.min().item(),
        "max": x.max().item(),
    }
```

为什么？

两个 tensor 即使：

```text
shape 完全正确
```

也可能发生：

```text
NaN
Inf
异常大值
全零
```

所以高级调试应该同时看：

```text
shape
+
numerical statistics
```

---

# 89. Position Embedding 的训练/评估差异

HybridEncoder 支持：

```text
eval_spatial_size
```

当固定评估尺寸时，可以预先构造相应：

```text
position embedding
```

例如：

```text
640×640
stride32
→ 20×20
```

这样 eval 时不必每次重新生成。

而训练时由于：

```text
multi-scale training
```

feature H/W 可能变化，所以常根据当前：

```text
h,w
```

动态构造。

---

# 90. 如果训练输入不是 640×640 呢？

假设输入：

```text
800×800
```

那么 stride：

```text
8 → 100×100
16 → 50×50
32 → 25×25
```

selected encoder feature：

```text
25×25
```

token 数：

\[
625
\]

所以 position embedding 也必须变成：

```text
[1,625,384]
```

这就是动态构造位置编码的意义。

---

# 91. 为什么 multi-scale training 会影响 Attention 成本？

假设 stride32：

### 640 input

```text
20×20
N=400
N²=160000
```

### 800 input

```text
25×25
N=625
N²=390625
```

虽然输入边长只增加：

\[
800/640=1.25
\]

attention matrix 大小却增加约：

\[
390625/160000
\approx2.44
\]

所以 Transformer 对输入分辨率变化很敏感。

---

# 92. 为什么 HybridEncoder 只在一个尺度做 Transformer 更适合多尺度训练？

因为 selected level 是：

```text
stride32
```

即使输入尺寸变化，它仍然是最小的 feature map。

如果在：

```text
stride8
```

做全局 attention，多尺度训练时高分辨率 batch 的成本会迅速爆炸。

所以这个结构不仅：

```text
640 inference 快
```

也有利于：

```text
training efficiency
```

---

# 93. FPN 的语义传播例子

假设：

```text
C5
```

已经非常确信图中某区域属于：

```text
person
```

但其 20×20 空间定位较粗。

Top-down：

```text
C5 semantic information
↓ upsample
C4
↓
C3
```

让高分辨率 feature 在保留：

```text
edge / local detail
```

的同时，也知道：

```text
“这里很可能是 person”
```

这有利于：

```text
更精确定位
```

---

# 94. PAN 的定位信息传播例子

反过来：

```text
P3
```

有更细空间结构。

Bottom-up：

```text
P3
↓
P4
↓
P5
```

让低分辨率语义层重新吸收：

```text
fine spatial cues
```

所以双向融合比单纯 top-down 更充分。

---

# 95. 为什么最终仍然保留三个输出？

因为 DFINETransformer 的 deformable cross-attention 后面会从多个尺度采样。

例如：

```text
P3
适合精细、小目标区域

P4
中等尺度

P5
大范围语义
```

decoder query 可以利用：

```text
multi-scale sampling
```

而不是被迫只依赖单一 feature map。

第 11 课会把这条链真正接上。

---

# 96. HybridEncoder 与 Deformable Attention 的接口

本课输出：

```text
[
  [B,384,80,80],
  [B,384,40,40],
  [B,384,20,20],
]
```

第 11 课 decoder 会把它们进一步处理成类似：

```text
flatten P3
→ 6400 positions

flatten P4
→ 1600 positions

flatten P5
→ 400 positions
```

总：

\[
8400
\]

然后记录：

```text
spatial_shapes
=
[[80,80],
 [40,40],
 [20,20]]
```

所以：

> 第 7 课 Deformable Attention 里的 8400 个位置，真正来源就是本课 HybridEncoder 的三个输出。

现在课程前后已经连接起来了。

---

# 97. `spatial_shapes` 为什么必须保留？

flatten 后：

```text
8400 positions
```

本身已经看不出：

```text
哪 6400 个属于 80×80
哪 1600 个属于 40×40
哪 400 个属于 20×20
```

所以 decoder 需要额外 metadata：

```text
spatial_shapes
level_start_index
```

恢复每个 level 的空间结构。

这就是为什么 HybridEncoder 不能只理解成：

```text
“输出三个 Tensor”
```

它决定了后面 deformable attention 的 multi-level geometry。

---

# 98. `out_channels` 和 `out_strides`

HybridEncoder 构造时会记录：

```text
out_channels
```

为每层：

```text
hidden_dim
```

所以 X：

```text
[384,384,384]
```

同时：

```text
out_strides
```

继承：

```text
[8,16,32]
```

这其实就是模块对外声明：

```text
我输出什么规格的 feature pyramid
```

---

# 99. 模块接口契约

可以把整个模型看成多个接口：

## Backbone

输出：

```text
channels:
[512,1024,2048]

strides:
[8,16,32]
```

## HybridEncoder

输入要求：

```text
in_channels:
[512,1024,2048]
```

输出：

```text
channels:
[384,384,384]

strides:
[8,16,32]
```

## DFINETransformer

输入要求：

```text
feat_channels:
[384,384,384]

feat_strides:
[8,16,32]
```

这三个模块正好闭合。

---

# 100. 这就是为什么配置文件不能随便改一个数字

如果只把：

```yaml
HybridEncoder:
    hidden_dim: 512
```

但忘了修改：

```yaml
DFINETransformer:
    feat_channels: [384,384,384]
```

就会破坏：

```text
encoder → decoder
```

接口。

所以以后改模型必须沿着：

```text
producer
→ interface
→ consumer
```

检查。

---

# 101. `fuse_op='sum'` 时 shape 会怎样？

如果：

```text
sum
```

那么：

```text
[B,384,H,W]
+
[B,384,H,W]
```

仍然：

```text
[B,384,H,W]
```

所以 Fuse Block 输入 channel：

```text
384
```

而不是：

```text
768
```

源码也会相应设置：

```text
input_dim = hidden_dim
```

而 cat：

```text
input_dim = 2 × hidden_dim
```

这是一个非常清楚的配置 → shape 影响链。

---

# 102. Sum 与 Cat 的计算差异

假设 Fuse Block 第一层主要处理输入 channel。

Cat：

```text
input = 768
```

Sum：

```text
input = 384
```

因此 Sum 可以减少：

```text
memory traffic
input projection cost
部分卷积计算
```

但 Cat 给后续网络更多自由：

```text
分别识别两个来源
```

所以两者没有绝对“谁更好”，而是不同：

```text
accuracy-efficiency trade-off
```

---

# 103. `expansion` 的意义

Fuse Block 内部不会所有地方都固定使用 384 channels。

例如：

\[
c4
=
round(
expansion
\times hidden\_dim / 2
)
\]

所以：

```text
expansion 越大
→ 内部 channel 越宽
→ capacity 增加
→ params/FLOPs 增加
```

X 基础路线：

```text
expansion = 1.0
```

所以内部较宽。

小模型可能使用：

```text
0.34
0.5
```

来减少计算。

---

# 104. `depth_mult` 的意义

源码：

\[
n=round(3\times depth\_mult)
\]

因此：

```text
depth_mult
```

控制 Fuse Block 内部重复 block 数量。

例如：

```text
depth_mult=1.0
→ 3 blocks

depth_mult≈0.67
→ 2 blocks

depth_mult≈0.34
→ 1 block
```

这是一种：

```text
depth scaling
```

机制。

---

# 105. hidden_dim / expansion / depth_mult 三者分工

可以记：

```text
hidden_dim
→ 整个 encoder 主通道宽度

expansion
→ fusion block 内部宽度

depth_mult
→ fusion block 内部深度
```

因此模型缩放不只是：

```text
hidden_dim
```

一个旋钮。

---

# 106. 为什么 X 使用 hidden_dim=384？

更大的模型希望：

```text
更高 representation capacity
```

所以相比典型：

```text
256
```

X 提升到：

```text
384
```

但代价是：

```text
projection 更贵
Transformer QKV 更贵
FFN 更贵
FPN/PAN conv 更贵
decoder 也更贵
```

因此：

```text
X
```

的精度与延迟都比小型号高。

---

# 107. Transformer 参数量粗略估算

忽略 bias。

Self-Attention 的：

```text
Q,K,V,out projection
```

大致：

\[
4D^2
\]

D=384：

\[
4\times384^2
=
589,824
\]

FFN：

\[
D\times F
+
F\times D
=
2DF
\]

其中：

```text
F=2048
```

所以：

\[
2\times384\times2048
=
1,572,864
\]

仅这两大部分约：

\[
2.16M
\]

参数。

这说明：

> 即使只有 1 层 Transformer Encoder，它也不是“免费”的。

---

# 108. 为什么 FFN 参数比 Attention projection 还多？

上面的粗略数字：

```text
Attention projections
≈ 0.59M

FFN
≈ 1.57M
```

因为：

```text
2048
```

远大于：

```text
384
```

所以 Transformer 中：

```text
FFN
```

经常是参数量大户。

这也是为什么小模型会明显降低：

```text
dim_feedforward
```

---

# 109. Attention 计算量与参数量不能混为一谈

参数量主要受：

```text
D
F
```

影响。

而 attention runtime 还受：

```text
N
```

强烈影响。

所以：

```text
20×20
```

和：

```text
80×80
```

使用完全相同 Transformer 参数，

参数量：

```text
一样
```

但计算量：

```text
差非常多
```

这是理解 Transformer efficiency 的关键。

---

# 110. HybridEncoder 是否改变 stride？

最终：

```text
输入 strides
[8,16,32]

输出 strides
[8,16,32]
```

所以整体上：

```text
不改变 pyramid level 定义
```

虽然内部：

```text
upsample
downsample
```

来回传播，

最终仍回到同样三层尺度。

---

# 111. HybridEncoder 是否改变 spatial position 总数？

最终仍然：

```text
6400 + 1600 + 400
=
8400
```

所以：

```text
position 数不变
```

变化的是：

```text
每个 position 的 representation
```

变得更适合 detection。

---

# 112. 为什么叫 Encoder？

因为它仍然承担：

```text
把 backbone feature
编码成 decoder 更容易消费的 memory representation
```

只是它不等同于经典 Transformer：

```text
Encoder × 6
```

而是：

```text
Transformer + CNN pyramid
```

的混合编码器。

---

# 113. 为什么不是直接叫 Neck？

从传统 CNN 检测器角度：

```text
它确实很像 neck
```

但 RT-DETR/D-FINE 体系希望强调：

```text
intra-scale Transformer encoding
```

也是其核心组成。

因此：

```text
HybridEncoder
```

比单纯：

```text
FPN Neck
```

更准确。

---

# 114. 与经典 FPN 的差别

经典 FPN 主要：

```text
lateral projection
+
top-down upsample
+
fusion
```

HybridEncoder 额外包含：

```text
selected-level Transformer encoding
+
stronger CSP/ELAN-style fuse blocks
+
bottom-up PAN
```

所以它不是普通 FPN 的简单复制。

---

# 115. 与 PANet 思想的联系

PAN 强调：

```text
bottom-up path augmentation
```

让低层定位信息向高层传播。

HybridEncoder 同样采用：

```text
top-down
+
bottom-up
```

双向路径。

但具体 block 与 DETR 接口是其自己的实现。

---

# 116. 与 BiFPN 的区别

BiFPN 也强调：

```text
bidirectional feature fusion
```

并常加入：

```text
learnable fusion weights
```

HybridEncoder 当前主线不是 BiFPN。

不要因为都有：

```text
top-down + bottom-up
```

就认为实现完全相同。

结构思想可能相近，但：

```text
fusion operator
block
channel strategy
attention placement
```

都不同。

---

# 117. 初学者最容易犯的错误 1

错误：

```text
HybridEncoder 就是 Transformer Encoder。
```

正确：

```text
HybridEncoder
=
少量 Transformer intra-scale interaction
+
大量 CNN-style cross-scale fusion
```

---

# 118. 错误 2

错误：

```text
三个尺度都经过 Transformer。
```

当前 X 主线：

```text
use_encoder_idx=[2]
```

只对：

```text
20×20
```

层做。

---

# 119. 错误 3

错误：

```text
use_encoder_idx=[2]
表示有两个 encoder。
```

错误。

它表示：

```text
选择 index=2 的 feature level。
```

层数由：

```text
num_encoder_layers
```

控制。

---

# 120. 错误 4

错误：

```text
hidden_dim=384
表示 backbone 输出 384 channels。
```

错误。

Backbone：

```text
512/1024/2048
```

HybridEncoder 通过 input projection 才统一成：

```text
384
```

---

# 121. 错误 5

错误：

```text
FPN 只是 upsample。
```

错误。

完整过程：

```text
lateral transform
→ upsample
→ fusion
→ learned Fuse Block
```

---

# 122. 错误 6

错误：

```text
cat 后仍然是 384 channels。
```

如果：

```text
384 + 384
```

沿 channel cat：

```text
768
```

Fuse Block 才把它变回：

```text
384
```

---

# 123. 错误 7

错误：

```text
PAN 与 FPN 做同一个方向。
```

正确：

```text
FPN:
20→40→80

PAN:
80→40→20
```

---

# 124. 错误 8

错误：

```text
P3/P4/P5 与 C3/C4/C5 是同一个 Tensor。
```

它们空间尺寸可能一样，但内容不同。

```text
C*
→ backbone raw features

P*
→ HybridEncoder fused features
```

---

# 125. 错误 9

错误：

```text
8400 positions 是 backbone 某一层产生的。
```

不是。

\[
8400
=
80^2+40^2+20^2
\]

是三个 feature levels flatten 后的总和。

---

# 126. 错误 10

错误：

```text
Attention 参数量大，所以高分辨率 Attention 贵。
```

只说了一半。

高分辨率真正致命的是：

\[
N^2
\]

即 token 数平方增长。

即使参数完全相同：

```text
80×80
```

也比：

```text
20×20
```

贵得多。

---

# 127. 源码阅读顺序建议

打开当前仓库：

```text
engine/deim/hybrid_encoder.py
```

推荐顺序：

```text
① HybridEncoder.__init__

② HybridEncoder.forward

③ build_2d_sincos_position_embedding

④ TransformerEncoderLayer

⑤ TransformerEncoder

⑥ SCDown

⑦ RepNCSPELAN4

⑧ CSPLayer / CSPLayer2

⑨ VGGBlock

⑩ ConvNormLayer / ConvNormLayer_fuse
```

不要从文件第一行机械读到底。

先大结构，再进入细节。

---

# 128. 阅读 `__init__` 时先只找 5 组模块

看到构造函数，先标：

```text
A. input_proj

B. encoder

C. lateral_convs

D. fpn_blocks

E. downsample_convs + pan_blocks
```

其他细节先放一边。

只要这五组能对应到结构图：

```text
projection
Transformer
top-down
bottom-up
```

就不会迷路。

---

# 129. 阅读 `forward` 时只追 4 个 list

重点变量：

```text
feats
proj_feats
inner_outs
outs
```

含义：

```text
feats
= backbone 原始输出

proj_feats
= channel 对齐后的 feature

inner_outs
= top-down FPN 结果

outs
= bottom-up PAN 最终输出
```

这是读 forward 最重要的变量地图。

---

# 130. 四个 list 的 shape

## feats

```text
[
 [B,512,80,80],
 [B,1024,40,40],
 [B,2048,20,20]
]
```

## proj_feats

```text
[
 [B,384,80,80],
 [B,384,40,40],
 [B,384,20,20]
]
```

## inner_outs

```text
[
 [B,384,80,80],
 [B,384,40,40],
 [B,384,20,20]
]
```

## outs

```text
[
 [B,384,80,80],
 [B,384,40,40],
 [B,384,20,20]
]
```

虽然后三者 shape 看起来相同：

> **它们的 feature 内容完全不同。**

这正是为什么不能只看 shape 判断“网络什么都没做”。

---

# 131. Shape 不变 ≠ 计算没发生

Transformer：

```text
[B,400,384]
→
[B,400,384]
```

shape 不变。

Fuse Block：

```text
[B,384,H,W]
→
[B,384,H,W]
```

很多时候 shape 也不变。

但 representation 已经发生巨大变化。

所以源码分析要区分：

```text
shape transformation
```

和：

```text
feature transformation
```

---

# 132. 一个更高级的理解：HybridEncoder 在做“信息路由”

可以把三层看成三个信息节点：

```text
S8
S16
S32
```

先：

```text
S32 内部全局通信
```

再：

```text
S32 → S16 → S8
```

最后：

```text
S8 → S16 → S32
```

所以 HybridEncoder 本质上设计了一个：

```text
高效的信息传播图
```

而不只是“堆几个卷积”。

---

# 133. 为什么先 Top-Down，再 Bottom-Up？

先让：

```text
高层语义
```

传播到：

```text
高分辨率层
```

这样新的 S8 feature 已经不是原始浅层 feature。

再从这个增强后的 S8：

```text
bottom-up
```

传播回去。

因此 PAN 使用的是：

```text
已经融合过高层语义的高分辨率 feature
```

而不是原始 C3。

这使双向融合形成连续的信息循环。

---

# 134. 一个信息路径例子

最终 P5 可以间接包含：

```text
原始 C5
```

当然有。

但它还可以通过：

```text
C5
↓ top-down
C4
↓
C3
↓ bottom-up
P4
↓
P5
```

重新吸收经过多尺度融合后的信息。

所以最终：

```text
P5
```

不是简单：

```text
projected C5
```

---

# 135. P3 也不只是 C3

P3：

```text
projected C3
+
来自 C4/C5 的 top-down semantic information
+
Fuse Block learned transformation
```

所以它既有：

```text
高空间分辨率
```

又有：

```text
更强 semantic context
```

这正是 detection pyramid 想要的。

---

# 136. 本课小型数学练习：Cat 输入宽度

设：

\[
D=384
\]

top-down 两个输入：

\[
A,B
\in
R^{B\times D\times H\times W}
\]

channel concat：

\[
Concat(A,B)
\in
R^{B\times2D\times H\times W}
\]

所以：

\[
2D=768
\]

如果改为 sum：

\[
A+B
\in
R^{B\times D\times H\times W}
\]

所以：

```text
cat 和 sum
```

不仅是语义差异，也直接改变后续模块的输入维度。

---

# 137. 本课小型数学练习：Attention 比例

C3：

\[
N_3=80^2=6400
\]

C5：

\[
N_5=20^2=400
\]

Self-attention matrix 比例：

\[
\frac{N_3^2}{N_5^2}
=
\left(
\frac{6400}{400}
\right)^2
=
16^2
=
256
\]

所以：

```text
空间边长只差 4 倍
```

attention matrix 却差：

```text
256 倍
```

因为：

```text
二维位置数先平方一次，
Attention 又对 token 数平方一次。
```

从边长角度近似是四次方增长。

---

# 138. 为什么这对视觉 Transformer 特别重要？

假设 feature 边长扩大：

\[
r
\]

二维 token 数：

\[
N\propto r^2
\]

Attention：

\[
N^2\propto r^4
\]

所以：

```text
resolution ×2
```

可能让 attention matrix：

```text
×16
```

这就是高分辨率视觉 attention 的根本成本问题之一。

---

# 139. 本课实验：比较 Attention Matrix 内存

```python
def attn_elements(h, w, heads=8):

    n = h * w

    return (
        heads
        * n
        * n
    )


for size in [20,40,80]:

    n = attn_elements(
        size,
        size
    )

    mb_fp32 = (
        n * 4 / 1024**2
    )

    print(
        size,
        n,
        f"{mb_fp32:.2f} MB"
    )
```

注意：

> 这只是理论 attention-score tensor 的元素估算，不代表 PyTorch 实际峰值显存。

Flash Attention、kernel fusion、backward 保存策略等都会改变真实内存行为。

---

# 140. 为什么不要把理论复杂度直接等同于实测 latency？

真实 GPU latency 还受：

```text
kernel implementation
memory bandwidth
tensor core utilization
operator fusion
batch size
precision
CUDA version
TensorRT
```

影响。

所以：

```text
理论 FLOPs
```

用于解释趋势；

```text
真实 latency
```

必须 benchmark。

这是以后做模型优化必须养成的习惯。

---

# 141. `ConvNormLayer_fuse` 的意义

HybridEncoder 中有些卷积层专门支持：

```text
Conv + BN fusion
```

训练：

```text
Conv
→ BN
→ Activation
```

部署：

```text
Fused Conv
→ Activation
```

输出数学上等价（在 BN 使用固定 running statistics 的推理状态下）。

这样可以：

```text
减少部署 operator
```

---

# 142. 为什么 `convert_to_deploy()` 不能随便在训练中调用？

因为转换后：

```text
原始 Conv
BN
```

可能被删除，替换成：

```text
fused Conv
```

这针对：

```text
inference deployment
```

设计。

如果还要继续普通训练：

```text
BN training behavior
```

已经不存在。

所以部署重参数化通常应该在：

```text
训练完成
```

之后使用。

---

# 143. HybridEncoder 与 BN 的关系

本模块中大量 CNN fusion block 使用：

```text
BatchNorm2d
```

所以前面训练 recipe 中：

```text
SyncBN
batch size
freeze_norm
```

不只是影响 backbone。

encoder 的 normalization 行为同样会影响训练。

这也是为什么训练系统不能只研究 model.forward。

---

# 144. 为什么 Transformer 用 LayerNorm，而 CNN 部分用 BatchNorm？

Transformer token：

```text
[B,N,D]
```

更常用：

```text
LayerNorm
```

对 feature dimension 做归一化。

CNN feature：

```text
[B,C,H,W]
```

传统高效卷积 block 常用：

```text
BatchNorm2d
```

HybridEncoder 同时存在：

```text
LayerNorm
+
BatchNorm
```

正好体现：

```text
Transformer + CNN
```

的混合结构。

---

# 145. 这也是“Hybrid”的另一个层面

不仅计算结构混合：

```text
Attention
+
Convolution
```

连数据表示也在来回切换：

```text
[B,C,H,W]
↓
[B,N,D]
↓
[B,C,H,W]
```

Normalization 也切换：

```text
BN
↔
LN
```

所以 HybridEncoder 是一个非常典型的：

```text
CNN/Transformer bridge
```

---

# 146. 本课最终源码逻辑伪代码

不要背官方变量名。

只背下面：

```python
def forward(features):

    # 1. channel alignment
    features = project_to_hidden_dim(
        features
    )

    # 2. selected-level transformer
    for level in selected_levels:

        x = feature_map_to_tokens(
            features[level]
        )

        pos = build_2d_position_encoding()

        x = transformer_encoder(
            x,
            pos
        )

        features[level] = tokens_to_map(x)

    # 3. top-down FPN
    inner = [features[-1]]

    for high_to_low:

        high = lateral(inner[0])
        high = upsample(high)

        low = corresponding_feature

        fused = fuse(high, low)

        inner.insert(
            0,
            fuse_block(fused)
        )

    # 4. bottom-up PAN
    outputs = [inner[0]]

    for low_to_high:

        low = downsample(outputs[-1])
        high = corresponding_inner

        fused = fuse(low, high)

        outputs.append(
            pan_block(fused)
        )

    return outputs
```

如果你能自己写出这段伪代码，本课已经真正掌握。

---

# 147. 从 640×640 到 Decoder 的完整链条现在已经延伸到这里

```text
Image
[B,3,640,640]

↓ HGNetv2-B5

C3
[B,512,80,80]

C4
[B,1024,40,40]

C5
[B,2048,20,20]

↓ HybridEncoder

P3
[B,384,80,80]

P4
[B,384,40,40]

P5
[B,384,20,20]

↓ 下一课 DFINETransformer

multi-scale flatten
8400 spatial positions

↓

query selection

↓

300 object queries

↓

decoder layers
```

现在你已经走完：

```text
Image → Backbone → Encoder
```

---

# 148. 本课练习 1

为什么：

```text
C3 [B,512,80,80]
```

经过：

```text
1×1 Conv 512→384
```

后 H/W 不变？

请用卷积输出公式解释。

---

# 149. 练习 2

计算三个 projection Conv 的权重参数量：

```text
512→384
1024→384
2048→384
```

并求总和。

---

# 150. 练习 3

为什么：

```text
use_encoder_idx=[2]
```

选择的是：

```text
20×20
```

而不是：

```text
40×40
```

？

---

# 151. 练习 4

`20×20` feature flatten 后：

```text
N = ?
```

Transformer 输入 shape 是什么？

---

# 152. 练习 5

hidden_dim=384，nhead=8。

每个 head 的 dimension 是多少？

---

# 153. 练习 6

为什么 position embedding dimension 必须能被 4 整除？

---

# 154. 练习 7

比较：

```text
80×80
```

与：

```text
20×20
```

self-attention matrix 的元素数量比例。

---

# 155. 练习 8

Top-down 第一次融合：

```text
[B,384,40,40]
```

和：

```text
[B,384,40,40]
```

使用 cat 后 shape 是什么？

---

# 156. 练习 9

如果改成：

```text
fuse_op=sum
```

上一题输出 shape 又是什么？

---

# 157. 练习 10

为什么 FPN 是：

```text
20→40→80
```

而 PAN 是：

```text
80→40→20
```

？

请从：

```text
semantic information
spatial information
```

两个角度解释。

---

# 158. 练习 11

SCDown：

```text
[B,384,80,80]
```

经过：

```text
1×1
+
depthwise 3×3 stride2
```

输出 shape 是什么？

---

# 159. 练习 12

为什么最终：

```text
P3/P4/P5
```

的 stride 与：

```text
C3/C4/C5
```

相同？

---

# 160. 练习 13

为什么：

```text
proj_feats
inner_outs
outs
```

可能 shape 完全一样，但不能认为它们是同一组 feature？

---

# 161. 练习 14

解释：

```text
intra-scale interaction
```

和：

```text
cross-scale fusion
```

的区别。

---

# 162. 练习 15

如果把：

```text
hidden_dim 384 → 512
```

需要至少检查后续哪个配置字段是否同步变化？

---

# 163. 练习答案摘要

## 1

1×1：

```text
K=1
S=1
P=0
```

所以 H/W 不变。

## 2

\[
512\times384
+
1024\times384
+
2048\times384
=
1,376,256
\]

仅 Conv 权重。

## 3

Python：

```text
0=C3
1=C4
2=C5
```

## 4

\[
20\times20=400
\]

输入：

```text
[B,400,384]
```

## 5

\[
384/8=48
\]

## 6

因为 x/y 各分 sin/cos：

```text
x sin
x cos
y sin
y cos
```

共四组。

## 7

\[
(6400/400)^2=256
\]

## 8

```text
[B,768,40,40]
```

## 9

```text
[B,384,40,40]
```

## 11

```text
[B,384,40,40]
```

## 12

encoder 内部虽上/下采样，但最终重新输出三个原有 pyramid levels。

## 15

至少检查：

```text
DFINETransformer.feat_channels
```

以及所有假定 hidden dimension 的相关模块。

---

# 164. 本课必须掌握的 12 个结论

进入第 11 课前，请确保能独立解释：

```text
1. HybridEncoder 为什么存在？

2. 为什么先把 512/1024/2048 统一成 384？

3. 为什么只在 stride32 做 Transformer？

4. [B,384,20,20] 怎样变成 [B,400,384]？

5. 2D sin-cos position embedding 为什么需要？

6. FPN 为什么从 20→40→80？

7. PAN 为什么从 80→40→20？

8. cat 为什么产生 768 channels？

9. Fuse Block 为什么还要把 768 压回 384？

10. SCDown 为什么比普通 3×3 downsample 更高效？

11. 为什么最终仍输出 stride 8/16/32？

12. 这三个输出怎样连接第 7 课的 8400 spatial positions？
```

如果这 12 个问题能真正回答，HybridEncoder 就不再是一个“夹在 backbone 和 decoder 中间、不知道有什么用的模块”。

---

# 165. 本课最核心的一句话

> **HybridEncoder 的本质，是用低成本的方式把“全局语义建模”和“双向多尺度融合”组合起来：只在低分辨率高层特征上使用昂贵的 Transformer，再通过 FPN/PAN 把信息传播到整个特征金字塔。**

这就是它为什么非常适合：

```text
Real-Time DETR
```

---

# 166. 下一课：DFINETransformer Decoder

下一课我们从本课最后三个 Tensor 直接开始：

```text
P3 [B,384,80,80]
P4 [B,384,40,40]
P5 [B,384,20,20]
```

然后真正进入：

```text
_prepare_encoder_input
↓
flatten multi-scale features
↓
8400 positions
↓
encoder output proposals
↓
classification score
↓
top-k query selection
↓
300 queries
↓
reference points
↓
DFINE decoder layers
```

到那时，我们会第一次真正回答：

> **Object Query 在 DEIM/D-FINE 中到底从哪里来？**

以及：

> **为什么第 5 课讲的“learnable object query”在现代实时 DETR 中已经演化成更复杂的 query initialization / selection 机制？**

---

# 167. 源码核对说明

本课按当前官方 DEIMv2 `engine/deim/hybrid_encoder.py` 的实现核对，并以课程主线 `deim_dfine_x` 的 X 配置进行具体 shape 推导。

当前实现中 `HybridEncoder` 的主要构成包括：

```text
input_proj
TransformerEncoder
2D sin-cos positional embedding
lateral_convs
fpn_blocks
downsample_convs
pan_blocks
```

其 forward 主流程是：

```text
projection
→ selected-level Transformer encoding
→ top-down fusion
→ bottom-up fusion
→ outputs
```

当前 X 级 D-FINE 配置将：

```text
HGNetv2-B5
return_idx=[1,2,3]
```

与：

```text
HybridEncoder hidden_dim=384
dim_feedforward=2048
```

以及：

```text
DFINETransformer
feat_channels=[384,384,384]
```

对齐。

课程中的：

```text
512/1024/2048
→
384/384/384
```

以及：

```text
80×80 / 40×40 / 20×20
```

均基于这条配置链推导。

由于仓库仍可能更新，如果你本地 checkout 的 commit 与课程核对版本不同，最终应以：

```text
你的 merged YAML
+
你本地源码
+
forward hook 实测 shape
```

为准。

---

# 168. 推荐源码位置

重点文件：

```text
engine/deim/hybrid_encoder.py
```

重点类/函数：

```text
HybridEncoder
TransformerEncoderLayer
TransformerEncoder
RepNCSPELAN4
CSPLayer
CSPLayer2
VGGBlock
SCDown
ConvNormLayer
ConvNormLayer_fuse
```

重点配置：

```text
configs/base/dfine_hgnetv2.yml
configs/deim_dfine/dfine_hgnetv2_x_coco.yml
```

---

# 169. 本课结束后的完整模型地图

```text
                        DEIM-D-FINE-X

Image
[B,3,640,640]

        │
        ▼

HGNetv2-B5
第 9 课

        │
        ├── C3 [B,512,80,80]
        ├── C4 [B,1024,40,40]
        └── C5 [B,2048,20,20]

        │
        ▼

HybridEncoder
第 10 课

        │
        ├── input projection
        │
        ├── C5 Transformer
        │
        ├── top-down FPN
        │
        └── bottom-up PAN
        │
        ▼

        ├── P3 [B,384,80,80]
        ├── P4 [B,384,40,40]
        └── P5 [B,384,20,20]

        │
        ▼

DFINETransformer
第 11～13 课

        │
        ▼

300 object predictions

        │
        ▼

class + bbox
```

到这里，我们已经从：

```text
[B,3,640,640]
```

连续追到了：

```text
三个 detection-ready multi-scale features
```

下一课开始，真正进入检测头。
