# 第 7 课：Deformable Attention------为什么不用全局 Attention

> **课程定位**：DEIMv2 从零到源码 · 第 7 / 24 课\
> **前置知识**：Tensor/Shape、Q/K/V Attention、Object Query、Hungarian
> O2O。\
> **目标**：彻底理解 reference point、sampling offset、sampling
> location、attention weight、multi-scale indexing、双线性插值，以及
> DEIM/D-FINE `MSDeformableAttention` 的真实 shape。

------------------------------------------------------------------------

# 1. 从 8400 个图像位置开始

640×640 输入，三个典型尺度：

``` text
stride 8 : 80×80 = 6400
stride 16: 40×40 = 1600
stride 32: 20×20 = 400
```

总位置数：

\[ N_v=6400+1600+400=8400 \]

Flatten + concat 后：

``` text
memory: [B,8400,C]
```

如果有：

``` text
300 Object Queries
8 attention heads
```

普通 dense cross-attention 的 Query-position score 数约为：

\[ 300`\times8400`{=tex}`\times8`{=tex} = 20,160,000 \]

每张图、每层就是约 2016 万个 head-specific interactions。

但检测 Query 往往已经有一个目标的大致空间参考。既然当前 Query
认为目标大概在右下方，它是否还需要每层都同等扫描左上天空等全部位置？

Deformable Attention 的答案是：

``` text
不 dense 地看完所有位置
↓
先给 Query 一个 Reference Point / Box
↓
Query 预测少量 Sampling Offsets
↓
只在多个尺度上取少量特征
↓
预测 Attention Weights
↓
加权融合
```

------------------------------------------------------------------------

# 2. 普通 Attention 与 Deformable Attention

普通 cross-attention：

\[ `\mathrm{Attention}`{=tex}(Q,K,V) = `\mathrm{softmax}`{=tex}
`\left`{=tex}( `\frac{QK^T}{\sqrt d}`{=tex} `\right`{=tex})V \]

若：

``` text
Q: [B,Nq,C]
K,V: [B,Nv,C]
```

score：

``` text
[B,h,Nq,Nv]
```

Deformable cross-attention 不为每个 Query 建立完整的 `Nv=8400` score
vector，而是只访问少量学习出来的位置。

  -----------------------------------------------------------------------
  普通 Cross-Attention                Deformable Cross-Attention
  ----------------------------------- -----------------------------------
  Query 访问全部 memory positions     Query 访问少量 sampling positions

  dense                               sparse

  通常通过 QK 相似度产生全局 scores   本实现由 Query 预测 sampling
                                      weights

  空间先验弱                          reference point / box 提供空间参考

  interaction 数随 `Nv` 增长          主要随 sampling point 数增长
  -----------------------------------------------------------------------

Deformable Attention 并不是"没有 Attention"，而是把候选信息源从全部
tokens 改成少量 learned sampling locations。

------------------------------------------------------------------------

# 3. Reference Point：从哪里附近找？

设：

\[ r=(x,y) \]

例如：

``` text
r=(0.60,0.40)
```

通常是 normalized coordinates：

``` text
x,y ∈ [0,1]
```

含义是：

``` text
图像宽度方向约 60%
图像高度方向约 40%
```

于是问题从：

``` text
“8400 个位置里目标在哪？”
```

变成：

``` text
“围绕这个空间参考，哪些位置最值得取特征？”
```

Reference Point 并不等于最终 bounding box；它是当前 decoder
阶段用于空间采样的参考。

------------------------------------------------------------------------

# 4. 为什么叫 Deformable？

如果永远使用固定 3×3 grid：

``` text
reference center
↓
固定邻域
```

这只是固定局部采样。

Deformable 的关键是：

``` text
Sampling Offset 由网络预测
```

例如：

``` text
Δp0=(-0.04,-0.02)
Δp1=(+0.08,+0.01)
Δp2=(+0.02,+0.10)
```

采样位置：

\[ p_k=r+`\Delta `{=tex}p_k \]

不同：

``` text
Query
Head
Decoder Layer
Feature Level
```

都可以学习不同的 sampling pattern。

------------------------------------------------------------------------

# 5. Multi-Scale Deformable Attention

检测器拥有多尺度特征：

``` text
P3: 80×80
P4: 40×40
P5: 20×20
```

一个 Query 可以同时从三个尺度各取少量信息。

抽象公式：

\[ `\mathrm{MSDeformAttn}`{=tex}(q) = `\sum`{=tex}\_m W_m `\left[
\sum_l
\sum_k
A_{mlqk}
W'_m
x_l
\left(
r_{ql}+\Delta p_{mlqk}
\right)
\right]`{=tex}\]

符号：

``` text
q = Query
m = attention head
l = feature level
k = sampling point
A = attention weight
Δp = sampling offset
x_l = 第 l 个 feature map
```

本质：

> 每个 Query、每个 head，在多个尺度上选少量点，取出 feature vectors
> 后加权求和。

------------------------------------------------------------------------

# 6. `spatial_shapes` 是什么？

三个 maps：

``` text
P3 [B,C,80,80]
P4 [B,C,40,40]
P5 [B,C,20,20]
```

保存：

``` python
spatial_shapes = [
    [80,80],
    [40,40],
    [20,20],
]
```

shape：

``` text
[L,2] = [3,2]
```

为什么需要它？

因为 flatten 后：

``` text
P3 → [B,6400,C]
P4 → [B,1600,C]
P5 → [B, 400,C]
```

concat：

``` text
[B,8400,C]
```

单看 8400 已经不知道各 token 原来的二维 feature level。Deformable
sampling 必须知道每个 level 的 `H,W`。

------------------------------------------------------------------------

# 7. `level_start_index`

各 level 长度：

``` text
6400
1600
400
```

在 concat memory 中的起点：

``` text
Level 0: 0
Level 1: 6400
Level 2: 8000
```

所以：

``` text
level_start_index = [0,6400,8000]
```

二维位置 `(h,w)` 在单个 level 的 flatten index：

\[ n=hW+w \]

例如 P4 的全局 index：

\[ 6400+h`\cdot40`{=tex}+w \]

经典 Deformable DETR 经常显式传 `level_start_index`。DEIM/D-FINE 的部分
Python core 可以直接依据：

``` python
[h*w for h,w in spatial_shapes]
```

把 memory `split` 回各 level，因此具体源码里未必始终显式传一个同名
Tensor。概念必须理解，但不要机械寻找变量名。

------------------------------------------------------------------------

# 8. DEIM/D-FINE `MSDeformableAttention` 的两条预测支路

核心结构可概括为：

``` text
Query [B,Nq,C]
│
├─ Linear → Sampling Offsets
│
└─ Linear → Attention Weights
```

翻译成人话：

``` text
Sampling Offsets:
“我要去哪些相对位置看？”

Attention Weights:
“这些位置中谁更重要？”
```

这就是整个模块的核心。

------------------------------------------------------------------------

# 9. 标准等采样点版本的 Shape

设：

``` text
B  = batch
Nq = queries
h  = heads
L  = levels
P  = points per level
C  = hidden dim
d  = C/h
```

Sampling offsets：

``` text
[B,Nq,h,L,P,2]
```

最后的 2：

``` text
Δx, Δy
```

Attention weights：

``` text
[B,Nq,h,L,P]
```

例如：

``` text
B=2
Nq=300
h=8
L=3
P=4
```

offset：

``` text
[2,300,8,3,4,2]
```

每个 Query 有：

\[ 8`\times3`{=tex}`\times4`{=tex}=96 \]

个 head-specific 2D offsets。

------------------------------------------------------------------------

# 10. DEIM 的关键细节：每个 Level 点数可以不同

真实实现支持：

``` text
num_points=[3,6,3]
```

即：

``` text
Level 0: 3
Level 1: 6
Level 2: 3
```

每个 head：

\[ 3+6+3=12 \]

个采样点。

源码因此可将 offsets 表示成：

``` text
[B,Nq,h,12,2]
```

weights：

``` text
[B,Nq,h,12]
```

之后按照：

``` text
[3,6,3]
```

再 split 回不同 levels。

这就是为什么读 DEIM 源码时不能死背：

``` text
[B,Nq,h,L,P,2]
```

标准形式表达数学更直观，但真实实现为了支持可变 `P_l`，会合并 point 维。

------------------------------------------------------------------------

# 11. Attention Weights

Query 经过 Linear：

``` text
[B,Nq,C]
→
[B,Nq,h,sum(P_l)]
```

例如：

``` text
[B,300,8,12]
```

然后在 sampling-point 维：

``` python
softmax(dim=-1)
```

对于固定：

``` text
batch
query
head
```

所有采样点权重之和：

\[ 1 \]

普通 attention 是：

``` text
Query → 对 8400 Keys 打分
```

这里则是：

``` text
Query → 直接预测少量采样点权重
```

------------------------------------------------------------------------

# 12. 2D Reference Point

reference：

``` text
[B,Nq,L,2]
```

最后：

``` text
(x,y)
```

通常 normalized 到 `[0,1]`。

核心关系：

\[ sampling_location = reference + `\frac{offset}{(W,H)}`{=tex} \]

为什么 spatial shape 通常保存：

``` text
(H,W)
```

却要变成：

``` text
(W,H)
```

？

因为 offset 顺序是：

``` text
(dx,dy)
```

所以必须：

``` text
dx/W
dy/H
```

而不是：

``` text
dx/H
dy/W
```

这也是源码中看到 `flip` 的原因之一。

------------------------------------------------------------------------

# 13. 为什么不同 Level 必须分别归一化？

在 80×80 map：

``` text
dx=4 cells
```

normalized：

\[ 4/80=0.05 \]

在 20×20 map：

\[ 4/20=0.20 \]

同样移动 4 个 feature cells，在不同尺度对应不同 normalized distance。

所以 offsets 必须根据每个 level 的 `W,H` 做 normalization。

------------------------------------------------------------------------

# 14. 4D Reference Box

实现还支持：

``` text
[...,4]
```

可以理解成：

``` text
(cx,cy,w,h)
```

这时 offset 可以根据当前 reference box 的：

``` text
w,h
```

缩放。

核心直觉：

``` text
小目标
→ 采样区域自然较小

大目标
→ 采样区域自然较大
```

抽象：

\[ offset' = `\Delta `{=tex}p `\times `{=tex}point_scale
`\times`{=tex}(w,h) `\times `{=tex}offset_scale \]

然后：

\[ sampling_location = (cx,cy)+offset' \]

因此 localization 与 feature sampling 开始互相影响。

------------------------------------------------------------------------

# 15. 为什么这和 Box Refinement 联系很深？

Decoder 可以形成循环：

``` text
当前 Query + 当前 Reference Box
↓
Deformable Cross-Attention
↓
获取相关视觉信息
↓
BBox Refinement
↓
更准确的 Reference Box
↓
下一 Decoder Layer
↓
围绕更准确位置再次采样
```

所以 bbox localization 并不是只在 decoder 最后一层发生。

当前定位结果会影响：

``` text
下一层去哪里读取 feature
```

第 11 课深入 decoder 时会追踪 reference box 的创建与更新。

------------------------------------------------------------------------

# 16. Sampling Location 为什么可以是小数？

例如：

``` text
(13.2,27.7)
```

feature map 只有整数 cell。

它附近是：

``` text
(13,27)  (14,27)
(13,28)  (14,28)
```

需要：

``` text
Bilinear Interpolation
```

来取得连续位置的 feature。

------------------------------------------------------------------------

# 17. 从 Linear Interpolation 到 Bilinear

一维：

``` text
x=0 → 10
x=1 → 20
```

取：

``` text
x=0.25
```

则：

\[ 10(0.75)+20(0.25)=12.5 \]

二维时使用四个邻居。

设：

\[ d_x=x-x_0,`\qquad `{=tex}d_y=y-y_0 \]

四个权重：

\[ w\_{00}=(1-d_x)(1-d_y) \]

\[ w\_{10}=d_x(1-d_y) \]

\[ w\_{01}=(1-d_x)d_y \]

\[ w\_{11}=d_xd_y \]

输出：

\[ v= w\_{00}v\_{00} +w\_{10}v\_{10} +w\_{01}v\_{01} +w\_{11}v\_{11} \]

------------------------------------------------------------------------

# 18. Bilinear 手算

四个值：

``` text
10  20
30  40
```

取：

``` text
dx=0.25
dy=0.50
```

权重：

``` text
0.375
0.125
0.375
0.125
```

输出：

\[ 10(0.375) +20(0.125) +30(0.375) +40(0.125) =22.5 \]

------------------------------------------------------------------------

# 19. 为什么不能直接 Round？

如果：

``` text
13.2 → 13
13.3 → 13
13.4 → 13
```

offset 连续改变，但采样值长期不变。

这会让 sampling coordinate 很难通过梯度学习。

Bilinear interpolation 让输出随坐标连续变化，因此 detection loss
可以训练：

``` text
sampling offsets
```

------------------------------------------------------------------------

# 20. PyTorch `grid_sample`

DEIM/D-FINE Python core 使用：

``` python
torch.nn.functional.grid_sample
```

做连续二维采样。

Reference/sampling locations 常是：

``` text
[0,1]
```

而 `grid_sample` 的 grid 使用：

``` text
[-1,1]
```

因此源码可见：

``` python
sampling_grids = 2 * sampling_locations - 1
```

映射：

``` text
0   → -1
0.5 →  0
1   → +1
```

------------------------------------------------------------------------

# 21. `grid_sample` Shape

输入 feature：

``` text
[N,C,H,W]
```

grid：

``` text
[N,Hout,Wout,2]
```

输出：

``` text
[N,C,Hout,Wout]
```

Deformable Attention 通过 `reshape/permute` 把：

``` text
head
query
point
```

重新排布成 `grid_sample` 能接受的 layout。

------------------------------------------------------------------------

# 22. Value 的 Head 拆分

Memory：

``` text
[B,Nv,C]
```

若：

``` text
C=h*d
```

可以拆：

``` text
[B,Nv,h,d]
```

某些优化实现则提前变成：

``` text
[B,h,d,Nv]
```

然后按：

``` text
6400
1600
400
```

split 成 levels。

重要原则：

> 不背死一个 layout；每次根据 `reshape / permute / split` 推导。

------------------------------------------------------------------------

# 23. Level Feature 恢复二维

假设：

``` text
B=2
h=8
d=32
Level 0 = 80×80
```

可恢复：

``` text
[B*h,d,80,80]
```

即：

``` text
[16,32,80,80]
```

为什么把 head 合入 batch？

因为 `grid_sample` 可以把每个 head 当作独立 batch item 一次处理。

------------------------------------------------------------------------

# 24. Sampling Grid Shape

标准：

``` text
sampling_locations:
[B,Nq,h,L,P,2]
```

取某个 level：

``` text
[B,Nq,h,P,2]
```

permute：

``` text
[B,h,Nq,P,2]
```

合并 B/h：

``` text
[B*h,Nq,P,2]
```

这正好可作为：

``` text
grid_sample grid
```

其中：

``` text
Hout=Nq
Wout=P
```

------------------------------------------------------------------------

# 25. `grid_sample` 后的 Shape

输入：

``` text
[B*h,d,H_l,W_l]
```

grid：

``` text
[B*h,Nq,P_l,2]
```

输出：

``` text
[B*h,d,Nq,P_l]
```

含义：

> 每个 batch、head、query 已经从该 feature level 取到 `P_l` 个 d 维
> feature vectors。

------------------------------------------------------------------------

# 26. 三个 Levels 合并

如果：

``` text
P=[3,6,3]
```

则：

``` text
Level 0:
[B*h,d,Nq,3]

Level 1:
[B*h,d,Nq,6]

Level 2:
[B*h,d,Nq,3]
```

concat：

``` text
[B*h,d,Nq,12]
```

weights：

``` text
[B*h,1,Nq,12]
```

broadcast multiply：

``` text
[B*h,d,Nq,12]
```

sum 最后一维：

``` text
[B*h,d,Nq]
```

恢复 heads：

``` text
[B,h*d,Nq]
=
[B,C,Nq]
```

transpose：

``` text
[B,Nq,C]
```

所以模块保持：

``` text
[B,Nq,C]
→
[B,Nq,C]
```

------------------------------------------------------------------------

# 27. 完整 Shape Trace

设：

``` text
B=2
Nq=300
C=256
h=8
d=32

spatial:
80×80
40×40
20×20

P=[3,6,3]
```

Memory：

``` text
[2,8400,256]
```

Query：

``` text
[2,300,256]
```

Offsets：

``` text
[2,300,8,12,2]
```

Weights：

``` text
[2,300,8,12]
```

Reference：

``` text
[2,300,3,2]
```

或：

``` text
[2,300,3,4]
```

采样后：

``` text
P3:
[16,32,300,3]

P4:
[16,32,300,6]

P5:
[16,32,300,3]
```

concat：

``` text
[16,32,300,12]
```

weighted sum：

``` text
[16,32,300]
```

最终：

``` text
[2,300,256]
```

------------------------------------------------------------------------

# 28. 为什么不同 Head 能学习不同空间模式？

offset 有 head 维：

``` text
[B,Nq,h,...,2]
```

因此不同 head 可以学：

``` text
目标中心
左边缘
右边缘
上下文
长轴方向
短轴方向
```

等不同 pattern。

这不是人工指定，而是模型具备这种表达能力。

------------------------------------------------------------------------

# 29. Offset 初始化

DEIM/D-FINE 实现会给不同 heads 构造不同初始方向，概念上使用：

``` text
cos(theta)
sin(theta)
```

使 heads 初始朝不同方向展开，并让不同 sampling points 有不同半径。

直觉：

``` text
不要让所有采样点一开始全部堆在 reference center
```

而是先提供空间多样性，再让训练学习真正有用的 pattern。

------------------------------------------------------------------------

# 30. Attention Weight 初始化

公开实现会把 weight projection 的参数初始化成让初始 logits 相同的状态。

经过 softmax：

``` text
各 sampling points 初始权重近似均匀
```

所以初始状态可以理解为：

``` text
位置先分散
权重先公平
训练再学习偏好
```

------------------------------------------------------------------------

# 31. 复杂度到底省在哪里？

Dense interaction：

\[ O(N_qN_v) \]

Sparse deformable sampling：

\[ O `\left`{=tex}( N_q`\sum`{=tex}\_lP_l `\right`{=tex}) \]

再乘 heads。

例：

``` text
Nq=300
Nv=8400
h=8
sum(P)=12
```

Dense：

\[ 300`\times8400`{=tex}`\times8`{=tex} =20,160,000 \]

Sparse sampling interactions：

\[ 300`\times12`{=tex}`\times8`{=tex} =28,800 \]

粗略 interaction-count 比：

\[ 700 \]

但绝对不能说：

``` text
“整个模型因此快 700 倍”
```

真实 runtime 还有：

``` text
Linear projections
grid_sample
memory layout
GPU kernel efficiency
Self-Attention
FFN
Encoder
Backbone
Backward
```

700 只是在这个特定 interaction-count 指标上的比较。

------------------------------------------------------------------------

# 32. Deformable Attention 不是"只能看局部"

因为：

``` text
offset 是 learned
```

而且存在：

``` text
multiple heads
multiple levels
multiple decoder layers
iteratively updated references
```

所以更准确的说法是：

> 它不 dense
> 地访问全部空间位置，而是学习稀疏、内容相关、reference-driven 的采样。

------------------------------------------------------------------------

# 33. Self-Attention 与 Deformable Cross-Attention

Decoder layer 可以理解成：

``` text
Object Queries
↓
Self-Attention
“Queries 之间交流”
↓
Deformable Cross-Attention
“每个 Query 去图像取视觉证据”
↓
FFN
“逐 Query 非线性变换”
```

Self-attention：

``` text
Query ↔ Query
```

Cross-attention：

``` text
Query ↔ Image Features
```

不要把两者混在一起。

------------------------------------------------------------------------

# 34. Reference Point 从哪里来？

在 DETR 后续模型中可能来自：

``` text
encoder proposal
query initialization
learned reference
previous decoder layer refined box
```

DEIM/D-FINE 还会结合 bbox refinement。

本课只要求理解 reference 的作用。

第 11 课会真正追踪：

``` text
在哪里创建
怎样变成 query reference
每层怎样更新
怎样送入下一 decoder layer
```

------------------------------------------------------------------------

# 35. 为什么这特别适合 Detection？

Detection 同时问：

``` text
What?
Where?
```

Reference point / box 把：

``` text
Where
```

直接引入 feature acquisition。

当前定位：

``` text
决定去哪里采样
```

采样到的视觉信息：

``` text
又帮助下一步定位
```

因此形成：

``` text
定位
↔
取特征
```

的闭环。

------------------------------------------------------------------------

# 36. PyTorch 实验 1：得到 8400 Memory Tokens

``` python
import torch

B=2
C=256

p3=torch.randn(B,C,80,80)
p4=torch.randn(B,C,40,40)
p5=torch.randn(B,C,20,20)

def flatten_feature(x):
    return x.flatten(2).transpose(1,2)

m3=flatten_feature(p3)
m4=flatten_feature(p4)
m5=flatten_feature(p5)

memory=torch.cat(
    [m3,m4,m5],
    dim=1
)

print(m3.shape)
print(m4.shape)
print(m5.shape)
print(memory.shape)
```

应得到：

``` text
[2,6400,256]
[2,1600,256]
[2,400,256]
[2,8400,256]
```

------------------------------------------------------------------------

# 37. 实验 2：`spatial_shapes` 与 `level_start_index`

``` python
spatial_shapes=torch.tensor([
    [80,80],
    [40,40],
    [20,20],
])

level_sizes=(
    spatial_shapes[:,0]
    *
    spatial_shapes[:,1]
)

level_start_index=torch.cat([
    torch.zeros(
        1,
        dtype=torch.long
    ),
    level_sizes.cumsum(0)[:-1]
])

print(level_sizes)
print(level_start_index)
```

应得到：

``` text
[6400,1600,400]

[0,6400,8000]
```

------------------------------------------------------------------------

# 38. 实验 3：Memory Split 回 Levels

``` python
parts=memory.split(
    [6400,1600,400],
    dim=1
)

print(
    [x.shape for x in parts]
)

p3_back=(
    parts[0]
    .transpose(1,2)
    .reshape(B,C,80,80)
)

print(p3_back.shape)
```

输出：

``` text
[2,256,80,80]
```

------------------------------------------------------------------------

# 39. 实验 4：Query 预测 Offsets

``` python
import torch.nn as nn

B=2
Nq=300
C=256
heads=8
points=[3,6,3]

Psum=sum(points)

query=torch.randn(
    B,Nq,C
)

offset_proj=nn.Linear(
    C,
    heads*Psum*2
)

offsets=offset_proj(
    query
).reshape(
    B,Nq,heads,Psum,2
)

print(offsets.shape)
```

输出：

``` text
[2,300,8,12,2]
```

------------------------------------------------------------------------

# 40. 实验 5：Attention Weights

``` python
weight_proj=nn.Linear(
    C,
    heads*Psum
)

weights=weight_proj(
    query
).reshape(
    B,Nq,heads,Psum
)

weights=weights.softmax(-1)

print(weights.shape)
print(weights[0,0,0].sum())
```

输出：

``` text
[2,300,8,12]
```

权重和应接近：

``` text
1.0
```

------------------------------------------------------------------------

# 41. 实验 6：手算 Bilinear

``` python
v00=torch.tensor(10.0)
v10=torch.tensor(20.0)
v01=torch.tensor(30.0)
v11=torch.tensor(40.0)

dx=0.25
dy=0.50

out=(
    (1-dx)*(1-dy)*v00
    +dx*(1-dy)*v10
    +(1-dx)*dy*v01
    +dx*dy*v11
)

print(out)
```

结果：

``` text
22.5
```

------------------------------------------------------------------------

# 42. 实验 7：真正调用 `grid_sample`

``` python
import torch.nn.functional as F

feature=torch.arange(
    16,
    dtype=torch.float32
).reshape(
    1,1,4,4
)

grid=torch.tensor([
    [[[0.0,0.0]]]
])

sample=F.grid_sample(
    feature,
    grid,
    mode="bilinear",
    padding_mode="zeros",
    align_corners=False
)

print(feature)
print(sample)
```

继续测试：

``` text
[-1,-1]
[1,1]
[0.5,0.5]
```

重点是理解：

``` text
continuous coordinates
→ differentiable feature sampling
```

------------------------------------------------------------------------

# 43. 实验 8：证明 Sampling Coordinate 有梯度

``` python
feature=torch.randn(
    1,1,8,8
)

grid=torch.tensor(
    [[[[0.2,-0.3]]]],
    requires_grad=True
)

out=F.grid_sample(
    feature,
    grid,
    mode="bilinear",
    align_corners=False
)

loss=out.sum()
loss.backward()

print(grid.grad)
```

若 gradient 非零：

``` text
sampling location
```

就可以通过 loss 学习。

这就是 learned offsets 能被训练的关键。

------------------------------------------------------------------------

# 44. 实验 9：最小 Single-Level Deformable Sampler

``` python
class TinyDeformableSampler(
    nn.Module
):
    def __init__(
        self,
        dim=32,
        num_points=4
    ):
        super().__init__()

        self.num_points=num_points

        self.offset=nn.Linear(
            dim,
            num_points*2
        )

        self.weight=nn.Linear(
            dim,
            num_points
        )

    def forward(
        self,
        query,
        feature,
        reference
    ):
        B,Nq,C=query.shape

        offsets=self.offset(
            query
        ).reshape(
            B,Nq,
            self.num_points,
            2
        )

        weights=self.weight(
            query
        ).softmax(-1)

        locations=(
            reference[:,:,None,:]
            +0.05*offsets
        )

        grid=2*locations-1

        sampled=F.grid_sample(
            feature,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False
        )

        out=(
            sampled
            *
            weights[:,None,:,:]
        ).sum(-1)

        return out.transpose(1,2)
```

输入：

``` text
query:     [B,Nq,C]
feature:   [B,C,H,W]
reference: [B,Nq,2]
```

输出：

``` text
[B,Nq,C]
```

------------------------------------------------------------------------

# 45. Tiny Sampler Shape Trace

Offset：

``` text
[B,Nq,C]
→ Linear
[B,Nq,P*2]
→ reshape
[B,Nq,P,2]
```

Reference：

``` text
[B,Nq,2]
→ unsqueeze
[B,Nq,1,2]
```

Broadcast：

``` text
locations:
[B,Nq,P,2]
```

`grid_sample`：

``` text
feature:
[B,C,H,W]

grid:
[B,Nq,P,2]

output:
[B,C,Nq,P]
```

weights：

``` text
[B,Nq,P]
→
[B,1,Nq,P]
```

weighted sum：

``` text
[B,C,Nq]
```

transpose：

``` text
[B,Nq,C]
```

------------------------------------------------------------------------

# 46. Tiny 与真实 DEIM 还差什么？

真实版本还包括：

``` text
multi-head
multi-level
不同 level 不同 point 数
reference box scaling
offset initialization
value layout optimization
self-attention
residual / normalization
FFN
iterative box refinement
denoising queries
```

但最核心机制已经完整：

``` text
Query
→ Offsets
→ Sampling Locations
→ Bilinear Sampling
→ Sampled Features
→ Attention Weights
→ Weighted Sum
```

------------------------------------------------------------------------

# 47. 源码逐句翻译法

看到：

``` python
sampling_offsets = ...
```

翻译：

``` text
Query 决定去哪些相对位置看。
```

看到：

``` python
attention_weights = ...
```

翻译：

``` text
Query 决定各采样点有多重要。
```

看到：

``` text
sampling_locations = reference + ...
```

翻译：

``` text
把相对位移变成真正访问位置。
```

看到：

``` python
F.grid_sample(...)
```

翻译：

``` text
从 feature map 的连续二维坐标取特征。
```

看到：

``` text
sampled * attention_weights
```

翻译：

``` text
按重要性融合这些视觉证据。
```

------------------------------------------------------------------------

# 48. 常见 Shape Bug

## Bug 1：H/W 与 x/y 对错

`spatial_shapes`：

``` text
(H,W)
```

offset：

``` text
(dx,dy)
```

归一化必须：

``` text
dx/W
dy/H
```

------------------------------------------------------------------------

## Bug 2：把 `num_levels` 与 `num_points` 混淆

``` text
L=3
P=[3,6,3]
```

总点数不是：

``` text
3×6
```

而是：

``` text
3+6+3=12
```

------------------------------------------------------------------------

## Bug 3：忘记 Head 维

完整 offsets：

``` text
[B,Nq,h,Psum,2]
```

不是：

``` text
[B,Nq,Psum,2]
```

------------------------------------------------------------------------

## Bug 4：把 Offset 当绝对坐标

Offset：

``` text
相对位移
```

必须结合：

``` text
reference
```

得到 sampling location。

------------------------------------------------------------------------

# 49. 常见概念 Bug

## "Reference Point 就是最终 Box"

错误。

它是当前阶段的空间参考；最终框通常经历 decoder 多层 refinement。

## "Deformable Attention 只看目标内部"

错误。

learned offsets 可以采：

``` text
目标内部
边缘
外部上下文
```

## "Deformable Attention 就是局部 Attention"

不够准确。

它是：

``` text
learned sparse spatial sampling
```

而不是固定 local window。

## "interaction 少 700 倍，所以模型快 700 倍"

错误。

那只是特定 interaction-count 的数量级比较，不是端到端 latency 比值。

------------------------------------------------------------------------

# 50. 为什么目标外上下文也有用？

例如：

``` text
船 ↔ 水面
汽车 ↔ 道路
人 ↔ 自行车
```

上下文可能帮助类别与定位判断。

所以 sampling points 不需要被强制限制在当前 bounding box 内。

------------------------------------------------------------------------

# 51. 为什么 Multi-Scale 很重要？

小目标在：

``` text
20×20
```

feature 上可能只剩很少 cells。

``` text
80×80
```

保留更多空间细节。

低分辨率 feature 又通常具有更强高级语义。

所以一个 Query
同时从不同尺度取少量信息，比只固定读取一个尺度更适合检测。

------------------------------------------------------------------------

# 52. 与前六课连接

第 2 课 Tensor：

``` text
reshape
permute
flatten
split
broadcast
```

本课全部用上。

第 3 课 Multi-Scale：

``` text
Backbone/Encoder
制造 P3/P4/P5
```

本课回答：

``` text
Decoder 怎样读取 P3/P4/P5
```

第 4 课 Attention：

``` text
选择信息 + 权重 + 聚合
```

本课把候选集合从：

``` text
全部 tokens
```

变成：

``` text
少量 learned sampling locations
```

第 5 课 Object Query：

``` text
Query 不只代表 prediction slot
还决定去哪里寻找视觉信息
```

第 6 课 Hungarian：

``` text
最终 matched loss
通过 backward
训练 Query representation、
sampling offsets、
attention weights 和 bbox refinement
```

------------------------------------------------------------------------

# 53. 与 `deimv2_dinov3_x` 的连接

DEIMv2 X 配置可确认：

``` text
DINOv3STAs:
name: dinov3_vits16plus
interaction_indexes: [5,8,11]
hidden_dim: 256

HybridEncoder:
in_channels: [256,256,256]
hidden_dim: 256

DEIMTransformer:
feat_channels: [256,256,256]
hidden_dim: 256
num_layers: 6
dim_feedforward: 2048
```

虽然 X backbone 是 DINOv3 + STA，而不是 HGNetv2，但最终 detector decoder
仍然接收多尺度 feature levels。

所以 Deformable Attention 是理解：

``` text
deim_dfine_x
```

与：

``` text
deimv2_dinov3_x
```

decoder 路线的共同基础。

------------------------------------------------------------------------

# 54. 总流程图

``` text
P3 80×80
P4 40×40
P5 20×20
    │
    ▼
Flatten + Concat
[B,8400,C]
    │
    ├──────────── spatial_shapes
    │
Object Query [B,Nq,C]
    │
    ├─→ Linear → Sampling Offsets
    │
    ├─→ Linear → Attention Weights
    │
    └─→ Reference Points / Boxes
                │
                ▼
        Sampling Locations
                │
                ▼
      Split Multi-Scale Value
                │
                ▼
        Bilinear Sampling
          (`grid_sample`)
                │
                ▼
       Sampled Feature Vectors
                │
                ▼
         Weighted Sum
                │
                ▼
           [B,Nq,C]
```

Decoder 更大的循环：

``` text
Current Query
+
Current Reference Box
↓
Deformable Cross-Attention
↓
Better Visual Evidence
↓
BBox Refinement
↓
Better Reference Box
↓
Next Decoder Layer
```

------------------------------------------------------------------------

# 55. 必须掌握的 12 个结论

1.  640 输入的 80×80、40×40、20×20 共 8400 个 positions。

2.  Deformable Attention 不 dense 地访问全部位置，而是学习少量 sampling
    locations。

3.  Reference Point / Box 提供当前空间参考。

4.  Sampling Offset 是相对位移。

5.  Sampling Location 是 reference 与适当归一化/缩放后的 offset 的组合。

6.  Attention Weight 决定 sampled features 的融合比例。

7.  `spatial_shapes` 保存各 feature level 的 `(H,W)`。

8.  `level_start_index` 描述各 level 在 concat sequence 的起点；DEIM
    core 也可能直接按 level sizes split。

9.  连续坐标通过 bilinear interpolation / `grid_sample` 取值。

10. DEIM 实现支持每个 level 不同采样点数，例如 `[3,6,3]`。

11. 多头意味着每个 head 可以学习不同 sampling pattern。

12. 最终输出仍恢复为 `[B,Nq,C]`。

------------------------------------------------------------------------

# 56. Shape Cheat Sheet

符号：

``` text
B    batch
Nq   query count
C    hidden dim
h    heads
d    C/h
L    levels
P_l  points at level l
Nv   total memory positions
```

Tensor：

``` text
Query:
[B,Nq,C]

Memory:
[B,Nv,C]

Spatial shapes:
[L,2]

Reference:
[B,Nq,L,2]
or
[B,Nq,L,4]

Offsets:
[B,Nq,h,sum(P_l),2]

Weights:
[B,Nq,h,sum(P_l)]

Level-l sampled:
[B*h,d,Nq,P_l]

All sampled:
[B*h,d,Nq,sum(P_l)]

Output:
[B,Nq,C]
```

640 example：

``` text
spatial_shapes:
[[80,80],
 [40,40],
 [20,20]]

level sizes:
[6400,1600,400]

level_start_index:
[0,6400,8000]

total:
8400
```

若：

``` text
P=[3,6,3]
h=8
```

每 Query：

``` text
12 points/head
96 head-specific sampling points
```

------------------------------------------------------------------------

# 57. 自测

1.  为什么 `[B,8400,C]` 本身不足以完成二维多尺度采样？

2.  `spatial_shapes` 保存什么？

3.  为什么 start indices 是 `[0,6400,8000]`？

4.  Reference Point 与 Sampling Offset 有什么区别？

5.  为什么 `dx` 要除以 `W`，`dy` 要除以 `H`？

6.  为什么 sampling location 可以是小数？

7.  为什么不直接 round？

8.  Bilinear interpolation 使用几个邻居？

9.  为什么 `[0,1]` sampling location 会转换到 `[-1,1]`？

10. `[3,6,3]` 每 head 共多少 points？

11. 8 heads 时每 Query 有多少 head-specific points？

12. 为什么不能声称整模型快 700 倍？

13. Query Self-Attention 与 Deformable Cross-Attention 分别连接谁？

14. Reference Box 的 `w,h` 为什么能帮助决定采样范围？

15. 为什么模块输出还能保持 `[B,Nq,C]`？

------------------------------------------------------------------------

# 58. 作业 A：320 输入

输入：

``` text
320×320
strides=[8,16,32]
```

自己计算：

``` text
spatial_shapes
level sizes
total positions
level_start_index
```

并比较 640 与 320 的 dense memory positions。

------------------------------------------------------------------------

# 59. 作业 B：Sampling Point 数量

分别计算：

``` text
[4,4,4]
[3,6,3]
[2,4,2]
[8,8,8]
```

对应：

``` text
sum(P_l)
8×sum(P_l)
300×8×sum(P_l)
```

解释增加 sampling points 的潜在收益与计算代价。

------------------------------------------------------------------------

# 60. 作业 C：可视化 Sampling Locations

设置：

``` text
reference=(0.5,0.5)
```

随机产生：

``` text
8 heads
每 head 4 offsets
```

画出：

``` text
reference
sampling points
```

观察不同 heads 的空间 pattern。

------------------------------------------------------------------------

# 61. 作业 D：验证坐标梯度

运行 `grid_sample` gradient 实验，把 grid 改成：

``` text
[0.0,0.0]
[0.2,0.2]
[0.8,-0.5]
```

打印：

``` text
grid.grad
```

解释 learned sampling offsets 为什么能通过 detection loss 更新。

------------------------------------------------------------------------

# 62. 作业 E：升级 Tiny Sampler

把 single-level 版本升级到：

``` text
P3 [B,C,80,80]
P4 [B,C,40,40]
P5 [B,C,20,20]
```

每层先使用：

``` text
4 points
```

最终输出：

``` text
[B,Nq,C]
```

不要复制官方实现。

------------------------------------------------------------------------

# 63. 作业 F：手工 Shape Trace

给：

``` text
B=4
Nq=300
C=384
h=8
L=3
P=[3,6,3]
```

手算：

``` text
d
offset shape
weight shape
每 level feature input
每 level grid
每 level sampled output
concat shape
final output
```

先手算，再用 PyTorch 验证。

------------------------------------------------------------------------

# 64. 以后读 `MSDeformableAttention` 固定问 12 个问题

``` text
1. query shape？
2. value shape？
3. hidden dim？
4. heads？
5. head dim？
6. levels？
7. 每 level points？
8. reference 是 2D 还是 4D？
9. offset 如何 normalize / scale？
10. sampling location shape？
11. sampled values 如何融合？
12. output 如何恢复 [B,Nq,C]？
```

能独立回答这 12 个问题，这个模块就不再是黑盒。

------------------------------------------------------------------------

# 65. 下一课预告

## 第 8 课：真正进入 `deim_dfine_x`------配置文件怎样"变成"模型？

我们会追：

``` text
train.py
↓
YAMLConfig
↓
__include__
↓
registry / register
↓
create
↓
__inject__
↓
DEIM
↓
HGNetv2
↓
HybridEncoder
↓
DFINETransformer
```

核心问题：

> YAML 中的一个名字，怎样最终实例化成真正的 `nn.Module`？

并亲手写一个最小：

``` text
Registry
+
Factory
+
Dependency Injection
```

实验。

------------------------------------------------------------------------

# 附录 A：与官方源码的对应

本课核对的 DEIM/D-FINE 公开实现中，`MSDeformableAttention`
的核心成员包括：

``` text
sampling_offsets
attention_weights
num_points_list
offset_scale
```

forward 核心输入包括：

``` text
query
reference_points
value
value_spatial_shapes
```

Query 预测 offsets/weights，构造 sampling locations；attention core 把各
level value 恢复为二维 feature maps，并通过
`torch.nn.functional.grid_sample` 进行 bilinear sampling，最后按 weights
加权求和。

源码存在不同 core/layout 版本，因此不要背死 `value`
的单一内存排列；始终跟随：

``` text
reshape
permute
split
```

重新推导。

------------------------------------------------------------------------

# 附录 B：DEIMv2 X 配置核对

`deimv2_dinov3_x_coco.yml` 当前公开配置可确认：

``` text
DINOv3STAs:
name: dinov3_vits16plus
interaction_indexes: [5,8,11]
hidden_dim: 256

HybridEncoder:
in_channels: [256,256,256]
hidden_dim: 256

DEIMTransformer:
feat_channels: [256,256,256]
hidden_dim: 256
num_layers: 6
dim_feedforward: 2048
```

本课使用 `[3,6,3]` 主要解释 DEIM decoder 家族"每个 level 可用不同 point
数"的真实机制。具体某个模型最终 `num_points` 值，后续必须以其 YAML
include/override 合并后的实际配置为准。

------------------------------------------------------------------------

# 附录 C：一手资料

1.  DEIMv2 官方仓库\
    https://github.com/Intellindust-AI-Lab/DEIMv2

2.  DEIMv2 DINOv3-X 配置\
    https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/configs/deimv2/deimv2_dinov3_x_coco.yml

3.  DEIM 官方仓库\
    https://github.com/Intellindust-AI-Lab/DEIM

4.  DEIM `MSDeformableAttention` / D-FINE Decoder\
    https://github.com/Intellindust-AI-Lab/DEIM/blob/main/engine/deim/dfine_decoder.py

5.  DEIM Deformable Attention Core\
    https://github.com/Intellindust-AI-Lab/DEIM/blob/main/engine/deim/utils.py

6.  Deformable DETR\
    https://arxiv.org/abs/2010.04159

7.  PyTorch `grid_sample`\
    https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html

------------------------------------------------------------------------

# 最终心智模型

以后看到：

``` python
cross_attn(
    query,
    reference_points,
    value,
    spatial_shapes
)
```

脑中展开：

``` text
Query [B,Nq,C]
        │
        ├───────────────┐
        │               │
        ▼               ▼
Linear(offset)      Linear(weight)
        │               │
        ▼               ▼
[B,Nq,h,Psum,2]  [B,Nq,h,Psum]
        │               │
        │            softmax
        │               │
Reference Point         │
        │               │
        ▼               │
Sampling Locations      │
        │               │
        ▼               │
Multi-Scale Features    │
        │               │
        ▼               │
Bilinear Sampling       │
        │               │
        └───────┬───────┘
                ▼
          Weighted Sum
                │
                ▼
           [B,Nq,C]
```

再把它放回 decoder：

``` text
当前 Query + Reference Box
↓
Deformable Cross-Attention
↓
更相关的视觉证据
↓
BBox Refinement
↓
更准确的 Reference Box
↓
下一层继续采样
```

这就是从基础 DETR 进入 D-FINE / DEIM / DEIMv2 Decoder
源码最关键的一座桥。
