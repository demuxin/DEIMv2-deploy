# 第 5 课：DETR 到底改变了什么------Object Query

> **课程定位**：DEIMv2 从零到源码 · 第 5 / 24 课\
> **本课目标**：真正理解 DETR 为什么是目标检测范式上的重要变化；彻底区分
> image feature/token、object query、anchor、proposal、prediction
> slot；理解 set prediction、Transformer decoder、query
> self-attention、query-image cross-attention，以及为什么最终可以得到
> `[B,Nq,K] + [B,Nq,4]`。\
> **重点连接**：DETR → Deformable DETR → D-FINE → DEIM / DEIMv2。\
> **本课最重要的问题**：**300 个 Object Queries 到底是什么？**

------------------------------------------------------------------------

# 0. 前 4 课终于在这里汇合

目前我们已经掌握：

``` text
第 1 课
Image → Backbone → Encoder → Decoder → Boxes

第 2 课
Tensor / Shape

第 3 课
CNN / Multi-scale Features

第 4 课
Transformer / QKV / Attention
```

现在可以第一次真正理解：

``` text
DETR
```

因为 DETR 的核心，就是把：

``` text
目标检测
```

重新写成：

``` text
Transformer + Set Prediction
```

------------------------------------------------------------------------

# 1. 目标检测到底要输出什么？

输入：

``` text
一张图片
```

我们希望输出：

``` text
object 1:
class = person
box   = ...

object 2:
class = car
box   = ...

object 3:
class = dog
box   = ...
```

也就是说：

\[ Image `\rightarrow`{=tex} { (c_i,b_i) }\_{i=1}\^{M} \]

其中：

``` text
M
```

不是固定的。

有的图片：

``` text
0 个目标
```

有的：

``` text
3 个
```

有的：

``` text
几十个
```

所以目标检测的输出天然是一个：

> **可变长度的目标集合。**

------------------------------------------------------------------------

# 2. 为什么叫 Set Prediction？

假设 GT 有三个目标：

``` text
person
car
dog
```

集合：

``` text
{person, car, dog}
```

和：

``` text
{dog, person, car}
```

从检测任务角度是同一个结果。

因为目标没有天然顺序。

数学上：

\[ {y_1,y_2,y_3} \]

不应该因为排列：

\[ (y_3,y_1,y_2) \]

就变成另一张图的答案。

所以 DETR 的一个核心思想：

> 直接预测一个 **unordered set of objects**。

------------------------------------------------------------------------

# 3. DETR 之前，检测器通常怎样思考？

这里先用高度简化的历史直觉。

很多经典检测器依赖：

``` text
dense locations
anchors
region proposals
hand-designed assignment rules
NMS
```

例如可以想象：

``` text
Feature Map
80×80

每个位置
放多个 anchors

每个 anchor
预测：
class
box offset
```

如果：

``` text
80×80
```

每个位置 3 anchors：

\[ 80`\times80`{=tex}`\times3`{=tex} = 19,200 \]

个候选。

然后还需要从大量重叠候选中筛掉重复框。

------------------------------------------------------------------------

# 4. 什么是 Anchor？

Anchor 可以理解成：

> 在特定空间位置预先定义的一组参考框。

例如某个 feature location 上：

``` text
small square
wide rectangle
tall rectangle
```

模型不是从完全空白开始预测 box，而是预测：

``` text
anchor → target box
```

的 offset/adjustment。

Anchor-based detector 的设计里通常要考虑：

``` text
anchor size
aspect ratio
feature level
positive/negative assignment
IoU threshold
```

------------------------------------------------------------------------

# 5. 为什么会需要 NMS？

假设图里只有一辆车。

Dense detector 可能预测：

``` text
box A score 0.95
box B score 0.93
box C score 0.90
box D score 0.88
```

它们都围着同一辆车。

我们不希望输出：

``` text
同一辆车 4 次
```

于是传统后处理常使用：

``` text
NMS
Non-Maximum Suppression
```

根据：

``` text
score
IoU overlap
```

保留一个，抑制重复框。

------------------------------------------------------------------------

# 6. DETR 的核心野心

DETR 希望把很多人工设计的检测组件简化掉。

核心思想可以概括为：

``` text
不要先生成成千上万个 dense candidates
再靠规则去重。

而是：

直接准备有限数量的 prediction slots，
让模型学会每个 slot 最终负责一个目标或 no-object。
```

这些 prediction slots 的核心表示，就是：

``` text
Object Queries
```

------------------------------------------------------------------------

# 7. Object Query 最先应该怎样理解？

先不要把它理解成框。

最安全的初始理解：

> **Object Query 是 Transformer Decoder 中用于产生目标预测的一组查询表示
> / prediction slots。**

例如：

``` text
Nq = 300
C  = 256
```

则：

``` text
query embeddings:
[300,256]
```

对 batch 扩展后：

``` text
[B,300,256]
```

这意味着：

``` text
每张图片有 300 个 prediction slots
```

每个 slot 最终可以预测：

``` text
一个类别
+
一个 bounding box
```

或者：

``` text
no-object / background-like empty prediction
```

具体类别表示方式会随 DETR 家族实现而变化。

------------------------------------------------------------------------

# 8. Object Query 不是"第 1 个目标、第 2 个目标......"

这是非常重要的误区。

Query 17 并不固定表示：

``` text
第 17 个 GT
```

Query 5 也不固定表示：

``` text
person
```

它们是可学习/构造的预测槽位。

训练中的 matching 会决定：

> 当前这张图片里，哪些 prediction slots 与哪些 GT 建立监督关系。

下一课会完整学习：

``` text
Hungarian Matching
```

------------------------------------------------------------------------

# 9. Object Query 也不等于 Anchor

经典 anchor：

``` text
有明确空间位置
有预定义 box size/aspect ratio
通常密集铺在 feature map 上
```

原始 DETR 的 learned object query：

``` text
不是传统意义上预定义的 anchor box
不是每个 feature cell 上固定摆一个矩形
```

它首先是一组：

``` text
learnable embeddings / decoder queries
```

但是必须注意：

> 后来的 DETR 变体（如 Deformable DETR、DINO、D-FINE 等）会引入
> reference points、two-stage proposal/query selection、anchor-like
> parameterization 等机制。

因此不能把：

``` text
“Object Query 永远完全没有空间参考”
```

当成所有 DETR 家族模型的永久结论。

本课先理解原始范式，后面再看 DEIM/D-FINE 如何升级。

------------------------------------------------------------------------

# 10. 300 个 Queries 为什么不等于 300 个物体？

假设：

``` text
Nq = 300
```

图里只有：

``` text
7 个 GT objects
```

那么训练匹配后，典型 one-to-one 情况下：

``` text
最多 7 个 prediction slots
与 7 个 GT 一一匹配
```

其余大量 slots：

``` text
没有对应 GT
```

它们会接受 empty/no-object 相关监督，具体损失定义依模型而异。

所以：

``` text
300 queries
≠
300 objects
```

而是：

``` text
最多提供 300 个预测槽位
```

------------------------------------------------------------------------

# 11. 为什么要比真实目标数多很多？

因为推理前不知道图片有多少目标。

如果只准备：

``` text
10 queries
```

但图片有：

``` text
25 objects
```

显然无法做到每个目标一个独立 slot。

所以选择一个较大的：

``` text
Nq
```

作为最大预测容量。

例如常见：

``` text
100
300
```

具体值取决于模型设计和数据集。

DEIM/D-FINE 配置中也会出现：

``` text
num_queries
```

这一关键超参数。

------------------------------------------------------------------------

# 12. Object Query 与 Image Token 是完全不同的东西

这是本课最重要的 shape 区分之一。

假设检测器有三尺度 feature：

``` text
80×80
40×40
20×20
```

flatten：

``` text
6400
1600
400
```

总 image positions：

\[ 6400+1600+400=8400 \]

所以 image memory 可以是：

``` text
[B,8400,C]
```

Object Queries：

``` text
[B,300,C]
```

这两个维度：

``` text
8400
300
```

含义完全不同。

------------------------------------------------------------------------

# 13. 8400 是什么？

``` text
8400
```

来自图片空间：

``` text
6400 + 1600 + 400
```

代表：

> 多尺度图像特征中的空间位置 / feature tokens。

它和输入分辨率、feature strides 密切相关。

------------------------------------------------------------------------

# 14. 300 是什么？

``` text
300
```

是：

> 模型设置的 object query / prediction slot 数量。

它不是：

``` text
图片像素数量
feature map spatial size
GT 数量
```

而是 decoder 的预测容量超参数。

------------------------------------------------------------------------

# 15. 一张最重要的图

``` text
IMAGE SIDE

Image
[B,3,640,640]
      │
      ▼
Backbone / Encoder
      │
      ▼
Image Memory
[B,8400,256]
      │
      │ K / V
      │
      │
      ▼
Cross-Attention
      ▲
      │
      │ Q
      │
Object Queries
[B,300,256]

      │
      ▼
Decoder Output
[B,300,256]

      ├──────────────┐
      ▼              ▼
Class Head        Box Head
[B,300,K]         [B,300,4]
```

这张图请反复看。

------------------------------------------------------------------------

# 16. Object Query 最开始里面有什么？

原始 DETR 使用：

``` text
learned object query embeddings
```

可以想象：

``` python
query_embed = nn.Embedding(
    num_queries,
    hidden_dim
)
```

例如：

``` text
[300,256]
```

这些参数：

``` text
训练前随机初始化
训练中通过反向传播学习
```

然后供每张图片使用。

------------------------------------------------------------------------

# 17. Query 为什么能对不同图片预测不同东西？

这是初学者经常困惑的问题。

如果：

``` text
query embedding
```

对所有图片都共享，

为什么 query 7 在图片 A 预测 person，在图片 B 又能预测 car？

因为最终 decoder query feature 不只取决于初始 query embedding。

它还通过：

``` text
cross-attention
```

读取当前图片的 image features。

所以：

``` text
共享 query prior
+
当前图片 memory
→
当前图片特定的 decoder feature
```

最终预测自然可以不同。

------------------------------------------------------------------------

# 18. Query Self-Attention 做什么？

假设：

``` text
queries:
[B,300,256]
```

Self-Attention：

``` text
Q/K/V
都来自 queries
```

attention matrix：

``` text
[B,h,300,300]
```

它让：

``` text
query 1
query 2
...
query 300
```

互相交换信息。

直觉上可以理解为：

> prediction slots 不需要彼此完全独立，它们可以协调自己的表示。

这与 set prediction / 去重复目标预测的学习过程相关，但不要简单说：

``` text
self-attention = NMS
```

二者不是同一算法。

------------------------------------------------------------------------

# 19. Query-Image Cross-Attention 做什么？

现在：

``` text
queries:
[B,300,C]
```

image memory：

``` text
[B,8400,C]
```

Cross-Attention：

``` text
Q
来自 object queries

K/V
来自 image memory
```

所以：

``` text
Q:
[B,h,300,d]

K/V:
[B,h,8400,d]
```

full cross-attention matrix：

``` text
[B,h,300,8400]
```

意义：

> 每个 object query 都从图像特征中读取与自己相关的信息。

------------------------------------------------------------------------

# 20. 一个 Query 怎样"找到物体"？

不要把它想成：

``` text
query 在图片上移动一个框
```

更好的理解：

``` text
query representation
↓
与 image features 交互
↓
聚合相关视觉信息
↓
decoder feature
↓
classification head + box head
↓
类别与位置
```

在原始 DETR 中，box 是 decoder feature 经过 prediction head
后预测出来的。

后来的 DETR 变体会让 query 与 reference point / proposal 更紧密地结合。

------------------------------------------------------------------------

# 21. Decoder 一层的简化结构

典型概念：

``` text
Queries
[B,Nq,C]
    │
    ▼
Query Self-Attention
    │
    ▼
[B,Nq,C]
    │
    ▼
Cross-Attention
with Image Memory
    │
    ▼
[B,Nq,C]
    │
    ▼
FFN
    │
    ▼
[B,Nq,C]
```

配合：

``` text
Residual
LayerNorm
```

构成一个 decoder layer。

多层堆叠：

``` text
Layer 1
↓
Layer 2
↓
...
↓
Layer L
```

------------------------------------------------------------------------

# 22. 为什么 Decoder 输出仍然是 300 个 Query Features？

输入：

``` text
[B,300,C]
```

decoder layer 一般保持 query sequence length：

``` text
300
```

所以最终：

``` text
[B,300,C]
```

每一个 query feature 都对应一个预测槽位。

然后：

``` text
class head
```

和：

``` text
box head
```

分别把它转换为目标类别与 box 参数。

------------------------------------------------------------------------

# 23. Classification Head

假设：

``` text
decoder output:
[B,300,256]
```

一个 Linear：

``` python
class_head = nn.Linear(
    256,
    K
)
```

得到：

``` text
[B,300,K]
```

如果 K 表示模型定义的类别输出维度，那么：

``` text
每个 query
→ 一组类别 logits
```

具体是否显式包含 no-object 类、使用 sigmoid 还是 softmax，取决于 DETR
变体与 criterion 实现。

后面进入 DEIM/D-FINE criterion 时会按真实源码讲。

------------------------------------------------------------------------

# 24. Box Head

简化教学版：

``` python
box_head = nn.Sequential(
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 4),
)
```

输入：

``` text
[B,300,256]
```

输出：

``` text
[B,300,4]
```

四个数可以表示：

``` text
cx
cy
w
h
```

通常是归一化坐标。

------------------------------------------------------------------------

# 25. 但 D-FINE 的 Box Regression 更复杂

这里必须提前提醒。

为了理解 DETR，我们现在用：

``` text
query feature
→ MLP
→ 4 numbers
```

这个经典简化模型。

但后面 D-FINE 会使用：

``` text
fine-grained distribution refinement
reg_max
reg_scale
distance/distribution representation
```

所以不能把本课教学用的：

``` text
Linear/MLP → 4
```

直接当成 `DFINETransformer` 的完整真实 box regression 实现。

第 12 课会专门拆。

------------------------------------------------------------------------

# 26. 一个最小 DETR Shape Pipeline

假设：

``` text
B = 2
C = 256
Nimage = 8400
Nq = 300
K = 80
```

Image Memory：

``` text
[2,8400,256]
```

Object Queries：

``` text
[2,300,256]
```

Decoder：

``` text
[2,300,256]
```

Class：

``` text
[2,300,80]
```

Boxes：

``` text
[2,300,4]
```

这就是 DETR 类模型最核心的 shape 地图。

------------------------------------------------------------------------

# 27. 为什么预测 300 个框不会输出 300 个最终目标？

模型会为 queries 产生：

``` text
classification confidence
```

推理阶段：

``` text
低置信度/empty-like predictions
```

不会全部作为最终检测结果保留。

所以：

``` text
300
```

是 prediction slots 数，

不是最终 detection count。

后处理具体规则依模型实现。

------------------------------------------------------------------------

# 28. DETR 为什么需要 Matching？

现在出现一个关键问题。

GT：

``` text
3 objects
```

预测：

``` text
300 slots
```

到底：

``` text
哪个 query 学 GT 1？
哪个 query 学 GT 2？
哪个 query 学 GT 3？
```

不能固定写：

``` text
query 0 → GT 0
query 1 → GT 1
```

因为 GT 本身是无序集合。

所以需要：

``` text
matching
```

在预测集合和 GT 集合之间建立一一对应关系。

------------------------------------------------------------------------

# 29. 一个极小 Matching 例子

GT：

``` text
GT A = person
GT B = car
```

Predictions：

``` text
Query 0
Query 1
Query 2
Query 3
```

我们计算每个：

``` text
Query ↔ GT
```

的 matching cost。

可能：

``` text
           GT A    GT B

Query 0     0.2     4.0
Query 1     1.5     0.3
Query 2     0.8     2.1
Query 3     3.0     1.2
```

理想一一匹配可能：

``` text
Query 0 ↔ GT A
Query 1 ↔ GT B
```

剩余：

``` text
Query 2
Query 3
```

没有 GT 匹配。

下一课会严格推导：

``` text
Hungarian algorithm
classification cost
L1 box cost
GIoU cost
```

------------------------------------------------------------------------

# 30. One-to-One 到底是什么意思？

训练 matching 中：

``` text
一个 GT
最多匹配一个 prediction
```

同时：

``` text
一个 prediction
最多匹配一个 GT
```

所以是：

``` text
one-to-one
```

而不是：

``` text
一个 GT
对应几十个 positive anchors
```

这正是 DETR set prediction 的核心特征之一。

------------------------------------------------------------------------

# 31. One-to-One 为什么能减少重复预测？

训练目标在推动：

``` text
每个真实目标
由一个预测槽位负责
```

其他 slots 不应该都重复预测同一个 GT。

因此模型被训练成：

``` text
集合级别的一一预测
```

这也是 DETR 原始设计能够避免传统 NMS 依赖的重要原因。

但注意：

> "不使用 NMS"来自整体 set prediction + matching + loss
> 设计，不应该粗暴归因于某一层 self-attention。

------------------------------------------------------------------------

# 32. DETR 的真正改变不只是"用了 Transformer"

这是本课最重要的结论之一。

如果只是：

``` text
CNN
+
Transformer
```

并不足以概括 DETR 的意义。

DETR 更重要的变化是：

``` text
把目标检测定义成
direct set prediction
```

配合：

``` text
fixed prediction slots / object queries
+
bipartite matching
+
set-based loss
```

把传统检测中的很多人工设计组件重新组织了。

------------------------------------------------------------------------

# 33. 原始 DETR 的整体结构

概念图：

``` text
Image
[B,3,H,W]
    │
    ▼
CNN Backbone
    │
    ▼
Feature Map
[B,C,H',W']
    │
    ▼
Flatten + Position Encoding
    │
    ▼
Transformer Encoder
    │
    ▼
Image Memory
[B,N,C]
    │
    │
    │ K,V
    ▼
Transformer Decoder
    ▲
    │ Q
Object Queries
[B,Nq,C]
    │
    ▼
Decoder Features
[B,Nq,C]
    │
    ├──────────────┐
    ▼              ▼
Class            Box
[B,Nq,K]         [B,Nq,4]
```

------------------------------------------------------------------------

# 34. 原始 DETR 为什么训练慢？

原始 DETR 是非常重要的范式，但也有实际问题，例如：

``` text
训练收敛较慢
高分辨率特征计算成本高
小目标表现曾是明显挑战
```

后续很多工作都在改进：

``` text
Deformable DETR
DAB-DETR
DN-DETR
DINO
RT-DETR
D-FINE
DEIM
...
```

DEIMv2 就位于这条长期演化路线的后面。

------------------------------------------------------------------------

# 35. Deformable DETR 改了什么方向？

最核心的一个方向：

> 不再让每个 query 对所有 image positions 做昂贵的 full
> cross-attention。

而是围绕：

``` text
reference points
```

在多尺度 feature maps 上只采样少量位置。

于是：

``` text
full:
300 × 8400

变成概念上的：
300 × levels × points
```

当然真实实现还包括：

``` text
sampling offsets
attention weights
spatial shapes
level start index
```

第 7 课完整拆。

------------------------------------------------------------------------

# 36. Reference Point 与 Object Query 的关系

后来的 DETR 变体中，query 不再只是抽象 embedding。

它通常会和：

``` text
reference point
reference box
proposal
```

结合。

例如可以把 query 想成：

``` text
content query
+
spatial reference
```

content：

``` text
我在找什么/当前目标表示
```

reference：

``` text
我当前重点看图像哪里
```

这比原始 DETR 的纯 learned query 更容易进行空间定位。

------------------------------------------------------------------------

# 37. Two-Stage Query Selection 是什么方向？

一些 DETR 变体会从 encoder/image features 中先产生候选：

``` text
encoder proposals
```

然后选出高质量位置作为 decoder queries/reference。

于是 query 不再完全是：

``` text
与当前图片无关的固定 learned embeddings
```

而是：

``` text
从当前图片 feature 中初始化/选择
```

D-FINE/DEIM 的 decoder 源码中就会看到：

``` text
encoder output
top-k selection
query/reference initialization
```

所以本课必须建立：

``` text
原始 DETR Object Query
```

这个基线，后面才看得懂它如何被改造。

------------------------------------------------------------------------

# 38. 为什么叫 Query？

回到 Attention：

``` text
Query
Key
Value
```

Object Query 进入 decoder 后，本质上就是：

> 一组用于向 image memory 发起信息检索的 query representations。

Cross-Attention：

``` text
Object Query
→ Q

Image Features
→ K/V
```

因此名字不是随便起的。

------------------------------------------------------------------------

# 39. 一个很有用但不完全严格的类比

可以把 300 个 queries 想成：

``` text
300 个侦探
```

每个侦探问图片：

> "有没有某个目标值得我负责？"

它通过 cross-attention 读取图像证据。

最后：

``` text
一些侦探
找到真实目标

大量侦探
没有找到有效目标
```

但这个类比有边界：

``` text
query 并不是有固定身份的人
query 之间也会 self-attention
query 的空间参考会被后续 DETR 变体显著增强
```

所以只用它建立直觉，不要当数学定义。

------------------------------------------------------------------------

# 40. Object Query 是否有固定类别？

没有。

不能说：

``` text
query 0 专门检测 person
query 1 专门检测 car
```

同一个 query slot 在不同图片上可以输出不同类别。

它学到的是：

``` text
适合参与集合预测的 query representation / slot behavior
```

而不是人工绑定类别。

------------------------------------------------------------------------

# 41. Object Query 是否有固定位置？

原始 learned object query：

``` text
不是传统 anchor 那种固定图片坐标框
```

但后续模型可能显式引入：

``` text
reference point
reference box
anchor-like query position
```

因此回答这个问题必须说明：

``` text
具体取决于 DETR 变体
```

对于 DEIM/D-FINE，后面必须看真实 query/reference 初始化逻辑。

------------------------------------------------------------------------

# 42. 为什么 300 Queries 可以检测 7 个目标？

因为训练 matching：

``` text
7 GT
```

只会选择一部分 query predictions 与它们对应。

例如：

``` text
Query 17 → person
Query 83 → car
Query 101 → dog
...
```

剩余：

``` text
293 queries
```

不承担这 7 个 GT 的一一匹配正样本角色。

这直接导致一个后面非常重要的问题：

> **正监督很稀疏。**

------------------------------------------------------------------------

# 43. 这里第一次看到 DEIM 的核心问题

假设：

``` text
300 queries
```

图里：

``` text
7 GT
```

普通 one-to-one：

``` text
最多只有 7 个 matched positive queries
```

比例：

\[ 7/300 `\approx
2.33`{=tex}% \]

也就是说：

``` text
绝大多数 prediction slots
不是 matched positive
```

这就是 DEIM 关注的关键训练效率问题之一：

> **O2O matching 带来的稀疏正监督。**

------------------------------------------------------------------------

# 44. Dense O2O 为什么会出现？

先只建立动机。

DEIM 会问：

> 能不能保持 one-to-one set prediction
> 的优点，同时让训练时获得更多有效正监督？

这就是：

``` text
Dense O2O
```

方向。

后面第 15 课会专门拆：

``` text
普通 O2O
vs
Dense O2O
```

并计算正样本密度与训练信号变化。

现在只要记住：

``` text
DETR Object Query
→ O2O Matching
→ Positive supervision sparse
→ DEIM Dense O2O
```

这条因果链。

------------------------------------------------------------------------

# 45. DEIM 的 MAL 又从哪里来？

当训练中引入更密集的 O2O supervision 后：

``` text
不同匹配的质量并不完全相同
```

有些：

``` text
定位质量高
```

有些：

``` text
匹配较弱
```

于是需要更好地考虑：

``` text
match quality
```

这就连接到：

``` text
MAL
Matchability-Aware Loss
```

后面第 16 课展开。

所以 DEIM 的训练创新不是凭空出现的。

它直接建立在：

``` text
DETR set prediction
+
O2O query matching
```

之上。

------------------------------------------------------------------------

# 46. 一个最小 PyTorch Object Query 实验

``` python
import torch
import torch.nn as nn

B = 2
Nq = 300
C = 256

query_embed = nn.Embedding(
    Nq,
    C
)

print(
    query_embed.weight.shape
)
```

输出：

``` text
[300,256]
```

扩展到 batch：

``` python
queries = (
    query_embed.weight
    .unsqueeze(0)
    .expand(B, -1, -1)
)

print(
    queries.shape
)
```

输出：

``` text
[2,300,256]
```

这就是最基础的 learned query tensor。

------------------------------------------------------------------------

# 47. 实验 2：构造 Image Memory

``` python
import torch

B = 2
C = 256

p3 = torch.randn(
    B, C, 80, 80
)

p4 = torch.randn(
    B, C, 40, 40
)

p5 = torch.randn(
    B, C, 20, 20
)

features = []

for x in [p3, p4, p5]:

    x = x.flatten(2)
    x = x.transpose(1, 2)

    features.append(x)

memory = torch.cat(
    features,
    dim=1
)

print(memory.shape)
```

输出：

``` text
[2,8400,256]
```

现在同时拥有：

``` text
queries:
[2,300,256]

memory:
[2,8400,256]
```

这就是 decoder cross-attention 最重要的两种输入。

------------------------------------------------------------------------

# 48. 实验 3：用 PyTorch MultiheadAttention 模拟 Cross-Attention

``` python
import torch
import torch.nn as nn

B = 2
Nq = 300
Nm = 840
C = 256

queries = torch.randn(
    B, Nq, C
)

memory = torch.randn(
    B, Nm, C
)

cross_attn = nn.MultiheadAttention(
    embed_dim=C,
    num_heads=8,
    batch_first=True
)

out, weights = cross_attn(
    query=queries,
    key=memory,
    value=memory
)

print(
    "queries:",
    queries.shape
)

print(
    "memory:",
    memory.shape
)

print(
    "output:",
    out.shape
)

print(
    "weights:",
    weights.shape
)
```

这里故意把 memory token 数缩到：

``` text
840
```

以便普通设备实验。

预期：

``` text
queries:
[2,300,256]

memory:
[2,840,256]

output:
[2,300,256]
```

注意：

> Cross-Attention 输出 sequence length 跟 Query 一样，是 300，而不是
> 840。

因为：

``` text
每个 query
从 memory 读取信息
```

最终仍然得到一个更新后的 query feature。

------------------------------------------------------------------------

# 49. 为什么 Cross-Attention 输出是 Nq，而不是 Nk？

数学：

``` text
Q:
[Nq,d]

K:
[Nk,d]

V:
[Nk,dv]
```

先：

\[ QK\^T \]

shape：

\[ \[Nq,d\] `\times
[d,Nk]`{=tex}= \[Nq,Nk\] \]

再：

\[ \[Nq,Nk\] `\times
[Nk,d_v]`{=tex}= \[Nq,d_v\] \]

所以输出数量由：

``` text
Query 数 Nq
```

决定。

这是理解 decoder 的关键。

------------------------------------------------------------------------

# 50. 实验 4：一个极简 DETR Head

``` python
import torch
import torch.nn as nn

B = 2
Nq = 300
C = 256
K = 80

decoder_output = torch.randn(
    B, Nq, C
)

class_head = nn.Linear(
    C,
    K
)

box_head = nn.Sequential(
    nn.Linear(C, C),
    nn.ReLU(),
    nn.Linear(C, 4)
)

class_logits = class_head(
    decoder_output
)

boxes = torch.sigmoid(
    box_head(decoder_output)
)

print(
    class_logits.shape
)

print(
    boxes.shape
)
```

输出：

``` text
[2,300,80]
[2,300,4]
```

这就是 DETR prediction slots 的最简形态。

------------------------------------------------------------------------

# 51. 实验 5：完整 Tiny DETR Decoder

``` python
import torch
import torch.nn as nn

class TinyDETR(nn.Module):

    def __init__(
        self,
        hidden_dim=256,
        num_queries=300,
        num_classes=80,
        num_heads=8
    ):
        super().__init__()

        self.query_embed = nn.Embedding(
            num_queries,
            hidden_dim
        )

        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True
        )

        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim * 4
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim * 4,
                hidden_dim
            )
        )

        self.class_head = nn.Linear(
            hidden_dim,
            num_classes
        )

        self.box_head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                4
            )
        )

    def forward(self, memory):

        B = memory.shape[0]

        q = (
            self.query_embed.weight
            .unsqueeze(0)
            .expand(B, -1, -1)
        )

        q2, _ = self.self_attn(
            q, q, q
        )

        q = q + q2

        q2, _ = self.cross_attn(
            query=q,
            key=memory,
            value=memory
        )

        q = q + q2

        q = q + self.ffn(q)

        logits = self.class_head(q)

        boxes = torch.sigmoid(
            self.box_head(q)
        )

        return logits, boxes
```

测试时为了省内存：

``` python
memory = torch.randn(
    2, 840, 256
)

model = TinyDETR()

logits, boxes = model(memory)

print(logits.shape)
print(boxes.shape)
```

预期：

``` text
[2,300,80]
[2,300,4]
```

这不是 DEIMv2 的真实 decoder。

它只是把：

``` text
Object Query
Self-Attention
Cross-Attention
FFN
Class Head
Box Head
```

最小化地串起来。

------------------------------------------------------------------------

# 52. 为什么 TinyDETR 还不是真实 DETR？

我们故意省略了：

``` text
position encoding
LayerNorm
dropout
multiple decoder layers
encoder
auxiliary losses
no-object handling
Hungarian matching
real box loss
multi-scale features
reference points
deformable attention
denoising
iterative box refinement
```

以及 DEIM/D-FINE 特有机制。

目的只有一个：

> 先让 Object Query 的数据流完全透明。

------------------------------------------------------------------------

# 53. Decoder Layer 的 Shape Trace

假设：

``` text
queries:
[2,300,256]

memory:
[2,8400,256]
```

Query Self-Attention：

``` text
[2,300,256]
→
[2,300,256]
```

Cross-Attention：

``` text
Q:
[2,300,256]

K/V:
[2,8400,256]

→
[2,300,256]
```

FFN：

``` text
[2,300,256]
→
[2,300,1024]
→
[2,300,256]
```

最终：

``` text
[2,300,256]
```

------------------------------------------------------------------------

# 54. 六层 Decoder 呢？

如果 6 层：

``` text
Layer 1:
[2,300,256]

Layer 2:
[2,300,256]

...

Layer 6:
[2,300,256]
```

如果保存所有层：

``` text
[6,2,300,256]
```

这就连接到第 2 课的：

``` text
stack
```

以及后面：

``` text
auxiliary loss
intermediate supervision
iterative refinement
```

------------------------------------------------------------------------

# 55. 为什么中间层也可以预测？

因为每一层都有：

``` text
[B,Nq,C]
```

query features。

所以可以在每层后都接：

``` text
class head
box head
```

产生中间预测。

训练时对中间层也监督：

``` text
auxiliary losses
```

可以帮助优化深层 decoder。

D-FINE 还会进一步做更复杂的 localization supervision / distillation。

------------------------------------------------------------------------

# 56. Query 是不是"空槽位"？

可以把它叫 prediction slot，但不要理解成：

``` text
完全没有 feature 的空数组
```

原始 DETR 的 query embedding 是：

``` text
可学习参数
```

后续变体可能：

``` text
由 encoder feature 初始化
带 reference point
带 content embedding
带 positional component
```

所以更准确：

> Query 是一个将被 decoder 更新、最终承载目标预测信息的表示。

------------------------------------------------------------------------

# 57. Query 的身份会不会固定？

训练后可能观察到某些 query 有统计偏好，但不能把设计原则描述为：

``` text
query 7 永远负责左上角
query 9 永远负责 car
```

DETR 的 matching 是按每张图片动态建立的。

因此：

``` text
query index
```

不是 GT 的固定语义编号。

------------------------------------------------------------------------

# 58. 为什么 Object Queries 是 Set Prediction 的自然接口？

因为：

``` text
固定 Nq 个 slots
```

可以输出一个固定大小 prediction set：

\[ `\hat{Y}`{=tex} = {
`\hat `{=tex}y_1,`\ldots`{=tex},`\hat `{=tex}y\_{N_q} } \]

GT：

\[ Y = { y_1,`\ldots`{=tex},y_M } \]

其中：

\[ M`\le `{=tex}N_q \]

Matching：

``` text
从 Nq 个 predictions 中
选出 M 个
与 M 个 GT 一一对应
```

剩余 predictions：

``` text
作为 unmatched / empty predictions 处理
```

这就把可变长度 GT 集合变成了可训练的固定槽位输出。

------------------------------------------------------------------------

# 59. DETR 为什么叫 End-to-End Detection？

这里的"end-to-end"重点是：

``` text
从图像
直接学习 set prediction
```

并通过 matching/loss 训练最终目标集合。

相较传统 pipeline，它减少了对：

``` text
anchor engineering
proposal heuristics
NMS-style duplicate removal
```

等手工组件的依赖。

但真实现代检测系统仍然会有：

``` text
data preprocessing
postprocessing
score filtering
deployment conversion
```

所以"end-to-end"不要理解成：

> 完全没有任何前后处理代码。

------------------------------------------------------------------------

# 60. 从 DETR 到 D-FINE/DEIM：什么保留下来了？

虽然后面模型复杂很多，但几个核心思想仍然保留：

``` text
fixed number of decoder queries
set prediction
one-to-one matching
decoder query refinement
class/localization prediction
```

变化主要发生在：

``` text
query initialization
reference points
attention efficiency
box representation
denoising training
matching/loss
positive supervision density
backbone/encoder
```

所以 DETR 是理解 DEIMv2 的祖先框架。

------------------------------------------------------------------------

# 61. D-FINE 会怎样改"Box"？

原始 DETR：

``` text
query feature
→ direct box prediction
```

D-FINE：

``` text
更细粒度的 distribution-based localization/refinement
```

所以：

``` text
Object Query
```

仍然是预测主体，

但：

``` text
“如何从 query 表示得到更准确的 box”
```

被显著升级。

------------------------------------------------------------------------

# 62. DEIM 会怎样改"Training"？

DEIM 重点不是推翻 object query。

它看到的问题是：

``` text
O2O set prediction
训练时正监督太稀疏
```

于是通过：

``` text
Dense O2O
MAL
```

等机制提升训练效率/质量。

所以：

``` text
DETR
提供 set prediction 骨架

D-FINE
增强 localization

DEIM
增强训练范式
```

这是一个非常有用的高层理解。

------------------------------------------------------------------------

# 63. DEIMv2 又进一步改了什么方向？

DEIMv2 进一步把：

``` text
更强视觉 Backbone
DINOv3
+
STA
+
升级的训练机制
```

带入这条实时端到端检测路线。

但是最深层仍然能看到 DETR 的遗产：

``` text
queries
decoder
matching
set prediction
```

所以本课不是历史背景，而是后面所有源码的语义基础。

------------------------------------------------------------------------

# 64. 本课最重要的 5 个 Shape

请记住：

``` text
Image:
[B,3,H,W]

Image Memory:
[B,Nimage,C]

Object Queries:
[B,Nq,C]

Class Predictions:
[B,Nq,K]

Box Predictions:
[B,Nq,4]
```

教学例：

``` text
[B,3,640,640]

[B,8400,256]

[B,300,256]

[B,300,80]

[B,300,4]
```

------------------------------------------------------------------------

# 65. 本课最重要的 5 个概念区别

``` text
Image Token
≠
Object Query

Object Query
≠
Anchor

Object Query
≠
Ground Truth

Number of Queries
≠
Number of Objects

Self-Attention
≠
Cross-Attention
```

如果这五个区别彻底清楚，本课已经成功一大半。

------------------------------------------------------------------------

# 66. Image Token vs Object Query

Image Token：

``` text
来源：
图片 feature

数量：
与 feature spatial size 相关

作用：
提供视觉信息
```

Object Query：

``` text
来源：
learned / selected / initialized query mechanism

数量：
模型超参数

作用：
作为 prediction slots，从图片特征中查询并形成目标预测
```

------------------------------------------------------------------------

# 67. Anchor vs Object Query

传统 Anchor：

``` text
预定义空间框
密集铺设
有尺度/比例
常与 feature location 绑定
```

原始 DETR Object Query：

``` text
learned embedding
不等同于预定义矩形
数量固定
通过 decoder 与图像交互
```

后续 DETR：

``` text
query + reference point/box
```

使二者在"空间先验"方面出现某些概念接近之处，但实现和训练范式仍不同。

------------------------------------------------------------------------

# 68. Proposal vs Query

Proposal：

``` text
通常表示从当前图片中产生的候选区域/候选位置
```

Query：

``` text
表示 decoder 的查询/预测槽位
```

Two-stage DETR 变体可能：

``` text
从 encoder proposals
选择 top-k
用于初始化 decoder queries
```

所以 proposal 和 query 可以发生联系，但不能直接画等号。

------------------------------------------------------------------------

# 69. Reference Point vs Query

Reference Point：

``` text
空间参考
```

Query：

``` text
内容/预测表示
```

在 Deformable Attention 中，query 可以围绕 reference point：

``` text
预测 sampling offsets
```

从多尺度 feature 中采样。

所以：

``` text
query
告诉模型“我在找什么”

reference
帮助模型“我主要去哪里找”
```

这是很有用的直觉。

------------------------------------------------------------------------

# 70. Query 与 Prediction 的关系

一个 query feature：

``` text
[B,i,C]
```

经过：

``` text
class head
box/localization head
```

得到：

``` text
prediction i
```

所以可以写：

\[ q_i `\rightarrow`{=tex} (`\hat `{=tex}c_i,`\hat `{=tex}b_i) \]

Nq 个 queries：

\[ {q_i}*{i=1}\^{N_q} `\rightarrow`{=tex} {
(`\hat `{=tex}c_i,`\hat `{=tex}b_i) }*{i=1}\^{N_q} \]

这就是 prediction set。

------------------------------------------------------------------------

# 71. 一个最小数学模型

设 encoder memory：

\[ M`\in`{=tex}`\mathbb{R}`{=tex}\^{N_m`\times `{=tex}C} \]

Object Queries：

\[ Q_0`\in`{=tex}`\mathbb{R}`{=tex}\^{N_q`\times `{=tex}C} \]

Decoder：

\[ Q_L = Decoder( Q_0,M ) \]

其中：

\[ Q_L`\in`{=tex}`\mathbb{R}`{=tex}\^{N_q`\times `{=tex}C} \]

分类：

\[ `\hat `{=tex}C = f\_{cls}(Q_L) \]

框：

\[ `\hat `{=tex}B = f\_{box}(Q_L) \]

最终：

\[ `\hat `{=tex}C `\in`{=tex} `\mathbb{R}`{=tex}\^{N_q`\times `{=tex}K}
\]

\[ `\hat `{=tex}B `\in`{=tex} `\mathbb{R}`{=tex}\^{N_q`\times4`{=tex}}
\]

------------------------------------------------------------------------

# 72. 本课自测

尽量不用代码。

## Q1

为什么目标检测输出天然更像"集合"而不是普通有序序列？

------------------------------------------------------------------------

## Q2

Object Query 最安全的定义是什么？

------------------------------------------------------------------------

## Q3

为什么：

``` text
300 queries
```

不意味着图片必须有 300 个物体？

------------------------------------------------------------------------

## Q4

``` text
8400
```

与：

``` text
300
```

分别是什么？

------------------------------------------------------------------------

## Q5

Object Query 为什么不是传统 Anchor？

------------------------------------------------------------------------

## Q6

为什么同一个 learned query 在不同图片可以预测不同类别/位置？

------------------------------------------------------------------------

## Q7

Query Self-Attention 的 Q/K/V 来自哪里？

------------------------------------------------------------------------

## Q8

Cross-Attention 的 Q 与 K/V 分别来自哪里？

------------------------------------------------------------------------

## Q9

如果：

``` text
Q = [B,8,300,32]
K = [B,8,8400,32]
```

attention matrix 是什么 shape？

------------------------------------------------------------------------

## Q10

为什么 cross-attention 输出仍然有 300 个 query features？

------------------------------------------------------------------------

## Q11

为什么 DETR 需要 matching？

------------------------------------------------------------------------

## Q12

如果一张图有 7 个 GT、300 queries，普通 O2O 最多有多少 matched positive
predictions？

------------------------------------------------------------------------

## Q13

这个数字为什么会成为 DEIM 的研究动机之一？

------------------------------------------------------------------------

## Q14

原始 DETR 与后续 Deformable/DINO/D-FINE 的 query
在空间参考上有什么演化？

------------------------------------------------------------------------

## Q15

为什么不能说：

``` text
DETR 的核心只是“把 CNN 换成 Transformer”
```

？

------------------------------------------------------------------------

# 73. 课后作业 A：画出 Query / Memory 数据流

不看本课，自己画：

``` text
Image
↓
Backbone
↓
Memory

Object Queries
↓
Decoder

Class
Box
```

必须标出：

``` text
[B,3,640,640]
[B,8400,256]
[B,300,256]
[B,300,K]
[B,300,4]
```

然后给每个维度写语义。

------------------------------------------------------------------------

# 74. 课后作业 B：实现 Tiny DETR

要求自己实现：

``` text
nn.Embedding
query self-attention
cross-attention
FFN
class head
box head
```

不能复制本课代码。

要求每一步打印：

``` text
input/output shape
```

最后必须得到：

``` text
class:
[B,300,K]

box:
[B,300,4]
```

------------------------------------------------------------------------

# 75. 课后作业 C：比较 Anchor 与 Query

自己写一个表格，至少比较：

``` text
数量来源
空间位置
是否预定义 box
是否密集铺设
训练 assignment
重复预测处理
与 image feature 的关系
```

注意：

> 比较"原始 DETR learned query"与"经典 anchor-based
> detector"，不要把所有后续 DETR 变体混成一个模型。

------------------------------------------------------------------------

# 76. 课后作业 D：Positive Supervision Density

假设：

``` text
num_queries = 300
```

分别计算图片有：

``` text
1
5
10
30
100
```

个 GT 时，普通 O2O 最大 matched-positive 比例：

\[ `\frac{M}{300}`{=tex} \]

是多少。

思考：

``` text
目标较少的图片
```

为什么 query-level 正监督尤其稀疏？

这会直接连接第 15 课 Dense O2O。

------------------------------------------------------------------------

# 77. 课后作业 E：自己解释 Cross-Attention

禁止使用：

> "query 去关注 image。"

这种一句话回答。

必须完整解释：

``` text
Object query 输入 shape
Image memory 输入 shape
Q projection
K/V projection
QKᵀ shape
softmax 维度
weighted V
输出 shape
为什么输出 token 数等于 query 数
```

能完成这项作业，才算真正理解 DETR Decoder 的核心。

------------------------------------------------------------------------

# 78. 本课 Cheat Sheet

``` text
Object Detection
=
predict an unordered set of objects

DETR
=
direct set prediction
+
Transformer
+
Object Queries
+
bipartite matching

Object Query
=
decoder query representation /
prediction slot

Image Memory
=
visual feature tokens

Nimage
=
由 feature spatial resolution 决定

Nq
=
query/prediction slot 超参数

Query Self-Attention
=
queries ↔ queries

Cross-Attention
=
queries ↔ image memory

Decoder Output
=
[B,Nq,C]

Class
=
[B,Nq,K]

Box
=
[B,Nq,4]
```

------------------------------------------------------------------------

# 79. 从 DETR 到 DEIMv2 的演化主线

请把这条线留在脑中：

``` text
DETR
│
├─ Set Prediction
├─ Object Queries
├─ O2O Matching
└─ Transformer Decoder
        │
        ▼
Deformable DETR
│
├─ Multi-scale
├─ Reference Points
└─ Sparse Sampling
        │
        ▼
DINO / related DETR improvements
│
├─ Better query initialization
├─ Denoising
└─ Better training
        │
        ▼
D-FINE
│
├─ Fine-grained box regression
└─ Localization refinement/distillation
        │
        ▼
DEIM
│
├─ Dense O2O
└─ MAL
        │
        ▼
DEIMv2
│
├─ DINOv3
├─ STA
└─ upgraded training/architecture
```

这不是完整论文谱系，而是为了理解 DEIMv2 源码而保留的主干逻辑。

------------------------------------------------------------------------

# 80. 下一课预告

## 第 6 课：Hungarian Matching 与 One-to-One------300 个 Query 到底谁负责哪个 GT？

下一课会真正解决：

``` text
GT:
3 个

Predictions:
300 个

到底怎么配对？
```

我们会从：

``` text
3 predictions
2 GT
```

的极小例子开始。

逐步构造：

``` text
classification cost
L1 box cost
GIoU cost
```

形成 cost matrix：

\[ C\_{ij} \]

然后理解 Hungarian algorithm 在优化：

\[ `\min`{=tex}\_{`\sigma`{=tex}} `\sum`{=tex}*i C*{i,`\sigma`{=tex}(i)}
\]

我们还会亲手用：

``` python
scipy.optimize.linear_sum_assignment
```

做 matching。

最后连接：

``` text
One-to-One
→ sparse positives
→ DEIM Dense O2O
```

这会是理解 DEIM 训练创新最关键的基础课之一。

------------------------------------------------------------------------

# 附录 A：与 DEIMv2 后续源码阅读的对应关系

后续读 decoder 时，重点寻找这些概念：

``` text
num_queries
query_embed
tgt
memory
reference_points
enc_outputs
topk
query selection
decoder layers
class_embed
bbox_embed
dn queries
aux_outputs
```

但请注意：

> DEIMv2 / D-FINE 的真实 decoder 已经比原始 DETR 复杂很多。

所以看到源码中 query 不是简单：

``` python
nn.Embedding(300,256)
```

时不要惊讶。

我们的学习方法是：

``` text
先掌握原始 DETR 的最小语义
↓
再逐项解释后续模型为什么修改它
```

------------------------------------------------------------------------

# 附录 B：参考资料

优先参考原始论文与官方项目：

1.  DETR 原论文：End-to-End Object Detection with Transformers\
    https://arxiv.org/abs/2005.12872

2.  DETR 官方实现\
    https://github.com/facebookresearch/detr

3.  Deformable DETR 原论文\
    https://arxiv.org/abs/2010.04159

4.  DEIMv2 官方仓库\
    https://github.com/Intellindust-AI-Lab/DEIMv2

5.  DEIM 官方仓库\
    https://github.com/Intellindust-AI-Lab/DEIM

> 本课的重点是建立 DETR Object Query / Set Prediction 的基础模型。DEIMv2
> 中 query selection、reference points、denoising、D-FINE localization
> 等具体实现，以后会单独对照官方源码，不把原始 DETR 的简化结构误认为
> DEIMv2 的最终实现。

------------------------------------------------------------------------

**第 5 课结束。**

真正达标的标准是：当别人问你

> "DETR 为什么要 300 个 Object Queries？"

你不再回答：

> "因为要检测 300 个框。"

而是能够解释：

``` text
Object Query 是固定数量的 decoder prediction slots；
它与 image tokens 不同，也不等同于传统 anchors；
decoder 通过 self-attention 和 cross-attention 更新这些 queries；
每个 query 最终形成类别与定位预测；
训练时通过 bipartite one-to-one matching，把可变长度 GT 集合分配给固定数量 prediction slots；
而这种 O2O 训练的稀疏正监督，又正好连接到 DEIM 的 Dense O2O 动机。
```
