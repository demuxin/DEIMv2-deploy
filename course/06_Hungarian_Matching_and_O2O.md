# DEIMv2 第 6 课：DETR 的训练核心——Hungarian Matching 与 O2O

> **本课目标**：从零理解 DETR 为什么需要 Hungarian Matching（匈牙利匹配）、什么是 One-to-One（O2O）匹配，并能从数学公式、Tensor shape 和 PyTorch 小实验三个层面解释：**模型产生 300 个预测后，训练时到底怎样决定“哪个 query 负责哪个 GT”？**
>
> 本课同时为后面的 **DEIM Dense O2O、MAL、DEIMv2 upgraded Dense O2O** 打基础。现在先把“标准 O2O 匹配”彻底吃透。

---

## 0. 先把本课放回整个课程地图

前 5 课我们已经建立了这些基础：

```text
第 1 课：图片 → backbone → encoder → decoder → bbox
第 2 课：Tensor / shape
第 3 课：CNN / stride / multi-scale feature
第 4 课：Transformer / Q K V / Attention
第 5 课：Object Query / Set Prediction
第 6 课：Hungarian Matching / O2O   ← 现在
第 7 课：Deformable Attention
...
第 15 课：DEIM Dense O2O
第 16 课：MAL
第 17 课：DEIMv2 Dense O2O v2
```

第 5 课结束时，我们已经知道 DETR 的 decoder 可以输出很多 object queries。

例如：

```text
num_queries = 300

decoder output:
[B, 300, C]
```

再经过分类头和框回归头：

```text
pred_logits : [B, 300, num_classes]
pred_boxes  : [B, 300, 4]
```

现在出现一个非常关键的问题。

假设一张图片里只有 3 个真实目标：

```text
GT 0 = person
GT 1 = dog
GT 2 = bicycle
```

但模型产生了：

```text
query 0
query 1
query 2
...
query 299
```

那么：

> **到底应该让哪个 query 学 person？哪个 query 学 dog？哪个 query 学 bicycle？**

这就是本课的核心。

---

# 1. 为什么普通的“按位置监督”在 DETR 中行不通？

先看传统 dense detector 的直觉。

例如某些 anchor/grid-based detector 会把预测与空间位置强绑定：

```text
feature map cell
       ↓
anchor / point
       ↓
负责附近的 GT
```

因此训练规则可以大量依赖空间位置。

但 DETR 的 object query 不具有这种固定语义：

```text
query 17 并不永远负责左上角
query 81 并不永远负责 person
query 203 也不是“大目标 query”
```

它们是一个 **set**。

因此：

```text
GT 顺序没有意义
prediction 顺序也没有固定意义
```

假设 GT 是：

```text
[person, dog]
```

下面两组预测从集合角度完全一样：

```text
预测 A:
query 0 → person
query 1 → dog
```

```text
预测 B:
query 0 → dog
query 1 → person
```

如果检测结果都正确，我们不应该因为“顺序不同”惩罚模型。

这叫：

## permutation invariance

即：

> **集合中元素的排列顺序不应该影响目标检测结果的正确性。**

于是 DETR 需要先寻找一个最佳的：

```text
prediction ↔ ground truth
```

对应关系。

---

# 2. DETR 的核心训练流程

一张图可以抽象成：

```text
                 Model

image
  │
  ▼
300 object queries
  │
  ├── q0 → class + bbox
  ├── q1 → class + bbox
  ├── q2 → class + bbox
  │
  ...
  └── q299 → class + bbox

                  │
                  ▼

         Hungarian Matcher

                  │

       GT0 ←→ query 37
       GT1 ←→ query 91
       GT2 ←→ query 208

                  │
                  ▼

              Criterion

        classification loss
        bbox L1 loss
        GIoU loss
        ...
```

请特别记住：

> **Matcher 不直接训练网络。**

Matcher 的任务是：

```text
决定谁和谁配对
```

Criterion/Loss 的任务才是：

```text
根据配对结果计算损失
```

这是源码阅读时非常容易混淆的一点。

---

# 3. 什么叫 One-to-One Matching？

假设有 3 个 GT：

```text
GT A
GT B
GT C
```

有 5 个预测：

```text
q0
q1
q2
q3
q4
```

One-to-One（O2O）要求：

```text
一个 GT
只能匹配一个 prediction

一个 prediction
也只能匹配一个 GT
```

例如：

```text
GT A ← q3
GT B ← q0
GT C ← q4
```

剩下：

```text
q1
q2
```

没有匹配 GT。

它们会作为 background / no-object 方向受到分类监督（具体分类损失形式依实现而异）。

因此：

```text
5 predictions
3 GT

→ 3 matched predictions
→ 2 unmatched predictions
```

如果是 DETR 常见的：

```text
300 queries
7 GT
```

那么标准 O2O 下：

```text
最多 7 个 query 能成为 matched positive
```

这句话非常重要。

后面 DEIM 的核心动机之一正是：

> **标准 O2O 的正监督太稀疏。**

---

# 4. 为什么不能简单找“离 GT 最近的 query”？

因为“一个预测好不好”不只取决于 bbox。

假设 GT 是：

```text
class = dog
bbox  = [cx, cy, w, h]
```

有两个预测：

```text
query A:
class confidence dog = 0.95
bbox IoU = 0.60
```

```text
query B:
class confidence dog = 0.20
bbox IoU = 0.90
```

哪个更应该负责这个 GT？

没有办法只看一个维度回答。

因此 DETR matcher 会构造综合匹配代价：

\[
C_{ij}
=
\lambda_{\mathrm{cls}}C^{\mathrm{cls}}_{ij}
+
\lambda_{\mathrm{bbox}}C^{\mathrm{bbox}}_{ij}
+
\lambda_{\mathrm{giou}}C^{\mathrm{giou}}_{ij}
\]

其中：

```text
i = prediction/query index
j = GT index
```

所以：

```text
C[i, j]
```

表示：

> **让第 i 个 query 负责第 j 个 GT，需要付出多大的匹配代价？**

越小越好。

---

# 5. Cost Matrix：整个 Hungarian Matching 的核心数据结构

假设：

```text
5 predictions
3 GT
```

那么 cost matrix 的 shape 是：

```text
[5, 3]
```

例如：

\[
C=
\begin{bmatrix}
0.8 & 4.2 & 2.1\\
3.5 & 0.4 & 1.9\\
1.1 & 2.2 & 3.8\\
0.3 & 4.0 & 2.7\\
2.9 & 1.7 & 0.2
\end{bmatrix}
\]

可以读成：

```text
             GT0    GT1    GT2

query0       0.8    4.2    2.1
query1       3.5    0.4    1.9
query2       1.1    2.2    3.8
query3       0.3    4.0    2.7
query4       2.9    1.7    0.2
```

肉眼容易看出：

```text
query3 ↔ GT0
query1 ↔ GT1
query4 ↔ GT2
```

总 cost：

\[
0.3+0.4+0.2=0.9
\]

Hungarian algorithm 做的就是：

> **在满足 one-to-one 约束的前提下，寻找总匹配 cost 最小的组合。**

---

# 6. 为什么不能对每个 GT 独立 argmin？

这是一个非常重要的陷阱。

假设：

\[
C=
\begin{bmatrix}
0.1 & 0.2\\
0.3 & 4.0\\
4.0 & 0.4
\end{bmatrix}
\]

如果分别对每一列找最小值：

```text
GT0 → query0
GT1 → query0
```

问题来了：

```text
query0 同时匹配两个 GT
```

这违反 O2O。

Hungarian matching 会寻找满足：

```text
一个 query 最多使用一次
一个 GT 最多使用一次
```

的全局组合。

例如可能得到：

```text
GT0 → query0
GT1 → query2
```

总代价：

\[
0.1+0.4=0.5
\]

所以 Hungarian algorithm 不是简单的：

```python
cost.argmin(dim=0)
```

而是一个：

## Linear Sum Assignment Problem

线性和分配问题（LSAP）。

数学形式可以写成：

\[
\sigma^*
=
\arg\min_{\sigma}
\sum_{j=1}^{M}
C_{\sigma(j),j}
\]

其中：

```text
M = GT 数量
σ(j) = 分配给 GT j 的 prediction index
```

并要求不同 GT 使用不同 prediction。

---

# 7. Matching Cost 的第一部分：Classification Cost

现在开始逐项拆：

\[
C_{ij}
=
\lambda_{cls}C_{cls}
+
\lambda_{bbox}C_{bbox}
+
\lambda_{giou}C_{giou}
\]

先看 classification。

假设 GT j 的类别是：

```text
dog
```

那么如果 query i 很确信它是 dog：

```text
P(dog) = 0.95
```

应该：

```text
cost 小
```

如果：

```text
P(dog) = 0.02
```

应该：

```text
cost 大
```

最简单的思想可以写成：

\[
C^{cls}_{ij}=-p_i(c_j)
\]

或者使用与 focal loss 思路相关的分类匹配代价。

实际 DETR 系列实现可能采用不同的分类 cost 形式，因此阅读源码时不要把：

```text
classification matching cost
```

机械理解成固定的 `-probability`。

DEIM / D-FINE 系列 matcher 使用 focal-style classification cost 参数，例如配置中可见：

```yaml
matcher:
  type: HungarianMatcher
  weight_dict:
    cost_class: 2
    cost_bbox: 5
    cost_giou: 2
  alpha: 0.25
  gamma: 2.0
```

也就是说，在我们重点研究的模型路线中，matcher 不只是看一个朴素 softmax probability。

这一点后面读真实 matcher 源码时会继续展开。

---

# 8. Matching Cost 的第二部分：L1 Bounding Box Cost

假设 bbox 使用：

```text
[cx, cy, w, h]
```

GT：

\[
b^{gt}=[0.50,0.50,0.20,0.40]
\]

prediction：

\[
b^{pred}=[0.52,0.47,0.25,0.38]
\]

L1 distance：

\[
C_{bbox}
=
|0.52-0.50|
+
|0.47-0.50|
+
|0.25-0.20|
+
|0.38-0.40|
\]

得到：

\[
C_{bbox}
=
0.02+0.03+0.05+0.02
=
0.12
\]

所以：

```text
box 越接近 GT
→ L1 distance 越小
→ matching cost 越小
```

PyTorch 可以直接写：

```python
torch.cdist(pred_boxes, target_boxes, p=1)
```

如果：

```text
pred_boxes   : [N, 4]
target_boxes : [M, 4]
```

那么：

```text
cost_bbox : [N, M]
```

例如：

```text
N = 300 queries
M = 7 GT
```

得到：

```text
cost_bbox.shape
= [300, 7]
```

注意：

> 这里不是得到 300 个值，也不是得到 7 个值，而是计算 **每一个 prediction 与每一个 GT 的两两距离**。

---

# 9. Matching Cost 的第三部分：GIoU Cost

只使用 L1 bbox distance 仍然不够。

例如两个框：

```text
中心位置接近
宽高数值也接近
```

不代表空间重叠关系一定最好。

因此 DETR matcher 通常还加入 IoU/GIoU。

## 9.1 IoU

\[
IoU(A,B)
=
\frac{|A\cap B|}{|A\cup B|}
\]

范围：

\[
0\le IoU\le1
\]

如果：

```text
IoU = 1
```

表示两个框完全一致。

---

## 9.2 GIoU

IoU 的一个问题是：

如果两个框完全不相交：

```text
IoU = 0
```

那么无论：

```text
离得很近
```

还是：

```text
离得非常远
```

IoU 都是 0。

GIoU 加入最小包围区域 \(C\)：

\[
GIoU
=
IoU
-
\frac{|C\setminus(A\cup B)|}{|C|}
\]

因此 GIoU 可以在不相交时继续反映两个框之间的空间关系。

范围：

\[
-1\le GIoU\le1
\]

matcher 希望：

```text
GIoU 越大越好
```

但 Hungarian solver 是：

```text
cost 越小越好
```

因此通常使用：

\[
C_{giou}=-GIoU
\]

---

# 10. 三个 Cost 怎样合并？

以 DEIM/D-FINE 常见配置为例：

```yaml
weight_dict:
  cost_class: 2
  cost_bbox: 5
  cost_giou: 2
```

那么：

\[
C
=
2C_{cls}
+
5C_{bbox}
+
2C_{giou}
\]

假设某个 prediction-GT pair：

```text
classification cost = -0.70
bbox cost           =  0.10
giou cost           = -0.60
```

那么：

\[
C
=
2(-0.70)+5(0.10)+2(-0.60)
\]

\[
=-1.4+0.5-1.2
\]

\[
=-2.1
\]

另一个 pair：

```text
classification cost = -0.30
bbox cost           = 0.25
giou cost           = -0.20
```

则：

\[
C
=
2(-0.30)+5(0.25)+2(-0.20)
\]

\[
=-0.6+1.25-0.4
=0.25
\]

Hungarian solver 更偏向：

```text
-2.1
```

因为 cost 更小。

注意：

> **cost 可以是负数。**

它只需要能用于比较候选匹配关系，不要求像“概率”那样限制在 0～1。

---

# 11. 一个完整的 shape 推导

现在使用后面经常出现的真实规模。

假设：

```text
batch size B = 2
num_queries N = 300
num_classes = 80
```

模型输出：

```text
pred_logits : [2, 300, 80]
pred_boxes  : [2, 300, 4]
```

两张图片分别有：

```text
image 0: 5 GT
image 1: 8 GT
```

targets 可以抽象成：

```python
targets = [
    {
        "labels": Tensor[5],
        "boxes":  Tensor[5, 4]
    },
    {
        "labels": Tensor[8],
        "boxes":  Tensor[8, 4]
    }
]
```

---

## 11.1 flatten predictions

很多 matcher 实现为了向量化，会先：

```python
out_prob = pred_logits.flatten(0, 1)
out_bbox = pred_boxes.flatten(0, 1)
```

于是：

```text
[2, 300, 80]
→ [600, 80]

[2, 300, 4]
→ [600, 4]
```

为什么是 600？

\[
B\times N
=
2\times300
=
600
\]

---

## 11.2 concatenate targets

两个 batch 中：

```text
5 + 8 = 13 GT
```

所以：

```text
target labels:
[13]

target boxes:
[13, 4]
```

---

## 11.3 两两计算 cost

分类：

```text
cost_class:
[600, 13]
```

bbox：

```text
cost_bbox:
[600, 13]
```

GIoU：

```text
cost_giou:
[600, 13]
```

合并：

```text
C:
[600, 13]
```

但这里要小心：

> image 0 的 query 不能匹配 image 1 的 GT。

因此实现随后需要恢复 batch 维度，并按照每张图 GT 数量切开。

概念上变成：

```text
image 0:
[300, 5]

image 1:
[300, 8]
```

然后分别运行 Hungarian solver。

最终可能得到：

```python
[
    (
        src_idx_image0,  # [5]
        tgt_idx_image0   # [5]
    ),
    (
        src_idx_image1,  # [8]
        tgt_idx_image1   # [8]
    )
]
```

例如：

```text
image 0:

src_idx = [12, 39, 81, 105, 240]
tgt_idx = [ 2,  0,  4,   1,   3]
```

它表示：

```text
query 12  ↔ GT 2
query 39  ↔ GT 0
query 81  ↔ GT 4
query 105 ↔ GT 1
query 240 ↔ GT 3
```

---

# 12. 为什么输出通常是 `(src_idx, tgt_idx)`？

你以后会在 criterion 中不断看到类似：

```python
indices = matcher(outputs, targets)
```

然后：

```python
for src, tgt in indices:
    ...
```

这里：

```text
src = source prediction index
tgt = target index
```

例如：

```python
src = tensor([3, 8, 15])
tgt = tensor([1, 0, 2])
```

就是：

```text
prediction 3  → GT 1
prediction 8  → GT 0
prediction 15 → GT 2
```

为什么 `tgt` 不一定：

```text
[0, 1, 2]
```

？

因为 Hungarian solver 返回的是最优 assignment 对应关系，不要求按 GT 原始顺序排列。

---

# 13. 匹配以后，Loss 才真正开始

这是本课最重要的区分之一。

```text
Hungarian matching
```

解决：

> 谁负责谁？

而：

```text
loss
```

解决：

> 匹配以后，模型应该受到多大的惩罚？

流程：

```text
predictions
targets
     │
     ▼
matcher
     │
     ▼
indices
     │
     ▼
criterion
     │
     ├── classification loss
     ├── bbox loss
     ├── giou loss
     └── ...
     │
     ▼
total loss
     │
     ▼
backward()
```

---

# 14. Matching Cost ≠ Training Loss

这是初学 DETR 时最容易犯的错误之一。

看到：

```yaml
matcher:
  weight_dict:
    cost_class: 2
    cost_bbox: 5
    cost_giou: 2
```

不要理解成：

```text
最终 loss 就一定是
2 * class_loss
+ 5 * bbox_loss
+ 2 * giou_loss
```

matcher cost 的作用是：

```text
选择 assignment
```

criterion loss weight 的作用是：

```text
决定梯度优化目标中不同 loss 的权重
```

二者可以数值相似，但逻辑角色完全不同。

例如：

```text
MATCHING PHASE

predictions
   ↓
cost_class
cost_bbox
cost_giou
   ↓
Hungarian
   ↓
indices
```

随后才：

```text
OPTIMIZATION PHASE

matched predictions
        +
targets
   ↓
loss_cls
loss_bbox
loss_giou
...
   ↓
weighted total loss
   ↓
backpropagation
```

请把这两阶段在脑中永久分开。

---

# 15. Hungarian Matching 本身需要梯度吗？

通常：

```text
不需要。
```

原因是 assignment 是离散选择：

```text
query 17 ↔ GT 2
```

这种组合优化结果不是普通连续可微运算。

训练过程更像：

```text
Step 1
根据当前 prediction 计算 matching cost

Step 2
求一个离散 assignment

Step 3
把这个 assignment 当作监督关系

Step 4
对真正的 loss 求梯度
```

所以不要期待：

```text
gradient 穿过 scipy Hungarian solver
```

模型参数的梯度来自：

```text
匹配之后计算的 differentiable losses
```

而不是 assignment solver 本身。

---

# 16. 为什么 O2O 能让 DETR 不依赖传统 NMS？

传统 dense detector 可能针对一个真实目标产生大量高分框：

```text
dog:

box A
box B
box C
box D
box E
```

因此推理时需要 NMS 去重。

而 DETR 的训练目标强调：

```text
一个 GT
↔
一个 prediction
```

也就是说训练会推动网络形成：

```text
one object
→ one responsible query
```

这使 set prediction detector 可以设计成 end-to-end prediction，而不必把 NMS 作为核心必需步骤。

当然，这并不意味着“只要写了 HungarianMatcher，网络就神奇地永远不会产生重复预测”。

真正起作用的是整套：

```text
set prediction objective
+
one-to-one assignment
+
classification of unmatched queries
+
decoder/query interaction
+
training dynamics
```

共同形成的行为。

---

# 17. O2O 的优点

标准 O2O 很漂亮。

## 17.1 assignment 明确

一个 GT：

```text
只选一个负责 prediction
```

不会出现大量 prediction 同时争抢同一 GT 的训练目标。

## 17.2 天然适合 set prediction

检测结果本身就是：

\[
\{\hat y_1,\hat y_2,\ldots,\hat y_N\}
\]

而不是必须绑定固定排列顺序。

## 17.3 可以支持端到端检测

避免依赖传统的手工 duplicate-removal pipeline。

---

# 18. 但 O2O 也有一个巨大问题：Sparse Supervision

这就是 DEIM 即将登场的原因。

假设：

```text
num_queries = 300
```

一张图只有：

```text
5 GT
```

那么标准 O2O：

```text
最多只有 5 个 matched positives
```

positive ratio：

\[
\frac{5}{300}
\approx1.67\%
\]

如果只有 2 个目标：

\[
\frac{2}{300}
\approx0.67\%
\]

也就是说：

```text
绝大多数 queries
并没有获得正目标的 box regression supervision
```

这会让 DETR 训练面临：

```text
positive supervision sparse
```

的问题。

注意这里不要误解：

> unmatched query 并不是“完全没有 loss”。

它们仍然可以接受背景/类别方向的监督。

真正稀疏的是：

```text
positive object supervision
```

尤其是：

```text
positive localization supervision
```

---

# 19. 一个非常重要的数字实验

假设训练 batch：

```text
B = 16
```

每张图平均：

```text
7 objects
```

那么标准 O2O 一次 batch 大约有：

\[
16\times7=112
\]

个 matched positives。

但 queries 总数：

\[
16\times300=4800
\]

positive proportion：

\[
\frac{112}{4800}
\approx2.33\%
\]

现在你应该开始理解 DEIM 的问题意识了：

```text
4800 query slots
```

中，真正对应 GT positive 的只有大约：

```text
112
```

当然，这只是用于建立直觉的简化统计；真实训练还存在 denoising queries、auxiliary decoder losses 等机制。

但 **O2O 本身的 positive assignment 稀疏性** 是确实存在的。

---

# 20. DEIM 的关键洞察将在后面出现

DEIM 的出发点可以先浓缩成一句话：

> **既然 O2O 很干净，但 supervision 太稀疏，能不能保留 O2O 的性质，同时让训练看到更多有效 targets？**

于是后面会出现：

```text
Standard O2O
     ↓
Sparse supervision

        vs

Dense O2O
     ↓
More targets / positives
     ↓
Denser supervision
```

但增加 positive matching 后又产生新的问题：

```text
不是所有 match 都一样可靠
```

于是进一步引出：

```text
MAL
Matchability-Aware Loss
```

所以你可以把后面的 DEIM 主线记成：

```text
问题 1：
O2O supervision 太稀疏
        ↓
Dense O2O

问题 2：
Dense 后出现不同质量的 match
        ↓
MAL
```

本课只负责把“问题 1 的起点”彻底建立起来。

---

# 21. 与 DEIMv2 配置真正对应起来

在 DEIMv2 的基础配置中，可以看到类似：

```yaml
DEIMCriterion:
  weight_dict:
    loss_mal: 1
    loss_bbox: 5
    loss_giou: 2
    loss_fgl: 0.15
    loss_ddf: 1.5

  matcher:
    type: HungarianMatcher

    weight_dict:
      cost_class: 2
      cost_bbox: 5
      cost_giou: 2

    alpha: 0.25
    gamma: 2.0
```

先只关注 matcher：

```text
HungarianMatcher
│
├── classification cost
├── bbox L1 cost
└── GIoU cost
```

也就是说，即使到了 DEIMv2：

> **Hungarian assignment 仍然是理解 O2O 训练的基础。**

DEIMv2 是在这个基础上进一步升级训练策略，而不是把“一对一分配”这个概念完全抛弃。

---

# 22. DEIMv2 的 `change_matcher` 是什么信号？

DEIMv2 配置还能看到：

```yaml
change_matcher: True
iou_order_alpha: 4.0
matcher_change_epoch: 45
```

不同具体模型的 epoch 设置可能不同。

现在暂时不要深挖实现，只建立一个认识：

```text
DEIMv2 并不把 matcher 当成永远固定不变的黑盒。
```

它会在训练过程中对 matching 策略进行进一步设计。

因此后面第 17 课研究 upgraded Dense O2O 时，我们会重新回来回答：

```text
为什么训练后期要改变 matching？
iou_order_alpha 是什么？
matcher_change_epoch 为什么存在？
它如何影响 query ↔ GT assignment？
```

本课先保证标准 Hungarian O2O 基础完全牢固。

---

# 23. 小型 PyTorch 实验 1：构造 L1 Cost Matrix

先从最容易理解的 bbox cost 开始。

```python
import torch

pred_boxes = torch.tensor([
    [0.50, 0.50, 0.20, 0.20],
    [0.20, 0.20, 0.10, 0.10],
    [0.80, 0.75, 0.15, 0.20],
    [0.48, 0.52, 0.22, 0.18],
    [0.78, 0.76, 0.14, 0.21],
])

target_boxes = torch.tensor([
    [0.50, 0.50, 0.20, 0.20],
    [0.80, 0.75, 0.15, 0.20],
    [0.20, 0.20, 0.10, 0.10],
])

cost_bbox = torch.cdist(
    pred_boxes,
    target_boxes,
    p=1
)

print(cost_bbox)
print(cost_bbox.shape)
```

shape：

```text
pred_boxes:
[5, 4]

target_boxes:
[3, 4]

cost_bbox:
[5, 3]
```

请自己解释：

```text
cost_bbox[i, j]
```

是什么。

答案：

> prediction i 与 GT j 的 bbox L1 distance。

---

# 24. 小型实验 2：真的运行 Hungarian Solver

SciPy 提供：

```python
linear_sum_assignment
```

实验：

```python
import torch
from scipy.optimize import linear_sum_assignment

C = torch.tensor([
    [0.8, 4.2, 2.1],
    [3.5, 0.4, 1.9],
    [1.1, 2.2, 3.8],
    [0.3, 4.0, 2.7],
    [2.9, 1.7, 0.2],
])

row_ind, col_ind = linear_sum_assignment(C.numpy())

print("prediction indices:", row_ind)
print("target indices:", col_ind)

print("matched costs:")
print(C[row_ind, col_ind])

print("total cost:")
print(C[row_ind, col_ind].sum())
```

预期对应关系接近：

```text
query3 ↔ GT0
query1 ↔ GT1
query4 ↔ GT2
```

你应该重点观察：

```text
row_ind
col_ind
```

而不是只看 total cost。

因为后面的 criterion 真正需要的是：

```text
哪些 prediction 被选中了？
它们对应哪些 GT？
```

---

# 25. 小型实验 3：证明 argmin 不等于 Hungarian

运行：

```python
import torch
from scipy.optimize import linear_sum_assignment

C = torch.tensor([
    [0.1, 0.2],
    [0.3, 4.0],
    [4.0, 0.4],
])

# 错误思路：每个 GT 独立找最小 prediction
independent = C.argmin(dim=0)

print("independent argmin:")
print(independent)

# 正确 O2O assignment
row_ind, col_ind = linear_sum_assignment(C.numpy())

print("Hungarian:")
print(row_ind)
print(col_ind)
```

独立 argmin 会得到：

```text
GT0 → query0
GT1 → query0
```

违反 O2O。

Hungarian 则会得到合法的全局 assignment。

---

# 26. 小型实验 4：自己组合三种 Cost

下面模拟：

```text
classification cost
bbox cost
giou cost
```

```python
import torch
from scipy.optimize import linear_sum_assignment

cost_class = torch.tensor([
    [-0.9, -0.1],
    [-0.4, -0.8],
    [-0.6, -0.5],
])

cost_bbox = torch.tensor([
    [0.10, 0.80],
    [0.70, 0.15],
    [0.25, 0.30],
])

cost_giou = torch.tensor([
    [-0.8, -0.1],
    [-0.2, -0.7],
    [-0.5, -0.4],
])

C = (
    2.0 * cost_class
    + 5.0 * cost_bbox
    + 2.0 * cost_giou
)

print("combined cost:")
print(C)

row_ind, col_ind = linear_sum_assignment(C.numpy())

print("matched prediction indices:", row_ind)
print("matched target indices:", col_ind)
```

然后修改：

```python
cost_bbox weight = 5
```

为：

```python
cost_bbox weight = 20
```

观察 assignment 是否变化。

这个实验会让你真正理解：

> matcher 权重会改变“谁负责谁”。

而不仅仅改变一个 loss 数值。

---

# 27. 小型实验 5：300 Queries 对 7 个 GT

现在模拟更接近 DETR 的规模：

```python
import torch

N = 300
M = 7

pred_boxes = torch.rand(N, 4)
target_boxes = torch.rand(M, 4)

cost_bbox = torch.cdist(
    pred_boxes,
    target_boxes,
    p=1
)

print(cost_bbox.shape)
```

输出：

```text
torch.Size([300, 7])
```

也就是：

\[
300\times7=2100
\]

个 candidate pair。

Hungarian 最终只会从这些 pair 中选择：

```text
7 pairs
```

因为只有：

```text
7 GT
```

这正是：

```text
many candidates
        ↓
global assignment
        ↓
few matched positives
```

---

# 28. 为什么不是让每个 GT 匹配多个 queries？

你现在可能自然产生一个问题：

> 如果 7 个 GT 只匹配 7 个 query 太稀疏，为什么不直接每个 GT 匹配 10 个 query？

这实际上会改变训练范式。

如果：

```text
GT A
↔ q1
↔ q5
↔ q17
↔ q81
...
```

那么就更接近 one-to-many supervision。

它可以提供更多 positives，但也会重新引入：

```text
多个 prediction 对应同一 object
```

的问题。

标准 DETR 的漂亮之处恰恰在于：

```text
one object
↔
one prediction
```

DEIM 的巧妙点并不是简单粗暴地说：

```text
“那就改成 one-to-many。”
```

而是想办法：

```text
保留 O2O 训练/推理范式的重要性质
+
提高训练中的有效监督密度
```

这就是为什么它叫：

# Dense O2O

而不是简单叫：

```text
One-to-Many DETR
```

后面第 15 课会专门拆。

---

# 29. Denoising Queries 会不会破坏这个理解？

D-FINE / DEIM 系列训练时还会看到：

```yaml
num_denoising: 100
label_noise_ratio: 0.5
box_noise_scale: 1.0
```

这意味着训练阶段除了普通 matching queries 外，还存在 denoising 训练机制。

因此真实训练时 decoder 中参与计算的 query 数量和监督路径可能比：

```text
300 normal queries
```

更复杂。

但是学习顺序必须分开：

```text
普通 matching queries
        ↓
Hungarian O2O
```

和：

```text
denoising queries
        ↓
已知 target 派生的额外训练信号
```

不是同一件事。

本课讨论的是：

> **普通 set-prediction 分支中 query 与 GT 的 O2O assignment。**

后面读 decoder 和 criterion 时，我们再把 denoising 分支接回来。

---

# 30. Auxiliary Decoder Loss 又会发生什么？

假设 decoder 有 6 层：

```text
layer 0
layer 1
layer 2
layer 3
layer 4
layer 5
```

很多 DETR 系列模型不仅监督最后一层，也监督中间层：

```text
layer 0 → auxiliary output
layer 1 → auxiliary output
...
layer 5 → final output
```

那么 matcher / indices / auxiliary loss 的关系也会进入源码。

初学阶段只需要记住：

```text
O2O matching
```

不是说整个网络一次 forward 永远只产生一个 loss。

可能存在：

```text
final output loss
+
auxiliary decoder losses
+
denoising losses
+
localization-related losses
```

但每一条 supervision branch 的“prediction 如何对应 target”仍然必须被明确处理。

---

# 31. 一个从模型输出到 backward 的完整伪代码

下面这段非常值得反复看。

```python
# 1. model forward
outputs = model(images)

# outputs
# pred_logits: [B, N, num_classes]
# pred_boxes : [B, N, 4]


# 2. matching
indices = matcher(outputs, targets)

# indices:
# [
#   (pred_indices_img0, target_indices_img0),
#   (pred_indices_img1, target_indices_img1),
#   ...
# ]


# 3. classification loss
loss_cls = compute_classification_loss(
    outputs,
    targets,
    indices
)


# 4. select matched boxes
matched_pred_boxes = ...
matched_target_boxes = ...


# 5. localization losses
loss_bbox = l1_loss(
    matched_pred_boxes,
    matched_target_boxes
)

loss_giou = giou_loss(
    matched_pred_boxes,
    matched_target_boxes
)


# 6. weighted total
loss = (
    w_cls * loss_cls
    + w_bbox * loss_bbox
    + w_giou * loss_giou
)


# 7. gradient
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

这就是 DETR 训练核心骨架。

后面的 DEIM、D-FINE、DEIMv2 虽然会加入大量机制，但你应该始终能看到这根主骨架。

---

# 32. 用一个具体例子完整走一次

假设：

```text
3 queries
2 GT
```

GT：

```text
GT0:
class = cat
box = [0.25, 0.40, 0.20, 0.30]

GT1:
class = dog
box = [0.75, 0.55, 0.25, 0.35]
```

预测：

```text
q0:
cat score = 0.90
dog score = 0.05
box ≈ GT0

q1:
cat score = 0.30
dog score = 0.60
box 在 GT0 和 GT1 中间

q2:
cat score = 0.05
dog score = 0.95
box ≈ GT1
```

构造出的 cost matrix 可能像：

\[
C=
\begin{bmatrix}
-2.4 & 3.1\\
1.0 & 0.8\\
3.4 & -2.7
\end{bmatrix}
\]

Hungarian：

```text
q0 ↔ GT0
q2 ↔ GT1
```

q1：

```text
unmatched
```

然后：

```text
q0:
classification → cat
bbox → GT0

q2:
classification → dog
bbox → GT1

q1:
classification → background / no-object direction
```

于是：

```text
q0
q2
```

获得 positive localization supervision。

这就是标准 O2O。

---

# 33. 从梯度角度理解 Sparse Supervision

假设 bbox head：

```python
pred_boxes = bbox_head(query_features)
```

300 个 query 都会产生 box：

```text
[300, 4]
```

但是 bbox regression loss 通常只对：

```text
matched positive predictions
```

计算。

如果 5 个 GT：

```text
matched queries = 5
```

那么 localization loss 直接使用的是：

```text
5 boxes
```

而不是：

```text
300 boxes
```

所以对于 localization：

```text
300 candidates
     ↓
Hungarian
     ↓
5 matched
     ↓
L1/GIoU
```

这就是为什么“positive supervision density”是值得研究的问题。

---

# 34. Hungarian Matching 的复杂度需要担心吗？

Hungarian algorithm 的经典复杂度通常可视为与 assignment 规模的三次方量级相关（具体实现与矩阵形状会影响实际开销）。

但 DETR 场景通常：

```text
queries ≈ 300
GT 数量远小于 300
```

而且 matcher 不参与梯度图。

在整个 detector 的计算中，backbone、encoder、decoder 等仍然占据大量计算。

所以学习阶段更应该关注：

```text
matching quality
```

和：

```text
training supervision
```

而不是把 Hungarian algorithm 当成整个模型的主要 FLOPs 瓶颈。

---

# 35. 为什么匹配需要同时看分类和定位？

我们构造一个反例。

GT：

```text
class = dog
```

prediction A：

```text
dog score = 0.99
IoU = 0.10
```

prediction B：

```text
dog score = 0.60
IoU = 0.90
```

只看分类：

```text
A 赢
```

只看 bbox：

```text
B 赢
```

但检测任务真正需要的是：

```text
类别正确
+
位置正确
```

所以 matcher 本质上是在问：

> 哪个 prediction 是这个 GT 的“综合最佳负责人”？

这也是为什么 classification/localization 的相对 cost 权重很重要。

---

# 36. 为什么匹配结果会随着训练不断变化？

epoch 1：

```text
query 17 ↔ GT dog
```

epoch 2 不一定还是 query 17。

因为模型参数更新以后：

```text
classification score 改变
bbox 改变
GIoU 改变
```

所以 cost matrix 改变：

\[
C^{(t+1)}\neq C^{(t)}
\]

于是 Hungarian assignment 也可能变化。

这就是 DETR 训练动态中非常有意思的一点：

```text
模型预测
  ↓
决定 assignment
  ↓
assignment 决定 supervision
  ↓
supervision 更新模型
  ↓
模型预测改变
  ↓
重新 assignment
```

它是一个不断演化的过程。

---

# 37. 为什么训练早期尤其困难？

训练初期：

```text
classification 还很差
bbox 也很差
```

所以：

```text
谁最适合匹配哪个 GT
```

本身可能就不稳定。

这意味着：

```text
assignment noisy / unstable
+
positive 数量少
```

会共同增加优化难度。

DETR 后续大量研究都围绕这些问题发展：

```text
better query initialization
better reference points
denoising training
better matching
better localization
dense supervision
```

DEIM 就属于：

```text
better matching / supervision
```

这条研究路线。

---

# 38. 本课和 DEIM 的真正连接

现在把逻辑串起来：

```text
DETR
│
├── set prediction
│
├── object queries
│
└── Hungarian O2O
        │
        ▼
   一个 GT ↔ 一个 query
        │
        ▼
  end-to-end / less duplicate assignment
        │
        ▼
      但是
        │
        ▼
positive supervision sparse
        │
        ▼
       DEIM
        │
        ▼
    Dense O2O
```

然后：

```text
Dense O2O
   │
   ▼
更多 positive supervision
   │
   ▼
但 match quality 不一致
   │
   ▼
MAL
```

再到 DEIMv2：

```text
DEIM
  ↓
upgraded Dense O2O
  ↓
Mosaic + MixUp + CopyBlend
  ↓
training-stage matching improvements
```

所以：

> 如果第 6 课不理解，后面第 15～17 课会变成“记名词”。

而理解了本课以后，DEIM 的设计动机会非常自然。

---

# 39. 本课必须记住的 10 个结论

### 1

DETR 输出的是一个 prediction set：

```text
N queries
→ N predictions
```

---

### 2

GT 也是一个 set：

```text
M objects
```

顺序本身没有检测语义。

---

### 3

训练前必须解决：

```text
prediction ↔ GT
```

的 assignment。

---

### 4

标准 DETR 使用 Hungarian matching 求全局 one-to-one assignment。

---

### 5

cost matrix：

```text
[N, M]
```

其中：

```text
C[i,j]
```

表示 prediction i 匹配 GT j 的代价。

---

### 6

常见 matching cost 包含：

\[
C
=
\lambda_{cls}C_{cls}
+
\lambda_{bbox}C_{bbox}
+
\lambda_{giou}C_{giou}
\]

---

### 7

Hungarian matching 不是：

```python
argmin()
```

因为 assignment 必须满足 one-to-one 约束。

---

### 8

Matching Cost 与最终 Training Loss 是两个不同概念。

---

### 9

如果：

```text
300 queries
5 GT
```

标准 O2O 最多只有：

```text
5 matched positives
```

---

### 10

这种 sparse positive supervision 正是理解 DEIM Dense O2O 的关键起点。

---

# 40. 本课练习

## 练习 1：shape

给定：

```text
B = 4
N = 300
num_classes = 80
```

四张图的 GT 数量：

```text
[3, 8, 5, 2]
```

回答：

1. `pred_logits` shape？
2. `pred_boxes` shape？
3. flatten 后 prediction 总数？
4. target concat 后总 GT 数？
5. 向量化大 cost matrix 的 shape？
6. 每张图真正 Hungarian 时对应矩阵 shape？

---

## 练习 2：O2O

一张图片：

```text
300 queries
12 GT
```

问：

标准 O2O 最多产生多少 matched positives？

positive/query ratio 是多少？

---

## 练习 3：手算 cost

给定：

```text
cost_class = -0.7
cost_bbox  =  0.12
cost_giou  = -0.5
```

权重：

```text
2 / 5 / 2
```

计算总 matching cost。

---

## 练习 4：解释

用自己的话解释：

> 为什么不能对每个 GT 单独 `argmin()`？

---

## 练习 5：概念辨析

解释：

```text
matching cost
```

和：

```text
training loss
```

的区别。

---

## 练习 6：思考 DEIM

如果我们希望增加 positive supervision，你能想到哪些方法？

先不要查答案。

尝试从：

```text
增加 GT？
增加 matching？
增加 auxiliary supervision？
```

三个方向思考。

等到第 15 课，再与 DEIM 的 Dense O2O 对照。

---

# 41. 练习答案

## 练习 1

```text
pred_logits:
[4, 300, 80]

pred_boxes:
[4, 300, 4]
```

flatten predictions：

\[
4\times300=1200
\]

总 GT：

\[
3+8+5+2=18
\]

大 cost matrix：

```text
[1200, 18]
```

每张图实际 assignment：

```text
image0: [300, 3]
image1: [300, 8]
image2: [300, 5]
image3: [300, 2]
```

---

## 练习 2

最多：

```text
12 matched positives
```

比例：

\[
12/300=0.04=4\%
\]

---

## 练习 3

\[
C
=
2(-0.7)+5(0.12)+2(-0.5)
\]

\[
=-1.4+0.6-1
\]

\[
=-1.8
\]

---

## 练习 4

因为不同 GT 独立 argmin 可能选择同一个 prediction。

O2O 要求：

```text
prediction 不能重复使用
```

因此需要求全局 assignment。

---

## 练习 5

```text
matching cost
→ 决定 prediction 和 GT 怎么配对
```

```text
training loss
→ 在配对以后决定网络参数怎样更新
```

---

# 42. 源码阅读时的“定位口诀”

以后看到 matcher 源码，先不要陷进每一行公式。

先找这 7 件事：

```text
① prediction logits 在哪里 flatten？
② prediction boxes 在哪里 flatten？
③ target labels 在哪里 concat？
④ target boxes 在哪里 concat？
⑤ cost_class 在哪里算？
⑥ cost_bbox / cost_giou 在哪里算？
⑦ linear_sum_assignment 在哪里调用？
```

只要这 7 个位置找到，整个 matcher 的骨架就已经看懂了。

然后再研究：

```text
focal-style class cost
alpha
gamma
change_matcher
iou_order_alpha
```

这样的 DEIM/DEIMv2 特有细节。

---

# 43. 下一次遇到 `[300, 7]` 时你应该立刻想到什么？

不要只想到：

```text
这是一个 tensor。
```

而应该想到：

```text
300
=
300 prediction queries

7
=
这张图的 7 个 GT

[300,7]
=
每一个 prediction
对
每一个 GT
的 pairwise matching cost
```

然后：

```text
Hungarian
↓
从 2100 个 candidate pairs
中选择 7 个合法 O2O pairs
```

如果这个画面已经形成，本课最重要的目标就完成了。

---

# 44. 本课总结图

```text
                     DETR training

image
  │
  ▼
backbone / encoder / decoder
  │
  ▼
300 queries
  │
  ├───────────────┐
  │               │
  ▼               ▼
class logits     boxes
[N,C]            [N,4]
  │               │
  └───────┬───────┘
          │
          ▼
   pairwise costs
          │
          ├── classification cost
          ├── L1 bbox cost
          └── GIoU cost
          │
          ▼
    Cost Matrix [N,M]
          │
          ▼
   Hungarian Algorithm
          │
          ▼
    One-to-One Matching
          │
    ┌─────┴─────┐
    ▼           ▼
 matched      unmatched
 queries       queries
    │
    ▼
criterion / losses
    │
    ▼
weighted total loss
    │
    ▼
backward()
```

而 DEIM 后面的故事，就是从这里继续：

```text
Standard O2O
    ↓
positive supervision sparse
    ↓
Dense O2O
    ↓
match quality varies
    ↓
MAL
    ↓
DEIMv2 upgraded Dense O2O
```

---

# 45. 建议你在进入第 7 课前做到

你不需要背 Hungarian algorithm 的内部矩阵操作步骤。

但应该能不看资料解释：

```text
1. 为什么 DETR 需要 matching？
2. 什么是 O2O？
3. 为什么不能直接 argmin？
4. cost matrix 的 [N,M] 分别是什么？
5. classification / L1 / GIoU 为什么都要参与？
6. matching cost 和 loss 有什么区别？
7. 300 queries + 5 GT 为什么只有最多 5 个 O2O positives？
8. 为什么这会引出 DEIM Dense O2O？
```

如果这 8 个问题都能回答，我们就已经具备继续读 DEIM matcher、criterion 和 Dense O2O 的基础。

---

## 参考入口

为了让本课与仓库/论文保持一致，建议后续源码学习时重点对照：

- **DETR 官方 `matcher.py`**：理解标准 Hungarian one-to-one assignment 的最干净入口。
- **DEIM 论文**：重点关注 sparse O2O supervision、Dense O2O 与 MAL 的动机。
- **DEIMv2 论文**：重点关注 upgraded Dense O2O。
- **DEIMv2 `configs/base/deimv2.yml`**：观察 `DEIMCriterion.matcher`、`cost_class / cost_bbox / cost_giou`、`change_matcher`、`iou_order_alpha` 与 `matcher_change_epoch`。

> 本课刻意没有提前展开 `change_matcher` 和 MAL 的内部公式。它们分别属于后续 DEIM/DEIMv2 训练优化课程。现在最重要的是把 **标准 Hungarian O2O** 建立成牢固的基线。
