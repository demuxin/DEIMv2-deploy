# DEIMv2 源码课程 · 第 16 课
# MAL——Matchability-Aware Loss

**本课核心目标**

第 15 课我们已经得到 DEIM 的第一条核心逻辑：

```text
Dense O2O
→ 每张训练输入中的 GT 更多
→ Hungarian O2O positive matches 更多
→ supervision 更密
```

但随之出现一个新问题：

```text
matched positive ≠ high-quality positive
```

特别是在训练早期和 Dense O2O 强增强场景下，Hungarian Matcher 可能不得不把一个 GT 分给一个 **IoU 很低、当前并不“好匹配”** 的 Query。

**MAL（Matchability-Aware Loss）就是 DEIM 为这个问题设计的分类损失。**

本课会从 BCE → Focal Loss → VFL → MAL，一步一步推导到 DEIM 官方 `loss_labels_mal()` 的真实 PyTorch 实现，并重点解释：

- `p`、`q`、`y` 分别是什么；
- 为什么 Hungarian matched 仍可能有 `IoU = 0`；
- VFL 为什么会弱化低 IoU positive；
- MAL 为什么把正样本 target 从 `q` 改成 `q^γ`；
- **最关键的一点：MAL 对 matched positive 不再额外乘一个 `q` 权重；**
- 为什么源码里 `target_score.pow(gamma)` 与论文公式完全对应；
- 为什么 `ious.detach()` 和 `pred_score.detach()` 很重要；
- `gamma=1.5` 到底怎样改变 supervision；
- MAL 与 Dense O2O 为什么是成套设计；
- MAL 在 `deim_dfine_x` 的 main / aux / encoder / DN loss 中怎样被调用。

---

## 0. 先记住本课最重要的一句话

**VFL 对低质量 matched positive 的问题，不只是“target 很低”，更关键的是它还用 IoU `q` 对整个 positive classification loss 再做一次缩放；MAL 去掉了这个 positive-side 的整体 `q` 衰减，让低质量但已经被 Hungarian 选中的 Query 仍能获得足够强的分类优化信号。**

用源码语言说：

```text
VFL positive:
    target = q
    weight = q

MAL positive:
    target = q^gamma
    weight = 1
```

这两行是整节课的钥匙。

---

# 1. 从第 15 课的问题继续

假设一张 Dense O2O 训练图中有：

```text
24 GT
300 Queries
```

Hungarian Matching 最终给出：

```text
24 matched Query-GT pairs
```

但这 24 个 pair 的质量可能完全不同：

```text
q3  ↔ GT0   IoU = 0.93
q8  ↔ GT1   IoU = 0.81
q21 ↔ GT2   IoU = 0.54
q37 ↔ GT3   IoU = 0.18
q44 ↔ GT4   IoU = 0.05
q71 ↔ GT5   IoU = 0.00
...
```

为什么 `IoU=0.05` 甚至 `IoU=0` 仍可能是 matched positive？

因为 Hungarian Matching 求的是：

$$
\sigma^*
=
\arg\min_{\sigma}
\sum_i C_{i,\sigma(i)}
$$

它要求的是：

```text
在当前所有 Query 中找到全局总代价最优的一对一 assignment
```

而不是：

```text
只有 IoU > 某阈值才能成为 positive
```

因此：

$$
\boxed{
\text{Hungarian matched}
\not\Rightarrow
\text{high IoU}
}
$$

---

# 2. “Matchability” 到底是什么意思？

可以先用直觉理解：

> 一个 Query 当前到底有多适合作为这个 GT 的匹配对象？

DEIM 中 MAL 使用匹配后的预测框与 GT 框之间的 IoU：

$$
q=\operatorname{IoU}(\hat b,b)
$$

作为 localization / matching quality 信号。

因此：

```text
q ≈ 1
→ 当前 box 与 GT 高度重合
→ 高质量 match

q ≈ 0
→ 当前 box 几乎没有定位好
→ 低质量 match
```

注意：

**matchability 不是 Hungarian cost 本身。**

Hungarian cost 可能综合：

```text
classification cost
L1 cost
GIoU cost
```

MAL 中这里使用的 `q` 则是匹配完成后计算出的 **IoU quality**。

---

# 3. 先从最普通的 BCE 开始

对某一个类别，设：

```text
logit = z
probability = p = sigmoid(z)
target = t
```

Binary Cross Entropy：

$$
L_{\text{BCE}}
=
-[
t\log p
+(1-t)\log(1-p)
]
$$

如果：

$$
t=1
$$

则：

$$
L=-\log p
$$

如果：

$$
t=0
$$

则：

$$
L=-\log(1-p)
$$

---

# 4. BCEWithLogitsLoss 为什么直接吃 logits？

PyTorch 通常不用：

```python
p = torch.sigmoid(logits)
loss = manual_bce(p, target)
```

而直接：

```python
F.binary_cross_entropy_with_logits(
    logits,
    target
)
```

因为它把：

```text
sigmoid + BCE
```

合并成数值更稳定的实现。

DEIM 的 VFL 和 MAL 最终都落到：

```python
F.binary_cross_entropy_with_logits(...)
```

---

# 5. DETR 分类为什么不是简单的普通 BCE？

假设：

```text
B = 2
Q = 300
C = 80
```

logits：

```text
[2, 300, 80]
```

如果每张图只有 6 个 GT，那么绝大多数 Query 都不是 matched positive。

因此分类存在严重的：

```text
positive / negative imbalance
```

这也是 Focal Loss 一类设计存在的重要背景。

---

# 6. Focal Loss 回顾

Focal Loss 的核心思想：

> easy samples 不应该长期主导梯度，把优化重点放到 hard samples。

二分类直觉形式：

$$
FL(p_t)
=
-\alpha_t(1-p_t)^\gamma\log p_t
$$

当样本已经非常容易：

```text
p_t → 1
```

则：

$$
(1-p_t)^\gamma\rightarrow0
$$

loss 被抑制。

---

# 7. 但检测还有另一个问题：分类置信度应该反映定位质量

假设两个 Query 都预测：

```text
class = person
confidence = 0.95
```

但：

```text
Query A: IoU = 0.95
Query B: IoU = 0.10
```

如果分类 target 都是硬标签：

```text
target = 1
```

那么 classification branch 并不知道：

```text
A 的 box 很好
B 的 box 很差
```

于是一个自然想法是：

> 把 localization quality 融进 classification target。

这正是 Varifocal Loss（VFL）类方法的重要思想。

---

# 8. VFL 中的 q

对 matched positive：

$$
q=\operatorname{IoU}(\hat b,b)
$$

并令 classification soft target：

$$
t=q
$$

因此：

```text
IoU = 0.95 → target score = 0.95
IoU = 0.50 → target score = 0.50
IoU = 0.05 → target score = 0.05
```

这样 classification confidence 被鼓励与 localization quality 对齐。

---

# 9. DEIM 论文中的 VFL positive

对 positive：

$$
L_{\text{VFL}}^+
=
-q[
q\log p
+(1-q)\log(1-p)
]
$$

你应该立刻看出它其实是：

$$
\boxed{
L_{\text{VFL}}^+
=
q\cdot
\operatorname{BCE}(p,q)
}
$$

即：

```text
target = q
```

并且：

```text
整个 positive BCE 再乘 q
```

---

# 10. 这正是低质量 matched positive 的问题

假设：

$$
q=0.05
$$

那么：

$$
L_{\text{VFL}}^+
=
0.05
\cdot
\operatorname{BCE}(p,0.05)
$$

整个 loss 被：

```text
× 0.05
```

强烈缩小。

但这个 Query 明明已经被 Hungarian 指定为：

```text
当前 GT 的 O2O positive
```

Dense O2O 又恰恰会增加这种低质量 match。

于是出现矛盾：

```text
Dense O2O:
想通过更多 matches 获得更多 supervision

VFL:
对低 IoU matches 又把 classification loss 压得很小
```

这会削弱 Dense O2O 新增 supervision 的价值。

---

# 11. 更极端：q = 0

如果一个 Hungarian matched Query 与 GT：

$$
q=0
$$

那么 VFL 的 quality target 是：

```text
0
```

且 positive weighting 也趋于：

```text
0
```

从分类优化角度看，这个已经被 Hungarian 认定的 positive 几乎失去了 positive supervision。

DEIM 论文明确把这视为 VFL 在 O2O / Dense O2O 场景中的问题之一。

---

# 12. MAL 的设计目标

MAL 要同时做到两件事：

### 目标 A

保留：

```text
classification confidence
↔ localization quality
```

的联系。

### 目标 B

不要因为：

```text
当前 IoU 很低
```

就把 Hungarian positive 的整个 classification loss 几乎抹掉。

因此 MAL 做了一个很小、但非常关键的修改。

---

# 13. MAL 公式

设：

- \(p\)：当前类别预测概率；
- \(q\)：matched prediction 与 GT 的 IoU；
- \(y\in\{0,1\}\)：是否属于 matched positive；
- \(\gamma\)：quality / focusing exponent。

MAL 可以用与官方源码完全一致的 BCE 形式写成：

### Positive：\(y=1\)

$$
t=q^\gamma
$$

$$
\boxed{
L_{\text{MAL}}^+
=
-[
q^\gamma\log p
+(1-q^\gamma)\log(1-p)
]
}
$$

即：

$$
\boxed{
L_{\text{MAL}}^+
=
\operatorname{BCE}(p,q^\gamma)
}
$$

### Negative：\(y=0\)

$$
\boxed{
L_{\text{MAL}}^-
=
-p^\gamma\log(1-p)
}
$$

当前 DEIM 源码还允许通过 `mal_alpha` 对 negative weight 再乘一个可选系数。

---

# 14. 注意论文排版公式与源码的符号

你可能在论文文本抽取或网页转录中看到 positive 公式第二项写成：

$$
+(1-q^\gamma)\log(1-p)
$$

容易造成符号困惑。

实际官方代码调用：

```python
F.binary_cross_entropy_with_logits(
    src_logits,
    target_score,
    weight=weight
)
```

因此 positive 的实际优化目标就是标准 BCE：

$$
-[
t\log p+(1-t)\log(1-p)
]
$$

其中：

$$
t=q^\gamma
$$

**读论文公式时遇到排版/文本抽取歧义，最终一定回到官方实现验证。**

---

# 15. MAL 和 VFL 最关键的差别

## VFL positive

$$
L^+_{\text{VFL}}
=
q\cdot\operatorname{BCE}(p,q)
$$

## MAL positive

$$
L^+_{\text{MAL}}
=
\operatorname{BCE}(p,q^\gamma)
$$

所以有两个变化：

```text
变化 1：
positive target
q → q^gamma

变化 2：
positive overall weight
q → 1
```

**第二个变化对理解 low-quality matching 尤其重要。**

---

# 16. gamma = 1.5 时 q^gamma 是多少？

当前 DEIM base 配置：

```yaml
DEIMCriterion:
  gamma: 1.5
```

因此：

$$
t=q^{1.5}
$$

具体：

| IoU \(q\) | MAL target \(q^{1.5}\) |
|---:|---:|
| 0.00 | 0.000 |
| 0.05 | 0.011 |
| 0.10 | 0.032 |
| 0.20 | 0.089 |
| 0.50 | 0.354 |
| 0.80 | 0.716 |
| 0.95 | 0.926 |
| 1.00 | 1.000 |

当：

$$
0<q<1,\quad\gamma>1
$$

有：

$$
q^\gamma<q
$$

所以 MAL 的 target score 比原始 IoU 更“严格”。

---

# 17. 一个很容易产生的误解

看到：

```text
q = 0.05
q^1.5 ≈ 0.011
```

你可能会问：

> target 不是更小了吗？为什么反而说 MAL 更重视低质量 match？

关键就在于：

**不能只看 target，还必须看 loss weight。**

VFL：

```text
target = 0.05
whole positive loss × 0.05
```

MAL：

```text
target ≈ 0.011
whole positive loss × 1
```

MAL 没有把整个 matched-positive BCE 再乘一个很小的 `q`。

---

# 18. 用梯度看得更清楚

对 BCEWithLogits：

$$
\frac{\partial L}{\partial z}
=
p-t
$$

其中：

$$
p=\sigma(z)
$$

对 MAL positive：

$$
t=q^\gamma
$$

所以：

$$
\boxed{
\frac{\partial L_{\text{MAL}}^+}{\partial z}
=
p-q^\gamma
}
$$

---

# 19. VFL positive 的梯度

因为：

$$
L_{\text{VFL}}^+
=
q\cdot\operatorname{BCE}(p,q)
$$

且源码中的 \(q\) 被 detach，可视为常量：

$$
\boxed{
\frac{\partial L_{\text{VFL}}^+}{\partial z}
=
q(p-q)
}
$$

这就是最漂亮的对比。

---

# 20. 低质量 match 的梯度对比

假设：

$$
q=0.05,\quad p=0.5,\quad\gamma=1.5
$$

MAL target：

$$
q^{1.5}\approx0.0112
$$

MAL：

$$
\frac{\partial L}{\partial z}
\approx
0.5-0.0112
=
0.4888
$$

VFL：

$$
\frac{\partial L}{\partial z}
=
0.05(0.5-0.05)
=
0.0225
$$

梯度绝对值相差约：

$$
21.7\times
$$

这非常直观地说明：

> VFL 的 `q` weighting 会让低 IoU positive 的 classification gradient 极小，而 MAL 不再这样做。

---

# 21. 但注意：MAL 并不是“强迫低 IoU positive 得到高置信度”

对：

$$
q=0.05
$$

MAL target：

$$
q^{1.5}\approx0.011
$$

也就是说它并没有说：

```text
这是 matched positive
→ class confidence 应该变成 1.0
```

恰恰相反：

```text
当前定位质量极差
→ quality-aware classification target 很低
```

但它会给这个 Query **足够强的梯度去校准当前 classification score**。

这是：

```text
optimize low-quality match
```

和：

```text
blindly boost low-quality score
```

的根本区别。

---

# 22. q = 0 时 MAL 为什么仍然有意义？

如果：

$$
q=0
$$

matched positive 的 MAL target：

$$
t=0^\gamma=0
$$

positive weight 仍为：

```text
1
```

所以：

$$
L=-\log(1-p)
$$

如果当前 Query 对 GT 类别错误地给出很高分：

```text
p = 0.8
```

它会收到明显的校准梯度。

与 VFL 的 `q × BCE` 相比，不会因为：

```text
q = 0
```

而让整个 positive classification term 消失。

---

# 23. 高质量 match 时又怎样？

假设：

$$
q=0.95,\quad\gamma=1.5
$$

$$
q^{1.5}\approx0.926
$$

所以 MAL target 仍很高：

```text
≈ 0.926
```

一个定位非常好的 Query 会被鼓励拥有：

```text
高 classification confidence
```

因此 MAL 仍保留：

```text
classification score
↔ localization quality
```

的排序能力。

---

# 24. MAL 可以看成“Quality Calibration + Match-Aware Optimization”

对 positive：

```text
q
↓
q^gamma
↓
soft classification target
```

这负责：

```text
quality calibration
```

同时：

```text
positive weight = 1
```

避免低质量 match 被整体抑制。

对 negative：

```text
weight = p^gamma
```

抑制 easy negatives，保留 hard negatives。

---

# 25. 现在进入真实 DEIM 源码

当前官方：

```text
engine/deim/deim_criterion.py
```

中：

```python
def loss_labels_mal(
    self,
    outputs,
    targets,
    indices,
    num_boxes,
    values=None
):
```

参数含义：

```text
outputs
    模型预测

targets
    GT

indices
    Hungarian matched indices

num_boxes
    distributed-normalized GT 数量

values
    可选的外部 quality 值
    当前 boxes_weight_format 等逻辑可传入 IoU
```

---

# 26. 第 1 行：必须有 pred_boxes

```python
assert 'pred_boxes' in outputs
```

为什么分类 loss 还需要 box？

因为 MAL 的 target：

$$
q^\gamma
$$

中：

$$
q=\operatorname{IoU}(
\text{pred box},
\text{GT box}
)
$$

所以 MAL 不是纯粹只看 logits 的 classification loss。

---

# 27. 得到 matched Query 的全局索引

```python
idx = self._get_src_permutation_idx(indices)
```

假设 batch=2：

```text
image 0:
src = [3, 17, 91]

image 1:
src = [8, 22]
```

会转换成类似：

```text
batch_idx = [0,0,0,1,1]
src_idx   = [3,17,91,8,22]
```

这样就能直接：

```python
outputs['pred_boxes'][idx]
```

取出所有 matched predictions。

---

# 28. 取 matched predicted boxes

```python
src_boxes = outputs['pred_boxes'][idx]
```

如果总 matched 数：

```text
M = 24
```

则：

```text
src_boxes
[24, 4]
```

box 格式：

```text
cx, cy, w, h
```

且通常已经归一化。

---

# 29. 取对应 GT boxes

```python
target_boxes = torch.cat([
    t['boxes'][i]
    for t, (_, i)
    in zip(targets, indices)
], dim=0)
```

输出：

```text
target_boxes
[M, 4]
```

顺序与：

```text
src_boxes
```

一一对应。

---

# 30. 计算 IoU Matrix

```python
ious, _ = box_iou(
    box_cxcywh_to_xyxy(src_boxes),
    box_cxcywh_to_xyxy(target_boxes)
)
```

如果：

```text
M = 24
```

则：

```text
ious
[24, 24]
```

为什么不是 `[24]`？

因为 `box_iou(A,B)` 计算：

```text
A 中每个 box
×
B 中每个 box
```

的 pairwise IoU matrix。

---

# 31. 为什么取 diag？

```python
ious = torch.diag(ious).detach()
```

因为：

```text
src_boxes[k]
```

已经与：

```text
target_boxes[k]
```

按 Hungarian matching 对齐。

所以真正需要：

$$
\operatorname{IoU}
(
\hat b_k,
b_k
)
$$

即矩阵主对角线。

shape：

```text
[M]
```

---

# 32. `detach()` 为什么非常重要？

```python
ious = torch.diag(ious).detach()
```

MAL 中 IoU 的作用是：

```text
构造 classification soft target
```

而不是让 classification loss 通过：

```text
target_score
→ IoU
→ pred_boxes
```

反向传播到 box branch。

`detach()` 切断这条梯度：

```text
pred_boxes
─X→ IoU target
─X→ MAL gradient
```

box branch 有自己明确的：

```text
L1
GIoU
FGL
...
```

来优化。

---

# 33. 如果 `values` 不为空

源码：

```python
else:
    ious = values
```

`get_loss_meta_info()` 可以根据配置提前构造 quality 值。

例如：

```text
boxes_weight_format = 'iou'
```

时可把 detach 后的 IoU 作为 `values` 传进 MAL。

这避免重复计算，也让 criterion 的 quality weighting 更统一。

---

# 34. 取 logits

```python
src_logits = outputs['pred_logits']
```

对 COCO：

```text
[B, Q, 80]
```

例如：

```text
[2, 300, 80]
```

注意 MAL 最终不是只计算 matched positions。

它会构造一个完整：

```text
[B,Q,C]
```

target tensor，让 unmatched Query / non-target classes 也作为 negatives 参与分类。

---

# 35. 取 matched GT labels

```python
target_classes_o = torch.cat([
    t["labels"][J]
    for t, (_, J)
    in zip(targets, indices)
])
```

如果：

```text
M = 24
```

则：

```text
target_classes_o
[24]
```

例如：

```text
[0, 0, 2, 17, 5, ...]
```

---

# 36. 先把所有 Query 设为 no-object sentinel

```python
target_classes = torch.full(
    src_logits.shape[:2],
    self.num_classes,
    dtype=torch.int64,
    device=src_logits.device
)
```

如果：

```text
num_classes = 80
```

那么 sentinel：

```text
80
```

shape：

```text
[B,Q]
```

---

# 37. matched Query 写入真实类别

```python
target_classes[idx] = target_classes_o
```

于是：

```text
unmatched Query:
80

matched person Query:
0

matched car Query:
2
```

---

# 38. One-hot 后去掉 no-object 维度

```python
target = F.one_hot(
    target_classes,
    num_classes=self.num_classes + 1
)[..., :-1]
```

shape：

```text
[B,Q,81]
→ [...,:-1]
→ [B,Q,80]
```

对 unmatched Query：

```text
one-hot 的 1 在第 80 维
```

切掉最后一维后变成：

```text
[0,0,...,0]
```

对 matched person：

```text
[1,0,0,...]
```

---

# 39. 先创建 IoU Score Map

```python
target_score_o = torch.zeros_like(
    target_classes,
    dtype=src_logits.dtype
)
```

shape：

```text
[B,Q]
```

初始全部：

```text
0
```

---

# 40. matched Query 写入 IoU

```python
target_score_o[idx] = ious.to(
    target_score_o.dtype
)
```

例如：

```text
q3  → 0.95
q17 → 0.20
q91 → 0.05
```

unmatched：

```text
0
```

---

# 41. IoU 与 one-hot 类别相乘

```python
target_score = (
    target_score_o.unsqueeze(-1)
    * target
)
```

shape：

```text
target_score_o
[B,Q]

unsqueeze
[B,Q,1]

target
[B,Q,C]

broadcast multiply
↓
target_score
[B,Q,C]
```

---

# 42. 一个具体例子

假设：

```text
C = 4
Query 7 matched class = 2
IoU = 0.8
```

one-hot：

```text
target[7]
=
[0,0,1,0]
```

IoU：

```text
0.8
```

相乘：

```text
target_score[7]
=
[0,0,0.8,0]
```

所以只有正确类别拥有 quality target。

---

# 43. 取预测概率，但 detach

```python
pred_score = F.sigmoid(
    src_logits
).detach()
```

shape：

```text
[B,Q,C]
```

这不是最终 BCE 的预测输入。

它只是用来构造：

```text
negative focusing weight
```

---

# 44. 为什么 `pred_score.detach()`？

后面：

```python
pred_score.pow(self.gamma)
```

只是作为 sample weight。

如果不 detach，weight 本身也会对 logits 求导，实际梯度会包含：

```text
loss value 对动态 weight 的导数
```

当前官方实现明确选择：

```text
weight 作为常量调制项
```

真正 BCE 对 logits 的梯度由：

```text
BCEWithLogits
```

主导。

---

# 45. MAL 最关键的一行：q → q^gamma

```python
target_score = target_score.pow(
    self.gamma
)
```

matched positive 的正确类别：

```text
q → q^gamma
```

negatives 原本：

```text
0
```

仍然：

```text
0^gamma = 0
```

所以最终：

```text
target_score
[B,Q,C]
```

就是 MAL 的 soft target。

---

# 46. 构造 MAL weight

默认：

```python
weight = (
    pred_score.pow(self.gamma)
    * (1 - target)
    + target
)
```

这行极其重要。

分两种情况。

---

# 47. 对 matched 正确类别

正确类别位置：

```text
target = 1
```

所以：

$$
w
=
p^\gamma(1-1)+1
=
1
$$

因此：

$$
\boxed{
w_{\text{positive}}=1
}
$$

这就是 MAL 与 VFL 的关键差别。

---

# 48. 对 negative class / unmatched Query

这些位置：

```text
target = 0
```

所以：

$$
w
=
p^\gamma(1-0)+0
=
p^\gamma
$$

因此：

$$
\boxed{
w_{\text{negative}}=p^\gamma
}
$$

easy negative：

```text
p ≈ 0
→ weight ≈ 0
```

hard negative：

```text
p 较大
→ weight 较大
```

这保留了 Focal-style hard-negative focusing。

---

# 49. `mal_alpha` 是什么？

源码：

```python
if self.mal_alpha != None:
    weight = (
        self.mal_alpha
        * pred_score.pow(self.gamma)
        * (1 - target)
        + target
    )
```

此时：

```text
positive weight = 1
negative weight = mal_alpha * p^gamma
```

默认 `mal_alpha=None` 时：

```text
positive = 1
negative = p^gamma
```

DEIM 论文强调 MAL 简化了正负权重，并可不依赖 VFL 中的 α；当前代码仍保留 `mal_alpha` 作为可选实验接口。

---

# 50. 最终 BCE

```python
loss = F.binary_cross_entropy_with_logits(
    src_logits,
    target_score,
    weight=weight,
    reduction='none'
)
```

shape：

```text
[B,Q,C]
```

所以每个：

```text
image
× query
× class
```

都有一个分类 loss。

---

# 51. Loss Reduction

源码：

```python
loss = (
    loss.mean(1).sum()
    * src_logits.shape[1]
    / num_boxes
)
```

假设：

```text
loss
[B,Q,C]
```

`mean(1)`：

```text
对 Q 维取平均
→ [B,C]
```

`sum()`：

```text
→ scalar
```

再乘：

```text
Q / num_boxes
```

因为：

$$
\operatorname{mean}_Q(\cdot)\times Q
=
\operatorname{sum}_Q(\cdot)
$$

所以整体可理解为：

```text
对 Query/class loss 汇总
再按 num_boxes 归一化
```

---

# 52. num_boxes 从哪里来？

Criterion forward 中：

```python
num_boxes = sum(
    len(t["labels"])
    for t in targets
)
```

分布式训练时还会：

```text
all_reduce
÷ world_size
clamp(min=1)
```

目的：

```text
不同 GPU / batch 中 GT 数不同
仍保持合理的 loss normalization
```

Dense O2O 会改变每图 GT 数，因此理解这个 normalization 非常重要。

---

# 53. VFL 源码与 MAL 源码并排看

## VFL

```python
target_score = q * one_hot

weight = (
    alpha * p.pow(gamma) * (1 - target)
    + target_score
)
```

对 positive：

```text
target_score = q
weight = q
```

## MAL

```python
target_score = (q * one_hot).pow(gamma)

weight = (
    p.pow(gamma) * (1 - target)
    + target
)
```

对 positive：

```text
target_score = q^gamma
weight = 1
```

**这几乎就是整篇 MAL 设计的源码浓缩。**

---

# 54. 一张表彻底分清 BCE / Focal / VFL / MAL

| Loss | Positive target | Positive weight | Negative target | Negative weight |
|---|---|---|---|---|
| BCE | 1 | 1 | 0 | 1 |
| Focal | 1 | hard-sample modulation | 0 | hard-sample modulation |
| VFL | \(q\) | \(q\) | 0 | \(\alpha p^\gamma\) |
| MAL | \(q^\gamma\) | **1** | 0 | \(p^\gamma\) |

MAL 的关键词：

```text
quality-aware target
+
no q attenuation for matched positive
+
focal-style negative weighting
```

---

# 55. 为什么 Dense O2O 特别需要 MAL？

第 15 课：

```text
Mosaic / MixUp
↓
GT per image ↑
↓
Hungarian matched positives ↑
```

但同时：

```text
小目标 ↑
裁切目标 ↑
人工上下文 ↑
hard targets ↑
early low-IoU matches ↑
```

如果继续使用：

```text
VFL positive weight = q
```

那么新增的 low-quality positive 很可能又被弱化。

MAL：

```text
positive weight = 1
```

才能真正让 Dense O2O 增加的 supervision 被有效利用。

所以：

$$
\boxed{
\text{Dense O2O + MAL}
}
$$

应当作为一个整体理解。

---

# 56. MAL 并不会直接改变 Hungarian Matching

非常容易混淆：

```text
Matcher
```

与：

```text
Criterion
```

Hungarian：

```text
决定谁和谁匹配
```

MAL：

```text
匹配完成以后
决定 classification 怎样优化这些 matches
```

即：

```text
predictions
+ targets
     ↓
Hungarian Matcher
     ↓
indices
     ↓
MAL
```

---

# 57. MAL 也不直接替代 Box Loss

MAL：

```text
classification loss
```

Box branch 仍有：

```text
L1
GIoU
FGL
DDF / localization distillation
```

MAL 只是使用 IoU 来构造：

```text
classification quality target
```

它不是 box regression loss。

---

# 58. 为什么 IoU 必须 detach？

如果不 detach：

```text
pred_boxes
↓
IoU
↓
target_score
↓
classification BCE
↓
gradient back to pred_boxes
```

这样 classification loss 会通过“标签本身”反向修改 box。

这会让：

```text
target
```

不再是固定监督信号。

当前实现明确：

```python
ious.detach()
```

把：

```text
quality measurement
```

与：

```text
box regression gradient
```

解耦。

---

# 59. 为什么 pred_score 也 detach？

negative weight：

$$
p^\gamma
$$

是用当前预测 confidence 衡量：

```text
这个 negative 有多 hard
```

`detach()` 后：

```text
它只决定“这一项 loss 有多重要”
```

而不额外产生：

```text
weight 自身的 gradient path
```

这是阅读 dynamic-weight loss 时非常重要的源码习惯：

> 每看到由 prediction 计算出的 weight，都要立刻检查它是否 `detach()`。

---

# 60. MAL 在 deim_dfine_x 中不是只算最后一层

`DEIMCriterion.forward()` 会对：

```text
main output
aux_outputs
pre_outputs
enc_aux_outputs
dn_outputs
dn_pre_outputs
```

按配置调用相应 loss。

因此你会在训练日志中看到类似：

```text
loss_mal
loss_mal_aux_0
loss_mal_aux_1
...
loss_mal_enc_0
...
loss_mal_dn_0
...
```

具体有哪些 key 取决于模型实际 outputs 和 weight_dict。

---

# 61. 一个非常关键的 Criterion 细节：MAL 与 Global Union 的 indices 不同

当前 `DEIMCriterion.forward()`：

```python
use_uni_set = (
    self.use_uni_set
    and (loss in ['boxes', 'local'])
)
```

也就是说：

```text
boxes / local
→ 可以使用 global union indices

MAL
→ 使用普通 O2O matcher indices
```

这是第 14、15、16 课必须串起来理解的地方。

---

# 62. 为什么 MAL 不直接使用 Global Union？

Classification O2O identity 要保持：

```text
一个 GT
↔
一个 classification-positive Query
```

而 localization 的 union set 可以吸收：

```text
不同 decoder layer
对同一个 GT 的不同优质 localization candidates
```

所以当前设计是：

```text
Classification:
strict O2O indices

Localization:
union indices（可选）
```

这体现了 DEIM / D-FINE 对：

```text
classification matching
```

与：

```text
localization supervision
```

的解耦。

---

# 63. MAL 与 DN Loss

DN query 的匹配身份是已知的：

```text
GT
→ noisy DN query
```

Criterion 使用：

```python
get_cdn_matched_indices(...)
```

构造 DN indices。

然后：

```text
MAL
boxes
local
```

仍可在 DN outputs 上计算。

但这里的 indices 不是 Hungarian 求出来的。

---

# 64. MAL 与 Encoder Auxiliary Output

Criterion 也可能对：

```text
enc_aux_outputs
```

计算 MAL。

如果 encoder auxiliary head 是：

```text
class_agnostic
```

源码会临时：

```python
self.num_classes = 1
```

并把 labels 变成：

```text
0
```

然后计算相应 auxiliary loss。

这说明 MAL 是整个训练 criterion 的 classification primitive，而不只是最终 decoder head 的一项损失。

---

# 65. 当前 deim_dfine 的默认 loss 配置

官方 DEIM base 中：

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

所以主损失可以概念化为：

$$
L
=
1L_{\text{MAL}}
+
5L_{L1}
+
2L_{\text{GIoU}}
+
0.15L_{\text{FGL}}
+
1.5L_{\text{DDF}}
+\cdots
$$

`...` 表示 auxiliary / encoder / DN 等分支中相应加权项。

---

# 66. 为什么 gamma 不能只理解成 Focal Loss 的 gamma？

在 MAL 中同一个：

```text
gamma
```

同时参与：

### Positive target

$$
q^\gamma
$$

### Negative weight

$$
p^\gamma
$$

因此调 gamma 同时改变：

```text
positive quality calibration
```

和：

```text
negative focusing strength
```

它不是一个只影响 negatives 的普通 focal hyperparameter。

---

# 67. gamma 增大时发生什么？

对：

$$
0<q<1
$$

gamma 越大：

$$
q^\gamma
$$

越小。

例如 q=0.5：

```text
gamma=1.0 → 0.500
gamma=1.5 → 0.354
gamma=2.0 → 0.250
gamma=3.0 → 0.125
```

同时 negative：

$$
p^\gamma
$$

对低 p 的抑制也更强。

因此 gamma 是一个同时控制：

```text
quality sharpness
+
negative focusing
```

的参数。

---

# 68. gamma = 0 会怎样？

纯数学上：

$$
q^0=1
$$

对正样本会退化成：

```text
hard target = 1
```

negative weight：

$$
p^0=1
$$

于是接近普通 BCE。

但注意：

```text
0^0
```

的数值/实现语义需要额外注意，因此不要把 gamma=0 当作实际推荐配置。

这个思想实验只是帮助理解 gamma 的作用。

---

# 69. gamma = 1 会怎样？

Positive：

$$
q^\gamma=q
$$

Negative：

$$
p^\gamma=p
$$

此时 MAL positive target 与 VFL 相同：

```text
target = q
```

但仍有一个关键区别：

```text
MAL positive weight = 1
VFL positive weight = q
```

所以即使 gamma=1：

```text
MAL ≠ VFL
```

---

# 70. PyTorch 实验 1：手写 MAL

```python
import torch
import torch.nn.functional as F


def mal_binary(logit, q, gamma=1.5, positive=True):
    p = torch.sigmoid(logit).detach()

    if positive:
        target = q.pow(gamma)
        weight = torch.ones_like(target)
    else:
        target = torch.zeros_like(q)
        weight = p.pow(gamma)

    return F.binary_cross_entropy_with_logits(
        logit,
        target,
        weight=weight,
        reduction="none"
    )
```

---

# 71. PyTorch 实验 2：手写 VFL Positive

```python
def vfl_positive(logit, q):
    return F.binary_cross_entropy_with_logits(
        logit,
        q,
        weight=q,
        reduction="none"
    )
```

然后比较：

```python
qs = torch.tensor([
    0.05,
    0.20,
    0.50,
    0.80,
    0.95,
])

logit = torch.zeros_like(qs)

print(
    "VFL:",
    vfl_positive(logit, qs)
)

print(
    "MAL:",
    mal_binary(
        logit,
        qs,
        gamma=1.5,
        positive=True
    )
)
```

`logit=0` 对应：

$$
p=0.5
$$

这是观察两种 loss weighting 差异最方便的起点。

---

# 72. PyTorch 实验 3：直接比较梯度

```python
import torch
import torch.nn.functional as F


def grad_of_vfl(q_value):
    z = torch.tensor(
        0.0,
        requires_grad=True
    )

    q = torch.tensor(q_value)

    loss = F.binary_cross_entropy_with_logits(
        z,
        q,
        weight=q
    )

    loss.backward()
    return z.grad.item()


def grad_of_mal(q_value, gamma=1.5):
    z = torch.tensor(
        0.0,
        requires_grad=True
    )

    q = torch.tensor(q_value)
    target = q.pow(gamma)

    loss = F.binary_cross_entropy_with_logits(
        z,
        target
    )

    loss.backward()
    return z.grad.item()


for q in [
    0.05,
    0.20,
    0.50,
    0.80,
    0.95,
]:
    print(
        q,
        grad_of_vfl(q),
        grad_of_mal(q)
    )
```

重点观察：

```text
q = 0.05
q = 0.20
```

的梯度量级。

---

# 73. PyTorch 实验 4：验证公式

对 MAL positive：

$$
\frac{\partial L}{\partial z}
=
p-q^\gamma
$$

可以验证：

```python
import math

p = 0.5
q = 0.05
gamma = 1.5

analytic = (
    p
    -
    q ** gamma
)

print(analytic)
```

与 autograd 结果比较。

---

# 74. PyTorch 实验 5：negative focusing

```python
import torch

gamma = 1.5

ps = torch.tensor([
    0.001,
    0.01,
    0.1,
    0.5,
    0.9,
])

weights = ps.pow(gamma)

for p, w in zip(ps, weights):
    print(
        f"p={p.item():.3f}",
        f"weight={w.item():.6f}"
    )
```

你应该看到：

```text
easy negative:
p≈0
→ weight≈0

hard false positive:
p较大
→ weight显著增大
```

---

# 75. PyTorch 实验 6：复现官方 target_score 构造

假设：

```text
B = 1
Q = 5
C = 3
```

```python
import torch
import torch.nn.functional as F

B, Q, C = 1, 5, 3

# Query 1 matched class 0, IoU=.2
# Query 3 matched class 2, IoU=.8

target_classes = torch.full(
    (B, Q),
    C,
    dtype=torch.long
)

target_classes[0, 1] = 0
target_classes[0, 3] = 2

target = F.one_hot(
    target_classes,
    num_classes=C + 1
)[..., :-1]

target_score_o = torch.zeros(
    B,
    Q
)

target_score_o[0, 1] = 0.2
target_score_o[0, 3] = 0.8

target_score = (
    target_score_o.unsqueeze(-1)
    * target
)

target_score = target_score.pow(1.5)

print(target)
print(target_score)
```

你应该能亲眼看到：

```text
只有 matched Query 的 matched class
才出现 q^gamma soft target
```

---

# 76. PyTorch 实验 7：复现官方 weight

```python
logits = torch.randn(
    B,
    Q,
    C
)

pred_score = logits.sigmoid().detach()

weight = (
    pred_score.pow(1.5)
    * (1 - target)
    + target
)

print(weight.shape)
```

检查：

```python
print(
    weight[0, 1, 0]
)
```

应该严格为：

```text
1
```

因为它是 matched positive class。

---

# 77. PyTorch 实验 8：验证 detach

```python
logits = torch.randn(
    4,
    requires_grad=True
)

p = logits.sigmoid()
p_detached = p.detach()

print(p.requires_grad)
print(p_detached.requires_grad)
```

输出：

```text
True
False
```

以后阅读所有：

```text
quality-aware loss
distillation loss
dynamic weighting
```

都应该主动检查这一点。

---

# 78. PyTorch 实验 9：IoU quality 不回传到 box

可以做一个最小实验：

```python
pred_box = torch.tensor(
    [0.2, 0.2, 0.6, 0.6],
    requires_grad=True
)

q = some_iou_function(
    pred_box,
    gt_box
)

target = q.detach().pow(1.5)

cls_loss = bce(
    cls_logit,
    target
)
```

只对：

```text
cls_loss
```

backward。

应观察：

```text
cls_logit.grad != None
pred_box.grad == None
```

然后再加入：

```text
box_loss
```

观察 box gradient 来自哪里。

---

# 79. PyTorch 实验 10：gamma sweep

```python
import torch

q = torch.linspace(
    0,
    1,
    11
)

for gamma in [
    1.0,
    1.5,
    2.0,
    3.0
]:
    print(
        gamma,
        q.pow(gamma)
    )
```

观察：

```text
gamma ↑
→ 中低 IoU target 更低
→ quality calibration 更尖锐
```

同时别忘了：

```text
negative p^gamma focusing 也同步改变
```

---

# 80. 推荐实验：画 VFL vs MAL Loss Curve

固定：

```text
q = 0.05
```

让：

```text
p: 0.001 → 0.999
```

分别画：

```text
VFL positive loss
MAL positive loss
```

再做：

```text
q = 0.95
```

你会复现论文 Figure 4 的核心现象。

图建议使用：

```text
x-axis: predicted probability p
y-axis: classification loss
```

并分别观察：

```text
low-quality match
high-quality match
```

---

# 81. 推荐实验：画 Gradient Curve

比 loss curve 更重要：

```text
dL / dz
```

对：

```text
q = 0.05
q = 0.20
q = 0.50
q = 0.80
q = 0.95
```

画：

```text
|gradient|
vs
p
```

你会真正理解 MAL 为什么改变 optimization dynamics。

---

# 82. 推荐实验：训练时记录 matched IoU

在 matcher 后统计：

```python
matched_ious = []
```

每个 batch 保存：

```text
mean
median
P10
P25
P50
P75
P90
```

并按 epoch 画：

```text
matched IoU distribution
```

你会看到训练早期：

```text
low-quality matches 更多
```

后期：

```text
distribution 向高 IoU 移动
```

这会让 MAL 从抽象公式变成真实训练现象。

---

# 83. 推荐实验：Dense O2O OFF / ON

比较：

```text
A. Dense O2O OFF
B. Dense O2O ON
```

记录：

```text
targets/image
matched positives/image
matched IoU histogram
MAL loss
classification gradient norm
AP
convergence speed
```

你要验证：

```text
Dense O2O
→ positives ↑
→ low-quality matches 也可能 ↑
```

这就是 MAL 的真实使用环境。

---

# 84. 推荐实验：VFL vs MAL

固定所有其他条件：

```text
model
seed
augmentation
optimizer
LR
epochs
```

只改：

```text
losses: ['vfl', ...]
```

与：

```text
losses: ['mal', ...]
```

记录：

```text
AP
AP50
AP75
convergence curve
matched-IoU buckets
```

尤其建议把 matched positives 按 IoU 分桶：

```text
[0.0,0.1)
[0.1,0.3)
[0.3,0.5)
[0.5,0.7)
[0.7,0.9)
[0.9,1.0]
```

分析每桶：

```text
loss
gradient
confidence
```

---

# 85. 推荐实验：gamma Ablation

```text
gamma = 1.0
gamma = 1.25
gamma = 1.5
gamma = 2.0
```

不要只看最终 AP。

同时记录：

```text
positive target distribution q^gamma
negative weight distribution p^gamma
classification calibration
convergence
```

这样才能知道 gamma 为什么有效。

---

# 86. 一个非常重要的调试陷阱：不要把 q 当成可训练 label

正确：

```python
q = iou(...).detach()
```

错误实验：

```python
q = iou(...)
```

然后直接把它作为 BCE target。

后者会让：

```text
classification loss
```

通过：

```text
q
```

改变 box prediction。

这不是官方 MAL 的优化图。

---

# 87. 第二个调试陷阱：只看 `target_score`

如果你只打印：

```text
q^gamma
```

会错误得出：

```text
MAL 更忽略低 IoU positives
```

必须同时打印：

```text
target_score
weight
final BCE
gradient
```

才能理解 MAL。

---

# 88. 第三个陷阱：把 matched positive 当作 target=1

MAL 不是：

```text
matched → 1
unmatched → 0
```

而是：

```text
matched correct class
→ q^gamma

unmatched / other class
→ 0
```

所以它是一种：

```text
quality-aware soft-label classification
```

---

# 89. 第四个陷阱：认为 q^gamma 是 loss weight

在 MAL 中：

```text
q^gamma
```

是：

```text
positive BCE target
```

不是：

```text
positive overall loss weight
```

Positive overall weight：

```text
1
```

这是必须牢牢记住的区别。

---

# 90. 第五个陷阱：认为 MAL 只关注 low-quality matches

MAL 不是一个：

```text
只为 IoU < 0.2 设计的特殊分支
```

它统一处理：

```text
low-quality
medium-quality
high-quality
negatives
```

只是相较 VFL，它特别修复了 low-quality matched positives 被过度削弱的问题。

---

# 91. 第六个陷阱：把 MAL 与 matcher cost 混为一谈

Matcher：

$$
C
=
\lambda_{cls}C_{cls}
+
\lambda_{bbox}C_{bbox}
+
\lambda_{giou}C_{giou}
$$

MAL：

```text
assignment 完成后
classification optimization
```

所以：

```text
改 MAL
```

不等于：

```text
改 Hungarian cost
```

DEIMv2 后续的 matcher schedule 是另一层优化，第 17/18 课再讲。

---

# 92. 第七个陷阱：认为 MAL 可以替代 box regression

不可以。

MAL 的 IoU 只是：

```text
quality label
```

真正让 box 位置变准的主要还是：

```text
L1
GIoU
FGL
localization distillation
```

---

# 93. MAL 与 confidence calibration

理想情况下：

```text
localization 好
→ classification score 高

localization 差
→ classification score 低
```

这对推理排序非常重要。

因为最终 detector 往往会依据：

```text
class score
```

选择 top predictions。

如果 classification confidence 与 localization quality 完全脱节：

```text
一个 IoU=0.1 的 box
可能排在 IoU=0.9 的 box 前面
```

quality-aware target 有助于改善这种 ranking。

---

# 94. 为什么 MAL 仍然是 classification loss？

虽然 target 使用 IoU，但优化变量仍然是：

```text
pred_logits
```

并且 IoU 被：

```text
detach
```

所以：

$$
\frac{\partial L_{\text{MAL}}}
{\partial \text{pred\_boxes}}
=0
$$

（只考虑 MAL 这条路径）

而：

$$
\frac{\partial L_{\text{MAL}}}
{\partial \text{pred\_logits}}
\neq0
$$

因此它严格属于 classification objective。

---

# 95. 从计算图角度看 MAL

```text
pred_boxes
   │
   ▼
  IoU
   │
detach
   │
   ▼
q^gamma ───────────┐
                   │
                   ▼
pred_logits ──→ BCEWithLogits ──→ loss_mal
     │
     └→ sigmoid → detach → p^gamma
                        │
                        └→ negative weight
```

backward：

```text
loss_mal
   │
   ▼
pred_logits

X pred_boxes
X target q
X dynamic weight path
```

这张计算图非常值得自己画一次。

---

# 96. 从整个 DEIM 训练系统看 MAL

```text
Images
  │
  ├─ Mosaic / MixUp
  │
  ▼
Dense Targets
  │
  ▼
Model
  │
  ├─ pred_logits
  └─ pred_boxes
         │
         ▼
Hungarian Matcher
         │
         ▼
      indices
         │
         ├─────────────┐
         │             │
         ▼             ▼
    matched IoU      box losses
         │
       detach
         │
         ▼
      q^gamma
         │
         ▼
        MAL
         │
         ▼
classification gradients
```

这就是：

```text
Dense O2O + MAL
```

的完整闭环。

---

# 97. 本课最重要的公式组

## IoU quality

$$
q=\operatorname{IoU}(\hat b,b)
$$

## VFL positive

$$
L_{\text{VFL}}^+
=
q\cdot\operatorname{BCE}(p,q)
$$

## MAL positive target

$$
t=q^\gamma
$$

## MAL positive

$$
L_{\text{MAL}}^+
=
\operatorname{BCE}(p,q^\gamma)
$$

## MAL negative

$$
L_{\text{MAL}}^-
=
p^\gamma
\operatorname{BCE}(p,0)
$$

## MAL positive logit gradient

$$
\frac{\partial L}{\partial z}
=
p-q^\gamma
$$

## VFL positive logit gradient

$$
\frac{\partial L}{\partial z}
=
q(p-q)
$$

---

# 98. 本课最重要的源码组

```python
# 1. matched IoU
ious = torch.diag(ious).detach()

# 2. full [B,Q,C] quality target
target_score = (
    target_score_o.unsqueeze(-1)
    * target
)

# 3. MAL quality transform
target_score = target_score.pow(
    self.gamma
)

# 4. detached prediction used only for weighting
pred_score = F.sigmoid(
    src_logits
).detach()

# 5. positive weight=1, negative weight=p^gamma
weight = (
    pred_score.pow(self.gamma)
    * (1 - target)
    + target
)

# 6. classification loss
loss = F.binary_cross_entropy_with_logits(
    src_logits,
    target_score,
    weight=weight,
    reduction='none'
)
```

如果你真正理解这六段，第 16 课的核心就掌握了。

---

# 99. 12 个必须掌握的结论

1. Hungarian matched 不代表 IoU 一定高。
2. Dense O2O 会增加 positive matches，也可能增加低质量 matches。
3. VFL positive 使用 `target=q`，同时整个 positive loss 还被 `q` 加权。
4. 低 q 时，VFL classification gradient 会被明显压小。
5. MAL positive 使用 `target=q^gamma`。
6. MAL matched-positive 的 overall weight 是 **1**，不是 `q` 或 `q^gamma`。
7. MAL negative weight 是 `p^gamma`，用于抑制 easy negatives。
8. `q^gamma` 是 soft target，不是 positive loss weight。
9. IoU 在 MAL 中 `detach()`，所以 MAL 不通过 quality target 训练 box branch。
10. `pred_score` 也 `detach()`，所以动态 negative weight 本身不建立额外梯度路径。
11. MAL 属于 criterion/classification optimization，不是 matcher。
12. DEIM 最核心的训练逻辑应整体理解为 **Dense O2O + MAL**。

---

# 100. 本课检查题

## A. 概念

1. 为什么 Hungarian matched Query 的 IoU 可以接近 0？
2. Matchability 在 MAL 中用什么量表示？
3. VFL 为什么会弱化 low-quality matched positive？
4. MAL 为什么不等于“给所有 matched positive target=1”？
5. 为什么 `q^gamma` 越低不代表 MAL 越忽略该 positive？
6. MAL 与 Dense O2O 为什么互补？
7. MAL 与 Hungarian Matcher 分别发生在哪个阶段？
8. MAL 为什么不能替代 L1/GIoU？

## B. 公式

9. 写出 VFL positive 的 BCE 形式。
10. 写出 MAL positive target。
11. 写出 MAL positive loss。
12. 写出 MAL negative weight。
13. 写出 MAL positive 对 logit 的梯度。
14. 写出 VFL positive 对 logit 的梯度。
15. 当 `q=.05, p=.5, gamma=1.5` 时，两者梯度为什么差很多？
16. gamma 增大时 `q^gamma` 怎样变化？

## C. Shape

17. `pred_logits` 在 COCO、B=2、Q=300 时是什么 shape？
18. 总 matched 数 M=24 时 `src_boxes` 是什么 shape？
19. `box_iou(src_boxes,target_boxes)` 为什么是 `[24,24]`？
20. `torch.diag()` 后是什么 shape？
21. `target_score_o.unsqueeze(-1)` 是什么 shape？
22. 最终 `target_score` 为什么与 logits 同 shape？

## D. 源码

23. 为什么 `ious.detach()`？
24. 为什么 `pred_score.detach()`？
25. `target` 与 `target_score` 有什么区别？
26. 为什么 positive weight 必然等于 1？
27. 为什么 unmatched Query 的 target_score 全是 0？
28. `mal_alpha` 影响 positive 还是 negative weighting？
29. `num_boxes` 为什么需要 distributed normalization？
30. MAL 是否使用 Global Union indices？

---

# 101. 答案速记

```text
1.
Hungarian 求当前候选中的全局最优一对一 assignment，
没有 IoU 必须超过某阈值的要求。

2.
matched predicted box 与 GT box 的 IoU q。

3.
positive target=q，
并且整个 positive BCE 又乘 q。

4.
MAL target=q^gamma，
仍然保留 localization-quality calibration。

5.
因为 positive overall weight=1；
q^gamma 是 target，不是 loss weight。

6.
Dense O2O 增加 matches；
MAL 让质量不同的 matches 都能得到合适的分类优化。

7.
Matcher 先决定 assignment；
MAL 在 assignment 后计算 classification loss。

8.
MAL 的 box-derived q 被 detach；
它主要优化 logits，不负责直接回归 box。

9.
q * BCE(p,q)

10.
q^gamma

11.
BCE(p,q^gamma)

12.
p^gamma

13.
p - q^gamma

14.
q(p-q)

15.
VFL 整体还被 q=.05 缩小；
MAL positive weight=1。

16.
0<q<1 时 gamma 越大，q^gamma 越小。

17.
[2,300,80]

18.
[24,4]

19.
box_iou 计算 pairwise matrix。

20.
[24]

21.
[B,Q,1]

22.
与 one-hot [B,Q,C] 广播相乘。

23.
阻止 MAL target 的梯度回传到 pred_boxes。

24.
让 p^gamma 只作为 detached dynamic weight。

25.
target 是 hard one-hot match mask；
target_score 是 q^gamma soft quality label。

26.
target=1：
p^gamma*(1-target)+target = 1。

27.
unmatched 的 one-hot real-class mask 全为 0。

28.
默认实现中只额外缩放 negative term。

29.
不同 GPU 的 GT 数可能不同，需要稳定 normalization。

30.
当前 criterion 中 MAL 使用 strict matcher indices；
boxes/local 才可使用 union indices。
```

---

# 102. 下一课预告

第 17 课：

# 《DEIMv2 Dense O2O v2——Mosaic + MixUp + CopyBlend》

我们会从训练数据本身进入 DEIMv2：

```text
原始 COCO sample
↓
Mosaic
↓
geometric transforms
↓
MixUp
↓
CopyBlend
↓
Dense target set
↓
matcher schedule
↓
Dense O2O v2
```

并真正推导：

```text
box 坐标怎样随 Mosaic 变化？
Resize 前后如何换算？
MixUp 怎样合并 targets？
CopyBlend 为什么与普通 copy-paste 不完全等价？
mixup_epochs=[4,29] 到底怎样控制训练阶段？
copyblend_epochs=[4,50] 的目的是什么？
DEIMv2 为什么还加入 matcher_change_epoch / iou_order_alpha？
```

这一课会把第 15、16 课的：

```text
Dense O2O
+
MAL
```

真正落到 DEIMv2 的数据管线和训练时间轴上。

---

# 103. 源码与论文核对范围

本课依据当前公开官方资料核对。

## DEIM 论文

**DEIM: DETR with Improved Matching for Fast Convergence**, CVPR 2025。

论文指出 Dense O2O 虽然增加有效训练匹配并加速收敛，但同时引入更多低质量匹配，因此提出 Matchability-Aware Loss。

论文对 VFL 的分析重点包括：

```text
低质量 match 的 loss 太小
q=0 的 match 会按 negative-like quality 处理
```

MAL 将 positive quality target 改为：

$$
q^\gamma
$$

并调整正负样本 weighting。

## DEIM 官方 Criterion

当前：

```text
engine/deim/deim_criterion.py
```

中真实实现：

```text
loss_labels_focal()
loss_labels_vfl()
loss_labels_mal()
loss_boxes()
loss_local()
```

`loss_labels_mal()` 明确执行：

```text
matched IoU
→ detach
→ one-hot quality map
→ q^gamma
→ positive weight 1
→ negative weight p^gamma
→ BCEWithLogits
```

## DEIM Base Config

当前：

```text
configs/base/deim.yml
```

配置：

```yaml
DEIMCriterion:
  weight_dict:
    loss_mal: 1
    loss_bbox: 5
    loss_giou: 2
    loss_fgl: 0.15
    loss_ddf: 1.5
  losses: ['mal', 'boxes', 'local']
  gamma: 1.5
```

---

# 104. 参考资料

1. **DEIM: DETR with Improved Matching for Fast Convergence**, CVPR 2025  
   https://openaccess.thecvf.com/content/CVPR2025/html/Huang_DEIM_DETR_with_Improved_Matching_for_Fast_Convergence_CVPR_2025_paper.html

2. DEIM paper PDF  
   https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_DEIM_DETR_with_Improved_Matching_for_Fast_Convergence_CVPR_2025_paper.pdf

3. DEIM official repository  
   https://github.com/Intellindust-AI-Lab/DEIM

4. DEIM Criterion — `loss_labels_mal()`  
   https://github.com/Intellindust-AI-Lab/DEIM/blob/main/engine/deim/deim_criterion.py

5. DEIM base configuration  
   https://github.com/Intellindust-AI-Lab/DEIM/blob/main/configs/base/deim.yml

---

## 版本提醒

DEIM / DEIMv2 仍可能更新。学习本地仓库时，请始终按以下优先级核对：

```text
1. 你本地实际使用的 merged YAML
2. 你本地 deim_criterion.py
3. runtime 中打印出的 target / weight / loss / gradient
4. 论文
5. 本讲义
```

特别建议你在第 16 课结束后亲自给 `loss_labels_mal()` 加一次临时 debug：

```python
print("iou:", ious[:10])
print("target:", target_score[idx][:10])
print("positive weight:", weight[idx][:10])
```

然后观察训练早期和训练后期的差异。

当你能从一个：

```text
IoU = 0.05
```

的 matched Query 出发，手工推到：

```text
q^1.5
→ target_score
→ weight=1
→ BCEWithLogits
→ dL/dz
```

并能解释它为什么不同于 VFL 时，MAL 就真正学懂了。
