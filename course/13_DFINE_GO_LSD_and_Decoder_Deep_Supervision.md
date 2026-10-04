# DEIMv2 源码课程 · 第 13 课
# D-FINE 的 GO-LSD 与 Decoder 多层监督

> 本课承接第 12 课 FDR。
>
> 上一课我们已经把 D-FINE 的定位路径拆到了最细：
>
> ```text
> Query feature
>   ↓
> 4 × 33 distribution logits
>   ↓
> Softmax
>   ↓
> non-uniform W(n)
>   ↓
> Integral
>   ↓
> l/t/r/b refinement
>   ↓
> distance2bbox
>   ↓
> refined box
> ```
>
> 现在出现一个自然问题：
>
> **6 层 Decoder 中，最后一层的定位分布通常经过最多次 refinement、包含最成熟的 localization knowledge。为什么只让最后一层自己用这些知识，而不把它反过来教给前面的浅层？**
>
> D-FINE 的答案就是：
>
> **GO-LSD — Global Optimal Localization Self-Distillation**
>
> 本课会把论文中的 GO-LSD、源码中的 `teacher_corners / teacher_logits / loss_ddf`、DETR 的 auxiliary supervision，以及 Hungarian Matching 的跨层关系完整串成一条训练链。
>
> 本课继续以 `deim_dfine_x` 为主要模型背景。关键配置包括：
>
> ```text
> num_layers = 6
> num_queries = 300
> reg_max = 32
> loss_fgl = 0.15
> loss_ddf = 1.5
> ```
>
> Criterion 中 DDF 的默认温度：
>
> ```text
> T = 5
> ```

---

# 1. 先建立整课的总地图

普通 DETR 多层 Decoder：

```text
Decoder Layer 0 ──→ prediction 0 ──→ auxiliary loss
Decoder Layer 1 ──→ prediction 1 ──→ auxiliary loss
Decoder Layer 2 ──→ prediction 2 ──→ auxiliary loss
Decoder Layer 3 ──→ prediction 3 ──→ auxiliary loss
Decoder Layer 4 ──→ prediction 4 ──→ auxiliary loss
Decoder Layer 5 ──→ prediction 5 ──→ final loss
```

D-FINE 仍然保留这些监督，但定位部分多出一条：

```text
                    Final Layer
                        │
                        │ teacher
                        ▼
               final localization
                 distributions
                        │
       ┌────────────────┼────────────────┐
       ▼                ▼                ▼
    Layer 0          Layer 1          Layer 4
    student          student          student
       │                │                │
       └────────── DDF / KL ─────────────┘
```

同时，论文中的 GO-LSD 并不是简单一句：

```text
“最后一层教前面几层”
```

它还处理：

```text
每层 Hungarian Matching 不完全相同
→ 构造跨层 localization 的 global union set
→ matched / unmatched predictions 使用不同权重
→ classification 仍保持 O2O
```

这几个点才构成完整 GO-LSD。

---

# 2. 为什么 DETR 本来就需要 Decoder 多层监督？

假设有 6 层 Decoder。

如果只监督最后一层：

```text
Layer 0
  ↓
Layer 1
  ↓
Layer 2
  ↓
Layer 3
  ↓
Layer 4
  ↓
Layer 5
  ↓
Loss
```

那么浅层只能通过很长的计算图间接获得梯度。

DETR 系列通常会给中间 Decoder 层增加：

> **Auxiliary Loss**

也就是：

```text
每一层都直接产生 class + box prediction
每一层都直接与 GT 计算 loss
```

这能让浅层更容易学习有效 Query 表示。

---

# 3. Auxiliary Loss ≠ Self-Distillation

这是本课第一处必须严格区分的概念。

### Auxiliary supervision

老师是：

```text
Ground Truth
```

例如：

```text
Layer 1 prediction
        ↓
Hungarian Matching
        ↓
GT
        ↓
classification / L1 / GIoU / FGL
```

### Self-distillation

老师是：

```text
同一个模型自己的更深层 prediction
```

例如：

```text
Final Layer distribution
        ↓ teacher
Shallow Layer distribution
        ↓ student
KL divergence
```

所以：

```text
Auxiliary Loss:
student ← GT

GO-LSD:
student ← final decoder layer
```

两者同时存在，而且解决的问题不同。

---

# 4. 为什么 GT 监督还不够？

你可能会问：

> 既然浅层已经直接看到了 GT，为什么还需要最后一层当 Teacher？

因为 FDR 的输出不是一个简单坐标，而是：

```text
33-bin localization distribution
```

GT 可以告诉网络：

```text
正确边界大约位于哪里
```

并通过 FGL 构造相邻 bins 的 target。

但最终层自己学到的 distribution 还包含更丰富的：

```text
soft probability structure
```

例如：

```text
bin 14: 0.04
bin 15: 0.18
bin 16: 0.46
bin 17: 0.24
bin 18: 0.06
...
```

这种 soft target 不只是：

```text
答案在哪
```

还包含：

```text
模型对附近位置的相对偏好和不确定性
```

Self-distillation 可以把这些知识传给浅层。

---

# 5. GO-LSD 的 Teacher 从哪里来？

答案非常直接：

> **最后一个 Decoder Layer 的 refined localization distributions。**

DEIM / D-FINE 源码在构造 auxiliary outputs 时，把最终层：

```text
out_corners[-1]
out_logits[-1]
```

作为：

```text
teacher_corners
teacher_logits
```

传给所有浅层。

核心结构可概括为：

```python
out["aux_outputs"] = self._set_aux_loss2(
    out_logits[:-1],
    out_bboxes[:-1],
    out_corners[:-1],
    out_refs[:-1],
    out_corners[-1],
    out_logits[-1],
)
```

注意：

```text
[:-1]
```

是 student layers；

```text
[-1]
```

是 teacher layer。

---

# 6. 对 `deim_dfine_x`，究竟有几个 Student？

X：

```text
num_layers = 6
```

索引：

```text
Layer 0
Layer 1
Layer 2
Layer 3
Layer 4
Layer 5
```

最终层：

```text
Layer 5
```

是主输出，也是 GO-LSD Teacher。

因此普通 auxiliary student layers：

```text
Layer 0
Layer 1
Layer 2
Layer 3
Layer 4
```

一共：

```text
5 个
```

所以你可以先建立最简单的图：

```text
                   Layer 5
                   Teacher
                     │
        ┌────────────┼─────────────┐
        │            │             │
        ▼            ▼             ▼
     Layer 0      Layer 1  ...   Layer 4
     Student      Student        Student
```

---

# 7. Teacher 的 Shape

上一课已经知道：

```text
reg_max = 32
```

所以：

```text
4 × (32+1)
=
132
```

最终层：

```text
teacher_corners
[B,Q,132]
```

普通 query 路线：

```text
Q = 300
```

因此：

```text
teacher_corners
[B,300,132]
```

逻辑 reshape：

```text
[B,300,4,33]
```

即：

```text
300 Queries
×
4 edges
×
33-bin distributions
```

---

# 8. Student 的 Shape

任意浅层：

```text
pred_corners
[B,300,132]
```

reshape：

```text
[B,300,4,33]
```

因此 Teacher 和 Student 可以逐 Query、逐边进行 distribution distillation。

---

# 9. `teacher_logits` 又是什么？

源码还把最终层分类输出：

```text
out_logits[-1]
```

传成：

```text
teacher_logits
```

shape：

```text
[B,Q,num_classes]
```

在 COCO：

```text
[B,300,80]
```

它不是用来做 localization KL 的 target distribution。

它主要用于：

> **给 unmatched predictions 的 localization distillation 提供 confidence weighting。**

这一点后面会详细推导。

---

# 10. `_set_aux_loss2()` 真正在做什么？

源码结构：

```python
def _set_aux_loss2(
    outputs_class,
    outputs_coord,
    outputs_corners,
    outputs_ref,
    teacher_corners=None,
    teacher_logits=None,
):
    return [
        {
            "pred_logits": a,
            "pred_boxes": b,
            "pred_corners": c,
            "ref_points": d,
            "teacher_corners": teacher_corners,
            "teacher_logits": teacher_logits,
        }
        for a, b, c, d in zip(...)
    ]
```

所以每个 auxiliary layer 的 dictionary 大概长这样：

```text
{
    pred_logits:      当前层 classification
    pred_boxes:       当前层 bbox
    pred_corners:     当前层 FDR distribution logits
    ref_points:       当前层 reference
    teacher_corners:  最终层 FDR distribution logits
    teacher_logits:   最终层 classification logits
}
```

这就是 GO-LSD 在模型输出接口上的实现入口。

---

# 11. 为什么 Teacher 不需要额外网络？

传统 Knowledge Distillation 常见：

```text
大 Teacher Model
       ↓
小 Student Model
```

需要：

```text
额外模型
额外 forward
额外显存
额外 checkpoint
```

GO-LSD：

```text
同一个 Decoder
```

里面天然已经有：

```text
浅层
深层
```

于是：

```text
Final Layer = Teacher
Earlier Layers = Students
```

这就是：

> **Self-Distillation**

Teacher 本来就会被计算，所以无需额外 teacher detector 做一次完整 forward。

---

# 12. 为什么最终层适合做 Teacher？

从 FDR 的 iterative refinement 角度：

```text
Layer 0 distribution
↓ residual refinement
Layer 1 distribution
↓ residual refinement
...
Layer 5 distribution
```

最终层经历最多轮：

```text
Query feature refinement
+
reference refinement
+
distribution refinement
```

论文的核心观察就是：

> 更深层的 refined distributions 通常包含更成熟、更精确的 localization knowledge。

因此让最终层指导浅层是自然选择。

---

# 13. Self-Distillation 到底 Distill 什么？

不是：

```text
feature map
```

也不是主要去蒸馏：

```text
最终 bbox 的 4 个坐标
```

而是：

> **FDR localization probability distributions**

即：

```text
left distribution
top distribution
right distribution
bottom distribution
```

每个：

```text
33 bins
```

因此蒸馏对象是：

```text
[B,Q,4,33]
```

的概率结构。

---

# 14. 为什么不只 Distill 最终 Box？

假设 Teacher 和 Student 最终 expectation 都是：

```text
0.10
```

但：

Teacher：

```text
[0.01, 0.08, 0.72, 0.17, 0.02]
```

Student：

```text
[0.20, 0.20, 0.20, 0.20, 0.20]
```

两者 expectation 可能接近。

但 Teacher 明显更确定。

如果只蒸馏：

```text
最终 scalar coordinate
```

这种结构信息就消失了。

蒸馏完整 distribution 能传递：

```text
peak
shape
uncertainty
neighbor relations
```

---

# 15. GO-LSD 的核心 Loss：KL Divergence

Teacher distribution：

\[
q(n)
\]

Student distribution：

\[
p(n)
\]

核心蒸馏项：

\[
D_{KL}(q\|p)
=
\sum_n
q(n)
\log
\frac{q(n)}{p(n)}
\]

源码使用：

```python
F.log_softmax(student / T)
F.softmax(teacher.detach() / T)
```

再交给：

```python
KLDivLoss
```

---

# 16. 为什么 Teacher 要 `.detach()`？

源码逻辑：

```python
F.softmax(
    target_corners.detach() / T,
    dim=1
)
```

意思是：

```text
Teacher distribution
只提供 target
```

不让当前 DDF loss 反过来为了迎合 Student 而修改 Teacher。

也就是：

```text
Teacher ─────→ Student
  target          learn

而不是：

Teacher ←────→ Student
双方互相迁就
```

这里的单向梯度非常重要。

---

# 17. Temperature T 是什么？

源码：

```text
T = 5
```

Student：

\[
p_T
=
softmax(z_s/T)
\]

Teacher：

\[
q_T
=
softmax(z_t/T)
\]

当：

```text
T > 1
```

logits 差异被缩小，distribution 会变得：

```text
更平滑
```

这样 Teacher 不只告诉 Student：

```text
“最大概率 bin 是谁”
```

还会更明显地暴露：

```text
其他 bins 的相对关系
```

这就是经典 knowledge distillation 中 temperature 的作用。

---

# 18. 一个简单 Temperature 例子

Teacher logits：

```text
[0, 1, 4, 1, 0]
```

T=1：

```text
Softmax
→ 非常集中在中间
```

T=5：

```text
Softmax
→ 更平滑
```

于是 Student 能看到：

```text
中心最好
邻近两个位置次之
远处更差
```

而不是接近 one-hot 的信息。

---

# 19. 为什么 Loss 乘 `T²`？

源码：

```text
(T ** 2) * KL(...)
```

这是知识蒸馏中的经典处理。

因为使用较大 T 会改变 softmax 对 logits 的梯度尺度。

乘：

\[
T^2
\]

用于补偿温度缩放导致的梯度幅度变化，使不同 T 下蒸馏梯度更合理。

---

# 20. DDF：为什么普通 KL 还不够？

如果直接：

```text
所有 Query
所有边
全部同权 KL
```

会有问题。

因为 DETR 中：

```text
300 Queries
```

绝大多数通常不是 GT 的 matched positive。

而且 matched query 的质量也不同。

因此：

```text
Query A:
IoU 很高

Query B:
IoU 很低

Query C:
unmatched 但 teacher confidence 很高

Query D:
unmatched 且 teacher confidence 很低
```

它们不应该拥有完全相同的 localization distillation 权重。

所以 D-FINE 提出：

> **DDF — Decoupled Distillation Focal Loss**

---

# 21. “Decoupled” 解耦什么？

主要把预测分成：

```text
Matched Predictions
```

和：

```text
Unmatched Predictions
```

然后采用不同质量指标：

```text
Matched:
用 IoU 权重

Unmatched:
用 Teacher Classification Confidence 权重
```

这就是最核心的 decoupled weighting。

---

# 22. 为什么 Matched Prediction 用 IoU？

一个 Query 如果通过 Hungarian Matching 对应某个 GT：

```text
Query ↔ GT
```

那么我们有明确目标框。

因此可以计算：

\[
IoU(
pred\_box,
GT
)
\]

IoU 越高：

```text
说明该 prediction 的 localization 更可靠
```

于是它的 localization distillation 可以获得更大权重。

---

# 23. 为什么不用 Classification Confidence 给 Matched Query 加权？

论文特别指出一个 DETR/O2O 现象：

```text
有些预测 localization 很好
但 classification confidence 还不高
```

如果 matched prediction 也用：

```text
classification confidence
```

作为蒸馏权重，就可能错误压低这些：

```text
高 IoU、低分类置信度
```

但 localization knowledge 很有价值的预测。

因此 matched 部分直接使用：

> **IoU**

而不是 classification confidence。

---

# 24. 为什么 Unmatched Prediction 用 Confidence？

对于 unmatched query：

```text
没有对应 GT
```

因此无法定义：

```text
该 Query 与哪个 GT 的 IoU
```

但 Teacher 仍可能认为：

```text
这个 Query 有较高 object/class confidence
```

所以源码使用：

```python
teacher_logits.sigmoid().max(dim=-1)[0]
```

得到：

```text
每个 Query 的最大类别 confidence
```

作为 unmatched distillation weight。

---

# 25. `teacher_logits.sigmoid().max(-1)` Shape 推导

输入：

```text
teacher_logits
[B,Q,C]
```

COCO：

```text
[B,300,80]
```

sigmoid：

```text
[B,300,80]
```

对类别维：

```python
.max(dim=-1)[0]
```

得到：

```text
[B,300]
```

每个 Query 一个：

```text
Teacher confidence
```

---

# 26. 为什么要 Repeat 4 次？

Localization distribution 有四个边：

```text
left
top
right
bottom
```

所以 Query-level weight：

```text
[B,Q]
```

要变成：

```text
[B,Q,4]
```

再 flatten：

```text
[B*Q*4]
```

与：

```text
pred_corners.reshape(-1,33)
```

一一对应。

源码：

```python
weight_targets_local = (
    weight_targets_local
    .unsqueeze(-1)
    .repeat(1, 1, 4)
    .reshape(-1)
)
```

---

# 27. Student / Teacher 在 Criterion 中怎样 Flatten？

原本：

```text
pred_corners
[B,Q,132]
```

reshape：

```python
pred_corners.reshape(
    -1,
    reg_max + 1
)
```

因为：

```text
reg_max + 1 = 33
```

所以：

```text
[B,Q,132]
→ [B*Q*4,33]
```

Teacher 同样：

```text
[B*Q*4,33]
```

于是每一行就是：

```text
某个 Query 的某一条边的 distribution
```

---

# 28. 用具体数字算 Shape

假设：

```text
B = 2
Q = 300
reg_max = 32
```

则：

```text
pred_corners
[2,300,132]
```

flatten：

\[
2\times300\times4
=
2400
\]

所以：

```text
Student
[2400,33]

Teacher
[2400,33]
```

KL：

```text
每一行 → 1 scalar
```

所以：

```text
loss_match_local
[2400]
```

之后再根据 matched / unmatched mask 分组。

---

# 29. Matched Mask 从哪里来？

Criterion 已经通过 Hungarian Matching 得到：

```text
indices
```

例如：

```text
Batch 0:
query [7, 19, 103]
↔
GT    [0,  1,   2]
```

然后：

```python
idx = self._get_src_permutation_idx(indices)
```

把 matched Query 索引组织成可用于 Tensor indexing 的形式。

---

# 30. Query-level Mask

先：

```python
mask = torch.zeros_like(
    weight_targets_local,
    dtype=torch.bool
)
```

shape：

```text
[B,Q]
```

然后：

```python
mask[idx] = True
```

于是：

```text
True
=
Hungarian matched query

False
=
unmatched query
```

---

# 31. 为什么 Mask 也 Repeat 4 次？

因为每个 Query 有：

```text
4 edge distributions
```

所以：

```text
[B,Q]
→ [B,Q,4]
→ [B*Q*4]
```

这样：

```text
一个 Query matched
```

就意味着它的：

```text
left
top
right
bottom
```

四个 distributions 都属于 matched group。

---

# 32. Matched Weight 如何替换成 IoU？

初始：

```text
weight_targets_local
=
teacher confidence
```

对 matched positions：

```python
weight_targets_local[idx] = ious
```

于是最终：

```text
matched query:
weight = IoU

unmatched query:
weight = teacher confidence
```

这正是 DDF 的 decoupled weighting。

---

# 33. DDF 的第一层公式

对任意 distribution：

\[
L_k
=
w_k
T^2
D_{KL}
\left(
q_k^T
\|
p_k^T
\right)
\]

其中：

```text
q = Teacher
p = Student
```

matched：

\[
w_k = IoU_k
\]

unmatched：

\[
w_k = Conf_k
\]

但这还没有结束。

DDF 还会处理：

> **Matched 和 Unmatched 数量极不平衡的问题。**

---

# 34. 为什么数量不平衡是严重问题？

假设一张图：

```text
300 Queries
```

只有：

```text
10 matched
```

那么：

```text
matched = 10
unmatched = 290
```

如果简单对所有 Query 求平均：

```text
unmatched
```

会在数量上占据绝对优势。

但如果只监督 matched：

```text
大量 unmatched Query 的 localization distribution
```

又完全没有得到 self-distillation。

GO-LSD 希望两者都利用，但需要平衡。

---

# 35. 论文里的数量平衡思想

设：

```text
K_m = matched 数量
K_u = unmatched 数量
```

论文为 matched / unmatched 设计了与：

\[
\sqrt{K_m}
\]

和：

\[
\sqrt{K_u}
\]

相关的组级权重。

直觉是：

```text
不让数量巨大的 unmatched group
按原始数量线性压倒 matched group
```

同时也不简单强制：

```text
两组完全 1:1
```

而使用平方根缓和数量差异。

---

# 36. 源码中的实现

源码计算：

```python
batch_scale = (
    8 / batch_size
)

num_pos = (
    mask.sum() * batch_scale
) ** 0.5

num_neg = (
    (~mask).sum() * batch_scale
) ** 0.5
```

然后分别：

```text
matched mean loss
unmatched mean loss
```

最终：

```python
loss_ddf = (
    loss_pos * num_pos
    +
    loss_neg * num_neg
) / (
    num_pos + num_neg
)
```

所以本质上是：

```text
sqrt(count)-weighted
group mean
```

---

# 37. 为什么是平方根？

假设：

```text
K_m = 10
K_u = 290
```

原始比例：

\[
290/10=29
\]

平方根：

\[
\sqrt{290}/\sqrt{10}
\approx5.39
\]

所以 unmatched 仍然因为数量多而有更大整体权重，但差距从：

```text
29 倍
```

压缩到：

```text
约 5.4 倍
```

这是一种温和的数量平衡。

---

# 38. `batch_scale = 8 / B` 是什么？

源码注释说明：

```text
Avoid the influence of batch size per GPU
```

如果每 GPU batch size 改变：

```text
mask.sum()
```

自然会随 batch size 大致线性改变。

因此乘：

```text
8 / B
```

把 count scale 归一到一个参考 batch 规模。

这不是 GO-LSD 理论公式最核心的部分，而是实现中的训练稳定性处理。

---

# 39. 为什么 Final Layer 自己不做 DDF？

最终层：

```text
pred_corners
```

和：

```text
teacher_corners
```

如果是同一个 Tensor，Teacher 和 Student 完全相同。

源码会检查：

```python
torch.equal(
    pred_corners,
    target_corners
)
```

如果相同：

```text
loss_ddf = 0
```

因此 DDF 的实际目标是：

```text
Teacher Final Layer
→ Earlier Layers
```

而不是让最终层蒸馏自己。

---

# 40. 到这里先总结 DDF

```text
Final Layer Distribution
         │
         │ detach
         ▼
       Teacher
         │
         ├──────── matched query
         │              │
         │              ▼
         │          weight = IoU
         │
         └──────── unmatched query
                        │
                        ▼
                weight = teacher confidence

Student Distribution
         │
         ▼
   KL divergence
      with T=5
         │
         ▼
 matched / unmatched
 sqrt-count balancing
         │
         ▼
      loss_ddf
```

---

# 41. 但这还不是完整 GO-LSD：为什么叫 “Global Optimal”？

如果只是：

```text
最后一层 Teacher
→ 浅层 Student
```

那只是普通 self-distillation。

“Global Optimal” 的关键来自：

> **不同 Decoder Layer 的 Hungarian Matching 可能选中不同 Query。**

例如同一个 GT：

```text
Layer 0:
GT A ↔ Query 17

Layer 1:
GT A ↔ Query 53

Layer 2:
GT A ↔ Query 53

Layer 3:
GT A ↔ Query 101

Layer 4:
GT A ↔ Query 101

Layer 5:
GT A ↔ Query 101
```

如果只看某一层自己的 match：

```text
它只知道一个局部 assignment。
```

GO-LSD 希望利用：

```text
所有层出现过的优质 localization candidates
```

形成更全局的定位监督集合。

---

# 42. Local Matching vs Global Union

设：

```text
M_0 = Layer 0 matched indices
M_1 = Layer 1 matched indices
...
M_5 = Layer 5 matched indices
```

GO-LSD 的 localization union：

\[
M_{global}
=
M_0
\cup
M_1
\cup
\cdots
\cup
M_5
\]

这就是论文所说的：

> unified union set

它让一个在某层表现出良好定位潜力、但在另一层未被选中的 Query，也有机会进入 localization distillation 的全局 matched 集合。

---

# 43. 为什么各层 Matching 会不同？

因为每层都有不同的：

```text
classification score
bbox
GIoU
```

Hungarian cost：

\[
C
=
\lambda_{cls}C_{cls}
+
\lambda_{L1}C_{L1}
+
\lambda_{GIoU}C_{GIoU}
\]

随着 Decoder refinement：

```text
predictions 在变化
```

因此最优 assignment 也可能变化。

这正是 DETR 深监督中的一个重要现象。

---

# 44. 一个极简例子

只有：

```text
3 Queries
1 GT
```

Layer 0：

```text
Q0 IoU=0.60
Q1 IoU=0.35
Q2 IoU=0.20

matched → Q0
```

Layer 1：

```text
Q0 IoU=0.55
Q1 IoU=0.75
Q2 IoU=0.25

matched → Q1
```

Layer 2：

```text
Q0 IoU=0.50
Q1 IoU=0.72
Q2 IoU=0.80

matched → Q2
```

如果只看最终层：

```text
matched set = {Q2}
```

而跨层 union：

```text
{Q0,Q1,Q2}
```

说明：

```text
三个 Query 都曾经在某一阶段成为该 GT 的最佳 assignment candidate。
```

GO-LSD 希望 localization knowledge 能覆盖这种跨层候选信息。

---

# 45. Classification 为什么不能也用 Union Set？

这是一个极其重要的问题。

如果分类也把：

```text
Q0
Q1
Q2
```

全部标成同一个 GT 的 positive：

```text
一个物体
→ 多个 positive object predictions
```

就会破坏 DETR：

> **One-to-One Set Prediction**

最终可能产生冗余框。

所以论文明确区分：

```text
Localization:
可以利用跨层 union set

Classification:
仍然严格 O2O matching
```

这是 GO-LSD 设计的关键边界。

---

# 46. Localization 与 Classification 被“部分解耦”

你可以这样理解：

```text
Classification:
谁负责这个 GT？
→ 必须唯一

Localization knowledge:
哪些 Query 曾经展示出好的定位能力？
→ 可以更广泛地学习
```

因此：

```text
O2O identity
```

仍由 classification 保持；

而：

```text
localization distribution learning
```

可以利用更丰富的候选。

---

# 47. 这和第 6 课 O2O 的关系

第 6 课你学到：

```text
一个 GT
↔
一个 Query
```

这是 DETR end-to-end、无需 NMS 的重要基础。

GO-LSD 并没有把推理改成：

```text
one-to-many detector
```

它做的是：

```text
训练期 localization knowledge transfer 更密集
```

但最终：

```text
classification assignment
```

仍保持 O2O。

所以不要误解成：

```text
GO-LSD 把 DETR 改成 O2M
```

---

# 48. 论文概念和源码应该怎样对应？

这里需要非常认真地区分两层理解。

### 论文层

GO-LSD 描述：

```text
每层 Hungarian Matching
↓
跨层 union set
↓
global localization optimization
↓
matched / unmatched DDF
```

### 源码层

你实际看到：

```text
teacher_corners
teacher_logits
indices
mask
IoU weighting
confidence weighting
sqrt-count balancing
KL
```

同时 Criterion 的 auxiliary-loss 调用和 matcher/index 聚合逻辑共同决定：

```text
当前 loss 使用哪些 localization matches。
```

读源码时不要只搜一个名为：

```text
GO_LSD()
```

的函数。

它并不存在为一个单独大模块。

GO-LSD 是由：

```text
Decoder output packaging
+
multi-layer matching
+
criterion indices handling
+
DDF weighting
+
KL distillation
```

共同实现的训练机制。

---

# 49. Decoder 多层输出到底有哪些？

训练模式下，主输出：

```text
pred_logits
pred_boxes
pred_corners
ref_points
up
reg_scale
```

此外：

```text
aux_outputs
enc_aux_outputs
pre_outputs
```

如果启用 denoising：

```text
dn_outputs
dn_pre_outputs
dn_meta
```

所以：

```text
“Decoder 多层监督”
```

实际上只是整个深监督系统的一部分。

---

# 50. `aux_outputs` 是什么？

对于 6 层 Decoder：

```text
Final:
Layer 5
```

主输出。

而：

```text
aux_outputs
```

通常包含：

```text
Layer 0
Layer 1
Layer 2
Layer 3
Layer 4
```

每个 dictionary 都可以计算：

```text
classification loss
bbox L1
GIoU
FGL
DDF
```

具体哪些 loss 被启用由 Criterion 的：

```text
losses
weight_dict
```

控制。

---

# 51. DEIM 的 Loss 配置

DEIM 的 D-FINE 路线配置：

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
```

本课最重要的是：

```text
loss_fgl = 0.15
loss_ddf = 1.5
```

其中：

```text
FGL:
GT → distribution

DDF:
Final Layer → shallow distribution
```

---

# 52. FGL 和 DDF 是互补关系

可以画成：

```text
             Ground Truth
                 │
                 │ FGL
                 ▼
         Student Distribution
                 ▲
                 │ DDF
                 │
       Final Layer Distribution
              Teacher
```

Student 同时受到：

```text
hard/geometry-aware target
+
soft teacher target
```

这就是 D-FINE localization training 的核心组合。

---

# 53. FGL 与 DDF 的数学角色

FGL：

\[
L_{FGL}
\]

要求：

```text
distribution 的期望和形状
朝 GT 所在的 bins 靠近
```

DDF：

\[
L_{DDF}
\]

要求：

```text
浅层 distribution
逼近最终层 refined distribution
```

总 localization loss 中还包括：

\[
L_{bbox}
\]

和：

\[
L_{giou}
\]

因此可以理解为：

\[
L_{loc}
=
\lambda_1L_{bbox}
+
\lambda_2L_{giou}
+
\lambda_3L_{FGL}
+
\lambda_4L_{DDF}
\]

DEIM 还会与其 classification/MAL 等 loss 一起构成总训练目标。

---

# 54. 为什么 `loss_ddf` 权重比 `loss_fgl` 大很多？

配置：

```text
loss_fgl = 0.15
loss_ddf = 1.5
```

不能简单得出：

```text
“DDF 比 FGL 重要 10 倍”
```

因为不同 loss：

```text
数值尺度
归一化方式
统计范围
```

都不同。

Loss weight 的意义是：

```text
调节最终梯度贡献尺度
```

不能仅根据系数大小比较概念重要性。

---

# 55. 为什么 DDF 对 Unmatched Query 也计算？

这是 GO-LSD 很有意思的一点。

传统 DETR：

```text
unmatched query
```

主要在分类上学习：

```text
no-object / background
```

但 D-FINE 发现：

```text
即使一个 Query 当前没有被 Hungarian Matching 选中
```

它的 localization distribution 仍可以从最终层获得有意义的 soft guidance。

因此 GO-LSD：

```text
matched → distill
unmatched → 也 distill
```

只是两者权重机制不同。

---

# 56. Unmatched Localization 为什么不会直接破坏 O2O？

因为：

```text
DDF 对 unmatched query
```

是在做：

```text
localization distribution alignment
```

不是把它的 classification label 改成某个 GT positive。

所以 classification 仍然保持：

```text
严格 O2O
```

这就是为什么可以：

```text
利用更多 localization knowledge
```

而不必在推理时引入多个 positive detections。

---

# 57. GO-LSD 为什么可以称为“Bidirectional Optimization”？

论文对 GO-LSD 的整体描述包含两个方向的收益：

### 深 → 浅

```text
Final refined distribution
→ distill to shallow layers
```

浅层获得更成熟 localization knowledge。

### 浅 → 深 / progressive residual optimization

FDR 本身是 residual distribution refinement：

```text
shallow distribution
→ deeper layer residual correction
→ final distribution
```

因此整个系统形成：

```text
forward:
浅层状态 → 深层 refinement

training knowledge:
深层 Teacher → 浅层 distillation
```

可以把它理解成训练信息上的双向协同。

---

# 58. FDR + GO-LSD 为什么天然搭配？

如果没有 FDR，Decoder 每层只输出：

```text
4 个 bbox coordinates
```

Teacher 能提供的信息相对有限。

有了 FDR：

```text
4 × 33 distributions
```

以后，每一层都存在丰富的 soft localization representation。

于是：

```text
FDR
提供可蒸馏的“知识载体”

GO-LSD
负责跨层传递这种知识
```

两者不是两个完全独立的小技巧。

---

# 59. 一个关键闭环

```text
Layer 0 distribution
      │
      ▼
residual refinement
      │
Layer 1 distribution
      │
      ▼
...
      │
Final distribution
      │
      └──────────────┐
                     │
                 Teacher
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
      shallow 0             shallow 4
          │                     │
          └────── DDF ──────────┘
```

Forward 方向：

```text
浅 → 深
```

Knowledge distillation：

```text
深 → 浅
```

---

# 60. PyTorch 实验 1：Temperature 如何改变 Distribution

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([
    0.0,
    1.0,
    4.0,
    1.0,
    0.0
])

for T in [1.0, 2.0, 5.0, 10.0]:
    p = F.softmax(
        logits / T,
        dim=-1
    )

    print(
        f"T={T}:",
        p
    )
```

观察：

```text
T 越大
→ distribution 越平滑
```

---

# 61. PyTorch 实验 2：实现最小 Distillation KL

```python
def distill_kl(
    student_logits,
    teacher_logits,
    T=5.0
):
    student_log_prob = F.log_softmax(
        student_logits / T,
        dim=-1
    )

    teacher_prob = F.softmax(
        teacher_logits.detach() / T,
        dim=-1
    )

    loss = F.kl_div(
        student_log_prob,
        teacher_prob,
        reduction="batchmean"
    )

    return loss * (T ** 2)
```

测试：

```python
student = torch.randn(
    100,
    33,
    requires_grad=True
)

teacher = torch.randn(
    100,
    33,
    requires_grad=True
)

loss = distill_kl(
    student,
    teacher
)

loss.backward()

print(
    student.grad is not None
)

print(
    teacher.grad
)
```

预期：

```text
student.grad != None
teacher.grad == None
```

因为：

```text
teacher.detach()
```

---

# 62. PyTorch 实验 3：验证 Teacher 越接近 Student，KL 越小

```python
teacher = torch.randn(
    32,
    33
)

student_far = torch.randn(
    32,
    33
)

student_close = (
    teacher
    + 0.05 * torch.randn_like(teacher)
)

loss_far = distill_kl(
    student_far,
    teacher
)

loss_close = distill_kl(
    student_close,
    teacher
)

print(
    "far:",
    loss_far.item()
)

print(
    "close:",
    loss_close.item()
)
```

通常：

```text
loss_close < loss_far
```

---

# 63. PyTorch 实验 4：模拟 `[B,Q,132]` 的 DDF

```python
B = 2
Q = 300
REG_MAX = 32

student = torch.randn(
    B,Q,4*(REG_MAX+1),
    requires_grad=True
)

teacher = torch.randn(
    B,Q,4*(REG_MAX+1)
)

student_flat = student.reshape(
    -1,
    REG_MAX+1
)

teacher_flat = teacher.reshape(
    -1,
    REG_MAX+1
)

print(
    student_flat.shape
)
```

输出：

```text
[2400,33]
```

然后：

```python
T = 5.0

per_edge_kl = (
    F.kl_div(
        F.log_softmax(
            student_flat/T,
            dim=-1
        ),
        F.softmax(
            teacher_flat.detach()/T,
            dim=-1
        ),
        reduction="none"
    ).sum(dim=-1)
    * T**2
)

print(
    per_edge_kl.shape
)
```

输出：

```text
[2400]
```

这已经非常接近源码 DDF 的核心。

---

# 64. PyTorch 实验 5：构造 Matched Mask

```python
B = 2
Q = 300

mask = torch.zeros(
    B,Q,
    dtype=torch.bool
)

# 假设 batch 0 有 5 个 matched queries
mask[0, [3, 7, 20, 81, 150]] = True

# batch 1 有 3 个
mask[1, [4, 99, 210]] = True

print(
    "matched queries:",
    mask.sum().item()
)

edge_mask = (
    mask
    .unsqueeze(-1)
    .repeat(1,1,4)
    .reshape(-1)
)

print(
    edge_mask.shape
)

print(
    "matched edge distributions:",
    edge_mask.sum().item()
)
```

如果 matched Query 总数是 8：

```text
matched edge distributions
=
8 × 4
=
32
```

---

# 65. PyTorch 实验 6：Matched 用 IoU，Unmatched 用 Confidence

```python
teacher_conf = torch.rand(
    B,Q
)

weights = teacher_conf.clone()

matched_ious = torch.tensor([
    0.90,
    0.75,
    0.63,
    0.88,
    0.55,
    0.72,
    0.81,
    0.69
])

weights[mask] = matched_ious

edge_weights = (
    weights
    .unsqueeze(-1)
    .repeat(1,1,4)
    .reshape(-1)
)

weighted_kl = (
    per_edge_kl
    * edge_weights
)

print(
    weighted_kl.shape
)
```

---

# 66. PyTorch 实验 7：Matched / Unmatched Group Balance

```python
pos_loss = (
    weighted_kl[edge_mask].mean()
)

neg_loss = (
    weighted_kl[~edge_mask].mean()
)

num_pos = (
    edge_mask.sum().float()
).sqrt()

num_neg = (
    (~edge_mask).sum().float()
).sqrt()

loss_ddf = (
    pos_loss * num_pos
    +
    neg_loss * num_neg
) / (
    num_pos + num_neg
)

print(
    "pos loss:",
    pos_loss.item()
)

print(
    "neg loss:",
    neg_loss.item()
)

print(
    "DDF:",
    loss_ddf.item()
)
```

这就是源码逻辑的最小版。

---

# 67. PyTorch 实验 8：为什么不能直接所有 Query 求 Mean？

比较：

```python
naive = weighted_kl.mean()

balanced = loss_ddf

print(
    "naive:",
    naive.item()
)

print(
    "balanced:",
    balanced.item()
)
```

然后改变：

```text
matched Query 数量
```

观察两种 loss 对 Query 数量分布的敏感性。

这能帮助你理解：

```text
DDF 不只是 KL
```

而是：

```text
KL
+
quality weighting
+
group balancing
```

---

# 68. PyTorch 实验 9：模拟 6 层 Self-Distillation

```python
B = 1
Q = 10
DISTR_DIM = 132

layers = [
    torch.randn(
        B,Q,DISTR_DIM,
        requires_grad=True
    )
    for _ in range(6)
]

teacher = layers[-1]

loss = 0.0

for student in layers[:-1]:
    loss = loss + distill_kl(
        student.reshape(-1,33),
        teacher.reshape(-1,33),
        T=5.0
    )

loss.backward()

for i, x in enumerate(layers):
    grad_norm = (
        None
        if x.grad is None
        else x.grad.norm().item()
    )

    print(
        i,
        grad_norm
    )
```

因为 teacher 被 detach：

```text
Layer 0~4
从这个 toy DDF 得到梯度

Layer 5
不会通过 DDF teacher path 得到梯度
```

真实训练中 Layer 5 当然还会通过：

```text
自己的 detection losses
```

获得梯度。

---

# 69. PyTorch 实验 10：模拟跨层 Matching Union

```python
layer_matches = [
    {3, 7, 20},
    {7, 20, 31},
    {7, 31, 42},
    {31, 42, 51},
    {42, 51, 60},
    {51, 60, 72},
]

global_union = set().union(
    *layer_matches
)

print(
    global_union
)

print(
    "local final:",
    len(layer_matches[-1])
)

print(
    "global union:",
    len(global_union)
)
```

这个 toy example 展示：

```text
final layer local matches
```

与：

```text
all-layer localization candidate union
```

为什么不是同一个概念。

---

# 70. 一个完整的 GO-LSD Shape Trace

假设：

```text
B=2
Q=300
C=80
reg_max=32
num_layers=6
```

最终 Teacher：

```text
teacher_corners
[2,300,132]

teacher_logits
[2,300,80]
```

某 Student：

```text
pred_corners
[2,300,132]
```

reshape：

```text
Student:
[2400,33]

Teacher:
[2400,33]
```

confidence：

```text
teacher_logits
[2,300,80]

↓ sigmoid + max class

[2,300]

↓ repeat 4

[2400]
```

mask：

```text
[2,300]

↓ repeat 4

[2400]
```

KL：

```text
[2400,33]
→ sum over bins
→ [2400]
```

最终：

```text
matched weighted mean
+
unmatched weighted mean
→ scalar loss_ddf
```

---

# 71. Decoder 多层监督的总 Loss 树

可以把训练 loss 想成：

```text
Total Loss
│
├── Final Decoder Layer
│   ├── classification / MAL
│   ├── L1
│   ├── GIoU
│   └── FGL
│
├── Aux Layer 0
│   ├── classification / MAL
│   ├── L1
│   ├── GIoU
│   ├── FGL
│   └── DDF ← Final Teacher
│
├── Aux Layer 1
│   ├── ...
│   └── DDF ← Final Teacher
│
├── ...
│
├── Aux Layer 4
│   ├── ...
│   └── DDF ← Final Teacher
│
├── Encoder auxiliary outputs
│
├── Preliminary outputs
│
└── Denoising outputs
    ├── DN detection losses
    └── DN localization/distillation path
```

后面的课程会继续拆 DN 和整个训练系统。

---

# 72. `enc_aux_outputs` 和 GO-LSD 是一回事吗？

不是。

Encoder auxiliary outputs：

```text
enc_topk_logits
enc_topk_bboxes
```

主要用于：

```text
encoder proposal supervision
```

而 GO-LSD 的核心 Teacher/Student distribution：

```text
teacher_corners
pred_corners
```

来自 Decoder 的 FDR localization distributions。

不要把所有：

```text
auxiliary outputs
```

都叫 self-distillation。

---

# 73. `pre_outputs` 又是什么？

第 12 课看到第一层先产生：

```text
pre_bboxes
pre_scores
```

它们被包装成：

```text
pre_outputs
```

用于对：

```text
FDR 前的 preliminary prediction
```

提供监督。

所以 D-FINE 的训练非常“深”：

```text
encoder proposal
↓
preliminary box
↓
每层 refined box/distribution
↓
final distribution teacher
```

多个阶段都有直接训练信号。

---

# 74. Denoising Outputs 也使用 Teacher

源码：

```python
out["dn_outputs"] = self._set_aux_loss2(
    dn_out_logits,
    dn_out_bboxes,
    dn_out_corners,
    dn_out_refs,
    dn_out_corners[-1],
    dn_out_logits[-1],
)
```

所以 DN 分支也把：

```text
DN final layer
```

作为 DN shallow layers 的 teacher。

注意这里 Teacher 是：

```text
DN 自己的 final output
```

而不是普通 matching query 的 final output。

---

# 75. 为什么 DN 和普通 Query 要分开？

训练时 Query 序列中可能同时有：

```text
denoising queries
+
matching queries
```

两者：

```text
来源不同
监督方式不同
assignment 方式不同
```

所以 Decoder 后：

```python
torch.split(...)
```

把它们拆开。

然后：

```text
normal outputs
```

和：

```text
dn outputs
```

分别组织 loss。

---

# 76. `loss_ddf` 在 DN 分支有什么特殊点？

Criterion 中会判断：

```text
"is_dn" in outputs
```

并使用相应的：

```text
fgl_targets_dn
```

以及不同的 bookkeeping。

源码中的：

```text
num_pos / num_neg
```

统计也主要在普通非-DN路径上更新，以避免不同分支反复改变组权重统计。

这里暂时先建立概念；完整 DN 机制在后面的 Decoder/训练课程继续展开。

---

# 77. 为什么 DDF 使用 Distribution KL，而不是 Feature MSE？

传统 feature distillation：

```text
Teacher feature
Student feature
↓
MSE
```

在 DETR 中会遇到：

```text
不同层 Query identity / matching 不稳定
```

的问题。

D-FINE 选择：

```text
localization distribution
```

作为知识载体，因为它直接对应：

```text
检测任务真正需要优化的边界位置
```

而不是要求不同深度的 hidden feature 必须数值相同。

---

# 78. 为什么不直接做 Classification Logit Distillation？

GO-LSD 的目标是：

```text
Localization Self-Distillation
```

而 DETR classification 必须严格维护：

```text
O2O assignment
```

如果无差别把跨层 classification soft targets 传播给大量候选，可能与：

```text
唯一 positive assignment
```

产生冲突。

因此设计重点放在：

```text
localization distributions
```

而分类主要作为：

```text
unmatched localization weight
```

的一种可靠性信号。

---

# 79. “Global Optimal” 不等于“知道真正全局最优 Query”

这个名称不要过度字面化。

GO-LSD 并不是在训练中神奇地知道：

```text
所有可能预测里的真实数学全局最优解
```

它的“Global”主要体现在：

```text
不局限于单一 Decoder Layer 的 local matching
```

而利用：

```text
跨层 matching information
```

构建更全局的 localization optimization / distillation 集合。

---

# 80. 为什么 Teacher 本身也可能错？

Self-distillation 的 Teacher：

```text
不是 GT
```

而是模型自己的 final layer。

因此它当然可能：

```text
定位错误
分类错误
distribution 不够好
```

这也是为什么不能简单：

```text
所有 Teacher predictions 等权强制 Student 模仿
```

DDF 使用：

```text
matched → IoU
unmatched → confidence
```

做质量加权，本质上就是在控制：

```text
Teacher knowledge 的可信程度。
```

---

# 81. 为什么 Matched 使用 Student IoU 还是 Teacher IoU？

读源码时要非常小心“当前 outputs”是谁。

Criterion 对某个 auxiliary output 计算 loss 时：

```text
outputs["pred_boxes"]
```

就是：

```text
当前被监督的 Student layer boxes
```

源码计算：

```text
IoU(
outputs["pred_boxes"][idx],
target_boxes
)
```

因此 matched DDF 权重与：

```text
当前 Student prediction 的 localization quality
```

直接相关。

Teacher distribution 提供 soft target；

当前 matched Student 的 IoU 提供质量权重。

---

# 82. 一个具体 DDF 数字例子

假设某 matched edge：

```text
KL = 0.08
IoU = 0.90
T = 5
```

温度补偿后：

\[
T^2 = 25
\]

该项：

\[
0.90\times25\times0.08
=
1.8
\]

另一个 unmatched edge：

```text
KL = 0.08
Teacher confidence = 0.15
```

则：

\[
0.15\times25\times0.08
=
0.3
\]

同样 KL 差异下：

```text
高质量 matched localization
```

获得明显更大权重。

---

# 83. 如果 Unmatched Teacher Confidence 很高呢？

假设：

```text
confidence = 0.85
```

则：

\[
0.85\times25\times0.08
=
1.7
\]

这意味着：

```text
虽然它没被当前 Hungarian assignment 选中
```

但 Teacher 对其 object/class prediction 很有信心，所以其 localization distribution 仍值得 Student 学习。

这正体现了 GO-LSD：

```text
不浪费 unmatched localization knowledge
```

的思想。

---

# 84. 为什么这对 O2O DETR 尤其有意义？

O2O 的一个结构特点：

```text
GT 数量远小于 Query 数量
```

例如：

```text
300 Queries
10 GT
```

单层严格 matching：

```text
只有 10 个 matched positives
```

如果 localization distillation 只利用 matched：

```text
290 个 Query 的 distribution knowledge
```

完全被浪费。

GO-LSD 对 unmatched 也进行 weighted distillation，因此可以利用更密集的内部监督信号。

---

# 85. 但 GO-LSD 和 DEIM Dense O2O 不是同一个东西

非常重要。

### GO-LSD

来自 D-FINE：

```text
跨 Decoder layer localization self-distillation
```

重点：

```text
FDR distribution
teacher/student
DDF
```

### DEIM Dense O2O

来自 DEIM：

```text
通过训练数据/匹配机制提升 O2O supervision density
```

重点：

```text
matching
MAL
augmentation
positive supervision
```

后面第 15–17 课会专门拆 DEIM。

两者可以共同存在于：

```text
deim_dfine_x
```

但不能混成一个概念。

---

# 86. `deim_dfine_x` 为什么同时有 FDR、GO-LSD 和 MAL？

因为它本质上是：

```text
D-FINE detector architecture/localization system
+
DEIM training framework
```

所以：

```text
FDR
GO-LSD
```

解决：

```text
bbox localization representation / decoder optimization
```

而：

```text
Dense O2O
MAL
```

解决：

```text
DETR matching / supervision efficiency
```

这是理解 `deim_dfine_x` 最重要的“大分工”。

---

# 87. 一张总架构图

```text
Image
  ↓
HGNetv2
  ↓
HybridEncoder
  ↓
DFINETransformer
  │
  ├─ Layer 0
  │    ├ class
  │    ├ box
  │    └ FDR distribution ─┐
  │                        │ student
  ├─ Layer 1               │
  │    └ FDR distribution ─┤
  │                        │
  ├─ ...                   │
  │                        │
  └─ Layer 5               │
       ├ class             │
       ├ final box         │
       └ final FDR ────────┘
              │
              │ Teacher
              ▼
            GO-LSD
              │
              ▼
          loss_ddf

Meanwhile:

GT
│
├ Hungarian Matching
├ classification/MAL
├ L1
├ GIoU
└ FGL

DEIM training:
Dense O2O + MAL + augmentation recipe
```

---

# 88. 源码阅读：第一站 `DFINETransformer.forward`

你应该找到训练输出：

```python
out = {
    "pred_logits": out_logits[-1],
    "pred_boxes": out_bboxes[-1],
    "pred_corners": out_corners[-1],
    "ref_points": out_refs[-1],
    "up": self.up,
    "reg_scale": self.reg_scale,
}
```

先确认：

```text
主输出就是最后 Decoder layer。
```

---

# 89. 源码阅读：第二站 `aux_outputs`

继续看：

```python
out["aux_outputs"] = self._set_aux_loss2(
    out_logits[:-1],
    out_bboxes[:-1],
    out_corners[:-1],
    out_refs[:-1],
    out_corners[-1],
    out_logits[-1],
)
```

请自己回答：

```text
谁是 Student？
谁是 Teacher？
```

答案：

```text
Student:
[:-1]

Teacher:
[-1]
```

---

# 90. 源码阅读：第三站 `_set_aux_loss2`

重点不是 Python list comprehension。

重点是：

```text
每个 Student dictionary
都持有同一个 final teacher_corners
和 teacher_logits
```

所以：

```text
5 个 shallow layers
```

共享：

```text
Layer 5 Teacher
```

---

# 91. 源码阅读：第四站 `DEIMCriterion.loss_local`

找到：

```python
if "teacher_corners" in outputs:
```

这就是：

```text
FGL
```

切换到：

```text
DDF self-distillation
```

的入口。

---

# 92. 源码阅读：第五站 Student / Teacher Reshape

重点观察：

```python
pred_corners = outputs[
    "pred_corners"
].reshape(
    -1,
    self.reg_max + 1
)

target_corners = outputs[
    "teacher_corners"
].reshape(
    -1,
    self.reg_max + 1
)
```

这两行把：

```text
[B,Q,132]
```

变成：

```text
[B*Q*4,33]
```

---

# 93. 源码阅读：第六站 Confidence Weight

```python
weight_targets_local = (
    outputs["teacher_logits"]
    .sigmoid()
    .max(dim=-1)[0]
)
```

请在调试器打印：

```text
shape
min
max
mean
```

理解：

```text
unmatched distillation quality weight
```

究竟是什么分布。

---

# 94. 源码阅读：第七站 Mask

```python
mask = torch.zeros_like(
    weight_targets_local,
    dtype=torch.bool
)

mask[idx] = True
```

这里把：

```text
Hungarian matching result
```

转换成：

```text
matched/unmatched binary partition
```

---

# 95. 源码阅读：第八站 IoU Override

```python
weight_targets_local[idx] = ious
```

这是一行非常短、但论文意义很大的代码。

它完成：

```text
unmatched → confidence

matched → IoU
```

的 decoupling。

---

# 96. 源码阅读：第九站 KL

核心：

```python
F.log_softmax(
    pred_corners / T,
    dim=1
)

F.softmax(
    target_corners.detach() / T,
    dim=1
)
```

你应该立刻能说出：

```text
Student:
log probability

Teacher:
soft probability

Teacher:
detach

Temperature:
T=5
```

---

# 97. 源码阅读：第十站 Group Balance

找到：

```text
num_pos
num_neg

loss_match_local1
loss_match_local2
```

并画出：

```text
all edge KL
   │
   ├ matched
   │
   └ unmatched
        ↓
separate mean
        ↓
sqrt-count weighted combine
```

---

# 98. 推荐断点 1：`_set_aux_loss2`

打印：

```python
print(
    len(out["aux_outputs"])
)
```

X 预期：

```text
5
```

再打印：

```text
aux_outputs[0]["pred_corners"].shape
aux_outputs[0]["teacher_corners"].shape
```

---

# 99. 推荐断点 2：检查所有 Student 的 Teacher 是否相同

可以比较：

```python
torch.equal(
    aux_outputs[0]["teacher_corners"],
    aux_outputs[4]["teacher_corners"]
)
```

应该为：

```text
True
```

因为它们都引用：

```text
final layer distribution
```

---

# 100. 推荐断点 3：检查 Teacher Detach

在 Criterion 里：

```text
teacher_corners
```

本身可能仍带计算图。

真正做 KL 时：

```text
teacher_corners.detach()
```

切断梯度。

所以不要因为 output dictionary 中 teacher Tensor：

```text
requires_grad=True
```

就误以为 DDF 会反向优化 Teacher。

关键是：

```text
使用时 detach。
```

---

# 101. 推荐断点 4：统计 Matched / Unmatched

打印：

```python
print(
    mask.sum().item(),
    (~mask).sum().item()
)
```

再考虑 repeat 4 后：

```text
数量都 ×4
```

你会直观看到 DETR 中：

```text
unmatched 数量通常远大于 matched
```

为什么 DDF 需要 group balancing。

---

# 102. 推荐断点 5：比较 Matched IoU 与 Teacher Confidence

打印：

```text
matched IoU mean
unmatched teacher confidence mean
```

不要期待固定数值。

重点是理解：

```text
两组权重来自完全不同的 quality signal。
```

---

# 103. 推荐断点 6：逐层看 DDF

训练一个 batch 时记录：

```text
loss_ddf_aux_0
loss_ddf_aux_1
...
loss_ddf_aux_4
```

具体 key 名可能带 Criterion 的 suffix。

观察：

```text
浅层离 Teacher 是否更远？
越深层是否通常 KL 更小？
```

这不是数学上强制保证，但通常是非常有价值的调试观察。

---

# 104. 推荐断点 7：直接看一个 Query 的 Distribution

选择一个 matched Query：

```text
q
```

打印：

```text
Layer 0 left distribution
Layer 1 left distribution
...
Layer 5 left distribution
```

可以输出：

```text
Top-3 bin indices
Top-3 probabilities
Expectation
Entropy
```

这样 GO-LSD 会从抽象概念变成真正可观察的 Tensor。

---

# 105. 小实验：Distribution Entropy Across Layers

```python
def entropy_from_logits(
    logits
):
    p = logits.softmax(-1)

    return -(
        p
        * torch.log(
            p + 1e-9
        )
    ).sum(-1)

# corners:
# [num_layers, B, Q, 4, 33]

for layer in range(6):
    h = entropy_from_logits(
        corners[layer]
    )

    print(
        layer,
        h.mean().item()
    )
```

不要机械认为：

```text
越深 entropy 必须越小。
```

但这个统计可以帮助你研究：

```text
distribution confidence 怎样随 refinement 演化。
```

---

# 106. 小实验：KL to Final Teacher Across Layers

```python
teacher = corners[-1].detach()

for layer in range(5):

    student = corners[layer]

    kl = F.kl_div(
        F.log_softmax(
            student / 5,
            dim=-1
        ),
        F.softmax(
            teacher / 5,
            dim=-1
        ),
        reduction="batchmean"
    ) * 25

    print(
        layer,
        kl.item()
    )
```

这是观察：

```text
各层 localization distribution
距离最终层有多远
```

的最直接实验。

---

# 107. 为什么 GO-LSD 只在训练期需要？

推理时：

```text
不需要 GT
不需要 Hungarian Matching
不需要 KL
不需要 teacher/student loss
```

模型只需要：

```text
forward
→ final prediction
```

因此：

```text
GO-LSD
```

是训练优化机制。

不会在推理阶段额外跑一个 Teacher 网络。

---

# 108. 为什么说它几乎不增加推理成本？

因为推理图不需要：

```text
loss_ddf
KL
matched mask
teacher packaging
```

最终仍然只输出：

```text
pred_logits
pred_boxes
```

所以 GO-LSD 的额外成本主要发生在：

```text
training
```

而不是 deployment inference。

---

# 109. 论文实验给出的重要信号

D-FINE 论文报告，GO-LSD 与 FDR 的组合能进一步改善定位性能；论文还专门比较了不同 distillation 方法，并指出传统 logit mimicking / feature imitation 在 DETR 的 O2O matching 不稳定背景下并不一定有效，而 GO-LSD 的 localization-oriented设计表现更好。

论文的超参数分析还指出：

```text
T = 5
```

在其实验设置中取得了最好的平衡。

这里的重点不是背 AP 数字，而是理解：

```text
为什么 D-FINE 没有直接套一个普通 KD loss，
而是专门围绕 DETR matching + FDR distribution
设计 GO-LSD。
```

---

# 110. 常见误解 1：GO-LSD 需要一个预训练 Teacher Model

错误。

Teacher 是：

```text
同一个模型的 final decoder layer。
```

---

# 111. 常见误解 2：Teacher 是 EMA Model

错误。

不要把：

```text
GO-LSD Teacher
```

和后面训练 recipe 中的：

```text
EMA model
```

混淆。

它们完全不同。

GO-LSD Teacher：

```text
final decoder layer
```

EMA：

```text
整个模型参数的指数移动平均副本。
```

---

# 112. 常见误解 3：GO-LSD 蒸馏的是 Classification Logits

核心不是。

主要蒸馏：

```text
localization distributions
```

`teacher_logits` 主要用于：

```text
unmatched localization distillation weighting。
```

---

# 113. 常见误解 4：Teacher Corners 是最终 Bounding Box

错误。

`teacher_corners`：

```text
[B,Q,132]
```

是：

```text
4 × 33 localization distribution logits
```

最终 box：

```text
[B,Q,4]
```

是另一个 Tensor。

---

# 114. 常见误解 5：所有 Query 的 KL 权重相同

错误。

matched：

```text
IoU weight
```

unmatched：

```text
Teacher confidence weight
```

然后还做：

```text
matched/unmatched group balancing。
```

---

# 115. 常见误解 6：只有 Matched Query 才参与 GO-LSD

错误。

论文和源码都明确利用：

```text
matched
+
unmatched
```

只是权重不同。

---

# 116. 常见误解 7：GO-LSD 破坏 O2O

错误。

Localization 可以利用更广泛的跨层信息，但：

```text
classification
```

仍保持 O2O assignment。

---

# 117. 常见误解 8：Auxiliary Loss 就是 GO-LSD

错误。

Auxiliary Loss 是：

```text
中间层直接接受训练监督
```

GO-LSD 是：

```text
final localization distribution
→ shallow distributions
```

Auxiliary output 只是 GO-LSD 的承载位置之一。

---

# 118. 常见误解 9：Final Layer 也需要模仿自己

没有意义。

源码发现：

```text
student == teacher
```

时 DDF 为 0。

---

# 119. 常见误解 10：`T=5` 是 Classification Threshold

完全错误。

这里：

```text
T
```

是：

> Knowledge Distillation Temperature

用于：

```text
softmax(logits/T)
```

---

# 120. 常见误解 11：DDF 中的 IoU 是 Hungarian Cost

不是。

Hungarian Matching 先产生：

```text
matched indices
```

然后 DDF 对 matched predictions 计算：

```text
pred box vs GT 的 IoU
```

作为 distillation quality weight。

---

# 121. 常见误解 12：Global Union 会让一个 GT 在最终预测出现多个框

不是。

Global union 是：

```text
training localization optimization
```

的概念。

最终 detection 的：

```text
classification O2O
```

仍然负责唯一 set prediction 结构。

---

# 122. 常见误解 13：GO-LSD 只解决浅层训练困难

不完全。

它当然给浅层更强监督，但更完整的作用包括：

```text
利用最终 refined distribution 的 soft knowledge
利用跨层 matching candidates
利用 unmatched predictions
平衡不同质量 localization knowledge
```

所以它不仅是“多加一个浅层 loss”。

---

# 123. 常见误解 14：DDF = KL

不完整。

真正 DDF 包含：

```text
temperature-scaled KL
+
matched IoU weighting
+
unmatched confidence weighting
+
matched/unmatched count balancing
```

只写：

```text
KL(student, teacher)
```

只能算一个极简蒸馏，不是完整 DDF。

---

# 124. 本课最重要的公式

Student layer \(l\)：

\[
P_l(n)
=
softmax(z_l/T)
\]

Final teacher：

\[
P_L(n)
=
softmax(z_L/T)
\]

单个 localization distribution 的 distillation：

\[
D_k
=
T^2
D_{KL}
(
P_L^k
\|
P_l^k
)
\]

matched：

\[
w_k^{m}
=
IoU_k
\]

unmatched：

\[
w_k^{u}
=
Conf_k
\]

再对 matched / unmatched group 进行数量平衡。

论文形式可概括为：

\[
L_{DDF}
=
T^2
\sum_{l=1}^{L-1}
\left[
\sum_{k\in M}
\alpha_k
KL(P_l^k,P_L^k)
+
\sum_{k\in U}
\beta_k
KL(P_l^k,P_L^k)
\right]
\]

其中：

```text
M = matched set
U = unmatched set
```

并用：

```text
IoU
confidence
sqrt-count balancing
```

构造 \(\alpha_k,\beta_k\)。

---

# 125. 本课 Shape 速查表

| Tensor | Shape | 含义 |
|---|---:|---|
| Student `pred_corners` | `[B,Q,132]` | 浅层 FDR logits |
| Teacher `teacher_corners` | `[B,Q,132]` | 最终层 FDR logits |
| Reshaped Student | `[B*Q*4,33]` | 每条边一个分布 |
| Reshaped Teacher | `[B*Q*4,33]` | Teacher edge distribution |
| `teacher_logits` | `[B,Q,C]` | 最终层分类 logits |
| Teacher confidence | `[B,Q]` | unmatched quality weight |
| Query match mask | `[B,Q]` | matched / unmatched |
| Edge match mask | `[B*Q*4]` | 四边展开后的 mask |
| Per-edge KL | `[B*Q*4]` | 每条边的蒸馏损失 |
| `loss_ddf` | scalar | DDF 总损失 |

---

# 126. 6 层 Decoder 的监督速查

```text
Layer 0:
GT supervision
+
Final Layer teacher

Layer 1:
GT supervision
+
Final Layer teacher

Layer 2:
GT supervision
+
Final Layer teacher

Layer 3:
GT supervision
+
Final Layer teacher

Layer 4:
GT supervision
+
Final Layer teacher

Layer 5:
GT supervision
+
作为 Teacher
```

这就是最简洁的多层监督心智模型。

---

# 127. 本课练习

### 练习 1

为什么普通 Auxiliary Loss 不等于 GO-LSD？

### 练习 2

`deim_dfine_x` 有 6 个 Decoder layers。普通 `aux_outputs` 有几个 Student layers？

### 练习 3

为什么：

```text
teacher_corners
```

shape 是：

```text
[B,300,132]
```

？

### 练习 4

为什么 Criterion 要把它 reshape 成：

```text
[B*300*4,33]
```

？

### 练习 5

Teacher 为什么必须 `.detach()`？

### 练习 6

Temperature \(T>1\) 会怎样改变 distribution？

### 练习 7

为什么 KL 前要乘：

\[
T^2
\]

？

### 练习 8

DDF 为什么把 matched / unmatched 分开？

### 练习 9

Matched prediction 为什么用 IoU 加权？

### 练习 10

Unmatched prediction 为什么用 Teacher classification confidence 加权？

### 练习 11

为什么 matched query 不统一使用 classification confidence？

### 练习 12

为什么 GO-LSD 还要学习 unmatched predictions？

### 练习 13

为什么 unmatched distillation 不会自动破坏 O2O classification？

### 练习 14

什么是跨层：

```text
global union set
```

？

### 练习 15

为什么不同 Decoder Layer 的 Hungarian matches 可能不同？

### 练习 16

为什么 localization 可以使用 union set，但 classification 仍应保持 O2O？

### 练习 17

为什么用：

\[
\sqrt{K}
\]

而不是直接用 \(K\) 做 matched/unmatched group weighting 能缓和数量失衡？

### 练习 18

`loss_fgl` 和 `loss_ddf` 分别是谁教谁？

### 练习 19

FDR 与 GO-LSD 为什么天然互补？

### 练习 20

GO-LSD Teacher 和 EMA Teacher 有什么区别？

### 练习 21

为什么 GO-LSD 几乎不增加推理成本？

### 练习 22

`teacher_logits` 是否直接作为 localization KL target？为什么？

### 练习 23

如果 Student distribution 已经与 Teacher 完全相同，DDF 应该是多少？

### 练习 24

为什么只蒸馏最终 bbox `[B,Q,4]` 会丢失一部分 FDR 知识？

---

# 128. 练习答案摘要

### 1

Auxiliary Loss：

```text
GT → intermediate layer
```

GO-LSD：

```text
final layer distribution → shallow layer distribution
```

### 2

```text
5
```

即 Layer 0–4。

### 3

\[
4(32+1)=132
\]

### 4

为了把每个 Query 的四条边拆成独立的 33-bin distributions。

### 5

Teacher 只提供目标，不应被 DDF 梯度拉向 Student。

### 6

使 distribution 更平滑，暴露更多 soft relative information。

### 8

因为两组的监督可靠性来源和数量规模都不同。

### 9

matched 有明确 GT，因此 IoU 能直接衡量 localization quality。

### 10

unmatched 没有对应 GT，Teacher confidence 是可用的可靠性 proxy。

### 14

所有 Decoder layers 的 localization matching indices 的联合信息。

### 16

分类必须维护一个 GT 对一个 Query 的 set prediction；localization knowledge transfer 可以更宽松。

### 18

```text
FGL:
GT → current localization distribution

DDF:
Final refined distribution → shallow distribution
```

### 20

GO-LSD Teacher 是同一次 forward 的最终 Decoder Layer；EMA 是整个模型参数的移动平均副本。

### 21

训练期 loss 和 matching 在推理时都移除，不需要额外 Teacher forward。

### 23

理论上：

```text
0
```

### 24

因为完整 distribution 还编码峰值形状、邻近 bins 概率和 localization uncertainty。

---

# 129. 进入第 14 课前必须能独立回答的问题

你至少应该能不看笔记解释：

```text
1. 为什么 D-FINE 已有 Auxiliary Loss 还要 GO-LSD？

2. Final Layer 为什么能成为 Teacher？

3. Teacher/Student 蒸馏的究竟是哪一个 Tensor？

4. [B,Q,132] 为什么 reshape 成 [B*Q*4,33]？

5. KL divergence 在这里比较什么？

6. T=5 的作用是什么？

7. 为什么 Teacher 要 detach？

8. matched query 为什么用 IoU？

9. unmatched query 为什么用 teacher confidence？

10. 为什么 matched/unmatched 还需要数量平衡？

11. Global union set 为什么存在？

12. Localization union 为什么不等于 classification O2M？

13. FGL 和 DDF 有什么不同？

14. FDR 和 GO-LSD 是怎样配合的？

15. 为什么 GO-LSD 不增加最终推理结构？
```

如果这些都能自己讲出来，那么你已经真正理解：

```text
D-FINE 不只是“每层预测一个 box”，
而是一个多层、分布式、带自蒸馏的 localization optimization system。
```

---

# 130. 第 11–13 课现在可以合成一张图

```text
HybridEncoder Features
        │
        ▼
Top-K Query Selection
        │
        ▼
┌──────────────────────────────┐
│ Decoder Layer 0              │
│ Self-Attn                    │
│ Deformable Cross-Attn        │
│ FFN                          │
│ FDR distribution             │
└──────────────┬───────────────┘
               │
               ▼
         refined reference
               │
               ▼
┌──────────────────────────────┐
│ Decoder Layer 1              │
│ ...                          │
│ distribution refinement      │
└──────────────┬───────────────┘
               │
              ...
               │
               ▼
┌──────────────────────────────┐
│ Decoder Layer 5              │
│ final refined distribution   │
│ final bbox                   │
│ final class                  │
└──────────────┬───────────────┘
               │
               │ Teacher
               ▼
             GO-LSD
               │
       ┌───────┼────────┐
       ▼       ▼        ▼
     L0 dist  L1 dist  L4 dist
       │       │        │
       └──── DDF KL ────┘

Meanwhile each layer:
GT
↓
Hungarian Matching
↓
classification / MAL
bbox L1
GIoU
FGL
```

---

# 131. 下一课：第一次把 `deim_dfine_x` 完整闭环

下一课是：

# 第 14 课《把 `deim_dfine_x` 从输入到 loss 完整走一遍》

前面几课我们已经分别拆开：

```text
第 9 课:
HGNetv2

第 10 课:
HybridEncoder

第 11 课:
DFINETransformer Decoder

第 12 课:
FDR

第 13 课:
GO-LSD + 多层监督
```

下一课会把它们重新合起来：

```text
[B,3,640,640]
↓
HGNetv2
↓
C3/C4/C5
↓
HybridEncoder
↓
P3/P4/P5
↓
8400 positions
↓
Top-K 300 Queries
↓
6-layer DFINETransformer
↓
FDR
↓
GO-LSD / aux outputs
↓
Hungarian Matching
↓
MAL
↓
L1 / GIoU
↓
FGL
↓
DDF
↓
weighted total loss
↓
backward()
```

并且会制作一张：

> **`deim_dfine_x` 完整 Tensor Shape Trace + Loss Graph**

到第 14 课结束，第一条重点主线 `deim_dfine_x` 就会形成真正完整的闭环。

---

# 132. 源码与论文核对说明

本课按当前公开 D-FINE / DEIM 实现与 D-FINE 论文核对。

重点源码：

```text
DEIM:
engine/deim/dfine_decoder.py
engine/deim/deim_criterion.py

D-FINE:
src/zoo/dfine/dfine_decoder.py
src/zoo/dfine/dfine_criterion.py
```

关键源码关系：

```text
final layer:
out_corners[-1]
out_logits[-1]

        ↓

_set_aux_loss2(
    shallow corners,
    shallow refs,
    teacher_corners=out_corners[-1],
    teacher_logits=out_logits[-1]
)

        ↓

loss_local()

        ↓

student corners
teacher corners.detach()

        ↓

temperature-scaled KL

        ↓

matched:
IoU weight

unmatched:
teacher confidence weight

        ↓

sqrt-count group balance

        ↓

loss_ddf
```

D-FINE 论文对 GO-LSD 的定义包括：

```text
final refined distributions → shallow layers
per-layer Hungarian matching
cross-layer unified union set
matched + unmatched localization distillation
DDF decoupled weighting
classification remains O2O
```

论文给出的 DDF 公式使用：

```text
KL divergence
temperature T
matched IoU weighting
unmatched confidence weighting
sqrt-count balancing
```

并在其超参数实验中采用：

```text
T = 5
```

作为表现最佳的温度设置。

DEIM 配置中保留：

```yaml
loss_fgl: 0.15
loss_ddf: 1.5
```

因此 `deim_dfine_x` 同时继承 D-FINE 的：

```text
FDR
GO-LSD
```

以及 DEIM 自己的：

```text
Dense O2O
MAL
training recipe
```

后者将在第 15–18 课重点展开。

---

## 参考资料

- Yansong Peng et al., *D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement*, ICLR 2025.
- D-FINE official implementation: `Peterande/D-FINE`
- DEIM official implementation: `Intellindust-AI-Lab/DEIM`
- DEIM `engine/deim/dfine_decoder.py`
- DEIM `engine/deim/deim_criterion.py`
- DEIM `configs/base/deim.yml`
- DEIM `configs/base/dfine_hgnetv2.yml`

> 仓库会持续更新。实际调试时，应以你所使用 checkout 的源码、最终 merged YAML，以及 debugger / forward hook 实测结果为最高依据。
