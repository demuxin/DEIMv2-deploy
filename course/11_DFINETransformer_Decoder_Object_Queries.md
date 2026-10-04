# DEIMv2 源码课程 · 第 11 课
# DFINETransformer Decoder——Query 真正开始检测目标

> 本课从第 10 课的 `HybridEncoder` 输出继续，目标是把 `deim_dfine_x` 中 **8400 个多尺度空间位置如何变成 300 个 Object Queries，再经过 6 层 Decoder 逐步变成类别和边界框** 完整讲清楚。
>
> 本课固定使用 X 级主线的典型尺寸：
>
> ```text
> P3 = [B,384,80,80]
> P4 = [B,384,40,40]
> P5 = [B,384,20,20]
>
> hidden_dim  = 384
> num_queries = 300
> num_layers  = 6
> nhead       = 8
> num_levels  = 3
> num_points  = [3,6,3]
> reg_max     = 32
> reg_scale   = 8   # X 配置覆盖基础配置
> ```
>
> 本课结束后，你应该能自己解释：
>
> ```text
> P3/P4/P5
> → [B,8400,384]
> → encoder proposals
> → Top-300
> → query content + reference boxes
> → self-attention
> → deformable cross-attention
> → iterative refinement
> → pred_logits / pred_boxes
> ```

---

## 1. 先把前 10 课接起来

我们已经走到：

```text
Image [B,3,640,640]
        ↓
HGNetv2-B5
        ↓
C3 [B, 512,80,80]
C4 [B,1024,40,40]
C5 [B,2048,20,20]
        ↓
HybridEncoder
        ↓
P3 [B,384,80,80]
P4 [B,384,40,40]
P5 [B,384,20,20]
```

DEIM 容器的主干可以理解为：

```python
x = backbone(x)
x = encoder(x)
x = decoder(x, targets)
```

这一课研究最后一句里的：

```text
decoder = DFINETransformer
```

注意：`DFINETransformer` 不是“一个 TransformerDecoderLayer”，而是完整检测头容器。它内部包含：

```text
multi-scale input projection
encoder proposal generation
encoder score head
encoder bbox head
Top-K query selection
denoising query construction
TransformerDecoder
classification heads
distribution bbox heads
Integral / FDR machinery
```

---

# 2. 第一个关键转换：三个 Feature Maps → 一个 Memory Sequence

输入：

```text
P3 [B,384,80,80]
P4 [B,384,40,40]
P5 [B,384,20,20]
```

对每个 level 做：

```text
[B,C,H,W]
→ flatten(H,W)
→ [B,C,HW]
→ transpose / permute
→ [B,HW,C]
```

于是：

### P3

\[
80	imes80=6400
\]

```text
[B,384,80,80]
→ [B,6400,384]
```

### P4

\[
40	imes40=1600
\]

```text
[B,384,40,40]
→ [B,1600,384]
```

### P5

\[
20	imes20=400
\]

```text
[B,384,20,20]
→ [B,400,384]
```

然后沿 token 维拼接：

\[
6400+1600+400=8400
\]

得到：

```text
memory = [B,8400,384]
```

这是本课第一个必须牢牢记住的数字。

---

# 3. 8400 不是 Query 数量

这是最容易混淆的地方。

```text
8400
=
三个多尺度 Feature Maps 的全部 spatial positions

300
=
最终送进 Decoder 的 detection queries
```

可以画成：

```text
8400 image positions
        ↓
proposal generation
        ↓
proposal scoring
        ↓
Top-K
        ↓
300 detection queries
```

所以：

> `8400` 是 dense image memory，`300` 是 sparse object-centric slots。

---

# 4. 为什么 Flatten 后还必须保存 `spatial_shapes`？

Flatten 后：

```text
memory = [B,8400,384]
```

已经看不出某个 token 原来属于：

```text
80×80
40×40
20×20
```

因此同时保存：

```text
spatial_shapes =
[
  [80,80],
  [40,40],
  [20,20]
]
```

它告诉后面的 Multi-Scale Deformable Attention：

```text
哪些 token 属于哪个 level？
每个 level 的 H/W 是多少？
怎样把一维索引重新解释成二维坐标？
```

这正是第 7 课 `spatial_shapes` 的真实来源。

---

# 5. 8400 个位置还会生成 8400 个 Anchors / Proposals

DFINETransformer 不只保存：

```text
8400 feature vectors
```

还会给这些位置建立几何先验：

```text
(cx, cy, w, h)
```

可以把它理解为：

```text
每个 spatial position
+
一个初始候选框
```

对于某层 H×W 网格，中心位置可以理解为：

\[
c_x=rac{x+0.5}{W},
\qquad
c_y=rac{y+0.5}{H}
\]

例如 80×80 左上角第一个 cell：

\[
c_x=c_y=rac{0.5}{80}=0.00625
\]

加 `0.5` 是因为我们取的是 grid cell 中心，而不是左上边界。

不同 level 的初始宽高会反映不同尺度的 proposal prior。

---

# 6. 为什么 Box 经常在 inverse-sigmoid 空间里更新？

归一化 box 希望位于：

```text
0 < x < 1
```

如果直接：

\[
p' = p+\Delta
\]

很容易越界。

DETR 系列常用：

\[
z=\sigma^{-1}(p)=\lograc{p}{1-p}
\]

然后：

\[
z'=z+\Delta
\]

最后：

\[
p'=\sigma(z')
\]

这样最终天然回到：

```text
(0,1)
```

例如：

\[
p=0.2
\]

\[
\sigma^{-1}(0.2)pprox-1.386
\]

若：

\[
\Delta=0.5
\]

则：

\[
p'=\sigma(-0.886)pprox0.292
\]

这就是后面 iterative refinement 的核心数学工具。

---

# 7. Encoder Proposal Stage

现在有：

```text
memory
[B,8400,384]

anchors
[...,8400,4]
```

memory 先经过一个 projection + normalization：

```text
[B,8400,384]
→ [B,8400,384]
```

然后分成两个 head。

## Classification head

COCO 有 80 类：

```text
[B,8400,384]
→ Linear(384,80)
→ [B,8400,80]
```

所以 8400 个 spatial candidates 都有类别分数。

## Bounding-box head

```text
[B,8400,384]
→ MLP
→ [B,8400,4]
```

它预测相对 anchor 的 box refinement。

因此每个 candidate 现在都有：

```text
feature
class logits
candidate bbox
```

---

# 8. Top-K：300 个 Query 真正出现

默认：

```text
query_select_method = default
```

对：

```text
outputs_logits
[B,8400,80]
```

先在 class 维取最大值：

```text
[B,8400,80]
→ max(class)
→ [B,8400]
```

然后：

```text
Top-K
K=300
```

得到：

```text
topk_indices
[B,300]
```

再用同一组 indices 同时从：

```text
memory
logits
bbox proposals
```

中 gather。

最终得到：

```text
enc_topk_memory
[B,300,384]

enc_topk_logits
[B,300,80]

enc_topk_bbox_unact
[B,300,4]
```

---

# 9. 一个 5 个位置的最小例子

假设只有：

```text
5 positions
3 classes
```

分类 logits：

```text
p0 [0.1, 2.0, 0.3]
p1 [0.2, 0.4, 0.1]
p2 [3.1, 0.1, 0.2]
p3 [0.7, 0.9, 0.2]
p4 [0.1, 0.2, 2.4]
```

每个位置取最大类别分数：

```text
[2.0, 0.4, 3.1, 0.9, 2.4]
```

若：

```text
K=2
```

选择：

```text
p2
p4
```

然后必须从：

```text
memory
bbox
logits
```

中同时取 `p2/p4`。

为什么一定要用同一组 index？

因为：

```text
query feature
```

必须和：

```text
它自己的 reference bbox
```

一一对应。

---

# 10. 现在回答本课最核心的问题：Object Query 到底从哪里来？

默认配置：

```text
learn_query_content = False
```

因此正常 detection query 的 content 来自：

```text
enc_topk_memory
```

也就是：

> **从当前图片的 8400 个 multi-scale spatial features 中，根据 encoder proposal score 选出的 Top-300 feature vectors。**

因此：

```text
query content = image-conditioned
```

而不是简单的：

```text
300 个与图片无关的固定 embedding
```

源码仍支持：

```text
learn_query_content=True
```

这时可以使用：

```text
Embedding(num_queries, hidden_dim)
```

但当前主线默认不是这种模式。

---

# 11. 和原始 DETR 的 Object Query 有什么区别？

教学版原始 DETR 常写成：

```text
learnable query 1
learnable query 2
...
learnable query N
```

而 D-FINE/RT-DETR 这类现代实时 DETR 更接近：

```text
dense image memory
↓
proposal scoring
↓
Top-K
↓
image-aware query initialization
```

因此 Decoder 不必完全从“空白槽位”开始搜索。

它一开始已经获得：

```text
一个比较有希望的 feature
+
一个比较有希望的 box
```

---

# 12. Query 其实至少包含两个概念

不要把 Query 只理解成一个 Tensor。

## Query Content

```text
[B,300,384]
```

它表示：

```text
这个 object hypothesis 当前“看起来是什么”
```

## Reference Box

```text
[B,300,4]
```

它表示：

```text
这个 object hypothesis 当前“大概在哪里、多大”
```

reference box：

```text
(cx,cy,w,h)
```

还会经过 `query_pos_head` 变成：

```text
query positional embedding
[B,300,384]
```

所以可以记成：

```text
Query
=
Content Hypothesis
+
Spatial Hypothesis
```

---

# 13. Query Positional Embedding

reference box：

```text
[B,300,4]
```

经过 MLP：

```text
4
→ 2D
→ D
```

X 中：

```text
4
→ 768
→ 384
```

得到：

```text
query_pos
[B,300,384]
```

这样 box geometry 就被编码成与 query content 同维度的向量。

这比固定 positional embedding 更自然，因为：

```text
reference box 会逐层变化
```

于是：

```text
query position 也会跟着目标位置动态更新
```

---

# 14. Decoder 第一层真正拿到什么？

忽略训练专属 Denoising：

```text
query content
[B,300,384]

reference boxes
[B,300,4]

image memory
[B,8400,384]

spatial_shapes
[[80,80],
 [40,40],
 [20,20]]
```

现在 Transformer Decoder 才真正开始。

---

# 15. 一个 Decoder Layer 的完整结构

先记住：

```text
Query Content
[B,Q,D]

      │
      ├── Reference Box
      │       ↓
      │   Query Pos MLP
      │
      ▼

Self-Attention
Query ↔ Query

      ↓

Multi-Scale Deformable Cross-Attention
Query ↔ Image

      ↓

Gate / Residual Fusion

      ↓

FFN

      ↓

Updated Query Feature

      ↓

Classification Head
BBox Heads

      ↓

Updated Reference Box
```

这里：

```text
Q = 300
D = 384
```

---

# 16. Query Self-Attention

概念上：

```text
Q = content + query_pos
K = content + query_pos
V = content
```

然后：

```text
Multi-Head Self-Attention
```

输入输出：

```text
[B,300,384]
→ [B,300,384]
```

为什么 queries 之间需要互相看？

因为目标检测是：

```text
set prediction
```

300 个 object hypotheses 需要协调：

```text
谁负责哪个目标？
是否多个 query 都在关注同一个区域？
场景中其他 object hypotheses 是什么？
```

---

# 17. Self-Attention 的具体数字

X：

```text
hidden_dim = 384
nhead = 8
```

每个 head：

\[
d_h=rac{384}{8}=48
\]

概念 shape：

```text
Q [B,8,300,48]
K [B,8,300,48]
V [B,8,300,48]
```

每个 head 的 attention matrix：

```text
300×300
```

即：

\[
90,000
\]

8 heads：

\[
720,000
\]

个 query-query attention scores / image。

---

# 18. Cross-Attention：Query 真正去看图像

Query：

```text
[B,300,384]
```

Image memory：

```text
[B,8400,384]
```

如果做普通 dense cross-attention：

\[
300	imes8400
=
2,520,000
\]

每个 head 的 query-position pairs。

8 heads：

\[
20,160,000
\]

所以 D-FINE 使用第 7 课已经学过的：

```text
Multi-Scale Deformable Attention
```

---

# 19. 第 7 课的 Reference Points 终于有来源了

当时我们学：

```text
reference_points
→ sampling_offsets
→ sampling_locations
→ bilinear sampling
→ attention_weights
```

现在完整来源是：

```text
P3/P4/P5
↓
8400 spatial positions
↓
anchors + encoder bbox predictions
↓
Top-300
↓
reference boxes
↓
Deformable Cross-Attention
```

所以 reference points 并不是凭空产生的。

---

# 20. `num_points=[3,6,3]`

三个 feature levels：

```text
P3 → 3 sampling points
P4 → 6 sampling points
P5 → 3 sampling points
```

因此每个：

```text
query × head
```

总共采样：

\[
3+6+3=12
\]

个位置。

正常 300 queries、8 heads：

\[
300	imes8	imes12
=
28,800
\]

个 query-head-sampling locations / decoder layer / image。

这与 dense cross-attention 的结构完全不同，因此不能直接当作严格 FLOPs 比值，但能非常直观地解释为什么 Deformable Attention 适合实时检测。

---

# 21. Sampling Offsets 与 Attention Weights

可以概念性理解：

```text
sampling_offsets
[B,300,8,12,2]
```

最后：

```text
2 = dx,dy
```

同时：

```text
attention_weights
[B,300,8,12]
```

每个采样位置有一个权重。

reference 是 4D box：

```text
(cx,cy,w,h)
```

因此 offset 可以根据：

```text
w,h
```

缩放。

直觉：

```text
小目标
→ 在较小区域采样

大目标
→ 在较大区域采样
```

---

# 22. 为什么 Sampling Location 可以是小数？

例如：

```text
x = 13.42
y = 7.81
```

它不落在整数 grid 上。

因此要用第 7 课的：

```text
bilinear interpolation
```

从四个邻近 feature points 加权取值。

最终 Multi-Scale Deformable Attention 聚合多个 level、多个 head、多个 sampling points 的信息，重新得到：

```text
[B,300,384]
```

---

# 23. Self-Attention 与 Cross-Attention 的职责

非常建议记成：

```text
Self-Attention
=
Query ↔ Query

Cross-Attention
=
Query ↔ Image

FFN
=
Feature-channel transformation
```

一个 Decoder Layer 就是在反复做这三种信息更新。

---

# 24. D-FINE Decoder 里的 Gate

D-FINE Decoder Layer 还包含门控融合结构。

直觉可以理解为：

```text
旧 Query State
+
从图像新读到的信息
↓
Learned Gate
↓
决定保留多少旧信息
决定吸收多少新信息
```

通用门控形式可写成：

\[
g=\sigma(f(x,y))
\]

\[
z=g\odot x+(1-g)\odot y
\]

具体实现以当前源码为准。

关键思想是：

> Cross-Attention 的新信息不是简单无条件覆盖原 Query，而是可以通过可学习门控融合。

---

# 25. FFN

X 的 FFN：

```text
384
→ 2048
→ 384
```

DEIM 训练配置将激活改成：

```text
SiLU
```

所以：

```text
[B,300,384]
→ Linear
→ [B,300,2048]
→ SiLU
→ Linear
→ [B,300,384]
```

shape 最终不变，但 representation 被更新。

---

# 26. 为什么需要 6 层 Decoder？

X：

```text
num_layers = 6
```

不是每层都“从头检测”。

更准确地说：

```text
Layer 0
粗定位

↓

Layer 1
更准确

↓

Layer 2
继续修正

↓

...

↓

Layer 5
最终预测
```

这叫：

```text
Iterative Refinement
```

---

# 27. Iterative Box Refinement 的核心公式

设第 i 层 reference：

\[
r_i
\]

网络预测修正量：

\[
\Delta_i
\]

则：

\[
r_{i+1}
=
\sigma
\left(
\sigma^{-1}(r_i)
+
\Delta_i
ight)
\]

例如：

```text
cx = 0.40
```

\[
\sigma^{-1}(0.4)pprox-0.405
\]

如果：

```text
Δcx = +0.2
```

则：

\[
-0.405+0.2=-0.205
\]

\[
\sigma(-0.205)pprox0.449
\]

于是：

```text
cx:
0.400 → 0.449
```

下一层就从更准确的 reference 继续采样。

---

# 28. 为什么 Iterative Refinement 很强？

因为形成：

```text
better reference
        ↓
better deformable sampling locations
        ↓
better image evidence
        ↓
better query representation
        ↓
better bbox
        ↓
next layer gets a better reference
```

这是一条非常重要的正反馈链。

这也是“Query 真正开始检测目标”的核心过程。

---

# 29. Classification Head

每层 query feature：

```text
[B,300,384]
```

经过分类 head：

```text
384 → 80
```

得到：

```text
[B,300,80]
```

也就是：

```text
300 queries
×
80 COCO classes
```

训练时中间 Decoder layers 也可以产生 prediction，形成 auxiliary supervision。

---

# 30. 为什么每层都监督？

如果只监督最后一层：

```text
Layer 0~4
```

只能依靠深层反向传播间接学习。

Auxiliary Loss 让中间层直接知道：

```text
你的分类应该往哪里改
你的 box 应该往哪里改
```

这通常能改善 DETR 的优化和收敛。

---

# 31. D-FINE 的 BBox Head 为什么输出 132，而不是 4？

这是下一课的核心。

D-FINE 引入：

```text
Fine-grained Distribution Refinement
```

当前：

```text
reg_max = 32
```

每个方向有：

\[
32+1=33
\]

个 distribution bins。

四个方向：

```text
left
top
right
bottom
```

所以：

\[
4	imes33=132
\]

bbox distribution head 会输出：

```text
[B,300,132]
```

reshape：

```text
[B,300,4,33]
```

再：

```text
softmax
→ Integral
→ distance
→ bbox
```

最终仍得到：

```text
[B,300,4]
```

因此：

> 132 不是 132 个 boxes，而是一个 box 四个方向各 33 个离散 localization logits。

第 12 课会完整推导这一过程。

---

# 32. `pre_bbox_head`

D-FINE 还包含 preliminary bbox refinement。

可以先把定位理解成：

```text
Reference Box
↓
Preliminary / Coarse Refinement
↓
Fine-grained Distribution Refinement
↓
Final Refined Box
```

也就是说 D-FINE 不满足于简单的一次：

```text
4-number regression
```

而是进一步对边界做细粒度分布式建模。

---

# 33. LQE：Localization Quality Estimation

源码还包含 LQE。

直觉：

```text
分类分数高
```

不一定代表：

```text
定位也可靠
```

例如：

```text
Query A:
dog score 很高
但 localization distribution 很不确定

Query B:
dog score 很高
且 localization distribution 很集中
```

二者的最终可信度不应该完全一样。

LQE 从 localization distribution 的统计信息中估计定位质量，再影响 score。

详细数学留到后续 FDR / GO-LSD 课程。

---

# 34. Training 时还有 Denoising Queries

到目前为止，我们为了理解主路径，一直假设：

```text
Q = 300
```

真实训练还有：

```text
num_denoising = 100
```

模型会根据 GT：

```text
labels
boxes
```

加入：

```text
label noise
box noise
```

构造 Denoising Queries。

它们与正常 300 matching queries 拼接后一起进入 Decoder。

---

# 35. Denoising 的直觉

普通 Query：

```text
“我需要自己找到目标在哪里。”
```

Denoising Query：

```text
“这里给你一个带噪声的 GT 类别和 GT Box，
请你把它恢复正确。”
```

因此它提供更直接的训练信号。

尤其训练早期：

```text
matching queries 很乱
Hungarian assignment 也不稳定
```

Denoising 可以帮助 Decoder 更快学会：

```text
分类
+
box refinement
```

---

# 36. `num_denoising=100` 不代表永远正好 100 个 DN Queries

这是非常容易误解的一点。

真实 DN 数量还受到：

```text
当前 batch 每张图 GT 数
denoising group 构造
padding
```

影响。

因此 Debug 时必须看：

```text
dn_meta
实际 Tensor shape
```

不要机械认为：

```text
训练 Query 数一定 = 300 + 100
```

---

# 37. Training 和 Inference 的 Query 数为什么可能不同？

推理没有 GT：

```text
→ 无法构造 GT-based denoising group
→ 没有 DN queries
→ 正常 300 detection queries
```

训练：

```text
DN queries
+
300 matching queries
```

所以训练阶段如果你看到：

```text
target.shape[1] > 300
```

不要立即认为模型出错。

---

# 38. 为什么 Denoising 需要 Attention Mask？

DN Queries 带有 GT-derived 信息。

如果允许它们和正常 queries 随意互相通信：

```text
答案信息可能泄漏
```

因此构造：

```text
attn_mask
```

限制某些 Query Groups 的 Self-Attention。

这样：

```text
Denoising Task
```

和：

```text
Normal Matching Task
```

能够在同一个 Decoder 中训练，又不发生不合理的信息泄露。

---

# 39. Decoder 输出为什么还要 Split？

训练输入是：

```text
DN Queries
+
Matching Queries
```

因此 Decoder 输出也是拼在一起的。

随后根据：

```text
dn_meta
```

拆成：

```text
dn_outputs
normal outputs
```

分别计算：

```text
Denoising losses
```

和：

```text
Hungarian / normal detection losses
```

---

# 40. 最终正常输出

Inference 最重要的两个 Tensor：

```text
pred_logits
[B,300,80]

pred_boxes
[B,300,4]
```

`pred_boxes` 通常是归一化：

```text
(cx,cy,w,h)
```

例如：

```text
[0.5,0.4,0.2,0.3]
```

对于 640×640：

```text
center x ≈ 320
center y ≈ 256
width    ≈ 128
height   ≈ 192
```

之后 PostProcessor 再转换为最终图像坐标和输出格式。

---

# 41. 训练输出为什么比推理复杂很多？

训练 output dict 还可能包含：

```text
aux_outputs
enc_aux_outputs
pre_outputs
dn_outputs
dn_pre_outputs
dn_meta
pred_corners
ref_points
```

它们分别服务于：

```text
中间 Decoder 层监督
Encoder Proposal 监督
Preliminary Box 监督
Denoising 监督
FDR localization loss
跨层 localization distillation
```

所以：

> 训练时的 DFINETransformer 不只是“返回最后一个 bbox”。

---

# 42. 为什么 Encoder Proposals 也需要 Loss？

Query selection 依赖：

```text
encoder score
+
encoder bbox proposal
```

如果这里学不好：

```text
Top-300
```

就会选择很差的 Decoder 起点。

因此需要直接监督：

```text
Encoder Proposal Stage
```

而不能只依赖最后 Decoder Loss。

---

# 43. `detach()`：源码阅读中必须理解的细节

Top-K 选出的某些：

```text
query content
reference boxes
```

在进入 Decoder 前会出现：

```text
detach()
```

`detach`：

```text
不改变数值
不改变 shape
```

但会：

```text
切断对应 autograd 路径
```

例如：

```python
y = f(x)
z = y.detach()
loss = g(z)
```

`loss` 不会通过：

```text
z → y → x
```

这条路径回传。

---

# 44. 为什么 Detach 后 Encoder 仍然能学？

因为 Encoder Proposal 有自己的：

```text
enc_aux_outputs
```

和直接监督。

可以理解成两条路径：

```text
Encoder Proposal
      │
      ├── direct encoder loss
      │
      └── selected proposal.detach()
               ↓
            Decoder
               ↓
          decoder losses
```

这使 Proposal Stage 和 Decoder Stage 的优化更加清晰。

---

# 45. Top-K 与 Hungarian Matching 完全不是一回事

这是本课必须消灭的一个混淆。

## Top-K Query Selection

发生在：

```text
Model Forward
```

作用：

```text
8400 candidates
→ 选择 300 个 Decoder 起点
```

依据：

```text
Encoder Proposal Score
```

## Hungarian Matching

发生在：

```text
Criterion / Training Loss
```

作用：

```text
300 predictions
↔
GT objects
```

依据：

```text
classification cost
bbox L1 cost
GIoU cost
```

所以：

```text
Top-K
≠
Hungarian
```

---

# 46. 一张图理解两次选择

```text
8400 spatial positions
        │
        │ encoder proposal score
        ▼
      Top-K
        │
        ▼
300 Decoder Queries
        │
        │ Decoder × 6
        ▼
300 predictions
        │
        │ Hungarian Matching
        ▼
GT ↔ matched predictions
```

Top-K 回答：

```text
Decoder 从哪里开始？
```

Hungarian 回答：

```text
训练时谁负责哪个 GT？
```

---

# 47. Query Self-Attention 也不是 Hungarian Matching

Self-Attention 只是：

```text
Query Feature Interaction
```

它没有显式解决：

```text
GT assignment
```

O2O assignment 是 Criterion/Matcher 层面的训练机制。

---

# 48. 为什么 300 Queries 不代表图片有 300 个物体？

假设：

```text
GT = 7
```

模型仍然输出：

```text
300 predictions
```

Hungarian 最多把：

```text
7 个 queries
```

与：

```text
7 个 GT
```

一一匹配。

剩余大量 Query 是 unmatched / negative slots。

这也是后面 DEIM Dense O2O 要解决：

```text
positive supervision 太稀疏
```

问题的来源。

---

# 49. 三种“数量”必须分清

```text
8400
=
Dense Spatial Positions

300
=
Decoder Query Capacity

GT count
≈
O2O Positive Matches
```

例如一张图只有 7 个 GT：

```text
8400 image positions
→ 300 queries
→ ~7 positive O2O assignments
```

这三层密度完全不同。

---

# 50. Query 数越多越好吗？

不是。

Self-Attention：

\[
O(Q^2)
\]

300：

\[
300^2=90,000
\]

600：

\[
600^2=360,000
\]

Query 翻倍，Self-Attention pair 数约变 4 倍。

Deformable Cross-Attention 则更接近随 Q 线性增长：

```text
300×8×12 = 28,800
600×8×12 = 57,600
```

所以：

```text
num_queries
```

本身也是 accuracy / latency / memory 的权衡参数。

---

# 51. Query 不是 Anchor

Anchor：

```text
几何 prior / proposal
```

Query：

```text
参与 Transformer 运算的 object representation
```

一个 Query 可以携带：

```text
reference box
```

但：

```text
Query ≠ Reference Box
Query ≠ Anchor
```

---

# 52. Query 也不是某个固定 Feature Map Pixel

虽然 Query Content 一开始可能从某个 Top-K spatial position 选出，但进入 Decoder 后：

```text
它通过 Multi-Scale Deformable Attention
在多个尺度、多个位置读取信息
```

所以它已经成为：

```text
独立的 object slot
```

不再永久绑定最初的单个 grid cell。

---

# 53. Query 的三个阶段

可以这样记：

```text
Stage 1
8400 Spatial Candidates

↓

Stage 2
300 Selected Proposals

↓

Stage 3
300 Object Queries
iteratively refined by Decoder
```

这就是：

```text
Dense Visual Representation
→ Sparse Object Representation
```

的转折。

---

# 54. 为什么这种设计适合实时 DETR？

如果一直在 8400 个位置做复杂 object-level computation：

```text
成本很高
```

而现在：

```text
Dense Feature Extraction
→ Cheap Proposal Selection
→ Sparse 300-Query Decoder
```

将昂贵的 object-centric reasoning 集中在：

```text
300 个 slots
```

上。

同时 Cross-Attention 又不是：

```text
300×8400 dense attention
```

而是少量 Deformable Sampling。

这就是实时性的关键来源之一。

---

# 55. 正常推理的完整 Shape Trace

```text
INPUT FROM HYBRID ENCODER

P3 [B,384,80,80]
P4 [B,384,40,40]
P5 [B,384,20,20]

        ↓ flatten

F3 [B,6400,384]
F4 [B,1600,384]
F5 [B, 400,384]

        ↓ concatenate

memory
[B,8400,384]

        ↓ proposal heads

enc_logits
[B,8400,80]

enc_boxes
[B,8400,4]

        ↓ Top-K 300

query_content
[B,300,384]

reference_boxes
[B,300,4]

        ↓ query_pos_head

query_pos
[B,300,384]

        ↓ Decoder × 6

decoder_feature
[B,300,384]

        ↓ heads

class_logits
[B,300,80]

distribution_logits
[B,300,132]

        ↓ FDR / Integral

pred_boxes
[B,300,4]
```

---

# 56. 一张完整结构图

```text
P3 80×80 ── flatten ── 6400 ──┐
P4 40×40 ── flatten ── 1600 ──┼─→ memory [B,8400,384]
P5 20×20 ── flatten ──  400 ──┘
                                      │
                          ┌───────────┴───────────┐
                          ▼                       ▼
                    Score Head               BBox Head
                  [B,8400,80]              [B,8400,4]
                          │                       │
                          └──────────┬────────────┘
                                     ▼
                                  Top-300
                                     │
                          ┌──────────┴──────────┐
                          ▼                     ▼
                    Query Content         Reference Box
                   [B,300,384]             [B,300,4]
                          │                     │
                          │               Query Pos MLP
                          │                     │
                          │               [B,300,384]
                          └──────────┬──────────┘
                                     ▼
                              Decoder Layer 0
                              ├ Self-Attn
                              ├ Deformable Cross-Attn
                              ├ Gate
                              ├ FFN
                              └ Box Refinement
                                     ↓
                              Decoder Layer 1
                                     ↓
                                   ...
                                     ↓
                              Decoder Layer 5
                                     │
                         ┌───────────┴───────────┐
                         ▼                       ▼
                 pred_logits                pred_boxes
                 [B,300,80]                [B,300,4]
```

---

# 57. PyTorch 实验 1：三个尺度 Flatten

```python
import torch

B = 2
D = 384

p3 = torch.randn(B,D,80,80)
p4 = torch.randn(B,D,40,40)
p5 = torch.randn(B,D,20,20)

def flatten_level(x):
    return x.flatten(2).permute(0,2,1)

f3 = flatten_level(p3)
f4 = flatten_level(p4)
f5 = flatten_level(p5)

print(f3.shape)
print(f4.shape)
print(f5.shape)

memory = torch.cat([f3,f4,f5], dim=1)

print(memory.shape)
```

预期：

```text
[2,6400,384]
[2,1600,384]
[2, 400,384]
[2,8400,384]
```

---

# 58. PyTorch 实验 2：Top-K Query Selection

```python
B = 2
N = 8400
D = 384
C = 80
K = 300

memory = torch.randn(B,N,D)
logits = torch.randn(B,N,C)
boxes  = torch.randn(B,N,4)

scores = logits.max(dim=-1).values

topk_scores, topk_idx = torch.topk(
    scores,
    k=K,
    dim=1
)

topk_memory = torch.gather(
    memory,
    dim=1,
    index=topk_idx[...,None].expand(-1,-1,D)
)

topk_boxes = torch.gather(
    boxes,
    dim=1,
    index=topk_idx[...,None].expand(-1,-1,4)
)

print(topk_memory.shape)
print(topk_boxes.shape)
```

预期：

```text
[2,300,384]
[2,300,4]
```

这段实验非常重要，因为它就是：

```text
8400 → 300
```

的最小复现。

---

# 59. PyTorch 实验 3：Reference Box → Query Position

```python
import torch.nn as nn

B = 2
Q = 300
D = 384

refs = torch.rand(B,Q,4)

query_pos_head = nn.Sequential(
    nn.Linear(4,2*D),
    nn.ReLU(),
    nn.Linear(2*D,D)
)

query_pos = query_pos_head(refs)

print(query_pos.shape)
```

输出：

```text
[2,300,384]
```

真实源码使用自己的 MLP 与配置激活；这个实验只复现 shape 与概念。

---

# 60. PyTorch 实验 4：Query Self-Attention

```python
attn = nn.MultiheadAttention(
    embed_dim=384,
    num_heads=8,
    batch_first=True
)

content = torch.randn(2,300,384)
pos = torch.randn(2,300,384)

q = content + pos
k = content + pos
v = content

out, _ = attn(
    q,k,v,
    need_weights=False
)

print(out.shape)
```

输出：

```text
[2,300,384]
```

---

# 61. PyTorch 实验 5：Iterative Refinement

```python
def inverse_sigmoid(x, eps=1e-6):
    x = x.clamp(eps,1-eps)
    return torch.log(x/(1-x))

ref = torch.tensor([
    [0.40,0.50,0.20,0.30]
])

delta = torch.tensor([
    [0.20,-0.10,0.30,0.10]
])

new_ref = torch.sigmoid(
    inverse_sigmoid(ref) + delta
)

print("old:", ref)
print("new:", new_ref)
```

再比较：

```python
direct = ref + delta
print("direct:", direct)
```

尝试：

```text
ref = 0.95
delta = +0.30
```

你会发现直接相加可能：

```text
> 1
```

而 logit-space refinement 后仍然：

```text
0~1
```

---

# 62. PyTorch 实验 6：FDR 的 132 个 Logits

```python
B = 2
Q = 300
reg_max = 32

pred = torch.randn(
    B,
    Q,
    4*(reg_max+1)
)

print(pred.shape)

pred = pred.reshape(
    B,
    Q,
    4,
    reg_max+1
)

print(pred.shape)

prob = pred.softmax(dim=-1)

print(prob.shape)
```

输出：

```text
[2,300,132]
[2,300,4,33]
[2,300,4,33]
```

第 12 课会继续把：

```text
[B,300,4,33]
→ Integral
→ distances
→ bbox
```

完整实现。

---

# 63. PyTorch 实验 7：模拟 6 层 Refinement

```python
B = 1
Q = 300

ref = torch.rand(
    B,Q,4
).clamp(0.05,0.95)

refs = [ref]

for layer in range(6):

    delta = (
        torch.randn_like(ref)
        * 0.1
    )

    ref = torch.sigmoid(
        inverse_sigmoid(ref)
        + delta
    )

    refs.append(ref)

for i,r in enumerate(refs):
    print(
        i,
        r.mean().item(),
        r.min().item(),
        r.max().item()
    )
```

真实网络的 delta 当然不是随机，而是：

```text
由当前 Query Feature 学出来的。
```

---

# 64. 一个最小版 Query Selector

```python
class ToyQuerySelector(nn.Module):

    def __init__(
        self,
        dim=384,
        num_classes=80,
        num_queries=300
    ):
        super().__init__()

        self.score = nn.Linear(
            dim,
            num_classes
        )

        self.box = nn.Linear(
            dim,
            4
        )

        self.num_queries = num_queries

    def forward(self, memory):

        logits = self.score(memory)
        boxes = self.box(memory)

        scores = logits.max(-1).values

        idx = scores.topk(
            self.num_queries,
            dim=1
        ).indices

        queries = torch.gather(
            memory,
            1,
            idx[...,None].expand(
                -1,-1,memory.shape[-1]
            )
        )

        refs = torch.gather(
            boxes,
            1,
            idx[...,None].expand(
                -1,-1,4
            )
        )

        return queries, refs
```

测试：

```python
selector = ToyQuerySelector()

memory = torch.randn(
    2,8400,384
)

q,refs = selector(memory)

print(q.shape)
print(refs.shape)
```

输出：

```text
[2,300,384]
[2,300,4]
```

---

# 65. 一个学习版 Decoder Layer 伪代码

不要把下面当官方源码；它用于建立结构地图：

```python
def decoder_layer(
    query,
    reference_box,
    memory,
    spatial_shapes
):

    query_pos = box_to_pos_embedding(
        reference_box
    )

    # 1. Query ↔ Query
    query = self_attention(
        query + query_pos,
        query + query_pos,
        query
    )

    # 2. Query ↔ Image
    image_info = deformable_attention(
        query=query + query_pos,
        reference_box=reference_box,
        memory=memory,
        spatial_shapes=spatial_shapes
    )

    # 3. Fuse
    query = gated_fusion(
        query,
        image_info
    )

    # 4. Channel transformation
    query = ffn(query)

    return query
```

---

# 66. Decoder 外层伪代码

```python
query = initial_content
ref = initial_reference

for layer in decoder_layers:

    query = layer(
        query,
        ref,
        memory,
        spatial_shapes
    )

    coarse_delta = pre_bbox_head(query)

    coarse_box = refine(
        ref,
        coarse_delta
    )

    distribution = bbox_head(query)

    fine_box = integral_refine(
        coarse_box,
        distribution
    )

    logits = class_head(query)

    save(logits, fine_box)

    ref = fine_box.detach()
```

这就是 D-FINE Decoder 的整体精神。

下一课我们把：

```text
integral_refine(...)
```

彻底展开。

---

# 67. 推荐真实源码阅读顺序

打开：

```text
engine/deim/dfine_decoder.py
```

不要从第一行机械读到底。

推荐：

```text
1. DFINETransformer.__init__

2. DFINETransformer.forward

3. encoder input / anchor generation helpers

4. _select_topk

5. decoder input construction

6. TransformerDecoder.forward

7. TransformerDecoderLayer.forward

8. MSDeformableAttention.forward

9. Integral

10. LQE
```

这基本就是数据真正流动的顺序。

---

# 68. 阅读 `DFINETransformer.__init__` 时先找什么？

先只找：

```text
input_proj

enc_output
enc_score_head
enc_bbox_head

tgt_embed
query_pos_head

decoder

dec_score_head
pre_bbox_head
dec_bbox_head

integral

denoising_class_embed
```

然后分别问：

```text
谁负责 Proposal？
谁负责 Query？
谁负责 Class？
谁负责 Box？
谁负责 Denoising？
```

先建立角色，不要一开始陷入初始化代码。

---

# 69. 阅读 `forward()` 时只追 7 个阶段

```text
① feats → memory

② generate anchors

③ build denoising queries

④ encoder proposals

⑤ Top-K query selection

⑥ decoder

⑦ split / package outputs
```

只要这七段不丢，Decoder 源码就不会乱。

---

# 70. 推荐断点

### 断点 A：DFINETransformer.forward 开头

应该看到：

```text
[B,384,80,80]
[B,384,40,40]
[B,384,20,20]
```

### 断点 B：Flatten 后

```text
memory
[B,8400,384]
```

### 断点 C：Encoder Score 后

```text
[B,8400,80]
```

### 断点 D：Top-K 后

```text
query content
[B,300,384]

reference
[B,300,4]
```

### 断点 E：Decoder 第一层前

推理时：

```text
target
[B,300,384]

reference_points
[B,300,4]

value
[B,8400,384]
```

### 断点 F：MSDeformableAttention 内

观察：

```text
sampling_offsets
attention_weights
sampling_locations
```

### 断点 G：每层 Box Refinement 后

观察：

```text
Layer 0 reference
Layer 1 reference
...
Layer 5 reference
```

---

# 71. Debug 不要打印整个 Tensor

不要：

```python
print(memory)
```

推荐：

```python
print(memory.shape)
print(memory.mean())
print(memory.std())
print(memory.min())
print(memory.max())
```

还可以：

```python
print(memory[0,:3,:8])
```

只看局部。

检测 NaN：

```python
def check(name, x):
    print(
        name,
        tuple(x.shape),
        "nan:",
        torch.isnan(x).any().item(),
        "inf:",
        torch.isinf(x).any().item(),
        "mean:",
        x.mean().item(),
        "std:",
        x.std().item()
    )
```

---

# 72. 本课最重要的 Shape 表

| 阶段 | Shape |
|---|---|
| P3 | `[B,384,80,80]` |
| P4 | `[B,384,40,40]` |
| P5 | `[B,384,20,20]` |
| Flatten P3 | `[B,6400,384]` |
| Flatten P4 | `[B,1600,384]` |
| Flatten P5 | `[B,400,384]` |
| Memory | `[B,8400,384]` |
| Encoder logits | `[B,8400,80]` |
| Encoder boxes | `[B,8400,4]` |
| Top-K indices | `[B,300]` |
| Query content | `[B,300,384]` |
| Initial refs | `[B,300,4]` |
| Query position | `[B,300,384]` |
| Decoder feature | `[B,300,384]` |
| Class logits | `[B,300,80]` |
| FDR corner logits | `[B,300,132]` |
| Final boxes | `[B,300,4]` |

建议把这张表作为第 11 课的速查表。

---

# 73. 初学者最容易犯的错误

### 错误 1

```text
8400 是 Object Query 数。
```

错。

```text
8400 = spatial memory positions
300  = object queries
```

### 错误 2

```text
300 Queries 是 300 个固定 learnable embeddings。
```

当前默认 `learn_query_content=False` 时不是。

正常 Query Content 来自：

```text
Top-K selected encoder memory
```

### 错误 3

```text
Query = Bounding Box。
```

错。

```text
Query Content = 384D feature
Reference Box = 4D geometry
```

### 错误 4

```text
Top-K 就是 Hungarian Matching。
```

完全不同。

### 错误 5

```text
Top-K 就是 NMS。
```

也不是。Top-K 不做传统 IoU suppression。

### 错误 6

```text
Decoder 每层都重新生成 300 Queries。
```

不是。

同一组 Object Slots 被逐层更新。

### 错误 7

```text
Deformable Attention 仍然看全部 8400 positions。
```

它围绕 reference 在多个尺度采样少量位置。

### 错误 8

```text
132 = 132 个 Boxes。
```

不是。

```text
4 directions × 33 bins
```

### 错误 9

```text
num_denoising=100 → 训练永远 400 Queries。
```

实际 DN 数应以 `dn_meta` 和真实 shape 为准。

### 错误 10

```text
detach 后前面的模块就完全不训练。
```

不是。Encoder Proposal Stage 仍有自己的直接监督。

---

# 74. 本课练习

1. 手算：

\[
80^2+40^2+20^2
\]

为什么得到 8400？

2. 为什么 8400 不是 Object Query 数？

3. `enc_score_head` 输入 `[B,8400,384]`，COCO 下输出是什么 shape？

4. 默认 query selection 怎样从 `[B,8400,80]` 得到 `[B,300]` indices？

5. 为什么必须用同一组 Top-K indices 同时 gather memory、logits 和 boxes？

6. `learn_query_content=False` 时 Query Content 来自哪里？

7. Query Content 和 Reference Box 的职责分别是什么？

8. 为什么 Reference Box 可以动态生成 Query Positional Embedding？

9. `hidden_dim=384,nhead=8`，每个 head dimension 是多少？

10. 300 Queries 的 Self-Attention 有多少 Query Pairs？

11. `[3,6,3]` 表示什么？

12. 300 Queries、8 Heads、12 Sampling Points，每层每张图有多少 query-head-sampling locations？

13. 为什么 Deformable Attention 比 300×8400 的 Dense Cross-Attention 更适合实时检测？

14. 写出 Iterative Refinement 的核心公式。

15. 为什么 Top-K Query Selection 和 Hungarian Matching 完全不同？

16. `reg_max=32` 时为什么 bbox distribution head 输出 132？

17. 为什么 Training 时 Query 数可能大于 300？

18. 为什么 Inference 没有 Denoising Queries？

19. 为什么 Encoder Proposal Stage 需要自己的监督？

20. 解释：

```text
better reference
→ better sampling
→ better query feature
→ better box
```

---

# 75. 练习答案摘要

### 1

\[
80^2+40^2+20^2
=
6400+1600+400
=
8400
\]

### 3

```text
[B,8400,80]
```

### 4

先：

```text
max over classes
```

得到：

```text
[B,8400]
```

再：

```text
topk(K=300)
```

得到：

```text
[B,300]
```

indices。

### 6

来自：

```text
Top-K selected encoder memory
```

### 9

\[
384/8=48
\]

### 10

\[
300^2=90,000
\]

### 11

三个 level 每个 head 分别采样：

```text
3
6
3
```

总：

\[
12
\]

### 12

\[
300	imes8	imes12=28,800
\]

### 14

\[
r_{i+1}
=
\sigma
\left(
\sigma^{-1}(r_i)+\Delta_i
ight)
\]

### 16

\[
4	imes(32+1)=132
\]

即：

```text
4 个方向
×
每方向 33 个 distribution logits
```

### 17

训练还会加入 Denoising Queries。

### 18

Inference 没有 GT，无法构造 GT-based Denoising Group。

---

# 76. 本课必须掌握的 15 个问题

进入第 12 课前，请确保能独立解释：

```text
1. P3/P4/P5 怎样变成 8400 memory positions？

2. spatial_shapes 为什么不能丢？

3. 8400 个 anchors/proposals 为什么存在？

4. Encoder Score Head 在做什么？

5. Encoder BBox Head 在做什么？

6. Top-300 到底依据什么选？

7. 默认 Query Content 来自哪里？

8. Reference Box 来自哪里？

9. Query Content 与 Query Position 有什么区别？

10. Self-Attention 为什么是 Query ↔ Query？

11. Deformable Cross-Attention 为什么是 Query ↔ Image？

12. [3,6,3] 到底是什么意思？

13. Box 为什么逐层 Refinement？

14. Training 为什么会出现 Denoising Queries？

15. Top-K 与 Hungarian Matching 为什么完全不同？
```

如果这 15 个问题能自己讲出来，你就不再只是“知道 DETR 有 Object Query”，而是真正知道 `deim_dfine_x` 的 Query 在代码里怎样出生、怎样看图像、怎样一步步变成目标预测。

---

# 77. 本课最核心的一句话

> **DFINETransformer 把 HybridEncoder 的 8400 个 dense multi-scale spatial features 转换成 300 个 sparse object-centric queries：先通过 encoder proposal scoring 选择高质量候选，以候选 feature 作为 Query Content、候选 box 作为 Reference，再通过多层 Query Self-Attention、Multi-Scale Deformable Cross-Attention 和 Iterative Refinement，让每个 Query 逐步收敛成一个具体目标预测。**

---

# 78. 下一课：D-FINE 的精细边界框回归——FDR

本课故意留下最大的黑盒：

```text
Query Feature
↓
132 Localization Logits
↓
???
↓
4D Bounding Box
```

第 12 课会把它彻底拆开：

```text
reg_max = 32
reg_scale = 8

4 × 33 distributions

softmax

Integral

weighting_function

distance2bbox

reference points

Fine-grained Distribution Refinement
```

我们会从一个很小的：

```text
reg_max = 4
```

实验开始，亲手把 distribution logits 还原成 bbox，然后逐行映射回 D-FINE 的真实实现。

---

# 79. 源码核对说明

本课按当前官方 DEIM / DEIMv2 公开代码路线核对。DEIMv2 官方仓库继续包含 `DFINETransformer` 路线；DEIM 的 `engine/deim/dfine_decoder.py` 中可以直接看到本课涉及的核心结构，包括：

```text
DFINETransformer
TransformerDecoder
TransformerDecoderLayer
MSDeformableAttention
Top-K Query Selection
Denoising
Integral
LQE
```

课程主线 X 配置对应：

```text
HGNetv2-B5
HybridEncoder hidden_dim=384
DFINETransformer feat_channels=[384,384,384]
reg_scale=8
```

基础配置提供：

```text
feat_strides=[8,16,32]
num_levels=3
num_layers=6
num_queries=300
num_denoising=100
reg_max=32
num_points=[3,6,3]
query_select_method=default
cross_attn_method=default
```

因此本课核心 Shape：

```text
[B,384,80,80]
[B,384,40,40]
[B,384,20,20]

→ [B,8400,384]

→ Top-300

→ [B,300,384]
```

来自这条真实配置链。

特别注意：

```text
reg_scale
```

基础配置可能是：

```text
4
```

而 X 配置覆盖为：

```text
8
```

所以以后分析任何模型，都必须查看：

```text
最终 merged YAML
```

而不是只看 base config。

如果你的本地 checkout 与课程核对版本不同，请始终以：

```text
你的本地源码
+
最终 merged config
+
forward hook 实测 shape
```

为最高依据。

---

# 80. 到第 11 课结束时的完整模型地图

```text
Image
[B,3,640,640]

        ↓

HGNetv2-B5
第 9 课

        ↓

C3 / C4 / C5

        ↓

HybridEncoder
第 10 课

        ↓

P3 [B,384,80,80]
P4 [B,384,40,40]
P5 [B,384,20,20]

        ↓

DFINETransformer
第 11 课

        ↓

memory
[B,8400,384]

        ↓

Encoder Proposals
8400 candidates

        ↓

Top-300

        ↓

Query Content
[B,300,384]

Reference Boxes
[B,300,4]

        ↓

Decoder × 6

        ├── Self-Attention
        ├── Deformable Cross-Attention
        ├── Gate
        ├── FFN
        └── Iterative Refinement

        ↓

pred_logits
[B,300,80]

pred_boxes
[B,300,4]

        ↓

Hungarian Matching
第 6 课

        ↓

Loss
```

到这里，我们已经第一次完整贯通：

```text
Image
→ Backbone
→ HybridEncoder
→ Query Initialization
→ Transformer Decoder
→ Class / Box Prediction
```

第 12 课开始，我们会钻进 D-FINE 最关键的定位内部机制：**FDR**。
