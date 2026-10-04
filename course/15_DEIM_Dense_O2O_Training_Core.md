# DEIMv2 源码课程 · 第 15 课
# DEIM Dense O2O——训练框架真正的核心

> 本课目标：把 DEIM 最重要的训练思想彻底讲清楚——**为什么 DETR 的 O2O 监督稀疏，Dense O2O 到底“Dense”在哪里，为什么它不是 O2M，以及 Mosaic / MixUp 怎样在不改变 O2O 匹配规则和推理结构的前提下增加正样本监督。**
>
> 第 16 课将继续解决 Dense O2O 带来的“匹配质量参差不齐”问题：MAL。第 17 课再进入 DEIMv2 的 Mosaic + MixUp + CopyBlend 升级版。

---

## 1. 一句话抓住 Dense O2O

Dense O2O **不是**：

```text
1 GT → 多个 Query
```

而是：

```text
一张训练输入中的 GT 数量增加
        ↓
每个 GT 仍只匹配 1 个 Query
        ↓
总 positive matches 增加
```

所以它保持：

\[
M_i=1
\]

但增大：

\[
N_{\text{GT}}
\]

这就是 DEIM 最关键的思想。

---

## 2. 为什么普通 DETR 的 O2O 监督稀疏？

假设：

```text
num_queries = 300
GT objects = 6
```

Hungarian Matching 后可能是：

```text
GT0 ← q17
GT1 ← q41
GT2 ← q88
GT3 ← q103
GT4 ← q207
GT5 ← q291
```

只有 6 个 Query 成为这 6 个 GT 的 positive matches。

教学上定义一个直观指标：

\[
\rho=\frac{N_{\text{positive}}}{N_{\text{query}}}
\]

O2O 下，只要 GT 数没有超过 Query 数：

\[
N_{\text{positive}}\approx N_{\text{GT}}
\]

因此：

\[
\rho\approx\frac{N_{\text{GT}}}{Q}
\]

这里：

\[
\rho=\frac6{300}=2\%
\]

这不是论文正式评价指标，只是帮助我们理解 sparse supervision。

---

## 3. 增大 Batch 能解决吗？

假设：

```text
batch = 32
每图平均 GT = 6
```

则：

```text
positive = 32 × 6 = 192
queries  = 32 × 300 = 9600
```

比例仍然：

\[
192/9600=2\%
\]

所以增大 batch 增加了总样本量，却没有改变**单张输入内部的 O2O 稀疏结构**。

---

## 4. O2O 为什么天然受 GT 数量限制？

设一张图有 \(N\) 个 GT。

O2O 要求每个 target 的 positive match 数：

\[
M_i=1
\]

因此：

\[
N_{\text{positive}}
=
\sum_{i=1}^{N}M_i
=
N
\]

如果：

```text
N = 5
Q = 300
```

则只有约：

```text
5 / 300
```

的 Query 成为 GT-positive。

这就是 DEIM 论文针对的核心 optimization bottleneck。

---

## 5. O2M 为什么监督更密？

One-to-Many 允许：

```text
GT0
├─ q7
├─ q31
├─ q88
└─ q121
```

如果每个 GT 平均匹配 \(M>1\) 个预测：

\[
N_{\text{positive}}^{O2M}\approx N\times M
\]

例如：

```text
6 GT × 10 matches/GT = 60 positives
```

比 O2O 的：

```text
6 positives
```

密得多。

但这样会改变 assignment 结构，也可能带来重复预测及训练/推理设计上的额外问题。

DEIM 问的是：

> **能不能不放弃 O2O，也获得更密集的监督？**

---

## 6. DEIM 的关键转念：不增加 M，增加 N

普通 O2O：

\[
S=N\times1=N
\]

其中 \(S\) 是 positive supervision count。

O2M 的思路：

```text
N 不变
M ↑
```

DEIM 的思路：

```text
M = 1 不变
N ↑
```

即：

\[
S_{\text{Dense O2O}}=N'
\quad,\quad N'>N
\]

所以：

```text
O2M:
增加每个 GT 的 matches

Dense O2O:
增加一张训练输入中的 GT instances
```

这是两者最根本的区别。

---

## 7. 为什么叫 “Dense O2O”？

“Dense”修饰的是：

```text
training supervision / target density
```

而不是：

```text
per-GT assignment multiplicity
```

正确心智模型：

```text
Sparse targets + O2O
        ↓
Dense targets + O2O
```

不是：

```text
O2O → O2M
```

---

## 8. 论文中的极简例子

假设原图只有：

```text
1 object
```

普通 O2O：

```text
1 GT → 1 positive Query
```

如果把图像内容组合到四个区域，使 composite image 中存在：

```text
4 target instances
```

则：

```text
4 GT → 4 positive Queries
```

但每个 GT 仍：

```text
只匹配一个 Query
```

所以：

\[
M_i=1
\]

没变；变化的是：

\[
N:1\rightarrow4
\]

---

## 9. Mosaic 为什么特别适合 Dense O2O？

Mosaic 可以把多张图组合成一个训练输入：

```text
┌────────────┬────────────┐
│  Image A   │  Image B   │
├────────────┼────────────┤
│  Image C   │  Image D   │
└────────────┴────────────┘
```

假设：

```text
A: 5 GT
B: 7 GT
C: 4 GT
D: 8 GT
```

四张原图平均：

\[
(5+7+4+8)/4=6
\]

理想组合后：

```text
24 GT
```

于是同一个 model input 中可供 Hungarian O2O 匹配的 target 数大幅增加。

---

## 10. Positive Density 的具体变化

普通：

```text
Q = 300
GT = 6
```

\[
6/300=2\%
\]

Mosaic 后假设：

```text
GT = 24
```

\[
24/300=8\%
\]

也就是教学例子中：

```text
2% → 8%
```

约 4 倍。

真实训练中不会机械地永远是 4 倍，因为还会受到裁剪、缩放、box sanitize、增强概率和每张原图目标数的影响。

---

## 11. 真实 Mosaic 后为什么可能只有 21 个 GT？

例如源图总共有：

```text
24 GT
```

经过：

```text
缩放
裁剪
平移
```

后：

```text
2 个完全出界
1 个过小被过滤
```

最终：

```text
21 GT
```

那么 positive-query ratio：

\[
21/300=7\%
\]

仍明显高于原始的：

```text
2%
```

因此 Dense O2O 的重点是**统计意义上的 target densification**，不是强求固定倍数。

---

## 12. Dense O2O 到底改了 Hungarian Matcher 什么？

最重要的答案：

> **没有把 Hungarian 的一对一约束改成一对多。**

改变的是 matcher 输入的 target 数。

普通：

```text
predictions = 300
targets     = 6

cost matrix:
[300, 6]
```

Dense：

```text
predictions = 300
targets     = 24

cost matrix:
[300, 24]
```

Hungarian 仍然求一对一 assignment。

只要：

```text
N_GT <= Q
```

就可以大致得到：

```text
N_GT 个 positive matches
```

---

## 13. 一个公式看懂全部

普通 O2O：

\[
C\in\mathbb{R}^{Q\times N}
\]

Dense O2O：

\[
C'\in\mathbb{R}^{Q\times N'}
\]

其中：

\[
N'>N
\]

但求解仍是：

\[
\sigma^*
=
\arg\min_\sigma
\sum_i C_{i,\sigma(i)}
\]

没有把：

```text
一个 target
```

复制成多个 matching slots。

---

## 14. 为什么 Dense O2O 不破坏 Set Prediction？

Composite image 中的 target：

\[
Y=\{y_1,y_2,\dots,y_{N'}\}
\]

仍是当前输入坐标系里的**不同 object instances**。

即使多个 instance 类别都是：

```text
person
```

它们仍是：

```text
person #1
person #2
...
```

Hungarian 对 instance set 做一对一匹配，所以 set prediction formulation 完全成立。

---

## 15. Dense O2O 不是“把同一个 GT 复制 4 份”

错误做法：

```text
同一个 box
GT_A1
GT_A2
GT_A3
GT_A4
```

完全重叠，却要求多个 Query 去预测同一个物体。

DEIM 不是这样。

Mosaic 中来自不同图像的目标经过空间变换后位于：

```text
不同位置
不同尺度
不同上下文
```

它们在 composite image 中是独立实例。

---

## 16. 为什么 300 Queries 通常还有空间？

普通 COCO 图像：

```text
GT 数量通常远低于 300
```

所以：

```text
Q=300
```

本身有较大的 object-slot capacity。

Dense O2O 做的是：

> **更充分地利用已经存在的 Query capacity。**

而不是首先把：

```text
300 → 600 → 1200
```

来获得更多 positives。

---

## 17. 为什么不直接增加 Query 数？

Decoder self-attention 含有随 Query 数近似平方增长的部分：

\[
O(Q^2)
\]

例如：

\[
300^2=90,000
\]

\[
600^2=360,000
\]

Query 翻倍，pair interactions 约变为 4 倍。

Dense O2O 更聪明的地方在于：

```text
不扩大模型 Query budget
而提高已有 Query 的 positive utilization
```

---

## 18. 如果 augmentation 后 GT > 300 呢？

O2O capacity：

\[
N_{\text{positive}}\le Q
\]

如果：

```text
GT = 420
Q  = 300
```

不可能让 420 个 GT 全部获得独立 Query。

因此在密集自定义数据集上必须统计：

```text
augmented targets/image
```

并和：

```text
num_queries
```

比较。

推荐统计：

```text
P50
P90
P95
P99
max
```

如果：

```text
P99 augmented GT = 80
Q = 300
```

capacity 很充足。

如果：

```text
P99 augmented GT = 400
Q = 300
```

就必须重新评估 Query 数和 augmentation recipe。

---

## 19. Dense O2O 为什么能帮助收敛？

训练早期最缺的是有效的 object-level positive supervision。

普通：

```text
300 queries
6 positives
```

Dense：

```text
300 queries
20~30 positives（示意）
```

于是一次 optimizer step 中：

```text
更多 Query 获得正类监督
更多 boxes 获得 localization supervision
更多 object-query pairs 参与优化
```

所以每个 training input 能提供更多有效检测监督。

---

## 20. 这不是简单的“增加数据集长度”

普通 augmentation：

```text
1 source image
→
1 transformed training image
```

Dense O2O 的关键量是：

```text
targets per final model input
```

而不是：

```text
dataset length
```

它把原本分散在多张图中的 object supervision 重新打包进更密集的训练输入。

---

## 21. 为什么称 DEIM 为训练框架？

DEIM 的核心并不要求你必须使用某一个特定 detector architecture。

论文把它与：

```text
RT-DETR
D-FINE
```

等 DETR detector 结合。

核心改变集中在：

```text
training sample construction
matching supervision density
classification objective
training recipe
```

所以它更像：

```text
DETR training framework
```

而不是：

```text
一个全新的 backbone/decoder。
```

---

## 22. 真实 DEIM 配置：Dense O2O 就写在数据管线里

当前官方 `configs/base/deim.yml` 的相关部分明确标记：

```yaml
# Dense O2O

train_dataloader:
  dataset:
    transforms:
      ops:
        - {type: Mosaic, ...}
        - {type: RandomPhotometricDistort, p: 0.5}
        - {type: RandomZoomOut, fill: 0}
        - {type: RandomIoUCrop, p: 0.8}
        - {type: SanitizeBoundingBoxes, min_size: 1}
        - {type: RandomHorizontalFlip}
        - {type: Resize, size: [640, 640]}
        - {type: SanitizeBoundingBoxes, min_size: 1}
        - {type: ConvertPILImage, dtype: 'float32', scale: True}
        - {type: ConvertBoxes, fmt: 'cxcywh', normalize: True}
```

这是非常关键的源码阅读经验：

> 论文名字叫 “Dense O2O Matching”，但你不能只去 matcher 文件寻找它。

它的关键入口首先是：

```text
data pipeline
```

---

## 23. 为什么仓库里不一定有 `DenseO2OMatcher`？

因为 Dense O2O 不是：

```python
class DenseO2OMatcher(...)
```

这种“换 matcher 算法”的实现。

实际逻辑是：

```text
Mosaic / MixUp
        ↓
targets per input ↑
        ↓
same Hungarian Matcher
        ↓
O2O positive pairs ↑
```

所以搜索源码时要沿：

```text
config
→ transforms
→ collate
→ targets
→ matcher
```

这条路径追。

---

## 24. 当前 DEIM Base 中 Mosaic 的关键配置

```yaml
type: Mosaic
output_size: 320
rotation_range: 10
translation_range: [0.1, 0.1]
scaling_range: [0.5, 1.5]
probability: 1.0
fill_value: 0
use_cache: False
max_cached_images: 50
random_pop: True
```

随后 pipeline 还有：

```yaml
Resize:
  size: [640, 640]
```

所以：

```text
output_size=320
```

不能简单理解成模型最终输入为 320×320。

第 17 课会逐行拆 Mosaic 的真实几何坐标变换。

---

## 25. `SanitizeBoundingBoxes` 为什么非常重要？

经过：

```text
Mosaic
crop
scale
translation
```

后，一些 box 可能：

```text
完全出界
只剩极小区域
宽高为 0
退化
```

所以必须过滤非法 box。

这也说明：

```text
最终 target count
```

不能简单等于：

```text
4 张源图 GT 数量之和。
```

---

## 26. 哪些增强真正直接增加 target density？

### 主要可以增加 target count

```text
Mosaic
MixUp
DEIMv2 的 CopyBlend
```

### 主要改变 appearance / geometry

```text
RandomPhotometricDistort
RandomZoomOut
RandomIoUCrop
HorizontalFlip
Resize
```

其中 RandomIoUCrop 甚至可能减少 GT 数量。

所以：

> “DEIM 的所有 augmentation 都等于 Dense O2O”是不准确的。

Dense O2O 是整套 recipe 的核心视角，但各 augmentation 还有其他职责。

---

## 27. MixUp 为什么配置在 `collate_fn`？

当前 DEIM base：

```yaml
collate_fn:
  mixup_prob: 0.5
  mixup_epochs: [4, 29]
```

而 Mosaic 位于：

```text
dataset transforms
```

这说明两者处于数据管线不同阶段。

概念上：

```text
Dataset sample
↓
Mosaic / transforms
↓
sample tensors
↓
Collate
↓
MixUp / batch-level processing
↓
batch
```

第 17 课会进入 `BatchImageCollateFunction` 逐行追踪。

---

## 28. Mosaic 和 MixUp 对 target density 的区别

Mosaic：

```text
空间拼接多个来源
→ objects 位于不同区域
```

MixUp：

```text
混合两个 image samples
→ 两边 targets 一起保留
```

因此二者都可以增加：

```text
targets per final input
```

但视觉分布和遮挡模式不同。

---

## 29. 为什么不能“增强越强越好”？

target density 上升同时会带来：

```text
目标变小
遮挡增加
人工上下文
边界裁切
matching 难度增加
```

因此：

```text
更多 positives
```

并不等于：

```text
所有 positives 都高质量。
```

这正是 DEIM 第二个核心贡献 MAL 出现的原因。

---

## 30. Dense O2O 的核心 Trade-off

```text
Target count ↑
      │
      ├── Positive supervision ↑
      │
      └── Hard / low-quality matches ↑
```

所以完整 DEIM 不是：

```text
Dense O2O
```

单独工作，而是：

```text
Dense O2O
+
MAL
+
augmentation schedule
```

---

## 31. 一个极端匹配例子

训练早期：

```text
GT: cat
```

Hungarian 最终选：

```text
Query 37
```

但：

```text
IoU(q37, GT) = 0.05
```

为什么仍会被匹配？

因为 Hungarian 求的是：

```text
当前候选中的全局最优 assignment
```

不是保证：

```text
每个 matched pair 的绝对质量都很高。
```

因此：

```text
matched positive
≠
high-quality prediction
```

这是第 16 课理解 MAL 的钥匙。

---

## 32. Dense O2O 没有制造“假 GT”

Mosaic / MixUp 的 annotations 仍来自真实标注，并经过正确的几何坐标变换。

所谓：

```text
low-quality match
```

指：

```text
当前模型 prediction 与 GT 的匹配质量低
```

不是：

```text
GT annotation 是假的。
```

---

## 33. Dense O2O 与 Denoising 不是一回事

### Dense O2O

```text
更多真实/组合 targets
↓
Hungarian O2O positives ↑
```

### Denoising

```text
GT
↓
添加 label / box noise
↓
构造已知 GT identity 的 DN queries
```

DN queries 不需要 Hungarian 去重新寻找身份。

二者作用点完全不同。

---

## 34. Dense O2O 与 Global Union 也不是一回事

第 14 课我们看到：

```text
Global Union Set
```

汇总不同 decoder layers 的 localization matches。

### Dense O2O

```text
增加 target 数
```

### Global Union

```text
同一 target 在不同 decoder layer
可能出现不同 localization candidates
→ 取 union
```

所以这是两种不同的“监督变密”。

---

## 35. 三种“增加监督”必须分清

```text
A. Dense O2O
   data/target level
   targets per image ↑

B. Denoising
   decoder training query level
   noisy GT queries ↑

C. Global Union / GO-LSD
   criterion level
   cross-layer localization candidates ↑
```

如果把这三者混成一个概念，后面读 `DEIMCriterion` 会非常混乱。

---

## 36. Dense O2O 如何影响 `deim_dfine_x` 的 Loss？

第 14 课：

```text
pred_logits
[B,300,80]

pred_boxes
[B,300,4]

pred_corners
[B,300,132]
```

这些 prediction shape 不因为 Dense O2O 改变。

改变的是：

```text
targets
[N,4]
```

例如：

```text
N: 6 → 24
```

于是：

```text
Hungarian matched pairs ↑
```

进一步让更多 predictions 参与：

```text
MAL
L1
GIoU
FGL
```

所以数据管线的改变最终一路传到 loss graph。

---

## 37. 普通 O2O vs Dense O2O 的完整 Shape 对比

假设：

```text
Q = 300
```

普通：

```text
targets:
[6,4]

cost:
[300,6]

matched:
6
```

Dense：

```text
targets:
[24,4]

cost:
[300,24]

matched:
24
```

但模型输出仍：

```text
pred_logits [300,80]
pred_boxes  [300,4]
```

这就是 Dense O2O 最优雅的地方：

```text
prediction side: 不变
target side:     更密
matching rule:   不变
inference:       不变
```

---

## 38. 为什么 Dense O2O 不增加推理结构？

推理阶段没有：

```text
Mosaic
MixUp
Hungarian Matcher
MAL
Dense target construction
```

仍然只是：

```text
image
↓
backbone
↓
encoder
↓
decoder
↓
predictions
↓
postprocess
```

所以 Dense O2O 本身不要求：

```text
额外 decoder
额外 head
额外 NMS
```

这对 Real-Time DETR 很重要。

---

## 39. “没有额外成本”应该怎样严谨表述？

不能说 Dense O2O 完全没有训练成本。

它仍有：

```text
Mosaic / MixUp 数据处理
box 坐标变换
更大的 matcher cost matrix
更多 matched loss 项
cache / collate 开销
```

更准确的说法是：

> 它不需要为了增加 positives 而增加额外的推理分支或把模型改造成 O2M inference architecture。

---

## 40. Matcher Cost 的变化

普通：

```text
300 × 6 = 1,800 cost entries
```

Dense：

```text
300 × 24 = 7,200 entries
```

matcher 计算增加。

但 detector 的主要 tensor：

```text
feature maps
[B,300,D] decoder queries
decoder layers
```

并不会因为 GT 从 6 变成 24 而变成 4 倍。

因此它与“复制多套 decoder”是完全不同的成本结构。

---

## 41. Epoch Policy：为什么强增强不是全程固定？

当前 DEIM base 中可看到：

```yaml
policy:
  epoch: [4, 29, 50]
  ops:
    - Mosaic
    - RandomPhotometricDistort
    - RandomZoomOut
    - RandomIoUCrop

mosaic_prob: 0.5
```

以及：

```yaml
mixup_prob: 0.5
mixup_epochs: [4, 29]
stop_epoch: 50
```

这说明增强策略是：

```text
epoch-dependent
```

而不是训练从头到尾完全不变。

具体每个 epoch 的开关语义必须结合当前 Python policy 实现读取，不能仅凭三个数字猜测。

---

## 42. 为什么前期、后期需要不同策略？

可以用一个教学上的三阶段直觉理解：

```text
stabilize
   ↓
densify / strong augmentation
   ↓
refine / weak augmentation
```

训练非常早期：

```text
prediction 接近随机
matching quality 很差
```

过强组合可能增加 optimization 难度。

中期：

```text
更多 targets
更多 object diversity
```

有助于提高训练信号密度。

后期：

```text
逐渐回到更接近真实测试分布
```

有助于精细 localization 和 confidence 收敛。

---

## 43. Dense O2O 和 Multi-Scale Training 不要混淆

Dense O2O：

```text
关注 target count
```

Multi-scale：

```text
关注 input H×W
```

Multi-scale 会改变：

```text
P3/P4/P5 shape
memory token count
```

但不一定增加 GT 数。

两者可以同时使用，但不是同一机制。

---

## 44. 为什么 RandomIoUCrop 可能减少 GT 还要用？

因为训练 recipe 的目标不是：

\[
\max N_{GT}
\]

而是：

\[
\max \text{useful supervision}
\]

有用监督还取决于：

```text
数据多样性
目标尺度
上下文
匹配质量
泛化
```

所以 Dense O2O 不是“GT 数量竞赛”。

---

## 45. 一个更完整的因果链

```text
DETR uses O2O
        ↓
1 GT → 1 positive Query
        ↓
few GT per image
        ↓
few positive queries
        ↓
sparse supervision
        ↓
slow optimization
        ↓
increase targets per training input
        ↓
Mosaic / MixUp
        ↓
same Hungarian O2O
        ↓
more positive matches
        ↓
denser supervision
        ↓
but more low-quality matches
        ↓
MAL
```

这就是 DEIM 论文最核心的逻辑链。

---

# 46. PyTorch 小实验：Positive Density

```python
def positive_density(num_gt, num_queries=300):
    positives = min(num_gt, num_queries)
    return positives, positives / num_queries

for n in [1, 3, 6, 12, 24, 48]:
    p, r = positive_density(n)
    print(
        f"GT={n:2d}",
        f"positive={p:2d}",
        f"density={r:.2%}"
    )
```

预期：

```text
GT= 1 positive= 1 density=0.33%
GT= 3 positive= 3 density=1.00%
GT= 6 positive= 6 density=2.00%
GT=12 positive=12 density=4.00%
GT=24 positive=24 density=8.00%
GT=48 positive=48 density=16.00%
```

---

# 47. 小实验：模拟 Mosaic Target Count

```python
counts = [5, 7, 4, 8]

original_mean = sum(counts) / len(counts)
mosaic_count = sum(counts)

print("original mean:", original_mean)
print("mosaic:", mosaic_count)
```

输出：

```text
original mean: 6.0
mosaic: 24
```

理想 densification：

\[
24/6=4
\]

---

# 48. 小实验：考虑被裁掉的 Box

```python
before = 24
removed = 3

after = before - removed

print("targets:", after)
print("positive-query ratio:", after / 300)
```

得到：

```text
targets = 21
ratio = 7%
```

仍高于原始约 2%。

---

# 49. 小实验：Cost Matrix Shape

```python
import torch

Q = 300

for N in [6, 24]:
    cost = torch.randn(Q, N)
    print(cost.shape)
```

输出：

```text
torch.Size([300, 6])
torch.Size([300, 24])
```

Dense O2O 的关键是：

```text
target dimension ↑
```

不是给同一 target 建多个 assignment slots。

---

# 50. 小实验：真正做一次 Hungarian

```python
import numpy as np
from scipy.optimize import linear_sum_assignment

def match(num_queries, num_gt):
    cost = np.random.rand(num_queries, num_gt)
    q_idx, gt_idx = linear_sum_assignment(cost)
    return q_idx, gt_idx

for n in [6, 24]:
    q, gt = match(300, n)

    print(
        f"{n} GT -> {len(q)} matches",
        f"unique q={len(np.unique(q))}",
        f"unique gt={len(np.unique(gt))}"
    )
```

只要：

```text
GT <= Q
```

你会看到：

```text
6 GT  → 6 matches
24 GT → 24 matches
```

而且 query 和 target indices 都是一对一的。

---

# 51. 小实验：O2M 对照

```python
def o2m_positive_count(num_gt, matches_per_gt):
    return num_gt * matches_per_gt

print(
    o2m_positive_count(
        num_gt=6,
        matches_per_gt=10
    )
)
```

输出：

```text
60
```

所以：

```text
Dense O2O: N ↑, M=1
O2M:       N 固定, M>1
```

---

# 52. 小实验：检查 Query Capacity

```python
Q = 300

augmented_counts = [
    18, 41, 72, 155, 220, 301, 420
]

for n in augmented_counts:
    assignable = min(n, Q)
    coverage = assignable / n

    print(
        f"GT={n:3d}",
        f"assignable={assignable:3d}",
        f"coverage={coverage:.2%}"
    )
```

当：

```text
GT > 300
```

O2O capacity 开始不足。

---

# 53. 小实验：在真实 DataLoader 中统计 Target Count

训练调试时最值得先加：

```python
for images, targets in train_loader:

    counts = [
        len(t["boxes"])
        for t in targets
    ]

    print(
        "targets/image:",
        counts
    )

    break
```

分别比较：

```text
Dense O2O OFF
Dense O2O ON
```

不要先看 AP。

先验证最基本的问题：

> target density 是否真的增加了？

---

# 54. 小实验：Matcher 后统计 Positive Matches

```python
positive_counts = [
    len(src_idx)
    for src_idx, tgt_idx
    in indices
]

print(
    positive_counts
)
```

如果：

```text
GT < num_queries
```

positive count 应基本随 GT count 增长。

---

# 55. 小实验：验证 O2O 没有被破坏

```python
for src_idx, tgt_idx in indices:

    assert (
        len(torch.unique(src_idx))
        ==
        len(src_idx)
    )

    assert (
        len(torch.unique(tgt_idx))
        ==
        len(tgt_idx)
    )
```

即：

```text
每个 matched Query 唯一
每个 matched GT 唯一
```

Dense O2O 后仍应成立。

---

# 56. 小实验：统计 Query Utilization

定义教学指标：

\[
U=\frac{N_{\text{matched}}}{Q}
\]

```python
Q = 300

util = [
    len(src) / Q
    for src, tgt in indices
]

print(
    sum(util) / len(util)
)
```

比较：

```text
Mosaic/MixUp OFF
vs
ON
```

它能帮助你观察已有 Query capacity 被利用得是否更充分。

但它不是最终模型质量指标；仍必须看 IoU 和 AP。

---

# 57. 推荐实验：统计 Augmented GT 分布

```python
import numpy as np

counts = []

for images, targets in train_loader:
    counts.extend(
        len(t["boxes"])
        for t in targets
    )

arr = np.asarray(counts)

for p in [50, 90, 95, 99]:
    print(
        f"P{p}:",
        np.percentile(arr, p)
    )

print("max:", arr.max())
```

然后与：

```text
num_queries = 300
```

比较。

这是把 DEIM 用到自定义数据集时非常重要的 sanity check。

---

# 58. 推荐实验：记录 Matched IoU Histogram

分别在：

```text
Dense O2O OFF
Dense O2O ON
```

情况下统计：

```text
IoU(pred matched box, GT)
```

并画 histogram。

你要观察的不只是：

```text
positive 数是否增加
```

还要观察：

```text
低 IoU positives 是否也增加
```

这会直接把你带到第 16 课的 MAL。

---

# 59. 推荐 Ablation

固定：

```text
model
seed
optimizer
batch size
epochs
```

依次实验：

```text
A. Mosaic OFF, MixUp OFF
B. Mosaic ON,  MixUp OFF
C. Mosaic OFF, MixUp ON
D. Mosaic ON,  MixUp ON
```

记录：

```text
mean targets/image
mean positive matches/image
mean matched IoU
training throughput
AP
AP50
AP75
```

这样你看到的是机制，而不仅是最终分数。

---

# 60. Dense O2O 的训练成本应该怎样评估？

不要只看：

```text
images / second
```

因为 Dense augmentation 可能让数据管线稍慢，却让每个 input 包含更多 object supervision。

更有意义的是：

```text
time-to-target-AP
```

例如：

```text
达到 50 AP 需要多少小时？
```

而不是只问：

```text
每秒能跑多少 batch？
```

DEIM 论文报告在其实验条件下可显著缩短训练时间；摘要给出最高约 50% 的训练时间降低。这个数字是论文实验结果，不应理解为任何自定义数据集上的固定保证。

---

# 61. Dense O2O 为什么特别适合 DETR？

传统 dense detector 常具有：

```text
大量 anchors / points
+
O2M assignment
```

一个 GT 本来就可能贡献多个 positive samples。

而 DETR：

```text
1 GT → 1 Query
```

因此：

```text
GT count
```

对 positive count 的限制非常直接。

Dense O2O 正是从这个结构性特点出发设计的。

---

# 62. Dense O2O 与 Classification

普通：

```text
6 matched positives
294 unmatched queries
```

Dense 示例：

```text
24 matched positives
276 unmatched queries
```

positive classification supervision 增加。

但这些 positives 的定位质量差异也更大，所以 classification objective 需要理解：

```text
matchability / localization quality
```

这就是 MAL 的入口。

---

# 63. Dense O2O 与 Box Regression

更多 matched pairs 意味着更多 Query 参与：

```text
L1
GIoU
```

对于 `deim_dfine_x`，还会进一步影响：

```text
FDR / FGL localization supervision
```

因此 Dense O2O 不只是“分类正样本更多”。

它会增加整个 object-level detection supervision。

---

# 64. Dense O2O 与 GO-LSD 可以同时工作

Dense O2O：

```text
targets ↑
```

Global Union / GO-related localization：

```text
每个 target 的跨层 localization candidates ↑
```

两者可以叠加。

可以用一个教学直觉表示：

```text
更多 targets
×
更丰富的跨层 localization candidates
```

但 classification 的 O2O identity 仍要保持清楚。

---

# 65. Dense O2O 与 Denoising 也可以同时工作

```text
Dense target set
       │
       ├─ Hungarian O2O
       │
       └─ DN query construction
```

所以一张 Dense O2O composite image 中更多 GT，也可能影响 DN training group 的构造规模。

但 DN 仍然不是 Hungarian positive assignment。

---

# 66. DEIM 的系统设计不是一个开关

不要期待配置中只有：

```yaml
dense_o2o: true
```

真正系统是：

```text
Mosaic
+
MixUp
+
geometric transforms
+
SanitizeBoundingBoxes
+
epoch policy
+
Hungarian O2O
+
MAL
+
scheduler / no-aug stage
```

DEIMv2 又继续加入：

```text
CopyBlend
matcher schedule
```

所以它必须作为完整 training system 学习。

---

# 67. DEIMv2 怎样继续 Dense O2O？

当前 DEIMv2 X 配置明确标记：

```text
Dense O2O:
Mosaic + Mixup + CopyBlend
```

并配置类似：

```yaml
mixup_epochs: [4, 29]
copyblend_epochs: [4, 50]
stop_epoch: 50
```

说明 DEIMv2 没有抛弃 Dense O2O，而是继续增强：

```text
target densification
+
matching optimization
```

第 17 课会专门拆：

```text
Mosaic
MixUp
CopyBlend
```

及其 box 变换与 epoch schedule。

---

# 68. 为什么本课先不展开 CopyBlend？

因为最重要的是先把原理固定下来：

\[
\boxed{
\text{Increase }N
\quad
\text{while keeping }M_i=1
}
\]

如果一开始同时塞入：

```text
CopyBlend
matcher_change_epoch
iou_order_alpha
```

很容易把：

```text
Dense O2O 的本质
```

和：

```text
DEIMv2 的后续升级
```

混为一谈。

课程顺序：

```text
第15课：Dense O2O
第16课：MAL
第17课：Dense O2O v2 / CopyBlend
第18课：完整训练 Recipe
```

---

# 69. 常见误解总表

| 误解 | 正确理解 |
|---|---|
| Dense O2O = O2M | 错；每个 GT 仍只匹配一个 Query |
| Dense O2O = 增加 Query 数 | 错；核心是增加 targets/image |
| 需要新 Dense Matcher | 不一定；核心仍可用 Hungarian O2O |
| Mosaic 只是防过拟合 | 在 DEIM 中还承担 target densification |
| Dense O2O 会破坏 NMS-free | 不会；O2O assignment 保留 |
| GT 越多越好 | 错；受 Query capacity、匹配质量、分布偏移等限制 |
| Dense O2O 只影响分类 | 错；box/localization 监督也增加 |
| Dense O2O = DN | 错；作用阶段不同 |
| Dense O2O = Global Union | 错；一个增 targets，一个增跨层 localization candidates |
| matched = 高 IoU | 错；Hungarian 是相对最优，不保证绝对质量 |
| Mosaic 4 图 = 必然 4× GT | 错；crop/sanitize 会改变数量 |
| 完全没有训练成本 | 错；有 augmentation、target processing、matcher 成本 |
| 强增强应全程开启 | 错；DEIM 使用 epoch-dependent recipe |

---

# 70. 一张图总结 DEIM 的核心因果关系

```text
                  Ordinary DETR
                       │
                       ▼
                 O2O Matching
                       │
                       ▼
               1 GT → 1 Query
                       │
                       ▼
              few GT per image
                       │
                       ▼
              sparse positives
                       │
                       ▼
              slow optimization
                       │
                       ▼
                     DEIM
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
     Keep O2O                  More targets
     M_i = 1                   per input
          │                         │
          │                  Mosaic / MixUp
          │                         │
          └────────────┬────────────┘
                       ▼
                  Dense O2O
                       │
                       ▼
             more positive matches
                       │
                       ▼
              denser supervision
                       │
                       ▼
            more quality variation
                       │
                       ▼
                      MAL
```

---

# 71. 本课最重要的 12 个结论

1. DETR O2O 的 sparse supervision 与“每个 GT 只有一个 positive match”直接相关。
2. Dense O2O **不是** O2M。
3. Dense O2O 保持 \(M_i=1\)。
4. 它通过增加 \(N_{\text{GT}}\) 增加 positive matches。
5. Mosaic / MixUp 在 DEIM 中承担 target densification 的特殊角色。
6. Matcher 仍然可以是 Hungarian Matcher。
7. 改变的是 cost matrix 的 target dimension：`[Q,N] → [Q,N']`。
8. Detector prediction tensor shape 不需要因此改变。
9. Dense O2O 不要求额外 inference branch 或 NMS。
10. target 数不能无限增加，必须考虑 Query capacity 和 match quality。
11. Dense O2O 会带来更多质量参差的 matches，因此 MAL 很关键。
12. DEIM 的本质是一套 training optimization framework，而不是单一网络模块。

---

# 72. 本课检查题

### 概念

1. O2O 为什么天然容易产生 sparse positive supervision？
2. Dense O2O 的 “Dense” 到底 dense 在哪里？
3. Dense O2O 与 O2M 的数学区别是什么？
4. 为什么 Dense O2O 仍可保持 NMS-free？
5. Mosaic 在 DEIM 中为什么不只是普通数据增强？
6. 为什么 ColorJitter 不直接增加 O2O positive count？
7. 为什么 targets/image 不能无限增加？
8. 为什么 Hungarian matched 不等于 high-quality matched？

### Shape / 数字

9. 300 Queries、6 GT 时 cost matrix 是什么 shape？
10. 24 GT 时是什么 shape？
11. 两种情况下理论 positive count 分别是多少？
12. 6/300 和 24/300 分别是多少？
13. 420 GT、300 Queries 时有什么 capacity 问题？
14. 为什么不简单把 300 Queries 增加到 600？

### 源码

15. 为什么理解 Dense O2O 应先看 `train_dataloader`？
16. Mosaic 当前位于 dataset transforms 还是 collate？
17. MixUp 配置在哪一层？
18. `SanitizeBoundingBoxes` 为什么必要？
19. 怎样验证 Dense O2O 后 assignment 仍是一对一？
20. 怎样统计真实 augmented targets/image？

### 系统

21. Dense O2O、Denoising、Global Union 分别在哪个阶段发生？
22. 为什么 Dense O2O 不直接增加 inference FLOPs？
23. 为什么它仍可能增加训练数据处理成本？
24. 为什么最后阶段通常减弱强 augmentation？
25. 为什么 Dense O2O 后还需要 MAL？

---

# 73. 答案速记

```text
1.
每个 GT 只有一个 positive Query，
且每图 GT 数通常远少于 Query 数。

2.
target / supervision density。

3.
Dense O2O: N↑, M=1
O2M:       N固定, M>1

4.
同一个 target 仍只有一个 O2O positive。

5.
它能增加 targets per final model input。

6.
它通常只改变 appearance，不增加 GT count。

7.
受 Query capacity、目标尺寸、遮挡、matching quality、
augmentation realism 等限制。

8.
Hungarian 求相对最优 assignment，不保证绝对 IoU 高。

9.
[300,6]

10.
[300,24]

11.
约 6 和 24（N <= Q 时）。

12.
2% 和 8%。

13.
最多 300 个独立 Query slots，无法覆盖全部 420 GT。

14.
Query self-attention 有 Q² 规模成分，会增加计算。

15.
Dense O2O 首先改变的是 target set。

16.
dataset transforms。

17.
collate_fn。

18.
增强后 box 可能越界、过小或退化。

19.
检查 matched src_idx 和 tgt_idx 都没有重复。

20.
在 DataLoader 后打印 len(target["boxes"])。

21.
Dense O2O：data/target
DN：decoder training query construction
Global Union：criterion / cross-layer localization matching

22.
模型推理 architecture 不需要改变。

23.
仍有 Mosaic/MixUp、box transforms、matcher matrix 等成本。

24.
让最终优化回到更接近真实测试分布。

25.
Dense O2O 引入更多质量差异大的 matched positives，
需要更合适的 classification optimization。
```

---

# 74. 下一课预告：MAL

第 16 课：

# 《MAL——Matchability-Aware Loss》

我们将从：

```text
Hungarian matched
≠
high-quality matched
```

开始，完整比较：

```text
BCE
↓
Focal Loss
↓
Varifocal Loss
↓
Matchability-Aware Loss
```

并推导：

\[
\mathrm{MAL}(p,q,y)
\]

重点回答：

```text
p 是什么？
q 是什么？
y 是什么？
为什么出现 q^γ？
IoU=0.05 时 MAL 为什么重要？
IoU=0.95 时又怎样？
Dense O2O 为什么尤其需要 MAL？
loss_labels_mal() 每一行对应哪部分数学公式？
```

还会做 PyTorch 数值实验，比较：

```text
IoU = 0.05
0.20
0.50
0.80
0.95
```

时 VFL 与 MAL 的 loss / gradient 行为。

---

# 75. 源码与论文核对范围

本课依据当前公开 DEIM / DEIMv2 资料核对：

### DEIM 论文

**DEIM: DETR with Improved Matching for Fast Convergence**, CVPR 2025。

论文明确说明 Dense O2O：

```text
保持 one-to-one matching structure
增加每张训练图中的 targets
```

以缓解 DETR O2O sparse supervision。

论文同时指出 Dense O2O 会引入更多低质量 matches，因此进一步提出 MAL。

### DEIM 官方配置

当前：

```text
configs/base/deim.yml
```

在数据增强段直接标记：

```text
# Dense O2O
```

并配置：

```text
Mosaic
RandomPhotometricDistort
RandomZoomOut
RandomIoUCrop
SanitizeBoundingBoxes
HorizontalFlip
Resize
MixUp
epoch policy
```

### DEIM Criterion

当前 `DEIMCriterion` 仍注入 matcher，并通过 Hungarian matching 处理 predictions 与 targets；这与 Dense O2O“增加 targets、保留 O2O”的核心逻辑一致。

### DEIMv2

当前 DEIMv2 DINOv3-X 配置明确标记：

```text
Dense O2O: Mosaic + Mixup + CopyBlend
```

说明 DEIMv2 延续并升级了 Dense O2O 路线。

---

# 76. 参考资料

1. DEIM: DETR with Improved Matching for Fast Convergence, CVPR 2025  
   https://openaccess.thecvf.com/content/CVPR2025/html/Huang_DEIM_DETR_with_Improved_Matching_for_Fast_Convergence_CVPR_2025_paper.html

2. DEIM paper PDF  
   https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_DEIM_DETR_with_Improved_Matching_for_Fast_Convergence_CVPR_2025_paper.pdf

3. DEIM official repository  
   https://github.com/Intellindust-AI-Lab/DEIM

4. DEIM Dense O2O base configuration  
   https://github.com/Intellindust-AI-Lab/DEIM/blob/main/configs/base/deim.yml

5. DEIM Criterion  
   https://github.com/Intellindust-AI-Lab/DEIM/blob/main/engine/deim/deim_criterion.py

6. DEIM base dataloader configuration  
   https://github.com/Intellindust-AI-Lab/DEIM/blob/main/configs/base/dataloader.yml

7. DEIMv2 official repository  
   https://github.com/Intellindust-AI-Lab/DEIMv2

8. DEIMv2 DINOv3-X configuration  
   https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/configs/deimv2/deimv2_dinov3_x_coco.yml

---

> **版本提醒**
>
> DEIM / DEIMv2 的训练代码仍可能迭代，尤其是 augmentation policy、CopyBlend、matcher schedule 和 collate 行为。
>
> 使用你的本地仓库时，优先级应为：
>
> ```text
> 1. 本地最终 merged YAML
> 2. 本地 Python implementation
> 3. DataLoader / Matcher 的真实 debug 输出
> 4. 本讲义
> ```
>
> 验证 Dense O2O 最有效的方法不是背配置，而是亲手比较：
>
> ```text
> targets per image
> matched positives per image
> matched IoU distribution
> ```
>
> 在：
>
> ```text
> Dense O2O OFF
> vs
> Dense O2O ON
> ```
>
> 两种情况下到底发生了什么。
