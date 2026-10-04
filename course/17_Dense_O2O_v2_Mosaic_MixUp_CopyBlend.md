# DEIMv2 源码课程 · 第 17 课
# DEIMv2 Dense O2O v2——Mosaic + MixUp + CopyBlend

**本课目标**

第 15 课我们已经知道 Dense O2O 的本质不是把 O2O 改成 O2M，而是：

\[
\boxed{\text{增加每个最终训练输入中的 GT 数量，同时保持每个 GT 只匹配一个 Query}}
\]

第 16 课又解决了 Dense O2O 带来的另一个问题：更多 matched positives 中会混入更多低质量匹配，因此 DEIM 使用 MAL 来改善分类优化。

本课进入 **DEIMv2 的升级版 Dense O2O**。重点不再停留在“为什么”，而是进入真正的数据管线：

```text
source images
    ↓
Mosaic
    ↓
geometric / photometric transforms
    ↓
Resize / Normalize / ConvertBoxes
    ↓
DataLoader collate
    ↓
MixUp 或 CopyBlend
    ↓
multi-scale resize
    ↓
dense targets
    ↓
model
    ↓
Hungarian O2O
    ↓
MAL + box losses
```

我们会重点围绕 `deimv2_dinov3_x_coco.yml` 来学习，并穿插当前 `BatchImageCollateFunction` 的真实实现逻辑。

---

## 0. 先记住本课的四句话

1. **Mosaic**：在 sample / dataset transform 阶段，把多张来源图像的目标组合到一张训练图里。
2. **MixUp**：在 batch collate 阶段，把 batch 中相邻（roll 后）的两张图线性混合，同时拼接两边的 targets。
3. **CopyBlend**：同样位于 collate 阶段，从当前 batch 建立 object pool，把满足条件的目标 patch 复制/混合到其他图像中，并把新 box/label 加入 targets。
4. 三者的共同目的之一都是让：

\[
N_{\text{GT per input}}\uparrow
\Rightarrow
N_{\text{O2O positives}}\uparrow
\]

但它们制造“密集监督”的方式完全不同。

---

# 1. DEIM → DEIMv2：Dense O2O 升级在哪里？

原始 DEIM 的核心数据增强路线可以概括为：

```text
Mosaic
+
MixUp
```

DEIMv2 在此基础上进一步加入：

```text
CopyBlend
```

并配合：

```text
augmentation schedule
multi-scale schedule
matcher schedule
no-augmentation stage
```

所以“Dense O2O v2”不应该理解成一个单独的新 Python 类。

更准确的理解是：

```text
一套更完整的 target densification + matching curriculum
```

---

# 2. `deimv2_dinov3_x_coco.yml` 中的真实入口

当前官方 X 配置明确写着：

```yaml
## Dense O2O: Mosaic + Mixup + CopyBlend
```

数据增强部分大致是：

```yaml
train_dataloader:
  dataset:
    transforms:
      ops:
        - {type: Mosaic, output_size: 320, ...}
        - {type: RandomPhotometricDistort, p: 0.5}
        - {type: RandomZoomOut, fill: 0}
        - {type: RandomIoUCrop, p: 0.8}
        - {type: SanitizeBoundingBoxes, min_size: 1}
        - {type: RandomHorizontalFlip}
        - {type: Resize, size: [640, 640]}
        - {type: SanitizeBoundingBoxes, min_size: 1}
        - {type: ConvertPILImage, dtype: 'float32', scale: True}
        - {type: Normalize, ...}
        - {type: ConvertBoxes, fmt: 'cxcywh', normalize: True}

      policy:
        epoch: [4, 29, 50]

  collate_fn:
    mixup_epochs: [4, 29]
    stop_epoch: 50
    copyblend_epochs: [4, 50]
    base_size_repeat: 3
```

X 配置通过 `__include__` 继承 base 配置，因此不能只看 X 文件里显式出现的字段；例如概率、CopyBlend 的对象筛选参数和 matcher 参数可能来自 `configs/base/deimv2.yml`。

**读 DEIMv2 配置时永远要看 merged config，而不是只看一个 YAML 文件。**

---

# 3. Base DEIMv2 中的重要默认值

当前公开 base 配置对应的关键默认值包括：

```yaml
mosaic_prob: 0.5

collate_fn:
  mixup_prob: 0.5
  mixup_epochs: [4, 29]

  copyblend_prob: 0.5
  copyblend_epochs: [4, 50]
  area_threshold: 100
  num_objects: 3
  with_expand: True
  expand_ratios: [0.1, 0.25]

  stop_epoch: 50
```

Matcher：

```yaml
matcher:
  change_matcher: True
  iou_order_alpha: 4.0
  matcher_change_epoch: 45
```

对 X 配置，最终值应以你本地运行时打印出的 merged YAML 为准。

---

# 4. 为什么 augmentation 分成 Dataset Transform 和 Collate 两层？

数据加载流程：

```text
Dataset.__getitem__(index)
      │
      ▼
single image + target
      │
      ▼
dataset transforms
      │
      ▼
transformed sample
      │
      ├────────┐
      │        │
      ▼        ▼
   sample 0  sample 1 ...
      │        │
      └────┬───┘
           ▼
      collate_fn
           │
           ▼
    batch images + targets
```

Mosaic 需要 Dataset 能够取得额外来源图像，因此适合放在 sample transform 侧。

MixUp / CopyBlend 则直接利用：

```text
当前 batch 中已经准备好的图像与 targets
```

所以放在 `BatchImageCollateFunction` 中很自然。

---

# 5. 第一部分：Mosaic

最常见的 2×2 Mosaic 直觉：

```text
┌──────────────┬──────────────┐
│   image A    │   image B    │
│  4 objects   │  7 objects   │
├──────────────┼──────────────┤
│   image C    │   image D    │
│  5 objects   │  6 objects   │
└──────────────┴──────────────┘
```

如果所有目标都保留：

\[
N_{\text{mosaic}}
=
4+7+5+6
=
22
\]

原始单图平均：

\[
\bar N=5.5
\]

因此一个最终 Mosaic sample 可以显著增加：

```text
targets per input
```

---

# 6. X 配置里的 `output_size: 320` 为什么不是最终 320×320？

配置中：

```yaml
Mosaic:
  output_size: 320
```

后面又有：

```yaml
Resize:
  size: [640, 640]
```

因此不要看到：

```text
output_size = 320
```

就说：

```text
X 模型训练输入是 320×320
```

最终 nominal input 仍是：

```text
640×640
```

并且 collate 阶段还可能继续进行 multi-scale resizing。

官方 README 对自定义输入尺寸也特别建议：

```text
Mosaic output_size ≈ input_size / 2
```

---

# 7. Mosaic 的 box 坐标为什么必须跟着图像一起变？

假设源图：

```text
W = 640
H = 480
```

一个 `xyxy` box：

```text
[100, 80, 300, 280]
```

如果整张图按：

```text
sx = 0.5
sy = 0.5
```

缩放，则：

\[
x'_1=s_xx_1,\quad
y'_1=s_yy_1
\]

\[
x'_2=s_xx_2,\quad
y'_2=s_yy_2
\]

得：

```text
[50, 40, 150, 140]
```

如果随后被放到 Mosaic 画布上的 offset：

```text
dx = 320
dy = 0
```

则：

```text
[370, 40, 470, 140]
```

这就是所有几何增强最基本的原则：

\[
\boxed{\text{image transform 与 box transform 必须严格同步}}
\]

---

# 8. 一般 affine 变换怎样作用于 box？

点：

\[
\mathbf p=
\begin{bmatrix}
x\\y\\1
\end{bmatrix}
\]

经过 affine：

\[
\mathbf p'
=
A\mathbf p
\]

对 box 的四个角：

```text
(x1,y1)
(x2,y1)
(x1,y2)
(x2,y2)
```

分别变换，再取：

\[
x'_{\min},y'_{\min},
x'_{\max},y'_{\max}
\]

作为新的 axis-aligned bounding box。

因此 rotation 后 box 往往会变大，因为要重新取旋转目标的外接矩形。

---

# 9. Mosaic 的 rotation / translation / scaling

X 配置：

```yaml
rotation_range: 10
translation_range: [0.1, 0.1]
scaling_range: [0.5, 1.5]
```

直觉：

```text
rotation:
    改变方向

translation:
    改变目标在画布中的位置

scaling:
    改变目标尺寸
```

它们除了增加 target count，也显著增加：

```text
scale diversity
spatial diversity
context diversity
```

---

# 10. 为什么 Mosaic 特别容易产生小目标？

四张图被组合到有限画布中时，单个来源图通常会被缩小。

假设一个目标原本：

```text
80×60
```

缩放 0.5：

```text
40×30
```

面积：

\[
4800\rightarrow1200
\]

缩小为原来的：

\[
0.5^2=0.25
\]

因此 Mosaic 不只是：

```text
object count ↑
```

也会改变 object-scale distribution。

---

# 11. 为什么 Mosaic 后必须 SanitizeBoundingBoxes？

几何变换后可能出现：

```text
box 完全出界
box 部分出界
width <= 0
height <= 0
box 太小
```

所以 X pipeline 中可以看到：

```yaml
SanitizeBoundingBoxes
```

而且不止一次。

这说明：

\[
N_{\text{after Mosaic}}
\neq
\sum N_{\text{source}}
\]

一些目标会被裁掉或过滤。

---

# 12. Mosaic cache 为什么值得注意？

X 配置：

```yaml
use_cache: True
max_cached_images: 50
random_pop: True
```

Mosaic 需要额外图像来源。

cache 可以降低反复加载额外样本的 I/O 开销，但也意味着：

```text
RAM 占用
worker 行为
随机性
```

都会受到影响。

在自定义数据集上出现：

```text
DataLoader 内存异常
worker 问题
```

时，Mosaic cache 是应该检查的地方之一。

---

# 13. Mosaic 的 `probability` 和 `mosaic_prob` 不要混淆

你可能同时看到：

```yaml
Mosaic:
  probability: 1.0
```

和：

```yaml
mosaic_prob: 0.5
```

这两个值处在不同层次的 transform/policy 控制中。

不应该脱离当前 `Compose / policy / Mosaic` 实现，只凭名字猜：

```text
最终 Mosaic 概率到底是多少。
```

正确方法是：

```text
1. 看 merged YAML
2. 看 transforms policy 的 set_epoch 逻辑
3. 看 Mosaic.forward / __call__
4. 用 debug counter 实测每个 epoch 的触发比例
```

---

# 14. 第二部分：MixUp

MixUp 的经典形式：

\[
\tilde x
=
\lambda x_i
+(1-\lambda)x_j
\]

分类任务通常还会：

\[
\tilde y
=
\lambda y_i
+(1-\lambda)y_j
\]

目标检测则不能只混一个类别标签。

因为每张图都有一个 object set：

\[
Y_i=
\{(b_k,c_k)\}
\]

所以 DEIMv2 要把两边的：

```text
boxes
labels
areas
```

都合并。

---

# 15. 当前 `BatchImageCollateFunction.apply_mixup()` 的真实逻辑

当前实现首先采样：

```python
beta = round(
    random.uniform(0.45, 0.55),
    6
)
```

当：

```text
random < mixup_prob
```

且：

```text
mixup_epochs[0]
<= epoch
< mixup_epochs[-1]
```

时启用 MixUp。

也就是说当前实现不是从完整 Beta 分布中取极端比例，而是把混合比例限制在接近：

```text
0.5 : 0.5
```

的范围。

---

# 16. Batch roll 是什么意思？

当前代码使用类似：

```python
images.roll(
    shifts=1,
    dims=0
)
```

假设 batch：

```text
[A, B, C, D]
```

roll 后：

```text
[D, A, B, C]
```

所以配对：

```text
A ↔ D
B ↔ A
C ↔ B
D ↔ C
```

不需要额外从 Dataset 再加载一批图片。

---

# 17. MixUp 图像公式

当前逻辑等价于：

\[
X_i'
=
\beta X_i
+(1-\beta)X_{i-1}
\]

例如：

```text
beta = 0.52
```

则：

```text
output A
=
0.52 × A
+
0.48 × D
```

shape 不变：

```text
before:
[B,3,H,W]

after:
[B,3,H,W]
```

---

# 18. MixUp targets 怎样变化？

假设：

```text
A: 6 boxes
D: 9 boxes
```

混合后 A 的 target set：

```text
6 + 9 = 15 boxes
```

即：

\[
Y'_A=Y_A\cup Y_D
\]

当前实现会 `torch.cat`：

```text
boxes
labels
area
```

所以：

```text
[6,4] + [9,4]
→ [15,4]
```

这就是 MixUp 对 Dense O2O 最直接的贡献。

---

# 19. MixUp 还保存了每个 object 的 mixing ratio

当前实现会在 target 中加入：

```text
target['mixup']
```

对原图对象：

```text
beta
```

对 roll 来源对象：

```text
1 - beta
```

例如：

```text
A 有 2 个目标
D 有 3 个目标
beta = .52
```

则：

```text
mixup =
[.52, .52, .48, .48, .48]
```

这让后续 criterion / target weighting 有机会知道：

```text
每个对象在混合图像中的视觉贡献比例。
```

---

# 20. MixUp 为什么能继续增加 Dense O2O？

假设 Mosaic 后：

```text
sample A = 20 GT
sample D = 18 GT
```

再 MixUp：

```text
output A = 38 GT
```

若：

```text
Q = 300
```

理论 positive utilization：

\[
20/300=6.67\%
\]

变成：

\[
38/300=12.67\%
\]

所以 Mosaic 与 MixUp 可以叠加 densification。

---

# 21. 但 MixUp 并不是永远叠加

当前 `apply_mixup()` 的结构是：

```text
if MixUp condition:
    MixUp

elif CopyBlend condition:
    CopyBlend
```

也就是说默认实现中：

```text
同一次 collate 调用
MixUp 命中后
不再进入 CopyBlend 分支
```

因此：

\[
\boxed{\text{MixUp 与 CopyBlend 默认是互斥式选择，而不是同 batch 串行叠加}}
\]

这是只看 YAML 很容易错过的源码细节。

---

# 22. 第三部分：CopyBlend

CopyBlend 的直觉：

```text
当前 batch 中已有多个 object
       ↓
筛选可复制对象
       ↓
截取 object patch
       ↓
选择新的目标位置
       ↓
blend / copy 到另一张图
       ↓
新增 box + label + area
```

它不是简单把两张完整图像叠在一起。

它是在：

```text
object level
```

增加目标实例。

---

# 23. CopyBlend 为什么对 Dense O2O 很自然？

Mosaic：

```text
image-level composition
```

MixUp：

```text
image-level linear mixture
```

CopyBlend：

```text
object-level densification
```

如果一张图原来：

```text
12 GT
```

CopyBlend 新增：

```text
3 objects
```

则：

```text
15 GT
```

Hungarian 仍然：

```text
1 GT → 1 Query
```

但 positives 从：

```text
12 → 15
```

---

# 24. CopyBlend 首先建立 object pool

当前实现遍历 batch：

```text
image 0
image 1
...
image B-1
```

收集：

```text
boxes
labels
areas
image_idx
image_height
image_width
```

形成：

```text
objects_pool
```

因此 CopyBlend 的 donor objects 来自：

```text
当前 batch
```

而不是另开一个 Dataset loader。

---

# 25. `area_threshold=100` 是什么？

当前 base 配置：

```yaml
area_threshold: 100
```

object pool 只保留：

```text
source_area >= area_threshold
```

的目标。

直觉原因：

```text
极小 patch
→ 复制后信息不足
→ 几何误差相对更大
→ augmentation 质量可能差
```

所以先过滤过小对象。

注意 area 的实际单位/尺度必须结合当前 target 在 collate 前保存的 `area` 定义检查，不能只凭 `100` 猜成某个固定物理尺度。

---

# 26. `num_objects=3`

当前 base：

```yaml
num_objects: 3
```

可以把它理解为 CopyBlend 对单张目标图像尝试复制的 object 数量预算。

某些模型配置还支持：

```text
random_num_objects
```

让实际数量在：

```text
1 ~ num_objects
```

间随机。

对 X 默认配置，应以 merged config 为准。

---

# 27. CopyBlend 为什么要记录 donor `image_idx`？

object pool 中一个 box：

```text
只告诉你目标在哪
```

但真正复制 patch 还需要知道：

```text
它来自哪张 image tensor
```

因此必须保存：

```text
image_idx
```

然后：

```text
source image
+
source box
```

才能裁出 donor patch。

---

# 28. 从归一化 cxcywh 恢复像素 patch

collate 前 targets 已可能是：

```text
normalized cxcywh
```

假设：

```text
box = [cx,cy,w,h]
```

图像：

```text
W,H
```

则像素中心：

\[
c_x^{px}=c_xW
\]

\[
c_y^{px}=c_yH
\]

宽高：

\[
w^{px}=wW
\]

\[
h^{px}=hH
\]

patch 边界：

\[
x_1=c_x^{px}-w^{px}/2
\]

\[
y_1=c_y^{px}-h^{px}/2
\]

\[
x_2=c_x^{px}+w^{px}/2
\]

\[
y_2=c_y^{px}+h^{px}/2
\]

最终还要 clip 到图像边界。

---

# 29. 一个具体 CopyBlend box 例子

假设：

```text
image = 640×640
normalized box:
[0.25, 0.50, 0.10, 0.20]
```

则：

```text
cx = 160
cy = 320
w  = 64
h  = 128
```

所以：

```text
x1 = 128
y1 = 256
x2 = 192
y2 = 384
```

donor patch：

```text
[C,128,64]
```

如果复制到新中心：

```text
(400,200)
```

新 box：

```text
x1=368
y1=136
x2=432
y2=264
```

再转回 normalized cxcywh：

```text
[0.625, 0.3125, 0.10, 0.20]
```

---

# 30. `with_expand=True` 是什么思路？

当前 base：

```yaml
with_expand: True
expand_ratios: [0.1, 0.25]
```

直觉：

```text
不只裁目标 box 的精确矩形
而允许向周围扩一点上下文
```

如果原 patch 宽：

```text
w
```

随机 expand ratio：

```text
r
```

教学上可以理解为裁剪区域可能向外扩：

\[
\Delta w\propto rw
\]

\[
\Delta h\propto rh
\]

具体上下左右如何扩、怎样 clip，以你本地 `BatchImageCollateFunction` 的实际实现为准。

---

# 31. 为什么复制一点上下文可能更自然？

如果严格只复制 box：

```text
┌──── object ────┐
```

边界可能非常突兀。

带少量 surrounding pixels：

```text
┌── context ──────────┐
│   ┌─ object ─┐      │
│   └──────────┘      │
└─────────────────────┘
```

可以降低：

```text
硬切边界
```

带来的视觉伪影。

---

# 32. CopyBlend 的 `blend` 到底做什么？

当前实现存在：

```text
copyblend_type='blend'
```

对目标区域：

\[
P_{\text{out}}
=
\beta P_{\text{dst}}
+(1-\beta)P_{\text{src}}
\]

即：

```text
目标背景 patch
与
donor object patch
```

做线性混合。

如果不是 blend 模式，则可以直接：

```text
destination patch = source patch
```

---

# 33. CopyBlend 与普通 Copy-Paste 的差别

普通 hard copy-paste：

```text
dst_patch = src_patch
```

CopyBlend 的 blend 模式：

```text
dst_patch
=
beta * dst_patch
+
(1-beta) * src_patch
```

因此 object 不是完全覆盖背景，而是以混合方式注入。

从训练角度看，它同时具有：

```text
object insertion
+
soft appearance mixing
```

的特点。

---

# 34. CopyBlend 后 targets 怎样更新？

对每个成功插入的 object：

```text
new box
new label
new area
```

会 append 到目标图像原有 targets。

例如：

```text
before:
boxes [12,4]
labels [12]

copied:
boxes [3,4]
labels [3]

after:
boxes [15,4]
labels [15]
```

这正是它能够增加 O2O positives 的根本原因。

---

# 35. CopyBlend 也维护 `mixup` ratio

当前实现会给更新后的 targets 构造：

```text
mixup
```

原有 objects：

```text
ratio = 1.0
```

新 blended objects：

```text
ratio = 对应 blend ratio
```

所以 `mixup` 这个 key 的语义不应狭义理解成：

```text
“只有 MixUp augmentation 才有”
```

它更像：

```text
object-level visual contribution / mixing weight metadata
```

---

# 36. CopyBlend 失败时怎么办？

如果当前 batch：

```text
没有 area >= threshold 的有效 object
```

则：

```text
objects_pool empty
```

当前实现直接返回：

```text
original images, targets
```

所以：

```text
copyblend_prob > 0
```

不代表每次命中概率后都一定能实际增加目标。

---

# 37. MixUp 与 CopyBlend 的概率关系

当前控制流近似：

```python
if mixup_active and random() < mixup_prob:
    do_mixup()

elif copyblend_active and random() < copyblend_prob:
    do_copyblend()
```

假设两者都 active：

```text
p_mix = 0.5
p_copy = 0.5
```

则理论上：

\[
P(\text{MixUp})=0.5
\]

CopyBlend 只有 MixUp 没触发后才有机会：

\[
P(\text{CopyBlend})
=
(1-0.5)\times0.5
=
0.25
\]

都不发生：

\[
0.5\times0.5=0.25
\]

这是一个非常重要的概率细节。

---

# 38. 为什么 YAML 中的 `copyblend_prob=0.5` 不等于最终 50% batch 都 CopyBlend？

因为：

```text
MixUp 在 if 分支
CopyBlend 在 elif 分支
```

当两者时间窗重叠时，CopyBlend 的实际触发率还受到：

```text
MixUp 是否先命中
```

的影响。

所以真实触发率必须：

```text
debug / counter
```

而不是只读一个配置数字。

---

# 39. 第四部分：训练时间轴

对 X 的核心时间点：

```text
epoch 4
epoch 29
epoch 45（base matcher schedule）
epoch 50
```

以及总训练/no-aug 相关配置共同构成 curriculum。

官方 README 给出的通用调 epoch 示例明确说明：

```text
policy epoch:
[start_epoch, flat_epoch, epoches-no_aug_epoch]

mixup_epochs:
[start_epoch, flat_epoch]

copyblend_epochs:
[start_epoch, epoches-no_aug_epoch]

matcher_change_epoch:
约训练增强阶段末尾的 90%
```

---

# 40. X 配置的教学时间轴

可以先用下面的“概念图”理解：

```text
epoch
0      4              29             45      50
|------|---------------|--------------|-------|

warm / stabilization
       │
       ├── MixUp active window ──────┤
       │
       ├──────── CopyBlend active ───────────┤
       │
       ├── strong transform curriculum ──────┤
                                     │
                                     └ matcher changes
                                             │
                                             └ strong aug / multiscale stop
```

这张图是教学抽象；**具体某个 transform 在某个区间的开关状态必须以当前 `policy` 实现为准**。

---

# 41. 为什么前 4 个 epoch 不急着上最强 Dense augmentation？

训练刚开始：

```text
query
classification
box
```

都接近随机。

此时 Hungarian assignment 本来就不稳定。

如果立刻加入最复杂：

```text
Mosaic
MixUp
CopyBlend
```

可能让：

```text
target distribution
matching difficulty
```

同时剧烈变化。

因此 DEIM/DEIMv2 recipe 采用阶段式策略，而不是“从 step 0 到最后全部拉满”。

---

# 42. 为什么 MixUp 比 CopyBlend 更早结束？

X：

```text
MixUp:
[4,29)

CopyBlend:
[4,50)
```

MixUp 会把整张图像都做线性叠加：

```text
global visual distribution shift
```

CopyBlend 更偏：

```text
localized object-level augmentation
```

所以 recipe 允许：

```text
中期先停止更强的全图混合
继续保留 object-level densification
```

直到后期。

这是合理的训练 curriculum 直觉；具体效果应以论文/ablation 和实际训练验证为准。

---

# 43. 为什么最后还要 no-augmentation stage？

强增强的分布：

```text
Mosaic image
MixUp image
CopyBlend image
```

与真实测试图像并不完全相同。

后期关闭强增强，让模型更多看到：

```text
natural single images
```

有助于：

```text
localization refinement
confidence calibration
train-test distribution alignment
```

这与很多现代 detector 的训练 recipe 一致。

---

# 44. 第五部分：Multi-scale Training

`BatchImageCollateFunction.__call__()` 在 augmentation 后还可能：

```python
if self.scales is not None \
   and self.epoch < self.stop_epoch:
    sz = random.choice(self.scales)
    images = F.interpolate(
        images,
        size=sz
    )
```

所以 nominal：

```text
640×640
```

不代表训练每个 batch 永远都是：

```text
640×640
```

---

# 45. `generate_scales()` 怎样生成尺寸？

当前逻辑以：

```text
base_size
```

为中心，从约：

```text
0.75×base_size
```

到：

```text
1.25×base_size
```

按 32 的倍数生成候选尺度，并通过：

```text
base_size_repeat
```

控制 base size 重复出现次数。

对 X：

```text
base_size = 640
base_size_repeat = 3
```

因此 640 会在候选列表中重复，提高抽到 nominal resolution 的概率。

---

# 46. Multi-scale 为什么不会破坏 normalized boxes？

collate 前：

```text
boxes 已转成 normalized cxcywh
```

例如：

```text
[0.5,0.5,0.2,0.1]
```

图像从：

```text
640×640
```

resize 到：

```text
800×800
```

normalized box 仍是：

```text
[0.5,0.5,0.2,0.1]
```

因为它表达的是相对比例。

这就是：

```text
ConvertBoxes(normalize=True)
```

在 multi-scale pipeline 中非常方便的原因。

---

# 47. 第六部分：DEIMv2 的 Matcher Schedule

当前 base matcher：

```yaml
change_matcher: True
iou_order_alpha: 4.0
matcher_change_epoch: 45
```

这说明 DEIMv2 的 Dense O2O v2 不只是：

```text
数据增强升级
```

还包含：

```text
后期 matching cost / ordering strategy 的变化
```

---

# 48. 为什么 matcher 也要分阶段？

训练早期：

```text
boxes 很差
IoU 本身噪声大
```

如果过早高度依赖 localization ordering：

```text
matcher 可能被不可靠的 IoU 主导
```

后期：

```text
boxes 已经比较准
```

localization quality 更有可信度。

因此可以让 matcher 后期更重视：

```text
high-quality localization ordering
```

形成 curriculum：

```text
early:
learn objects

late:
refine which Query is the best match
```

---

# 49. `iou_order_alpha=4.0` 不要凭名字硬猜公式

看到：

```yaml
iou_order_alpha: 4.0
```

不能直接断言：

\[
\text{cost}=\text{cost}-4\cdot IoU
\]

或：

\[
IoU^4
\]

具体公式必须去当前：

```text
HungarianMatcher
```

的 `forward()` / matching cost 构造代码中逐行确认。

本课只需要建立：

```text
DEIMv2 后期会切换 matching strategy
```

这个系统概念。

Matcher 的精确公式会在完整训练 recipe / matcher 源码分析中继续展开。

---

# 50. Dense O2O v2 的完整因果链

```text
Original images
      │
      ▼
Mosaic
      │
      ├─ more objects
      ├─ smaller objects
      └─ new spatial contexts
      │
      ▼
transforms + sanitize
      │
      ▼
batch collate
      │
      ├─ MixUp
      │    └─ union two target sets
      │
      └─ CopyBlend
           └─ inject selected object patches
      │
      ▼
targets / input ↑
      │
      ▼
Hungarian O2O
      │
      ▼
positive Query supervision ↑
      │
      ├─ more useful positives
      └─ more low-quality positives
      │
      ▼
MAL
      │
      ▼
better classification optimization
      │
      ▼
late matcher schedule
      │
      ▼
better high-quality assignment/refinement
      │
      ▼
no-aug refinement
```

---

# 51. 一个完整数字例子

假设原始 batch 中 4 张图：

```text
A: 5 GT
B: 7 GT
C: 6 GT
D: 4 GT
```

Mosaic 后，假设每个 sample 组合后平均：

```text
20 GT
```

经过 crop / sanitize：

```text
18 GT
```

如果 MixUp 命中：

```text
18 + 17
=
35 GT
```

对：

```text
Q = 300
```

positive ratio：

\[
35/300
\approx
11.7\%
\]

而原始平均：

\[
5.5/300
\approx
1.83\%
\]

这是一个示意例子，但它非常直观地展示了 Dense O2O 的目标。

---

# 52. 如果 CopyBlend 命中

假设：

```text
Mosaic + sanitize 后：
18 GT
```

CopyBlend 成功加入：

```text
3 GT
```

则：

```text
21 GT
```

positive ratio：

\[
21/300=7\%
\]

它没有 MixUp 那么激进地把整套 target union 进来，但视觉扰动也更局部。

---

# 53. Mosaic / MixUp / CopyBlend 对比表

| 项目 | Mosaic | MixUp | CopyBlend |
|---|---|---|---|
| 作用层 | Dataset transform | Batch collate | Batch collate |
| 粒度 | image composition | whole-image mixing | object patch |
| 是否增加 targets | 是 | 是 | 是 |
| 图像 shape 是否立即改变 | 会重组画布 | 不变 | 不变 |
| targets 更新 | 几何变换+合并 | concat | append |
| appearance 改变 | 强 | 强 | 局部 |
| object context 改变 | 强 | 叠加上下文 | 局部插入 |
| Dense O2O 作用 | 大幅增加 GT | target set union | 精细增加 GT |
| 当前 collate 中与另一项互斥 | — | 与 CopyBlend | 与 MixUp |

---

# 54. 为什么不是“GT 越多越好”？

继续增加 GT 会遇到：

```text
1. Query capacity
2. excessive overlap
3. tiny objects
4. unrealistic contexts
5. harder matching
6. low-quality positives
7. data-loading overhead
```

Dense O2O 的目标不是：

\[
\max N_{\text{GT}}
\]

而是：

\[
\max
\text{useful O2O supervision}
\]

---

# 55. Query Capacity 再复习

若：

```text
Q = 300
GT = 40
```

最多：

```text
40 positives
```

若：

```text
GT = 420
```

最多：

```text
300 independent Query slots
```

因此自定义密集场景数据集必须统计：

```text
augmented GT P50
P90
P95
P99
max
```

---

# 56. 为什么 CopyBlend 的 `num_objects=3` 是一种控制？

CopyBlend 如果无限复制：

```text
target count
overlap
visual artifacts
```

都可能迅速上升。

`num_objects` 提供一个直接的 densification budget。

可以把它看成：

```text
每张图最多额外注入多少 object-level supervision
```

的重要超参数。

---

# 57. Dense augmentation 为什么与 MAL 是配套的？

强 augmentation 后：

```text
more GT
→ more Hungarian positives
```

但：

```text
more small / occluded / blended objects
→ match quality variance ↑
```

第 16 课：

```text
MAL positive weight = 1
target = q^gamma
```

所以低质量但已经 matched 的 Query 不会像 VFL 那样再被整体乘一个很小的 q。

这就是：

```text
Dense O2O
+
MAL
```

的协同。

---

# 58. MixUp ratio 与 MAL quality 是两个不同概念

`target['mixup']`：

```text
表示 augmentation 中 object 的视觉混合贡献
```

MAL 的：

```text
q = IoU(pred_box, GT)
```

表示：

```text
当前 prediction 的 localization quality
```

两者不要混淆：

```text
mixup ratio ≠ IoU
```

一个来自：

```text
data construction
```

一个来自：

```text
model prediction
```

---

# 59. PyTorch 小实验 1：模拟 MixUp

```python
import torch

B, C, H, W = 4, 3, 640, 640
images = torch.rand(B, C, H, W)

beta = 0.52

mixed = (
    images.roll(shifts=1, dims=0)
    * (1.0 - beta)
    + images * beta
)

print(images.shape)
print(mixed.shape)
```

输出：

```text
[4,3,640,640]
[4,3,640,640]
```

---

# 60. PyTorch 小实验 2：模拟 target concat

```python
import torch

boxes_a = torch.rand(6, 4)
labels_a = torch.randint(0, 80, (6,))

boxes_b = torch.rand(9, 4)
labels_b = torch.randint(0, 80, (9,))

boxes = torch.cat(
    [boxes_a, boxes_b],
    dim=0
)

labels = torch.cat(
    [labels_a, labels_b],
    dim=0
)

print(boxes.shape)
print(labels.shape)
```

输出：

```text
[15,4]
[15]
```

---

# 61. PyTorch 小实验 3：构造 MixUp metadata

```python
beta = 0.52

mixup = torch.tensor(
    [beta] * 6
    +
    [1.0 - beta] * 9
)

print(mixup.shape)
print(mixup)
```

shape：

```text
[15]
```

与：

```text
15 objects
```

一一对应。

---

# 62. PyTorch 小实验 4：归一化 box → 像素 box

```python
import torch

box = torch.tensor([
    0.25,
    0.50,
    0.10,
    0.20
])

W = H = 640

cx, cy, w, h = box

cx *= W
cy *= H
w  *= W
h  *= H

x1 = cx - w / 2
y1 = cy - h / 2
x2 = cx + w / 2
y2 = cy + h / 2

print(
    x1.item(),
    y1.item(),
    x2.item(),
    y2.item()
)
```

应得到约：

```text
128, 256, 192, 384
```

---

# 63. PyTorch 小实验 5：复制 patch

```python
import torch

image_src = torch.rand(
    3, 640, 640
)

image_dst = torch.rand(
    3, 640, 640
)

patch = image_src[
    :,
    256:384,
    128:192
]

print(patch.shape)

image_dst[
    :,
    136:264,
    368:432
] = patch
```

这就是最小版：

```text
Copy-Paste
```

---

# 64. PyTorch 小实验 6：CopyBlend

```python
beta = 0.5

dst_patch = image_dst[
    :,
    136:264,
    368:432
]

blended = (
    beta * dst_patch
    +
    (1 - beta) * patch
)

image_dst[
    :,
    136:264,
    368:432
] = blended
```

这就是 CopyBlend 最核心的像素操作。

真实源码还必须处理：

```text
object selection
location selection
clipping
expand
boxes
labels
area
mixup ratio
```

---

# 65. PyTorch 小实验 7：概率树

```python
p_mix = 0.5
p_copy = 0.5

p_mix_final = p_mix
p_copy_final = (
    1 - p_mix
) * p_copy

p_none = (
    1 - p_mix
) * (
    1 - p_copy
)

print(
    p_mix_final,
    p_copy_final,
    p_none
)
```

输出：

```text
0.5
0.25
0.25
```

这是 `if / elif` 带来的实际概率结构。

---

# 66. PyTorch 小实验 8：Monte Carlo 验证实际概率

```python
import random

N = 100000

counts = {
    "mixup": 0,
    "copyblend": 0,
    "none": 0,
}

for _ in range(N):
    if random.random() < 0.5:
        counts["mixup"] += 1
    elif random.random() < 0.5:
        counts["copyblend"] += 1
    else:
        counts["none"] += 1

for k, v in counts.items():
    print(
        k,
        v / N
    )
```

应接近：

```text
MixUp    50%
CopyBlend 25%
None      25%
```

---

# 67. PyTorch 小实验 9：生成 multi-scale candidates

```python
def generate_scales(
    base_size,
    base_size_repeat
):
    start = int(
        base_size * 0.75 / 32
    ) * 32

    scale_repeat = (
        base_size - start
    ) // 32

    scales = [
        start + i * 32
        for i in range(scale_repeat)
    ]

    scales += [
        base_size
    ] * base_size_repeat

    scales += [
        int(
            base_size * 1.25 / 32
        ) * 32
        - i * 32
        for i in range(scale_repeat)
    ]

    return scales


print(
    generate_scales(
        640,
        3
    )
)
```

然后数一下：

```text
640 出现多少次？
```

---

# 68. 实验 10：统计 augmentation 后 targets/image

最值得做的实验：

```python
counts = []

for images, targets in train_loader:
    counts.extend(
        len(t["boxes"])
        for t in targets
    )
```

然后：

```python
import numpy as np

arr = np.asarray(counts)

print(
    "mean",
    arr.mean()
)

for p in [
    50, 90, 95, 99
]:
    print(
        p,
        np.percentile(arr, p)
    )

print(
    "max",
    arr.max()
)
```

分别在：

```text
epoch 0
epoch 5
epoch 20
epoch 35
epoch 46
epoch 51
```

采样，你会直接看到 curriculum。

---

# 69. 实验 11：给 MixUp / CopyBlend 加触发计数器

调试时可以临时加入：

```python
self.mix_count += 1
```

和：

```python
self.copy_count += 1
```

每个 epoch 输出：

```text
MixUp batches
CopyBlend batches
No-mix batches
```

这比只看：

```text
prob = 0.5
```

更可靠。

---

# 70. 实验 12：统计 CopyBlend 真正新增多少 objects

```python
before = [
    len(t["boxes"])
    for t in targets
]

images2, targets2 = collate.apply_mixup(
    images,
    targets
)

after = [
    len(t["boxes"])
    for t in targets2
]

delta = [
    b - a
    for a, b
    in zip(before, after)
]

print(delta)
```

注意函数名虽然叫：

```text
apply_mixup
```

当前实现内部实际上同时管理：

```text
MixUp
CopyBlend
```

这是源码命名与实际职责不完全一致的一个小细节。

---

# 71. 实验 13：保存 augmentation 可视化

强烈建议至少保存：

```text
20 Mosaic
20 MixUp
20 CopyBlend
```

并画 box。

重点检查：

```text
box 是否偏移
box 是否越界
label 是否错位
CopyBlend box 是否包住新目标
resize 后 box 是否仍正确
```

目标检测增强中：

> **一个 box 坐标 bug 足以让整个训练 recipe 失效。**

---

# 72. 实验 14：检查 target 与 image shape

在 collate 后：

```python
print(
    images.shape
)

for i, t in enumerate(targets):
    print(
        i,
        t["boxes"].shape,
        t["labels"].shape,
        t["area"].shape
    )
```

必须满足：

```text
len(boxes)
=
len(labels)
=
len(area)
```

若有 `mixup`：

```text
len(mixup)
=
len(boxes)
```

---

# 73. 实验 15：检查 box 合法性

normalized `cxcywh`：

```python
boxes = t["boxes"]

assert torch.isfinite(
    boxes
).all()

assert (
    boxes[:, 2:] > 0
).all()
```

根据 pipeline 语义还可以检查：

```text
cx,cy,w,h
```

是否大致位于合理范围。

不要机械要求所有值都严格 `<1`，除非你确认当前 sanitize / clipping 已保证这一点。

---

# 74. 实验 16：Dense O2O 的最终验证

augmentation 不是目的。

最终要验证：

```text
matched positives
```

是否增加。

Matcher 后：

```python
pos_counts = [
    len(src_idx)
    for src_idx, tgt_idx
    in indices
]
```

比较：

```text
Dense O2O v2 OFF
Dense O2O v2 ON
```

并同时统计：

```text
matched IoU
```

才能知道增加的是：

```text
useful supervision
```

还是只是：

```text
noisy supervision
```

---

# 75. 推荐 Ablation 1：逐个增加 augmentation

```text
A. baseline
B. + Mosaic
C. + Mosaic + MixUp
D. + Mosaic + CopyBlend
E. + Mosaic + MixUp + CopyBlend schedule
```

记录：

```text
targets/image
matched positives/image
matched IoU
training time
AP
AP50
AP75
APS
APM
APL
```

这会让你真正理解每个模块的职责。

---

# 76. 推荐 Ablation 2：CopyBlend object 数

```text
num_objects:
1
2
3
5
```

观察：

```text
target density
overlap
matched IoU
APS
overall AP
```

不要假设：

```text
5 一定比 3 好。
```

---

# 77. 推荐 Ablation 3：area_threshold

```text
25
50
100
200
```

观察：

```text
donor pool size
copied-object size distribution
augmentation success rate
small-object AP
```

这对目标尺寸分布与 COCO 很不同的自定义数据集尤其重要。

---

# 78. 推荐 Ablation 4：Copy vs Blend

```text
copyblend_type = copy
```

与：

```text
copyblend_type = blend
```

比较：

```text
visual artifacts
convergence
AP
confidence calibration
```

这能帮助你理解：

```text
“增加 object”
```

与：

```text
“怎样把 object 放进去”
```

是两个不同问题。

---

# 79. 推荐 Ablation 5：schedule

比较：

```text
MixUp 长时间开启
vs
中期关闭
```

```text
CopyBlend 中期关闭
vs
后期关闭
```

重点观察：

```text
AP75
calibration
late-stage localization loss
```

后期强增强过多往往最容易影响精细 localization。

---

# 80. 常见误解 1：Mosaic 就等于 4 倍 positives

错。

因为：

```text
crop
scaling
sanitize
tiny-object filtering
```

都会改变最终 GT 数。

正确做法：

```text
实测 len(target["boxes"])
```

---

# 81. 常见误解 2：MixUp 只混像素

错。

检测 MixUp 必须同步合并：

```text
boxes
labels
area
```

当前实现还记录：

```text
mixup ratio
```

---

# 82. 常见误解 3：CopyBlend 只是 Copy-Paste 换名字

不准确。

当前实现支持：

```text
blend
```

即 donor 与 destination patch 做线性混合，而不只是硬覆盖。

还包含：

```text
object pool
area filtering
object budget
optional expand
target metadata
```

---

# 83. 常见误解 4：MixUp 和 CopyBlend 会一起执行

默认当前控制流不是。

它们处于：

```text
if
elif
```

因此同一次 collate 通常只执行其中一种。

---

# 84. 常见误解 5：copyblend_prob=0.5 就是 50% batch

当 MixUp 同时 active 时不一定。

因为 CopyBlend 只有在：

```text
MixUp 没命中
```

后才进入概率判断。

---

# 85. 常见误解 6：`output_size=320` 就是模型输入 320

错。

X pipeline 后面：

```text
Resize → 640×640
```

collate 还可能：

```text
multi-scale resize
```

---

# 86. 常见误解 7：augmentation 只影响 DataLoader

它最终会一路影响：

```text
targets count
↓
Hungarian cost matrix
↓
matched indices
↓
MAL
↓
L1/GIoU
↓
gradient
```

所以它实际上改变整个 optimization graph 的监督结构。

---

# 87. 常见误解 8：Dense O2O v2 改成 O2M

仍然错误。

即使：

```text
Mosaic + MixUp + CopyBlend
```

让一张图有 40 个 GT，

Hungarian 仍然：

```text
GT0 → q7
GT1 → q19
...
GT39 → q288
```

每个 GT：

```text
一个 O2O positive
```

---

# 88. 常见误解 9：matcher change 就是 Dense O2O

不是。

Dense O2O：

```text
target-density strategy
```

Matcher change：

```text
assignment strategy curriculum
```

DEIMv2 把二者组合使用，但概念必须分开。

---

# 89. 常见误解 10：强 augmentation 应一直开到最后

DEIMv2 明确使用：

```text
epoch windows
stop_epoch
no_aug_epoch
```

所以训练 recipe 本身就是：

```text
curriculum
```

而不是固定增强。

---

# 90. 你现在应该怎样读这部分源码？

建议按：

```text
1. deimv2_dinov3_x_coco.yml
2. configs/base/deimv2.yml
3. DataLoader.set_epoch()
4. Dataset / Compose.set_epoch()
5. Mosaic
6. BatchImageCollateFunction.__init__()
7. apply_mixup()
8. __call__()
9. HungarianMatcher
10. DEIMCriterion
```

这条顺序读。

不要直接搜索：

```text
DenseO2Ov2
```

因为它不是一个独立 class。

---

# 91. 一张最终结构图

```text
                 DEIMv2 Dense O2O v2
                          │
         ┌────────────────┼────────────────┐
         │                │                │
         ▼                ▼                ▼
      Mosaic            MixUp          CopyBlend
         │                │                │
image-level          image-level       object-level
composition          mixture           injection
         │                │                │
         └────────────────┼────────────────┘
                          ▼
                 targets / image ↑
                          │
                          ▼
                    Hungarian O2O
                          │
                          ▼
                  positive matches ↑
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
          useful positives    hard/low-IoU matches
                │                   │
                │                   ▼
                │                  MAL
                │                   │
                └─────────┬─────────┘
                          ▼
                   denser gradients
                          │
                          ▼
                late matcher change
                          │
                          ▼
                  no-aug refinement
```

---

# 92. 本课最重要的 15 个结论

1. DEIMv2 的 Dense O2O v2 是一套训练系统，不是一个单独模块。
2. X 配置明确使用 Mosaic + MixUp + CopyBlend。
3. Mosaic 位于 dataset transform 阶段。
4. MixUp / CopyBlend 位于 batch collate 阶段。
5. Mosaic 通过 image composition 增加目标。
6. MixUp 通过两套 target set 的 union 增加目标。
7. CopyBlend 通过 object-level patch injection 增加目标。
8. 当前 MixUp 使用 batch roll 配对，无需额外加载第二批图片。
9. 当前 MixUp ratio 接近 0.5。
10. 当前 MixUp 与 CopyBlend 默认通过 `if/elif` 互斥执行。
11. CopyBlend 使用 object pool、area threshold、num_objects 和 optional expand。
12. Dense O2O v2 仍然保持 Hungarian one-to-one assignment。
13. MAL 用来处理更多 matches 带来的质量差异。
14. DEIMv2 后期还会改变 matcher，并最终进入弱/无强增强 refinement。
15. 评价增强是否有效，必须统计 target count、matched count、matched IoU 和 AP，而不是只看配置是否开启。

---

# 93. 本课检查题

## 概念

1. Dense O2O v2 为什么仍然不是 O2M？
2. Mosaic、MixUp、CopyBlend 分别在哪个数据阶段？
3. 三者怎样分别增加 target density？
4. 为什么 CopyBlend 比 MixUp 更偏 object-level augmentation？
5. 为什么强增强后仍需要 MAL？
6. 为什么后期需要关闭强 augmentation？

## Shape / 数字

7. `[B,3,640,640]` 经过 MixUp 后 shape 是否改变？
8. 6 boxes 与 9 boxes MixUp 后 target boxes 是什么 shape？
9. `Q=300, GT=35` 时最多有多少 O2O positives？
10. positive-query ratio 是多少？
11. normalized `[.25,.5,.1,.2]` 在 640 图像上的像素 box 是多少？

## 源码

12. 为什么 MixUp 使用 `roll(shifts=1,dims=0)`？
13. 为什么 target 要 `deepcopy()`？
14. `target['mixup']` 保存什么？
15. CopyBlend 为什么需要 `image_idx`？
16. `area_threshold` 用在哪里？
17. `num_objects` 控制什么？
18. `with_expand` 的目的是什么？
19. 为什么 MixUp 命中后 CopyBlend 不再执行？
20. 为什么 `copyblend_prob=.5` 不一定等于 50% 实际触发率？

## Schedule

21. X 的 MixUp window 是什么？
22. CopyBlend window 是什么？
23. `stop_epoch` 控制什么重要行为？
24. matcher_change_epoch 为什么通常放在后期？
25. 为什么 `policy.epoch` 不能脱离 Python policy 实现硬猜？

---

# 94. 答案速记

```text
1.
target 数增加，但每个 GT 仍只匹配一个 Query。

2.
Mosaic: Dataset transform
MixUp: collate
CopyBlend: collate

3.
Mosaic: 多图组合
MixUp: 两套 targets concat
CopyBlend: 新增 object patch + annotation

4.
它只选择/插入若干目标区域，而不是混整张图。

5.
更多 targets 会带来更多质量参差的 matched positives。

6.
后期需要回到更接近真实测试分布做精细 refinement。

7.
不改变，仍 [B,3,H,W]。

8.
[15,4]

9.
35

10.
35/300 ≈ 11.67%

11.
[128,256,192,384] xyxy

12.
用 batch 内相邻样本构造 MixUp partner，
无需额外 Dataset fetch。

13.
避免原 targets 被原地修改造成副作用。

14.
每个 object 的 augmentation mixing contribution。

15.
必须知道 donor patch 来自哪张图。

16.
筛掉过小 donor objects。

17.
控制每张目标图像尝试注入的 object 数量预算。

18.
裁 donor 时可带少量上下文，减轻硬边界伪影。

19.
当前代码是 if / elif。

20.
CopyBlend 只有 MixUp 未触发后才有机会。

21.
X 当前配置 [4,29)。

22.
X 当前配置 [4,50)。

23.
当前 collate 中用于停止 multi-scale 等后期训练行为；
具体还要结合 solver/collate 实现。

24.
后期 box 已较可靠，IoU/localization ordering 更可信。

25.
三个 epoch 数只是配置参数，
真正开关语义由 set_epoch / policy code 决定。
```

---

# 95. 下一课预告

第 18 课：

# 《完整训练 Recipe——LR、AdamW、EMA、AMP、DDP》

我们会把前面：

```text
Dense O2O
MAL
augmentation schedule
matcher schedule
```

与真正的 optimizer/system training 串起来。

重点进入：

```text
AdamW
weight decay
parameter groups
DINOv3 backbone LR
detector LR
warmup
flat-cosine scheduler
no-aug stage
EMA
EMA restart
AMP
GradScaler
gradient clipping
DDP
batch-size scaling
```

尤其会拆 X 配置为什么：

```text
DINOv3 LR = 1e-5
detector LR = 5e-4
```

即约：

```text
50×
```

的学习率差异，以及为什么预训练 foundation backbone 不能和随机/新初始化检测模块用同样 LR。

---

# 96. 源码核对说明

本讲义按当前公开 DEIMv2 主仓库配置和可检索到的当前 `BatchImageCollateFunction` 实现逻辑核对。

其中必须特别注意：

1. `deimv2_dinov3_x_coco.yml` 会通过 `__include__` 继承 base 配置。
2. 因此只看 X YAML 可能看不到 `mixup_prob`、`copyblend_prob`、`area_threshold`、`num_objects`、`matcher_change_epoch` 等最终参数。
3. 官方仓库仍在迭代；不同 commit 的 collate、CopyBlend、matcher schedule 可能发生变化。
4. `policy.epoch=[4,29,50]` 的逐区间精确行为必须结合你本地版本的 transform policy 实现验证。
5. CopyBlend 的 patch location、expand 和 conflict handling 等细节应以你本地 `engine/data/dataloader.py` 为最高优先级。

学习时建议始终遵守：

```text
merged YAML
    ↓
local Python source
    ↓
runtime print/debug
    ↓
paper/README
    ↓
本讲义
```

---

# 97. 参考资料

1. DEIMv2 official repository  
   https://github.com/Intellindust-AI-Lab/DEIMv2

2. `deimv2_dinov3_x_coco.yml`  
   https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/configs/deimv2/deimv2_dinov3_x_coco.yml

3. DEIMv2 README — input size / epoch customization / training recipe  
   https://github.com/Intellindust-AI-Lab/DEIMv2

4. Original DEIM repository  
   https://github.com/Intellindust-AI-Lab/DEIM

5. Original DEIM Dense O2O base config  
   https://github.com/Intellindust-AI-Lab/DEIM/blob/main/configs/base/deim.yml

6. DEIM paper: *DEIM: DETR with Improved Matching for Fast Convergence*, CVPR 2025  
   https://openaccess.thecvf.com/content/CVPR2025/html/Huang_DEIM_DETR_with_Improved_Matching_for_Fast_Convergence_CVPR_2025_paper.html

---

## 本课最终记忆图

```text
                 Dense O2O v2
                      │
       ┌──────────────┼──────────────┐
       │              │              │
     Mosaic         MixUp        CopyBlend
       │              │              │
  多图空间组合      全图混合       目标级注入
       │              │              │
       └──────────────┼──────────────┘
                      ▼
               GT per input ↑
                      ▼
              O2O positives ↑
                      ▼
                quality variance ↑
                      ▼
                     MAL
                      ▼
            late matching refinement
                      ▼
                no-aug refinement
```

当你能够从 `deimv2_dinov3_x_coco.yml` 的：

```yaml
## Dense O2O: Mosaic + Mixup + CopyBlend
```

一路追到：

```text
Dataset transforms
→ BatchImageCollateFunction
→ targets 数量变化
→ Hungarian cost matrix
→ matched positives
→ MAL
→ late matcher
```

并能用实际数字解释每一步时，DEIMv2 的训练优化主线就已经真正串起来了。
