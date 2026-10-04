# DEIMv2 源码课程 · 第 9 课

# `deim_dfine_x` Backbone——HGNetv2-X / HGNetv2-B5 从源码到每一个 Shape

> **本课目标**
>
> 第 8 课我们已经从 YAML 还原出第一条重点路线：
>
> ```text
> DEIM
> ├── backbone: HGNetv2-B5
> ├── encoder: HybridEncoder
> └── decoder: DFINETransformer
> ```
>
> 这一课只做一件事：把 `HGNetv2-B5` 真正拆开。
>
> 学完以后，你应该能够从 `[B,3,640,640]` 开始，不靠背答案，自己推导出：
>
> ```text
> Stem   → [B,  64,160,160]
> Stage1 → [B, 128,160,160]
> Stage2 → [B, 512, 80, 80]   ← return
> Stage3 → [B,1024, 40, 40]   ← return
> Stage4 → [B,2048, 20, 20]   ← return
> ```
>
> 并真正理解 `ConvBNAct`、`LightConvBNAct`、`HG_Block`、`HG_Stage`、`EseModule`、dense aggregation、depthwise convolution、residual、`return_idx`、冻结策略和预训练权重加载分别在做什么。

---

# 0. 本课在 24 课中的位置

```text
第 8 课
第一次完整读 deim_dfine_x 配置
        │
        ▼
第 9 课
HGNetv2-B5 Backbone            ← 现在
        │
        ▼
第 10 课
HybridEncoder
        │
        ▼
第 11～13 课
DFINETransformer / FDR / GO-LSD
        │
        ▼
第 14 课
完整 forward → matcher → loss
```

这一课结束时，我们只走完：

```text
Image
  │
  ▼
HGNetv2-B5
  │
  ├── C3
  ├── C4
  └── C5
```

还没有进入：

```text
HybridEncoder
DFINETransformer
HungarianMatcher
loss
```

一定要把模型层级分清楚。

---

# 1. 先确认 X 配置到底选择了哪个 Backbone

`deim_dfine_x` 这一课程简称对应的 D-FINE-X / DEIM-D-FINE-X 路线中，X 配置选择：

```yaml
HGNetv2:
    name: 'B5'
    return_idx: [1, 2, 3]
```

因此本课不是泛泛研究所有 HGNetv2。

我们只重点追：

```text
HGNetv2
name = B5
return_idx = [1,2,3]
```

对应的真实执行路径。

官方 `HGNetv2.arch_configs` 中 B5 的核心结构可以整理为：

```text
Stem:
[3, 32, 64]

Stage1:
in=64
mid=64
out=128
blocks=1
downsample=False
light=False
kernel=3
layers/block=6

Stage2:
in=128
mid=128
out=512
blocks=2
downsample=True
light=False
kernel=3
layers/block=6

Stage3:
in=512
mid=256
out=1024
blocks=5
downsample=True
light=True
kernel=5
layers/block=6

Stage4:
in=1024
mid=512
out=2048
blocks=2
downsample=True
light=True
kernel=5
layers/block=6
```

这张表几乎就是本课所有 shape 推导的“源代码地图”。

---

# 2. Backbone 到底负责什么？

目标检测器可以先抽象为：

```text
RGB image
[B,3,H,W]

    │
    ▼

Backbone

    │
    ▼

visual features
```

Backbone 的核心任务不是直接输出 bbox，而是：

> **把像素转换成越来越抽象的视觉特征。**

浅层 feature 更容易保留：

```text
edge
texture
local geometry
fine spatial detail
```

深层 feature 更容易形成：

```text
object part
semantic pattern
high-level representation
```

与此同时，空间分辨率逐步下降：

```text
640
↓
320
↓
160
↓
80
↓
40
↓
20
```

而 channel 通常增加：

```text
3
↓
32
↓
64
↓
128
↓
512
↓
1024
↓
2048
```

这是一种非常经典的 CNN hierarchy：

```text
空间越来越小
语义越来越强
通道越来越多
```

---

# 3. 为什么检测需要多尺度 Backbone 输出？

如果只返回最后的：

```text
[B,2048,20,20]
```

那么对于 640×640 输入：

```text
一个 feature cell
```

大致对应原图：

```text
32×32
```

的步长。

对于很小的目标，这可能太粗。

因此检测器通常同时取：

```text
stride 8
stride 16
stride 32
```

三个层级：

```text
C3 → 80×80
C4 → 40×40
C5 → 20×20
```

于是：

```text
高分辨率 feature
→ 更有利于细粒度定位、小目标

低分辨率 feature
→ 更强语义、更大感受野
```

后面的 HybridEncoder 再把三者融合。

所以：

```text
HGNetv2
```

负责：

```text
产生 hierarchy
```

而：

```text
HybridEncoder
```

负责：

```text
跨尺度进一步交互和融合
```

不要把这两个职责混在一起。

---

# 4. 先复习卷积输出公式

第 3 课已经学过：

\[
H_
==

\left\lfloor
\frac{
H_{in}+2P-D(K-1)-1
}{S}
+1
\right\rfloor
\]

其中：

```text
K = kernel size
S = stride
P = padding
D = dilation
```

如果：

```text
K=3
S=2
P=1
D=1
```

那么：

\[
H_
==

\left\lfloor
\frac{H_{in}-1}{2}+1
\right\rfloor
\]

对于偶数尺寸：

```text
640 → 320
320 → 160
160 → 80
80  → 40
40  → 20
```

这就是 HGNetv2 多尺度 hierarchy 的空间基础。

---

# 5. 第一个基础模块：`ConvBNAct`

HGNetv2 中大量结构最终都建立在一个基本单元上：

```text
Conv
↓
BatchNorm
↓
Activation
```

可以把它记作：

```text
ConvBNAct
```

概念源码：

```python
class ConvBNAct(nn.Module):

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
```

真实实现还支持：

```text
groups
padding
use_act
use_lab
```

等选项。

---

# 6. Conv 为什么后面接 BatchNorm？

卷积输出：

\[
y = W*x+b
\]

不同 batch / channel 的激活尺度可能差异较大。

BatchNorm 训练时大致执行：

\[
\hat
====

\frac{x-\mu_B}
{\sqrt{\sigma_B^2+\epsilon}}
\]

再学习：

\[
y
=

\gamma\hat{x}+\beta
\]

直觉上：

```text
Conv
→ 提取特征

BN
→ 调整/稳定特征分布

Activation
→ 引入非线性
```

HGNetv2 中大量卷积都采用这种组合。

---

# 7. 为什么 Conv + BN 时 Conv 常常不需要 bias？

如果：

```text
Conv bias = b
```

然后马上：

```text
BatchNorm
```

BN 自己已经有：

```text
β
```

这样的平移参数。

因此 Conv bias 往往是冗余的。

这也是现代 CNN 中常见：

```python
nn.Conv2d(..., bias=False)
nn.BatchNorm2d(...)
```

的原因。

以后计算参数量时要注意：

> 不能看到 Conv 就机械地额外加 `C_out` 个 bias 参数，要看实际 `bias` 设置。

---

# 8. 激活函数：ReLU / SiLU 与 `use_lab`

普通激活可能是：

```text
ReLU
SiLU
```

SiLU：

\[
SiLU(x)=x\sigma(x)
\]

其中：

\[
\sigma(x)=\frac1{1+e^{-x}}
\]

HGNetv2 代码还支持：

```text
LearnableAffineBlock
```

简称可以理解为 LAB：

\[
y = a x+b
\]

其中：

```text
a
b
```

不是常量，而是可学习参数。

因此启用 `use_lab` 时，激活之后还可以有一个很轻量的可学习 affine 调整。

X/B5 路线是否启用它，要以最终 merged config 为准；理解源码时先知道该分支存在即可。

---

# 9. 第二个基础模块：`LightConvBNAct`

Stage3 / Stage4 的 B5 配置中：

```text
light_block=True
```

因此 HG block 内部会使用更轻量的卷积结构。

它的核心思想是：

```text
1×1 pointwise conv
        ↓
depthwise spatial conv
```

而不是每次都做昂贵的普通大卷积。

---

# 10. 什么是普通卷积？

假设：

```text
Cin = 256
Cout = 256
K = 5
```

普通 Conv 参数量：

\[
Params
======

C_{in}C_{out}K^2
\]

即：

\[
256\times256\times25
====================

1,638,400
\]

仅权重就约：

```text
1.64M
```

---

# 11. 什么是 Depthwise Convolution？

Depthwise Conv：

```text
groups = channels
```

意味着：

```text
每个 channel
只使用自己的空间卷积核
```

若：

```text
C = 256
K = 5
```

参数量：

\[
Params_
=======

C K^2
\]

\[
==

256\times25
===========

6400
\]

与普通 256→256 的 5×5 卷积：

```text
1,638,400
```

相比小很多。

---

# 12. 但 Depthwise Conv 不负责充分混合 channel

Depthwise：

```text
channel 0 → channel 0
channel 1 → channel 1
...
```

因此 channel 间交互不足。

所以常配：

```text
1×1 Pointwise Conv
```

执行 channel mixing。

于是轻量卷积的经典组合：

```text
Pointwise
↓
Depthwise
```

或者：

```text
Depthwise
↓
Pointwise
```

具体顺序由实现决定。

HGNetv2 的 `LightConvBNAct` 正是利用这类思想降低 Stage3 / Stage4 的计算量。

---

# 13. 为什么越深的 Stage 越适合 Light Block？

Stage3 / Stage4 的 channel 很大：

```text
512
1024
2048
```

如果大量使用：

```text
5×5 full convolution
```

参数量和 FLOPs 会迅速爆炸。

但这些 stage 的 spatial size 已经较小：

```text
40×40
20×20
```

通过：

```text
light block
+
large kernel
```

可以同时获得：

```text
较大的空间感受范围
+
较低计算成本
```

这是一种效率与表达能力的折中。

---

# 14. `EseModule` 是什么？

HGNetv2 的 aggregation 后还会出现 channel attention 风格的模块。

核心流程：

```text
feature
[B,C,H,W]

   │
   ▼
global average pooling

[B,C,1,1]

   │
   ▼
1×1 Conv

[B,C,1,1]

   │
   ▼
Sigmoid

[B,C,1,1]

   │
   ▼
multiply original feature

[B,C,H,W]
```

数学上：

\[
z_c
===

\frac1{HW}
\sum_{h=1}^{H}
\sum_{w=1}^{W}
x_{c,h,w}
\]

然后：

\[
a=\sigma(Wz+b)
\]

最后：

\[
y_
==

a_c x_{c,h,w}
\]

---

# 15. ESE 的直觉

假设 feature 有：

```text
1024 channels
```

不是每个 channel 对当前图片都同样重要。

ESE 学习：

```text
channel 0 → 0.12
channel 1 → 0.94
channel 2 → 0.33
...
```

然后：

```text
重要 channel
→ 放大/保留

不重要 channel
→ 抑制
```

因此它属于：

```text
channel reweighting
```

而不是 spatial attention。

---

# 16. 为什么叫 Effective / Efficient SE？

传统 SE block 通常有：

```text
Global Pool
↓
FC reduce
↓
activation
↓
FC expand
↓
sigmoid
```

而 HGNetv2 里的 ESE 更简洁：

```text
Global Pool
↓
1×1 Conv
↓
Sigmoid
```

因此额外开销更低。

对于实时 detector：

```text
少量性能收益
+
很小 latency overhead
```

通常比复杂 attention 更有吸引力。

---

# 17. 现在进入本课最重要的模块：`HG_Block`

HGNetv2 的核心不是普通 residual block。

它使用一种 dense feature aggregation 思想。

可以先画成：

```text
input x0
   │
   ├───────────────┐
   ▼               │
 layer1            │
   │               │
   ▼               │
  x1               │
   │               │
   ▼               │
 layer2            │
   │               │
   ▼               │
  x2               │
   │               │
  ...              │
   │               │
   ▼               │
  x6               │
                   │
x0,x1,x2,...,x6 ───┘
        │
        ▼
      concat
        │
        ▼
  aggregation conv
        │
        ▼
       ESE
        │
        ▼
      output
```

这是理解 HGNetv2 的核心。

---

# 18. B5 中一个 HG Block 有多少内部 layer？

B5 每个 stage 的配置最后一个数字：

```text
layer_num = 6
```

所以一个 HG Block 内部会产生：

```text
x1
x2
x3
x4
x5
x6
```

并保留最初：

```text
x0
```

最后 concat：

```text
[x0,x1,x2,x3,x4,x5,x6]
```

一共：

```text
7 份 feature
```

---

# 19. Dense aggregation 与普通串行 CNN 有什么区别？

普通串行：

```text
x0
↓
x1
↓
x2
↓
x3
```

最后只把：

```text
x3
```

交给下一层。

HG Block 更像：

```text
x0
x1
x2
x3
...
```

全部保留并聚合。

因此后面的 aggregation 同时能看到：

```text
较浅 feature
+
较深 feature
```

这有利于：

```text
feature reuse
gradient flow
multi-level representation
```

---

# 20. HG Block 的核心 forward 逻辑

为了避免被源码细节淹没，可以把真实逻辑压缩成：

```python
outputs = [x]

for layer in self.layers:
    x = layer(x)
    outputs.append(x)

x = torch.cat(outputs, dim=1)

x = self.aggregation(x)

if self.residual:
    x = x + identity

return x
```

你真正应该盯住的不是变量名，而是：

```text
① 保存 input
② 连续产生 6 个 feature
③ 全部 concat
④ 压缩/聚合
⑤ 可选 residual
```

---

# 21. Stage1 第一个 HG Block：精确算 concat channel

Stage1：

```text
in_chs  = 64
mid_chs = 64
out_chs = 128
layer_num = 6
```

输入：

```text
x0:
[B,64,160,160]
```

每个内部 layer 输出：

```text
[B,64,160,160]
```

所以：

```text
x1 = 64 ch
x2 = 64 ch
x3 = 64 ch
x4 = 64 ch
x5 = 64 ch
x6 = 64 ch
```

concat channel：

\[
64+6\times64
============

448
\]

因此：

```text
concat:
[B,448,160,160]
```

然后 aggregation：

```text
448
↓
128
```

得到：

```text
[B,128,160,160]
```

这就是 Stage1 的输出。

---

# 22. 这说明什么？

HG Block 并不是：

```text
64 → 128
```

这么简单。

内部真实发生：

```text
64
↓
连续生成 6×64 feature
↓
concat 成 448 channels
↓
聚合到 128
```

所以：

```text
out_chs
```

只是 block 最终输出 channel。

并不等于 block 内部所有中间 feature 的 channel。

---

# 23. Stage2 第一个 HG Block

进入 Stage2 前先 downsample：

```text
Stage1:
[B,128,160,160]

↓ stride 2

[B,128,80,80]
```

Stage2 第一个 HG Block：

```text
in = 128
mid = 128
out = 512
layers = 6
```

concat channel：

\[
128+6\times128
==============

896
\]

所以：

```text
before aggregation:
[B,896,80,80]

after aggregation:
[B,512,80,80]
```

---

# 24. Stage2 为什么有两个 HG Blocks？

B5：

```text
Stage2 block_num = 2
```

第一个 block：

```text
128
→ 512
```

第二个 block 的输入已经是：

```text
512
```

但 mid 仍然：

```text
128
```

所以第二个 block concat：

\[
512+6\times128
==============

1280
\]

然后：

```text
1280
↓ aggregation
512
```

最终：

```text
Stage2 output
[B,512,80,80]
```

---

# 25. 第二个 block 为什么可以 residual？

HG Stage 创建 block 时，通常：

```text
第一个 block
residual=False
```

因为：

```text
input channels
```

和：

```text
output channels
```

往往不同。

例如：

```text
128 → 512
```

不能直接：

```python
output + input
```

因为 shape 不同。

而第二个 block：

```text
512 → 512
```

shape 一致，所以可以：

```text
output + identity
```

这就是 residual 是否启用背后的 shape 条件。

---

# 26. Residual connection 为什么重要？

残差：

\[
y=F(x)+x
\]

梯度：

\[
\frac
=====

\frac{\partial F}{\partial x}+1
\]

其中直接的：

```text
+1
```

给梯度提供一条 identity path。

直觉上：

```text
深层网络
更容易优化
```

同时 block 只需要学习：

```text
在输入基础上应该修改什么
```

而不是从零重新构造全部表示。

---

# 27. Stage3：开始使用 Light Block

Stage3：

```text
in = 512
mid = 256
out = 1024
block_num = 5
downsample = True
light_block = True
kernel = 5
layer_num = 6
```

首先：

```text
[B,512,80,80]
↓ stride 2
[B,512,40,40]
```

第一个 HG Block concat channel：

\[
512+6\times256
==============

2048
\]

所以：

```text
[B,2048,40,40]
↓ aggregation
[B,1024,40,40]
```

---

# 28. Stage3 后续 HG Block 的 concat channel

第 2～5 个 block：

```text
input = 1024
mid = 256
```

concat：

\[
1024+6\times256
\]

\[
==

1024+1536
\]

\[
==

2560
\]

所以每个后续 block 都大致经历：

```text
[B,1024,40,40]
        ↓
6 个 256-channel intermediate features
        ↓
concat
[B,2560,40,40]
        ↓
aggregation
[B,1024,40,40]
        ↓
residual
[B,1024,40,40]
```

一共重复：

```text
4 个 residual HG blocks
```

加上第一个 transition block：

```text
总计 5 blocks
```

---

# 29. Stage4

Stage4：

```text
in = 1024
mid = 512
out = 2048
block_num = 2
downsample = True
light = True
kernel = 5
layers = 6
```

downsample：

```text
[B,1024,40,40]
↓
[B,1024,20,20]
```

第一个 block concat：

\[
1024+6\times512
===============

4096
\]

所以：

```text
[B,4096,20,20]
↓
aggregation
[B,2048,20,20]
```

第二个 block：

```text
input = 2048
mid = 512
```

concat：

\[
2048+6\times512
===============

5120
\]

然后：

```text
[B,5120,20,20]
↓
[B,2048,20,20]
```

最终得到：

```text
Stage4
[B,2048,20,20]
```

---

# 30. 到这里先做一张 B5 HG Block 总表

| Stage |  Spatial | First input | Mid |  Out | Blocks | Light | Kernel |
| ----- | -------: | ----------: | --: | ---: | -----: | ----- | -----: |
| 1     | 160×160 |          64 |  64 |  128 |      1 | No    |      3 |
| 2     |   80×80 |         128 | 128 |  512 |      2 | No    |      3 |
| 3     |   40×40 |         512 | 256 | 1024 |      5 | Yes   |      5 |
| 4     |   20×20 |        1024 | 512 | 2048 |      2 | Yes   |      5 |

这张表非常值得保存。

---

# 31. 现在回头拆 Stem

B5：

```text
stem_channels = [3,32,64]
```

Stem 的目标：

```text
[B,3,640,640]
↓
[B,64,160,160]
```

也就是：

```text
总 stride = 4
```

但它不是简单：

```text
Conv stride2
↓
Conv stride2
```

中间还有一个双分支结构。

---

# 32. Stem 总体结构

可以画成：

```text
Input
[B,3,640,640]

   │
   ▼
stem1
3×3, stride 2

[B,32,320,320]

   │
   ▼
special padding

[B,32,321,321]

   ├─────────────────────┐
   │                     │
   ▼                     ▼
MaxPool               stem2a
branch                2×2 Conv
   │                     │
   │                     ▼
   │                  stem2b
   │                  2×2 Conv
   │                     │
   ▼                     ▼
[B,32,320,320]     [B,32,320,320]

   │                     │
   └──────────┬──────────┘
              ▼
            concat

       [B,64,320,320]

              │
              ▼
            stem3
         3×3 stride2

       [B,32,160,160]

              │
              ▼
            stem4
             1×1

       [B,64,160,160]
```

这是 HGNetv2 Stem 比普通“两次 stride-2 Conv”更有意思的地方。

---

# 33. Stem 第一步：`stem1`

B5：

```text
in = 3
mid = 32
```

`stem1`：

```text
3×3
stride = 2
```

所以：

```text
[B,3,640,640]
↓
[B,32,320,320]
```

总 stride：

```text
2
```

---

# 34. 为什么随后手工 `F.pad`？

代码会对 feature 的：

```text
right
bottom
```

补 1 个像素。

于是：

```text
320×320
↓
321×321
```

这是为了让后面的：

```text
2×2 conv
```

和：

```text
2×2 max pool
```

两个分支在 spatial shape 上能够对齐。

这种写法不要用“常规 same padding”想当然。

源码阅读时：

> 遇到显式 `F.pad`，一定要把 padding 后的 H/W 单独写出来。

---

# 35. Conv 分支

从：

```text
[B,32,321,321]
```

开始。

`stem2a`：

```text
2×2
stride 1
```

如果不额外 padding：

\[
321-2+1
=======

320
\]

所以：

```text
[B,16,320,320]
```

然后再次对右/下补 1：

```text
[B,16,321,321]
```

`stem2b` 再做：

```text
2×2 stride1
```

得到：

```text
[B,32,320,320]
```

---

# 36. Pool 分支

另一边：

```text
[B,32,321,321]
```

经过：

```text
MaxPool 2×2
stride1
```

得到与 Conv 分支匹配的：

```text
[B,32,320,320]
```

于是两个分支可以：

```python
torch.cat(..., dim=1)
```

---

# 37. Stem concat

两个分支：

```text
pool branch:
[B,32,320,320]

conv branch:
[B,32,320,320]
```

沿 channel concat：

```text
32 + 32 = 64
```

所以：

```text
[B,64,320,320]
```

这再次用到了第 2 课的：

```python
torch.cat
```

注意：

```text
cat(dim=1)
```

改变的是：

```text
channel
```

不是：

```text
height / width
```

---

# 38. `stem3`

然后：

```text
3×3 stride2
```

把：

```text
320×320
```

降到：

```text
160×160
```

channel：

```text
64 → 32
```

所以：

```text
[B,64,320,320]
↓
[B,32,160,160]
```

---

# 39. `stem4`

最后：

```text
1×1 stride1
```

channel：

```text
32 → 64
```

spatial 不变：

```text
160×160
```

所以 Stem 最终：

```text
[B,64,160,160]
```

即：

```text
stride = 4
```

---

# 40. 为什么 Stem 不直接一次 stride=4？

如果粗暴地：

```text
640
↓ stride4
160
```

会一次丢掉大量局部信息。

HGNetv2 Stem 使用：

```text
逐步下采样
+
pool / conv 双分支
```

让网络在早期更充分地保留局部模式。

目标检测尤其关心：

```text
edge
corner
fine spatial structure
small objects
```

所以早期 feature extraction 的质量很重要。

---

# 41. Stage1 为什么不下采样？

Stem 已经：

```text
640 → 160
```

Stage1：

```text
downsample=False
```

因此保持：

```text
160×160
```

只做 feature transformation：

```text
64 channels
→
128 channels
```

所以：

```text
Stem:
[B,64,160,160]

Stage1:
[B,128,160,160]
```

此时整体 stride 仍然：

```text
4
```

---

# 42. Stage2 下采样到底用什么？

`HG_Stage` 在：

```text
downsample=True
```

时会先建立一个：

```text
3×3
stride=2
groups=in_chs
```

的 depthwise convolution。

注意：

> 它不是普通 full convolution。

例如 Stage2：

```text
in_chs = 128
```

所以：

```text
groups = 128
```

意味着：

```text
128 个 channel
各自独立做 3×3 stride2 spatial downsampling
```

这比普通：

```text
128 → 128 full 3×3
```

便宜很多。

---

# 43. Stage2 Downsample 参数量对比

普通 3×3：

\[
128\times128\times9
===================

147456
\]

depthwise 3×3：

\[
128\times9
==========

1152
\]

比例：

\[
147456/1152=128
\]

也就是说，仅卷积核权重而言：

```text
depthwise
```

少约：

```text
128 倍
```

这就是 groups convolution 的威力。

---

# 44. Stage3 / Stage4 也是 depthwise stride-2 downsample

Stage3：

```text
C=512
```

普通 3×3：

\[
512^2\times9
============

2,359,296
\]

depthwise：

\[
512\times9
==========

4608
\]

Stage4：

```text
C=1024
```

普通：

\[
1024^2\times9
=============

9,437,184
\]

depthwise：

\[
1024\times9
===========

9216
\]

因此 HGNetv2 在 downsampling 上非常重视效率。

---

# 45. 现在完整走一次 640×640

输入：

```text
x
[B,3,640,640]
```

Stem：

```text
stem1
→ [B,32,320,320]

branches + concat
→ [B,64,320,320]

stem3
→ [B,32,160,160]

stem4
→ [B,64,160,160]
```

Stage1：

```text
[B,64,160,160]
↓
[B,128,160,160]
```

Stage2：

```text
[B,128,160,160]
↓ downsample
[B,128,80,80]
↓ HG blocks
[B,512,80,80]
```

Stage3：

```text
[B,512,80,80]
↓
[B,512,40,40]
↓
[B,1024,40,40]
```

Stage4：

```text
[B,1024,40,40]
↓
[B,1024,20,20]
↓
[B,2048,20,20]
```

---

# 46. 完整 stride 表

| 位置        | Shape               | 相对输入 stride |
| ----------- | ------------------- | --------------: |
| Input       | `[B,3,640,640]`   |               1 |
| stem1       | `[B,32,320,320]`  |               2 |
| Stem output | `[B,64,160,160]`  |               4 |
| Stage1      | `[B,128,160,160]` |               4 |
| Stage2      | `[B,512,80,80]`   |               8 |
| Stage3      | `[B,1024,40,40]`  |              16 |
| Stage4      | `[B,2048,20,20]`  |              32 |

所以：

```text
Stage2 = C3
Stage3 = C4
Stage4 = C5
```

在检测 neck/encoder 的语境中非常自然。

---

# 47. `return_idx=[1,2,3]` 到底在选什么？

HGNetv2 的 stage 列表可以看成：

```text
index 0 → Stage1
index 1 → Stage2
index 2 → Stage3
index 3 → Stage4
```

配置：

```yaml
return_idx: [1,2,3]
```

因此返回：

```text
Stage2
Stage3
Stage4
```

而不返回：

```text
Stage1
```

所以最终 backbone output 是一个：

```python
list[Tensor]
```

而不是单个 Tensor。

---

# 48. Backbone 返回值的真实概念

```python
feats = backbone(images)
```

那么：

```text
feats[0]
=
[B,512,80,80]

feats[1]
=
[B,1024,40,40]

feats[2]
=
[B,2048,20,20]
```

注意：

```text
feats.shape
```

不存在。

因为：

```text
feats
```

是 Python list。

应该：

```python
for f in feats:
    print(f.shape)
```

这是实际调试时很常见的错误。

---

# 49. 为什么不返回 Stage1？

Stage1：

```text
stride 4
160×160
```

位置数：

\[
160^2
=====

25600
\]

非常大。

如果加入后续 multi-scale encoder：

```text
显存
计算
attention/feature fusion cost
```

都会明显增加。

而实时 detector 需要在：

```text
small-object detail
```

和：

```text
latency
```

之间折中。

X 路线选择：

```text
stride 8 / 16 / 32
```

作为检测的三个主尺度。

---

# 50. 三个返回 feature 一共有多少 spatial positions？

```text
80×80 = 6400
40×40 = 1600
20×20 = 400
```

总计：

\[
6400+1600+400=8400
\]

这就是我们前几课反复出现的：

```text
8400
```

但现在你终于知道：

> 它最初来自 HGNetv2-B5 的 Stage2 / Stage3 / Stage4。

不是 Transformer 凭空产生的。

---

# 51. Backbone feature 还不是 Transformer token

这一点非常重要。

当前输出：

```text
[B,C,H,W]
```

例如：

```text
[B,512,80,80]
```

这是：

```text
CNN feature map
```

还不是：

```text
[B,N,D]
```

形式的 token sequence。

后面的 HybridEncoder / decoder 会进行：

```text
channel projection
flatten
transpose
multi-scale concatenation
```

等操作。

所以现在不要提前把：

```text
80×80 feature map
```

直接等同于：

```text
6400 token tensor
```

虽然它之后可以被 flatten 成 6400 个 spatial positions。

---

# 52. Channel 为什么会 128 → 512 突然增加这么多？

很多 CNN 习惯：

```text
64 → 128 → 256 → 512
```

而 B5：

```text
Stage1 out = 128
Stage2 out = 512
Stage3 out = 1024
Stage4 out = 2048
```

HGNetv2 的 HG Block 使用：

```text
多个中间 feature
↓
concat
↓
aggregation
```

因此最终 `out_chs` 可以明显大于 `mid_chs`。

例如 Stage2：

```text
mid = 128
out = 512
```

不要误以为 Stage2 内部每一层都是 512 channels。

实际内部大量计算仍在：

```text
128 mid channels
```

上完成。

这也是控制计算成本的重要手段。

---

# 53. `mid_chs` 与 `out_chs` 的角色

可以把 HG Block 想成：

```text
wide external representation
+
narrower internal transformations
```

例如 Stage3：

```text
external:
1024 channels

internal:
256 channels
```

于是：

```text
输入 1024
↓
内部连续产生 256-channel features
↓
dense concat
↓
聚合回 1024
```

这与很多网络中的：

```text
bottleneck
```

思想有相似之处。

---

# 54. 为什么 Stage3 有 5 个 block？

B5 是较大的 backbone。

它把较多计算深度放在：

```text
Stage3
40×40
```

这个尺度。

这是一个很合理的折中点：

```text
80×80
→ 太贵

20×20
→ spatial detail 已经较粗

40×40
→ 计算量和空间信息之间较平衡
```

因此 B5：

```text
Stage3 = 5 blocks
```

比：

```text
Stage1 = 1
Stage2 = 2
Stage4 = 2
```

更深。

---

# 55. 参数量与 FLOPs 是两回事

一个层参数量：

\[
Params
\]

主要由：

```text
Cin
Cout
K
groups
```

决定。

但 FLOPs 还强烈依赖：

```text
H×W
```

例如相同 Conv：

```text
80×80
```

上的计算量约是：

```text
20×20
```

的：

\[
\frac{80^2}{20^2}=16
\]

倍。

所以：

> 深层 channel 多，不代表一定比浅层 spatial 大的卷积更耗计算。

阅读实时检测器时一定要同时看：

```text
channel
+
spatial resolution
```

---

# 56. 普通 Conv 参数公式

groups=1：

\[
Params_
=======

C_{out}
\cdot
C_{in}
\cdot
K^2
\]

如果带 bias：

\[
+C_{out}
\]

BN 训练参数：

\[
2C_{out}
\]

分别是：

```text
gamma
beta
```

running mean / variance 是 buffer，不是 optimizer 学习参数。

---

# 57. Group Conv 参数公式

若：

```text
groups = G
```

则：

\[
Params
======

C_{out}
\cdot
\frac{C_{in}}{G}
\cdot
K^2
\]

Depthwise：

```text
G = Cin
Cout = Cin
```

则：

\[
Params
======

C_{in}K^2
\]

这三个公式必须熟练。

---

# 58. Conv MACs 的粗略公式

忽略 bias 等小项：

\[
MACs
\approx
H_{out}W_{out}
C_{out}
\frac{C_{in}}{G}
K^2
\]

如果把一次 multiply-add 算成两个 FLOPs：

\[
FLOPs \approx 2\times MACs
\]

不同 benchmark 工具对 FLOPs/MACs 的定义可能不同。

因此比较论文数字时必须确认：

```text
工具
输入尺寸
是否算 BN
MAC 是否算 1 FLOP 或 2 FLOPs
```

---

# 59. 小实验：普通卷积 vs Depthwise 参数量

```python
def conv_params(
    cin,
    cout,
    k,
    groups=1,
    bias=False
):
    n = (
        cout
        * (cin // groups)
        * k
        * k
    )

    if bias:
        n += cout

    return n


normal = conv_params(
    512,
    512,
    3,
    groups=1
)

depthwise = conv_params(
    512,
    512,
    3,
    groups=512
)

print(normal)
print(depthwise)
print(normal / depthwise)
```

你应该得到：

```text
普通卷积参数量
=
depthwise 的 512 倍
```

仅讨论卷积核权重时成立。

---

# 60. 感受野：为什么连续小卷积能看到更大区域？

假设连续：

```text
3×3 stride1
```

第一层感受野：

```text
3×3
```

第二层：

```text
5×5
```

第三层：

```text
7×7
```

所以网络越深：

```text
effective receptive field
```

会逐渐增大。

HG Block 内部连续 6 层，因此不仅做 feature transformation，也持续扩大感受范围。

---

# 61. 感受野的递推公式

维护：

```text
r = receptive field
j = jump / effective stride
```

初始化：

```text
r0 = 1
j0 = 1
```

每层：

\[
j_l
===

j_{l-1}s_l
\]

\[
r_l
===

r_{l-1}
+
(k_l-1)d_lj_{l-1}
\]

例如：

```text
3×3 stride2
```

第一层：

\[
j=2
\]

\[
r=1+(3-1)\times1=3
\]

再一层：

```text
3×3 stride2
```

得到：

\[
j=4
\]

\[
r=3+2\times2=7
\]

所以一个 feature cell 已经能依赖原图更大区域。

---

# 62. 为什么 Stage3/4 使用 5×5？

5×5 单层对感受野增长：

\[
(k-1)=4
\]

而 3×3：

\[
(k-1)=2
\]

所以在相同 effective stride 下：

```text
5×5
```

能更快扩大 spatial context。

但普通 5×5 很贵。

因此 HGNetv2 在深层使用：

```text
LightConv
+
5×5 depthwise
```

是非常自然的组合：

```text
大 kernel
+
低成本
```

---

# 63. `freeze_at` 是什么？

HGNetv2 constructor 支持：

```text
freeze_at
```

它控制前面若干 stage 是否冻结。

冻结的本质：

```python
param.requires_grad = False
```

意味着：

```text
forward 仍然计算
```

但：

```text
optimizer 不更新这些参数
```

所以：

```text
freeze
≠
删除层
```

也：

```text
freeze
≠
不执行 forward
```

---

# 64. `freeze_norm` 是什么？

如果：

```text
freeze_norm=True
```

框架会把 normalization 行为冻结/替换为适合冻结推理统计的形式。

为什么要关心？

BN 在训练时不仅有：

```text
gamma / beta
```

还有：

```text
running mean
running variance
```

而 detector 的 batch size 常常较小。

小 batch 的 BN statistics 可能不稳定，因此检测框架经常：

```text
freeze BN
```

或者：

```text
SyncBN
```

或者使用：

```text
FrozenBatchNorm
```

---

# 65. DEIM 为什么又强调 Unfreezing BN？

第 8 课已经看到 DEIM base recipe 会调整：

```text
freeze_at
freeze_norm
```

这说明 DEIM 的训练策略不仅作用于：

```text
matcher / loss
```

也重新考虑 backbone fine-tuning。

如果 backbone 的 BN 不冻结：

```text
其 affine 参数和/或统计行为
```

可以进一步适应 detection training distribution。

这可能带来收益，但也依赖：

```text
batch size
distributed training
SyncBN
pretraining
```

等条件。

因此它属于完整训练 recipe，而不是纯 architecture。

---

# 66. `freeze_stem_only` 是什么思想？

X baseline 配置里还能看到：

```text
freeze_stem_only
```

这类选项用于更细粒度地控制：

```text
只冻结最早的 Stem
```

而让后面的 stage 继续 fine-tune。

为什么 Stem 更适合冻结？

因为最早层通常学习：

```text
edges
textures
basic local patterns
```

这些特征在不同视觉任务间更通用。

而深层：

```text
task-specific semantics
```

更需要适应 detection。

不过最终 DEIM 训练时到底哪些参数被冻结，必须看：

```text
merged config
+
constructor freezing logic
+
requires_grad
```

不能只凭某一个 YAML 片段判断。

---

# 67. 最可靠的冻结检查方式

真正运行模型后：

```python
for name, p in model.backbone.named_parameters():

    print(
        name,
        p.requires_grad
    )
```

如果：

```text
False
```

才说明该 parameter 真正被冻结。

这比只看：

```yaml
freeze_at
```

更可靠。

---

# 68. 预训练权重为什么重要？

Backbone 通常不是从随机参数开始。

它会加载：

```text
classification / large-scale pretraining
```

得到的 HGNetv2 权重。

这意味着训练 detector 时：

```text
Day 1
```

的 backbone 已经能提取有意义的视觉特征。

因此：

```text
更快收敛
更稳定
通常更高 AP
```

---

# 69. 为什么 Backbone LR 比 detector 小？

第 8 课已经算过：

```text
backbone LR
≈ detector LR 的 1%
```

现在从结构角度更容易理解：

```text
HGNetv2-B5
已经有预训练视觉表示
```

而：

```text
HybridEncoder / DFINETransformer
```

需要更强地适应 detection。

所以：

```text
backbone
→ 小 LR

new detector modules
→ 大 LR
```

这是典型 fine-tuning 策略。

---

# 70. 预训练权重加载为什么经常出现 `strict=False`？

预训练模型可能来自：

```text
classification backbone
```

而当前 detector 中：

```text
只需要 backbone stages
```

原始预训练 checkpoint 可能还包含：

```text
classification head
```

或命名不完全一致。

因此加载时常需要：

```text
过滤 key
```

或：

```text
strict=False
```

目标是：

```text
能匹配的 backbone 权重加载
不匹配的 detection-specific 部分忽略
```

---

# 71. 代码中的 `arch_configs` 为什么非常重要？

`HGNetv2` 不是写死一个网络。

而是：

```python
arch_configs = {
    "B0": {...},
    "B4": {...},
    "B5": {...},
    "B6": {...},
}
```

然后：

```text
name
```

决定选择哪套 stage configuration。

所以：

```yaml
name: B5
```

实际上是在选择一整张网络结构表。

这就是：

```text
data-driven architecture definition
```

的一个例子。

---

# 72. B5 与其他规模的本质差别是什么？

主要可以通过：

```text
stem width
mid channels
out channels
block number
layer number
```

控制。

例如更小模型可以：

```text
更少 channel
更少 block
```

更大模型：

```text
更多 channel
更深 stage
```

因此 HGNetv2 family 可以形成：

```text
速度 ↔ 精度
```

不同 trade-off。

而 X 选择 B5，是因为它属于高容量路线。

---

# 73. 为什么 X 不是简单“所有 channel ×2”？

高效 architecture scaling 往往不是机械：

```text
width ×2
depth ×2
```

因为不同 stage 的计算成本不同。

例如：

```text
160×160
```

非常昂贵。

所以不适合在那里无限加深。

而：

```text
40×40
```

可以承载更多 blocks。

因此 architecture scaling 通常需要：

```text
stage-aware allocation
```

B5 的：

```text
1 / 2 / 5 / 2 blocks
```

就是这种思路。

---

# 74. 从内存角度看三个返回 feature

假设：

```text
float32
= 4 bytes
```

单张图片、仅计算 tensor 元素：

### C3

\[
512\times80\times80
===================

3,276,800
\]

约：

\[
3,276,800\times4
\approx13.1MB
\]

### C4

\[
1024\times40\times40
====================

1,638,400
\]

约：

```text
6.55 MB
```

### C5

\[
2048\times20\times20
====================

819,200
\]

约：

```text
3.28 MB
```

所以虽然深层 channel 更大：

```text
spatial size 的下降
```

让 feature tensor 本身反而更小。

训练时实际显存远高于这些数字，因为还要保存：

```text
intermediate activations
gradients
optimizer states
workspace
```

---

# 75. 为什么 C3 往往最“贵”？

C3：

```text
80×80
```

虽然 channel 只有：

```text
512
```

但 spatial positions：

```text
6400
```

C5：

```text
20×20
```

只有：

```text
400
```

positions。

比例：

\[
6400/400=16
\]

所以高分辨率 feature 在：

```text
memory
FPN fusion
attention
convolution
```

中通常都很昂贵。

这也是实时检测器不会无限加入 stride-4 feature 的原因之一。

---

# 76. PyTorch 实验 1：只验证 shape，不依赖仓库

我们可以先做一个 shape-only toy backbone：

```python
import torch
import torch.nn as nn


class ToyBackbone(nn.Module):

    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                3, 64,
                kernel_size=3,
                stride=4,
                padding=1
            )
        )

        self.stage1 = nn.Conv2d(
            64, 128,
            3, 1, 1
        )

        self.stage2 = nn.Conv2d(
            128, 512,
            3, 2, 1
        )

        self.stage3 = nn.Conv2d(
            512, 1024,
            3, 2, 1
        )

        self.stage4 = nn.Conv2d(
            1024, 2048,
            3, 2, 1
        )

    def forward(self, x):

        x = self.stem(x)
        print("stem  ", x.shape)

        x = self.stage1(x)
        print("stage1", x.shape)

        c3 = self.stage2(x)
        print("stage2", c3.shape)

        c4 = self.stage3(c3)
        print("stage3", c4.shape)

        c5 = self.stage4(c4)
        print("stage4", c5.shape)

        return [c3, c4, c5]


x = torch.randn(
    2, 3, 640, 640
)

model = ToyBackbone()

ys = model(x)
```

注意：

> 这个实验只复现 shape hierarchy，不是 HGNetv2 的真实结构。

它的目的只是验证：

```text
stride chain
```

---

# 77. PyTorch 实验 2：手写 Dense Aggregation

```python
import torch
import torch.nn as nn


class TinyHGBlock(nn.Module):

    def __init__(
        self,
        in_ch,
        mid_ch,
        out_ch,
        layer_num=3
    ):
        super().__init__()

        layers = []

        for i in range(layer_num):

            layers.append(
                nn.Conv2d(
                    in_ch if i == 0
                    else mid_ch,
                    mid_ch,
                    3, 1, 1
                )
            )

        self.layers = nn.ModuleList(
            layers
        )

        concat_ch = (
            in_ch
            + layer_num * mid_ch
        )

        self.aggregate = nn.Conv2d(
            concat_ch,
            out_ch,
            1
        )

    def forward(self, x):

        outputs = [x]

        for layer in self.layers:
            x = layer(x)
            outputs.append(x)

        print(
            "channels before cat:",
            [t.shape[1] for t in outputs]
        )

        x = torch.cat(
            outputs,
            dim=1
        )

        print(
            "after cat:",
            x.shape
        )

        x = self.aggregate(x)

        return x
```

测试：

```python
x = torch.randn(
    2, 64, 80, 80
)

block = TinyHGBlock(
    in_ch=64,
    mid_ch=64,
    out_ch=128,
    layer_num=6
)

y = block(x)

print(y.shape)
```

你应该看到：

```text
cat channel
=
64 + 6×64
=
448
```

---

# 78. PyTorch 实验 3：手写 ESE

```python
class TinyESE(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=1
        )

    def forward(self, x):

        weight = x.mean(
            dim=(2,3),
            keepdim=True
        )

        weight = self.conv(
            weight
        )

        weight = torch.sigmoid(
            weight
        )

        return x * weight
```

测试：

```python
x = torch.randn(
    2, 128, 80, 80
)

ese = TinyESE(128)

y = ese(x)

print(x.shape)
print(y.shape)
```

你会发现：

```text
shape 完全不变
```

因为 ESE 改的是：

```text
channel importance
```

而不是 spatial size/channel count。

---

# 79. PyTorch 实验 4：Depthwise Downsample

```python
x = torch.randn(
    2, 512, 80, 80
)

down = nn.Conv2d(
    512,
    512,
    kernel_size=3,
    stride=2,
    padding=1,
    groups=512
)

y = down(x)

print(y.shape)
```

得到：

```text
[2,512,40,40]
```

注意：

```text
H/W ÷ 2
channel 不变
```

这就是 `HG_Stage.downsample` 的核心 shape 行为。

---

# 80. PyTorch 实验 5：用 Hook 打印真实 Backbone

在真实 DEIM 环境中，最推荐的实验是：

```python
import torch

# model 已通过 YAMLConfig 创建
backbone = model.backbone


def hook_fn(name):

    def hook(
        module,
        inputs,
        output
    ):

        if isinstance(output, torch.Tensor):

            print(
                name,
                tuple(output.shape)
            )

        elif isinstance(output, (list, tuple)):

            print(
                name,
                [
                    tuple(x.shape)
                    for x in output
                    if isinstance(x, torch.Tensor)
                ]
            )

    return hook


for name, module in backbone.named_modules():

    if (
        "stem" in name
        or "stage" in name
    ):
        module.register_forward_hook(
            hook_fn(name)
        )
```

然后：

```python
x = torch.randn(
    1,3,640,640
).cuda()

backbone.eval()

with torch.no_grad():
    ys = backbone(x)
```

最终重点验证：

```text
[B,512,80,80]
[B,1024,40,40]
[B,2048,20,20]
```

---

# 81. 为什么 Hook 比疯狂改源码 `print()` 更好？

如果直接在：

```python
forward()
```

里插几十个：

```python
print(x.shape)
```

会：

```text
污染源码
容易忘记删除
影响多人协作
```

forward hook：

```text
不修改模型源码
```

就可以观察中间输出。

这是以后读：

```text
HGNetv2
HybridEncoder
DFINETransformer
DINOv3 STA
```

都非常有用的技巧。

---

# 82. 但 Hook 也有陷阱

例如：

```text
一个 module
可能被重复调用
```

于是 hook 会打印多次。

还有：

```text
module output
```

可能是：

```text
Tensor
list
tuple
dict
```

不能机械：

```python
output.shape
```

所以调试函数最好先检查类型。

---

# 83. PyTorch 实验 6：统计 trainable 参数

```python
total = 0
trainable = 0

for p in backbone.parameters():

    total += p.numel()

    if p.requires_grad:
        trainable += p.numel()

print(
    "total:",
    total
)

print(
    "trainable:",
    trainable
)

print(
    "ratio:",
    trainable / total
)
```

如果 freeze 策略改变：

```text
trainable
```

会变化。

这可以验证第 8 课配置 merge 的实际效果。

---

# 84. PyTorch 实验 7：统计每个 Stage 参数量

```python
for name, module in backbone.named_children():

    n = sum(
        p.numel()
        for p in module.parameters()
    )

    print(
        name,
        f"{n/1e6:.3f} M"
    )
```

这会回答：

```text
模型参数主要集中在哪些 stage？
```

然后再结合：

```text
spatial size
```

思考：

```text
参数最多
```

和：

```text
计算最贵
```

是否一定是同一个 stage。

答案通常：

```text
不一定。
```

---

# 85. PyTorch 实验 8：检查 `return_idx`

```python
with torch.no_grad():

    feats = backbone(
        torch.randn(
            1,3,640,640,
            device=device
        )
    )

for i, f in enumerate(feats):

    print(
        i,
        tuple(f.shape)
    )
```

预期：

```text
0 → (1,512,80,80)
1 → (1,1024,40,40)
2 → (1,2048,20,20)
```

如果不是：

```text
不要先怀疑数学
```

先检查：

```text
最终 merged config
input resolution
return_idx
模型版本
```

---

# 86. 一个非常重要的初学者问题：为什么叫 C3/C4/C5？

不同论文命名略有差异，但检测中常把 backbone 多尺度输出叫：

```text
C3
C4
C5
```

而 neck/FPN 融合后的 feature 叫：

```text
P3
P4
P5
```

所以：

```text
HGNetv2 output
→ C3 C4 C5

HybridEncoder output
→ P3 P4 P5
```

这种叫法非常有助于区分：

```text
原始 backbone features
```

与：

```text
融合后的 pyramid features
```

---

# 87. C3 为什么不是 Stage3？

这里很容易混淆。

本课程中：

```text
Stage2 output
```

可能被称作：

```text
C3
```

因为：

```text
C3/C4/C5
```

是 detection feature pyramid 的语义层级命名。

而：

```text
Stage1/2/3/4
```

是 HGNetv2 自己的模块索引。

两套编号体系不是同一回事。

所以：

```text
HGNet Stage2
→ detection C3
```

完全不矛盾。

---

# 88. Backbone 到 Encoder 的接口契约

第 8 课看到：

```yaml
HybridEncoder:
    in_channels:
        [512,1024,2048]

    feat_strides:
        [8,16,32]
```

现在我们终于能证明它为什么这样写：

```text
HGNetv2 Stage2
→ 512 channels
→ stride 8

HGNetv2 Stage3
→ 1024 channels
→ stride 16

HGNetv2 Stage4
→ 2048 channels
→ stride 32
```

所以：

```text
Backbone output metadata
```

必须和：

```text
Encoder input config
```

完全一致。

---

# 89. 如果 `in_channels` 写错会怎样？

假设 backbone 输出：

```text
512 channels
```

但 HybridEncoder 的 input projection 以为：

```text
256 channels
```

那么 Conv 会期待：

```text
input.shape[1] = 256
```

实际却：

```text
512
```

PyTorch 会直接报 channel mismatch。

所以配置中的：

```yaml
in_channels
```

不是装饰信息。

它是：

```text
模块接口契约
```

---

# 90. 如果 `feat_strides` 写错呢？

这可能更危险。

有时 tensor shape 本身还能运行，但：

```text
positional encoding
anchor generation
reference points
feature scale interpretation
```

会被错误理解。

所以：

```text
feat_strides
```

必须真实对应 backbone 的下采样倍率。

这种错误不一定第一行就 crash，却可能让训练表现异常。

---

# 91. 为什么我们一定要逐层做 shape 推导？

因为 shape 是读视觉源码最可靠的“物理定律”。

变量名可能：

```text
x
feat
memory
src
out
```

随作者改变。

但：

```text
[B,512,80,80]
```

告诉你的信息非常具体：

```text
batch
channel
spatial scale
模块位置
```

只要 shape 推导没有断，你通常不会完全迷路。

---

# 92. HGNetv2-B5 的核心设计可以压缩成四句话

### 1. 高效 Stem

```text
逐步下采样
+
pool/conv 双分支
```

### 2. HG Block dense aggregation

```text
保留多个中间 feature
+
concat
+
aggregation
```

### 3. 深层 LightConv

```text
pointwise / depthwise
+
larger kernel
```

### 4. Multi-scale outputs

```text
stride 8 / 16 / 32
```

供 detector 使用。

---

# 93. HGNetv2 与普通 ResNet 的直觉对比

ResNet block 的核心：

```text
x
↓
F(x)
↓
F(x)+x
```

HGNetv2 HG Block 更强调：

```text
x0
↓
x1
↓
x2
↓
...
↓
x6

[x0,x1,...,x6]
↓
concat
↓
aggregation
↓
optional residual
```

所以：

```text
ResNet
→ residual learning 是核心

HGNetv2
→ dense aggregation + efficient convolution
   再结合 residual
```

---

# 94. 与 DenseNet 又有什么相似？

DenseNet 也强调：

```text
前面 feature
被后面重复利用
```

HG Block 同样：

```text
保存多层输出
最后 concat
```

所以思想上有：

```text
dense feature reuse
```

的相似性。

但具体：

```text
block topology
aggregation方式
channel strategy
attention
downsample
```

都不同。

不要把 HGNetv2 简化成：

```text
“就是 DenseNet”
```

---

# 95. 为什么这种 Backbone 适合实时检测？

需要同时满足：

```text
表达能力
+
多尺度输出
+
低延迟
```

HGNetv2 使用：

```text
dense aggregation
→ 增强 feature reuse

depthwise/light conv
→ 控制计算

large-kernel light blocks
→ 扩大 spatial context

multi-scale hierarchy
→ 直接适配 detector
```

所以它很适合作为实时 DETR 系列的 CNN backbone。

---

# 96. 但 Backbone 强不等于检测器一定强

目标检测最终还取决于：

```text
Backbone
+
HybridEncoder
+
Decoder
+
Matcher
+
Loss
+
Augmentation
+
Optimization
```

所以不要看到：

```text
HGNetv2-B5 feature 很强
```

就把 DEIM-D-FINE-X 的性能全部归功于 backbone。

后面课程正是要逐个拆开这些因素。

---

# 97. 本课最重要的 Shape Trace

请尽量能默写：

```text
Input
[B,3,640,640]

        │
        ▼
Stem
[B,64,160,160]
stride 4

        │
        ▼
Stage1
[B,128,160,160]
stride 4

        │
        ▼
Stage2
[B,512,80,80]
stride 8
        │
        └── return_idx=1
            C3

        │
        ▼
Stage3
[B,1024,40,40]
stride 16
        │
        └── return_idx=2
            C4

        │
        ▼
Stage4
[B,2048,20,20]
stride 32
        │
        └── return_idx=3
            C5
```

最终：

```python
[
    C3,
    C4,
    C5
]
```

---

# 98. 本课最重要的 HG Block Shape Trace

以 Stage3 第一个 block：

```text
input
[B,512,40,40]

│
├── x0
│   512 ch
│
├── x1
│   256 ch
│
├── x2
│   256 ch
│
├── x3
│   256 ch
│
├── x4
│   256 ch
│
├── x5
│   256 ch
│
└── x6
    256 ch

        │
        ▼
concat

512 + 6×256
=
2048 channels

        │
        ▼
aggregation

[B,1024,40,40]

        │
        ▼
ESE / channel reweight

[B,1024,40,40]
```

如果这一张图能真正理解，HGNetv2 最核心的 block 已经吃透大半。

---

# 99. 常见误区检查

## 误区 1

```text
return_idx=[1,2,3]
```

表示返回：

```text
第 1/2/3 个 Tensor channel。
```

错误。

它表示选定 stage outputs。

---

## 误区 2

```text
Stage3 = C3
```

不一定。

HGNetv2 stage 编号与 detection C-level 命名不是同一套编号。

---

## 误区 3

```text
2048 channels
```

意味着内部每层都在 2048 channels 上做卷积。

错误。

HG Block 有：

```text
mid_chs
```

大量内部计算在更窄 channel 上完成。

---

## 误区 4

Depthwise Conv 会充分混合 channels。

错误。

它主要做：

```text
per-channel spatial filtering
```

channel mixing 需要 pointwise 等操作。

---

## 误区 5

ESE 改变 feature shape。

错误。

通常：

```text
[B,C,H,W]
→
[B,C,H,W]
```

它改变的是 channel 权重。

---

## 误区 6

冻结参数后 forward 不执行。

错误。

冻结通常只是：

```text
requires_grad=False
```

---

## 误区 7

Backbone 已经输出多尺度，所以 HybridEncoder 没意义。

错误。

Backbone 只是产生不同层级 feature。

HybridEncoder 还要做：

```text
channel alignment
intra-scale interaction
top-down fusion
bottom-up fusion
```

下一课会详细解释。

---

# 100. 本课练习

## 练习 1：Stem

从：

```text
[B,3,640,640]
```

推导 B5 Stem 最终：

```text
[B,64,160,160]
```

并解释两次下采样分别发生在哪里。

---

## 练习 2：Stage

写出：

```text
Stage1
Stage2
Stage3
Stage4
```

的：

```text
output channels
spatial size
stride
```

---

## 练习 3：return_idx

为什么：

```yaml
return_idx: [1,2,3]
```

最终返回的是：

```text
512 / 1024 / 2048
```

三个 channel 的 feature？

---

## 练习 4：HG Block

Stage2 第一个 block：

```text
in=128
mid=128
layer_num=6
```

concat 后 channel 是多少？

---

## 练习 5

Stage2 第二个 block：

```text
in=512
mid=128
layer_num=6
```

concat 后 channel 是多少？

---

## 练习 6

Stage3 第一个 block：

```text
in=512
mid=256
layer_num=6
```

concat channel？

---

## 练习 7

Stage4 第二个 block：

```text
in=2048
mid=512
layer_num=6
```

concat channel？

---

## 练习 8

为什么 Stage2 第一个 block 不适合直接 residual，而第二个 block适合？

---

## 练习 9

计算：

```text
512-channel
3×3
full conv
```

和：

```text
512-channel
3×3
depthwise conv
```

的卷积核参数量比例。

---

## 练习 10

解释：

```text
C3/C4/C5
```

和：

```text
P3/P4/P5
```

的区别。

---

# 101. 练习答案

## 练习 2

```text
Stage1
[B,128,160,160]
stride 4

Stage2
[B,512,80,80]
stride 8

Stage3
[B,1024,40,40]
stride 16

Stage4
[B,2048,20,20]
stride 32
```

---

## 练习 4

\[
128+6\times128
==============

896
\]

---

## 练习 5

\[
512+6\times128
==============

1280
\]

---

## 练习 6

\[
512+6\times256
==============

2048
\]

---

## 练习 7

\[
2048+6\times512
===============

5120
\]

---

## 练习 8

第一个 block：

```text
128 → 512
```

input/output shape 不同。

第二个：

```text
512 → 512
```

shape 一致，可以直接 identity add。

---

## 练习 9

普通：

\[
512\times512\times9
\]

Depthwise：

\[
512\times9
\]

比例：

\[
512
\]

---

## 练习 10

```text
C3/C4/C5
→ backbone 原始 multi-scale features

P3/P4/P5
→ encoder/neck 融合后的 pyramid features
```

---

# 102. 建议你自己做的“源码追踪表”

读真实 `hgnetv2.py` 时，建议创建下面的表并边运行边填：

| Module      | Input             | Operation  | Output              |
| ----------- | ----------------- | ---------- | ------------------- |
| stem1       | `[1,3,640,640]` | 3×3 s2    | `[1,32,320,320]`  |
| stem branch | ...               | pool/conv  | ...                 |
| stem3       | ...               | 3×3 s2    | ...                 |
| stem4       | ...               | 1×1       | `[1,64,160,160]`  |
| stage1      | ...               | HG Block   | `[1,128,160,160]` |
| stage2      | ...               | DW s2 + HG | `[1,512,80,80]`   |
| stage3      | ...               | DW s2 + HG | `[1,1024,40,40]`  |
| stage4      | ...               | DW s2 + HG | `[1,2048,20,20]`  |

这比只看网络结构图有效得多。

---

# 103. 源码阅读顺序建议

打开：

```text
engine/backbone/hgnetv2.py
```

不要从第 1 行一直机械读到最后。

推荐顺序：

```text
① HGNetv2
   ↓
② arch_configs["B5"]
   ↓
③ HGNetv2.__init__
   ↓
④ HGNetv2.forward
   ↓
⑤ StemBlock
   ↓
⑥ HG_Stage
   ↓
⑦ HG_Block
   ↓
⑧ LightConvBNAct
   ↓
⑨ ConvBNAct
   ↓
⑩ EseModule
```

为什么倒着读？

因为先知道：

```text
大结构需要哪些小模块
```

再进入小模块，会更容易理解“这个类为什么存在”。

---

# 104. 遇到一个类时固定问 6 个问题

以后读任何视觉模型，都可以用这套方法。

例如看到：

```python
class HG_Block(...)
```

先问：

```text
1. 它为什么存在？
2. 输入 shape 是什么？
3. 输出 shape 是什么？
4. 哪一步改变 H/W？
5. 哪一步改变 C？
6. 哪些参数决定这些变化？
```

然后才看：

```text
每一行 Python
```

这样不会陷入局部语法。

---

# 105. 本课与下一课的接口

现在 backbone 返回：

```text
C3
[B,512,80,80]

C4
[B,1024,40,40]

C5
[B,2048,20,20]
```

但 DFINETransformer 希望处理统一维度的多尺度 feature。

于是下一个问题自然出现：

> **512、1024、2048 三个 channel 完全不同，怎样让它们互相融合？**

以及：

> **为什么 C5 要先经过 Transformer encoder，而 C3/C4 又通过 FPN/PAN 与它交换信息？**

这就是第 10 课：

# HybridEncoder

---

# 106. 下一课的第一步 shape

HybridEncoder 首先会把：

```text
C3 [B, 512,80,80]
C4 [B,1024,40,40]
C5 [B,2048,20,20]
```

投影到统一的 X hidden dimension：

```text
[B,384,80,80]
[B,384,40,40]
[B,384,20,20]
```

然后才开始：

```text
AIFI / Transformer interaction
+
top-down FPN
+
bottom-up PAN
```

所以本课最终输出：

```text
[512,1024,2048]
```

恰好就是下一课的输入。

---

# 107. 本课最终知识图

```text
                 HGNetv2-B5

RGB image
[B,3,640,640]

        │
        ▼

┌─────────────────────────┐
│          Stem           │
│                         │
│ Conv s2                 │
│ Pool/Conv branches      │
│ concat                  │
│ Conv s2                 │
└───────────┬─────────────┘
            │
            ▼
[B,64,160,160]
stride 4

            │
            ▼

┌─────────────────────────┐
│ Stage1                  │
│ HG Block ×1             │
│ dense aggregation       │
└───────────┬─────────────┘
            │
            ▼
[B,128,160,160]

            │
            ▼
     DW downsample s2

            │
            ▼

┌─────────────────────────┐
│ Stage2                  │
│ HG Block ×2             │
│ normal conv blocks      │
└───────────┬─────────────┘
            │
            ▼
C3
[B,512,80,80]
stride 8

            │
            ▼
     DW downsample s2

            │
            ▼

┌─────────────────────────┐
│ Stage3                  │
│ HG Block ×5             │
│ LightConv + 5×5         │
└───────────┬─────────────┘
            │
            ▼
C4
[B,1024,40,40]
stride 16

            │
            ▼
     DW downsample s2

            │
            ▼

┌─────────────────────────┐
│ Stage4                  │
│ HG Block ×2             │
│ LightConv + 5×5         │
└───────────┬─────────────┘
            │
            ▼
C5
[B,2048,20,20]
stride 32

            │
            ▼

return_idx = [1,2,3]

            │
            ▼

[
  C3,
  C4,
  C5
]

            │
            ▼

HybridEncoder
```

---

# 108. 本课必须能独立回答的 15 个问题

进入第 10 课前，请尽量做到不看讲义回答：

```text
1. HGNetv2 在 DEIM-D-FINE-X 中负责什么？

2. X 为什么选择 B5？

3. B5 Stem 如何把 640×640 变成 160×160？

4. 为什么 Stem 中有 pool/conv 两个分支？

5. HG Block 为什么保存所有中间 layer 输出？

6. concat 后 channel 怎样计算？

7. mid_chs 和 out_chs 有什么区别？

8. Stage2 为什么输出 512 channels？

9. Stage3/4 为什么使用 Light Block？

10. Depthwise Conv 为什么便宜？

11. ESE 做了什么？

12. return_idx=[1,2,3] 到底返回什么？

13. 为什么最终得到 stride 8/16/32？

14. 为什么三个返回 feature 的 channels 是
    512/1024/2048？

15. 为什么下一步还需要 HybridEncoder？
```

如果这 15 个问题都能讲清楚，你已经不只是“知道 HGNetv2 是 backbone”，而是真正开始具备独立读 backbone 源码的能力。

---

# 109. 本课源码核对说明

本课结构依据官方 DEIM 仓库当前 `engine/backbone/hgnetv2.py` 与 X 级 D-FINE/HGNetv2 配置体系核对。当前官方实现中的 B5 architecture table明确给出：

```text
stem_channels = [3,32,64]

Stage1 = [64,64,128,1,False,False,3,6]
Stage2 = [128,128,512,2,True,False,3,6]
Stage3 = [512,256,1024,5,True,True,5,6]
Stage4 = [1024,512,2048,2,True,True,5,6]
```

其中 stage tuple 的语义为：

```text
in_channels
mid_channels
out_channels
num_blocks
downsample
light_block
kernel_size
layer_num
```

官方 X 配置使用：

```text
name = B5
return_idx = [1,2,3]
```

因此 640×640 输入的三个检测 backbone 输出为：

```text
[B,512,80,80]
[B,1024,40,40]
[B,2048,20,20]
```

后续课程若仓库版本发生改变，应以你实际 checkout 的 commit 和运行时 hook 输出为最终准绳。

---

# 110. 第 10 课预告

下一课：

# 《HybridEncoder——DEIM 系列最容易被低估的模块》

我们会从本课三个输出直接开始：

```text
C3 [B,512,80,80]
C4 [B,1024,40,40]
C5 [B,2048,20,20]
```

逐行追踪：

```text
input_proj
↓
[B,384,80,80]
[B,384,40,40]
[B,384,20,20]

↓
Transformer intra-scale interaction

↓
top-down FPN

↓
bottom-up PAN

↓
P3 / P4 / P5
```

重点回答：

```text
Backbone 已经多尺度了，
为什么还需要 HybridEncoder？

为什么只对某个 level 做 Transformer？

FPN 为什么是 top-down？

PAN 为什么还要 bottom-up？

concat 后 channel 怎么算？

C3/C4/C5 与 P3/P4/P5 到底有什么区别？
```

到第 10 课结束，我们就会第一次真正拥有：

```text
可以交给 DFINETransformer 的三尺度检测特征。
```
