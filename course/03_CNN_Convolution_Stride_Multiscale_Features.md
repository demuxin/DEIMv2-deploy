# 第 3 课：CNN、卷积、Stride 与多尺度特征

> **课程定位**：DEIMv2 从零到源码 · 第 3 / 24 课\
> **本课目标**：从零理解 CNN 的卷积运算、kernel / stride / padding /
> dilation /
> groups、通道变化、下采样、参数量、计算量与感受野，并最终能够从
> `640×640` 输入手算 HGNetv2/检测 Backbone 的多尺度输出 shape。\
> **重点连接**：`deim_dfine_x → HGNetv2-B5 → HybridEncoder`。\
> **学习原则**：这一课不是背 CNN 名词，而是要做到"看到一个
> `Conv2d`，能立即推导输出 shape、参数量和它对空间分辨率做了什么"。

------------------------------------------------------------------------

# 0. 从第 1、2 课接着走

第 1 课我们建立了总地图：

``` text
Image
[B,3,640,640]
       │
       ▼
Backbone
       │
       ▼
multi-scale features
       │
       ▼
Encoder
       │
       ▼
Decoder
       │
       ▼
Bounding Boxes
```

第 2 课我们已经知道如何把：

``` text
[B,C,H,W]
```

变成：

``` text
[B,HW,C]
```

但还有一个非常重要的问题没有解决：

> `[B,3,640,640]` 到底怎样变成
> `[B,C,80,80]`、`[B,C,40,40]`、`[B,C,20,20]`？

答案主要就在：

``` text
Convolution
+
Stride
+
Hierarchical Backbone
```

这一课解决这件事。

------------------------------------------------------------------------

# 1. CNN 为什么需要卷积？

先假设一张图片：

``` text
[3,640,640]
```

如果直接把所有像素 flatten：

\[ 3`\times640`{=tex}`\times640`{=tex}=1,228,800 \]

得到超过 120 万维输入。

如果直接接一个全连接层：

``` text
1,228,800 → 1024
```

仅权重就需要：

\[ 1,228,800`\times1024`{=tex} = 1,258,291,200 \]

约 **12.58 亿个参数**。

这显然非常昂贵。

卷积的核心思想不同：

> 用一个很小的局部 kernel 在整张图片上重复扫描，并共享同一组参数。

例如一个：

``` text
3×3 convolution kernel
```

只关注当前位置周围的局部区域。

------------------------------------------------------------------------

# 2. 最简单的二维卷积

先不考虑 channel。

假设输入：

``` text
5 × 5
```

kernel：

``` text
3 × 3
```

stride：

``` text
1
```

padding：

``` text
0
```

kernel 第一次覆盖：

``` text
input

┌───────┐ . .
│ x x x │ . .
│ x x x │ . .
│ x x x │ . .
└───────┘ . .
. . . . .
. . . . .
```

然后向右移动一格：

``` text
. ┌───────┐ .
. │ x x x │ .
. │ x x x │ .
. │ x x x │ .
. └───────┘ .
```

对于每个位置，kernel 与局部像素做：

\[ y = `\sum`{=tex}*{i,j}x*{ij}w\_{ij}+b \]

这就是一个输出值。

------------------------------------------------------------------------

# 3. 卷积输出尺寸公式

这是本课必须掌握的公式。

对于一维空间尺寸：

\[ O= `\left`{=tex}`\lfloor`{=tex} `\frac{
I+2P-D(K-1)-1
}{S}`{=tex} +1 `\right`{=tex}`\rfloor`{=tex} \]

其中：

``` text
I = Input size
K = Kernel size
S = Stride
P = Padding
D = Dilation
O = Output size
```

二维情况分别对：

``` text
Height
Width
```

计算。

所以：

\[ H\_{out} = `\left`{=tex}`\lfloor`{=tex} `\frac{
H_{in}+2P_h-D_h(K_h-1)-1
}{S_h}`{=tex} +1 `\right`{=tex}`\rfloor`{=tex} \]

\[ W\_{out} = `\left`{=tex}`\lfloor`{=tex} `\frac{
W_{in}+2P_w-D_w(K_w-1)-1
}{S_w}`{=tex} +1 `\right`{=tex}`\rfloor`{=tex} \]

------------------------------------------------------------------------

# 4. 第一次手算：5×5 输入 + 3×3 卷积

参数：

``` text
I = 5
K = 3
S = 1
P = 0
D = 1
```

代入：

\[ O= `\frac{5-1(3-1)-1}{1}`{=tex}+1 \]

# \[

5-2-1+1 \]

\[ =3 \]

所以：

``` text
5×5
↓ Conv3×3, stride=1, padding=0
3×3
```

直觉上：

``` text
3×3 kernel
```

在 5×5 上横向可以放 3 次，纵向也可以放 3 次。

------------------------------------------------------------------------

# 5. Padding 为什么存在？

现在仍然：

``` text
Input = 5×5
Kernel = 3×3
Stride = 1
```

但：

``` text
Padding = 1
```

相当于在输入四周补一圈。

原：

``` text
5×5
```

逻辑上变成：

``` text
7×7
```

代入：

\[ O= `\frac{5+2(1)-3}{1}`{=tex}+1 \]

\[ =5 \]

于是：

``` text
5×5
↓ 3×3, stride=1, padding=1
5×5
```

这就是非常常见的：

> **保持空间尺寸不变的 3×3 convolution。**

------------------------------------------------------------------------

# 6. Stride 是什么？

Stride 表示：

> kernel 每次移动多少个像素位置。

如果：

``` text
stride = 1
```

kernel：

``` text
一步一步移动
```

如果：

``` text
stride = 2
```

则：

``` text
每次跨两个位置
```

因此 stride=2 常常会让：

``` text
H → H/2
W → W/2
```

也就是下采样。

------------------------------------------------------------------------

# 7. 最重要的 640 → 320

假设：

``` python
nn.Conv2d(
    in_channels=3,
    out_channels=32,
    kernel_size=3,
    stride=2,
    padding=1
)
```

输入：

``` text
[B,3,640,640]
```

计算：

\[ O = `\left`{=tex}`\lfloor`{=tex} `\frac{640+2-3}{2}`{=tex}+1
`\right`{=tex}`\rfloor`{=tex} \]

# \[

`\left`{=tex}`\lfloor
319.5`{=tex}+1 `\right`{=tex}`\rfloor`{=tex} = 320 \]

输出：

``` text
[B,32,320,320]
```

注意这里同时发生两件事：

``` text
Channel:
3 → 32

Spatial:
640×640 → 320×320
```

这两个变化必须分开理解。

------------------------------------------------------------------------

# 8. 为什么 channel 变多，而空间变小？

这是 CNN Backbone 中非常常见的设计：

``` text
空间分辨率逐渐下降
Channel 数逐渐增加
```

例如：

``` text
[B,3,640,640]
      ↓
[B,32,320,320]
      ↓
[B,64,160,160]
      ↓
[B,128,80,80]
      ↓
[B,256,40,40]
      ↓
[B,512,20,20]
```

直觉上：

``` text
浅层：
空间信息丰富
每个位置描述比较简单

深层：
空间位置变少
但每个位置拥有更丰富的 feature channels
```

所以 CNN 在做一种：

> **空间压缩 + 语义扩展**

但请注意：

这只是理解 CNN 的直觉，不意味着 channel
越多就一定"语义越高级"；真正能力还取决于网络结构、训练方式、参数等。

------------------------------------------------------------------------

# 9. `Conv2d` 的输入输出 shape

PyTorch：

``` python
conv = torch.nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    stride=2,
    padding=1
)
```

输入：

``` text
[B,3,640,640]
```

输出：

``` text
[B,64,320,320]
```

规律：

``` text
Batch
B
→ 不变

Channel
Cin
→ Cout

Height/Width
→ 根据 K/S/P/D 公式计算
```

因此：

\[ \[B,C\_{in},H,W\] `\rightarrow
[B,C_{out},H',W']`{=tex}\]

------------------------------------------------------------------------

# 10. 一个卷积 kernel 其实有多深？

假设：

``` text
Cin = 3
K = 3
```

一个输出 channel 对应的 kernel 并不是：

``` text
3×3
```

而是：

``` text
3×3×3
```

因为它必须同时观察：

``` text
R
G
B
```

三个输入 channels。

如果：

``` text
Cin = 256
```

一个普通 3×3 kernel 实际有：

\[ 256`\times3`{=tex}`\times3`{=tex} \]

个权重。

如果输出：

``` text
Cout = 512
```

则需要 512 组这样的 kernel。

所以普通 Conv2d 权重数：

\[ C\_{out}`\times `{=tex}C\_{in}`\times `{=tex}K_h`\times `{=tex}K_w \]

如果有 bias：

\[ Params = C\_{out} (C\_{in}K_hK_w+1) \]

------------------------------------------------------------------------

# 11. 手算卷积参数量

假设：

``` python
nn.Conv2d(
    256,
    512,
    kernel_size=3,
    bias=False
)
```

参数量：

\[ 512`\times256`{=tex}`\times3`{=tex}`\times3`{=tex} \]

先：

\[ 256`\times9`{=tex}=2304 \]

再：

\[ 2304`\times512`{=tex} = 1,179,648 \]

所以约：

``` text
1.18 M parameters
```

只是一层卷积。

这也解释了为什么现代网络会使用：

``` text
1×1 convolution
depthwise convolution
group convolution
bottleneck
```

来控制计算量。

------------------------------------------------------------------------

# 12. `1×1 Conv` 到底有什么用？

初学者第一次看到：

``` python
nn.Conv2d(
    256,
    128,
    kernel_size=1
)
```

容易觉得：

> 1×1 能看什么？

它虽然不扩大空间邻域，但可以在 **channel 维度做线性组合**。

输入一个位置：

``` text
256-dimensional feature
```

1×1 Conv 可以把它变成：

``` text
128-dimensional feature
```

因此：

``` text
[B,256,H,W]
↓ Conv1×1
[B,128,H,W]
```

空间尺寸不变。

------------------------------------------------------------------------

# 13. `1×1 Conv` 参数量优势

假设：

``` text
256 → 128
```

使用 1×1：

\[ 128`\times256`{=tex} = 32,768 \]

如果使用 3×3：

\[ 128`\times256`{=tex}`\times9`{=tex} = 294,912 \]

是前者的：

\[ 9`\times`{=tex} \]

所以 1×1 Conv 非常适合：

``` text
channel projection
channel compression
feature alignment
```

后面 HybridEncoder 里就会频繁遇到 channel projection。

------------------------------------------------------------------------

# 14. Groups 是什么？

普通 convolution：

``` text
每个输出 channel
都可以连接所有输入 channels
```

如果：

``` python
groups > 1
```

输入/输出 channels 被分组。

例如：

``` text
Cin = 64
Cout = 64
groups = 4
```

相当于分成 4 组：

``` text
每组：
16 input channels
→
16 output channels
```

不同组之间不直接通过该卷积混合。

------------------------------------------------------------------------

# 15. Depthwise Convolution

特殊情况：

``` text
groups = Cin
```

并且常见：

``` text
Cout = Cin
```

则每个 channel 独立做自己的 spatial convolution。

例如：

``` python
nn.Conv2d(
    256,
    256,
    kernel_size=3,
    groups=256
)
```

普通 3×3 Conv 参数：

\[ 256`\times256`{=tex}`\times9`{=tex} = 589,824 \]

Depthwise Conv：

\[ 256`\times1`{=tex}`\times9`{=tex} = 2,304 \]

差距巨大。

这也是轻量网络常用结构。

------------------------------------------------------------------------

# 16. 真实 HGNetv2：Stage Downsample 就用了 Depthwise Conv

HGNetv2 的 stage 中，官方实现的 downsample 可以概括为：

``` python
ConvBNAct(
    in_chs,
    in_chs,
    kernel_size=3,
    stride=2,
    groups=in_chs,
    use_act=False,
)
```

注意：

``` text
Cin = Cout = in_chs
groups = in_chs
```

这正是 depthwise convolution。

而：

``` text
stride = 2
```

负责：

``` text
H → H/2
W → W/2
```

所以它用很低的参数成本完成空间下采样。

------------------------------------------------------------------------

# 17. LightConvBNAct 为什么也有 Depthwise 思想？

HGNetv2 中的轻量卷积模块可以概括为：

``` text
1×1 Conv
   ↓
Depthwise K×K Conv
```

即：

``` python
conv1:
1×1 channel transform

conv2:
K×K, groups=out_chs
```

这是很典型的：

``` text
Pointwise
+
Depthwise
```

组合。

直觉：

``` text
1×1
→ 负责 channels 之间的信息混合

Depthwise K×K
→ 负责每个 channel 的空间信息提取
```

把"channel mixing"和"spatial filtering"部分拆开，可以显著减少计算。

------------------------------------------------------------------------

# 18. Dilation 是什么？

普通 3×3 kernel：

``` text
x x x
x x x
x x x
```

dilation=2 时：

``` text
x . x . x
. . . . .
x . x . x
. . . . .
x . x . x
```

kernel 仍然只有：

``` text
3×3 = 9
```

个采样点，

但覆盖的有效范围更大。

有效 kernel size：

\[ K\_{eff} = D(K-1)+1 \]

例如：

``` text
K=3
D=2
```

则：

\[ K\_{eff} = 2(3-1)+1 = 5 \]

所以 3×3 dilation=2 在空间覆盖范围上相当于 5×5。

------------------------------------------------------------------------

# 19. Stride 与 Dilation 的本质区别

``` text
Stride
→ 控制 kernel 每次移动多少
→ 常用于改变输出分辨率

Dilation
→ 控制 kernel 内采样点之间的间距
→ 扩大覆盖范围
```

不要混淆：

``` text
stride=2
```

通常会下采样。

``` text
dilation=2
```

如果 padding 合适，完全可以保持 H/W 不变。

------------------------------------------------------------------------

# 20. Pooling 是什么？

CNN 早期经常使用：

``` text
MaxPool
AveragePool
```

例如：

``` python
nn.MaxPool2d(
    kernel_size=2,
    stride=2
)
```

可以：

``` text
640×640
→
320×320
```

与卷积不同：

``` text
Pooling
通常没有需要学习的卷积 kernel 权重。
```

MaxPool：

\[ y=`\max`{=tex}(x_1,x_2,`\ldots`{=tex}) \]

AveragePool：

\[ y=`\frac{1}{N}`{=tex}`\sum`{=tex}\_i x_i \]

HGNetv2 Stem 中也使用了 MaxPool 分支。

------------------------------------------------------------------------

# 21. BatchNorm 是什么？

真实 CNN 中很少只有：

``` text
Conv
```

更常见：

``` text
Conv
↓
BatchNorm
↓
Activation
```

HGNetv2 直接把它封装为：

``` text
ConvBNAct
```

可以先理解：

``` text
Conv
→ 提取/变换 feature

BatchNorm
→ 对中间 activation 做归一化与可学习缩放/平移

Activation
→ 引入非线性
```

后面第 9 课读 HGNetv2 时会进一步讲。

------------------------------------------------------------------------

# 22. ReLU 为什么需要？

如果所有层都只是线性变换：

\[ y=W_2(W_1x) \]

则：

\[ y=(W_2W_1)x \]

很多层依然可以合并成一个线性变换。

加入非线性：

\[ y=W_2`\sigma`{=tex}(W_1x) \]

就不能简单合并。

ReLU：

\[ ReLU(x)=`\max`{=tex}(0,x) \]

HGNetv2 的 `ConvBNAct` 默认使用 ReLU。

------------------------------------------------------------------------

# 23. 什么叫 Downsampling？

下采样就是降低空间分辨率。

例如：

``` text
640×640
→
320×320
→
160×160
→
80×80
→
40×40
→
20×20
```

如果每一步都是：

``` text
stride=2
```

那么总 stride：

``` text
2
4
8
16
32
```

例如：

``` text
640 / 8 = 80
```

所以：

``` text
80×80
```

feature map 被称为：

``` text
stride-8 feature
```

或者：

``` text
1/8 resolution feature
```

------------------------------------------------------------------------

# 24. "Stride 8"不是说最后一层 stride 参数一定等于 8

这是一个非常重要的概念。

假设连续三层：

``` text
Layer 1: stride=2
Layer 2: stride=2
Layer 3: stride=2
```

总 stride：

\[ 2`\times2`{=tex}`\times2`{=tex}=8 \]

所以第三层相对于原图是：

``` text
stride 8
```

虽然每一个单独 Conv 的：

``` text
stride
```

都只是 2。

因此以后看到：

``` text
feat_strides = [8,16,32]
```

表示的是：

> 这些输出 feature maps 相对于原始输入图片的累计 stride。

------------------------------------------------------------------------

# 25. 从 640 手算检测器最经典的三尺度

输入：

``` text
640×640
```

stride 8：

\[ 640/8=80 \]

stride 16：

\[ 640/16=40 \]

stride 32：

\[ 640/32=20 \]

所以：

``` text
stride 8  → 80×80
stride 16 → 40×40
stride 32 → 20×20
```

这就是第 1、2 课反复出现的：

``` text
P3
P4
P5
```

常见空间尺度。

------------------------------------------------------------------------

# 26. 为什么目标检测特别喜欢多尺度？

假设图片里有：

``` text
一个很小的人
一辆中等大小的车
一辆占据半张图的大巴
```

如果只有：

``` text
20×20
```

feature map，

那么 640×640 图片被压缩得很厉害。

小目标可能只对应非常少的 feature cells。

如果有：

``` text
80×80
```

feature map，

空间定位更精细。

因此检测器常常保留：

``` text
高分辨率 feature
+
中分辨率 feature
+
低分辨率 feature
```

然后再进行融合。

------------------------------------------------------------------------

# 27. 但是"P3 只检测小目标"是错误的简化

初学时常听：

``` text
P3 → 小目标
P4 → 中目标
P5 → 大目标
```

这只是直觉。

现代检测器会：

``` text
top-down fusion
bottom-up fusion
attention
cross-scale interaction
```

所以不同层的信息会互相交换。

更准确的说法：

> 高分辨率层通常保留更细的空间信息，低分辨率深层通常具有更大的有效上下文和更抽象的特征；多尺度融合让检测器综合利用这些信息。

------------------------------------------------------------------------

# 28. 现在进入真实 `deim_dfine_x` Backbone：HGNetv2-B5

`deim_dfine_x` 使用 HGNetv2-B5 路线。

官方 HGNetv2 的 B5 配置核心可以整理为：

``` text
Stem:
3 → 32 → 64

Stage 1:
64 → 128
downsample = False

Stage 2:
128 → 512
downsample = True

Stage 3:
512 → 1024
downsample = True
light_block = True

Stage 4:
1024 → 2048
downsample = True
light_block = True
```

B5 的 stage 结构参数还包括：

``` text
Stage 1:
mid=64
out=128
blocks=1
kernel=3
layers/block=6

Stage 2:
mid=128
out=512
blocks=2
kernel=3
layers/block=6

Stage 3:
mid=256
out=1024
blocks=5
kernel=5
layers/block=6

Stage 4:
mid=512
out=2048
blocks=2
kernel=5
layers/block=6
```

这里先重点关注：

``` text
spatial size
channel
downsample
```

HGBlock 内部第 9 课再彻底拆。

------------------------------------------------------------------------

# 29. HGNetv2 自己声明了哪些输出 stride？

官方实现中：

``` python
self._out_strides = [4, 8, 16, 32]
```

对应：

``` text
Stage 1 → stride 4
Stage 2 → stride 8
Stage 3 → stride 16
Stage 4 → stride 32
```

而常见：

``` python
return_idx = [1,2,3]
```

表示返回：

``` text
Stage 2
Stage 3
Stage 4
```

所以检测器得到的正好是：

``` text
stride 8
stride 16
stride 32
```

三层。

------------------------------------------------------------------------

# 30. HGNetv2 Stem：为什么输出已经是 stride 4？

官方 Stem 的关键路径可以抽象为：

``` text
input
  │
  ▼
stem1: 3×3, stride=2
  │
  ├───────────────┐
  │               │
pool branch     conv branch
  │               │
  └────── cat ────┘
          │
          ▼
stem3: 3×3, stride=2
          │
          ▼
stem4: 1×1, stride=1
```

有两次：

``` text
stride=2
```

所以累计 stride：

\[ 2`\times2`{=tex}=4 \]

因此：

``` text
640
→ 320
→ 160
```

Stem 最终：

``` text
stride 4
```

------------------------------------------------------------------------

# 31. HGNetv2-B5 Stem 的真实 Shape 直觉推导

输入：

``` text
[B,3,640,640]
```

第一层：

``` text
3 → 32
stride 2
```

得到：

``` text
[B,32,320,320]
```

经过 Stem 内部分支、concat、第二次 stride=2 后：

``` text
[B,32,160,160]
```

最后：

``` text
1×1 Conv
32 → 64
```

得到：

``` text
[B,64,160,160]
```

所以 Stem：

``` text
[B,3,640,640]
→
[B,64,160,160]
```

累计 stride：

``` text
4
```

------------------------------------------------------------------------

# 32. Stage 1 为什么仍然是 160×160？

B5 Stage 1：

``` text
downsample = False
```

所以不会先执行 stride=2 downsample。

Channel：

``` text
64 → 128
```

于是：

``` text
[B,64,160,160]
→
[B,128,160,160]
```

仍是：

``` text
stride 4
```

------------------------------------------------------------------------

# 33. Stage 2：得到第一个检测尺度

Stage 2：

``` text
downsample = True
```

先：

``` text
3×3 depthwise conv
stride=2
```

所以：

``` text
160×160
→
80×80
```

随后 HG blocks 把 feature 转换到：

``` text
512 channels
```

所以：

``` text
Stage 2 output
=
[B,512,80,80]
```

累计 stride：

``` text
8
```

这就是第一个返回的 feature。

------------------------------------------------------------------------

# 34. Stage 3：第二个检测尺度

输入：

``` text
[B,512,80,80]
```

downsample：

``` text
80×80
→
40×40
```

输出 channel：

``` text
1024
```

得到：

``` text
[B,1024,40,40]
```

累计 stride：

``` text
16
```

------------------------------------------------------------------------

# 35. Stage 4：第三个检测尺度

输入：

``` text
[B,1024,40,40]
```

downsample：

``` text
40×40
→
20×20
```

输出 channel：

``` text
2048
```

得到：

``` text
[B,2048,20,20]
```

累计 stride：

``` text
32
```

------------------------------------------------------------------------

# 36. 所以 `deim_dfine_x` Backbone 输出是什么？

对于标准：

``` text
640×640
```

输入，我们得到核心三尺度：

``` text
C3:
[B, 512, 80, 80]
stride = 8

C4:
[B,1024, 40, 40]
stride = 16

C5:
[B,2048, 20, 20]
stride = 32
```

这就是以后第 9、10 课进入：

``` text
HGNetv2
→
HybridEncoder
```

时非常关键的输入。

------------------------------------------------------------------------

# 37. 为什么 HybridEncoder 不能直接假设 channel 都一样？

Backbone 给：

``` text
512
1024
2048
```

三个不同 channel 数。

但是 Transformer/feature fusion 常希望统一 hidden dimension。

例如 X 路线会进一步把它们投影到统一的 encoder hidden dimension。

概念上：

``` text
[B, 512,80,80]
     │
     └─ projection ─→ [B,C,80,80]

[B,1024,40,40]
     │
     └─ projection ─→ [B,C,40,40]

[B,2048,20,20]
     │
     └─ projection ─→ [B,C,20,20]
```

之后：

``` text
channel 统一
spatial resolution 不同
```

这样才更方便进行跨尺度融合。

第 10 课会正式拆 `HybridEncoder`。

------------------------------------------------------------------------

# 38. HGBlock 在做什么？先建立直觉

HGNetv2 的 `HG_Block` 不是简单：

``` text
Conv → Conv → Conv
```

它会保留中间输出。

概念上：

``` text
input x
  │
  ├──────────────┐
  ▼              │
layer 1          │
  │              │
  ▼              │
layer 2          │
  │              │
  ▼              │
layer 3 ...      │
  │              │
  ▼              │
             collect all
                  │
                  ▼
                 cat
                  │
                  ▼
             aggregation
                  │
                  ▼
                output
```

官方逻辑可以概括为：

``` python
output = [x]

for layer in self.layers:
    x = layer(x)
    output.append(x)

x = torch.cat(output, dim=1)
x = self.aggregation(x)
```

这与第 2 课学过的：

``` python
torch.cat(...)
```

直接连接起来了。

------------------------------------------------------------------------

# 39. 手算 HGBlock 的 concat channel

假设：

``` text
input channels = 128
mid channels = 64
layer_num = 6
```

`output` 中包含：

``` text
原 input:
128 channels

6 个 layer output:
每个 64 channels
```

所以 concat：

\[ C\_{total} = 128+6`\times64`{=tex} \]

# \[

# 128+384

512 \]

这正对应官方实现中的思想：

``` python
total_chs = in_chs + layer_num * mid_chs
```

然后 aggregation 再把：

``` text
512
```

投影到目标 `out_chs`。

你现在已经能从第 2 课的 `cat` 理解 HGBlock 的 channel aggregation。

------------------------------------------------------------------------

# 40. ESE/SE 类模块为什么存在？

HGNetv2 中有 channel attention 风格的聚合模块。

其直觉：

先对空间：

``` text
H×W
```

做全局平均：

``` text
[B,C,H,W]
→
[B,C,1,1]
```

公式：

\[ z_c = `\frac{1}{HW}`{=tex} `\sum`{=tex}*{i=1}\^{H}
`\sum`{=tex}*{j=1}\^{W} x\_{c,i,j} \]

得到每个 channel 的全局描述。

然后生成 channel 权重：

``` text
[B,C,1,1]
```

再广播乘回：

``` text
[B,C,H,W]
```

这与第 2 课的 broadcasting 又连接起来。

------------------------------------------------------------------------

# 41. 参数量和计算量不是同一个概念

假设同一个：

``` text
3×3 Conv
Cin=256
Cout=256
```

参数量与输入 H/W 无关：

\[ Params = 256`\times256`{=tex}`\times9`{=tex} \]

但计算量与输出空间尺寸有关。

每个输出位置都要执行卷积。

粗略 MACs：

\[ MACs = H\_{out}W\_{out} C\_{out} `\frac{C_{in}}{groups}`{=tex} K_hK_w
\]

所以：

``` text
80×80
```

上的同一个卷积，

比：

``` text
20×20
```

昂贵得多。

------------------------------------------------------------------------

# 42. 手算一个 Conv 的 MACs

假设：

``` text
Input/Output spatial = 80×80
Cin = 256
Cout = 256
K = 3
groups = 1
```

则：

\[ MACs = 80`\times80`{=tex} `\times256`{=tex} `\times256`{=tex}
`\times9`{=tex} \]

约：

\[ 3.77`\times10`{=tex}\^9 \]

也就是约：

``` text
3.77 G MACs
```

如果同一卷积放到：

``` text
20×20
```

则：

\[ 20`\times20`{=tex} `\times256`{=tex} `\times256`{=tex}
`\times9`{=tex} \]

约：

``` text
0.236 G MACs
```

因为空间面积：

\[ 80^2/20^2=16 \]

相差 16 倍。

------------------------------------------------------------------------

# 43. 为什么网络越早的高分辨率层特别贵？

因为前面 feature map 很大。

例如：

``` text
320×320
160×160
80×80
```

即使 channel 不特别大，

空间位置数已经非常多。

所以网络设计常常：

``` text
早期：
channel 较少

后期：
空间变小后
channel 再增加
```

这是一种计算资源分配。

------------------------------------------------------------------------

# 44. FLOPs 与 MACs 的关系

不同工具定义略有差异。

常见：

``` text
1 MAC
=
1 multiplication + 1 addition
```

有些统计把：

``` text
1 MAC ≈ 2 FLOPs
```

但也有工具直接把 MAC 当作一次操作统计。

因此比较论文数字时必须确认：

> 它报告的是 FLOPs、MACs，还是具体工具的计算定义？

不要机械比较数字。

------------------------------------------------------------------------

# 45. 什么叫 Receptive Field（感受野）？

假设第一层 3×3 Conv。

一个输出位置直接观察：

``` text
3×3
```

输入区域。

第二层再 3×3。

它的一个输出位置会依赖第一层的 3×3
邻域，而这些第一层位置又各自依赖输入。

所以第二层相对于原始输入的感受野会扩大。

直觉：

``` text
网络越深
→ 一个 feature location 能“看到”的原图范围通常越大
```

这对目标检测很重要：

``` text
小局部
→ 边缘/纹理

更大上下文
→ 物体结构/场景关系
```

------------------------------------------------------------------------

# 46. 感受野的递推公式

可以用两个量：

``` text
r = receptive field size
j = jump / effective stride
```

初始化：

``` text
r0 = 1
j0 = 1
```

每一层：

\[ j_l = j\_{l-1}S_l \]

\[ r_l = r\_{l-1} + (K\_{eff,l}-1)j\_{l-1} \]

其中：

\[ K\_{eff} = D(K-1)+1 \]

这可以精确追踪理论感受野。

------------------------------------------------------------------------

# 47. 感受野手算

两层：

``` text
Conv3×3 stride=2
Conv3×3 stride=2
```

初始：

``` text
r0=1
j0=1
```

第一层：

\[ j_1=1`\times2`{=tex}=2 \]

\[ r_1=1+(3-1)`\times1`{=tex}=3 \]

第二层：

\[ j_2=2`\times2`{=tex}=4 \]

\[ r_2=3+(3-1)`\times2`{=tex} \]

\[ =7 \]

所以第二层一个 feature position 理论上对应原图：

``` text
7×7
```

感受野，

而位置之间的有效步长：

``` text
4
```

------------------------------------------------------------------------

# 48. Effective Receptive Field 与理论感受野

理论公式告诉我们：

``` text
哪些输入像素有可能影响这个 feature
```

但真实训练后，各位置影响权重通常并不相同。

所以还存在：

``` text
effective receptive field
```

概念。

本课程中：

-   shape 推导主要使用理论 stride；
-   模型语义理解时记住真实有效感受野更加复杂。

------------------------------------------------------------------------

# 49. 为什么检测 Backbone 不直接一路压到 1×1？

分类网络最终可以：

``` text
feature map
→ global average pooling
→ class
```

因为分类只需要回答：

> 图片里是什么？

目标检测还必须回答：

> 它在哪里？

如果过早把空间压成：

``` text
1×1
```

会丢失大量定位信息。

因此检测器需要保留：

``` text
80×80
40×40
20×20
```

这样的空间结构。

------------------------------------------------------------------------

# 50. CNN 与 ViT 在 DEIMv2 中的角色对照

这一课主要学习 CNN，是因为 `deim_dfine_x` 用 HGNetv2。

但 `deimv2_dinov3_x` 用的是 DINOv3 ViT。

两者产生特征的方式不同。

CNN：

``` text
Image
↓
local convolution
↓
hierarchical downsampling
↓
天然形成多尺度 feature maps
```

ViT：

``` text
Image
↓
patch embedding
↓
tokens
↓
Transformer blocks
↓
强语义 token features
```

DINOv3 的检测适配需要解决一个重要问题：

> 如何把 ViT 特征变成检测器需要的多尺度空间特征？

这就是后面 STA 的重要动机之一。

所以本课 CNN 多尺度概念，也是在给第 20 课 STA 铺路。

------------------------------------------------------------------------

# 51. 小型 PyTorch 实验 1：验证卷积输出公式

``` python
import torch
import torch.nn as nn

x = torch.randn(
    2, 3, 640, 640
)

conv = nn.Conv2d(
    in_channels=3,
    out_channels=32,
    kernel_size=3,
    stride=2,
    padding=1
)

y = conv(x)

print("input :", x.shape)
print("output:", y.shape)
```

运行前先算：

\[ O = `\left`{=tex}`\lfloor`{=tex} `\frac{640+2-3}{2}`{=tex}+1
`\right`{=tex}`\rfloor`{=tex} = 320 \]

所以预期：

``` text
input:
[2,3,640,640]

output:
[2,32,320,320]
```

------------------------------------------------------------------------

# 52. 小型实验 2：写一个 Conv Shape Calculator

自己实现：

``` python
import math

def conv_out_size(
    input_size,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1
):
    return math.floor(
        (
            input_size
            + 2 * padding
            - dilation * (kernel_size - 1)
            - 1
        ) / stride
        + 1
    )
```

测试：

``` python
print(
    conv_out_size(
        640,
        kernel_size=3,
        stride=2,
        padding=1
    )
)
```

应该：

``` text
320
```

以后读任何 Conv，都先手算再用 PyTorch 验证。

------------------------------------------------------------------------

# 53. 小型实验 3：模拟检测 Backbone

``` python
import torch
import torch.nn as nn

class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.s1 = nn.Conv2d(
            3, 32,
            3, 2, 1
        )

        self.s2 = nn.Conv2d(
            32, 64,
            3, 2, 1
        )

        self.s3 = nn.Conv2d(
            64, 128,
            3, 2, 1
        )

        self.s4 = nn.Conv2d(
            128, 256,
            3, 2, 1
        )

        self.s5 = nn.Conv2d(
            256, 512,
            3, 2, 1
        )

    def forward(self, x):
        x = self.s1(x)
        print("stride 2 :", x.shape)

        x = self.s2(x)
        print("stride 4 :", x.shape)

        c3 = self.s3(x)
        print("stride 8 :", c3.shape)

        c4 = self.s4(c3)
        print("stride 16:", c4.shape)

        c5 = self.s5(c4)
        print("stride 32:", c5.shape)

        return [c3, c4, c5]


model = TinyBackbone()

x = torch.randn(
    2, 3, 640, 640
)

features = model(x)
```

预期：

``` text
stride 2:
[2,32,320,320]

stride 4:
[2,64,160,160]

stride 8:
[2,128,80,80]

stride 16:
[2,256,40,40]

stride 32:
[2,512,20,20]
```

这不是 HGNetv2。

但它精确展示了：

> 多尺度 Backbone 最核心的空间 shape 原理。

------------------------------------------------------------------------

# 54. 小型实验 4：普通 Conv vs Depthwise Conv 参数量

``` python
import torch.nn as nn

normal = nn.Conv2d(
    256,
    256,
    kernel_size=3,
    padding=1,
    groups=1,
    bias=False
)

depthwise = nn.Conv2d(
    256,
    256,
    kernel_size=3,
    padding=1,
    groups=256,
    bias=False
)

def params(m):
    return sum(
        p.numel()
        for p in m.parameters()
    )

print(
    "normal:",
    params(normal)
)

print(
    "depthwise:",
    params(depthwise)
)
```

预期：

``` text
normal:
589824

depthwise:
2304
```

比例：

\[ 589824/2304=256 \]

所以这个例子里 depthwise 的卷积权重只有普通卷积的：

\[ 1/256 \]

------------------------------------------------------------------------

# 55. 小型实验 5：模拟 HGBlock 的 Feature Aggregation

``` python
import torch

B = 2
H = 80
W = 80

x0 = torch.randn(
    B, 128, H, W
)

outputs = [x0]

for _ in range(6):
    x = torch.randn(
        B, 64, H, W
    )
    outputs.append(x)

y = torch.cat(
    outputs,
    dim=1
)

print(y.shape)
```

手算：

\[ 128+6`\times64`{=tex} = 512 \]

所以：

``` text
[2,512,80,80]
```

这就是 HGBlock 中：

``` text
feature aggregation
```

的 shape 直觉。

------------------------------------------------------------------------

# 56. 小型实验 6：模拟 ESE 的 Broadcasting

``` python
import torch

x = torch.randn(
    2, 512, 80, 80
)

channel_summary = x.mean(
    dim=(2, 3),
    keepdim=True
)

print(
    channel_summary.shape
)

weights = torch.sigmoid(
    channel_summary
)

y = x * weights

print(y.shape)
```

shape：

``` text
x:
[2,512,80,80]

channel_summary:
[2,512,1,1]

weights:
[2,512,1,1]

broadcast multiply:
[2,512,80,80]
```

这直接复习了第 2 课的 broadcasting。

------------------------------------------------------------------------

# 57. 小型实验 7：自动打印每个 Stage Shape

以后真正加载 HGNetv2 时，可以使用 forward hook。

通用写法：

``` python
def hook_fn(name):
    def hook(module, inputs, output):
        if isinstance(output, torch.Tensor):
            print(
                name,
                tuple(output.shape)
            )
    return hook
```

然后：

``` python
handle = some_module.register_forward_hook(
    hook_fn("stage")
)
```

forward 时自动打印。

后续第 9 课我们会升级成完整：

``` text
module name
input shape
output shape
parameter count
stride
requires_grad
```

的 Backbone tracer。

------------------------------------------------------------------------

# 58. 一个非常重要的工程问题：640 不一定永远是 640

训练时可能存在：

``` text
multi-scale training
random resize
```

所以模型不能把：

``` text
80
40
20
```

全部硬编码。

更通用的关系是：

``` text
H8  = H / 8
W8  = W / 8

H16 = H / 16
W16 = W / 16

H32 = H / 32
W32 = W / 32
```

例如输入：

``` text
800×800
```

则：

``` text
100×100
50×50
25×25
```

这也是为什么源码经常动态读取：

``` python
h, w = x.shape[-2:]
```

而不是直接写死数字。

------------------------------------------------------------------------

# 59. 如果输入不是 stride 的整数倍怎么办？

例如：

``` text
641×641
```

经过 stride=2 Conv，

输出尺寸需要严格使用：

\[ `\left`{=tex}`\lfloor`{=tex} `\frac{I+2P-D(K-1)-1}{S}`{=tex}+1
`\right`{=tex}`\rfloor`{=tex} \]

不能简单说：

``` text
641 / 2
```

因为涉及：

``` text
floor
padding
kernel
```

所以真正读源码时：

> 整齐的 640 是教学方便，实际 shape 必须以公式和代码为准。

------------------------------------------------------------------------

# 60. `padding="same"` 为什么有时很麻烦？

所谓 same padding 通常希望：

``` text
stride=1
```

时保持：

``` text
Hout = Hin
Wout = Win
```

但：

``` text
偶数 kernel
stride > 1
不同 dilation
```

时 padding 左右/上下可能不对称。

HGNetv2 Stem 中就存在显式：

``` python
F.pad(...)
```

以及：

``` text
kernel_size=2
```

等组合。

所以第 9 课逐行拆 Stem 时，我们会严格算每一次：

``` text
pad
conv
pool
cat
```

而不是粗暴说"Stem 降采样 4 倍"。

------------------------------------------------------------------------

# 61. 为什么本课暂时没有逐像素手算整个 HGNetv2？

因为 HGNetv2-B5 包含大量：

``` text
HG_Block
LightConvBNAct
aggregation
ESE
residual
multiple blocks
```

如果第 3 课直接完整展开，会把：

``` text
CNN 基础
```

和：

``` text
HGNetv2 专属架构
```

混在一起。

课程设计是：

``` text
第 3 课
先掌握所有 CNN shape 原理

        ↓

第 9 课
用这些原理完整逐行拆 HGNetv2-B5
```

届时你不是"听我说 shape"，而是能自己推导。

------------------------------------------------------------------------

# 62. 本课的 DEIMv2 结构地图

现在我们已经能理解：

``` text
deim_dfine_x

Image
[B,3,640,640]
       │
       ▼
HGNetv2-B5 Stem
       │
       ▼
[B,64,160,160]
stride 4
       │
       ▼
Stage 1
[B,128,160,160]
       │
       ▼
Stage 2
[B,512,80,80]
stride 8
       │
       ├─────────────── C3
       ▼
Stage 3
[B,1024,40,40]
stride 16
       │
       ├─────────────── C4
       ▼
Stage 4
[B,2048,20,20]
stride 32
       │
       └─────────────── C5

C3/C4/C5
       │
       ▼
HybridEncoder
```

这就是本课最终需要真正记住的 DEIM 连接。

------------------------------------------------------------------------

# 63. 再连接第 2 课：这些 Feature Map 以后怎样变成 Tokens？

Backbone：

``` text
C3
[B,512,80,80]

C4
[B,1024,40,40]

C5
[B,2048,20,20]
```

HybridEncoder 会进行 channel projection 和融合。

假设最后统一到：

``` text
C
```

那么可以得到：

``` text
[B,C,80,80]
[B,C,40,40]
[B,C,20,20]
```

第 2 课已经知道：

``` text
flatten(2)
+
transpose(1,2)
```

得到：

``` text
[B,6400,C]
[B,1600,C]
[B, 400,C]
```

cat：

``` text
[B,8400,C]
```

于是现在：

``` text
CNN 世界
```

和：

``` text
Transformer 世界
```

已经接上了。

------------------------------------------------------------------------

# 64. 本课必须真正理解的 12 个概念

``` text
Convolution
Kernel
Stride
Padding
Dilation
Channel
Groups
Depthwise Conv
Pointwise 1×1 Conv
Downsampling
Multi-scale Feature
Receptive Field
```

不要只背定义。

每个词至少能回答：

``` text
它改变 H/W 吗？
它改变 channel 吗？
它影响参数量吗？
它为什么对检测有用？
```

------------------------------------------------------------------------

# 65. 本课自测

请尽量不运行代码。

## Q1

输入：

``` text
[B,3,640,640]
```

经过：

``` text
Conv3×3
Cout=32
stride=2
padding=1
```

输出 shape？

------------------------------------------------------------------------

## Q2

为什么：

``` text
stride=2
```

通常会把 H/W 减半？

------------------------------------------------------------------------

## Q3

连续三个：

``` text
stride=2
```

的层，相对于原图的累计 stride 是多少？

------------------------------------------------------------------------

## Q4

输入 640，stride 8/16/32 的 feature map 分别是多少？

------------------------------------------------------------------------

## Q5

普通：

``` text
Conv3×3
Cin=256
Cout=256
```

无 bias 的参数量是多少？

------------------------------------------------------------------------

## Q6

同样 channel 的 depthwise 3×3 参数量是多少？

------------------------------------------------------------------------

## Q7

`groups=Cin` 在 `Cin=Cout` 的典型情况下意味着什么？

------------------------------------------------------------------------

## Q8

为什么 1×1 Conv 虽然不看邻域，仍然非常有用？

------------------------------------------------------------------------

## Q9

HGNetv2-B5：

``` text
Stage2
Stage3
Stage4
```

在 640 输入下分别输出什么 spatial size？

------------------------------------------------------------------------

## Q10

对应 channel 分别是多少？

------------------------------------------------------------------------

## Q11

为什么目标检测不希望 Backbone 最终只留下 1×1 feature？

------------------------------------------------------------------------

## Q12

为什么：

``` text
80×80
```

上的同一个 Conv 通常比：

``` text
20×20
```

上的贵很多？

------------------------------------------------------------------------

# 66. 课后作业 A：完整手算 Backbone

设计：

``` text
Input
[B,3,640,640]

Conv1:
3→32
K=3 S=2 P=1

Conv2:
32→64
K=3 S=2 P=1

Conv3:
64→128
K=3 S=2 P=1

Conv4:
128→256
K=3 S=2 P=1

Conv5:
256→512
K=3 S=2 P=1
```

要求对每层写：

``` text
Input shape
Output shape
Cumulative stride
Parameter count
```

然后用 PyTorch 验证。

------------------------------------------------------------------------

# 67. 课后作业 B：普通卷积 vs Depthwise + Pointwise

比较两种结构。

方案 A：

``` text
Conv3×3
256 → 256
```

方案 B：

``` text
Depthwise 3×3
256 → 256

+

Pointwise 1×1
256 → 256
```

分别计算参数量。

方案 A：

\[ 256`\times256`{=tex}`\times9`{=tex} \]

方案 B：

\[ 256`\times9`{=tex} + 256`\times256`{=tex} \]

算出具体数字和比例。

思考：

> 为什么 MobileNet、HGNetv2 的轻量模块会喜欢类似结构？

------------------------------------------------------------------------

# 68. 课后作业 C：推导 HGNetv2-B5 的三个返回尺度

从：

``` text
Input = 640×640
Stem cumulative stride = 4
```

开始。

已知：

``` text
Stage1 downsample=False
Stage2=True
Stage3=True
Stage4=True
```

要求自己推导：

``` text
Stage1
Stage2
Stage3
Stage4
```

每一层的：

``` text
spatial size
cumulative stride
```

最后验证：

``` text
return_idx=[1,2,3]
```

为什么对应：

``` text
stride [8,16,32]
```

------------------------------------------------------------------------

# 69. 课后作业 D：感受野

假设网络：

``` text
Conv3×3 S=2
Conv3×3 S=1
Conv3×3 S=2
Conv3×3 S=1
```

从：

``` text
r0=1
j0=1
```

开始，

使用：

\[ j_l=j\_{l-1}S_l \]

\[ r_l=r\_{l-1}+(K_l-1)j\_{l-1} \]

计算每层：

``` text
jump
receptive field
```

不要跳步骤。

------------------------------------------------------------------------

# 70. 课后作业 E：把第 2、3 课连接起来

假设 HybridEncoder 已把三层 feature 投影成：

``` text
[B,384,80,80]
[B,384,40,40]
[B,384,20,20]
```

请手写 PyTorch：

``` text
① flatten spatial dims
② 转成 [B,N,C]
③ cat
```

最后必须得到：

``` text
[B,8400,384]
```

然后解释：

``` text
8400
384
```

分别代表什么。

------------------------------------------------------------------------

# 71. 本课 Cheat Sheet

## 卷积输出

\[ O= `\left`{=tex}`\lfloor`{=tex} `\frac{
I+2P-D(K-1)-1
}{S}`{=tex}+1 `\right`{=tex}`\rfloor`{=tex} \]

------------------------------------------------------------------------

## 有效 Kernel

\[ K\_{eff}=D(K-1)+1 \]

------------------------------------------------------------------------

## 普通 Conv 参数量（无 bias）

\[ Params = C\_{out} C\_{in} K_hK_w \]

------------------------------------------------------------------------

## Group Conv 参数量

\[ Params = C\_{out} `\frac{C_{in}}{groups}`{=tex} K_hK_w \]

------------------------------------------------------------------------

## Depthwise Conv（典型 Cin=Cout=groups=C）

\[ Params = CK_hK_w \]

------------------------------------------------------------------------

## Conv MACs 粗略公式

\[ MACs = H\_{out}W\_{out} C\_{out} `\frac{C_{in}}{groups}`{=tex} K_hK_w
\]

------------------------------------------------------------------------

## 累计 stride

\[ S\_{total} = `\prod`{=tex}\_l S_l \]

------------------------------------------------------------------------

## 理论感受野

\[ j_l=j\_{l-1}S_l \]

\[ r_l = r\_{l-1} + (K\_{eff,l}-1)j\_{l-1} \]

------------------------------------------------------------------------

# 72. 第 3 课结束后，你应该能看懂什么？

当以后看到：

``` python
self.downsample = ConvBNAct(
    in_chs,
    in_chs,
    kernel_size=3,
    stride=2,
    groups=in_chs,
    use_act=False,
)
```

你不应该只说：

> "这是一个卷积层。"

而应该立刻读出：

``` text
① Cin=Cout=in_chs
   → channel 数不变

② groups=in_chs
   → depthwise convolution

③ kernel=3
   → 3×3 spatial filtering

④ stride=2
   → H/W 约减半

⑤ 参数量很低
   → 因为不是普通 dense convolution

⑥ 它在 HGNetv2 Stage 开头
   → 主要负责低成本下采样
```

这就是从：

``` text
会看 Python
```

进步到：

``` text
会读神经网络源码
```

------------------------------------------------------------------------

# 73. 下一课预告

## 第 4 课：Transformer 从零------Token、Q/K/V 与 Attention

下一课将正式从：

\[ Q=XW_Q \]

\[ K=XW_K \]

\[ V=XW_V \]

推导：

\[ Attention(Q,K,V) = softmax `\left`{=tex}(
`\frac{QK^T}{\sqrt{d_k}}`{=tex} `\right`{=tex})V \]

并且会拿真实 shape：

``` text
[B,1600,256]
```

拆成：

``` text
[B,8,1600,32]
```

逐步计算：

``` text
QKᵀ
softmax
attention weights
V
multi-head merge
residual
LayerNorm
FFN
```

最后自己不用：

``` python
nn.MultiheadAttention
```

手写一个可以运行的 Multi-Head Self-Attention。

学完第 4 课，我们就具备进入 DETR 的两块最基础积木：

``` text
CNN
+
Transformer
```

------------------------------------------------------------------------

# 附录 A：本课真实源码依据

本课关于 HGNetv2 的结构说明依据官方实现中的：

``` text
ConvBNAct
LightConvBNAct
StemBlock
EseModule
HG_Block
HG_Stage
HGNetv2
```

核心源码来自 DEIM/DEIMv2 使用的 HGNetv2 实现。

官方实现明确给出了：

``` python
self._out_strides = [4, 8, 16, 32]
```

以及 B5 的 stage 配置：

``` text
Stage1: out=128,  downsample=False
Stage2: out=512,  downsample=True
Stage3: out=1024, downsample=True
Stage4: out=2048, downsample=True
```

因此对于 `640×640` 输入以及常见 `return_idx=[1,2,3]`：

``` text
Stage2 → [B,512,80,80]
Stage3 → [B,1024,40,40]
Stage4 → [B,2048,20,20]
```

这是由源码配置和 stride 关系直接推导出的 shape。

------------------------------------------------------------------------

# 附录 B：参考资料

优先使用官方/一手资料：

1.  DEIMv2 官方仓库\
    https://github.com/Intellindust-AI-Lab/DEIMv2

2.  DEIM 官方 HGNetv2 实现\
    https://github.com/Intellindust-AI-Lab/DEIM/blob/main/engine/backbone/hgnetv2.py

3.  DEIM 的 `deim_hgnetv2_x_coco.yml`\
    https://github.com/Intellindust-AI-Lab/DEIM/blob/main/configs/deim_dfine/deim_hgnetv2_x_coco.yml

4.  DEIMv2 README（当前模型 Backbone 说明）\
    https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/README.md

> 注：第 3 课的目标是 CNN 与多尺度 shape 基础，因此只抽取 HGNetv2
> 中与卷积、下采样和多尺度输出直接相关的真实结构。HGNetv2-B5 的每个
> Stem/HGBlock/Stage 将在第 9 课按真实 forward 调用顺序进一步逐行展开。

------------------------------------------------------------------------

**第 3 课结束。**

如果你现在看到：

``` text
[B,3,640,640]
→
[B,512,80,80]
[B,1024,40,40]
[B,2048,20,20]
```

已经能够解释：

``` text
为什么空间尺寸会这样变化，
为什么 channel 会变化，
stride 8/16/32 是怎样累计出来的，
为什么检测器需要保留三种尺度，
```

那么 CNN Backbone 的第一层核心逻辑已经真正建立起来了。
