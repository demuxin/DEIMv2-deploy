# DEIMv2 源码课程 · 第 12 课

# D-FINE 的精细边界框回归——FDR（Fine-grained Distribution Refinement）

> 本课承接第 11 课：我们已经知道 `DFINETransformer` 如何从 8400 个多尺度位置选出 300 个 Query，并通过 Decoder 让 Query 逐层读取图像信息。
>
> 这一课只追一个问题，而且把它追到底：
>
> **一个 Query 的 `[B, Q, D]` 特征，究竟怎样通过 D-FINE 的 FDR 变成精确的 `[B, Q, 4]` Bounding Box？**
>
> 我们会真正拆开下面这条链：
>
> ```text
> Decoder Query Feature
>        ↓
> preliminary box
>        ↓
> reference box
>        ↓
> 4 × (reg_max + 1) logits
>        ↓
> 4 × 33 probability distributions
>        ↓
> non-uniform weighting function W(n)
>        ↓
> Integral / expectation
>        ↓
> left / top / right / bottom refinement
>        ↓
> distance2bbox
>        ↓
> refined [cx, cy, w, h]
> ```
>
> 本课重点使用 `deim_dfine_x` 的实际关键配置：
>
> ```text
> hidden_dim = 384
> num_queries = 300
> num_layers = 6
> reg_max = 32
> reg_scale = 8
> up = 0.5
> ```
>
> 其中要特别注意：DEIM 的基础 D-FINE 配置里 `reg_scale=4`，但 X 配置会覆盖成 `reg_scale=8`。因此本课涉及 X 的具体数字时使用 **8**。

---

# 1. 为什么这一课非常重要？

如果只看最终输出，你会看到：

```text
pred_boxes
[B,300,4]
```

很容易误以为 D-FINE 仍然只是：

```text
Linear / MLP
384 → 4
```

然后直接回归：

```text
cx, cy, w, h
```

但 D-FINE 最核心的结构创新恰恰不是这样。

它把定位问题改写成：

> **Fine-grained Distribution Refinement**

也就是：

```text
直接预测一个坐标值
```

变成：

```text
预测一个离散概率分布
↓
用概率分布表达定位的不确定性与细粒度位置
↓
把分布积分成连续 refinement
↓
在 reference box 基础上逐层修正
```

所以：

```text
最终输出仍然只有 4 个数
```

不代表：

```text
网络内部只预测了 4 个数
```

D-FINE 的内部定位表示丰富得多。

---

# 2. 先回忆传统 DETR 的 Box Regression

一个最简单的 DETR-style bbox head 可以理解为：

```python
bbox_delta = bbox_head(query)
```

其中：

```text
query
[B,Q,D]

bbox_delta
[B,Q,4]
```

然后利用 reference point / reference box 做 iterative refinement。

抽象写成：

$$
b_{i+1}
=
\sigma
\left(
\sigma^{-1}(b_i)
+
\Delta_i
\right)
$$

其中：

```text
b_i
=
当前 reference box

Δ_i
=
这一层直接回归出来的 4D correction
```

这种方式的问题不是“不能工作”。

事实上很多 DETR 都这样工作得很好。

D-FINE 提出的问题是：

> **定位是一个高度精细、具有连续性和不确定性的任务，只让网络输出一个确定的 4D 数值，内部表示是否太粗？**

---

# 3. 从 Point Regression 到 Distribution Regression

假设我们只预测一个边界位置。

传统回归：

```text
网络输出：

2.37
```

意思是：

```text
我认为答案就是 2.37
```

Distribution Regression 则可能输出：

```text
bin 0 : 0.01
bin 1 : 0.05
bin 2 : 0.50
bin 3 : 0.38
bin 4 : 0.06
```

它表达：

```text
答案主要位于 bin 2 和 bin 3 之间
```

然后可以计算期望：

$$
\hat d
=
\sum_n p_n W(n)
$$

这里：

```text
p_n
=
第 n 个 bin 的概率

W(n)
=
第 n 个 bin 所代表的连续位置
```

这就是本课最重要的公式之一。

---

# 4. 为什么“分布”比“一个数”信息更多？

考虑两个预测。

### Prediction A

```text
[0.00, 0.02, 0.96, 0.02, 0.00]
```

非常集中。

模型基本确定：

```text
就在 bin 2 附近
```

### Prediction B

```text
[0.10, 0.20, 0.30, 0.25, 0.15]
```

期望值可能也接近 bin 2。

但是它明显更不确定。

如果最后只保留：

```text
expected coordinate
```

A 和 B 可能得到很相近的最终位置。

但在内部：

```text
A = 高置信定位
B = 模糊定位
```

这是 distribution representation 能额外表达的信息。

后面你看到：

```text
LQE
Localization Quality Estimation
```

就会理解为什么它可以利用 distribution 的形状来辅助判断定位质量。

---

# 5. D-FINE 不是直接对 `(cx,cy,w,h)` 做 4 个分布

这是一个非常重要的细节。

D-FINE 的 FDR 预测的是围绕 reference box 的四个边界方向：

```text
left
top
right
bottom
```

也就是：

```text
l, t, r, b
```

所以内部结构不是：

```text
cx distribution
cy distribution
w distribution
h distribution
```

而更接近：

```text
left-edge refinement distribution
top-edge refinement distribution
right-edge refinement distribution
bottom-edge refinement distribution
```

最终再通过：

```text
distance2bbox(...)
```

把四个方向还原成：

```text
x1, y1, x2, y2
```

最后转换成：

```text
cx, cy, w, h
```

---

# 6. 为什么是 132 个输出？

真实配置：

```text
reg_max = 32
```

离散 bin 数：

$$
reg\_max + 1 = 33
$$

四个边界方向：

```text
left
top
right
bottom
```

所以：

$$
4\times33=132
$$

因此：

```text
Query Feature
[B,300,384]

        ↓ dec_bbox_head

pred_corners
[B,300,132]
```

reshape：

```text
[B,300,132]
→ [B,300,4,33]
```

四个方向，每个方向有 33 个 logits。

---

# 7. 132 个数到底长什么样？

对一个 Query：

```text
pred_corners[q]
shape = [132]
```

reshape 后：

```text
[
    left_logits[33],
    top_logits[33],
    right_logits[33],
    bottom_logits[33]
]
```

例如教学版 `reg_max=4`：

```text
left:
[-2.1, -0.4, 1.8, 0.9, -1.2]

top:
[...]

right:
[...]

bottom:
[...]
```

每组先经过：

```text
softmax
```

得到概率分布。

---

# 8. 真实源码：`dec_bbox_head`

在 `DFINETransformer` 初始化里，定位 head 的核心结构可以概括为：

```python
self.dec_bbox_head = nn.ModuleList(
    [
        MLP(
            hidden_dim,
            hidden_dim,
            4 * (self.reg_max + 1),
            3
        )
        ...
    ]
)
```

X 中：

```text
hidden_dim = 384
reg_max = 32
```

因此一个 bbox head 的最后输出维度：

$$
4(32+1)=132
$$

所以：

```text
384
→ MLP
→ 132
```

而不是：

```text
384
→ 4
```

这就是 D-FINE 与普通直接 box regression 最直观的代码差异之一。

---

# 9. `pred_corners` 这个名字为什么容易误解？

源码叫：

```text
pred_corners
```

但你不要把它简单理解成：

```text
[x1,y1,x2,y2]
```

它实际上是：

```text
四个边界方向的 distribution logits
```

shape：

```text
[B,Q,4*(reg_max+1)]
```

也就是 X 中：

```text
[B,Q,132]
```

直到经过：

```text
Integral
+
distance2bbox
```

以后才真正得到 box。

---

# 10. FDR 的第一层：先得到 Preliminary Box

Decoder 第一层输出 query feature 后，源码先做：

```python
pre_bboxes = sigmoid(
    pre_bbox_head(output)
    + inverse_sigmoid(ref_points_detach)
)
```

这一步非常重要。

它先用一个普通 4D regression head：

```text
pre_bbox_head
```

得到一个 preliminary box。

所以第一层不是直接：

```text
原始 Top-K reference
→ FDR
```

而是：

```text
Top-K reference
        ↓
pre_bbox_head
        ↓
preliminary refined box
        ↓
作为 FDR 的 reference
```

---

# 11. Preliminary Box 的 Shape

假设正常推理：

```text
output
[B,300,384]
```

经过：

```text
pre_bbox_head
384 → 384 → 384 → 4
```

得到：

```text
[B,300,4]
```

再和：

```text
inverse_sigmoid(ref_points_detach)
```

相加：

```text
[B,300,4]
+
[B,300,4]
```

最终：

```text
pre_bboxes
[B,300,4]
```

---

# 12. 为什么还需要 Preliminary Box？

可以把整个定位任务分成：

```text
第一步：
先得到一个不错的粗框

第二步：
围绕这个粗框做精细边界分布修正
```

这比让一个 distribution 从非常差的 reference 开始承担全部定位工作更合理。

直觉上：

```text
粗定位：
“车大概在这里”

FDR：
“左边界再向左一点，
 上边界再向下一点，
 右边界再向右一点……”
```

---

# 13. 第一层的关键 Reference

源码中：

```python
ref_points_initial = pre_bboxes.detach()
```

所以 FDR 使用：

```text
ref_points_initial
```

作为当前 reference box。

注意：

```text
detach()
```

不改变：

```text
数值
shape
```

但切断这条梯度路径。

第 11 课已经讲过 detach 的意义。

---

# 14. FDR 真正开始：预测 Distribution Residual

源码核心思想：

```python
pred_corners =
    bbox_head[i](output + output_detach)
    + pred_corners_undetach
```

第一层时：

```text
output_detach = 0
pred_corners_undetach = 0
```

因此大致就是：

```text
pred_corners
=
bbox_head[0](output)
```

shape：

```text
[B,Q,132]
```

后续层则会把：

```text
上一层 distribution logits
```

作为 residual accumulation 的基础。

这正是：

> **Distribution Refinement**

而不只是每层重新独立预测一个分布。

---

# 15. “Refinement”究竟 refine 什么？

普通 iterative box refinement：

```text
Layer i:
box_i
↓
预测 4D delta
↓
box_(i+1)
```

FDR：

```text
Layer i:
distribution_i
↓
预测 distribution residual
↓
distribution_(i+1)
↓
Integral
↓
refined box
```

所以 Decoder 层间不仅有：

```text
Box Refinement
```

还有更细粒度的：

```text
Distribution Refinement
```

---

# 16. 为什么 Distribution Logits 可以直接相加？

设上一层：

$$
z_i
$$

是 distribution logits。

当前层预测：

$$
\Delta z_i
$$

则：

$$
z_{i+1}
=
z_i+\Delta z_i
$$

再：

$$
p_{i+1}
=
softmax(z_{i+1})
$$

所以更深层不必：

```text
从零重新学习完整 distribution
```

而可以学习：

```text
怎样修正上一层的 localization distribution
```

这就是 residual learning 的思想。

---

# 17. 现在进入 `Integral`

真实源码的核心逻辑非常短：

```python
shape = x.shape

x = softmax(
    x.reshape(-1, reg_max + 1),
    dim=1
)

x = linear(x, project).reshape(-1, 4)

return x.reshape(
    list(shape[:-1]) + [-1]
)
```

但这几行非常关键。

我们逐行拆。

---

# 18. `Integral` 第一步：reshape

输入：

```text
x
[B,Q,132]
```

因为：

```text
132 = 4 × 33
```

reshape：

```text
[B,Q,132]
→ [-1,33]
```

如果：

```text
B=2
Q=300
```

那么：

$$
2\times300\times4=2400
$$

所以：

```text
[2,300,132]
→ [2400,33]
```

也就是说：

```text
2400 个独立的边界方向分布
```

每个分布有：

```text
33 logits
```

---

# 19. `Integral` 第二步：Softmax

对每一行：

```text
[33 logits]
```

做：

$$
p_n
=
\frac{e^{z_n}}
{\sum_j e^{z_j}}
$$

于是：

```text
[2400,33]
```

仍然是相同 shape，但每行满足：

$$
\sum_{n=0}^{32}p_n=1
$$

现在才是真正的 probability distribution。

---

# 20. `Integral` 第三步：Project

如果是传统均匀 Distribution Focal Regression，可能使用：

```text
project =
[0,1,2,...,32]
```

然后：

$$
d
=
\sum_{n=0}^{32}p_n n
$$

但 D-FINE 不是简单这样做。

它使用：

> **Non-uniform Weighting Function**

也就是：

```text
project = W(n)
```

所以：

$$
d
=
\sum_{n=0}^{32}
p_n W(n)
$$

这是理解 D-FINE FDR 的核心。

---

# 21. 为什么不用均匀 Bin？

如果：

```text
W(n) = n
```

相邻 bins 的间隔永远相同。

例如：

```text
0
1
2
3
4
5
...
```

这意味着：

```text
靠近当前 reference 的微小修正
```

和：

```text
离 reference 很远的大幅修正
```

使用同样的离散分辨率。

但检测中通常更希望：

```text
reference 已经不错时
→ 中心附近需要非常细的分辨率

reference 很差时
→ 又需要保留较大的可修正范围
```

这两个要求是冲突的。

D-FINE 的 non-uniform weighting 就是为了同时兼顾：

```text
near zero → fine-grained
far away  → large range
```

---

# 22. `weighting_function()` 的真实公式结构

源码首先计算：

$$
upper_1
=
|up|\cdot|reg\_scale|
$$

$$
upper_2
=
2|up|\cdot|reg\_scale|
$$

然后：

$$
step
=
(upper_1+1)^{\frac{2}{reg\_max-2}}
$$

中心：

```text
W(mid) = 0
```

左右大部分 bin 使用指数形式生成：

$$
-(step)^i+1
$$

和：

$$
(step)^i-1
$$

最外侧两个 endpoint 则放到：

$$
-upper_2,\quad +upper_2
$$

因此得到一个：

```text
关于 0 对称
中心密
两端稀
```

的离散坐标轴。

---

# 23. `deim_dfine_x` 的真实 W(n)

X：

```text
reg_max = 32
reg_scale = 8
up = 0.5
```

因此：

$$
upper_1=0.5\times8=4
$$

$$
upper_2=8
$$

$$
step
=
(4+1)^{2/30}
\approx1.11326
$$

33 个 W(n) 约为：

```text
[
 -8.0000,
 -4.0000,
 -3.4913,
 -3.0344,
 -2.6239,
 -2.2552,
 -1.9240,
 -1.6265,
 -1.3593,
 -1.1193,
 -0.9037,
 -0.7100,
 -0.5360,
 -0.3797,
 -0.2394,
 -0.1133,
  0.0000,
  0.1133,
  0.2394,
  0.3797,
  0.5360,
  0.7100,
  0.9037,
  1.1193,
  1.3593,
  1.6265,
  1.9240,
  2.2552,
  2.6239,
  3.0344,
  3.4913,
  4.0000,
  8.0000
]
```

请认真观察中心附近：

```text
-0.2394
-0.1133
 0
 0.1133
 0.2394
```

非常密。

而边缘：

```text
-8
-4
...
4
8
```

跨度非常大。

这就是：

```text
fine near center
+
large correction range at tails
```

---

# 24. 为什么中心 W=0 特别重要？

后面 `distance2bbox()` 的公式设计使得：

```text
distance = [0,0,0,0]
```

恰好表示：

> **不要改变 reference box。**

这是非常漂亮的设计。

因此 distribution 如果集中在：

```text
W=0
```

附近，就意味着：

```text
当前 reference 已经不错，
只需要很小的修正。
```

---

# 25. `Integral` 本质上就是期望

假设一个方向只有 5 个教学 bins：

```text
W =
[-2,-1,0,1,2]
```

概率：

```text
P =
[0.05,0.10,0.60,0.20,0.05]
```

那么：

$$
d
=
0.05(-2)
+0.10(-1)
+0.60(0)
+0.20(1)
+0.05(2)
$$

$$
d
=
-0.1-0.1+0+0.2+0.1
=
0.1
$$

虽然 bin 是离散的，但期望：

```text
0.1
```

是连续值。

所以：

> **离散分布预测并不会把最终 box 限制在离散坐标上。**

---

# 26. 为什么这仍然可微？

因为：

```text
logits
↓
softmax
↓
weighted sum
↓
distance
↓
bbox
↓
loss
```

整个链都是可微的。

梯度可以从：

```text
bbox loss
```

一路回传到：

```text
33 个 distribution logits
```

因此网络可以学习：

```text
应该把概率质量移动到哪些 bins。
```

---

# 27. `F.linear(x, project)` 为什么能做积分？

假设：

```text
x
[N,33]
```

每一行是 probability：

```text
p0,p1,...,p32
```

而：

```text
project
[33]
```

是：

```text
W0,W1,...,W32
```

那么线性运算本质上得到：

$$
p_0W_0+p_1W_1+\cdots+p_{32}W_{32}
$$

也就是：

$$
E[W]
$$

所以源码虽然只是一行：

```python
F.linear(x, project)
```

数学意义却是：

> **计算 localization distribution 在非均匀坐标轴上的期望。**

---

# 28. Integral 后的 Shape

输入：

```text
pred_corners
[B,Q,132]
```

reshape + softmax：

```text
[B,Q,4,33]
```

逻辑上做四次期望：

```text
left  → 1 scalar
top   → 1 scalar
right → 1 scalar
bottom→ 1 scalar
```

于是：

```text
integral(...)
[B,Q,4]
```

注意：

```text
这 4 个数还不是最终 cxcywh
```

而是四个：

```text
edge refinement distances
```

---

# 29. 进入 `distance2bbox()`

真实源码接下来：

```python
inter_ref_bbox = distance2bbox(
    ref_points_initial,
    integral(pred_corners, project),
    reg_scale
)
```

输入：

```text
points
[B,Q,4]
=
[x,y,w,h]

distance
[B,Q,4]
=
[l,t,r,b]

reg_scale
=
8 for X
```

输出：

```text
[B,Q,4]
=
[cx,cy,w,h]
```

---

# 30. `distance2bbox()` 的真实公式

设 reference：

$$
P=(x,y,w,h)
$$

distribution integral 得到：

$$
d=(d_l,d_t,d_r,d_b)
$$

源码计算：

$$
x_1
=
x-
\left(
\frac{reg\_scale}{2}+d_l
\right)
\frac{w}{reg\_scale}
$$

$$
y_1
=
y-
\left(
\frac{reg\_scale}{2}+d_t
\right)
\frac{h}{reg\_scale}
$$

$$
x_2
=
x+
\left(
\frac{reg\_scale}{2}+d_r
\right)
\frac{w}{reg\_scale}
$$

$$
y_2
=
y+
\left(
\frac{reg\_scale}{2}+d_b
\right)
\frac{h}{reg\_scale}
$$

然后：

```text
[x1,y1,x2,y2]
→ [cx,cy,w,h]
```

---

# 31. 把公式化简，你会突然看懂 FDR

例如左边界：

$$
x_1
=
x-
\left(
\frac{s}{2}+d_l
\right)
\frac{w}{s}
$$

其中：

$$
s=reg\_scale
$$

展开：

$$
x_1
=
x-\frac{w}{2}-d_l\frac{w}{s}
$$

而 reference box 原本左边界就是：

$$
x_{1,ref}=x-\frac{w}{2}
$$

因此：

$$
x_1
=
x_{1,ref}
-
d_l\frac{w}{s}
$$

同理：

$$
y_1
=
y_{1,ref}
-
d_t\frac{h}{s}
$$

$$
x_2
=
x_{2,ref}
+
d_r\frac{w}{s}
$$

$$
y_2
=
y_{2,ref}
+
d_b\frac{h}{s}
$$

现在 FDR 的几何意义非常清楚了：

> **它预测的是 reference box 四条边应该相对移动多少。**

---

# 32. 为什么 `distance=0` 会保持 Reference Box 不变？

令：

$$
d_l=d_t=d_r=d_b=0
$$

则：

$$
x_1=x-\frac{w}{2}
$$

$$
y_1=y-\frac{h}{2}
$$

$$
x_2=x+\frac{w}{2}
$$

$$
y_2=y+\frac{h}{2}
$$

这恰好就是 reference box 自己。

所以：

```text
distribution 集中在 W=0
```

意味着：

```text
identity refinement
```

这是理解整个设计的关键。

---

# 33. 正距离和负距离是什么意思？

以左边界为例：

$$
x_1
=
x_{1,ref}
-
d_l\frac{w}{s}
$$

如果：

```text
d_l > 0
```

那么：

```text
x1 变小
```

左边界向左扩张。

如果：

```text
d_l < 0
```

那么：

```text
x1 变大
```

左边界向右收缩。

右边界：

$$
x_2
=
x_{2,ref}
+
d_r\frac{w}{s}
$$

所以：

```text
d_r > 0
→ 右边界向右扩张

d_r < 0
→ 右边界向左收缩
```

---

# 34. 一个完整数字例子

假设 reference：

```text
cx = 0.50
cy = 0.50
w  = 0.20
h  = 0.40
```

所以 reference 边界：

```text
x1 = 0.40
y1 = 0.30
x2 = 0.60
y2 = 0.70
```

X：

```text
reg_scale = 8
```

假设 Integral 得到：

```text
d_l = +1.0
d_t = -0.5
d_r = +2.0
d_b = 0
```

单位：

$$
w/s=0.20/8=0.025
$$

$$
h/s=0.40/8=0.05
$$

于是：

### 左边界

$$
x_1
=
0.40-1.0(0.025)
=
0.375
$$

### 上边界

$$
y_1
=
0.30-(-0.5)(0.05)
=
0.325
$$

### 右边界

$$
x_2
=
0.60+2.0(0.025)
=
0.65
$$

### 下边界

$$
y_2
=
0.70+0(0.05)
=
0.70
$$

最终：

```text
xyxy =
[0.375, 0.325, 0.650, 0.700]
```

转换：

```text
cx = (0.375+0.650)/2 = 0.5125
cy = (0.325+0.700)/2 = 0.5125

w = 0.650-0.375 = 0.275
h = 0.700-0.325 = 0.375
```

最终：

```text
[cx,cy,w,h]
=
[0.5125,0.5125,0.275,0.375]
```

---

# 35. 为什么距离要按 Reference Width/Height 缩放？

注意：

$$
\Delta x
=
d\frac{w}{reg\_scale}
$$

不是固定像素位移。

这意味着：

```text
同一个 d
```

对：

```text
大 box
```

产生较大的绝对移动；

对：

```text
小 box
```

产生较小的绝对移动。

因此 FDR 的 refinement 是：

> **相对于当前 box 尺寸归一化的。**

这非常适合目标检测，因为不同目标尺度差异巨大。

---

# 36. `reg_scale=8` 到底在控制什么？

从：

$$
\Delta x=d\frac{w}{s}
$$

看，`s=reg_scale` 影响几何缩放。

同时它还进入：

```text
weighting_function()
```

决定 W(n) 的曲率和范围。

所以 `reg_scale` 同时影响：

```text
distribution coordinate system
+
distance-to-box decoding scale
```

这两部分是配套设计的，不能只看其中一个。

---

# 37. 为什么 X 用 `reg_scale=8`？

这里最重要的不是死记“8 比 4 好”。

而是理解：

```text
reg_scale
```

会改变：

```text
W(n) 的非均匀分布
```

以及：

```text
distance → physical edge movement
```

X 配置明确覆盖基础配置：

```yaml
DFINETransformer:
    feat_channels: [384,384,384]
    reg_scale: 8
```

所以分析 `deim_dfine_x` 时必须使用 8。

以后如果你改这个参数，不能只认为：

```text
“我只是改了一个 loss 超参数”
```

它直接改变了 bbox representation / decoding geometry。

---

# 38. FDR 的完整 Forward 数据流

对第 i 层 Decoder：

```text
output_i
[B,Q,D]

        ↓

bbox_head[i]

        ↓

distribution residual
[B,Q,132]

        +

previous distribution logits

        ↓

pred_corners_i
[B,Q,132]

        ↓ reshape

[B,Q,4,33]

        ↓ softmax

4 probability distributions

        ↓ W(n)

Integral / expectation

        ↓

distance_i
[B,Q,4]

        ↓

distance2bbox(
    reference_box_i,
    distance_i
)

        ↓

refined_box_i
[B,Q,4]
```

---

# 39. Decoder 层间 FDR 怎样连接？

源码中的关键状态：

```text
pred_corners_undetach
ref_points_detach
output_detach
```

每层结束：

```python
pred_corners_undetach = pred_corners
ref_points_detach = inter_ref_bbox.detach()
output_detach = output.detach()
```

下一层继续：

```text
新的 Query Feature
+
上一层 Query Feature 的 detached representation
```

以及：

```text
当前 distribution residual
+
上一层 distribution logits
```

这让更深层执行：

```text
refinement
```

而不是：

```text
independent re-prediction
```

---

# 40. 为什么叫 Fine-grained Distribution **Refinement**？

现在可以精确定义：

### Fine-grained

因为：

```text
一个边界不是一个 scalar
```

而是：

```text
33-bin probability distribution
```

并且 W(n) 在 0 附近具有更细的 resolution。

### Distribution

因为网络内部预测：

```text
P(left)
P(top)
P(right)
P(bottom)
```

### Refinement

因为深层 Decoder：

```text
在前一层 distribution logits 基础上学习 residual correction
```

而不是每层从零预测。

---

# 41. FDR 与普通 DFL 有什么相似和不同？

如果你学过 YOLO 系列里的 DFL（Distribution Focal Loss），会感觉很熟悉：

```text
logits
→ softmax
→ expectation
→ continuous coordinate
```

相似点确实存在。

但 D-FINE 的关键区别包括：

```text
1. 它处在 DETR iterative decoder 框架里

2. 它围绕 reference box 做 refinement

3. distribution 在 Decoder layers 间迭代修正

4. 使用 non-uniform weighting function W(n)

5. 与 GO-LSD / localization distillation 联动
```

所以不能简单说：

```text
FDR = 把 DFL 搬进 DETR
```

它是围绕 DETR Decoder 的定位机制重新组织的。

---

# 42. 为什么 W(n) 非均匀很重要？

想象只有固定数量 33 个 bins。

你希望同时覆盖：

```text
很大的修正范围
```

又希望：

```text
0 附近非常精细
```

如果均匀分布在：

```text
[-8,8]
```

间隔就是：

$$
16/32=0.5
$$

中心附近只有：

```text
-0.5
0
0.5
```

而 D-FINE X 的真实 W(n) 中心附近约：

```text
-0.2394
-0.1133
0
0.1133
0.2394
```

细得多。

同时两端仍然能到：

```text
±8
```

这就是 non-uniform discretization 的意义。

---

# 43. 一个非常重要的认识：FDR 不是“把坐标量化成 33 档”

这是初学者很容易产生的误解。

虽然内部有：

```text
33 bins
```

但最终：

$$
d=\sum p_nW(n)
$$

是连续值。

例如：

```text
W2 = 0
W3 = 0.1133
```

如果：

```text
P2 = 0.3
P3 = 0.7
```

则：

$$
d
=
0.3(0)+0.7(0.1133)
=
0.07931
$$

它并不等于任何一个 bin。

所以：

```text
discrete representation
```

最终仍能产生：

```text
continuous localization
```

---

# 44. 训练时 GT 怎么变成 Distribution Target？

Forward 解码已经懂了。

现在反过来：

```text
GT box
```

怎样监督：

```text
33-bin distribution
```

核心链：

```text
GT bbox
+
reference box
↓
bbox2distance
↓
continuous target edge refinement
↓
translate_gt
↓
找到左右两个相邻 W(n)
↓
计算 interpolation weights
↓
监督 distribution logits
```

---

# 45. `bbox2distance()` 是 `distance2bbox()` 的逆方向

Forward：

```text
reference
+
predicted distance
↓
bbox
```

Training Target：

```text
reference
+
GT bbox
↓
target distance
```

例如左边：

$$
d_l
=
\frac{x-x_{1,gt}}
{w/reg\_scale}
-
\frac{reg\_scale}{2}
$$

这正好对应 `distance2bbox()` 的逆关系。

---

# 46. 为什么还要 `translate_gt()`？

`bbox2distance()` 得到连续目标，例如：

```text
d_gt = 0.18
```

但网络输出的是：

```text
33 个 logits
```

因此要在 W(n) 中找到：

```text
左邻 bin
右邻 bin
```

例如：

```text
W_left  = 0.1133
W_right = 0.2394
```

因为：

```text
0.1133 < 0.18 < 0.2394
```

然后用线性插值决定两个 bin 的监督权重。

---

# 47. GT 的双 Bin 插值

设：

$$
a=W_k
$$

$$
b=W_{k+1}
$$

GT：

$$
g\in[a,b]
$$

则右侧权重：

$$
w_r
=
\frac{g-a}{b-a}
$$

左侧：

$$
w_l
=
1-w_r
$$

例如：

```text
a = 0.1133
b = 0.2394
g = 0.18
```

则：

$$
w_r
=
\frac{0.18-0.1133}
{0.2394-0.1133}
\approx0.529
$$

$$
w_l
\approx0.471
$$

所以 target 不是粗暴 one-hot：

```text
bin k = 1
```

而是：

```text
bin k   ≈ 0.471
bin k+1 ≈ 0.529
```

这保留了连续 GT 在两个离散 bin 之间的位置关系。

---

# 48. FGL Loss

DEIM/D-FINE Criterion 中定位分布监督叫：

```text
loss_fgl
```

也就是 Fine-Grained Localization Loss。

配置中典型权重：

```yaml
loss_fgl: 0.15
```

Criterion 的 `losses` 中包含：

```text
local
```

它会利用：

```text
pred_corners
ref_points
GT boxes
```

生成 distribution targets。

---

# 49. FGL 的核心形式

对 target 位于：

```text
left bin = k
right bin = k+1
```

时，loss 可以理解为：

$$
L_{FGL}
=
w_l CE(z,k)
+
w_r CE(z,k+1)
$$

其中：

```text
z
=
当前方向的 33-bin logits
```

所以网络被要求：

```text
把概率集中到 GT 附近的两个 bins
```

而不是只优化最终 expectation。

---

# 50. 为什么同时还要 L1 / GIoU？

你可能会问：

> 已经有 FGL，为什么还需要 `loss_bbox` 和 `loss_giou`？

因为它们监督的层次不同。

### FGL

监督：

```text
内部 localization distribution
```

### L1

监督：

```text
最终 box coordinates
```

### GIoU

监督：

```text
最终 box 的几何重叠质量
```

所以：

```text
distribution-level supervision
+
coordinate-level supervision
+
overlap-level supervision
```

可以同时存在。

---

# 51. 配置里的 Localization Loss

DEIM 基础 D-FINE 配置中：

```yaml
weight_dict:
    loss_bbox: 5
    loss_giou: 2
    loss_fgl: 0.15
    loss_ddf: 1.5
```

你现在已经理解：

```text
loss_fgl
```

在监督什么。

下一课 GO-LSD 会重点解释：

```text
loss_ddf
```

为什么存在。

---

# 52. `pred_corners` 为什么训练时必须返回？

推理只需要：

```text
pred_logits
pred_boxes
```

但训练时模型额外返回：

```text
pred_corners
ref_points
up
reg_scale
```

因为 Criterion 需要：

```text
pred_corners
```

计算 distribution-level localization loss。

所以：

```text
pred_corners
```

主要是训练内部表示；

```text
pred_boxes
```

才是最终检测框表示。

---

# 53. LQE：从 Distribution 看定位质量

源码还有：

```text
LQE
Localization Quality Estimation
```

它先：

```text
pred_corners
[B,Q,132]
```

reshape：

```text
[B,Q,4,33]
```

softmax 后，每个方向取：

```text
Top-K probabilities
```

源码：

```text
k = 4
```

然后还计算：

```text
top-k mean
```

---

# 54. LQE 的 Shape 推导

每个方向：

```text
top 4 probabilities
+
mean
```

所以：

```text
5 numbers / direction
```

四个方向：

$$
4\times5=20
$$

因此 LQE MLP 输入：

```text
20
```

然后：

```text
20
→ hidden 64
→ 1 quality score
```

最后：

```text
classification score
+
quality score
```

---

# 55. LQE 的直觉

假设两个 Query 分类都是：

```text
car score = 0.9
```

Query A 的 localization distribution：

```text
非常尖锐、集中
```

Query B：

```text
非常平、分散
```

它们的定位可信度明显不同。

LQE 试图从：

```text
distribution shape
```

提取：

```text
localization confidence
```

因此 D-FINE 不只是用 distribution 算坐标，还进一步利用 distribution 的形状信息。

---

# 56. 为什么 LQE 使用 Top-K Probability Statistics？

因为它不需要读取完整 33 bins 的绝对位置。

它更关心：

```text
分布有多集中？
峰值有多高？
前几个峰值是什么形态？
```

Top-K probabilities 可以提供一种紧凑的：

```text
distribution sharpness / confidence summary
```

---

# 57. 第一层 FDR 的完整真实逻辑

把第 11 课和本课连起来：

```text
Top-K Query
[B,Q,384]

        ↓

Decoder Layer 0

        ↓

output_0
[B,Q,384]

        ├───────────────────────────────┐
        │                               │
        ▼                               ▼
pre_bbox_head                     bbox_head[0]
        │                               │
        ▼                               ▼
4D coarse delta                  132 logits
        │                               │
        ▼                               │
pre_bboxes                       pred_corners_0
[B,Q,4]                          [B,Q,132]
        │                               │
        │                               ▼
        │                         softmax + W(n)
        │                               │
        │                               ▼
        │                           [B,Q,4]
        │                               │
        └──────── reference ────────────┤
                                        ▼
                                  distance2bbox
                                        │
                                        ▼
                                  refined box_0
                                  [B,Q,4]
```

---

# 58. 第二层以后发生什么？

Layer 1：

```text
previous refined box
↓ detach
reference_1

previous pred_corners
+
new residual distribution logits
↓
pred_corners_1

reference_1
+
Integral(pred_corners_1)
↓
refined box_1
```

继续：

```text
Layer 2
Layer 3
...
```

所以：

```text
box reference
```

和：

```text
distribution representation
```

都在逐层演化。

---

# 59. 为什么上一层 Distribution 不 detach？

源码：

```python
pred_corners_undetach = pred_corners
```

注意名字：

```text
undetach
```

说明 distribution refinement 保留了跨层梯度连接。

而：

```python
ref_points_detach = inter_ref_bbox.detach()
```

box reference 路径则 detach。

这意味着：

```text
几何 reference 用于下一层 attention / decoding 时切断梯度
```

但：

```text
distribution residual chain
```

仍可以保留跨层优化关系。

这是读源码时非常值得注意的细节。

---

# 60. `output + output_detach` 又是什么？

源码中：

```python
pred_corners =
    bbox_head[i](
        output + output_detach
    )
    + pred_corners_undetach
```

上一层结束时：

```text
output_detach
=
previous output.detach()
```

所以后续 bbox head 不只看：

```text
当前 Query Feature
```

还显式融合：

```text
上一层 detached Query Feature
```

可以把它理解为：

```text
当前层在上一层定位状态基础上继续做 refinement
```

同时通过 detach 控制梯度耦合。

---

# 61. FDR 的“两个 Residual”

你可以把它看成两个层次的 residual 思想。

### Feature-side

```text
current output
+
previous output.detach()
```

### Distribution-side

```text
current bbox distribution residual
+
previous pred_corners
```

因此 FDR 不是简单：

```text
每层独立 132-way prediction
```

而是真正设计成：

```text
progressive refinement process
```

---

# 62. PyTorch 实验 1：生成 X 的真实 W(n)

```python
import torch

def weighting_function(
    reg_max,
    up,
    reg_scale
):
    upper_bound1 = abs(up) * abs(reg_scale)
    upper_bound2 = (
        abs(up)
        * abs(reg_scale)
        * 2
    )

    step = (
        upper_bound1 + 1
    ) ** (
        2 / (reg_max - 2)
    )

    left_values = [
        -(step ** i) + 1
        for i in range(
            reg_max // 2 - 1,
            0,
            -1
        )
    ]

    right_values = [
        step ** i - 1
        for i in range(
            1,
            reg_max // 2
        )
    ]

    values = (
        [-upper_bound2]
        + left_values
        + [0.0]
        + right_values
        + [upper_bound2]
    )

    return torch.tensor(values)

W = weighting_function(
    reg_max=32,
    up=0.5,
    reg_scale=8
)

print(W)
print(W.shape)
```

预期：

```text
shape = [33]
```

重点观察：

```text
W[16] = 0
```

以及中心附近间隔明显小于两端。

---

# 63. PyTorch 实验 2：画出 Non-uniform Bin Spacing

```python
import matplotlib.pyplot as plt

x = torch.arange(len(W))

plt.figure()
plt.plot(
    x.numpy(),
    W.numpy(),
    marker="o"
)
plt.xlabel("Bin index")
plt.ylabel("W(n)")
plt.title("D-FINE Non-uniform Weighting Function")
plt.grid(True)
plt.show()
```

你应该观察：

```text
中心区域较平缓、密集
两端变化更陡
```

这就是源码注释里：

```text
reg_scale controls curvature
```

的直观含义。

---

# 64. PyTorch 实验 3：自己实现 Integral

```python
def integral(
    logits,
    project
):
    B,Q,_ = logits.shape

    prob = logits.reshape(
        B,Q,4,-1
    ).softmax(dim=-1)

    distance = (
        prob
        * project.view(1,1,1,-1)
    ).sum(dim=-1)

    return distance

B = 2
Q = 300

logits = torch.randn(
    B,Q,4*33
)

distance = integral(
    logits,
    W
)

print(distance.shape)
```

输出：

```text
[2,300,4]
```

---

# 65. PyTorch 实验 4：验证源码 `F.linear` 与手算等价

```python
import torch.nn.functional as F

prob = torch.softmax(
    torch.randn(10,33),
    dim=-1
)

manual = (
    prob * W
).sum(dim=-1)

linear = F.linear(
    prob,
    W.unsqueeze(0)
).squeeze(-1)

print(
    torch.allclose(
        manual,
        linear
    )
)
```

应该得到：

```text
True
```

这能帮助你真正理解：

```text
Integral.forward()
```

里那一行 `F.linear()`。

---

# 66. PyTorch 实验 5：实现 `distance2bbox`

```python
def distance2bbox(
    points,
    distance,
    reg_scale
):
    s = abs(reg_scale)

    x = points[...,0]
    y = points[...,1]
    w = points[...,2]
    h = points[...,3]

    dl = distance[...,0]
    dt = distance[...,1]
    dr = distance[...,2]
    db = distance[...,3]

    x1 = x - (
        0.5*s + dl
    ) * (w/s)

    y1 = y - (
        0.5*s + dt
    ) * (h/s)

    x2 = x + (
        0.5*s + dr
    ) * (w/s)

    y2 = y + (
        0.5*s + db
    ) * (h/s)

    cx = (x1+x2)/2
    cy = (y1+y2)/2
    bw = x2-x1
    bh = y2-y1

    return torch.stack(
        [cx,cy,bw,bh],
        dim=-1
    )
```

测试：

```python
ref = torch.tensor([
    [0.5,0.5,0.2,0.4]
])

d = torch.tensor([
    [1.0,-0.5,2.0,0.0]
])

box = distance2bbox(
    ref,
    d,
    reg_scale=8
)

print(box)
```

应接近：

```text
[0.5125,0.5125,0.2750,0.3750]
```

---

# 67. PyTorch 实验 6：验证 Zero Distance = Identity

```python
ref = torch.rand(
    100,4
)

# 保证 w/h 合理
ref[:,2:] = (
    ref[:,2:] * 0.5
    + 0.05
)

zero = torch.zeros_like(ref)

decoded = distance2bbox(
    ref,
    zero,
    reg_scale=8
)

print(
    torch.max(
        torch.abs(
            decoded-ref
        )
    )
)
```

理论上只应有浮点误差。

这个实验非常重要，因为它揭示：

```text
W=0
```

在 FDR 中的几何意义。

---

# 68. PyTorch 实验 7：完整模拟 `[384 → 132 → 4 → bbox]`

```python
import torch.nn as nn

B = 2
Q = 300
D = 384

query = torch.randn(
    B,Q,D
)

bbox_head = nn.Sequential(
    nn.Linear(D,D),
    nn.SiLU(),
    nn.Linear(D,D),
    nn.SiLU(),
    nn.Linear(D,132)
)

pred_corners = bbox_head(query)

print(
    "pred_corners:",
    pred_corners.shape
)

distance = integral(
    pred_corners,
    W
)

print(
    "distance:",
    distance.shape
)

reference = torch.rand(
    B,Q,4
)

reference[...,2:] = (
    reference[...,2:] * 0.4
    + 0.05
)

pred_box = distance2bbox(
    reference,
    distance,
    8
)

print(
    "pred_box:",
    pred_box.shape
)
```

输出：

```text
pred_corners [2,300,132]
distance     [2,300,4]
pred_box     [2,300,4]
```

---

# 69. PyTorch 实验 8：模拟 Distribution Refinement

```python
B = 1
Q = 3

pred_corners = torch.zeros(
    B,Q,132
)

for layer in range(6):

    residual = (
        torch.randn(
            B,Q,132
        ) * 0.2
    )

    pred_corners = (
        pred_corners
        + residual
    )

    distance = integral(
        pred_corners,
        W
    )

    print(
        "layer",
        layer,
        distance[0,0]
    )
```

真实模型中：

```text
residual
```

当然来自：

```text
bbox_head[i](query features)
```

但这个实验能让你看到：

```text
logit residual accumulation
→ distribution 改变
→ expectation 改变
→ box refinement 改变
```

---

# 70. PyTorch 实验 9：比较“尖锐分布”和“平坦分布”

```python
sharp = torch.full(
    (33,),
    -5.0
)

sharp[16] = 5.0

flat = torch.zeros(33)

p_sharp = sharp.softmax(-1)
p_flat = flat.softmax(-1)

d_sharp = (
    p_sharp * W
).sum()

d_flat = (
    p_flat * W
).sum()

print(
    "sharp expectation:",
    d_sharp.item()
)

print(
    "flat expectation:",
    d_flat.item()
)

print(
    "sharp max prob:",
    p_sharp.max().item()
)

print(
    "flat max prob:",
    p_flat.max().item()
)
```

由于 W 对称，两者 expectation 都可能接近：

```text
0
```

但：

```text
distribution confidence
```

完全不同。

这正是为什么：

```text
只看最终坐标
```

会丢失信息，而 LQE 可以利用分布形状。

---

# 71. PyTorch 实验 10：GT 双 Bin 插值

```python
def translate_one_gt(
    gt,
    W
):
    diffs = W - gt

    mask = diffs <= 0

    left_idx = (
        mask.sum() - 1
    ).item()

    left_idx = max(
        0,
        min(
            left_idx,
            len(W)-2
        )
    )

    left = W[left_idx]
    right = W[left_idx+1]

    wr = (
        torch.abs(gt-left)
        /
        (
            torch.abs(gt-left)
            + torch.abs(right-gt)
        )
    )

    wl = 1-wr

    return (
        left_idx,
        left,
        right,
        wl,
        wr
    )

result = translate_one_gt(
    torch.tensor(0.18),
    W
)

print(result)
```

你应该找到 0.18 位于：

```text
0.1133
和
0.2394
```

之间。

---

# 72. 一个小型 FGL Loss

```python
def toy_fgl_loss(
    logits,
    left_idx,
    weight_left,
    weight_right
):
    logits = logits.unsqueeze(0)

    left_target = torch.tensor(
        [left_idx]
    )

    right_target = torch.tensor(
        [left_idx+1]
    )

    loss_left = F.cross_entropy(
        logits,
        left_target
    )

    loss_right = F.cross_entropy(
        logits,
        right_target
    )

    return (
        weight_left * loss_left
        +
        weight_right * loss_right
    )
```

这就是：

```text
两个相邻 bins
+
连续插值权重
```

的最小训练版本。

---

# 73. Shape 全追踪：X 模型一个 Query 的定位

```text
Query feature
[384]

↓ bbox head

[132]

↓ reshape

[4,33]

↓ softmax

left  [33]
top   [33]
right [33]
bottom[33]

↓ non-uniform expectation

[4]
=
[d_l,d_t,d_r,d_b]

reference
[4]
=
[cx,cy,w,h]

↓ distance2bbox

xyxy
[4]

↓ box_xyxy_to_cxcywh

refined box
[4]
```

扩展到 batch：

```text
[B,300,384]
→ [B,300,132]
→ [B,300,4,33]
→ [B,300,4]
→ [B,300,4]
```

---

# 74. 一个非常值得记住的公式链

FDR 可以浓缩成：

$$
z
\in
\mathbb{R}^{4\times33}
$$

$$
p_{k,n}
=
softmax(z_{k,:})_n
$$

$$
d_k
=
\sum_{n=0}^{32}
p_{k,n}W(n)
$$

其中：

```text
k ∈ {left, top, right, bottom}
```

然后：

$$
x_1
=
x-\frac{w}{2}
-d_l\frac{w}{s}
$$

$$
y_1
=
y-\frac{h}{2}
-d_t\frac{h}{s}
$$

$$
x_2
=
x+\frac{w}{2}
+d_r\frac{w}{s}
$$

$$
y_2
=
y+\frac{h}{2}
+d_b\frac{h}{s}
$$

最后：

$$
(cx,cy,w,h)
=
xyxy\_to\_cxcywh
(x_1,y_1,x_2,y_2)
$$

这几行就是 FDR inference 的数学核心。

---

# 75. FDR 的训练链

训练时再加一条逆变换：

```text
GT bbox
+
reference box
↓
bbox2distance
↓
continuous d_gt
↓
find neighboring W bins
↓
(left_idx, right_idx)
+
(weight_left, weight_right)
↓
FGL loss
↓
supervise 33-bin logits
```

所以：

```text
Inference:
distribution → distance → box

Training:
GT box → target distance → target distribution
```

这两条链刚好互相对应。

---

# 76. 为什么要保留 `ref_points`？

训练 output 中：

```text
ref_points
```

不是多余信息。

Criterion 要计算：

```text
GT box
```

相对于：

```text
当前 reference box
```

应该对应什么 target distances。

所以 FGL Target 不是：

```text
只由 GT 决定
```

而是：

```text
GT
+
当前 Query Reference
```

共同决定。

---

# 77. 为什么同一个 GT 对不同 Reference 有不同 Distribution Target？

假设同一个 GT：

```text
GT box = 固定
```

Query A 的 reference 已经很准：

```text
需要很小 correction
```

所以 target distribution 接近：

```text
W=0
```

Query B 的 reference 偏得较多：

```text
需要大 correction
```

所以 target distribution 会移动到：

```text
较远 bins
```

因此 FDR 是：

> **reference-conditioned localization representation**

而不是固定绝对坐标分类。

---

# 78. 为什么这种表示特别适合 Decoder Iterative Refinement？

因为 Decoder 本来就是：

```text
Layer 0:
粗 reference

Layer 1:
更好 reference

Layer 2:
更好 reference

...
```

随着 reference 越来越准：

```text
target correction
```

应该越来越靠近：

```text
0
```

而 W(n) 恰好在 0 附近提供最高的细粒度分辨率。

这两个设计天然匹配。

---

# 79. 这就是 D-FINE 的关键直觉

可以把它浓缩成：

```text
早期：
reference 不够准
→ distribution 可以使用较远 bins
→ 大范围修正

后期：
reference 已经较准
→ distribution 聚集到 0 附近
→ 利用密集 bins 做细粒度修正
```

这比固定精度的均匀 regression representation 更适合：

```text
coarse-to-fine iterative decoder
```

---

# 80. FDR 与第 11 课 Iterative Refinement 的关系

第 11 课我们简单写过：

```text
reference
→ bbox refinement
→ new reference
```

现在必须升级成更准确的理解：

```text
reference_i
        │
        │
query_i ├─→ distribution residual_i
        │
        ▼
distribution_i
        │
        ▼
Integral with W(n)
        │
        ▼
edge distances_i
        │
        ▼
distance2bbox
        │
        ▼
refined_box_i
        │
        └── detach → reference_(i+1)
```

这才是 D-FINE 的真正 localization path。

---

# 81. FDR 与 Deformable Attention 形成闭环

别忘了：

```text
reference box
```

不仅用于 box regression。

它还影响下一层：

```text
Deformable Cross-Attention
```

的采样区域。

所以：

```text
FDR
→ better box
→ better reference
→ better sampling locations
→ better image features
→ better distribution refinement
```

形成闭环。

这就是为什么定位模块不能孤立理解。

---

# 82. 一个完整的正反馈链

```text
Query Feature
      ↓
FDR
      ↓
More Accurate Box
      ↓
Next Layer Reference
      ↓
More Accurate Deformable Sampling
      ↓
Better Visual Evidence
      ↓
Better Query Feature
      ↓
Better FDR Distribution
      ↓
Even More Accurate Box
```

Decoder 的多层结构与 FDR 是协同设计的。

---

# 83. 常见误解 1：`reg_max=32` 表示最大距离是 32

错误。

`reg_max=32` 主要表示：

```text
最大 bin index
```

因此：

```text
33 bins
```

真正每个 bin 对应的距离不是：

```text
0,1,2,...,32
```

而是：

```text
W(n)
```

这个非均匀坐标轴。

---

# 84. 常见误解 2：132 个输出就是 132 个坐标

错误。

它是：

```text
4 directions
×
33 logits
```

---

# 85. 常见误解 3：Softmax 后取 argmax

错误。

FDR 不是：

```text
argmax bin
```

而是：

```text
probability-weighted expectation
```

即：

$$
\sum p_nW(n)
$$

这样才能得到连续 refinement。

---

# 86. 常见误解 4：W(n) 是可学习参数

当前实现中不是。

它由：

```text
reg_max
up
reg_scale
```

按固定公式生成。

`up` 与 `reg_scale` 在模型中也是：

```text
requires_grad=False
```

所以这里是：

```text
设计好的 coordinate mapping
```

不是网络自由学习出来的 bin positions。

---

# 87. 常见误解 5：FDR 直接预测最终绝对 Box

不是。

它预测：

```text
相对于 reference box 四条边的 refinement
```

---

# 88. 常见误解 6：`distance=0` 表示 box 大小为 0

完全错误。

在 `distance2bbox()` 的设计里：

```text
distance = 0
```

表示：

```text
保持 reference box 不变
```

因为公式中本来就含有：

```text
0.5 * reg_scale
```

这个 base term。

---

# 89. 常见误解 7：FGL 取代了 L1/GIoU

不是。

它们监督不同层次：

```text
FGL
→ distribution representation

L1
→ final coordinates

GIoU
→ geometric overlap
```

---

# 90. 常见误解 8：每个 Decoder Layer 独立预测 Distribution

不是。

后续层使用：

```text
previous pred_corners
+
new residual
```

做 distribution refinement。

---

# 91. 常见误解 9：非均匀 W(n) 会导致最终坐标离散

不会。

Integral 使用期望：

```text
weighted sum
```

所以最终 distance 是连续的。

---

# 92. 常见误解 10：LQE 就是另一个 bbox head

不是。

LQE 读取：

```text
localization distribution statistics
```

生成：

```text
quality score
```

用于调整分类 score。

---

# 93. 源码阅读顺序

本课建议同时打开：

```text
engine/deim/dfine_decoder.py
engine/deim/dfine_utils.py
engine/deim/deim_criterion.py
```

然后按这个顺序追：

```text
1. DFINETransformer.__init__
   ↓
   reg_max
   reg_scale
   up
   dec_bbox_head
   integral

2. TransformerDecoder.forward
   ↓
   pre_bboxes
   ref_points_initial
   pred_corners
   integral(...)
   distance2bbox(...)

3. Integral.forward

4. dfine_utils.weighting_function

5. dfine_utils.distance2bbox

6. dfine_utils.bbox2distance

7. dfine_utils.translate_gt

8. Criterion loss_local
```

这样会形成：

```text
Forward Decode
+
Training Target Encode
```

完整闭环。

---

# 94. 推荐断点 A：`weighting_function`

打印：

```python
print(project)
print(project.shape)
```

X 应该看到：

```text
33 values
```

并确认：

```text
project[16] ≈ 0
```

---

# 95. 推荐断点 B：`bbox_head[i]`

打印：

```text
output.shape
pred_corners.shape
```

正常 Query 路线：

```text
output:
[B,300,384]

pred_corners:
[B,300,132]
```

训练有 DN 时 Q 维可能大于 300。

---

# 96. 推荐断点 C：`Integral.forward`

打印：

```text
input shape
reshaped shape
prob sum
output shape
```

检查：

```python
prob.sum(dim=-1)
```

应接近：

```text
1
```

---

# 97. 推荐断点 D：`distance2bbox`

选一个 Query：

```python
q = 0

print(points[0,q])
print(distance[0,q])
print(output_box[0,q])
```

然后自己用公式手算一次。

这是理解 FDR 最有效的调试方式之一。

---

# 98. 推荐断点 E：跨 Decoder Layers 看 Distribution

训练时记录同一个 Query：

```text
Layer 0 pred_corners
Layer 1 pred_corners
Layer 2 pred_corners
...
```

分别 softmax 后观察：

```text
peak position
entropy
expectation
```

你会真正看到：

```text
distribution refinement
```

而不是只看最终 bbox。

---

# 99. 推荐一个 Distribution Entropy 实验

定义：

$$
H(p)
=
-\sum_n p_n\log p_n
$$

代码：

```python
def entropy(prob):
    return -(
        prob
        * torch.log(
            prob + 1e-9
        )
    ).sum(dim=-1)
```

对不同 Decoder layers 比较：

```text
entropy
```

注意：

```text
更低 entropy
```

通常意味着 distribution 更集中，但不能简单把它等同于“bbox 一定更准”。

它只是一个理解 distribution shape 的工具。

---

# 100. 本课 Shape 速查表

| Tensor                |          Shape | 含义                            |
| --------------------- | -------------: | ------------------------------- |
| Decoder output        |  `[B,Q,384]` | Query feature                   |
| Preliminary delta     |    `[B,Q,4]` | 粗 box correction               |
| Preliminary box       |    `[B,Q,4]` | FDR reference                   |
| `pred_corners`      |  `[B,Q,132]` | 4×33 distribution logits       |
| Reshaped corners      | `[B,Q,4,33]` | 四个边界分布                    |
| Softmax probabilities | `[B,Q,4,33]` | localization distributions      |
| `project=W(n)`      |       `[33]` | 非均匀 bin 坐标                 |
| Integral distance     |    `[B,Q,4]` | l/t/r/b refinement              |
| Refined bbox          |    `[B,Q,4]` | cx/cy/w/h                       |
| LQE top-k stats       |   `[B,Q,20]` | 4×(4 top probs + mean)         |
| LQE quality           |    `[B,Q,1]` | localization quality correction |

---

# 101. 本课最重要的参数表

```text
reg_max = 32
```

意味着：

```text
33 bins / edge
132 logits / query
```

```text
up = 0.5
```

参与决定：

```text
W(n) 的边界范围
```

```text
reg_scale = 8  # X
```

同时影响：

```text
W(n) curvature
+
distance2bbox geometry
```

```text
k = 4  # LQE
```

表示每个方向取：

```text
Top-4 localization probabilities
```

---

# 102. 本课练习

练习 1

为什么：

```text
reg_max=32
```

对应 33 个 bins？

练习 2

为什么四个方向最终输出：

$$
4\times33=132
$$

而不是 128？

练习 3

解释：

```text
pred_corners
```

为什么不是最终的：

```text
[x1,y1,x2,y2]
```

练习 4

写出 Integral 的核心公式。

练习 5

为什么 D-FINE 不直接使用：

```text
W=[0,1,...,32]
```

练习 6

X 配置：

```text
up=0.5
reg_scale=8
```

求：

```text
upper_bound1
upper_bound2
```

练习 7

为什么：

```text
W(mid)=0
```

非常重要？

练习 8

证明：

```text
distance=[0,0,0,0]
```

时 `distance2bbox` 输出等于 reference box。

练习 9

Reference：

```text
[cx,cy,w,h]
=
[0.5,0.5,0.2,0.4]
```

`reg_scale=8`，若：

```text
d_l=1
```

左边界相对 reference 左边界移动多少？

练习 10

为什么同样的：

```text
d_l=1
```

对大目标和小目标产生不同绝对位移？

练习 11

为什么 distribution regression 最终仍能产生连续坐标？

练习 12

`translate_gt()` 为什么需要左右两个 bins，而不是只选最近 bin？

练习 13

FGL、L1、GIoU 分别监督什么？

练习 14

为什么 LQE 能区分：

```text
相同期望值
但不同尖锐程度
```

的两个 distribution？

练习 15

为什么更深 Decoder Layer 不需要从零预测完整 distribution？

练习 16

解释：

```text
pred_corners =
current residual
+
previous pred_corners
```

的意义。

练习 17

为什么：

```text
ref_points_detach
```

和：

```text
pred_corners_undetach
```

的梯度策略不同？

练习 18

为什么 FDR 特别适合：

```text
coarse-to-fine Decoder
```

练习 19

画出：

```text
FDR → reference → deformable attention → FDR
```

闭环。

练习 20

如果把 `reg_scale` 从 8 改成 4，你至少应该检查哪两部分数学行为发生变化？

---

# 103. 练习答案摘要

1、

bin index：

```text
0,1,...,32
```

总共：

$$
32+1=33
$$

2、

四个边界方向，每个 33 logits：

$$
4(32+1)=132
$$

4、

$$
d
=
\sum_n p_nW(n)
$$

6、

$$
upper_1
=
0.5\times8
=
4
$$

$$
upper_2
=
0.5\times8\times2
=
8
$$

7、

因为：

```text
W=0
```

对应：

```text
identity refinement
```

即 reference box 无需改变。

9、

$$
\Delta x
=
d_l\frac{w}{s}
=
1\times\frac{0.2}{8}
=
0.025
$$

左边界向左移动：

```text
0.025
```

10、

因为位移按：

```text
reference w/h
```

缩放。

11、

因为不是 argmax，而是：

```text
probability-weighted expectation
```

12、

为了保留连续 GT 位于两个离散坐标之间的精确位置。

13、

```text
FGL  → distribution
L1   → box coordinate
GIoU → overlap geometry
```

18、

因为 reference 越来越准时，需要的 residual 越接近 0，而 W(n) 恰好在 0 附近最细。

20、

至少检查：

```text
W(n) 的曲率/范围
distance2bbox 的几何缩放
```

---

# 104. 进入下一课前必须能独立回答的 12 个问题

```text
1. 为什么 D-FINE 的 bbox head 输出 132？

2. 132 怎样变成 [4,33]？

3. 为什么要对 33 logits 做 Softmax？

4. Integral 的数学意义是什么？

5. W(n) 为什么不是均匀的？

6. X 的 reg_scale 为什么必须按最终 merged config 取 8？

7. distance2bbox 到底怎样使用 reference box？

8. 为什么 distance=0 表示保持 reference 不变？

9. GT box 怎样变成 distribution target？

10. 为什么 target 要分配给相邻两个 bins？

11. FGL 和最终 L1/GIoU Loss 有什么区别？

12. Decoder layers 怎样进行 distribution refinement？
```

如果这些都能自己讲出来，你已经真正理解了：

```text
D-FINE 为什么不是普通的 4D box regression。
```

---

# 105. 本课最核心的一句话

> **FDR 不直接把 Query Feature 回归成四个最终坐标，而是围绕当前 Reference Box，为左、上、右、下四条边分别预测一个 33-bin 的细粒度概率分布；这些分布通过非均匀 Weighting Function 的期望被还原为连续边界修正量，再由 `distance2bbox` 转换成新的 Bounding Box，并在 Decoder 层间以 residual distribution 的形式继续迭代精炼。**

---

# 106. 把第 11、12 课连成一条完整链

现在你已经可以理解：

```text
HybridEncoder
↓
P3/P4/P5
↓
8400 spatial positions
↓
Encoder proposals
↓
Top-300
↓
Query Content + Reference Box
↓
Decoder Layer
├ Self-Attention
├ Deformable Cross-Attention
└ FFN
↓
Query Feature
[B,300,384]
↓
Preliminary Box
[B,300,4]
↓
FDR Head
[B,300,132]
↓
4 × 33 distributions
↓
Integral with non-uniform W(n)
↓
LTRB refinement
[B,300,4]
↓
distance2bbox
↓
Refined Box
[B,300,4]
↓
Next Decoder Layer
```

现在 `deim_dfine_x` 的 Decoder 定位主路径已经基本打开。

---

# 107. 下一课：GO-LSD 与 Decoder 多层监督

FDR 解决的是：

```text
怎样把定位表示做得更细？
```

但还有另一个问题：

```text
最后一层 Decoder 的 localization distribution
通常最好。

那么能不能让浅层 Decoder
也直接学习深层已经得到的定位知识？
```

这就是下一课：

# 第 13 课《D-FINE 的 GO-LSD 与 Decoder 多层监督》

我们会重点拆：

```text
teacher_corners
teacher_logits

loss_ddf

Global Optimal Localization Self-Distillation

为什么 teacher 来自模型自己？

为什么叫 global optimal？

为什么要把最终层知识传给浅层？

FGL 与 DDF 怎样同时工作？

detach 在 self-distillation 中意味着什么？

Decoder Layer 0~5 的 loss 到底怎样组织？
```

并把：

```text
loss_fgl
loss_ddf
aux_outputs
teacher_corners
```

从 Criterion 源码中完整串起来。

---

# 108. 源码核对说明

本课内容按当前公开 DEIM / D-FINE 实现核对，核心源码位置包括：

```text
DEIM:
engine/deim/dfine_decoder.py
engine/deim/dfine_utils.py
engine/deim/deim_criterion.py

D-FINE:
src/zoo/dfine/dfine_decoder.py
```

本课使用的关键真实实现关系为：

```text
dec_bbox_head:
4 * (reg_max + 1)

Integral:
softmax
+
non-uniform project W(n)

TransformerDecoder:
pred_corners
=
bbox_head[i](...)
+
pred_corners_undetach

inter_ref_bbox
=
distance2bbox(
    ref_points_initial,
    integral(pred_corners, project),
    reg_scale
)
```

`dfine_utils.py` 中：

```text
weighting_function
translate_gt
distance2bbox
bbox2distance
```

构成 FDR 编码/解码的核心数学工具。

配置链中：

```text
base/dfine_hgnetv2.yml:
reg_max = 32
reg_scale = 4

dfine_hgnetv2_x_coco.yml:
reg_scale = 8
```

因此本课针对 `deim_dfine_x` 的具体推导采用：

```text
reg_max = 32
reg_scale = 8
up = 0.5
```

---

# 109. 建议你保存的最终 FDR 心智模型

不要把 FDR 记成：

```text
“D-FINE 用了 DFL”
```

而应该记成：

```text
                    FDR
                     │
          ┌──────────┴──────────┐
          │                     │
      Representation         Refinement
          │                     │
          ▼                     ▼
  4 edge distributions    layer-to-layer
      × 33 bins           residual update
          │                     │
          └──────────┬──────────┘
                     ▼
             non-uniform W(n)
                     │
                     ▼
                  Integral
                     │
                     ▼
           continuous L/T/R/B
                     │
                     ▼
             Reference Box
                     │
                     ▼
              distance2bbox
                     │
                     ▼
              Refined Box
                     │
                     ▼
         Next Decoder Reference
```

真正理解这张图以后，你再看到：

```python
pred_corners
integral
project
reg_max
reg_scale
up
distance2bbox
bbox2distance
translate_gt
loss_fgl
```

就不会把它们当成一堆孤立函数，而会知道它们都属于同一个完整的定位系统。

---

## 参考源码与论文

本课建议对照以下公开资料阅读：

- DEIM 官方仓库：`Intellindust-AI-Lab/DEIM`
- DEIM `engine/deim/dfine_decoder.py`
- DEIM `engine/deim/dfine_utils.py`
- DEIM `configs/base/dfine_hgnetv2.yml`
- DEIM `configs/deim_dfine/dfine_hgnetv2_x_coco.yml`
- D-FINE 官方论文：*D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement*
- D-FINE 官方实现：`Peterande/D-FINE`

> 注：仓库会持续更新。后续课程实际逐行调试时，应始终以你使用的 checkout、最终 merged YAML，以及 forward hook / debugger 实测 Tensor shape 为最高依据。
