# 第 2 课：PyTorch 与 Tensor------以后所有 Shape 推导的基础

> **课程定位**：DEIMv2 从零到源码 · 第 2 / 24 课
> **本课目标**：彻底掌握阅读 DEIMv2 源码时最常见的 Tensor 维度语言与shape 操作。学完后，看到`flatten / reshape / view / transpose / permute / unsqueeze / cat / stack / split / expand / repeat / unbind`时，不再靠猜，而能在纸上准确推出输入输出 shape。
> **与 DEIMv2 的关系**：DINOv3 Patch Embedding、Multi-Head Attention、HybridEncoder 多尺度特征、Transformer 输入、Object Query、Reference Point 等都依赖这些操作。
> **建议方式**：本课所有实验都亲手运行，并在运行前先写出你预测的 shape。

---

# 0. 为什么第 2 课必须专门学 Tensor？

以后读 DEIMv2，你会不断遇到类似代码：

```python
x = self.proj(x).flatten(2).transpose(1, 2)
```

或者：

```python
qkv = self.qkv(x) \
    .reshape(B, N, 3, self.num_heads, C // self.num_heads) \
    .permute(2, 0, 3, 1, 4)

q, k, v = qkv.unbind(0)
```

如果不熟悉 Tensor，这些代码看起来像"魔法"。

但实际上它们几乎都只是在做三件事：

```text
① 改变维度的组织方式
② 调整维度顺序
③ 合并/拆分多个 Tensor
```

神经网络真正困难的数学反而常常只有一两行。

因此，读视觉 Transformer 源码时必须养成：

> **任何一行 Tensor 操作，都先写 shape，再看数值。**

---

# 1. Tensor 到底是什么？

先从最简单的开始。

## 1.1 标量：0 维 Tensor

```python
import torch

x = torch.tensor(5.0)

print(x)
print(x.shape)
print(x.ndim)
```

输出：

```text
tensor(5.)
torch.Size([])
0
```

数学上：

\[ x=5 \]

---

## 1.2 向量：1 维 Tensor

```python
x = torch.tensor([1, 2, 3, 4])

print(x.shape)
```

```text
torch.Size([4])
```

数学上：

\[ x`\in`{=tex}`\mathbb{R}`{=tex}\^{4} \]

---

## 1.3 矩阵：2 维 Tensor

```python
x = torch.randn(3, 4)
```

shape：

```text
[3,4]
```

可以理解为：

```text
3 rows
4 columns
```

数学上：

\[ X`\in`{=tex}`\mathbb{R}`{=tex}\^{3`\times4`{=tex}} \]

---

## 1.4 图像 Batch：4 维 Tensor

视觉模型最常见：

```python
images = torch.randn(2, 3, 640, 640)
```

shape：

```text
[B,C,H,W]
=
[2,3,640,640]
```

其中：

```text
B = Batch
C = Channel
H = Height
W = Width
```

元素总数：

\[ 2`\times3`{=tex}`\times640`{=tex}`\times640`{=tex} = 2,457,600 \]

可以验证：

```python
print(images.numel())
```

---

# 2. Shape 是 Tensor 的"语法"

假设：

```python
x = torch.randn(2, 256, 80, 80)
```

不要只看到四个数字。

应该立即翻译：

```text
2   → 两张图片
256 → 每个空间位置有 256 个 feature channels
80  → feature map height
80  → feature map width
```

所以：

```text
[B,C,H,W]
```

不是装饰性符号。

它告诉你每个维度的**语义**。

同样：

```text
[B,N,C]
```

在 Transformer 中通常意味着：

```text
B = batch
N = token/query 数量
C = embedding dimension
```

例如：

```text
[2,8400,256]
```

表示：

```text
2 张图片
每张 8400 个 feature tokens
每个 token 256 维
```

而：

```text
[2,300,256]
```

可能表示：

```text
2 张图片
每张 300 个 object queries
每个 query 256 维
```

虽然最后一维相同，但 **8400 和 300 的语义完全不同**。

---

# 3. Tensor 的 dimension 编号

假设：

```python
x = torch.randn(2, 256, 80, 80)
```

维度编号：

```text
dim=0 → B → 2
dim=1 → C → 256
dim=2 → H → 80
dim=3 → W → 80
```

Python 也允许负数索引：

```text
dim=-1 → 最后一维 → W
dim=-2 → H
dim=-3 → C
dim=-4 → B
```

实验：

```python
print(x.shape[0])
print(x.shape[1])
print(x.shape[-1])
```

输出：

```text
2
256
80
```

以后源码里：

```python
C = x.shape[-1]
```

经常意味着：

> 最后一维是 embedding dimension。

---

# 4. `reshape()`：重新解释 Tensor 的形状

先构造：

```python
x = torch.arange(24)

print(x)
print(x.shape)
```

shape：

```text
[24]
```

现在：

```python
y = x.reshape(2, 12)
```

得到：

```text
[2,12]
```

还可以：

```python
z = x.reshape(2, 3, 4)
```

得到：

```text
[2,3,4]
```

关键规律：

> reshape 前后元素总数必须相同。

所以：

\[ 24=2`\times12`{=tex}=2`\times3`{=tex}`\times4`{=tex} \]

---

# 5. `-1`：让 PyTorch 自动计算某个维度

```python
x = torch.randn(2, 3, 4)
```

元素总数：

\[ 2`\times3`{=tex}`\times4`{=tex}=24 \]

执行：

```python
y = x.reshape(2, -1)
```

PyTorch 自动求：

\[ 24/2=12 \]

所以：

```text
[2,12]
```

再例如：

```python
y = x.reshape(-1, 4)
```

得到：

```text
[6,4]
```

因为：

\[ 24/4=6 \]

注意：

```python
x.reshape(-1, -1)
```

不允许，因为 PyTorch 无法唯一推断两个未知维度。

---

# 6. `view()` 与 `reshape()` 有什么区别？

表面：

```python
x.view(...)
x.reshape(...)
```

都能改 shape。

但它们与 Tensor 的内存布局有关。

先记结论：

```text
view
→ 更严格，需要兼容当前内存布局

reshape
→ 更灵活，必要时可能创建合适的新布局/副本
```

对于初学源码阅读：

> 如果只是推导 shape，两者先都理解成"改变 Tensor 的形状"。

但以后看到：

```python
x = x.permute(...)
x = x.contiguous().view(...)
```

就需要理解内存布局。

本课后面专门讲。

---

# 7. `flatten()`：把多个连续维度压成一个维度

这是 DEIMv2 中极重要的操作。

假设：

```python
x = torch.randn(2, 256, 80, 80)
```

shape：

```text
[B,C,H,W]
```

执行：

```python
y = x.flatten(2)
```

`2` 表示：

> 从 dim=2 开始，把后面的维度全部 flatten。

所以：

```text
[B,C,H,W]
→
[B,C,H×W]
```

具体：

```text
[2,256,80,80]
→
[2,256,6400]
```

因为：

\[ 80`\times80`{=tex}=6400 \]

---

# 8. `flatten(2)` 为什么在 ViT/DETR 中这么常见？

CNN 输出：

```text
[B,C,H,W]
```

Transformer 通常喜欢：

```text
[B,N,C]
```

第一步：

```python
x = x.flatten(2)
```

得到：

```text
[B,C,HW]
```

还不够。

我们希望：

```text
[B,HW,C]
```

所以还要：

```python
x = x.transpose(1, 2)
```

最终：

```text
[B,C,H,W]
      ↓ flatten(2)
[B,C,HW]
      ↓ transpose(1,2)
[B,HW,C]
```

这条转换请记住：

\[ `\boxed{ BCHW \rightarrow BNC }`{=tex} \]

其中：

\[ N=H`\times `{=tex}W \]

---

# 9. 真实 DEIMv2 源码：Patch Embedding

DEIMv2 官方 `engine/backbone/vit_tiny.py` 的 Patch Embedding 核心
forward 是：

```python
def forward(self, x):
    return self.proj(x).flatten(2).transpose(1, 2)
```

现在我们已经能逐步读懂。

假设：

```text
input:
[B,3,640,640]
```

若：

```python
self.proj = nn.Conv2d(
    3,
    256,
    kernel_size=16,
    stride=16
)
```

那么卷积后：

```text
[B,256,40,40]
```

因为：

\[ 640/16=40 \]

接着：

```python
.flatten(2)
```

得到：

```text
[B,256,1600]
```

因为：

\[ 40`\times40`{=tex}=1600 \]

然后：

```python
.transpose(1,2)
```

得到：

```text
[B,1600,256]
```

所以一行：

```python
self.proj(x).flatten(2).transpose(1, 2)
```

实际上完成：

```text
Image
[B,3,640,640]

↓ patch projection

[B,256,40,40]

↓ flatten spatial dimensions

[B,256,1600]

↓ move embedding dimension to last

[B,1600,256]
```

这就是：

> **二维图片 → Transformer token sequence**

的核心 shape 变化。

---

# 10. `transpose()`：交换两个维度

假设：

```python
x = torch.randn(2, 256, 6400)
```

执行：

```python
y = x.transpose(1, 2)
```

只交换 dim 1 和 dim 2：

```text
[2,256,6400]
→
[2,6400,256]
```

注意：

> `transpose()` 不会改变元素数量，也不是矩阵乘法。

它只是交换两个 axis。

---

# 11. `permute()`：任意重新排列所有维度

`transpose()` 只能交换两个维度。

`permute()` 可以重新指定整个维度顺序。

例如：

```python
x = torch.randn(2, 3, 4, 5)
```

原维度：

```text
dim0 = 2
dim1 = 3
dim2 = 4
dim3 = 5
```

执行：

```python
y = x.permute(0, 2, 3, 1)
```

表示新顺序：

```text
old dim0
old dim2
old dim3
old dim1
```

所以：

```text
[2,3,4,5]
→
[2,4,5,3]
```

这在视觉里常见于：

```text
BCHW
→
BHWC
```

即：

```python
x = x.permute(0, 2, 3, 1)
```

---

# 12. 真实 DEIMv2 Attention：`reshape + permute`

DEIMv2 `vit_tiny.py` 中 Attention 有类似：

```python
B, N, C = x.shape

qkv = self.qkv(x) \
    .reshape(
        B,
        N,
        3,
        self.num_heads,
        C // self.num_heads
    ) \
    .permute(2, 0, 3, 1, 4)

q, k, v = qkv.unbind(0)
```

这是本课最重要的真实源码 shape 推导之一。

假设：

```text
B = 2
N = 1600
C = 256
num_heads = 8
```

则：

\[ head_dim=256/8=32 \]

---

## 第一步

输入：

```text
x:
[2,1600,256]
```

---

## 第二步：`self.qkv(x)`

通常：

```python
self.qkv = nn.Linear(C, C * 3)
```

因此：

```text
[2,1600,256]
→
[2,1600,768]
```

为什么 768？

\[ 768=3`\times256`{=tex} \]

因为一次生成：

```text
Q
K
V
```

三份表示。

---

## 第三步：reshape

```python
.reshape(
    B,
    N,
    3,
    num_heads,
    C // num_heads
)
```

得到：

```text
[2,1600,3,8,32]
```

检查元素：

\[ 3`\times8`{=tex}`\times32`{=tex} = 768 \]

完全一致。

---

## 第四步：permute

原：

```text
[B,N,3,heads,head_dim]
```

也就是：

```text
[2,1600,3,8,32]
```

执行：

```python
.permute(2,0,3,1,4)
```

变成：

```text
[3,B,heads,N,head_dim]
```

具体：

```text
[3,2,8,1600,32]
```

为什么把 3 放最前面？

因为下一句：

```python
q, k, v = qkv.unbind(0)
```

正好沿 dim 0 拆成三个 Tensor。

于是：

```text
q = [2,8,1600,32]
k = [2,8,1600,32]
v = [2,8,1600,32]
```

这就是 Multi-Head Attention 的 Q/K/V shape。

---

# 13. `unbind()`：删除某个维度并逐片拆开

最简单实验：

```python
x = torch.randn(3, 2, 4)

a, b, c = x.unbind(0)
```

原：

```text
[3,2,4]
```

沿 dim 0 有 3 份。

所以：

```text
a = [2,4]
b = [2,4]
c = [2,4]
```

注意：

> `unbind()` 拆完后，被拆的那个维度消失。

这正适合：

```text
[3,B,heads,N,d]
```

变成：

```text
Q [B,heads,N,d]
K [B,heads,N,d]
V [B,heads,N,d]
```

---

# 14. Attention 的 shape 现在可以推导了

已有：

```text
Q = [B,h,N,d]
K = [B,h,N,d]
V = [B,h,N,d]
```

其中：

```text
h = num_heads
d = head_dim
C = h × d
```

Attention：

\[ Attention(Q,K,V) = softmax`\left`{=tex}( `\frac{QK^T}{\sqrt d}`{=tex}
`\right`{=tex})V \]

先看：

```python
k.transpose(-2, -1)
```

原：

```text
[B,h,N,d]
```

交换最后两维：

```text
[B,h,d,N]
```

所以：

```text
Q @ Kᵀ
```

shape：

```text
[B,h,N,d]
@
[B,h,d,N]
```

得到：

```text
[B,h,N,N]
```

这就是 attention matrix。

再：

```text
[B,h,N,N]
@
[B,h,N,d]
```

得到：

```text
[B,h,N,d]
```

最后重新合并 heads：

```text
[B,h,N,d]
→
[B,N,h,d]
→
[B,N,C]
```

因为：

\[ C=h`\times `{=tex}d \]

以后看到：

```text
[B,8,1600,32]
```

应该立即知道：

```text
8 × 32 = 256
```

它只是把原本 256 维 embedding 拆成了 8 个 attention heads。

---

# 15. `unsqueeze()`：增加一个长度为 1 的维度

假设：

```python
x = torch.randn(300, 256)
```

shape：

```text
[300,256]
```

如果想给它增加 batch 维：

```python
y = x.unsqueeze(0)
```

得到：

```text
[1,300,256]
```

再例如：

```python
x.unsqueeze(1)
```

得到：

```text
[300,1,256]
```

注意：

```text
unsqueeze(dim)
```

是在指定位置插入一个大小为 1 的维度。

---

# 16. `squeeze()`：删除长度为 1 的维度

```python
x = torch.randn(1, 300, 1, 256)
```

执行：

```python
y = x.squeeze()
```

得到：

```text
[300,256]
```

因为所有大小为 1 的维度都被移除。

但源码中更推荐明确指定：

```python
x.squeeze(0)
```

因为它只删除 dim 0。

这样不容易因为其他维度碰巧等于 1 而产生 bug。

---

# 17. Object Query 为什么经常需要 `unsqueeze + repeat/expand`？

假设 query embedding 参数：

```text
[300,256]
```

它代表一套共享的 300 个 queries。

但是 batch 有：

```text
B = 4
```

我们希望：

```text
[4,300,256]
```

可以：

```python
queries = query_embed.unsqueeze(0)
```

得到：

```text
[1,300,256]
```

然后：

```python
queries = queries.expand(4, -1, -1)
```

得到：

```text
[4,300,256]
```

这里：

```text
-1
```

表示：

> 这个维度保持原大小。

---

# 18. `expand()` 和 `repeat()`：看起来一样，其实不同

假设：

```python
x = torch.tensor([[1, 2, 3]])
```

shape：

```text
[1,3]
```

## expand

```python
y = x.expand(4, 3)
```

shape：

```text
[4,3]
```

逻辑上：

```text
1 2 3
1 2 3
1 2 3
1 2 3
```

但 `expand` 通常利用 stride/broadcast 视图，不真正复制所有数据。

---

## repeat

```python
z = x.repeat(4, 1)
```

同样得到：

```text
[4,3]
```

但 `repeat` 实际重复数据。

先记：

```text
expand
→ 通常更省内存
→ 只能扩展原来 size=1 的相关维度

repeat
→ 真正重复数据
→ 更自由但会占更多内存
```

---

# 19. Broadcasting：很多"神秘加法"的根源

假设：

```text
A = [2,300,256]
B = [256]
```

执行：

```python
C = A + B
```

为什么可以？

PyTorch 会从最后一维开始比较：

```text
A: [2,300,256]
B: [      256]
```

B 会逻辑上广播为：

```text
[2,300,256]
```

所以可以相加。

---

## 19.1 Broadcasting 基本规则

从右向左比较维度。

两个维度兼容，如果：

```text
① 大小相同
或
② 其中一个大小为 1
或
③ 某一方不存在该维度
```

例如：

```text
[2,300,256]
[1,  1,256]
```

可以。

```text
[2,300,256]
[       256]
```

可以。

```text
[2,300,256]
[2,  1,256]
```

可以。

但：

```text
[2,300,256]
[2,200,256]
```

不能直接 broadcast，因为：

```text
300 != 200
```

且都不是 1。

---

# 20. 一个与 Reference Point 很像的广播实验

假设：

```python
B = 2
N = 300
L = 3

reference_points = torch.randn(B, N, 1, 2)
valid_ratios = torch.randn(B, 1, L, 2)

out = reference_points * valid_ratios
```

shape：

```text
reference_points
[2,300,1,2]

valid_ratios
[2,1,3,2]
```

广播：

```text
[2,300,1,2]
[2,  1,3,2]
-----------
[2,300,3,2]
```

这类操作在多尺度 deformable attention 中非常常见。

意思可以理解为：

> 每一个 query/reference point 都需要针对多个 feature levels
> 得到相应坐标。

后面第 7 课会真正进入这类代码。

---

# 21. `cat()`：沿已有维度拼接

这是多尺度检测极常见的操作。

假设：

```text
P3 tokens = [B,6400,256]
P4 tokens = [B,1600,256]
P5 tokens = [B, 400,256]
```

执行：

```python
memory = torch.cat(
    [p3, p4, p5],
    dim=1
)
```

因为 dim 1 是 token 数：

```text
6400 + 1600 + 400
=
8400
```

所以：

```text
memory
=
[B,8400,256]
```

---

# 22. `cat()` 的规则

除了被拼接的那个维度，其他维度必须相同。

例如：

```text
A = [2,6400,256]
B = [2,1600,256]
```

可以：

```python
torch.cat([A, B], dim=1)
```

得到：

```text
[2,8000,256]
```

但如果：

```text
A = [2,6400,256]
B = [2,1600,128]
```

不能沿 dim 1 直接 cat。

因为最后一维：

```text
256 != 128
```

这也是为什么 HybridEncoder 常需要 input projection：

> 不同 Backbone feature channels 先投影到统一 hidden
> dimension，再方便后续融合。

---

# 23. `stack()`：增加一个新维度

这是 `cat()` 与 `stack()` 最重要的区别。

假设：

```text
A = [2,300,256]
B = [2,300,256]
C = [2,300,256]
```

执行：

```python
x = torch.stack([A, B, C], dim=0)
```

得到：

```text
[3,2,300,256]
```

注意：

> stack 创建了一个新的维度。

而：

```python
torch.cat([A,B,C], dim=0)
```

得到：

```text
[6,300,256]
```

因为 cat 是扩大已有 dim 0。

口诀：

```text
cat   → 接长一个已有维度
stack → 新建一个维度
```

---

# 24. 为什么 Decoder 各层输出适合 `stack()`？

假设 decoder 有 6 层。

每层输出：

```text
[B,300,256]
```

如果保存所有层：

```python
outputs = torch.stack(layer_outputs)
```

得到：

```text
[6,B,300,256]
```

第一维：

```text
6
```

现在具有新的语义：

```text
decoder layer index
```

于是：

```text
outputs[0]
```

是第一层 decoder 输出，

```text
outputs[-1]
```

是最后一层。

这对：

```text
auxiliary loss
intermediate supervision
layer-wise refinement
```

都非常重要。

---

# 25. `split()`：按指定大小拆 Tensor

假设：

```python
x = torch.randn(2, 8400, 256)
```

我们知道 8400 来自：

```text
6400
1600
400
```

可以：

```python
p3, p4, p5 = torch.split(
    x,
    [6400, 1600, 400],
    dim=1
)
```

得到：

```text
p3 = [2,6400,256]
p4 = [2,1600,256]
p5 = [2, 400,256]
```

如果再知道空间尺寸：

```text
80×80
40×40
20×20
```

就能恢复 feature maps。

---

# 26. 从 Token 恢复成 Feature Map

假设：

```text
p3 = [B,6400,256]
```

目标：

```text
[B,256,80,80]
```

先：

```python
p3 = p3.transpose(1, 2)
```

得到：

```text
[B,256,6400]
```

再：

```python
p3 = p3.reshape(B, 256, 80, 80)
```

得到：

```text
[B,256,80,80]
```

所以：

```text
Feature Map
[B,C,H,W]

      ↕ 可逆的 shape organization

Tokens
[B,HW,C]
```

只要你还知道：

```text
H
W
```

就可以在两种表示之间转换。

---

# 27. `chunk()`：平均拆成若干份

```python
x = torch.randn(2, 300, 768)

q, k, v = x.chunk(3, dim=-1)
```

最后一维：

```text
768 / 3 = 256
```

所以：

```text
q = [2,300,256]
k = [2,300,256]
v = [2,300,256]
```

这也是生成 Q/K/V 的一种写法。

与前面的：

```text
reshape + permute + unbind
```

相比，只是组织方式不同。

---

# 28. Indexing 与 Slicing：源码里到处都是

假设：

```python
x = torch.randn(2, 1601, 256)
```

为什么可能是 1601？

```text
1 CLS token
+
1600 patch tokens
```

于是：

```python
cls_token = x[:, 0]
```

shape：

```text
[2,256]
```

而：

```python
patch_tokens = x[:, 1:]
```

shape：

```text
[2,1600,256]
```

DEIMv2 的 ViT 源码就有这样的模式：

```python
outs.append((x[:, 1:], x[:, 0]))
```

也就是把：

```text
patch tokens
```

和：

```text
CLS token
```

分开保存。

---

# 29. `...`：Ellipsis 索引

假设：

```python
x = torch.randn(2, 8, 1600, 32)
```

源码可能写：

```python
x[..., :16]
```

意思是：

> 前面所有维度全部保留，只在最后一维取前 16 个元素。

结果：

```text
[2,8,1600,16]
```

例如官方 ViT 的 RoPE 辅助函数中会看到：

```python
x1 = x[..., : x.shape[-1] // 2]
x2 = x[..., x.shape[-1] // 2 :]
```

如果：

```text
last dim = 32
```

则：

```text
x1 = [...,16]
x2 = [...,16]
```

---

# 30. `contiguous()` 到底是什么？

这是 PyTorch 初学者经常被一句话带过的地方。

考虑：

```python
x = torch.arange(12).reshape(3, 4)
```

Tensor 底层数据在内存中按某种顺序排列。

执行：

```python
y = x.transpose(0, 1)
```

逻辑 shape 变成：

```text
[4,3]
```

但 PyTorch 不一定真的把所有元素复制到新的连续内存。

它可能只是改变：

```text
shape
stride
```

告诉 Tensor：

> "以后按照另一种方式解释同一块数据。"

因此：

```python
y.is_contiguous()
```

可能为：

```text
False
```

---

# 31. 什么是 stride？注意它不是 CNN stride

这里的 Tensor stride 与卷积 stride 是不同概念。

Tensor stride 表示：

> 沿某一维移动一个位置，在底层 storage 中需要跨过多少个元素。

实验：

```python
x = torch.arange(12).reshape(3, 4)

print(x.shape)
print(x.stride())
```

典型：

```text
shape  = [3,4]
stride = [4,1]
```

含义：

```text
沿 row 走一步 → 跨 4 个元素
沿 column 走一步 → 跨 1 个元素
```

transpose 后：

```python
y = x.transpose(0, 1)

print(y.shape)
print(y.stride())
```

会看到 stride 也交换。

所以：

> Tensor 的逻辑维度顺序变了，但底层数据不一定被重新排列。

---

# 32. 为什么经常出现 `.contiguous().view(...)`？

因为 `view()` 要求底层布局满足它重新解释 shape 的条件。

如果刚刚：

```python
x = x.permute(...)
```

得到非连续 Tensor，

直接：

```python
x.view(...)
```

可能失败。

于是源码常见：

```python
x = x.permute(...).contiguous().view(...)
```

`contiguous()` 会创建符合连续布局的数据表示。

简化理解：

```text
permute
→ 改逻辑维度顺序

contiguous
→ 必要时按这个新顺序重新排好内存

view
→ 再安全地重新解释 shape
```

---

# 33. Shape 操作是否会改变数值？

这是一个非常重要的分类。

## 通常只是改变"怎么看数据"

例如：

```text
reshape
view
flatten
transpose
permute
unsqueeze
squeeze
```

这些操作本身通常不做神经网络意义上的：

```text
feature extraction
classification
attention
convolution
```

它们主要改变数据组织。

---

## 会组合/复制/选择数据

例如：

```text
cat
stack
repeat
indexing
split
unbind
```

它们可能构造新 Tensor、选择数据或重新组织数据。

---

## 真正做数值计算

例如：

```text
Linear
Conv2d
LayerNorm
softmax
matmul
attention
activation
```

以后读源码要学会区分：

```text
这一行是在“算新的 feature”

还是

这一行只是“换一种 shape 组织 feature”
```

这是源码阅读速度提升非常明显的分界点。

---

# 34. PyTorch 中的 dtype

Tensor 不只有 shape。

还有：

```python
x.dtype
```

常见：

```text
torch.float32
torch.float16
torch.bfloat16
torch.int64
torch.bool
```

例如图片 feature：

```python
x = torch.randn(2, 256, 80, 80)
print(x.dtype)
```

通常：

```text
float32
```

类别 index：

```python
labels = torch.tensor([1, 5, 9])
```

通常：

```text
int64
```

mask：

```python
mask = torch.tensor([True, False, True])
```

是：

```text
bool
```

以后 AMP/FP16 会让部分计算使用低精度 dtype，以减少显存并提高吞吐。

---

# 35. Device：CPU 与 GPU

```python
x.device
```

可能：

```text
cpu
```

或者：

```text
cuda:0
```

例如：

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

x = torch.randn(2, 3, 640, 640).to(device)
```

重要规则：

> 参与同一次普通运算的 Tensor 通常必须位于兼容的 device 上。

例如：

```text
GPU Tensor + CPU Tensor
```

往往会报错。

所以源码里经常看到：

```python
device=x.device
dtype=x.dtype
```

用已有 Tensor 创建新的 Tensor 时继承设备和 dtype。

---

# 36. 一个非常实用的调试函数

以后读 DEIMv2，可以自己写：

```python
def inspect(name, x):
    print(
        f"{name:20s}",
        f"shape={tuple(x.shape)}",
        f"dtype={x.dtype}",
        f"device={x.device}",
        f"contiguous={x.is_contiguous()}"
    )
```

例如：

```python
x = torch.randn(2, 256, 80, 80)

inspect("feature", x)

tokens = x.flatten(2).transpose(1, 2)

inspect("tokens", tokens)
```

以后我们会进一步升级成：

```text
ShapeTracer
forward hooks
parameter counter
gradient tracer
```

用于真正追踪 DEIMv2。

---

# 37. 综合实验 1：模拟三层 Feature Maps

先不要运行，自己预测每一行。

```python
import torch

B = 2
C = 256

p3 = torch.randn(B, C, 80, 80)
p4 = torch.randn(B, C, 40, 40)
p5 = torch.randn(B, C, 20, 20)

p3 = p3.flatten(2).transpose(1, 2)
p4 = p4.flatten(2).transpose(1, 2)
p5 = p5.flatten(2).transpose(1, 2)

memory = torch.cat([p3, p4, p5], dim=1)

print(p3.shape)
print(p4.shape)
print(p5.shape)
print(memory.shape)
```

答案：

```text
[2,6400,256]
[2,1600,256]
[2, 400,256]

[2,8400,256]
```

数学：

\[ N = 80^2+40^2+20\^2 = 6400+1600+400 = 8400 \]

---

# 38. 综合实验 2：模拟 Multi-Head QKV

```python
import torch
import torch.nn as nn

B = 2
N = 300
C = 256
num_heads = 8

x = torch.randn(B, N, C)

qkv_layer = nn.Linear(C, C * 3)

qkv = qkv_layer(x)

print("after linear:", qkv.shape)

qkv = qkv.reshape(
    B,
    N,
    3,
    num_heads,
    C // num_heads
)

print("after reshape:", qkv.shape)

qkv = qkv.permute(2, 0, 3, 1, 4)

print("after permute:", qkv.shape)

q, k, v = qkv.unbind(0)

print("Q:", q.shape)
print("K:", k.shape)
print("V:", v.shape)

scores = q @ k.transpose(-2, -1)

print("attention scores:", scores.shape)
```

预测：

```text
after linear:
[2,300,768]

after reshape:
[2,300,3,8,32]

after permute:
[3,2,8,300,32]

Q/K/V:
[2,8,300,32]

attention scores:
[2,8,300,300]
```

请自己验证：

\[ 32=256/8 \]

以及：

\[ \[300,32\]`\times[32,300]`{=tex}= \[300,300\] \]

---

# 39. 综合实验 3：从 8400 Tokens 恢复三层 Feature Maps

```python
import torch

B = 2
C = 256

memory = torch.randn(B, 8400, C)

p3, p4, p5 = torch.split(
    memory,
    [6400, 1600, 400],
    dim=1
)

p3 = p3.transpose(1, 2).reshape(B, C, 80, 80)
p4 = p4.transpose(1, 2).reshape(B, C, 40, 40)
p5 = p5.transpose(1, 2).reshape(B, C, 20, 20)

print(p3.shape)
print(p4.shape)
print(p5.shape)
```

输出：

```text
[2,256,80,80]
[2,256,40,40]
[2,256,20,20]
```

现在应该能看到：

```text
BCHW
↔
BNC
```

本质上只是两种不同的数据组织方式。

---

# 40. 综合实验 4：Object Queries 扩展到整个 Batch

```python
import torch

num_queries = 300
hidden_dim = 256
B = 4

query_embed = torch.randn(
    num_queries,
    hidden_dim
)

print(query_embed.shape)

queries = query_embed.unsqueeze(0)

print(queries.shape)

queries = queries.expand(B, -1, -1)

print(queries.shape)
```

得到：

```text
[300,256]
[1,300,256]
[4,300,256]
```

注意：

```text
300
```

始终是 query 数。

```text
4
```

只是 batch size。

---

# 41. 综合实验 5：Broadcasting

```python
import torch

B = 2
N = 300
L = 3

reference_points = torch.randn(
    B, N, 1, 2
)

valid_ratios = torch.randn(
    B, 1, L, 2
)

locations = reference_points * valid_ratios

print(reference_points.shape)
print(valid_ratios.shape)
print(locations.shape)
```

预测：

```text
[2,300,1,2]
[2,1,3,2]
[2,300,3,2]
```

请解释：

```text
300
```

为什么没有和：

```text
3
```

发生冲突？

答案是因为它们处在不同维度，而另一边对应维度大小为 1，可以广播。

---

# 42. 综合实验 6：`cat` 与 `stack` 的区别

```python
import torch

a = torch.randn(2, 300, 256)
b = torch.randn(2, 300, 256)
c = torch.randn(2, 300, 256)

x1 = torch.cat([a, b, c], dim=0)
x2 = torch.stack([a, b, c], dim=0)

print(x1.shape)
print(x2.shape)
```

结果：

```text
cat:
[6,300,256]

stack:
[3,2,300,256]
```

问自己：

> 如果 `a/b/c` 分别代表 decoder layer 1/2/3 的输出，哪个更合理？

通常：

```text
stack
```

因为我们希望保留一个明确的：

```text
layer dimension
```

---

# 43. 综合实验 7：观察 contiguous

```python
import torch

x = torch.randn(2, 3, 4)

print(
    "x:",
    x.shape,
    x.stride(),
    x.is_contiguous()
)

y = x.permute(0, 2, 1)

print(
    "y:",
    y.shape,
    y.stride(),
    y.is_contiguous()
)

z = y.contiguous()

print(
    "z:",
    z.shape,
    z.stride(),
    z.is_contiguous()
)
```

观察：

```text
shape
stride
contiguous
```

怎样同时变化。

这个实验比死记：

> "permute 后要 contiguous"

更重要。

---

# 44. DEIMv2 源码阅读挑战：你现在已经能读这一段

真实 ViT Attention 的核心 shape 代码可以抽象成：

```python
B, N, C = x.shape

qkv = self.qkv(x) \
    .reshape(
        B,
        N,
        3,
        self.num_heads,
        C // self.num_heads
    ) \
    .permute(2, 0, 3, 1, 4)

q, k, v = qkv.unbind(0)

x = scaled_dot_product_attention(
    q, k, v
)

x = x.transpose(1, 2).reshape(B, N, C)
```

现在逐行写 shape。

假设：

```text
B = 2
N = 1600
C = 256
heads = 8
```

则：

```text
x
[2,1600,256]

self.qkv(x)
[2,1600,768]

reshape
[2,1600,3,8,32]

permute
[3,2,8,1600,32]

unbind
Q,K,V:
[2,8,1600,32]

attention output
[2,8,1600,32]

transpose(1,2)
[2,1600,8,32]

reshape
[2,1600,256]
```

注意最终 shape 与输入一致：

```text
[B,N,C]
→ Attention
→ [B,N,C]
```

这就是 Transformer block 可以做 residual：

```python
x = x + attention(x)
```

的重要原因之一。

两边 shape 必须相同。

---

# 45. Shape 推导的四条铁律

以后看到任何陌生代码，优先检查这四件事。

## 铁律 1：元素总数

reshape 前后：

\[ `\prod `{=tex}shape\_{before} = `\prod `{=tex}shape\_{after} \]

例如：

```text
[2,1600,768]
```

和：

```text
[2,1600,3,8,32]
```

因为：

\[ 768=3`\times8`{=tex}`\times32`{=tex} \]

---

## 铁律 2：矩阵乘法内维必须一致

例如：

```text
[...,N,d]
@
[...,d,M]
```

输出：

```text
[...,N,M]
```

---

## 铁律 3：cat 除拼接维之外必须兼容

```text
[B,N1,C]
[B,N2,C]

cat dim=1

→ [B,N1+N2,C]
```

---

## 铁律 4：broadcast 从右向左检查

例如：

```text
[B,N,1,2]
[B,1,L,2]

→ [B,N,L,2]
```

只要掌握这四条，绝大多数 shape 都能推出来。

---

# 46. DEIMv2 中常见的 Shape 字母表

以后课程统一使用：

```text
B = batch size
C = channel / hidden dimension
H = feature height
W = feature width

N = token/query 数量
L = number of feature levels
M = number of GT / another sequence length

h = number of attention heads
d = head dimension

K = number of classes
P = number of sampling points
```

典型：

```text
Image
[B,3,H,W]

Feature Map
[B,C,H,W]

Tokens
[B,N,C]

Queries
[B,Nq,C]

Q/K/V
[B,h,N,d]

Attention
[B,h,N,N]

Multi-level reference
[B,N,L,2]

Class logits
[B,Nq,K]

Boxes
[B,Nq,4]
```

以后每一课都会继续使用这套符号。

---

# 47. 初学者最容易犯的 10 个 Shape 错误

### 1. 把 channel 和 token 数搞反

```text
[B,256,8400]
```

和：

```text
[B,8400,256]
```

不是同一个 layout。

---

### 2. 认为 transpose 会改变元素数量

不会。

---

### 3. 把 `cat` 和 `stack` 当成同一件事

不是。

---

### 4. 忘记 `unbind` 会删除被拆的维度

```text
[3,B,N,C]
→
3 × [B,N,C]
```

---

### 5. 看到 `-1` 就不知道 shape

先计算总元素数。

---

### 6. 认为 `expand` 与 `repeat` 内存行为一样

不是。

---

### 7. 忘记 broadcasting 是从右向左比较

这是最常见的广播错误。

---

### 8. `permute` 后直接乱用 `view`

先考虑 contiguous/layout。

---

### 9. 把 Tensor stride 与 CNN stride 混为一谈

两个概念完全不同。

---

### 10. 只看数字，不看维度语义

例如：

```text
300
```

可能是：

```text
queries
GT slots
sequence length
```

数字本身没有意义。

必须知道它对应哪个 axis。

---

# 48. 一个更专业的 Shape 注释习惯

以后自己改 DEIMv2 时，建议写：

```python
# x: [B, C, H, W]
x = x.flatten(2)

# x: [B, C, HW]
x = x.transpose(1, 2)

# x: [B, HW, C]
```

Attention：

```python
# qkv: [B, N, 3C]
qkv = self.qkv(x)

# qkv: [B, N, 3, h, d]
qkv = qkv.reshape(...)

# qkv: [3, B, h, N, d]
qkv = qkv.permute(...)

# q/k/v: [B, h, N, d]
q, k, v = qkv.unbind(0)
```

这样阅读大型模型时会非常省脑力。

---

# 49. 本课自测

不要运行 Python，先在纸上完成。

## Q1

```python
x = torch.randn(4, 256, 80, 80)
y = x.flatten(2)
```

`y.shape = ?`

---

## Q2

接着：

```python
z = y.transpose(1, 2)
```

`z.shape = ?`

---

## Q3

```python
x = torch.randn(2, 1600, 768)

x = x.reshape(
    2,
    1600,
    3,
    8,
    32
)
```

为什么这个 reshape 合法？

---

## Q4

```text
Q = [2,8,300,32]
K = [2,8,300,32]
```

`Q @ K.transpose(-2,-1)` 的 shape 是什么？

---

## Q5

```text
P3 = [2,6400,256]
P4 = [2,1600,256]
P5 = [2,400,256]
```

沿 dim=1 cat 后是什么？

---

## Q6

三个：

```text
[2,300,256]
```

Tensor 沿新的 dim=0 stack 后是什么？

---

## Q7

```text
A = [2,300,1,2]
B = [2,1,3,2]
```

`A * B` 的 shape 是什么？

---

## Q8

为什么：

```python
x.permute(...).view(...)
```

有时会报错？

---

## Q9

`expand()` 与 `repeat()` 最重要的区别是什么？

---

## Q10

下面两个数字的语义为什么不同？

```text
[B,8400,256]

[B,300,256]
```

---

# 50. 课后作业 A：手算 320 输入

假设输入：

```text
[B,3,320,320]
```

三个 feature strides：

```text
8
16
32
```

请算：

```text
P3 spatial shape
P4 spatial shape
P5 spatial shape
```

以及 flatten 后：

```text
N3
N4
N5
N_total
```

然后与 640 输入的：

```text
8400
```

比较。

你应该发现：

```text
输入边长缩小 1/2
```

空间 token 总量大约缩小为：

```text
1/4
```

原因：

\[ (H/2)(W/2)=HW/4 \]

这会直接影响后面计算量。

---

# 51. 课后作业 B：写一个通用 Feature Flattener

目标：

输入：

```python
features = [
    torch.randn(B, 256, 80, 80),
    torch.randn(B, 256, 40, 40),
    torch.randn(B, 256, 20, 20),
]
```

写函数：

```python
def flatten_features(features):
    ...
```

返回：

```text
memory:
[B,8400,256]

spatial_shapes:
[[80,80],
 [40,40],
 [20,20]]
```

进一步计算：

```text
level_start_index
```

应该是：

```text
[0,6400,8000]
```

因为：

```text
P3 starts at 0
P4 starts after 6400
P5 starts after 6400+1600 = 8000
```

这已经非常接近后面 Deformable Attention 的真实输入组织。

---

# 52. 课后作业 C：手写 QKV Shape Pipeline

禁止直接复制本课代码。

自己写：

```python
def make_qkv(x, num_heads):
    ...
    return q, k, v
```

输入：

```text
[B,N,C]
```

输出必须：

```text
[B,h,N,d]
```

并加入：

```python
assert C % num_heads == 0
```

然后测试：

```text
B=2
N=300
C=256
h=8
```

要求：

```text
Q/K/V
=
[2,8,300,32]
```

---

# 53. 课后作业 D：自己解释官方 DEIMv2 PatchEmbed

看到：

```python
def forward(self, x):
    return self.proj(x).flatten(2).transpose(1, 2)
```

请不要说：

> "先卷积，再 flatten，再 transpose。"

而要完整解释：

```text
输入语义是什么？
self.proj 为什么可以实现 patch embedding？
640×640 在 patch=16 后为什么是 40×40？
为什么有 1600 个 patch tokens？
flatten(2) 到底合并哪两个维度？
transpose(1,2) 为什么必要？
最终 [B,1600,C] 每个维度是什么意思？
```

能完整回答，才算真正读懂这一行。

---

# 54. 本课最终 Cheat Sheet

```text
reshape
改变 shape，元素总数不变

view
类似 reshape，但更依赖内存布局

flatten
把连续多个维度合并

transpose(a,b)
交换两个维度

permute(...)
任意重新排列所有维度

unsqueeze(dim)
插入一个 size=1 的维度

squeeze(dim)
删除一个 size=1 的维度

cat
沿已有维度拼接

stack
增加新维度后拼接

split
按指定大小拆分

chunk
近似平均拆分

unbind
沿某维拆开，并删除该维

expand
利用广播式视图扩展 size=1 维度

repeat
真正重复数据

contiguous
获得连续内存布局

shape
Tensor 各维大小

stride
Tensor 各维在 storage 中的步进方式

dtype
数据类型

device
CPU / GPU 等设备
```

---

# 55. 把本课重新连接回 DEIMv2

现在回头看整个模型。

```text
Image
[B,3,640,640]

↓ Backbone

multi-scale feature maps
[B,C,H,W]

↓ flatten + transpose

multi-scale tokens
[B,N,C]

↓ cat

memory
[B,N_total,C]

↓ object queries

[B,Nq,C]

↓ attention reshape

[B,h,Nq,d]
[B,h,Nfeature,d]

↓ decoder

[B,Nq,C]

↓ prediction heads

class / localization
```

所以第 2 课不是一节独立的 PyTorch 基础课。

它实际上是在学习：

> **DEIMv2 源码所使用的"张量语言"。**

以后看到：

```python
reshape
permute
flatten
cat
unsqueeze
expand
```

我们就不再停下来解释最基础的 shape 规则，而能把注意力放到：

```text
为什么要这样组织信息？
这一层在检测算法上解决什么问题？
```

---

# 56. 下一课预告

## 第 3 课：CNN、卷积、Stride 与多尺度特征

下一课会正式回答：

> 为什么一张 `[B,3,640,640]` 图片经过 Backbone 后，会出现
> `80×80 / 40×40 / 20×20` 这样的 feature maps？

会从零推导：

\[ H\_{out} = `\left`{=tex}`\lfloor`{=tex}
`\frac{H+2P-D(K-1)-1}{S}`{=tex}+1 `\right`{=tex}`\rfloor`{=tex} \]

并亲手计算：

```text
kernel size
stride
padding
dilation
channels
parameters
FLOPs
receptive field
```

随后进入真实 HGNetv2 结构，为后面完整拆 `deim_dfine_x` 做准备。

---

# 附录：本课使用的官方源码位置

主要核对：

```text
DEIMv2/
└── engine/
    └── backbone/
        └── vit_tiny.py
```

其中本课重点观察：

```text
PatchEmbed.forward
Attention.forward
VisionTransformer.forward
RopePositionEmbedding.forward
```

官方仓库：

https://github.com/Intellindust-AI-Lab/DEIMv2

本课对应源码：

https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/engine/backbone/vit_tiny.py

后续正式进入 DINOv3/STA、HybridEncoder、DEIMTransformer
时，会继续以官方当前源码为准核对，而不是只根据通用 ViT/DETR
教程推测实现。

---

**第 2 课结束。**

这一课真正的达标标准不是记住 API，而是：

> 给你一段 DEIMv2 Tensor
> 代码，你能够不运行程序，先在纸上准确写出每一步的
> shape，并说明每个维度代表什么。

如果做到这一点，后面的 CNN、Attention、DETR、D-FINE、STA 和 Deformable
Attention 会明显容易很多。
