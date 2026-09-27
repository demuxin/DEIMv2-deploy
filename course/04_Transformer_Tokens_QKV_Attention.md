# 第 4 课：Transformer 从零------Token、Q/K/V 与 Attention

> **课程定位**：DEIMv2 从零到源码 · 第 4 / 24 课\
> **本课目标**：从零理解 Token、Embedding、Q/K/V、Scaled Dot-Product
> Attention、Multi-Head
> Attention、Residual、LayerNorm、FFN，并能够独立推导 DEIMv2 中 ViT
> Attention 的每一步 Tensor shape。\
> **重点连接**：DINOv3/ViT Backbone、DETR/DEIM Transformer
> Decoder、Deformable Attention 的共同基础。\
> **建议方式**：每看到一个矩阵乘法，先在纸上写 shape，再看公式；所有
> PyTorch 实验尽量亲手运行。

------------------------------------------------------------------------

# 0. 我们现在已经有了两块基础

第 2 课学会：

``` text
[B,C,H,W]
↔
[B,N,C]
```

第 3 课学会：

``` text
Image
→ CNN
→ multi-scale feature maps
```

现在开始学习 DEIMv2 的另一块核心积木：

``` text
Transformer
```

DEIMv2 中至少有两种非常重要的 Transformer 角色：

``` text
DINOv3 / ViT
→ 用 Transformer 理解图片

DEIM / D-FINE Decoder
→ 用 Transformer/attention 让 object queries 从图片特征中寻找目标
```

所以本课不是只为 DINOv3 服务。

它也是后面：

``` text
DETR
Deformable Attention
DFINETransformer
DEIMTransformer
```

的数学基础。

------------------------------------------------------------------------

# 1. Transformer 到底解决什么问题？

假设我们已经有一组 tokens：

``` text
Token 1
Token 2
Token 3
...
Token N
```

每个 token 是一个向量：

``` text
C dimensions
```

于是：

``` text
X = [N,C]
```

加入 batch：

``` text
X = [B,N,C]
```

Transformer 最核心的问题是：

> **一个 token 应该从其他哪些 token 获取信息？获取多少？**

Attention 就是用数据本身动态计算这种关系。

------------------------------------------------------------------------

# 2. Token 是什么？

Token 不是 Transformer 专属的"神秘对象"。

它本质就是：

> 一个向量形式的信息单元。

在 NLP：

``` text
一个词 / 子词
→ 一个 token
```

在 ViT：

``` text
一个 image patch
→ 一个 token
```

在 DETR：

``` text
一个 object query
→ 也可以看成一个 token-like representation
```

在多尺度检测特征：

``` text
feature map 上一个空间位置
→ flatten 后成为一个 feature token
```

所以：

``` text
Token
=
一个 C 维向量
```

------------------------------------------------------------------------

# 3. 从图片变成 ViT Tokens

假设：

``` text
Image:
[B,3,640,640]
```

patch size：

``` text
16×16
```

那么每个方向：

\[ 640/16=40 \]

patch 数：

\[ 40`\times40`{=tex}=1600 \]

所以：

``` text
1600 image patches
```

如果每个 patch 映射到：

``` text
C = 256
```

维：

``` text
[B,1600,256]
```

这就是 Transformer 可以处理的 token sequence。

第 2 课已经见过真实源码模式：

``` python
x = self.proj(x).flatten(2).transpose(1, 2)
```

其 shape：

``` text
[B,3,640,640]
→
[B,256,40,40]
→
[B,256,1600]
→
[B,1600,256]
```

------------------------------------------------------------------------

# 4. Embedding Dimension 是什么？

假设：

``` text
[B,1600,256]
```

其中：

``` text
1600 = token 数
256  = 每个 token 的 feature dimension
```

每个 token：

``` text
x_i ∈ R^256
```

可以想象：

``` text
token 1:
[0.12, -0.8, ..., 0.34]

token 2:
[-0.2, 0.91, ..., 0.03]

...
```

这些数字不是人工定义的"颜色/边缘标签"。

它们是神经网络学习出的 feature representation。

------------------------------------------------------------------------

# 5. 为什么需要 Attention？

假设图片里：

``` text
一个人的头
一个人的身体
一辆车
背景天空
```

一个 patch 只看自己，很难完整理解：

> "这个局部属于一个人。"

它需要和其他区域交换信息。

Attention 的目标：

``` text
当前 token
   │
   ├── 应该关注 token 2 多少？
   ├── 应该关注 token 3 多少？
   ├── 应该关注 token 4 多少？
   └── ...
```

最后把相关 token 的信息加权汇总。

------------------------------------------------------------------------

# 6. Query、Key、Value 的直觉

Attention 为每个 token 生成三个向量：

``` text
Query
Key
Value
```

最常用的直觉：

``` text
Query:
“我正在寻找什么？”

Key:
“我这里有什么，可以被谁匹配？”

Value:
“如果你关注我，我实际提供什么信息？”
```

不要把这当作严格数学定义。

真正数学定义是：

\[ Q=XW_Q \]

\[ K=XW_K \]

\[ V=XW_V \]

也就是说：

> Q、K、V 都是同一个输入 X 经过不同可学习线性投影得到的表示。

------------------------------------------------------------------------

# 7. 从一个 Token 推导 Q/K/V

假设一个 token：

\[ x`\in`{=tex}`\mathbb{R}`{=tex}\^{256} \]

定义：

\[ W_Q`\in`{=tex}`\mathbb{R}`{=tex}\^{256`\times256`{=tex}} \]

则：

\[ q=xW_Q \]

得到：

\[ q`\in`{=tex}`\mathbb{R}`{=tex}\^{256} \]

同理：

\[ k=xW_K \]

\[ v=xW_V \]

所以：

``` text
x
[256]
 │
 ├── WQ → q [256]
 ├── WK → k [256]
 └── WV → v [256]
```

------------------------------------------------------------------------

# 8. 一批 Tokens 的 Q/K/V

输入：

``` text
X:
[B,N,C]
```

假设：

``` text
B = 2
N = 1600
C = 256
```

则：

``` text
X:
[2,1600,256]
```

经过三个 Linear：

``` text
Q:
[2,1600,256]

K:
[2,1600,256]

V:
[2,1600,256]
```

shape 不一定必须保持 256，但经典 self-attention 中通常总 hidden
dimension 保持一致。

------------------------------------------------------------------------

# 9. 为什么 Q 和 K 要做点积？

假设：

``` text
q_i
```

是 token i 的 Query，

``` text
k_j
```

是 token j 的 Key。

计算：

\[ score\_{ij}=q_i`\cdot `{=tex}k_j \]

也就是：

\[ score\_{ij} = `\sum`{=tex}*{d=1}\^{C} q*{i,d}k\_{j,d} \]

得到一个标量。

直觉：

``` text
score 高
→ token i 的 Query 与 token j 的 Key 更匹配

score 低
→ 相关性较低
```

这是模型学习出来的关系，不是人工规定的相似度规则。

------------------------------------------------------------------------

# 10. 一个极小的 Attention 例子

假设只有 3 个 tokens，每个 Q/K 是二维：

\[ Q=
```{=tex}
\begin{bmatrix}
1&0\\
0&1\\
1&1
\end{bmatrix}
```
\]

\[ K=
```{=tex}
\begin{bmatrix}
1&0\\
0&1\\
1&1
\end{bmatrix}
```
\]

计算：

\[ QK\^T \]

先：

\[ K\^T=
```{=tex}
\begin{bmatrix}
1&0&1\\
0&1&1
\end{bmatrix}
```
\]

所以：

\[ QK\^T =
```{=tex}
\begin{bmatrix}
1&0&1\\
0&1&1\\
1&1&2
\end{bmatrix}
```
\]

shape：

``` text
Q  = [3,2]
Kᵀ = [2,3]

QKᵀ = [3,3]
```

每一行表示：

> 一个 Query 对所有 Keys 的匹配分数。

------------------------------------------------------------------------

# 11. 为什么 Attention Matrix 是 N×N？

一般：

``` text
Q = [N,d]
K = [N,d]
```

则：

``` text
Kᵀ = [d,N]
```

矩阵乘法：

\[ \[N,d\]`\times[d,N]`{=tex}= \[N,N\] \]

所以 self-attention：

``` text
N 个 query tokens
×
N 个 key tokens
```

产生：

``` text
N×N
```

关系矩阵。

加 batch：

``` text
[B,N,N]
```

加 multi-head：

``` text
[B,h,N,N]
```

------------------------------------------------------------------------

# 12. 为什么除以 (`\sqrt{d_k}`{=tex})？

标准 Scaled Dot-Product Attention：

\[ Attention(Q,K,V) = softmax `\left`{=tex}(
`\frac{QK^T}{\sqrt{d_k}}`{=tex} `\right`{=tex})V \]

为什么不是：

\[ softmax(QK\^T)V \]

？

假设 q/k 每个分量：

``` text
mean ≈ 0
variance ≈ 1
```

点积：

\[ q`\cdot `{=tex}k = `\sum`{=tex}\_{i=1}\^{d_k}q_i k_i \]

当：

``` text
d_k
```

变大时，点积的方差也会随维度增长。

结果：

``` text
logits 绝对值容易变大
→ softmax 更容易饱和
→ 梯度可能变得很小
```

除以：

\[ `\sqrt{d_k}`{=tex} \]

可以把尺度控制得更稳定。

------------------------------------------------------------------------

# 13. Softmax 在 Attention 中做什么？

假设某个 query 对 3 个 keys 的分数：

``` text
[2.0, 1.0, 0.0]
```

softmax：

\[ softmax(z_i) = `\frac{e^{z_i}}`{=tex} {`\sum`{=tex}\_j e\^{z_j}} \]

得到大约：

``` text
[0.665, 0.245, 0.090]
```

满足：

\[ 0.665+0.245+0.090=1 \]

所以可以理解为：

``` text
对 key 1：
关注 66.5%

对 key 2：
关注 24.5%

对 key 3：
关注 9.0%
```

这些是对某个 query 的归一化 attention weights。

------------------------------------------------------------------------

# 14. Attention Weight 为什么乘 V？

假设：

``` text
attention weights:
[0.665,0.245,0.090]
```

对应 Value：

\[ v_1,v_2,v_3 \]

输出：

\[ y = 0.665v_1 + 0.245v_2 + 0.090v_3 \]

也就是说：

> Query/Key 决定"关注谁"，Value 决定"真正拿走什么信息"。

这就是 Q/K/V 分工最重要的理解。

------------------------------------------------------------------------

# 15. 完整单头 Attention

输入：

``` text
X:
[B,N,C]
```

先：

\[ Q=XW_Q \]

\[ K=XW_K \]

\[ V=XW_V \]

假设仍：

``` text
Q/K/V:
[B,N,C]
```

计算：

``` text
K.transpose(-2,-1):
[B,C,N]
```

所以：

``` text
Q @ Kᵀ:
[B,N,N]
```

scale：

``` text
[B,N,N]
```

softmax：

``` text
[B,N,N]
```

再：

``` text
AttentionWeights @ V
```

即：

``` text
[B,N,N]
@
[B,N,C]
```

得到：

``` text
[B,N,C]
```

因此：

\[ `\boxed{
[B,N,C]
\rightarrow
Attention
\rightarrow
[B,N,C]
}`{=tex} \]

------------------------------------------------------------------------

# 16. 第一个 PyTorch 实验：手写单头 Attention

``` python
import math
import torch

B = 2
N = 4
C = 8

q = torch.randn(B, N, C)
k = torch.randn(B, N, C)
v = torch.randn(B, N, C)

scores = q @ k.transpose(-2, -1)

print(
    "scores:",
    scores.shape
)

scores = scores / math.sqrt(C)

weights = torch.softmax(
    scores,
    dim=-1
)

print(
    "weights:",
    weights.shape
)

out = weights @ v

print(
    "output:",
    out.shape
)
```

预期：

``` text
scores:
[2,4,4]

weights:
[2,4,4]

output:
[2,4,8]
```

检查：

``` python
print(
    weights.sum(dim=-1)
)
```

应该接近：

``` text
1
```

------------------------------------------------------------------------

# 17. 为什么需要 Multi-Head Attention？

如果只用一个 attention：

``` text
每个 token 只有一套 Q/K/V 关系空间
```

Multi-Head 的思想：

> 把 hidden dimension 拆成多个子空间，让不同 head 可以学习不同关系。

例如：

``` text
C = 256
num_heads = 8
```

则：

\[ d=256/8=32 \]

所以：

``` text
原：
[B,N,256]

拆成：
[B,8,N,32]
```

8 个 head 各自做 attention。

------------------------------------------------------------------------

# 18. Multi-Head 不是复制 8 份完整 256 维 Attention

这是常见误解。

典型做法：

``` text
总 hidden dim = 256

8 heads

每个 head dim = 32
```

因为：

\[ 8`\times32`{=tex}=256 \]

所以只是把：

``` text
256-dimensional representation
```

重新组织为：

``` text
8 × 32
```

并让每个 head 独立计算 attention。

------------------------------------------------------------------------

# 19. QKV 一次 Linear 为什么输出 3C？

源码常见：

``` python
self.qkv = nn.Linear(
    C,
    C * 3
)
```

输入：

``` text
[B,N,C]
```

输出：

``` text
[B,N,3C]
```

例如：

``` text
C=256
```

则：

``` text
[B,N,768]
```

然后 reshape：

``` text
[B,N,3,h,d]
```

因为：

\[ 3hd=3C \]

这比写三个独立 Linear 在代码组织上更紧凑。

------------------------------------------------------------------------

# 20. DEIMv2 风格 QKV Shape 完整推导

假设：

``` text
B = 2
N = 1600
C = 256
h = 8
d = 32
```

输入：

``` text
x:
[2,1600,256]
```

QKV Linear：

``` text
[2,1600,768]
```

reshape：

``` text
[2,1600,3,8,32]
```

permute：

``` text
[3,2,8,1600,32]
```

unbind：

``` text
Q:
[2,8,1600,32]

K:
[2,8,1600,32]

V:
[2,8,1600,32]
```

这正是第 2 课已经学过的 shape 操作，现在终于知道为什么这样做。

------------------------------------------------------------------------

# 21. 每个 Head 的 Attention Matrix

Q：

``` text
[B,h,N,d]
```

K：

``` text
[B,h,N,d]
```

transpose K：

``` text
[B,h,d,N]
```

所以：

``` text
Q @ Kᵀ
```

得到：

``` text
[B,h,N,N]
```

例如：

``` text
[2,8,1600,1600]
```

这意味着：

``` text
2 batches
8 heads
每个 head
1600 queries × 1600 keys
```

------------------------------------------------------------------------

# 22. 1600×1600 到底有多大？

每个 head：

\[ 1600\^2 = 2,560,000 \]

8 heads：

\[ 20,480,000 \]

batch=2：

\[ 40,960,000 \]

仅 attention score 就有约：

``` text
4096 万个元素
```

如果 float32：

``` text
4 bytes / element
```

仅这一矩阵理论原始存储规模约：

\[ 40,960,000`\times4`{=tex} `\approx
163.84`{=tex} MB \]

训练还需要中间 activation/gradient，因此实际内存压力更复杂。

这就是为什么标准全局 attention 的：

\[ O(N\^2) \]

复杂度如此重要。

------------------------------------------------------------------------

# 23. 这为什么与 Deformable Attention 有关？

以后检测特征可能不是：

``` text
1600 tokens
```

而是：

``` text
80×80 + 40×40 + 20×20
=
8400
```

如果对 8400 tokens 做完整：

\[ 8400\^2 = 70,560,000 \]

每个 head 就超过 7000 万个 pairwise scores。

Deformable Attention 的核心动机之一就是：

> 不需要每个 query 都和所有空间位置做完整
> attention，而是只在少量关键采样位置取信息。

所以第 7 课学习 Deformable Attention 时，本课的：

``` text
N×N full attention
```

就是最重要的对照基线。

------------------------------------------------------------------------

# 24. Multi-Head Attention 的输出怎样合回来？

每个 head 输出：

``` text
[B,h,N,d]
```

例如：

``` text
[2,8,1600,32]
```

先：

``` python
x = x.transpose(1, 2)
```

得到：

``` text
[B,N,h,d]
```

即：

``` text
[2,1600,8,32]
```

再 reshape：

``` python
x = x.reshape(B, N, C)
```

得到：

``` text
[2,1600,256]
```

因为：

\[ 8`\times32`{=tex}=256 \]

最后通常还有：

``` python
self.proj(x)
```

即 output projection。

------------------------------------------------------------------------

# 25. 完整 Multi-Head Attention 公式

可以写成：

\[ head_i = Attention( QW_i\^Q, KW_i\^K, VW_i\^V ) \]

然后：

\[ MultiHead(Q,K,V) = Concat( head_1,`\ldots`{=tex},head_h )W\^O \]

其中：

``` text
W_i^Q
W_i^K
W_i^V
```

是每个 head 对应的投影子空间，

``` text
W^O
```

是最终 output projection。

实际实现常把所有 heads 的 Q/K/V projection 合并成大矩阵一次完成。

------------------------------------------------------------------------

# 26. 第二个 PyTorch 实验：手写 Multi-Head Attention

``` python
import math
import torch
import torch.nn as nn

class TinyMultiHeadAttention(nn.Module):

    def __init__(
        self,
        dim=256,
        num_heads=8
    ):
        super().__init__()

        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = (
            dim // num_heads
        )

        self.qkv = nn.Linear(
            dim,
            dim * 3
        )

        self.proj = nn.Linear(
            dim,
            dim
        )

    def forward(self, x):

        B, N, C = x.shape

        qkv = self.qkv(x)

        qkv = qkv.reshape(
            B,
            N,
            3,
            self.num_heads,
            self.head_dim
        )

        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )

        q, k, v = qkv.unbind(0)

        scores = (
            q @ k.transpose(-2, -1)
        )

        scores = (
            scores
            / math.sqrt(self.head_dim)
        )

        attn = torch.softmax(
            scores,
            dim=-1
        )

        out = attn @ v

        out = out.transpose(
            1, 2
        ).reshape(
            B, N, C
        )

        out = self.proj(out)

        return out
```

测试：

``` python
x = torch.randn(
    2, 300, 256
)

attn = TinyMultiHeadAttention(
    dim=256,
    num_heads=8
)

y = attn(x)

print(x.shape)
print(y.shape)
```

应该：

``` text
[2,300,256]
[2,300,256]
```

------------------------------------------------------------------------

# 27. 为什么输入输出 Shape 一样很重要？

Transformer block 常使用 residual connection：

``` python
x = x + attention(x)
```

如果：

``` text
x:
[B,N,C]
```

那么：

``` text
attention(x)
```

也必须：

``` text
[B,N,C]
```

才能逐元素相加。

所以 Multi-Head Attention 最后要重新合并 heads，并通常投影回原 hidden
dimension。

------------------------------------------------------------------------

# 28. Residual Connection 是什么？

最简单：

\[ y=x+F(x) \]

其中：

``` text
F(x)
```

可以是 attention 或 FFN。

例如：

``` python
x = x + self.attn(x)
```

直觉：

> 新层不是完全重写 x，而是在原信息基础上学习一个更新量。

这有助于深层网络的信息和梯度传播。

------------------------------------------------------------------------

# 29. Residual 的 Shape 条件

如果：

``` text
x:
[B,N,256]
```

那么：

``` text
F(x)
```

也必须是：

``` text
[B,N,256]
```

才能：

``` python
x + F(x)
```

这就是为什么很多 Transformer 子层最终都保持 hidden dimension。

------------------------------------------------------------------------

# 30. LayerNorm 是什么？

Transformer 中常见：

``` python
nn.LayerNorm(C)
```

假设：

``` text
x:
[B,N,C]
```

LayerNorm 通常沿最后一个 feature dimension 归一化。

对一个 token：

\[ x_i`\in`{=tex}`\mathbb{R}`{=tex}\^{C} \]

计算：

\[ `\mu`{=tex} = `\frac{1}{C}`{=tex} `\sum`{=tex}\_{j=1}\^{C}x_j \]

\[ `\sigma`{=tex}\^2 = `\frac{1}{C}`{=tex} `\sum`{=tex}\_{j=1}\^{C}
(x_j-`\mu`{=tex})\^2 \]

归一化：

\[ `\hat{x}`{=tex}\_j = `\frac{x_j-\mu}`{=tex}
{`\sqrt{\sigma^2+\epsilon}`{=tex}} \]

再学习：

\[ y_j = `\gamma`{=tex}\_j`\hat{x}`{=tex}\_j+`\beta`{=tex}\_j \]

其中：

``` text
γ
β
```

是可学习参数。

------------------------------------------------------------------------

# 31. LayerNorm 与 BatchNorm 的直觉区别

CNN 中常见：

``` text
BatchNorm
```

Transformer 中常见：

``` text
LayerNorm
```

非常粗略地理解：

``` text
BatchNorm
→ 统计方式与 batch/channel/spatial 结构有关

LayerNorm
→ 对单个 token 的 feature dimension 做归一化
```

对于：

``` text
[B,N,C]
```

LayerNorm(C) 不需要依赖其他 batch 样本才能定义每个 token 的归一化。

------------------------------------------------------------------------

# 32. Pre-Norm 与 Post-Norm

Transformer 有两种常见组织。

Post-Norm：

``` text
x
│
├─────── residual ──────┐
│                       │
▼                       │
Attention               │
│                       │
└──────── + ────────────┘
          │
          ▼
       LayerNorm
```

形式：

\[ x'=LN(x+Attn(x)) \]

Pre-Norm：

``` text
x
│
├───────────────────────┐
▼                       │
LayerNorm               │
▼                       │
Attention               │
│                       │
└──────── + ────────────┘
```

形式：

\[ x'=x+Attn(LN(x)) \]

不同实现可能不同。

所以以后读源码不能只说：

> "Transformer 都是 Attention + Norm。"

必须看：

``` text
Norm 在 Attention 前还是后？
```

------------------------------------------------------------------------

# 33. FFN 是什么？

Attention 负责：

> token 之间交换信息。

FFN 负责：

> 对每个 token 自己的 feature dimension 做非线性变换。

典型：

\[ FFN(x) = W_2`\sigma`{=tex}(W_1x+b_1)+b_2 \]

例如：

``` text
hidden dim = 256
FFN dim = 1024
```

shape：

``` text
[B,N,256]
↓ Linear
[B,N,1024]
↓ GELU/ReLU
[B,N,1024]
↓ Linear
[B,N,256]
```

所以输出仍：

``` text
[B,N,256]
```

可以 residual。

------------------------------------------------------------------------

# 34. Attention 与 FFN 的分工

非常重要：

``` text
Attention
→ token mixing

FFN
→ channel/feature transformation
```

Attention：

``` text
token 1 可以读取 token 2/3/4 的信息
```

FFN：

``` text
对每一个 token 的 C 维 feature 独立做相同 MLP
```

这两个模块组合形成 Transformer Block 的核心。

------------------------------------------------------------------------

# 35. 一个标准 Transformer Block

简化 Pre-Norm：

``` python
def forward(x):

    x = x + attn(
        norm1(x)
    )

    x = x + ffn(
        norm2(x)
    )

    return x
```

shape：

``` text
x
[B,N,C]

↓ norm
[B,N,C]

↓ attention
[B,N,C]

↓ residual
[B,N,C]

↓ norm
[B,N,C]

↓ FFN
[B,N,C]

↓ residual
[B,N,C]
```

整个 block 不改变 shape。

但 feature 内容已经发生复杂变化。

------------------------------------------------------------------------

# 36. 第三个 PyTorch 实验：手写 Transformer Block

``` python
import torch
import torch.nn as nn

class TinyTransformerBlock(nn.Module):

    def __init__(
        self,
        dim=256,
        num_heads=8,
        mlp_ratio=4
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.attn = (
            TinyMultiHeadAttention(
                dim=dim,
                num_heads=num_heads
            )
        )

        self.norm2 = nn.LayerNorm(dim)

        hidden = dim * mlp_ratio

        self.ffn = nn.Sequential(
            nn.Linear(
                dim,
                hidden
            ),
            nn.GELU(),
            nn.Linear(
                hidden,
                dim
            )
        )

    def forward(self, x):

        x = x + self.attn(
            self.norm1(x)
        )

        x = x + self.ffn(
            self.norm2(x)
        )

        return x
```

测试：

``` python
x = torch.randn(
    2, 300, 256
)

block = TinyTransformerBlock()

y = block(x)

print(x.shape)
print(y.shape)
```

都是：

``` text
[2,300,256]
```

------------------------------------------------------------------------

# 37. Self-Attention 是什么？

如果：

``` text
Q
K
V
```

都来自同一个 sequence：

``` text
X
```

则称：

``` text
Self-Attention
```

即：

\[ Q=XW_Q \]

\[ K=XW_K \]

\[ V=XW_V \]

一个 token 可以从同一序列其他 token 获取信息。

ViT 中典型就是 self-attention。

------------------------------------------------------------------------

# 38. Cross-Attention 是什么？

如果：

``` text
Q
```

来自一个序列，

``` text
K/V
```

来自另一个序列，

就是 cross-attention。

例如 DETR Decoder：

``` text
Object Queries
→ Q

Image Features
→ K,V
```

于是：

``` text
Query:
“我这个 object query 想在图里找什么？”

Image Key:
“这个图像位置有什么可匹配特征？”

Image Value:
“如果关注这个位置，我提供什么视觉信息？”
```

这是理解 DETR 的关键。

------------------------------------------------------------------------

# 39. Self-Attention vs Cross-Attention Shape

Self-Attention：

``` text
X:
[B,N,C]

Q/K/V:
[B,h,N,d]

Attention:
[B,h,N,N]
```

Cross-Attention：

``` text
Query sequence:
[B,Nq,C]

Image sequence:
[B,Nk,C]
```

则：

``` text
Q:
[B,h,Nq,d]

K:
[B,h,Nk,d]

V:
[B,h,Nk,d]
```

Attention matrix：

``` text
[B,h,Nq,Nk]
```

注意：

> Cross-attention 不要求 query token 数和 key token 数相同。

------------------------------------------------------------------------

# 40. DETR 风格例子

假设：

``` text
object queries:
Nq = 300

image features:
Nk = 8400

heads = 8
head_dim = 32
```

则：

``` text
Q:
[B,8,300,32]

K:
[B,8,8400,32]

V:
[B,8,8400,32]
```

QKᵀ：

``` text
[B,8,300,8400]
```

这意味着：

> 每个 object query 都可以与大量 image positions 建立关系。

后面 Deformable Attention 会大幅减少实际采样位置。

------------------------------------------------------------------------

# 41. 为什么 DETR Decoder 还需要 Query Self-Attention？

Decoder 中常见：

``` text
① query self-attention
② query-image cross-attention
③ FFN
```

Self-attention：

``` text
300 queries
↔
300 queries
```

让不同 queries 之间交换信息。

Cross-attention：

``` text
300 queries
↔
image features
```

让 query 从图像读取视觉信息。

所以两种 attention 的角色不同。

------------------------------------------------------------------------

# 42. Position 信息为什么重要？

Attention 本身如果只看 token feature，没有额外位置机制，它并不知道：

``` text
token 17 在图片左上角

token 900 在图片右下角
```

图像任务对空间位置极其重要。

所以 Transformer 需要位置相关信息，例如：

``` text
absolute position embedding
relative position
sin/cos encoding
RoPE
reference points
```

DINOv3/DEIMv2 中会遇到位置编码与 RoPE 相关实现。

本课先记：

> **Attention 负责内容关系，位置机制负责告诉模型 token
> 在哪里/彼此空间关系如何。**

------------------------------------------------------------------------

# 43. CLS Token 是什么？

ViT 中常见一个额外 token：

``` text
[CLS]
```

如果 patch tokens：

``` text
1600
```

加入 CLS：

``` text
1601 tokens
```

shape：

``` text
[B,1601,C]
```

CLS token 不对应一个具体 patch。

它通常作为全局表示载体。

源码里可能看到：

``` python
x[:, 0]
```

代表：

``` text
CLS token
```

而：

``` python
x[:, 1:]
```

代表：

``` text
patch tokens
```

第 2 课已经见过这种 indexing。

------------------------------------------------------------------------

# 44. Dropout / DropPath 是什么角色？

Transformer 中常见：

``` text
Dropout
DropPath / Stochastic Depth
```

它们主要属于正则化。

Dropout：

``` text
随机把部分 activation 置零
```

DropPath：

``` text
训练时随机跳过某些 residual branch
```

例如：

``` text
x = x + drop_path(attn(...))
```

推理时通常关闭随机丢弃行为。

后面读 DINOv3 block 时会再次遇到。

------------------------------------------------------------------------

# 45. Attention Mask 是什么？

有时不希望某些 query 看某些 key。

可以对 attention logits 加 mask。

例如：

``` text
允许：
score

禁止：
-∞
```

softmax 后：

\[ e\^{-`\infty`{=tex}}=0 \]

所以被 mask 的位置权重为 0。

在不同 Transformer 任务中 mask 用途不同。

检测 Decoder 中还会遇到 denoising queries 与 attention mask 的设计。

------------------------------------------------------------------------

# 46. `scaled_dot_product_attention` 是什么？

现代 PyTorch 提供：

``` python
torch.nn.functional.scaled_dot_product_attention
```

可以直接完成：

``` text
QKᵀ
scale
mask
softmax
dropout
×V
```

并可能根据设备/条件选择更高效实现。

因此真实源码未必显式写：

``` python
scores = q @ k.transpose(...)
scores /= sqrt(d)
scores = softmax(scores)
out = scores @ v
```

但数学本质仍然是 Scaled Dot-Product Attention。

------------------------------------------------------------------------

# 47. Flash Attention 为什么会出现？

标准 attention 如果显式存：

``` text
[B,h,N,N]
```

内存开销很大。

Flash Attention 类方法通过更聪明的分块和内存访问方式，避免把完整中间
attention matrix 以朴素方式写入显存，从而：

``` text
减少显存访问
提高速度
```

但：

> 它没有改变 attention 的核心数学目标。

所以先学懂本课公式，再看 Flash Attention 才不会把"高效实现"与"Attention
定义"混淆。

------------------------------------------------------------------------

# 48. 为什么 Self-Attention 是 (O(N\^2))？

因为：

``` text
N queries
```

都要与：

``` text
N keys
```

计算关系。

pair 数：

\[ N`\times `{=tex}N=N\^2 \]

所以：

``` text
N 翻倍
→ pair 数约变 4 倍
```

这和第 3 课中：

``` text
图像边长减半
→ 空间 token 数约变 1/4
```

可以连接起来。

------------------------------------------------------------------------

# 49. 图像分辨率与 Attention 复杂度的危险关系

patch size 固定为 16。

图片：

``` text
640×640
```

patch tokens：

\[ 40\^2=1600 \]

如果图片边长翻倍：

``` text
1280×1280
```

tokens：

\[ 80\^2=6400 \]

token 数：

``` text
4×
```

full attention pair 数：

\[ 1600\^2 `\rightarrow
6400`{=tex}\^2 \]

变成：

``` text
16×
```

所以高分辨率视觉 Transformer 的 attention 成本非常敏感。

------------------------------------------------------------------------

# 50. 参数量：QKV Linear 有多少参数？

假设：

``` text
C = 256
```

QKV：

``` python
nn.Linear(
    256,
    768
)
```

不算 bias：

\[ 256`\times768`{=tex} = 196,608 \]

output projection：

``` python
nn.Linear(
    256,
    256
)
```

参数：

\[ 256\^2 = 65,536 \]

所以 attention projection 总参数约：

\[ 262,144 \]

不含 bias。

注意：

> Attention 计算量随 token 数 N 强烈变化，但这些 Linear 的参数量主要由
> hidden dimension C 决定。

------------------------------------------------------------------------

# 51. FFN 往往也很占参数

假设：

``` text
C = 256
FFN hidden = 1024
```

第一层：

\[ 256`\times1024`{=tex} = 262,144 \]

第二层：

\[ 1024`\times256`{=tex} = 262,144 \]

合计：

\[ 524,288 \]

不含 bias。

所以一个 Transformer block 中：

> FFN 参数量往往并不少，甚至比 attention projection 更多。

------------------------------------------------------------------------

# 52. 一个 Transformer Block 的信息流

把所有东西放在一起：

``` text
Input tokens
[B,N,C]
    │
    ▼
LayerNorm
    │
    ▼
Q / K / V
    │
    ▼
Multi-Head Self-Attention
    │
    ▼
[B,N,C]
    │
    ├──────── residual from input
    ▼
Add
    │
    ▼
LayerNorm
    │
    ▼
FFN
C → hidden → C
    │
    ├──────── residual
    ▼
Add
    │
    ▼
Output tokens
[B,N,C]
```

一个 block 不改变 token 数和 hidden dimension，但让 feature 更强。

------------------------------------------------------------------------

# 53. 多层 Transformer 会发生什么？

假设 12 个 blocks：

``` text
Block 1
↓
Block 2
↓
...
↓
Block 12
```

每层：

``` text
[B,N,C]
→
[B,N,C]
```

shape 看起来完全不变。

但 feature 内容不断变化。

这提醒我们：

> **Shape 不变 ≠ 网络没有做事。**

CNN 中很多 stage 会改变 shape。

Transformer block 常常主要改变 feature 内容，而保持 shape。

------------------------------------------------------------------------

# 54. DINOv3 为什么会取多个 Block 的中间特征？

DEIMv2 的 DINOv3+STA 路线会关注多个 Transformer block
的中间输出，例如配置中的 interaction indexes。

直觉上：

``` text
较早 block
→ 相对低层的视觉表示

较深 block
→ 更高级、更全局的语义表示
```

STA 再把这些信息适配到检测所需的空间多尺度结构。

后面第 19、20 课会详细展开。

------------------------------------------------------------------------

# 55. Self-Attention 与 CNN 卷积对比

CNN 3×3：

``` text
一个位置
主要直接读取局部 3×3 邻域
```

Global Self-Attention：

``` text
一个 token
理论上可以直接与所有 tokens 建立关系
```

CNN：

``` text
local inductive bias 强
```

Transformer：

``` text
global interaction 灵活
```

现代视觉架构常结合两者优势。

DEIMv2 的 DINOv3 + STA 就可以从这个角度理解一部分设计动机。

------------------------------------------------------------------------

# 56. Attention Weight 是不是"模型解释"？

需要谨慎。

Attention weights 确实表示：

``` text
在该 attention 运算中
不同 key/value 位置参与输出的权重
```

但不能简单断言：

``` text
attention weight 高
=
模型最终做出预测的唯一原因
```

深层模型还有：

``` text
多层 attention
FFN
residual
normalization
multiple heads
decoder
loss training
```

所以 attention map 可以帮助观察内部行为，但不能自动等价为完整因果解释。

------------------------------------------------------------------------

# 57. Self-Attention 的矩阵版本完整推导

设：

\[ X`\in`{=tex}`\mathbb{R}`{=tex}\^{N`\times `{=tex}C} \]

投影：

\[ Q=XW_Q \]

\[ K=XW_K \]

\[ V=XW_V \]

其中：

\[ Q,K`\in`{=tex}`\mathbb{R}`{=tex}\^{N`\times `{=tex}d_k} \]

\[ V`\in`{=tex}`\mathbb{R}`{=tex}\^{N`\times `{=tex}d_v} \]

分数：

\[ S=QK\^T \]

所以：

\[ S`\in`{=tex}`\mathbb{R}`{=tex}\^{N`\times `{=tex}N} \]

缩放：

\[ `\hat `{=tex}S = `\frac{S}{\sqrt{d_k}}`{=tex} \]

softmax：

\[ A = softmax(`\hat `{=tex}S) \]

所以：

\[ A`\in`{=tex}`\mathbb{R}`{=tex}\^{N`\times `{=tex}N} \]

输出：

\[ O=AV \]

于是：

\[ \[N,N\]`\times[N,d_v]`{=tex}= \[N,d_v\] \]

------------------------------------------------------------------------

# 58. 为什么 Softmax 通常沿最后一维？

假设：

``` text
scores:
[B,h,Nq,Nk]
```

对每个：

``` text
batch
head
query
```

我们需要在：

``` text
所有 keys
```

之间形成归一化权重。

所以：

``` python
softmax(
    scores,
    dim=-1
)
```

最后一维正是：

``` text
Nk
```

结果：

``` python
attn.sum(dim=-1)
```

应该接近 1。

------------------------------------------------------------------------

# 59. 第四个实验：观察 Attention Weight

``` python
import torch
import math

q = torch.tensor([
    [[1.0, 0.0]]
])

k = torch.tensor([
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]
])

v = torch.tensor([
    [
        [10.0, 0.0],
        [0.0, 10.0],
        [5.0, 5.0],
    ]
])

scores = (
    q @ k.transpose(-2, -1)
)

scores = scores / math.sqrt(2)

weights = torch.softmax(
    scores,
    dim=-1
)

out = weights @ v

print("scores:")
print(scores)

print("weights:")
print(weights)

print("output:")
print(out)
```

尝试在运行前回答：

``` text
query [1,0]
```

会更偏向哪些 keys？

然后观察最终 Value 的加权组合。

------------------------------------------------------------------------

# 60. 第五个实验：Attention 不只是"相似度平均"

改变：

``` text
V
```

但保持：

``` text
Q
K
```

不变。

你会发现：

``` text
attention weights 不变
```

但：

``` text
output 改变
```

这能非常清楚地理解：

``` text
Q/K
→ 决定读取位置/权重

V
→ 决定读取的内容
```

------------------------------------------------------------------------

# 61. 第六个实验：自己检查 Head Merge

``` python
import torch

B = 2
h = 8
N = 300
d = 32

x = torch.randn(
    B, h, N, d
)

print(
    "multi-head:",
    x.shape
)

x = x.transpose(
    1, 2
)

print(
    "after transpose:",
    x.shape
)

x = x.reshape(
    B, N, h * d
)

print(
    "merged:",
    x.shape
)
```

输出：

``` text
[2,8,300,32]
→
[2,300,8,32]
→
[2,300,256]
```

------------------------------------------------------------------------

# 62. 第七个实验：Self-Attention 与 Cross-Attention

``` python
import torch

B = 2
h = 8
d = 32

Nq = 300
Nk = 8400

q = torch.randn(
    B, h, Nq, d
)

k = torch.randn(
    B, h, Nk, d
)

v = torch.randn(
    B, h, Nk, d
)

scores = (
    q @ k.transpose(-2, -1)
)

print(scores.shape)
```

预期：

``` text
[2,8,300,8400]
```

不要真的在低显存设备上随意扩大
B/N/head，因为这个矩阵会快速占用大量内存。

------------------------------------------------------------------------

# 63. 第八个实验：一个完整 Tiny Transformer Encoder

``` python
import torch
import torch.nn as nn

class TinyEncoder(nn.Module):

    def __init__(
        self,
        dim=256,
        depth=4,
        num_heads=8
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            TinyTransformerBlock(
                dim=dim,
                num_heads=num_heads
            )
            for _ in range(depth)
        ])

    def forward(self, x):

        for i, block in enumerate(
            self.blocks
        ):
            x = block(x)

            print(
                f"block {i}:",
                x.shape
            )

        return x


x = torch.randn(
    2, 300, 256
)

model = TinyEncoder()

y = model(x)
```

你会看到每一层：

``` text
[2,300,256]
```

shape 都不变。

------------------------------------------------------------------------

# 64. 从 ViT 到 DETR：Attention 的角色发生变化

ViT Backbone：

``` text
patch tokens
↓
self-attention
↓
patch tokens
```

主要是：

``` text
image ↔ image
```

DETR Decoder：

``` text
object queries
↓
self-attention
↓
queries

queries
↓
cross-attention with image
↓
queries
```

主要是：

``` text
query ↔ query
query ↔ image
```

所以同一个 Attention 数学模块，在不同位置承担不同功能。

------------------------------------------------------------------------

# 65. DEIMv2 中你以后会看到的 Attention 家族

本课程后面会逐步遇到：

``` text
ViT Self-Attention

Multi-Head Attention

DETR Query Self-Attention

Cross-Attention

Multi-Scale Deformable Attention

DINOv3/RoPE-related Attention

Decoder localization-related interactions
```

它们都可以追溯到本课最基础的问题：

> Query 如何从 Key/Value 中选择并聚合信息？

------------------------------------------------------------------------

# 66. 真实源码阅读时的 8 个问题

以后看到任何 Attention 实现，固定问：

``` text
1. Q 从哪里来？
2. K 从哪里来？
3. V 从哪里来？
4. Q shape 是什么？
5. K/V shape 是什么？
6. Attention 在哪个维度 softmax？
7. 每个 query 可以看多少 key/value positions？
8. 输出 shape 怎样回到 residual 所需的 shape？
```

只要能回答这 8 个问题，大多数 Attention 代码就不会完全陌生。

------------------------------------------------------------------------

# 67. 常见误区 1：Q/K/V 是三份不同输入

不一定。

Self-Attention：

``` text
Q/K/V
```

都来自：

``` text
同一个 X
```

只是经过不同线性投影。

Cross-Attention 才通常是：

``` text
Q
来自 query sequence

K/V
来自 source/image sequence
```

------------------------------------------------------------------------

# 68. 常见误区 2：Attention 就是找最相似的一个 Token

不是。

Softmax 通常给所有允许位置一个权重分布：

``` text
0.6
0.2
0.1
0.05
...
```

输出是多个 Value 的加权组合。

并不一定只选一个位置。

------------------------------------------------------------------------

# 69. 常见误区 3：Multi-Head 就是把模型复制多份

不是。

它把 feature dimension 划分为多个 attention 子空间，并行学习不同关系。

------------------------------------------------------------------------

# 70. 常见误区 4：Transformer 没有空间信息

更准确地说：

> 纯内容 self-attention 本身不天然编码二维图像位置关系，因此视觉
> Transformer 通常需要 position embedding、relative position、RoPE
> 等位置机制。

不能简单说：

``` text
Transformer 完全不知道位置
```

因为真实模型会加入位置设计。

------------------------------------------------------------------------

# 71. 常见误区 5：Attention Matrix 越大越好

不是。

Full attention 提供灵活全局交互，但：

``` text
N×N
```

计算和内存很昂贵。

检测器需要高分辨率多尺度特征，因此 Deformable Attention
等方法会选择更稀疏、更高效的交互方式。

------------------------------------------------------------------------

# 72. 常见误区 6：Transformer 不需要 CNN

不同架构选择不同。

DEIMv2 本身就展示两条路线：

``` text
HGNetv2 CNN backbone

以及

DINOv3 ViT + STA backbone
```

而 STA/检测 neck 仍会涉及空间、多尺度和卷积式处理思想。

所以现实模型不是简单的：

``` text
CNN vs Transformer
```

二选一。

------------------------------------------------------------------------

# 73. 本课 Shape Cheat Sheet

## ViT tokens

``` text
Image:
[B,3,640,640]

Patch=16:
[B,C,40,40]

Flatten:
[B,1600,C]
```

------------------------------------------------------------------------

## QKV

``` text
Input:
[B,N,C]

QKV Linear:
[B,N,3C]

Reshape:
[B,N,3,h,d]

Permute:
[3,B,h,N,d]

Unbind:
Q/K/V
[B,h,N,d]
```

其中：

\[ C=hd \]

------------------------------------------------------------------------

## Self-Attention

``` text
Q:
[B,h,N,d]

Kᵀ:
[B,h,d,N]

QKᵀ:
[B,h,N,N]

Softmax:
[B,h,N,N]

× V:
[B,h,N,d]
```

------------------------------------------------------------------------

## Merge Heads

``` text
[B,h,N,d]

transpose

[B,N,h,d]

reshape

[B,N,C]
```

------------------------------------------------------------------------

## Cross-Attention

``` text
Q:
[B,h,Nq,d]

K/V:
[B,h,Nk,d]

Attention:
[B,h,Nq,Nk]

Output:
[B,h,Nq,d]
```

------------------------------------------------------------------------

# 74. 本课公式 Cheat Sheet

QKV：

\[ Q=XW_Q \]

\[ K=XW_K \]

\[ V=XW_V \]

Scaled Dot-Product Attention：

\[ Attention(Q,K,V) = softmax `\left`{=tex}(
`\frac{QK^T}{\sqrt{d_k}}`{=tex} `\right`{=tex})V \]

Multi-Head：

\[ head_i = Attention( QW_i\^Q, KW_i\^K, VW_i\^V ) \]

\[ MultiHead = Concat(head_1,`\ldots`{=tex},head_h)W\^O \]

FFN：

\[ FFN(x) = W_2`\sigma`{=tex}(W_1x+b_1)+b_2 \]

Residual：

\[ y=x+F(x) \]

LayerNorm：

\[ `\hat{x}`{=tex} = `\frac{x-\mu}`{=tex}
{`\sqrt{\sigma^2+\epsilon}`{=tex}} \]

\[ y=`\gamma`{=tex}`\hat{x}`{=tex}+`\beta`{=tex} \]

------------------------------------------------------------------------

# 75. 本课自测

尽量不用代码。

## Q1

``` text
X = [2,1600,256]
```

`nn.Linear(256,768)` 后是什么 shape？

------------------------------------------------------------------------

## Q2

为什么：

``` text
768 = 3×8×32
```

在 QKV 中有意义？

------------------------------------------------------------------------

## Q3

``` text
Q = [2,8,1600,32]
K = [2,8,1600,32]
```

`Q @ K.transpose(-2,-1)` 是什么 shape？

------------------------------------------------------------------------

## Q4

为什么除以：

\[ `\sqrt{32}`{=tex} \]

而不是：

\[ `\sqrt{256}`{=tex} \]

？

提示：每个 head 实际做点积的维度是多少？

------------------------------------------------------------------------

## Q5

Softmax 为什么沿 key dimension？

------------------------------------------------------------------------

## Q6

Self-Attention 与 Cross-Attention 的 Q/K/V 来源有什么区别？

------------------------------------------------------------------------

## Q7

如果：

``` text
Q:
[B,8,300,32]

K:
[B,8,8400,32]
```

attention matrix shape 是什么？

------------------------------------------------------------------------

## Q8

为什么 Multi-Head 输出最终又能恢复成：

``` text
[B,N,256]
```

？

------------------------------------------------------------------------

## Q9

Attention 与 FFN 的主要职责分别是什么？

------------------------------------------------------------------------

## Q10

Residual 为什么要求子层输出 shape 与输入兼容？

------------------------------------------------------------------------

## Q11

为什么 full self-attention 的复杂度与：

\[ N\^2 \]

相关？

------------------------------------------------------------------------

## Q12

为什么这个问题对目标检测特别重要？

------------------------------------------------------------------------

# 76. 课后作业 A：不用 `nn.MultiheadAttention` 手写 Attention

要求：

``` text
Input:
[B,N,256]

heads:
8

Output:
[B,N,256]
```

必须自己实现：

``` text
qkv projection
reshape
permute
QKᵀ
scale
softmax
×V
merge heads
output projection
```

并在每一步：

``` python
print(shape)
```

------------------------------------------------------------------------

# 77. 课后作业 B：验证 Attention 权重

构造：

``` text
N = 4
d = 2
```

手工指定 Q/K/V。

要求：

``` text
① 手算 QKᵀ
② 除以 sqrt(2)
③ 手算/用计算器求 softmax
④ 与 PyTorch 对比
⑤ 手算 weighted sum of V
```

这是本课最值得认真完成的作业。

------------------------------------------------------------------------

# 78. 课后作业 C：Self vs Cross

实现两个函数：

``` python
self_attention(x)

cross_attention(
    queries,
    memory
)
```

其中：

``` text
queries:
[B,300,256]

memory:
[B,8400,256]
```

先不要真的用很大 batch。

要求写出所有 shape，并解释：

``` text
300
8400
```

分别是什么。

------------------------------------------------------------------------

# 79. 课后作业 D：计算 Attention Matrix 大小

分别计算：

``` text
N = 300
N = 1600
N = 8400
```

时：

\[ N\^2 \]

是多少。

再假设：

``` text
8 heads
batch=2
float32=4 bytes
```

粗略计算显式 attention scores 的内存规模。

然后回答：

> 为什么 Deformable Attention 对检测模型很有吸引力？

------------------------------------------------------------------------

# 80. 课后作业 E：读一段真实 ViT Attention

找到 DEIMv2 ViT/DINOv3 相关 Attention 源码。

对下面这类链式代码：

``` python
qkv = self.qkv(x) \
    .reshape(...) \
    .permute(...)

q, k, v = qkv.unbind(0)
```

逐行写：

``` text
输入 shape
Linear 后 shape
reshape 后 shape
permute 后 shape
Q shape
K shape
V shape
```

要求不运行代码也能完成。

------------------------------------------------------------------------

# 81. 把前 4 课拼起来

现在我们已经有：

``` text
第 1 课：
完整目标检测地图

第 2 课：
Tensor / Shape 语言

第 3 课：
CNN / Multi-scale features

第 4 课：
Transformer / Attention
```

于是我们已经能理解：

``` text
Image
[B,3,640,640]

      ↓ CNN / Patch Embedding

Features / Tokens

      ↓ Transformer

Context-aware Features

      ↓ Object Queries

Detection
```

下一步就可以正式进入：

``` text
DETR
```

因为 DETR 的核心正是：

> **把 Transformer 与目标检测结合起来。**

------------------------------------------------------------------------

# 82. 下一课预告

## 第 5 课：DETR 与 Object Query------为什么 300 个 Query 能检测一张图？

下一课会回答：

``` text
什么是 Set Prediction？
为什么 DETR 不需要传统 anchor？
Object Query 到底是不是一个框？
为什么通常有 300 个 queries？
300 个 queries 与 8400 image tokens 有什么关系？
Decoder query self-attention 在做什么？
Cross-attention 在做什么？
为什么最终是 [B,300,K] + [B,300,4]？
```

我们会从一个最小 DETR 开始，亲手构造：

``` text
image memory
[B,8400,256]

object queries
[B,300,256]

↓ decoder

query features
[B,300,256]

↓ class head
[B,300,K]

↓ box head
[B,300,4]
```

并为第 6 课的：

``` text
Hungarian Matching
One-to-One Assignment
```

做好准备。

------------------------------------------------------------------------

# 附录 A：本课与 DEIMv2 源码的对应关系

本课概念会直接对应后续源码中的：

``` text
Patch Embedding
Attention
Transformer Block
LayerNorm
MLP / FFN
DINOv3 backbone blocks
DEIM/D-FINE decoder self-attention
decoder cross-attention
multi-scale deformable attention
```

读源码时重点寻找：

``` text
qkv
num_heads
head_dim
reshape
permute
unbind
scaled_dot_product_attention
proj
norm
mlp
residual
```

看到这些词，就回到本课的 shape 与公式。

------------------------------------------------------------------------

# 附录 B：参考资料

课程后续以官方源码和原论文为主：

1.  DEIMv2 官方仓库\
    https://github.com/Intellindust-AI-Lab/DEIMv2

2.  DEIMv2 中 ViT 相关实现\
    https://github.com/Intellindust-AI-Lab/DEIMv2/tree/main/engine/backbone

3.  PyTorch Scaled Dot-Product Attention 官方文档\
    https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

4.  Transformer 原论文：Attention Is All You Need\
    https://arxiv.org/abs/1706.03762

5.  Vision Transformer 原论文\
    https://arxiv.org/abs/2010.11929

> 本课只建立标准 Attention/Transformer 基础。DEIMv2 的
> DINOv3、RoPE、STA、Deformable Attention、DEIMTransformer
> 等具体实现，会在对应课程中逐一对照官方源码展开，避免把"标准
> Transformer"错误等同于"DEIMv2 中所有 Attention 实现"。

------------------------------------------------------------------------

**第 4 课结束。**

真正的达标标准是：当你看到

``` text
[B,N,C]
→ Q/K/V
→ [B,h,N,d]
→ QKᵀ
→ softmax
→ ×V
→ merge heads
→ [B,N,C]
```

时，你不仅能背出 shape，还能解释：

``` text
每一步为什么存在，
每个维度代表什么，
矩阵乘法为什么合法，
Attention 到底在聚合什么信息，
以及为什么这套机制可以进一步变成 DETR 的 query-image cross-attention。
```
