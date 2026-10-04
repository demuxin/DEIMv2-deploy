# DEIMv2 源码课程 · 第 20 课
# STA——DEIMv2 最核心的新结构

> 本课目标：真正理解 `deimv2_dinov3_x` 中 STA（Spatial Tuning Adapter）为什么存在、输入输出是什么、如何把 DINOv3 的单尺度强语义表示改造成检测器需要的 1/8、1/16、1/32 多尺度空间特征，以及它与后续 HybridEncoder 的职责为什么不同。

本课固定示例：

```text
B = batch size
input = 640 × 640
DINOv3 = dinov3_vits16plus
interaction_indexes = [5, 8, 11]
conv_inplane = 64
detector hidden_dim = 256
```

版本原则：公开仓库可能继续更新，因此本课把“官方 X 配置直接确认的事实”“ViT Adapter / DINOv3 Adapter 可确认的机制”“教学简化实验”分开说明。最终以你实际 commit 的源码和运行时 `tensor.shape` 为准。


---

# 1. 第 19 课留下的核心矛盾

上一课我们确认，640×640 输入经过 patch size 16 的 DINOv3 后，原生 patch grid 是：

```text
40 × 40 = 1600 patch positions
```

而实时目标检测希望得到：

```text
stride 8  → 80×80
stride 16 → 40×40
stride 32 → 20×20
```

所以真正的问题不是“DINOv3 特征够不够强”，而是：

> 怎样把 foundation model 的强表示转化成 detection-friendly spatial pyramid？

STA 就是这座桥。


---

# 2. 从真实 X 配置读出 STA 的接口

官方 X 配置的关键部分：

```yaml
DEIM:
  backbone: DINOv3STAs

DINOv3STAs:
  name: dinov3_vits16plus
  interaction_indexes: [5,8,11]
  finetune: True
  conv_inplane: 64
  hidden_dim: 256

HybridEncoder:
  in_channels: [256,256,256]
  hidden_dim: 256
```

因此可以确定 backbone 与 HybridEncoder 之间的契约：STA 最终要给 detector 三个 256-channel feature levels。公开的 backbone 导出/数值对比信息也明确展示 X 路线的三个输出：

```text
c2 [B,256,80,80]
c3 [B,256,40,40]
c4 [B,256,20,20]
```

对应 stride：

```text
640 / 80 = 8
640 / 40 = 16
640 / 20 = 32
```


---

# 3. STA 不是简单 FPN，也不是 HybridEncoder

三者职责要严格区分。

```text
DINOv3
↓
强 foundation representation，但 plain ViT 原生不是 CNN pyramid

STA
↓
补充 spatial prior，并把 ViT representation 适配成 1/8、1/16、1/32

HybridEncoder
↓
在已经形成的多尺度 feature 上继续做 intra-scale / cross-scale 融合
```

所以：

```text
STA = 创建并调谐 detection-friendly spatial hierarchy
HybridEncoder = 进一步融合 hierarchy
```

STA 更靠近 backbone；DEIMv2 工程上也把 `DINOv3 + STA` 整体注册成 `DINOv3STAs` backbone。


---

# 4. 为什么纯插值不是 STA

错误的最简方案：

```python
f16 = dinov3_feature
f8 = interpolate(f16, scale_factor=2)
f32 = avg_pool2d(f16, 2)
```

shape 会变成 80×80、40×40、20×20，但 40×40 插值到 80×80不会凭空恢复已经在 patch-16 tokenization 中压缩掉的细粒度空间信息。

STA 的关键思想是额外维护一个从原始图像建立的 CNN-like spatial-prior pathway，再让它和 ViT representation 交换信息。因此 stride-8 feature 的高分辨率信息不是“复制出来”的。


---

# 5. STA 总结构图

```text
                         Image
                    [B,3,640,640]
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
       DINOv3 ViT branch         Spatial Prior branch
             │                         │
       patch16 tokens             CNN-like hierarchy
             │                         │
       strong semantics           spatial detail
             │                         │
             └──────────┬──────────────┘
                        │
                 Interaction stages
                  Spatial ↔ ViT
                        │
                        ▼
                 adapted pyramid
             ┌──────────┼──────────┐
             ▼          ▼          ▼
           80×80      40×40      20×20
           1/8        1/16       1/32
             │          │          │
             └──────────┼──────────┘
                        ▼
                  HybridEncoder
```


---

# 6. Spatial Prior 到底是什么

这里的 prior 不是人工写死的边缘模板，而是 architecture 提供的空间归纳偏置。CNN-like branch 天然具有：

```text
locality
spatial continuity
hierarchical downsampling
```

它从原图逐步形成多尺度表示。一个教学版可以是：

```python
class MiniSPM(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.to_s8 = nn.Conv2d(64, dim, 3, 2, 1)
        self.to_s16 = nn.Conv2d(dim, dim, 3, 2, 1)
        self.to_s32 = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        x = self.stem(x)       # 1/4
        c2 = self.to_s8(x)     # 1/8
        c3 = self.to_s16(c2)   # 1/16
        c4 = self.to_s32(c3)   # 1/32
        return c2, c3, c4
```

这段是教学模型，不是对官方 STA 的逐字符复刻。


---

# 7. `conv_inplane=64` 与 `hidden_dim=256`

X 配置里的：

```text
conv_inplane = 64
```

更接近 Spatial Prior CNN branch 的内部容量设置，不代表最终输出只有 64 channels。

最终 detector interface 是：

```text
hidden_dim = 256
```

并与：

```yaml
HybridEncoder:
  in_channels: [256,256,256]
```

对齐。

还要继续保持第 19 课的警惕：foundation ViT 的原生 embedding dimension 与 DEIMv2 detector-side `hidden_dim` 不是同一个概念；遇到版本差异时必须检查当前 backbone constructor 和运行时 shape。


---

# 8. 为什么 `interaction_indexes=[5,8,11]` 不是三个尺度

这是本课最重要的纠错之一。

```text
5 / 8 / 11
```

描述的是 ViT **深度轴**上的 block index。

```text
1/8 / 1/16 / 1/32
```

描述的是 feature **空间尺度轴**。

它们是两个不同维度。

plain ViT 中 block 5、8、11 的 patch grid 不会因为层数加深就自动从 80×80 → 40×40 → 20×20。`interaction_indexes` 控制的是在哪些 ViT 深度让 spatial representation 与 ViT representation 交互。

如果 block index 从 0 开始：

```text
5  → 第 6 个 block
8  → 第 9 个 block
11 → 第 12 个 block
```


---

# 9. 多阶段 interaction 的直觉

可以先用下面的概念图理解 X：

```text
ViT blocks 0 ... 5
        ↓
Interaction stage
        ↓
ViT blocks ...
        ↓
Interaction stage around index 8
        ↓
ViT blocks ...
        ↓
Interaction stage around index 11
        ↓
STA output pyramid
```

具体 implementation 是在 block 前还是后交互、index 如何划分 block 区间，应以实际 `forward()` 为准；但不会改变核心结论：这些 index 是 ViT depth 上的交互节点，而不是 spatial stride。


---

# 10. Spatial feature flatten 后为什么又出现 8400

三个尺度：

```text
80×80 → 6400
40×40 → 1600
20×20 → 400
```

总数：

\[
6400+1600+400=8400
\]

feature map 转 token：

```python
def to_tokens(x):
    return x.flatten(2).transpose(1, 2)
```

得到：

```text
c2 → [B,6400,256]
c3 → [B,1600,256]
c4 → [B, 400,256]

cat → [B,8400,256]
```

注意：

```text
1600 = DINOv3 原生 40×40 patch positions
8400 = detection pyramid 三尺度总 positions
```

两者来源完全不同。


---

# 11. `spatial_shapes` 与 `level_start_index`

concat 后必须保留各 level 的空间边界：

```python
spatial_shapes = torch.tensor([
    [80,80],
    [40,40],
    [20,20],
])
```

对应：

```text
level 0 starts at 0
level 1 starts at 6400
level 2 starts at 6400+1600=8000
```

所以：

```text
level_start_index = [0,6400,8000]
```

这和第 7 课 Deformable Attention 的数据结构完全相同。


---

# 12. Level Embedding 为什么存在

当三个尺度 flatten 后都变成 `[B,N,C]`，网络需要区分 token 来自哪个 level。

概念上：

\[
C_l' = C_l + E_l
\]

教学代码：

```python
level_embed = nn.Parameter(torch.zeros(3, 256))

t2 = t2 + level_embed[0]
t3 = t3 + level_embed[1]
t4 = t4 + level_embed[2]
```

shape 完全不变，但 representation 获得了“尺度身份”。


---

# 13. 为什么 ViT 与 Spatial branch 不能直接相加

假设经过投影后二者 channel 都是 256：

```text
ViT:
[B,1600,256]

Spatial pyramid:
[B,8400,256]
```

sequence length 不同：

```text
1600 ≠ 8400
```

所以不能直接：

```python
vit + spatial
```

必须建立位置对应关系。完整 cross-attention 又需要处理约：

\[
1600	imes8400=13,440,000
\]

个 query-key pair（尚未乘 heads/batch），因此 sparse/deformable interaction 非常自然。


---

# 14. 为什么第 7 课的 Deformable Attention 在这里再次出现

Deformable Attention 的核心：

```text
reference point
+
learned offsets
+
few sampling points
+
bilinear interpolation
+
attention weights
```

它允许每个 query 不看全部 8400 positions，而只从多个 feature levels 的少量关键位置采样。

因此复杂度更接近：

```text
N_query × heads × levels × points
```

而不是：

```text
N_query × N_all_spatial
```

这非常适合 foundation ViT 与多尺度 spatial hierarchy 的交互。


---

# 15. Injector：Spatial → ViT

可以用一个抽象公式理解：

\[
X' = X + \gamma\,DA(Q=X,V=C)
\]

其中：

```text
X = ViT tokens
C = spatial pyramid tokens
DA = deformable attention
```

直觉是：

```text
每个 ViT token
→ 根据自己的空间 reference point
→ 去 spatial pyramid 的相关位置采少量 feature
→ 把局部/多尺度空间信息注入自己
```

因此 foundation token 获得更强的 detection-oriented localization information。

公式是教学抽象，不代表官方某一行代码的逐字符形式。


---

# 16. Extractor：ViT → Spatial

反方向：

\[
C' = C + DA(Q=C,V=X')
\]

也就是：

```text
spatial tokens
→ 从 ViT representation 中抽取强语义
```

最终 spatial maps 不再只是浅层 CNN feature，而是：

```text
local spatial prior
+
foundation-model semantics
```

这正是目标检测想要的组合。


---

# 17. 为什么双向交互比单向融合强

只有 Spatial → ViT：

```text
ViT 被空间调谐
```

但最终 detector 输出的 spatial pyramid 未必充分吸收 DINO semantics。

只有 ViT → Spatial：

```text
spatial maps 获得语义
```

但 ViT 本身没有受到 detection spatial prior 的调节。

双向：

```text
Spatial ↔ ViT
```

让两条 representation 在多个深度阶段共同演化。


---

# 18. Reference Point 在 STA 中是什么意思

ViT patch 本身位于规则的 40×40 grid。

例如 patch `(row=10,col=20)`，可以构造归一化中心：

\[
x=rac{20+0.5}{40},\qquad
y=rac{10+0.5}{40}
\]

使用 `[0,1]` 坐标后，同一个位置可映射到：

```text
80×80
40×40
20×20
```

三个 level。

注意：STA 的 feature-token reference point 与 decoder 中 object-query / bounding-box refinement 的 reference point 不是同一种语义，虽然都使用“reference point”这个词。


---

# 19. 同一个 normalized point 如何跨尺度

例如：

```text
(x,y)=(0.5,0.5)
```

大致对应：

```text
80×80 → (40,40)
40×40 → (20,20)
20×20 → (10,10)
```

所以一个 ViT token 可以从不同尺度的同一图像区域获得信息，然后再通过 learnable offsets 偏离中心位置。


---

# 20. Sampling Offset 与 Bilinear Sampling

网络预测 offset：

```text
Δx, Δy
```

然后：

```text
sampling_location
=
reference_point + normalized_offset
```

位置通常不是整数，因此用双线性插值：

\[
V(x,y)=\sum_{i,j}w_{ij}V(i,j)
\]

这使 sampling location 可微，offset 能通过 detection loss 学习。


---

# 21. `grid_sample` 最小实验

```python
import torch
import torch.nn.functional as F

feat = torch.randn(1, 256, 80, 80)

# grid_sample 坐标使用 [-1,1]
grid = torch.tensor(
    [[[[0.0, 0.0]]]],
    dtype=feat.dtype
)

sample = F.grid_sample(
    feat,
    grid,
    mode="bilinear",
    align_corners=False
)

print(sample.shape)
# [1,256,1,1]
```

Deformable Attention 本质上把这个操作扩展到：

```text
many queries
many heads
many levels
many learned points
learned attention weights
```


---

# 22. Spatial Prior 的 stride-8 为什么对小目标重要

一个 12×12 pixel 目标：

```text
stride16:
不到一个 feature cell

stride8:
大约覆盖 1.5×1.5 cells
```

stride-8 不能保证解决所有小目标，但 spatial sampling density 明显更高，更有利于精确定位。

这也是为什么“强语义”不能替代“高分辨率 spatial feature”。


---

# 23. stride-32 又为什么不可缺

对于大目标和全局语义：

```text
20×20
```

的 coarse representation 计算更经济，也更适合高层语义。

所以 detector 不是只追求越高分辨率越好，而是希望：

```text
high-resolution localization
+
mid-scale balance
+
coarse high-level semantics
```

共同存在。


---

# 24. STA 输出到 HybridEncoder 的完整接口

```text
Input
[B,3,640,640]

↓ DINOv3STAs

[
  [B,256,80,80],
  [B,256,40,40],
  [B,256,20,20]
]

↓ HybridEncoder

[
  [B,256,80,80],
  [B,256,40,40],
  [B,256,20,20]
]
```

HybridEncoder 输入输出 shape 可以相同，但 feature 内容已经通过 intra-scale / cross-scale processing 改变。

“shape 不变”绝不意味着“模块没做事”。


---

# 25. 为什么 STA 不能被 HybridEncoder 替代

如果只有 DINO 原生 40×40 feature，HybridEncoder 可以做一些 resize/fusion，但无法凭空恢复原图中在 patch-16 tokenization 时被压缩的细粒度信息。

STA 的 spatial-prior branch从原图建立高分辨率结构，这是关键差别。


---

# 26. 为什么 HybridEncoder 也不能被 STA 替代

STA 重点是：

```text
foundation-model adaptation
+
multi-scale spatial representation creation
```

HybridEncoder 重点是：

```text
已经形成的 P3/P4/P5 之间继续进行 feature fusion
```

所以两者是串联的两阶段问题：

```text
先“造出好 pyramid”
再“把 pyramid 融合得更好”
```


---

# 27. `finetune=True` 与 STA 的关系

X 配置：

```yaml
finetune: True
```

说明 DINOv3 不只是 frozen feature extractor。

Detection loss 的梯度可以经过：

```text
DEIMTransformer
↓
HybridEncoder
↓
STA
↓
DINOv3
```

继续小幅调节 foundation representation。

这与第 18 课的小 backbone LR 完全配套：

```text
DINOv3 → gentle finetuning
STA / detector → stronger task adaptation
```


---

# 28. 如果冻结 DINOv3 会怎样

概念上：

```text
loss
↓
decoder
↓
encoder
↓
STA
X
DINO frozen
```

优点：

```text
更省训练显存
减少 pretrained representation 被破坏的风险
```

代价：

```text
foundation backbone 本身不能针对 detection/domain 调整
```

所以 freeze / finetune 是可实验的 trade-off。


---

# 29. STA 为什么是“梯度桥”

STA 不只是 forward feature bridge。

当 `finetune=True`：

```text
Detection Loss
      │
      ▼
STA outputs
 ┌────┴────┐
 ▼         ▼
Spatial   ViT branch
branch      │
            ▼
        DINOv3 blocks
```

所以 STA 同时连接：

```text
forward representation flow
```

与：

```text
backward task gradient flow
```


---

# 30. 不同模型的 interaction indexes 不一定相同

不要把 X 的：

```text
[5,8,11]
```

当成 STA 定义本身。

公开的 S/M distilled ViT 路线可见：

```text
[3,7,11]
```

而 L/X DINOv3-S/S+ 路线使用：

```text
[5,8,11]
```

因此：

```text
interaction_indexes
```

是 model/config-specific architecture hyperparameter。


---

# 31. 为什么配置版本必须核对

公开生态中曾出现 YAML 与 notebook/exported config 对 interaction indexes 不一致的讨论。

所以阅读仓库时优先级建议：

```text
1. 你实际运行的 YAML
2. 当前 commit 的 Python source
3. checkpoint 对应 config
4. runtime tensor.shape / module attributes
5. README / notebook / 教程
```

尤其不要因为旧 notebook 里出现一个数字，就覆盖当前训练 YAML。


---

# 32. SPM 实验：真实 shape

```python
import torch
import torch.nn as nn

x = torch.randn(2,3,640,640)

spm = MiniSPM(dim=256)
c2,c3,c4 = spm(x)

print(c2.shape)
print(c3.shape)
print(c4.shape)
```

目标：

```text
[2,256,80,80]
[2,256,40,40]
[2,256,20,20]
```


---

# 33. Flatten / Concat 实验

```python
def flatten_feature(x):
    return x.flatten(2).transpose(1,2)

t2 = flatten_feature(c2)
t3 = flatten_feature(c3)
t4 = flatten_feature(c4)

print(t2.shape)  # [B,6400,256]
print(t3.shape)  # [B,1600,256]
print(t4.shape)  # [B, 400,256]

spatial = torch.cat([t2,t3,t4], dim=1)
print(spatial.shape)  # [B,8400,256]
```


---

# 34. Split / Restore 实验

```python
t2_,t3_,t4_ = torch.split(
    spatial,
    [6400,1600,400],
    dim=1
)

def to_map(t,h,w):
    B,N,C = t.shape
    return (
        t.transpose(1,2)
         .reshape(B,C,h,w)
    )

c2_ = to_map(t2_,80,80)
c3_ = to_map(t3_,40,40)
c4_ = to_map(t4_,20,20)
```

如果中间没有 interaction，可以验证：

```python
assert torch.allclose(c2, c2_)
assert torch.allclose(c3, c3_)
assert torch.allclose(c4, c4_)
```


---

# 35. 自动计算 `level_start_index`

```python
spatial_shapes = torch.tensor([
    [80,80],
    [40,40],
    [20,20],
])

sizes = (
    spatial_shapes[:,0]
    * spatial_shapes[:,1]
)

level_start_index = torch.cat([
    torch.zeros(1,dtype=torch.long),
    sizes.cumsum(0)[:-1]
])

print(sizes)
# [6400,1600,400]

print(level_start_index)
# [0,6400,8000]
```


---

# 36. 教学版 Level Embedding

```python
level_embed = nn.Parameter(
    torch.zeros(3,256)
)

t2 = t2 + level_embed[0]
t3 = t3 + level_embed[1]
t4 = t4 + level_embed[2]
```

注意广播：

```text
[B,N,256] + [256]
```

shape 不变。


---

# 37. Full Attention 为什么太重

教学版 cross-attention：

```python
scores = q @ k.transpose(-2,-1)
```

如果：

```text
q = 1600 ViT tokens
k = 8400 spatial tokens
```

则：

```text
scores = [B,1600,8400]
```

每个 sample 已有：

\[
13.44	ext{ million}
\]

个 score。再乘 heads、保存反向传播中间量，成本迅速增加。

因此真实 adapter 更适合 sparse/deformable interaction。


---

# 38. 固定局部采样与 Deformable Sampling

固定 3×3：

```text
(-1,-1) ... (+1,+1)
```

永远看同一邻域。

Deformable：

```text
offset = function(query)
```

不同 query 可以学习不同 sampling geometry：

```text
目标边缘
物体内部
另一个尺度
远处相关区域
```

因此比固定局部窗口更灵活。


---

# 39. 一个最小 Learnable Offset 实验

```python
offset_head = nn.Linear(256, 2)

query = torch.randn(2,1600,256)

offset = offset_head(query)

print(offset.shape)
# [2,1600,2]
```

下一步可以把 offset 做适当缩放，再与 normalized reference point 相加，然后用 `grid_sample` 采 feature。

这就是 deformable sampling 的最小原型。


---

# 40. STA 与 Decoder Reference Point 不要混淆

STA：

```text
feature token 的几何位置
```

用于 ViT/spatial feature interaction。

Decoder：

```text
object query 的 reference point / box state
```

用于目标查询和 iterative box refinement。

二者都使用归一化坐标和 sampling，但所在阶段与语义完全不同。


---

# 41. STA 中没有 300 Object Queries

STA 仍然处于 backbone：

```text
Image
↓
DINOv3STAs
↓
8400 multi-scale positions
```

之后：

```text
HybridEncoder
↓
encoder proposals / top-k selection
↓
300 object queries
```

所以：

```text
STA ≠ query selection
STA ≠ decoder
```


---

# 42. STA 也不负责 Matching / Loss

Hungarian Matching、MAL、Dense O2O 都属于训练监督机制。

STA 属于：

```text
feature representation architecture
```

因此 DEIMv2 的改进要分成两条轴：

```text
Architecture:
DINOv3 + STA + detector structure

Training:
Dense O2O v2 + MAL + augmentation/optimization recipe
```


---

# 43. 1600 → 8400 不是 token repeat

不能理解成：

```python
vit_tokens.repeat(...)
```

真实意义是：

```text
原图 spatial prior hierarchy
+
ViT semantic interaction
→
三个尺度各自具有空间意义的 feature positions
```

所以增加的是 detection-oriented representation density，而不是简单复制。


---

# 44. 为什么 stride16 是特殊对齐层

DINO patch size 16，因此：

```text
DINO patch grid = 40×40
Spatial stride16 = 40×40
```

二者天然同分辨率。

而：

```text
stride8 = 2× spatial resolution
stride32 = 1/2 spatial resolution
```

所以 stride16 是 ViT geometry 与 pyramid geometry 最自然的对齐点。


---

# 45. 输入尺寸变化时的推导

800×800、patch16：

```text
ViT grid:
50×50 = 2500

STA pyramid:
100×100
50×50
25×25
```

总 detector positions：

\[
10000+2500+625=13125
\]

输入变大不仅增加 detector pyramid cost，还会增加 ViT global attention cost。


---

# 46. STA 并没有消除 ViT 的二次复杂度

STA 的 deformable interaction可以避免 ViT↔spatial 使用巨大的 dense cross-attention。

但 DINOv3 自身的 self-attention 仍然要运行。

所以：

```text
efficient adapter
≠
free ViT backbone
```

这也是 DEIMv2 S/M 使用蒸馏 Tiny ViT、L/X 使用 DINOv3-S/S+ 的模型族设计背景。


---

# 47. 为什么 STA 是接口统一器

不同 ViT backbone 可能具有不同：

```text
embed_dim
heads
capacity
pretraining
```

但 detector 最终希望看到统一的：

```text
stride8 feature
stride16 feature
stride32 feature
```

STA 把 foundation-model 世界适配成 detector 世界，因此具有很强的工程模块化价值。


---

# 48. X 与 L 的 Adapter 容量也会缩放

公开配置中可以看到不同 variant 的：

```text
conv_inplane
hidden_dim
HybridEncoder capacity
decoder capacity
```

会一起变化。

所以 X 相比 L 并不是只把 DINOv3-S 换成 S+；这是 system-level scaling，而不是单模块 scaling。


---

# 49. 如何在本地真实验证 backbone 输出

```python
model.backbone.eval()

with torch.no_grad():
    feats = model.backbone(images)

for i,f in enumerate(feats):
    print(i, tuple(f.shape))
```

对 X、640 输入，应该重点核对：

```text
3 levels?
channels = 256?
spatial = 80/40/20?
```

如果你的 commit 不同，以 runtime 为准并记录差异。


---

# 50. 推荐的递归 Shape Printer

```python
def summarize(x):
    if torch.is_tensor(x):
        return tuple(x.shape)

    if isinstance(x,(list,tuple)):
        return [summarize(v) for v in x]

    if isinstance(x,dict):
        return {
            k:summarize(v)
            for k,v in x.items()
        }

    return type(x).__name__
```

它比只处理单 tensor 的 hook 更适合 adapter，因为 interaction block 很可能返回 tuple/list。


---

# 51. 给 STA 内部模块装 Hook

```python
def make_hook(name):
    def hook(module, inputs, outputs):
        print("\n", name)
        print("IN :", summarize(inputs))
        print("OUT:", summarize(outputs))
    return hook

handles = []

for name,module in model.backbone.named_modules():
    lower = name.lower()

    if any(k in lower for k in [
        "spm",
        "inter",
        "inject",
        "extract",
    ]):
        handles.append(
            module.register_forward_hook(
                make_hook(name)
            )
        )
```

这能把“结构图”真正对应到当前 commit 的 Python object。


---

# 52. 检查梯度是否进入 DINOv3

训练一个 batch 后：

```python
loss.backward()

for name,p in model.backbone.named_parameters():
    if "dinov3" in name.lower():
        if p.grad is not None:
            print(
                name,
                p.grad.norm().item()
            )
            break
```

如果 `finetune=True` 且 optimizer/graph 设置正常，应能找到参与训练的 DINOv3 参数梯度。


---

# 53. 检查 STA 与 DINO 的学习率组

```python
for i,g in enumerate(optimizer.param_groups):
    print(
        i,
        g["lr"],
        len(g["params"])
    )
```

进一步建立 parameter name → param group 的映射，验证：

```text
DINOv3
→ small LR

STA / encoder / decoder
→ detector-side LR
```

这会把第 18 课训练 recipe 与本课 architecture 完整连接起来。


---

# 54. Feature Statistics 调试

```python
for i,f in enumerate(feats):
    print(
        i,
        "mean=", f.mean().item(),
        "std=", f.std().item(),
        "max=", f.abs().max().item()
    )
```

重点排查：

```text
NaN
Inf
全 0
极端 std
```

这些可能提示：

```text
weight loading
normalization
precision
adapter interaction
```

存在问题。


---

# 55. Feature Energy Map

```python
f = feats[0]

energy = (
    f.pow(2)
     .mean(1)
     .sqrt()
)

print(energy.shape)
# [B,H,W]
```

可以可视化三个 level 的 spatial activation。

但注意：

```text
某层平均激活更大
≠
某层一定更重要
```


---

# 56. STA 的核心数学抽象

把 ViT state 记为：

\[
X_t
\]

spatial pyramid state 记为：

\[
C_t
\]

一次 interaction 可抽象为：

\[
X_{t+1}
=
F_{vit}
\left(
X_t + I(C_t,X_t)
ight)
\]

再：

\[
C_{t+1}
=
C_t + E(X_{t+1},C_t)
\]

其中：

```text
I = Injector
E = Extractor
F_vit = 一段 ViT blocks
```

所以这是一个耦合系统：

\[
X_{t+1}=f(X_t,C_t)
\]

\[
C_{t+1}=g(C_t,X_{t+1})
\]


---

# 57. 为什么这比 concat 更强

简单 concat 只表示：

```text
把 feature 放到一起
```

interaction 则可以学习：

```text
从什么尺度取信息
在什么位置采样
offset 多大
每个 sampling point 权重多少
什么时候注入
什么时候抽取
```

所以 adapter 真正学习的是 representation transformation，而不是 tensor 拼接。


---

# 58. 推荐的 Ablation 实验

如果以后自己研究，可以比较：

```text
A. DINO final feature + simple resize
B. Spatial branch only
C. DINO + simple FPN
D. DINO + STA
E. DINO + STA + HybridEncoder
```

记录：

```text
AP
APs
APm
APl
Params
GFLOPs
Latency
```

特别关注 `APs`，它能帮助分析 stride-8 spatial adaptation 对小目标的价值。


---

# 59. Interaction Schedule Ablation

比较：

```text
[11]
[8,11]
[5,8,11]
更多 interaction stages
```

观察：

```text
accuracy
memory
latency
```

不要预设“越多越好”，因为 DEIMv2 的目标包含 real-time efficiency。


---

# 60. Freeze / Finetune Ablation

比较：

```text
A. DINO completely frozen
B. only late blocks trainable
C. all DINO trainable with small LR
```

记录：

```text
trainable params
peak memory
iteration time
convergence
AP
```

这会帮助你理解 adapter 与 foundation-backbone finetuning 的分工。


---

# 61. 本课最常见的误解

必须避免：

```text
STA = FPN
STA = interpolate
block 5 = stride8
block 8 = stride16
block 11 = stride32
hidden_dim=256 = DINO native dim
conv_inplane=64 = final output channels
STA produces 300 object queries
STA does Hungarian matching
STA replaces HybridEncoder
HybridEncoder replaces STA
8400 = DINO patch tokens
interaction_indexes are fixed for all variants
```

这些都不正确或过度简化。


---

# 62. 本课必须记住的数字

对于 `deimv2_dinov3_x`、640×640：

```text
DINO patch size:
16

DINO patch grid:
40×40

DINO patch positions:
1600

STA target levels:
80×80
40×40
20×20

positions:
6400
1600
400

total:
8400

STA detector channels:
256

interaction indexes:
[5,8,11]

conv_inplane:
64

HybridEncoder input channels:
[256,256,256]
```


---

# 63. 最终 Shape Map

```text
Image
[B,3,640,640]
        │
        ├──────────────────────────────┐
        │                              │
        ▼                              ▼
DINOv3 ViT-S+                   Spatial Prior
patch16                         CNN-like branch
        │                              │
        ▼                              ├─ 80×80
40×40 patch grid                      ├─ 40×40
        │                              └─ 20×20
        ▼                              │
ViT representation                    │
        │                              │
        └──────────┬───────────────────┘
                   ▼
            Spatial ↔ ViT
          Interaction stages
             [5,8,11]
                   │
                   ▼
              STA outputs
       ┌───────────┼───────────┐
       ▼           ▼           ▼
[B,256,80,80] [B,256,40,40] [B,256,20,20]
       │           │           │
       └───────────┼───────────┘
                   ▼
             HybridEncoder
                   │
                   ▼
             enhanced pyramid
                   │
                   ▼
           8400 spatial positions
                   │
                   ▼
            DEIMTransformer
```


---

# 64. 本课检查题

1. STA 的全称是什么？
2. STA 解决 DINOv3 用于 detection 时的什么核心问题？
3. 640、patch16 时 DINO patch grid 是多少？
4. 为什么 block 5/8/11 不能直接叫 P3/P4/P5？
5. depth axis 与 scale axis 有什么区别？
6. Spatial Prior 为什么从原图建立？
7. 为什么不能只把 40×40 插值成 80×80？
8. X 的 `conv_inplane` 是多少？
9. detector-side `hidden_dim` 是多少？
10. STA 三个最终 feature shape 是什么？
11. 三尺度总 positions 是多少？
12. `level_embed` 的作用是什么？
13. Injector 的方向是什么？
14. Extractor 的方向是什么？
15. 为什么适合 deformable interaction？
16. STA reference point 与 decoder reference point 是否同义？
17. STA 会生成 300 queries 吗？
18. STA 会做 Hungarian matching 吗？
19. STA 与 HybridEncoder 有什么区别？
20. `interaction_indexes` 是否所有 variant 固定？
21. `finetune=True` 对梯度路径意味着什么？
22. 为什么 DINO LR 应比新 detector modules 小？


---

# 65. 检查题答案

```text
1. Spatial Tuning Adapter。
2. 把强但原生非 CNN-pyramid 的 ViT representation 适配成 detection-friendly multi-scale spatial features。
3. 40×40。
4. 因为 5/8/11 是 ViT 深度索引，不是 spatial stride。
5. depth 是网络层深；scale 是 feature resolution。
6. 为了保留/建立真实细粒度 spatial hierarchy。
7. 插值不会凭空恢复已经压缩掉的空间细节。
8. 64。
9. 256。
10. [B,256,80,80]、[B,256,40,40]、[B,256,20,20]。
11. 8400。
12. 告诉 concat 后的 token 来自哪个 pyramid level。
13. Spatial → ViT。
14. ViT → Spatial。
15. 它能以少量 learnable sampling points 做多尺度空间交互，避免巨大 full cross-attention。
16. 不是；STA 偏 feature geometry，decoder 偏 object query / box state。
17. 不会。
18. 不会。
19. STA 创建并调谐 foundation-model spatial pyramid；HybridEncoder 进一步融合已经形成的 pyramid。
20. 不是。
21. detection gradient 可以经 STA 回到参与训练的 DINOv3 参数。
22. 为保护 pretrained representation，同时让新模块更快适应 detection。
```


---

# 66. 课后实验清单

完成以下实验：

```text
A. MiniSPM：640 → 80/40/20
B. 三尺度 flatten → [B,8400,256]
C. split → restore，并用 torch.allclose 验证
D. 自动计算 spatial_shapes / level_start_index
E. level embedding
F. grid_sample 单点/多点采样
G. learnable offset
H. 简化双向 Spatial ↔ ViT interaction
I. 对真实 DINOv3STAs 安装 recursive shape hooks
J. 检查 DINO / STA gradient
K. 检查 optimizer param groups
L. frozen vs finetune 对比
```


---

# 67. 你必须能口述的 STA 解释

学完后应能不看讲义说出：

```text
deimv2_dinov3_x 并不是直接把 DINOv3 最后一层 40×40 patch feature
交给检测器。DINOv3STAs 同时维护 foundation ViT representation
和从原图建立的 CNN-like spatial-prior hierarchy。

Spatial branch 提供 1/8、1/16、1/32 的空间结构；
ViT branch 提供 DINOv3 的强语义 representation。
二者在 interaction_indexes 指定的 ViT 深度进行稀疏、空间感知的交互，
最终形成 256-channel 的 80×80、40×40、20×20 feature maps。

这些 feature maps 再进入 HybridEncoder 做进一步多尺度融合，
之后才进入 DEIMTransformer 的 query selection 与 decoder。
```


---

# 68. 下一课预告

第 21 课：

# 《DEIMTransformer——DEIMv2 为什么不继续照搬 DFINETransformer》

下一课从：

```text
[B,256,80,80]
[B,256,40,40]
[B,256,20,20]
```

开始追：

```text
8400 spatial positions
↓
encoder proposals
↓
top-k query selection
↓
300 queries
↓
decoder layers
↓
class + bbox
```

并系统比较：

```text
DFINETransformer
vs
DEIMTransformer
```


---

# 69. 源码核对与版本原则

本课重点核对：

```text
DEIMv2 official X config
DEIMv2 official README
DINOv3 official dense-prediction adapter implementation
DEIMv2 public backbone output information
different DEIMv2 variant configs
```

实际读仓库时优先级：

```text
runtime tensor.shape
>
current commit Python source
>
actual YAML
>
checkpoint-matched config
>
README / notebook / tutorial
```

这样可以最大限度避免仓库更新和导出配置差异造成的误解。


---

# 70. 参考资料

- DEIMv2 official repository  
  `https://github.com/Intellindust-AI-Lab/DEIMv2`

- DEIMv2 X config  
  `https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/configs/deimv2/deimv2_dinov3_x_coco.yml`

- DINOv3 official repository  
  `https://github.com/facebookresearch/dinov3`

- DINOv3 dense-prediction adapter  
  `https://github.com/facebookresearch/dinov3/blob/main/dinov3/eval/segmentation/models/backbone/dinov3_adapter.py`


---

# 71. 一句话总结

```text
STA 的本质不是“给 DINOv3 加一个 FPN”，
而是同时维护 ViT semantic branch 与 CNN-like spatial-prior branch，
在多个 ViT 深度上通过稀疏、空间感知的交互让两种 representation 双向融合，
最终把以 stride-16 patch tokens 为中心的 foundation model，
转化成实时 DETR 可以直接消费的 stride-8 / stride-16 / stride-32 多尺度检测特征。
```
