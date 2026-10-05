# DEIMv2 源码课程 · 第 18 课
# 完整训练 Recipe——LR、AdamW、EMA、AMP、DDP

**本课定位**

前 17 课已经把“模型怎样预测”和“监督怎样构造”拆开了。本课把它们重新装回真正的训练系统：

```text
batch
  ↓
forward
  ↓
Hungarian Matching
  ↓
MAL + box losses
  ↓
backward
  ↓
AMP unscale
  ↓
gradient clipping
  ↓
AdamW
  ↓
LR schedule
  ↓
EMA
  ↓
DDP synchronization
```

我们重点回答一个非常实际的问题：

**为什么同一个 `deimv2_dinov3_x` 网络，换一套 LR / optimizer / EMA / AMP / batch-size recipe，训练结果可能明显不同？**

本课以当前官方 `deimv2_dinov3_x_coco.yml` 和 DEIMv2 README 为主线，同时把 `deim_dfine_x` / HGNetv2 路线放进同一套框架比较。

---

## 0. 本课先记住 12 句话

1. Optimizer 不只是“更新参数”，它决定每个参数怎样把 gradient 变成 parameter update。
2. DEIMv2 使用 AdamW，并对不同模块设置不同 LR / weight decay。
3. DINOv3 是大规模预训练 backbone，检测器的 STA / encoder / decoder 则更需要快速适配，因此 LR 不应该一刀切。
4. X 配置中 DINOv3 LR 为 `1e-5`，检测模块默认 LR 为 `5e-4`，相差 **50 倍**。
5. norm / BN / bias 通常被放入 `weight_decay=0` 的参数组。
6. Warmup 解决训练初期“模型尚未稳定但 LR 已很大”的问题。
7. DEIM 系列使用 flat-cosine 风格的训练日程，而不是只依赖传统固定 step decay。
8. EMA 维护一份训练参数的平滑版本，通常用于更稳定的评估。
9. DEIMv2 还把 no-augmentation 阶段与 EMA 生命周期结合起来。
10. AMP 主要降低显存和提高吞吐；GradScaler 主要解决 FP16 梯度下溢。
11. DDP 是“一张 GPU 一个进程 + 梯度同步”，不是把一个 batch 简单复制到每张卡。
12. 改 global batch size 时，LR、warmup、EMA 等时间尺度也要一起重新考虑。

---

# 1. 一次训练 iteration 到底发生了什么？

把工程细节先全部压缩成：

```python
for images, targets in train_loader:
    optimizer.zero_grad()

    outputs = model(images, targets)
    loss_dict = criterion(outputs, targets)
    loss = sum(...)

    loss.backward()
    clip_grad_norm_(...)
    optimizer.step()

    ema.update(model)
```

如果启用 AMP：

```python
optimizer.zero_grad()

with autocast(...):
    outputs = model(images, targets)
    loss_dict = criterion(outputs, targets)
    loss = sum(...)

scaler.scale(loss).backward()

scaler.unscale_(optimizer)
clip_grad_norm_(...)

scaler.step(optimizer)
scaler.update()

ema.update(model)
```

如果启用 DDP：

```text
GPU0 / rank0 ─ forward ─ backward ─┐
GPU1 / rank1 ─ forward ─ backward ─┼─ all-reduce gradients
GPU2 / rank2 ─ forward ─ backward ─┤
GPU3 / rank3 ─ forward ─ backward ─┘
                                 ↓
                          optimizer.step()
```

本课就是把上面每一行拆透。

---

# 2. 先区分 4 个最容易混淆的量

## 2.1 Loss

$$
L(\theta)
$$

衡量当前预测有多差。

## 2.2 Gradient

$$
g_t=\nabla_\theta L(\theta_t)
$$

告诉我们参数往哪个方向变化会使 loss 下降。

## 2.3 Learning Rate

$$
\eta
$$

控制“这一步走多远”。

## 2.4 Optimizer

optimizer 决定：

```text
gradient
    ↓
怎样平滑？
怎样归一化？
怎样加入 weight decay？
    ↓
parameter update
```

SGD 最简单：

$$
\theta_{t+1}=\theta_t-\eta g_t
$$

AdamW 则远比这复杂。

---

# 3. 为什么 DEIMv2 用 AdamW？

Transformer / ViT 系列训练普遍适合 AdamW：

```text
first moment   → 梯度方向的指数平均
second moment  → 梯度平方的指数平均
adaptive step  → 不同参数得到不同尺度的更新
decoupled WD   → weight decay 与梯度更新解耦
```

对复杂检测器：

```text
DINOv3
STA
HybridEncoder
Transformer Decoder
class head
bbox head
```

不同参数的 gradient scale 可能差异很大，AdamW 的自适应更新非常有用。

---

# 4. Adam：一阶矩与二阶矩

给定：

$$
g_t=\nabla_\theta L_t
$$

一阶矩：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

二阶矩：

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

DEIMv2 X：

```yaml
betas: [0.9, 0.999]
```

所以：

$$
\beta_1=0.9,\qquad\beta_2=0.999
$$

一阶矩更快响应当前梯度；
二阶矩的时间尺度更长，用于估计梯度幅值。

---

# 5. Bias correction 为什么存在？

初始化：

$$
m_0=0,\qquad v_0=0
$$

所以训练刚开始：

$$
m_t,v_t
$$

会天然偏向 0。

Adam 使用：

$$
\hat m_t=\frac{m_t}{1-\beta_1^t}
$$

$$
\hat v_t=\frac{v_t}{1-\beta_2^t}
$$

修正这种初始化偏差。

更新近似：

$$
\theta_{t+1}
=
\theta_t
-
\eta
\frac{\hat m_t}
{\sqrt{\hat v_t}+\epsilon}
$$

---

# 6. Adam 与 AdamW 最大区别：Weight Decay

AdamW 的核心思想是把参数衰减从 loss gradient 中解耦：

$$
\theta_{t+1}
=
(1-\eta\lambda)\theta_t
-
\eta
\frac{\hat m_t}
{\sqrt{\hat v_t}+\epsilon}
$$

其中：

$$
\lambda=\text{weight decay}
$$

第一项：

$$
(1-\eta\lambda)\theta_t
$$

直接把参数向 0 收缩。

这就是“decoupled weight decay”。

---

# 7. X 模型的真实 optimizer 配置

当前官方 `deimv2_dinov3_x_coco.yml`：

```yaml
optimizer:
  type: AdamW

  params:
    -
      # except norm/bn/bias in self.dinov3
      params: '^(?=.*.dinov3)(?!.*(?:norm|bn|bias)).*$'
      lr: 0.00001

    -
      # including norm/bn/bias in self.dinov3
      params: '^(?=.*.dinov3)(?=.*(?:norm|bn|bias)).*$'
      lr: 0.00001
      weight_decay: 0.

    -
      # norm/bn/bias in sta|encoder|decoder
      params: '^(?=.*(?:sta|encoder|decoder))(?=.*(?:norm|bn|bias)).*$'
      weight_decay: 0.

  lr: 0.0005
  betas: [0.9, 0.999]
  weight_decay: 0.000125
```

这是本课最重要的一段配置之一。

---

# 8. 这段配置实际上把参数分成了什么？

可以理解成：

```text
Group A
DINOv3 normal weights
LR = 1e-5
WD = 1.25e-4（继承默认）

Group B
DINOv3 norm / bn / bias
LR = 1e-5
WD = 0

Group C
STA / encoder / decoder 的 norm / bn / bias
LR = default 5e-4
WD = 0

Group D
没被特殊 regex 覆盖的普通 detector weights
LR = default 5e-4
WD = 1.25e-4
```

最终准确分组必须以仓库的 optimizer param-group builder 对 regex 的处理为准，但理解配置时可以先建立这张地图。

---

# 9. 为什么 DINOv3 LR 只有 `1e-5`？

DINOv3 已经有强大的预训练 representation。

我们不希望：

```text
COCO detection gradients
        ↓
用很大的 LR
        ↓
快速破坏 pretrained representation
```

这类问题常被概括为：

```text
catastrophic forgetting / destructive finetuning
```

所以 backbone 使用较小 LR。

---

# 10. 为什么 detector LR 是 `5e-4`？

STA / encoder / decoder 必须学习：

```text
foundation features
      ↓
detection-specific representation
      ↓
object queries
      ↓
class + bbox
```

它们相对 DINOv3 更需要快速适配检测任务。

因此：

$$
\eta_{\text{detector}}
=
5\times10^{-4}
$$

$$
\eta_{\text{DINOv3}}
=
1\times10^{-5}
$$

比值：

$$
\frac{5\times10^{-4}}
{1\times10^{-5}}
=
50
$$

即：

$$
\boxed{\text{detector LR = 50 × DINOv3 LR}}
$$

---

# 11. 一个具体 update 数字例子

为了建立直觉，暂时忽略 Adam 的 adaptive denominator。

假设同样的有效更新方向：

```text
update_signal = 0.01
```

DINOv3：

$$
\Delta\theta
\approx
10^{-5}\times0.01
=
10^{-7}
$$

detector：

$$
\Delta\theta
\approx
5\times10^{-4}\times0.01
=
5\times10^{-6}
$$

后者就是前者的 50 倍。

真实 AdamW 更新不能这样简单计算，但这个例子很好地解释了 param-group LR 的意图。

---

# 12. 为什么 norm / BN / bias 常设 `weight_decay=0`？

对普通卷积/线性 weight：

```text
weight decay
→ 抑制参数无限增大
→ regularization
```

但：

```text
bias
normalization scale
normalization shift
```

的语义不同。

对这些参数施加相同的 L2-style shrinkage 往往没有必要，因此现代 Transformer / detector recipe 经常：

```text
norm / bn / bias:
weight_decay = 0
```

DEIMv2 的 regex 参数组正是在做这件事。

---

# 13. Regex 参数组必须会读

例如：

```regex
^(?=.*.dinov3)(?!.*(?:norm|bn|bias)).*$
```

教学上拆成：

```text
(?=.*.dinov3)
    必须包含 dinov3

(?!.*(?:norm|bn|bias))
    不能包含 norm / bn / bias
```

因此它抓的是：

```text
DINOv3 中的普通 weights
```

---

# 14. 第二个 regex

```regex
^(?=.*.dinov3)(?=.*(?:norm|bn|bias)).*$
```

意思：

```text
必须属于 dinov3
AND
名字中包含 norm / bn / bias
```

给：

```text
lr = 1e-5
weight_decay = 0
```

---

# 15. 第三个 regex

```regex
^(?=.*(?:sta|encoder|decoder))
 (?=.*(?:norm|bn|bias)).*$
```

意思：

```text
属于 STA / encoder / decoder
AND
是 norm / bn / bias
```

给：

```text
WD = 0
```

LR 若未在组内覆盖，就使用 optimizer 默认 LR。

---

# 16. 为什么必须实际打印 param groups？

Regex 最危险的 bug 是：

```text
你以为某参数进入 Group A
实际进入 Group D
```

因此不要只看 YAML。

应该验证：

```python
for name, p in model.named_parameters():
    ...
```

最终输出：

```text
group 0
lr
wd
parameter count
sample parameter names
```

后面给完整实验。

---

# 17. `deim_dfine_x` 为什么也要 differential LR？

HGNetv2 路线虽然不是 DINOv3，但 backbone 同样通常来自预训练。

DEIM / DEIMv2 HGNetv2 大型号常见：

```text
backbone LR << detector LR
```

例如公开 L 配置：

```text
backbone:
2.5e-5

detector:
5e-4
```

比值：

$$
20
$$

所以：

```text
小 backbone LR
```

并不是 DINOv3 独有，而是 pretrained backbone finetuning 的通用思想。

---

# 18. Learning Rate 到底控制什么？

简单 SGD：

$$
\theta_{t+1}
=
\theta_t-\eta g_t
$$

LR 太小：

```text
学得慢
很久都没适配 detection
```

LR 太大：

```text
loss oscillation
NaN
pretrained feature 被破坏
matching 不稳定
```

所以 LR 不是：

```text
越大越快
```

而是：

```text
稳定性与优化速度之间的平衡
```

---

# 19. 为什么需要 Warmup？

训练最开始：

```text
detector head 尚未稳定
query prediction 很差
Hungarian matching 变化剧烈
Adam moments 尚未积累
```

如果 step 0 就使用完整 LR：

```text
update 可能过猛
```

Warmup：

```text
very small LR
     ↓
gradually increase
     ↓
target LR
```

---

# 20. Linear Warmup 数学

一个简化版本：

$$
\eta_t
=
\eta_{\max}
\frac{t}{T_w},
\qquad
0\le t<T_w
$$

如果：

```text
target LR = 5e-4
warmup = 2000 iterations
```

则：

```text
iter 0       ≈ 0
iter 500     ≈ 1.25e-4
iter 1000    ≈ 2.5e-4
iter 1500    ≈ 3.75e-4
iter 2000    ≈ 5e-4
```

实际仓库 warmup 的起始 factor / scheduler 实现应以本地源码为准。

---

# 21. `warmup_iter` 与 `warmup_duration` 不要混淆

DEIM 系列配置中你可能同时看到：

```text
warmup_iter
```

与：

```yaml
lr_warmup_scheduler:
  warmup_duration: ...
```

它们属于不同层次/版本的 scheduler 配置。

读源码时必须确认：

```text
当前 Solver 实际读取哪个字段？
flatcosine 分支是否覆盖普通 scheduler？
warmup scheduler 在 epoch 还是 iteration 更新？
```

不要仅凭变量名推断。

---

# 22. Flat-Cosine：为什么叫 Flat？

直觉：

```text
LR
^
|          ┌───────────────┐
|         /                │
| warmup /                 │
|       /                  │
|______/                   ╲
|                           ╲
|                            ╲____
+------------------------------------> training
```

可以理解成三个阶段：

```text
① warmup
② flat / high-LR phase
③ cosine decay
```

DEIM 配置里：

```text
lrsheduler: flatcosine
flat_epoch: ...
lr_gamma: ...
```

控制这套 schedule。

---

# 23. Cosine decay 的经典形式

一个标准 cosine：

$$
\eta(t)
=
\eta_{\min}
+
\frac12
(\eta_{\max}-\eta_{\min})
\left[
1+\cos\left(
\pi\frac{t-T_s}{T-T_s}
\right)
\right]
$$

其中：

```text
Ts = cosine 开始位置
T  = decay 结束位置
```

DEIMv2 的具体 flatcosine 公式与 `lr_gamma` 的用法必须以当前 scheduler 实现为准。

本课先掌握：

```text
warmup → flat → cosine decay
```

的设计目的。

---

# 24. 为什么 DETR 类模型喜欢较长的 flat 阶段？

DETR 训练不是普通分类：

```text
model predictions
     ↓
Hungarian matching
     ↓
labels for queries
     ↓
gradient
```

early predictions 一变：

```text
assignment 也可能变
```

因此训练动态本身更复杂。

维持一段稳定 LR，有利于：

```text
query specialization
matching stabilization
detector adaptation
```

后期再降低 LR 做 refinement。

---

# 25. 为什么最后还有 no-aug epoch？

第 17 课已经学过：

```text
Mosaic / MixUp / CopyBlend
```

会制造与真实测试图不同的训练分布。

所以后期：

```text
strong augmentation OFF
multi-scale OFF
smaller LR
```

让模型专注：

```text
clean-image localization
confidence calibration
fine refinement
```

---

# 26. DEIMv2 README 的 epoch 设计非常重要

官方 README 给出一个 finetune 示例：

```yaml
epoches: 32
# 20 training + 12 EMA/no-aug related tail

flat_epoch: 14
no_aug_epoch: 12

policy:
  epoch: [4, 14, 20]

mixup_epochs: [4, 14]
stop_epoch: 20
copyblend_epochs: [4, 20]
```

这告诉我们：

```text
total epochs
≠
strong augmentation training epochs
```

后面保留了一段特殊的 refinement / EMA tail。

---

# 27. 现在进入 EMA

EMA：

**Exponential Moving Average**

假设：

```text
θ_t = 当前训练模型参数
\bar θ_t = EMA 参数
```

更新：

$$
\bar\theta_t
=
d\bar\theta_{t-1}
+
(1-d)\theta_t
$$

其中：

$$
d=\text{decay}
$$

---

# 28. `decay=0.9999` 是什么意思？

$$
\bar\theta_t
=
0.9999\bar\theta_{t-1}
+
0.0001\theta_t
$$

当前 step 参数只贡献：

```text
0.01%
```

但 EMA 不断累计，因此得到一个非常平滑的参数轨迹。

---

# 29. EMA 不是“平均最近 N 个 checkpoint”

它是每次更新：

```text
old EMA
× decay
+
current model
× (1-decay)
```

历史参数的权重指数衰减。

距离当前 k steps 的贡献近似：

$$
(1-d)d^k
$$

---

# 30. EMA 的有效时间尺度

一个常用粗略直觉：

$$
N_{\text{eff}}
\approx
\frac{1}{1-d}
$$

如果：

$$
d=0.9999
$$

则：

$$
N_{\text{eff}}
\approx
10000
$$

steps。

如果：

$$
d=0.9998
$$

则：

$$
N_{\text{eff}}
\approx
5000
$$

steps。

这就是为什么 batch size 改变后，EMA decay 也值得重新调整。

---

# 31. 为什么 EMA 常常能提高检测模型评估稳定性？

SGD / AdamW 训练轨迹：

```text
θ0 → θ1 → θ2 → θ3 → ...
```

每一步都有：

```text
mini-batch noise
augmentation noise
matching noise
```

EMA：

```text
对参数轨迹做低通滤波
```

因此通常：

```text
evaluation variance ↓
final generalization 更稳定
```

---

# 32. 为什么 DEIMv2 特别值得使用 EMA？

DEIMv2 训练中噪声来源很多：

```text
Mosaic
MixUp
CopyBlend
multi-scale
Hungarian assignment
denoising queries
DINOv3 finetuning
```

参数更新本身波动较强。

EMA 可以把：

```text
short-term optimizer noise
```

与：

```text
long-term learned solution
```

部分分离。

---

# 33. EMA warmup

训练最初：

```text
model 变化很快
```

如果从第一个 step 就用极高 decay：

```text
EMA 会过度滞后
```

所以很多实现使用：

```text
EMA warmup
```

让 decay 从较小值逐步靠近目标 decay。

当前仓库 runtime 中常见：

```text
ema_decay ≈ 0.9999
ema_warmups ≈ 1000 / 2000
```

具体值以当前 merged config 为准。

---

# 34. DEIMv2 的 `ema_restart_decay`

第 17 课看到 collate 配置里还有：

```yaml
ema_restart_decay: 0.9999
```

这说明数据增强阶段结束时，训练系统不仅切换：

```text
augmentation
multi-scale
```

还可能协调：

```text
EMA behavior
```

所以 DEIMv2 的 no-aug tail 不是“只关 Mosaic”那么简单，而是一套后期 refinement recipe。

具体 restart 条件和 EMA 参数重置逻辑要以你本地 solver / collate 源码为准。

---

# 35. Batch Size 为什么会影响 EMA？

假设数据集固定。

Global batch 从：

```text
32 → 64
```

每个 epoch 的 optimizer steps 约减半。

若 EMA decay 不变：

```text
EMA 按 step 计的时间尺度没变
```

但：

```text
按“看过多少样本”计的时间尺度变了
```

所以官方 README 明确建议在 batch size 翻倍时同步调整 EMA decay / warmup。

---

# 36. 官方 batch-size scaling 示例

README 以 DEIMv2-S：

```text
total batch 32 → 64
```

为例，建议：

```text
LR 按 linear scaling law 调整
EMA decay 调整
EMA warmup 减半
LR warmup duration 减半
```

其中给出的 EMA decay 规则：

$$
d_{\text{new}}
=
1-(1-d_{\text{old}})\times2
$$

若：

$$
d_{\text{old}}=0.9999
$$

则：

$$
d_{\text{new}}
=
1-0.0001\times2
=
0.9998
$$

---

# 37. 为什么 warmup steps 要减半？

batch 翻倍：

```text
每一步看到的样本 ×2
```

如果希望 warmup 覆盖的“样本总量”大致不变：

```text
warmup steps ÷2
```

例如：

```text
batch 32 × 500 steps
=
16000 samples
```

batch 64：

```text
64 × 250
=
16000 samples
```

这就是官方示例中 warmup duration 减半的直觉。

---

# 38. Linear Scaling Rule

常见经验：

$$
\eta_{\text{new}}
=
\eta_{\text{old}}
\frac{B_{\text{new}}}
{B_{\text{old}}}
$$

如果：

```text
B: 32 → 64
```

则：

```text
LR ×2
```

但这是经验法则，不是数学定理。

对：

```text
foundation backbone
detector head
very small datasets
unstable matching
```

仍需要实际验证。

---

# 39. 为什么 X 的 DINOv3 LR 更不能机械线性放大？

X backbone：

```text
pretrained DINOv3
```

检测模块：

```text
task-specific
```

二者对 LR 的敏感度不同。

所以你在自定义数据集上改 batch size 时，应该分别观察：

```text
backbone gradient norm
detector gradient norm
validation AP
feature drift
```

而不是简单：

```text
所有 param group LR × 同一个倍数
```

虽然官方 COCO recipe 会给出可参考的 linear-scaling 起点。

---

# 40. 现在进入 AMP

AMP：

**Automatic Mixed Precision**

核心思想：

```text
某些计算用 FP16 / BF16
某些敏感计算保留 FP32
```

PyTorch 自动选择合适精度。

DEIMv2 官方训练命令对 ViT / HGNetv2 都展示：

```bash
--use-amp
```

---

# 41. FP32 与 FP16

FP32：

```text
1 sign
8 exponent
23 fraction
```

FP16：

```text
1 sign
5 exponent
10 fraction
```

FP16：

```text
memory ↓
Tensor Core throughput ↑
dynamic range ↓
precision ↓
```

所以不能把整个训练粗暴地全部 `.half()`。

---

# 42. autocast 做什么？

典型：

```python
with torch.autocast(
    device_type="cuda",
    dtype=torch.float16
):
    outputs = model(images, targets)
    loss = ...
```

autocast 会根据算子选择合适 dtype。

例如很多：

```text
matmul
conv
```

可以用低精度加速；

某些数值敏感操作会保留更高精度。

---

# 43. 为什么还需要 GradScaler？

FP16 能表示的非常小的数有限。

假设某梯度：

$$
g=10^{-8}
$$

可能在 FP16 中：

```text
underflow → 0
```

那么参数就收不到这个梯度。

GradScaler 的思想：

$$
L'=sL
$$

则：

$$
\nabla L'
=
s\nabla L
$$

把很小的 gradient 临时放大。

---

# 44. 一个数字例子

原 gradient：

$$
10^{-8}
$$

scale：

$$
s=65536
$$

则 scaled gradient：

$$
6.5536\times10^{-4}
$$

这就更容易在 FP16 范围中保留下来。

optimizer.step 前再：

```text
unscale
```

恢复真实梯度尺度。

---

# 45. AMP 的正确操作顺序

非常重要：

```text
autocast forward
     ↓
scaler.scale(loss).backward()
     ↓
scaler.unscale_(optimizer)
     ↓
gradient clipping
     ↓
scaler.step(optimizer)
     ↓
scaler.update()
```

为什么 clip 前要 unscale？

因为如果 gradient 还被乘了 65536：

```text
你看到的 gradient norm 是假的
```

clipping 阈值就失去意义。

---

# 46. Gradient Clipping

runtime 中常见：

```text
clip_max_norm = 0.1
```

global norm：

$$
\|g\|_2
=
\sqrt{
\sum_i
\|g_i\|_2^2
}
$$

如果：

$$
\|g\|_2>C
$$

则按比例缩小：

$$
g_i'
=
g_i
\frac{C}{\|g\|_2}
$$

这里：

$$
C=0.1
$$

---

# 47. 为什么 DETR / Transformer 特别常用 clipping？

可能出现：

```text
attention instability
difficult Hungarian assignments
strong augmentation
noisy hard examples
denoising queries
```

某个 batch 产生异常大的 gradient。

Gradient clipping 相当于：

```text
给单步 update 加一个保险丝
```

防止少数 batch 破坏训练。

---

# 48. AMP 是否改变最终数学目标？

理想情况下：

```text
不改变 loss 定义
不改变 optimizer 目标
```

它主要改变：

```text
数值表示
计算精度
吞吐
显存
```

但实际浮点运算不是精确实数，所以：

```text
AMP 与 FP32 不会逐 bit 相同
```

这很正常。

---

# 49. BF16 为什么值得知道？

BF16：

```text
exponent bits 与 FP32 接近
fraction 较少
```

因此：

```text
dynamic range 比 FP16 更大
```

新 GPU 上训练大型 Transformer 时很常用。

但 DEIMv2 当前命令行 `--use-amp` 最终采用哪种 autocast dtype，要以当前代码和硬件路径为准。

---

# 50. 现在进入 DDP

DDP：

**DistributedDataParallel**

官方训练命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
--master_port=7777 \
--nproc_per_node=4 \
train.py \
-c configs/deimv2/deimv2_dinov3_x_coco.yml \
--use-amp \
--seed=0
```

这意味着：

```text
4 GPUs
4 processes
1 process / GPU
```

---

# 51. rank 是什么？

4 个进程：

```text
rank 0 → GPU0
rank 1 → GPU1
rank 2 → GPU2
rank 3 → GPU3
```

每个 rank：

```text
有自己的 model replica
读取自己的 mini-batch shard
做自己的 forward/backward
```

backward 时同步 gradients。

---

# 52. DDP 的梯度同步

假设两张 GPU：

```text
GPU0 gradient = g0
GPU1 gradient = g1
```

all-reduce 后常见语义相当于平均：

$$
g=
\frac{g_0+g_1}{2}
$$

然后两张卡：

```text
optimizer.step()
```

使用相同同步梯度，所以参数继续保持一致。

---

# 53. DDP 不是“训练 4 个不同模型”

每张 GPU 虽然有 model replica：

```text
θ0
θ1
θ2
θ3
```

但梯度每一步同步，因此：

```text
optimizer step 后它们保持一致
```

逻辑上仍然是一份模型。

---

# 54. Global Batch Size

假设：

```text
world_size = 4
per-GPU batch = 8
```

则：

$$
B_{\text{global}}
=
4\times8
=
32
$$

如果 8 GPUs：

$$
8\times8=64
$$

这就是为什么：

```text
GPU 数变化
```

可能导致：

```text
global batch 变化
→ LR recipe 变化
```

---

# 55. `total_batch_size` 与 per-GPU batch

DEIMv2 配置通常用：

```yaml
train_dataloader:
  total_batch_size: ...
```

框架会结合：

```text
world_size
```

分配每个 rank 的 batch。

所以看到：

```text
total_batch_size=32
```

在 4 GPUs 下，通常意味着：

```text
~8 images / GPU
```

具体必须看 DataLoader builder 是否要求整除以及 drop_last 行为。

---

# 56. DistributedSampler 为什么需要？

如果 4 个 rank 都遍历完整 dataset：

```text
同一张图会被训练 4 次
```

DDP 就失去意义。

DistributedSampler 把数据分片：

```text
dataset
├─ shard rank0
├─ shard rank1
├─ shard rank2
└─ shard rank3
```

每个 epoch 再通过：

```python
sampler.set_epoch(epoch)
```

改变 shuffle 顺序。

---

# 57. 为什么 `set_epoch()` 对 DEIMv2 更重要？

DEIMv2 不只是 sampler 需要 epoch。

第 17 课已经看到：

```text
Mosaic policy
MixUp window
CopyBlend window
multi-scale stop
```

都依赖：

```text
current epoch
```

所以 epoch 必须传播到：

```text
sampler
dataset transforms
collate function
criterion / matcher（如果有 schedule）
```

这就是为什么 DEIMv2 的训练循环比普通 PyTorch classification 更有状态。

---

# 58. SyncBatchNorm 是什么？

普通 BatchNorm 在每张 GPU 上分别计算：

```text
mean / variance
```

如果：

```text
per-GPU batch 很小
```

统计量可能不稳定。

SyncBN：

```text
跨 GPU 同步 BN statistics
```

得到更接近 global batch 的统计。

runtime 中存在：

```text
sync_bn
```

这样的配置。

---

# 59. DINOv3 路线是否特别依赖 SyncBN？

ViT 主体大量使用：

```text
LayerNorm
```

LayerNorm 不依赖 batch statistics。

但：

```text
STA
encoder
convolutional components
```

仍可能含 BN。

因此 SyncBN 是否有明显收益要看实际模型结构和 per-GPU batch。

---

# 60. `find_unused_parameters`

DDP 需要知道：

```text
哪些参数参与当前 backward graph
```

某些条件分支可能导致：

```text
某参数这一轮没被使用
```

`find_unused_parameters=True` 可以帮助 DDP 处理这类情况，但会有额外 graph traversal 开销。

如果模型所有 trainable parameters 每次都参与：

```text
False
```

通常更高效。

---

# 61. DDP + EMA：EMA 应该在哪个模型上更新？

DDP 包装后常见：

```text
model.module
```

才是底层原始 model。

EMA 需要从同步后的训练参数更新。

由于各 rank 的 model 参数在 optimizer step 后应一致：

```text
每个 rank 理论上可得到一致 EMA
```

但 checkpoint / evaluation 通常只需要 rank0 负责保存和日志。

具体 DEIMv2 对 `de_parallel(model)` / EMA 的处理以 solver 源码为准。

---

# 62. 为什么只有 rank0 打印 / 保存？

如果 8 个 rank 都：

```text
print()
save checkpoint
write TensorBoard
```

会得到：

```text
8 倍重复日志
文件写冲突
```

所以分布式框架常用：

```text
is_main_process()
```

控制：

```text
checkpoint
log
evaluation summary
```

---

# 63. AMP + DDP + AdamW + EMA 的正确总体顺序

```text
rank-local batch
      ↓
autocast forward
      ↓
loss
      ↓
scaler.scale(loss).backward()
      ↓
DDP gradient synchronization
      ↓
scaler.unscale_(optimizer)
      ↓
gradient clipping
      ↓
scaler.step(optimizer)
      ↓
scaler.update()
      ↓
LR update（按仓库 schedule 时机）
      ↓
EMA update
```

scheduler 到底在 optimizer step 后还是 epoch 边界更新，要以当前 DEIMv2 solver 实现为准。

---

# 64. 一次训练 iteration 的“数据 + 参数”双路线

```text
DATA ROUTE

images
  ↓
Mosaic / MixUp / CopyBlend
  ↓
model
  ↓
predictions
  ↓
matcher
  ↓
criterion
  ↓
loss


PARAMETER ROUTE

loss
  ↓
gradients
  ↓
AMP unscale
  ↓
clipping
  ↓
AdamW
  ↓
updated model
  ↓
EMA
```

第 15–17 课主要学 DATA/SUPERVISION route；
本课主要学 PARAMETER route。

---

# 65. PyTorch 实验 1：观察 AdamW 的参数更新

```python
import torch

w = torch.nn.Parameter(
    torch.tensor([1.0])
)

opt = torch.optim.AdamW(
    [w],
    lr=1e-2,
    betas=(0.9, 0.999),
    weight_decay=0.1,
)

for step in range(5):
    opt.zero_grad()

    loss = (w - 3.0).pow(2).sum()
    loss.backward()

    print(
        "before",
        step,
        "w=", w.item(),
        "grad=", w.grad.item(),
    )

    opt.step()

    print(
        "after ",
        step,
        "w=", w.item(),
    )
```

观察：

```text
gradient
与
parameter update
```

并不是简单：

```text
update = -lr * grad
```

因为 AdamW 有 moments 和 decay。

---

# 66. 实验 2：比较有无 weight decay

```python
import torch

p1 = torch.nn.Parameter(
    torch.tensor([1.0])
)
p2 = torch.nn.Parameter(
    torch.tensor([1.0])
)

o1 = torch.optim.AdamW(
    [p1],
    lr=1e-2,
    weight_decay=0.1
)

o2 = torch.optim.AdamW(
    [p2],
    lr=1e-2,
    weight_decay=0.0
)

for _ in range(100):
    p1.grad = torch.zeros_like(p1)
    p2.grad = torch.zeros_like(p2)

    o1.step()
    o2.step()

print(p1.item())
print(p2.item())
```

即使 gradient 为 0：

```text
有 WD 的参数仍会逐步缩小
```

这就是 decoupled weight decay 最直观的实验。

---

# 67. 实验 3：模拟 differential LR

```python
import torch
import torch.nn as nn

backbone = nn.Linear(4, 4)
head = nn.Linear(4, 2)

opt = torch.optim.AdamW([
    {
        "params": backbone.parameters(),
        "lr": 1e-5,
    },
    {
        "params": head.parameters(),
        "lr": 5e-4,
    }
])

for i, g in enumerate(
    opt.param_groups
):
    print(
        i,
        g["lr"]
    )
```

输出：

```text
0  1e-05
1  0.0005
```

正好对应 X 模型 50× LR 差异的核心思想。

---

# 68. 实验 4：打印 DEIMv2 真实 optimizer groups

在你本地仓库初始化：

```python
model = cfg.model
optimizer = cfg.optimizer
```

后，先建立 parameter id → name：

```python
id_to_name = {
    id(p): name
    for name, p
    in model.named_parameters()
}
```

再：

```python
for gi, group in enumerate(
    optimizer.param_groups
):
    print(
        f"\nGROUP {gi}",
        "lr=", group["lr"],
        "wd=", group["weight_decay"],
        "n=", len(group["params"]),
    )

    names = [
        id_to_name.get(
            id(p),
            "<unknown>"
        )
        for p in group["params"]
    ]

    for name in names[:20]:
        print("  ", name)
```

这是本课最推荐的源码实验。

---

# 69. 实验 5：检查所有 trainable 参数是否恰好进入一个 group

```python
from collections import Counter

assigned = []

for group in optimizer.param_groups:
    assigned += [
        id(p)
        for p in group["params"]
    ]

counter = Counter(assigned)

trainable = {
    id(p): name
    for name, p
    in model.named_parameters()
    if p.requires_grad
}

missing = [
    name
    for pid, name
    in trainable.items()
    if pid not in counter
]

duplicate = [
    trainable.get(pid, str(pid))
    for pid, n
    in counter.items()
    if n != 1
]

print("missing:", missing)
print("duplicate:", duplicate)
```

理想情况：

```text
missing = []
duplicate = []
```

---

# 70. 实验 6：画一个最小 Warmup + Cosine

```python
import math

base_lr = 5e-4
warmup = 100
total = 1000
flat_end = 500
min_lr = 5e-5

def lr_at(t):
    if t < warmup:
        return (
            base_lr
            * t / warmup
        )

    if t < flat_end:
        return base_lr

    progress = (
        (t - flat_end)
        / (total - flat_end)
    )

    return (
        min_lr
        + 0.5
        * (base_lr - min_lr)
        * (
            1
            + math.cos(
                math.pi * progress
            )
        )
    )

for t in [
    0, 25, 50, 100,
    300, 500, 750, 999
]:
    print(t, lr_at(t))
```

注意：这是教学版，不等同于仓库 flatcosine 的逐行实现。

---

# 71. 实验 7：手写 EMA

```python
import torch

theta = torch.tensor(10.0)
ema = theta.clone()

decay = 0.9

values = [
    11.0,
    13.0,
    12.0,
    15.0,
]

for x in values:
    theta = torch.tensor(x)

    ema = (
        decay * ema
        + (1 - decay) * theta
    )

    print(
        "model:",
        theta.item(),
        "ema:",
        ema.item()
    )
```

你会看到：

```text
model 参数快速跳动
EMA 缓慢变化
```

---

# 72. 实验 8：比较不同 EMA decay

```python
def ema_trace(
    values,
    decay
):
    e = values[0]
    out = [e]

    for x in values[1:]:
        e = (
            decay * e
            + (1 - decay) * x
        )
        out.append(e)

    return out


values = [
    0, 1, 2, 3, 4, 5
]

print(
    ema_trace(
        values,
        0.9
    )
)

print(
    ema_trace(
        values,
        0.99
    )
)
```

decay 越大：

```text
EMA 越平滑
但响应越慢
```

---

# 73. 实验 9：验证 EMA 有效时间尺度

```python
for d in [
    0.9,
    0.99,
    0.999,
    0.9999,
]:
    print(
        d,
        1 / (1 - d)
    )
```

得：

```text
.9     → 10
.99    → 100
.999   → 1000
.9999  → 10000
```

这是理解 DEIMv2 EMA 配置非常有用的直觉。

---

# 74. 实验 10：AMP 最小训练

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(256, 512),
    nn.GELU(),
    nn.Linear(512, 80),
).cuda()

opt = torch.optim.AdamW(
    model.parameters(),
    lr=5e-4
)

scaler = torch.amp.GradScaler(
    "cuda"
)

x = torch.randn(
    32, 256,
    device="cuda"
)

y = torch.randn(
    32, 80,
    device="cuda"
)

opt.zero_grad()

with torch.autocast(
    "cuda",
    dtype=torch.float16
):
    pred = model(x)
    loss = (
        pred - y
    ).pow(2).mean()

scaler.scale(
    loss
).backward()

scaler.unscale_(opt)

torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    0.1
)

scaler.step(opt)
scaler.update()
```

PyTorch API 版本可能略有差异，以你的本地 PyTorch 为准。

---

# 75. 实验 11：观察 AMP dtype

```python
class PrintDtype(
    torch.nn.Module
):
    def forward(self, x):
        print(
            "inside:",
            x.dtype
        )
        return x @ x.T


m = PrintDtype().cuda()
x = torch.randn(
    256, 256,
    device="cuda"
)

print(
    "outside:",
    x.dtype
)

with torch.autocast(
    "cuda",
    dtype=torch.float16
):
    y = m(x)
    print(
        "result:",
        y.dtype
    )
```

这个实验帮助你理解：

```text
autocast 不是把所有输入 tensor 永久改成 FP16
```

---

# 76. 实验 12：Gradient clipping

```python
import torch

model = torch.nn.Linear(
    10, 1
)

x = torch.randn(
    32, 10
) * 1000

y = torch.randn(
    32, 1
) * 1000

loss = (
    model(x) - y
).pow(2).mean()

loss.backward()

before = torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=0.1
)

print(
    "norm before clip:",
    before.item()
)

sq = 0.0

for p in model.parameters():
    if p.grad is not None:
        sq += (
            p.grad.norm() ** 2
        ).item()

print(
    "norm after clip:",
    sq ** 0.5
)
```

---

# 77. 实验 13：Global batch 计算

```python
def global_batch(
    per_gpu,
    world_size,
    accum=1,
):
    return (
        per_gpu
        * world_size
        * accum
    )


print(
    global_batch(
        8,
        4
    )
)
```

输出：

```text
32
```

如果还有 gradient accumulation：

```text
effective global batch
=
per_gpu
× world_size
× accumulation_steps
```

---

# 78. Gradient Accumulation 与 DDP 不一样

Gradient accumulation：

```text
多个 micro-batch
gradient 累积
最后一次 optimizer.step
```

DDP：

```text
多个 GPU 并行
每个 iteration 同步 gradient
```

两者可以组合。

但 DEIMv2 默认 recipe 是否使用 accumulation，应以配置/solver 为准，不要默认存在。

---

# 79. 实验 14：LR linear scaling

```python
def scaled_lr(
    old_lr,
    old_batch,
    new_batch,
):
    return (
        old_lr
        * new_batch
        / old_batch
    )


print(
    scaled_lr(
        2.5e-5,
        32,
        64
    )
)
```

得：

```text
5e-5
```

这正是官方 README batch doubling 示例的思路。

---

# 80. 实验 15：EMA decay batch scaling

```python
old_decay = 0.9999
batch_ratio = 2

new_decay = (
    1
    - (
        1 - old_decay
    )
    * batch_ratio
)

print(new_decay)
```

得：

```text
0.9998
```

---

# 81. 为什么不能只比较 loss 判断 recipe 好坏？

两套 recipe：

```text
A: train loss 更低
B: validation AP 更高
```

完全可能。

因为：

```text
augmentation strength
EMA
weight decay
LR
```

都会改变训练 loss 的可比性。

正确观察：

```text
train loss
validation AP
AP50 / AP75
APS / APM / APL
gradient norm
LR
EMA AP
raw-model AP
```

---

# 82. 建议训练时记录哪些曲线？

至少：

```text
total loss
loss_mal
loss_bbox
loss_giou

LR per param group

gradient norm

AMP scaler value

targets / image
matched positives / image

validation AP
validation AP75

raw model AP
EMA model AP
```

这样你才能判断：

```text
问题来自 model？
matching？
data？
LR？
AMP？
optimizer？
```

---

# 83. 为什么要分别画每个 param group 的 LR？

X 至少有：

```text
DINOv3 LR
detector LR
```

如果只打印：

```python
optimizer.param_groups[0]["lr"]
```

你可能误以为：

```text
整个模型都是这个 LR
```

应该：

```python
[
    group["lr"]
    for group
    in optimizer.param_groups
]
```

全部记录。

---

# 84. 一个非常实用的 LR 日志

```python
lrs = {
    f"lr/group_{i}":
        group["lr"]
    for i, group
    in enumerate(
        optimizer.param_groups
    )
}
```

然后每个 epoch / 若干 step 写入：

```text
TensorBoard
WandB
CSV
```

都可以。

---

# 85. 监控 Gradient Norm

在 clipping 前：

```python
grad_norm = (
    torch.nn.utils
    .clip_grad_norm_(
        model.parameters(),
        max_norm
    )
)
```

这个返回值通常是：

```text
clipping 前的 total norm
```

如果你看到：

```text
大多数 step = 0.02
偶尔 = 200
```

说明某些 batch 产生异常梯度。

应继续检查：

```text
augmentation
boxes
matcher
loss
NaN/Inf
```

---

# 86. AMP scaler 也是健康指标

GradScaler 会动态：

```text
increase scale
或
decrease scale
```

如果经常检测到：

```text
inf / nan gradient
```

scale 会下降。

如果 scaler 长期快速下降：

```text
数值稳定性可能有问题
```

不能只把它当成“PyTorch 内部细节”。

---

# 87. 自定义数据集：是否应该保留 X 的 `1e-5 / 5e-4`？

它是非常好的起点，但不保证最优。

如果数据集：

```text
与 DINOv3 预训练视觉域相近
数据量小
```

可以倾向：

```text
更小 backbone LR
```

如果：

```text
domain shift 很大
数据量足够
```

可能需要：

```text
更积极 finetune backbone
```

但应该通过实验，而不是凭感觉直接放大 10 倍。

---

# 88. 一个推荐的 backbone LR ablation

保持 detector：

```text
5e-4
```

尝试：

```text
5e-6
1e-5
2e-5
5e-5
```

观察：

```text
convergence
AP
AP75
early instability
backbone gradient norm
```

不建议一开始就：

```text
DINOv3 LR = detector LR
```

---

# 89. Freeze DINOv3 与小 LR finetune 有什么区别？

Freeze：

```text
requires_grad=False
```

$$
\Delta\theta_{\text{DINO}}=0
$$

小 LR：

```text
requires_grad=True
lr=1e-5
```

$$
\Delta\theta_{\text{DINO}}\neq0
$$

只是更新很慢。

所以：

```text
freeze
≠
small LR
```

---

# 90. 为什么 freeze 能省显存？

如果 backbone 不求梯度：

```text
不需要为其参数保存 optimizer states
部分 activation graph 也可减少
```

AdamW 每个参数通常还维护：

```text
m
v
```

所以冻结大 backbone 对显存影响很明显。

但会降低适应新 domain 的能力。

---

# 91. AdamW 的 optimizer state 为什么很吃显存？

每个 trainable parameter：

```text
parameter
gradient
first moment m
second moment v
```

如果都用 FP32 粗略理解：

```text
4 bytes
+ 4 bytes
+ 4 bytes
+ 4 bytes
```

约：

```text
16 bytes / parameter
```

实际 AMP、master weights、框架实现会改变精确数字。

但结论是：

```text
AdamW state 往往是训练显存的重要部分。
```

---

# 92. AMP 为什么不能解决所有显存问题？

AMP 可以降低：

```text
activations
部分计算 tensor
```

但：

```text
optimizer states
DDP buffers
model states
```

不一定全部减半。

所以 OOM 时还要考虑：

```text
per-GPU batch
input resolution
DINOv3 train/freeze
gradient checkpointing（若支持）
model size
```

---

# 93. DDP 为什么可能比单卡更快？

如果 global batch 固定不变：

```text
每卡处理更少样本
并行计算
```

但有：

```text
gradient all-reduce communication
```

所以加速不是线性的。

大模型：

```text
compute / communication 比高
```

通常更容易获得较好 scaling。

---

# 94. 为什么小 batch + 多卡可能不划算？

假设：

```text
global batch = 8
GPUs = 8
```

每卡：

```text
batch = 1
```

此时：

```text
GPU utilization 低
communication 比例高
BN 统计困难
```

所以“GPU 越多越快”不是必然。

---

# 95. DDP 常见错误：global batch 被偷偷放大

原来：

```text
1 GPU × batch 8 = 8
```

改成：

```text
4 GPU
```

如果仍然每卡 batch 8：

```text
global batch = 32
```

此时如果 LR 不变：

```text
你实际上已经换了 training recipe
```

所以对比实验必须明确：

```text
固定 per-GPU batch？
还是固定 global batch？
```

---

# 96. 可复现实验中的 seed

官方命令：

```text
--seed=0
```

但分布式训练要兼顾：

```text
global seed
rank offset
DataLoader worker seed
sampler epoch seed
augmentation randomness
```

即使 seed 相同：

```text
CUDA kernel
distributed reduction order
library version
```

仍可能导致轻微非确定性。

所以：

```text
reproducible
≠
bitwise identical
```

---

# 97. Resume 与 Tuning 不一样

**Resume** 通常意味着恢复：

```text
model
optimizer
scheduler
EMA
scaler
epoch
```

继续同一次训练。

**Tuning / finetune** 通常意味着：

```text
加载 model weights
但重新建立新的 optimizer / schedule
```

官方命令中：

```text
-r model.pth
```

与：

```text
-t model.pth
```

对应不同语义。

精确字段以 `train.py` 参数解析和 solver load 逻辑为准。

---

# 98. 为什么 Resume 必须恢复 optimizer state？

AdamW 不只保存参数：

```text
θ
```

还保存：

```text
m_t
v_t
step
```

如果只加载 model：

```text
Adam moments 归零
```

训练动态会突然改变。

同理：

```text
LR schedule
EMA
GradScaler
```

也应恢复，才能真正 resume。

---

# 99. Checkpoint 最好包含什么？

完整训练 checkpoint：

```text
model
EMA model
optimizer
LR scheduler
warmup scheduler
GradScaler
epoch
config
```

推理 checkpoint 则可能只需要：

```text
model / EMA weights
```

两者用途不同。

---

# 100. X 模型完整训练 recipe 地图

```text
               deimv2_dinov3_x
                      │
             pretrained DINOv3
                      │
            ┌─────────┴─────────┐
            │                   │
        DINOv3 params      detector params
            │                   │
         LR 1e-5             LR 5e-4
            │                   │
            └─────────┬─────────┘
                      │
                    AdamW
                      │
              norm/bias WD=0
                      │
                LR warmup
                      │
                flat phase
                      │
               cosine decay
                      │
     ┌────────────────┼─────────────────┐
     │                │                 │
Dense O2O          AMP+Scaler          DDP
     │                │                 │
matching/loss      clip grad       all-reduce
     └────────────────┼─────────────────┘
                      │
               optimizer.step
                      │
                     EMA
                      │
            no-augmentation tail
                      │
              final evaluation
```

---

# 101. 一次 iteration 的完整伪代码

```python
model.train()

for images, targets in loader:

    images = images.to(device)
    targets = move_to_device(
        targets,
        device
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    with autocast_if_enabled():

        outputs = model(
            images,
            targets
        )

        loss_dict = criterion(
            outputs,
            targets
        )

        loss = weighted_sum(
            loss_dict
        )

    if use_amp:

        scaler.scale(
            loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        grad_norm = clip_grad_norm_(
            model.parameters(),
            max_norm
        )

        scaler.step(
            optimizer
        )

        scaler.update()

    else:

        loss.backward()

        grad_norm = clip_grad_norm_(
            model.parameters(),
            max_norm
        )

        optimizer.step()

    if use_ema:
        ema.update(model)

    update_lr_if_step_based()
```

这段是教学伪代码，不是对官方文件的逐字复制。

---

# 102. Epoch 边界又发生什么？

```python
for epoch in range(...):

    sampler.set_epoch(epoch)
    dataset.set_epoch(epoch)
    collate_fn.set_epoch(epoch)
    criterion.set_epoch(epoch)

    train_one_epoch(...)

    scheduler.step_if_epoch_based()

    evaluate(...)

    save_checkpoint(...)
```

对 DEIMv2：

```text
epoch
```

同时控制：

```text
data augmentation
matcher
multi-scale
EMA behavior
LR stage
```

因此它是整个 curriculum 的“全局时钟”。

---

# 103. 最值得你在源码里追的调用链

下一次打开仓库时，按：

```text
train.py
  ↓
YAMLConfig
  ↓
solver.fit()
  ↓
setup()
  ↓
optimizer construction
  ↓
DDP wrapping
  ↓
EMA / scaler construction
  ↓
train_one_epoch()
  ↓
autocast
  ↓
criterion
  ↓
backward
  ↓
clip
  ↓
optimizer.step
  ↓
EMA update
  ↓
scheduler
  ↓
evaluate / checkpoint
```

把这条线追完，你才真正理解“训练框架”。

---

# 104. 常见误区 1：LR 是整个模型一个数字

错。

X 至少有：

```text
DINOv3 LR
detector LR
```

还会因：

```text
norm/bias
```

有不同 weight decay。

---

# 105. 常见误区 2：Weight decay 就是把 L2 loss 加进去

对 AdamW 不应这样理解。

AdamW 的重点是：

```text
decoupled weight decay
```

参数衰减与 adaptive gradient update 分离。

---

# 106. 常见误区 3：EMA 参数也参与 backward

错。

EMA 通常：

```text
不求梯度
不被 optimizer 更新
```

它只通过：

$$
\bar\theta
\leftarrow
d\bar\theta+(1-d)\theta
$$

更新。

---

# 107. 常见误区 4：AMP 就是 `model.half()`

错。

AMP：

```text
autocast
+
GradScaler（FP16 常见）
```

动态选择计算精度。

---

# 108. 常见误区 5：GradScaler 会改变真正的 gradient

它只是：

```text
backward 前临时放大
optimizer 前再 unscale
```

理想数学目标不变。

---

# 109. 常见误区 6：clip 应在 scaled gradients 上做

错。

AMP 下应先：

```text
scaler.unscale_(optimizer)
```

再：

```text
clip_grad_norm_
```

---

# 110. 常见误区 7：4 GPU 就等于 LR 自动 ×4

不一定。

关键是：

```text
global batch 是否变了？
```

如果 global batch 不变：

```text
没理由仅因为 GPU 数增加就自动 ×4 LR。
```

---

# 111. 常见误区 8：DDP 会自动帮你选择最优 batch size

不会。

DDP 只负责：

```text
distributed execution
gradient synchronization
```

batch / LR recipe 仍由你配置。

---

# 112. 常见误区 9：EMA decay 越接近 1 越好

错。

太大：

```text
EMA 响应太慢
```

太小：

```text
平滑不足
```

而且最优值还和：

```text
batch size
steps/epoch
training length
```

有关。

---

# 113. 常见误区 10：Resume 只加载 model.pth 就行

真正 resume 应尽量恢复：

```text
optimizer
scheduler
scaler
EMA
epoch
```

否则更接近：

```text
finetune from weights
```

---

# 114. 常见误区 11：训练 loss 下降说明 LR 一定合理

不一定。

backbone LR 太大时：

```text
train loss 也可能下降
```

但：

```text
pretrained representation 被破坏
validation AP 变差
```

所以必须看 validation。

---

# 115. 常见误区 12：AMP 一定让训练快 2 倍

不保证。

速度取决于：

```text
GPU architecture
tensor shapes
dataloader bottleneck
communication
unsupported ops
memory bandwidth
```

AMP 最大价值通常是：

```text
更高吞吐 + 更低 activation memory
```

但实际倍数要 benchmark。

---

# 116. 自定义数据集的推荐调参顺序

不要同时改 10 个超参数。

建议：

```text
① 先复现官方 recipe
② 确认数据/box/matcher 正确
③ 调 global batch
④ 按比例调整 LR / warmup / EMA
⑤ 调 backbone LR
⑥ 调 detector LR
⑦ 调 weight decay
⑧ 调 augmentation schedule
⑨ 调 EMA
```

否则 AP 变化后你无法知道原因。

---

# 117. 推荐 Ablation A：DINOv3 LR

```text
detector = 5e-4

backbone:
5e-6
1e-5
2e-5
5e-5
```

记录：

```text
AP
AP75
gradient norm
early loss stability
```

---

# 118. 推荐 Ablation B：Weight Decay

```text
5e-5
1e-4
1.25e-4
2e-4
```

注意：

```text
norm / bias 仍保持 WD=0
```

比较才公平。

---

# 119. 推荐 Ablation C：EMA

```text
no EMA
0.999
0.9998
0.9999
```

同时比较：

```text
raw model AP
EMA model AP
```

不要只保存 EMA 结果。

---

# 120. 推荐 Ablation D：AMP

```text
FP32
AMP
```

比较：

```text
images/s
max GPU memory
AP
NaN frequency
```

如果 AP 基本一致而：

```text
throughput ↑
memory ↓
```

AMP 就达到了目的。

---

# 121. 推荐 Ablation E：Global Batch

```text
16
32
64
```

每次按官方 scaling 规则同步考虑：

```text
LR
warmup
EMA
```

否则你比较的不是单纯 batch size。

---

# 122. 推荐 Ablation F：Gradient Clip

```text
no clip
0.05
0.1
0.5
1.0
```

同时记录：

```text
unclipped grad norm histogram
```

如果 99% step 的 norm 都小于 0.1：

```text
clip 几乎不介入
```

如果大量 step 都被 clip：

```text
你可能需要重新检查 LR / loss scale / data。
```

---

# 123. 一张训练故障排查表

| 现象 | 优先检查 |
|---|---|
| loss 第一轮就 NaN | AMP、box 合法性、LR、grad norm |
| loss 剧烈震荡 | LR、warmup、augmentation、matcher |
| backbone 很快退化 | DINOv3 LR |
| AP75 明显差 | late LR、no-aug、bbox loss、matcher |
| raw AP 好但 EMA 差 | EMA decay/warmup/restart |
| EMA AP 好很多 | raw updates 可能噪声较大 |
| 多卡 AP 与单卡差异大 | global batch、LR scaling、BN、seed |
| AMP 下 NaN，FP32 正常 | scaler、敏感 op、LR、overflow |
| GPU 利用率低 | DataLoader、Mosaic cache、per-GPU batch |
| DDP hang | rank/collective、unused params、DataLoader 异常 |

---

# 124. 一张“谁控制什么”的总表

| 组件 | 主要控制 |
|---|---|
| LR | 单步更新尺度 |
| param groups | 不同模块的学习速度 |
| AdamW | adaptive update + decoupled WD |
| weight decay | 参数正则化 |
| warmup | early-stage stability |
| flat/cosine | 中后期优化节奏 |
| grad clipping | 极端 gradient update |
| EMA | 参数轨迹平滑 |
| AMP | 数值精度 / 显存 / 吞吐 |
| GradScaler | FP16 gradient dynamic range |
| DDP | 多 GPU 数据并行 |
| SyncBN | 跨 rank BN statistics |
| batch size | gradient noise / steps / recipe scale |

---

# 125. 本课最重要的 20 个结论

1. DEIMv2 的训练性能来自 architecture + supervision + optimization recipe 三者共同作用。
2. X 使用 AdamW。
3. X 的 DINOv3 LR 当前配置为 `1e-5`。
4. X detector 默认 LR 为 `5e-4`。
5. 二者相差 50 倍。
6. DINOv3 小 LR 是为了温和 finetune pretrained representation。
7. detector 大 LR 是为了快速学习 detection-specific mapping。
8. norm / BN / bias 常使用 `weight_decay=0`。
9. 参数组是靠 parameter-name regex 构造的，必须运行时验证。
10. Warmup 主要保护训练初期。
11. Flat-cosine 提供 warmup → stable learning → refinement 的节奏。
12. no-aug tail 与后期小 LR 共同用于精细收敛。
13. EMA 是参数的指数移动平均，不参与 backward。
14. EMA decay 与 optimizer step 数、batch size 有关。
15. AMP = autocast +（FP16 场景常见的）GradScaler，而不是简单 `.half()`。
16. AMP 下 clip gradient 前必须先 unscale。
17. DDP 是每 GPU 一个进程，并通过 all-reduce 同步 gradient。
18. 真正决定 LR scaling 的是 global batch，而不是 GPU 数本身。
19. batch size 改变时，LR / warmup / EMA 都应该一起重新考虑。
20. 真正调试训练 recipe 时，应同时记录 LR、gradient norm、AMP scale、raw AP 和 EMA AP。

---

# 126. 本课检查题

## AdamW

1. AdamW 的一阶矩和二阶矩分别是什么？
2. `beta1=0.9`、`beta2=0.999` 表示什么？
3. AdamW 为什么叫 decoupled weight decay？
4. 为什么 norm/bias 常不做 weight decay？

## Param Groups

5. X 的 DINOv3 LR 是多少？
6. detector LR 是多少？
7. 相差多少倍？
8. 为什么不统一使用 `5e-4`？
9. Regex param group 最大的工程风险是什么？

## Scheduler

10. Warmup 解决什么问题？
11. Flat-cosine 可以分成哪三个阶段？
12. no-aug tail 为什么通常配合较小 LR？

## EMA

13. 写出 EMA 公式。
14. decay 越大，EMA 响应更快还是更慢？
15. `0.9999` 的粗略有效时间尺度是多少 steps？
16. batch 翻倍为什么要重新考虑 decay？

## AMP

17. autocast 与 GradScaler 分别做什么？
18. 为什么需要 loss scaling？
19. 为什么 gradient clipping 前必须 unscale？
20. AMP 是否保证与 FP32 bitwise 一致？

## DDP

21. 4 GPU DDP 通常有几个进程？
22. `8 images/GPU × 4 GPU` 的 global batch 是多少？
23. DDP 同步的是参数还是 gradient？
24. 为什么 DistributedSampler 必须 `set_epoch()`？
25. SyncBN 解决什么问题？

## 工程

26. Resume 为什么应该恢复 optimizer state？
27. Tuning 与 Resume 的核心区别是什么？
28. 为什么应该同时保存 raw model 和 EMA model 的评估结果？
29. OOM 时 AMP 之外还能调什么？
30. 如果 AMP 下 NaN 而 FP32 正常，你会按什么顺序排查？

---

# 127. 答案速记

```text
1.
m = gradient EMA
v = squared-gradient EMA

2.
控制两个 moment 的指数平滑时间尺度。

3.
WD 直接作用于参数，不混入 adaptive gradient。

4.
这些参数的语义不同，通常不需要相同的 shrinkage。

5.
1e-5

6.
5e-4

7.
50×

8.
DINOv3 已预训练，过大 LR 容易破坏 representation；
detector 需要更快适配任务。

9.
参数可能进入错误 group；必须运行时打印验证。

10.
防止训练初期 full LR 造成不稳定。

11.
warmup → flat → cosine decay

12.
clean data + small LR 用于后期 refinement。

13.
ema = d * ema + (1-d) * model

14.
更慢。

15.
~10000 optimizer steps。

16.
batch 变化会改变每 epoch 的 optimizer step 数和每 step 的样本量。

17.
autocast 选择运算精度；
scaler 保护 FP16 小梯度。

18.
防止 gradient underflow。

19.
否则 norm 仍带 scale，clip 阈值失真。

20.
不保证。

21.
4。

22.
32。

23.
backward 期间 all-reduce gradients；
因此各 replica 随后保持一致。

24.
每 epoch 改变 deterministic shuffle partition。

25.
小 per-GPU batch 时跨 GPU 聚合 BN statistics。

26.
AdamW 的 m/v/step 是训练状态的一部分。

27.
Resume 恢复训练状态；
tuning 通常只继承权重并开启新优化过程。

28.
才能知道提升来自训练模型本身还是 EMA 平滑。

29.
per-GPU batch、resolution、冻结 backbone、model size 等。

30.
检查 scaler/overflow → grad norm → LR → loss/box →
数值敏感 op → PyTorch/CUDA 版本。
```

---

# 128. 下一课预告

第 19 课：

# 《ViT 与 DINOv3——从 CNN 世界切换到 Foundation Model》

从这一课开始，我们正式进入第二条重点主线：

```text
deimv2_dinov3_x
```

会从：

```text
[B,3,640,640]
```

开始推：

```text
patch_size = 16
↓
40 × 40 patches
↓
1600 patch tokens
↓
positional information
↓
ViT blocks
↓
block 5 / 8 / 11
↓
STA
```

并重点解释：

```text
为什么 DINOv3 feature 很强，
却不能像 HGNetv2 那样直接给 detector
1/8、1/16、1/32 三层 feature map？
```

这会直接为第 20 课 STA 铺路。

---

# 129. 源码核对与版本说明

本讲义按 2026-10 前后可访问的 DEIMv2 官方公开仓库内容核对，重点参考：

- `configs/deimv2/deimv2_dinov3_x_coco.yml`
- DEIMv2 `README.md`
- `configs/base/deimv2.yml` 的当前公开配置语义
- DEIM / DEIMv2 runtime 输出中公开可见的 AMP、EMA、clip、scheduler 字段

**需要特别注意：**

1. DEIMv2 仍在迭代，不同 commit 的 solver / scheduler / EMA restart 细节可能变化。
2. X YAML 通过 `__include__` 继承 `runtime.yml`、`base/optimizer.yml`、`base/deimv2.yml`，因此单看 X 文件不是最终 merged config。
3. `flatcosine` 的精确逐 step 公式必须以你本地版本的 solver/scheduler 实现为准；本课中的 cosine 公式是教学模型。
4. EMA restart 的精确触发和 decay 更新必须以你本地 solver / collate 代码为准。
5. AMP API 随 PyTorch 版本有变化，实验代码按现代 `torch.autocast` / `torch.amp.GradScaler` 思路书写。
6. 官方 README 的 batch scaling 示例是 recipe 起点，不是对所有数据集都保证最优的数学定律。

学习优先级始终是：

```text
本地 commit 源码
    ↓
merged YAML
    ↓
runtime print
    ↓
官方 README / paper
    ↓
本讲义
```

---

# 130. 参考资料

1. DEIMv2 official repository  
   https://github.com/Intellindust-AI-Lab/DEIMv2

2. DEIMv2 X config  
   https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/configs/deimv2/deimv2_dinov3_x_coco.yml

3. DEIMv2 README — training / batch-size scaling / epoch customization  
   https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/README.md

4. PyTorch AdamW documentation  
   https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

5. PyTorch AMP documentation  
   https://pytorch.org/docs/stable/amp.html

6. PyTorch DistributedDataParallel documentation  
   https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html

7. AdamW paper — *Decoupled Weight Decay Regularization*  
   https://arxiv.org/abs/1711.05101

---

## 本课最终记忆图

```text
                   TRAINING RECIPE
                          │
        ┌─────────────────┼──────────────────┐
        │                 │                  │
     Optimization      Stability          Scale
        │                 │                  │
      AdamW             Warmup              DDP
      LR groups         Grad Clip         Global Batch
      Weight Decay      AMP/Scaler        SyncBN
      Flat-Cosine       EMA               Sampler
        │                 │                  │
        └─────────────────┼──────────────────┘
                          ▼
                    model update
                          │
                          ▼
                       EMA model
                          │
                          ▼
                 no-aug refinement
                          │
                          ▼
                    final detector
```

当你看到 X 配置中的：

```yaml
lr: 0.00001
...
lr: 0.0005
betas: [0.9, 0.999]
weight_decay: 0.000125
```

不应该再把它看成几个孤立的超参数。

你应该立即想到：

```text
pretrained DINOv3
     ↓ small LR

detection modules
     ↓ large LR

AdamW
     ↓
warmup / flat / cosine
     ↓
AMP + clipping
     ↓
DDP synchronized gradients
     ↓
optimizer step
     ↓
EMA smoothing
     ↓
no-aug refinement
```

这才是 DEIMv2 的完整训练 Recipe。
