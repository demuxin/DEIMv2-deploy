# DEIMv2 源码课程 · 第 8 课

# 第一次完整读 `deim_dfine_x` 配置——从一份 YAML 还原整个模型与训练系统

> **本课定位**
> 前 7 课解决的是“看懂检测模型需要哪些基础”。从这一课开始，我们正式进入第一条重点源码主线：**DEIM + D-FINE-X**。
>
> 本课暂时不深入 HGNetv2、HybridEncoder、DFINETransformer 每个算子的内部实现——它们分别会在第 9～13 课拆开。本课要完成一个更基础、也更重要的能力：
>
> **看到一份 `deim_hgnetv2_x_coco.yml`，能够把它还原成真正会被 Python 创建出来的模型、损失、数据、优化器和训练策略。**
>
> 这也是以后阅读 `deimv2_dinov3_x_coco.yml` 的基础。

---

## 0. 先纠正一个命名问题：我们说的 `deim_dfine_x` 到底是什么？

课程里为了方便，我们一直把第一条重点路线简称为：

```text
deim_dfine_x
```

但在官方 DEIM 仓库中，更准确的配置关系是：

```text
configs/deim_dfine/
├── dfine_hgnetv2_x_coco.yml
└── deim_hgnetv2_x_coco.yml
```

其中：

```text
dfine_hgnetv2_x_coco.yml
```

定义的是：

```text
D-FINE-X 的基础模型/训练配置
```

而：

```text
deim_hgnetv2_x_coco.yml
```

进一步：

```yaml
__include__: [
    './dfine_hgnetv2_x_coco.yml',
    '../base/deim.yml'
]
```

也就是说可以把它理解为：

```text
D-FINE-X detector
        +
DEIM training recipe
        =
DEIM-D-FINE-X
```

所以以后课程里说：

```text
deim_dfine_x
```

主要指的就是：

> **以 D-FINE-X 为 detector，本体使用 HGNetv2-B5 + HybridEncoder + DFINETransformer，再叠加 DEIM Dense O2O / MAL 等训练机制的 X 级模型。**

这一点非常重要，因为：

```text
D-FINE
```

和：

```text
DEIM
```

不是同一层面的名字。

粗略来说：

```text
D-FINE
→ detector architecture / localization design

DEIM
→ improved matching + training framework
```

因此：

```text
DEIM-D-FINE-X
```

才是我们这一阶段真正研究的对象。

---

# 1. 本课结束以后，你应该能回答什么？

看到下面命令：

```bash
python train.py \
    -c configs/deim_dfine/deim_hgnetv2_x_coco.yml \
    --use-amp
```

你应该能够解释：

```text
① train.py 怎样找到 YAML？
② YAML 为什么可以 include 其他 YAML？
③ 多份 YAML 出现同名字段时谁覆盖谁？
④ model: DEIM 为什么能变成 Python 的 DEIM(...) 对象？
⑤ backbone: HGNetv2 为什么不是普通字符串？
⑥ encoder: HybridEncoder 怎么被注入 DEIM？
⑦ decoder: DFINETransformer 怎么被注入 DEIM？
⑧ criterion 怎样变成 DEIMCriterion？
⑨ matcher 怎样进一步变成 HungarianMatcher？
⑩ optimizer 中的正则表达式怎样选择参数？
⑪ 为什么 backbone LR 与 encoder/decoder LR 不一样？
⑫ Dense O2O 的 Mosaic / MixUp 在配置树哪里进入？
⑬ 一个 YAML 最终怎样控制整个训练系统？
```

如果这些问题全部搞懂，以后看到任何 DEIM/DEIMv2 配置都会快很多。

---

# 2. 先建立“配置不是参数表，而是对象图”的认识

很多初学者第一次看 YAML，会把它理解成：

```text
一张超参数表
```

例如：

```yaml
hidden_dim: 384
num_layers: 6
lr: 0.0005
```

这种理解只对了一半。

DEIM 的配置系统更接近：

```text
YAML
↓
描述整个 Python 对象图
↓
Registry / create()
↓
实例化对象
```

例如最终需要创建：

```text
Detection Solver
│
├── model: DEIM
│   ├── backbone: HGNetv2
│   ├── encoder: HybridEncoder
│   └── decoder: DFINETransformer
│
├── criterion: DEIMCriterion
│   └── matcher: HungarianMatcher
│
├── optimizer: AdamW
│
├── train_dataloader
│   ├── CocoDetection
│   ├── transforms
│   └── collate_fn
│
├── val_dataloader
│
├── lr_scheduler
├── warmup_scheduler
├── EMA
├── AMP scaler
├── evaluator
└── postprocessor
```

所以 YAML 本质上是在描述：

# 一个依赖关系图

而不只是：

```text
key → number
```

---

# 3. 从训练命令开始

官方 DEIM 的典型训练命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
--master_port=7777 \
--nproc_per_node=4 \
train.py \
-c configs/deim_dfine/deim_hgnetv2_x_coco.yml \
--use-amp \
--seed=0
```

先只看最核心部分：

```bash
train.py -c xxx.yml
```

也就是：

```text
train.py
   │
   ▼
config path
   │
   ▼
YAMLConfig
```

`train.py` 中的核心逻辑可以概括为：

```python
cfg = YAMLConfig(args.config, **update_dict)

solver = TASKS[cfg.yaml_cfg['task']](cfg)

if args.test_only:
    solver.val()
else:
    solver.fit()
```

不要被 `train.py` 很短这件事骗了。

它只是入口。

真正复杂的东西都通过：

```python
cfg
```

被延迟创建。

---

# 4. 第一行关键源码：`YAMLConfig`

核心代码结构：

```python
cfg = YAMLConfig(args.config, **update_dict)
```

这里：

```text
args.config
```

就是：

```text
configs/deim_dfine/deim_hgnetv2_x_coco.yml
```

进入：

```text
engine/core/yaml_config.py
```

核心逻辑可以概括为：

```python
class YAMLConfig(BaseConfig):

    def __init__(self, cfg_path, **kwargs):
        super().__init__()

        cfg = load_config(cfg_path)

        cfg = merge_dict(cfg, kwargs)

        self.yaml_cfg = copy.deepcopy(cfg)
```

逐行理解。

---

## 4.1 `load_config(cfg_path)`

```python
cfg = load_config(cfg_path)
```

它不是简单：

```python
yaml.safe_load(file)
```

就结束。

因为配置可能包含：

```yaml
__include__:
```

所以它还必须递归读取其他 YAML。

最后得到一个：

```python
dict
```

例如概念上：

```python
cfg = {
    "task": "detection",

    "model": "DEIM",

    "DEIM": {...},

    "HGNetv2": {...},

    "HybridEncoder": {...},

    "DFINETransformer": {...},

    "criterion": "DEIMCriterion",

    "DEIMCriterion": {...},

    "optimizer": {...},

    "train_dataloader": {...},

    ...
}
```

---

## 4.2 `merge_dict(cfg, kwargs)`

命令行还可以覆盖 YAML。

例如：

```bash
--use-amp
```

最终会进入：

```python
update_dict
```

再：

```python
cfg = merge_dict(cfg, kwargs)
```

所以优先级概念上是：

```text
YAML defaults
     ↓
command-line override
     ↓
final config
```

这意味着：

```text
配置文件不是最终真相
```

真正训练使用的是：

```text
YAML + CLI override
```

合并后的结果。

---

# 5. X 配置不是一个文件，而是一棵 include 树

现在看真正的：

```text
deim_hgnetv2_x_coco.yml
```

核心：

```yaml
__include__: [
    './dfine_hgnetv2_x_coco.yml',
    '../base/deim.yml'
]
```

这意味着：

```text
deim_hgnetv2_x_coco.yml
│
├── dfine_hgnetv2_x_coco.yml
│
│   ├── dataset/coco_detection.yml
│   ├── runtime.yml
│   ├── base/dataloader.yml
│   ├── base/optimizer.yml
│   └── base/dfine_hgnetv2.yml
│
└── base/deim.yml
```

所以真正的配置不是 37 行。

而是：

```text
多份 YAML merge 后形成的一棵大配置树
```

这是阅读仓库时第一个非常重要的技巧：

> **不要只看当前 YAML 文件。**

看到：

```yaml
__include__:
```

就必须继续往下追。

---

# 6. 配置继承可以理解成“先给默认值，再局部覆盖”

假设基础配置：

```yaml
DFINETransformer:
    hidden_dim: 256
    num_layers: 6
    reg_scale: 4
```

X 配置：

```yaml
DFINETransformer:
    reg_scale: 8
```

最终不是：

```yaml
DFINETransformer:
    reg_scale: 8
```

而是概念上：

```yaml
DFINETransformer:
    hidden_dim: 256
    num_layers: 6
    reg_scale: 8
```

即：

```text
基础配置
+
X-specific override
```

这就是为什么一个 X 配置本身只有几十行，却能描述一个非常复杂的模型。

---

# 7. 先读基础模型：`base/dfine_hgnetv2.yml`

这份文件给出了 D-FINE + HGNetv2 路线的大量默认设置。

核心入口：

```yaml
task: detection

model: DEIM

criterion: DEIMCriterion

postprocessor: PostProcessor
```

这四行非常关键。

它们告诉框架：

```text
任务
=
detection

模型容器
=
DEIM

训练 criterion
=
DEIMCriterion

推理后处理
=
PostProcessor
```

注意：

```yaml
model: DEIM
```

不是说整个模型只有一个叫 DEIM 的神经网络层。

它更像：

```text
顶层模型容器的类名
```

---

# 8. `DEIM` 是模型容器

基础配置继续：

```yaml
DEIM:
    backbone: HGNetv2
    encoder: HybridEncoder
    decoder: DFINETransformer
```

把它翻译成 Python 直觉：

```python
model = DEIM(
    backbone=HGNetv2(...),
    encoder=HybridEncoder(...),
    decoder=DFINETransformer(...)
)
```

也就是：

```text
DEIM
│
├── HGNetv2
├── HybridEncoder
└── DFINETransformer
```

forward 的总体逻辑就是：

```python
x = self.backbone(x)
x = self.encoder(x)
x = self.decoder(x, targets)
```

所以：

```yaml
backbone: HGNetv2
```

不是：

```text
把字符串 "HGNetv2" 传进 forward
```

而是配置系统会把它解析成真正的：

```python
HGNetv2(...)
```

对象。

---

# 9. 这里开始涉及 Registry / Dependency Injection

如果手写普通 PyTorch，你可能这样写：

```python
backbone = HGNetv2(...)
encoder = HybridEncoder(...)
decoder = DFINETransformer(...)

model = DEIM(
    backbone,
    encoder,
    decoder
)
```

但仓库希望做到：

```text
只改 YAML
就能换组件
```

例如：

```yaml
backbone: HGNetv2
```

以后可以变成：

```yaml
backbone: DINOv3STAs
```

而 `train.py` 不需要重写。

这就需要：

```text
Registry
+
Factory
+
Dependency Injection
```

---

# 10. Registry 是什么？

先做一个最小版本。

```python
REGISTRY = {}

def register(cls):
    REGISTRY[cls.__name__] = cls
    return cls
```

使用：

```python
@register
class HGNetv2:
    pass

@register
class HybridEncoder:
    pass
```

之后：

```python
REGISTRY
```

可能是：

```python
{
    "HGNetv2": HGNetv2,
    "HybridEncoder": HybridEncoder,
}
```

于是：

```python
name = "HGNetv2"
cls = REGISTRY[name]
obj = cls()
```

就能：

```text
字符串
↓
找到 Python class
↓
创建 object
```

这就是 registry 最核心的思想。

---

# 11. Factory / `create()` 又是什么？

有 registry 之后，可以写：

```python
def create(name, config):
    cls = REGISTRY[name]
    kwargs = config[name]
    return cls(**kwargs)
```

例如：

```python
config = {
    "HGNetv2": {
        "name": "B5",
        "return_idx": [1, 2, 3]
    }
}
```

调用：

```python
backbone = create(
    "HGNetv2",
    config
)
```

等价于：

```python
backbone = HGNetv2(
    name="B5",
    return_idx=[1, 2, 3]
)
```

真实仓库的 `create()` 比这个复杂很多，因为还要处理：

```text
__inject__
__share__
默认参数
全局配置
嵌套对象
```

但核心思想没有变化。

---

# 12. Dependency Injection 是什么？

假设：

```python
class DEIM(nn.Module):

    __inject__ = [
        "backbone",
        "encoder",
        "decoder"
    ]

    def __init__(
        self,
        backbone,
        encoder,
        decoder
    ):
        ...
```

配置：

```yaml
DEIM:
    backbone: HGNetv2
    encoder: HybridEncoder
    decoder: DFINETransformer
```

`create("DEIM")` 时：

```text
发现 backbone 是依赖
        ↓
create("HGNetv2")
        ↓
发现 encoder 是依赖
        ↓
create("HybridEncoder")
        ↓
发现 decoder 是依赖
        ↓
create("DFINETransformer")
        ↓
最终：
DEIM(backbone, encoder, decoder)
```

所以对象创建树是递归的：

```text
create(DEIM)
│
├── create(HGNetv2)
├── create(HybridEncoder)
└── create(DFINETransformer)
```

这就是依赖注入。

---

# 13. `YAMLConfig.model` 是什么时候真正创建模型的？

这是非常漂亮的设计。

`YAMLConfig` 中：

```python
@property
def model(self):

    if self._model is None \
       and 'model' in self.yaml_cfg:

        self._model = create(
            self.yaml_cfg['model'],
            self.global_cfg
        )

    return super().model
```

假设：

```yaml
model: DEIM
```

那么第一次访问：

```python
cfg.model
```

才执行：

```python
create("DEIM", global_cfg)
```

这叫：

# lazy initialization

也就是：

```text
需要时才创建
```

而不是 YAML 一加载就把所有对象全部实例化。

---

# 14. 为什么 lazy initialization 很有用？

例如：

```python
cfg = YAMLConfig(...)
```

此时：

```text
配置已经读完
```

但模型不一定已经创建。

直到：

```python
cfg.model
```

才真正：

```text
build model
```

同样：

```python
cfg.optimizer
```

才真正 build optimizer。

```python
cfg.train_dataloader
```

才真正 build dataloader。

这样：

```text
配置解析
```

和：

```text
对象实例化
```

被分开。

这对大型训练框架非常常见。

---

# 15. 现在真正读 X Backbone 配置

`dfine_hgnetv2_x_coco.yml`：

```yaml
DEIM:
    backbone: HGNetv2

HGNetv2:
    name: 'B5'
    return_idx: [1, 2, 3]
    freeze_stem_only: True
    freeze_at: 0
    freeze_norm: True
```

但注意：

```text
DEIM 的 base/deim.yml
```

后续会覆盖部分 freeze 设置。

我们先只理解 D-FINE-X baseline。

---

# 16. `name: B5`

```yaml
HGNetv2:
    name: 'B5'
```

表示选择：

```text
HGNetv2-B5
```

也就是 X 级模型使用的较大 backbone。

第 9 课会逐层拆：

```text
Stem
↓
Stage 1
↓
Stage 2
↓
Stage 3
↓
Stage 4
```

本课先记：

```text
X
→ HGNetv2-B5
```

---

# 17. `return_idx: [1, 2, 3]`

这行非常关键：

```yaml
return_idx: [1, 2, 3]
```

表示 backbone 不只返回最后一层。

而是返回多个 stage 的 feature。

对于该路线，基础 encoder 配置定义：

```yaml
HybridEncoder:
    in_channels: [512, 1024, 2048]
    feat_strides: [8, 16, 32]
```

所以对于 640×640 输入，我们预期送入 encoder 的三个尺度是：

```text
stride 8
→ 640 / 8
→ 80

stride 16
→ 40

stride 32
→ 20
```

因此 shape 概念上：

```text
C3:
[B, 512, 80, 80]

C4:
[B, 1024, 40, 40]

C5:
[B, 2048, 20, 20]
```

这里：

```text
512 / 1024 / 2048
```

是通道数。

```text
80 / 40 / 20
```

来自 stride。

---

# 18. 第一次做真实 X 模型 shape 推导

输入：

```text
[B, 3, 640, 640]
```

HGNetv2-B5：

```text
[B, 3, 640, 640]
        │
        ▼
     HGNetv2
        │
        ├── C3 [B, 512,  80, 80]
        ├── C4 [B,1024,  40, 40]
        └── C5 [B,2048,  20, 20]
```

到这里，我们还没有进入 DETR decoder。

这些只是：

```text
CNN feature maps
```

---

# 19. `HybridEncoder` 基础配置

基础文件：

```yaml
HybridEncoder:

    in_channels: [512, 1024, 2048]

    feat_strides: [8, 16, 32]

    hidden_dim: 256

    use_encoder_idx: [2]

    num_encoder_layers: 1

    nhead: 8

    dim_feedforward: 1024

    dropout: 0.

    enc_act: 'gelu'

    expansion: 1.0

    depth_mult: 1

    act: 'silu'
```

但 X 配置覆盖：

```yaml
HybridEncoder:

    hidden_dim: 384

    dim_feedforward: 2048
```

所以最终 X 的关键参数变成：

```text
in_channels
=
[512, 1024, 2048]

hidden_dim
=
384

dim_feedforward
=
2048
```

---

# 20. 为什么 `in_channels` 和 `hidden_dim` 不一样？

输入：

```text
C3 = 512 channels
C4 = 1024 channels
C5 = 2048 channels
```

但 encoder 希望内部统一：

```text
384 channels
```

所以需要 input projection：

```text
512
 ↓
384

1024
 ↓
384

2048
 ↓
384
```

因此 HybridEncoder 输出的多尺度 feature 可以概念化为：

```text
P3:
[B,384,80,80]

P4:
[B,384,40,40]

P5:
[B,384,20,20]
```

注意：

> X 配置把 `hidden_dim` 从基础的 256 提升到 384。

这就是模型容量增大的一个来源。

---

# 21. `use_encoder_idx: [2]` 是什么意思？

基础配置：

```yaml
use_encoder_idx: [2]
```

三个尺度：

```text
index 0 → stride 8
index 1 → stride 16
index 2 → stride 32
```

所以：

```text
index 2
```

对应：

```text
20×20
```

的最低分辨率 feature。

也就是说 HybridEncoder 中昂贵的 Transformer intra-scale encoder 并不是对所有尺度都完整运行。

而是选择指定 level。

为什么？

直觉上：

```text
80×80
= 6400 positions

40×40
= 1600 positions

20×20
= 400 positions
```

Self-Attention 对 token 数非常敏感。

如果对：

```text
6400 tokens
```

做全局 attention，成本会非常高。

而：

```text
400 tokens
```

就便宜得多。

这是实时 DETR encoder 设计的重要思想。

第 10 课会彻底拆。

---

# 22. `nhead: 8`

```yaml
nhead: 8
```

如果：

```text
hidden_dim = 384
```

那么每个 head 的维度：

\[
d_
==

384/8
=====

48
\]

所以：

```text
Q/K/V

[B,N,384]
```

可以拆成：

```text
[B,8,N,48]
```

这正是第 4 课学过的 Multi-Head Attention。

---

# 23. `dim_feedforward: 2048`

Transformer encoder layer 中通常：

```text
Attention
↓
FFN
```

FFN 类似：

```text
384
↓
2048
↓
384
```

所以：

```yaml
dim_feedforward: 2048
```

控制的是 FFN 中间维度。

它不是：

```text
feature map channel
```

也不是：

```text
query 数量
```

这一点一定要区分。

---

# 24. 进入 `DFINETransformer`

基础配置：

```yaml
DFINETransformer:

    feat_channels: [256, 256, 256]

    feat_strides: [8, 16, 32]

    hidden_dim: 256

    num_levels: 3

    num_layers: 6

    eval_idx: -1

    num_queries: 300

    num_denoising: 100

    label_noise_ratio: 0.5

    box_noise_scale: 1.0

    reg_max: 32

    reg_scale: 4

    layer_scale: 1

    num_points: [3, 6, 3]

    cross_attn_method: default

    query_select_method: default
```

X 覆盖：

```yaml
DFINETransformer:

    feat_channels: [384, 384, 384]

    reg_scale: 8
```

这里有一个值得特别观察的点：

基础配置的：

```text
hidden_dim = 256
```

而 X 的 HybridEncoder：

```text
hidden_dim = 384
```

X 又明确：

```yaml
feat_channels: [384,384,384]
```

这说明 decoder 接收的是 384-channel multi-scale features，并在 decoder 内部按其实现进行投影/处理。后续第 11～13 课会以真实 constructor 和 forward 为准继续追踪，不在本课仅凭 YAML 猜内部所有 projection。

这也是源码阅读必须养成的习惯：

> **YAML 能告诉我们接口约束，但内部 tensor 怎么变换，必须继续看 Python 实现。**

---

# 25. `num_levels: 3`

```yaml
num_levels: 3
```

表示 decoder 使用：

```text
3 个 feature levels
```

也就是：

```text
P3
P4
P5
```

对应：

```text
80×80
40×40
20×20
```

flatten 后 spatial positions 总数：

\[
80\times80
+
40\times40
+
20\times20
\]

\[
==

6400+1600+400
\]

\[
==

8400
\]

所以以后你看到：

```text
8400
```

应该立刻想到：

```text
P3 + P4 + P5
flatten 后的总 spatial positions
```

---

# 26. `num_queries: 300`

```yaml
num_queries: 300
```

这就是第 5、6 课已经学习过的：

```text
object queries
```

所以正常 matching 分支最终会围绕：

```text
300 queries
```

进行检测。

典型 decoder representation：

```text
[B,300,D]
```

最后：

```text
classification
→ [B,300,num_classes]

bbox
→ [B,300,4]
```

训练时再：

```text
Hungarian O2O
```

匹配 GT。

---

# 27. `num_denoising: 100`

```yaml
num_denoising: 100
```

这是训练时额外加入的 denoising queries。

所以训练阶段不能简单认为 decoder 永远只有：

```text
300
```

个 query。

概念上：

```text
matching queries
300

+

denoising queries
100-ish configured budget
```

实际构造数量还会受到 targets 等逻辑影响。

因此：

```text
训练 query 数
```

和：

```text
推理 query 数
```

不一定完全一样。

这是 DETR 系列源码中一个很重要的区别。

---

# 28. `label_noise_ratio` 与 `box_noise_scale`

```yaml
label_noise_ratio: 0.5

box_noise_scale: 1.0
```

用于 denoising training。

直觉：

```text
从 GT 构造 noisy queries
↓
故意扰动 label / box
↓
让 decoder 学习把 noisy target 恢复回来
```

所以：

```text
label_noise_ratio
```

控制 label 噪声。

```text
box_noise_scale
```

控制 bbox 扰动尺度。

这不是第 6 课 Hungarian matching 的同一条监督路径。

---

# 29. `reg_max: 32`

```yaml
reg_max: 32
```

这是 D-FINE 最重要的配置之一。

它与：

```text
Fine-grained Distribution Refinement
```

相关。

普通 bbox regression：

```text
直接预测一个连续值
```

D-FINE 则引入：

```text
distribution-based localization representation
```

例如：

```text
reg_max = 32
```

意味着 localization distribution 的离散 bins / representation 范围与这个参数相关。

第 12 课会从：

```text
logits
↓
softmax
↓
distribution
↓
expectation
↓
distance
↓
bbox
```

完整推导。

现在只记：

```text
reg_max
≠ num_queries
≠ num_classes
```

它属于：

```text
bbox localization representation
```

---

# 30. `reg_scale: 8`

X 配置：

```yaml
DFINETransformer:
    reg_scale: 8
```

基础值：

```text
4
```

X 改为：

```text
8
```

说明 X 模型对 D-FINE localization regression 的尺度参数做了专门设置。

它的真实作用不能只看名字猜。

后面第 12 课会进入：

```text
DFINETransformer
distance2bbox
integral / distribution
reg_scale
```

真实源码继续解释。

本课先学会：

```text
看到 override
↓
知道它来自 X-specific config
↓
知道必须追进 decoder implementation
```

---

# 31. `num_points: [3, 6, 3]`

这是第 7 课刚学完的内容。

三个 feature levels：

```text
P3
P4
P5
```

采样点数：

```text
3
6
3
```

所以每个 head/query 合计：

\[
3+6+3=12
\]

个采样点。

而不是对全部：

```text
8400 positions
```

做完整 cross-attention。

因此：

```text
query
↓
reference point
↓
sampling offsets
↓
P3: 3 points
P4: 6 points
P5: 3 points
↓
weighted aggregation
```

这就是第 7 课和当前配置真正接上的地方。

---

# 32. `eval_idx: -1`

```yaml
eval_idx: -1
```

表示推理/评估使用 decoder 的指定层输出。

Python 中：

```text
-1
```

通常表示最后一层。

所以 6 层 decoder：

```text
layer 0
layer 1
layer 2
layer 3
layer 4
layer 5
          ↑
       eval_idx=-1
```

训练时可以对中间层做 auxiliary supervision，但评估通常选指定层作为最终输出。

---

# 33. 到目前为止，模型图已经可以从 YAML 还原出来

现在我们第一次不看 architecture diagram，仅凭配置还原：

```text
Input
[B,3,640,640]

        │
        ▼

HGNetv2-B5

        │
        ├── [B,512,80,80]
        ├── [B,1024,40,40]
        └── [B,2048,20,20]

        │
        ▼

HybridEncoder
hidden_dim=384

        │
        ├── [B,384,80,80]
        ├── [B,384,40,40]
        └── [B,384,20,20]

        │
        ▼

DFINETransformer

num_levels=3
num_layers=6
num_queries=300
num_points=[3,6,3]
reg_max=32
reg_scale=8

        │
        ▼

class predictions
+
D-FINE localization predictions

        │
        ▼

PostProcessor
```

这就是“会读配置”和“只会看参数”的区别。

---

# 34. 接下来读 `base/deim.yml`

现在我们还只有：

```text
D-FINE-X
```

但用户真正要学的是：

```text
DEIM-D-FINE-X
```

`deim_hgnetv2_x_coco.yml` include：

```yaml
'../base/deim.yml'
```

这个文件加入：

# DEIM training recipe

---

# 35. Dense O2O augmentation

`base/deim.yml` 中包含：

```yaml
train_dataloader:
    dataset:
        transforms:
            ops:
                - Mosaic
                - RandomPhotometricDistort
                - RandomZoomOut
                - RandomIoUCrop
                - SanitizeBoundingBoxes
                - RandomHorizontalFlip
                - Resize
                - ...
```

其中 Mosaic：

```yaml
output_size: 320
```

最后：

```yaml
Resize:
    size: [640,640]
```

所以可以形成：

```text
multiple images
↓
Mosaic
↓
320-scale mosaic construction
↓
other augmentations
↓
Resize
↓
640×640 network input
```

这里最重要的是：

> 在 DEIM 中，这些增强不只是为了普通意义上的“数据多样性”。

它们还与：

```text
Dense O2O
```

的监督密度有关。

第 15～17 课会专门推导。

---

# 36. 为什么 Mosaic 会影响 O2O supervision density？

假设普通图片：

```text
5 GT
```

300 queries 下：

```text
最多 5 O2O positive matches
```

Mosaic 把多张图组合后，可能让一张训练样本中出现：

```text
更多 objects
```

例如：

```text
20 GT
```

那么：

```text
最多 20 positive O2O matches
```

所以 augmentation 在 DEIM 中还承担：

```text
增加 target density
↓
增加 positive O2O supervision
```

的作用。

这就是 DEIM 训练思想最关键的直觉之一。

---

# 37. `policy.epoch`

DEIM-X 最终配置：

```yaml
policy:
    epoch: [4, 29, 50]
```

这意味着 augmentation policy 会随 epoch 改变。

不要把训练理解成：

```text
epoch 0
到
epoch 57

全部使用同一套数据增强
```

而是：

```text
training timeline
────────────────────────────►

0      4          29          50      58
│──────│──────────│───────────│───────│
       augmentation stages change
```

这就是：

```text
stage-aware training recipe
```

---

# 38. `mixup_epochs`

X 配置：

```yaml
mixup_epochs: [4, 29]
```

表示 MixUp 只在特定 epoch 区间启用。

所以：

```text
MixUp
```

不是从第一轮一直跑到最后一轮。

为什么？

因为强 augmentation：

```text
训练早中期
→ 提供丰富监督

训练后期
→ 可能妨碍模型在真实数据分布上收敛
```

因此后期常关闭强 augmentation。

---

# 39. `stop_epoch: 50`

```yaml
stop_epoch: 50
```

与训练后期的数据策略切换有关。

注意：

```text
stop_epoch
```

不是：

```text
训练在 epoch 50 结束
```

真正总训练 epoch 是：

```yaml
epoches: 58
```

所以：

```text
epoch 50
↓
某些数据策略停止

epoch 58
↓
整个训练结束
```

这是配置阅读时非常容易误解的地方。

---

# 40. DEIM 会覆盖 Backbone freeze 设置

D-FINE-X baseline：

```yaml
HGNetv2:
    freeze_stem_only: True
    freeze_at: 0
    freeze_norm: True
```

而 `base/deim.yml`：

```yaml
HGNetv2:
    freeze_at: -1
    freeze_norm: False
```

也就是说 DEIM recipe 明确调整了 backbone training behavior。

最终有效配置需要按照 merge 规则确认。

概念上 DEIM 希望：

```text
更多 backbone 参数参与训练
```

并：

```text
不冻结 normalization
```

这再次说明：

> **不能只看 `dfine_hgnetv2_x_coco.yml` 就认为那是最终 DEIM-X 配置。**

---

# 41. DEIM 还改变 decoder activation

`base/deim.yml`：

```yaml
DFINETransformer:
    activation: silu
    mlp_act: silu
```

所以 DEIM 并不只改：

```text
matcher / loss
```

它还对训练 recipe 和部分模块设置进行了调整。

因此“DEIM = 只有 MAL”这种理解是不完整的。

---

# 42. Criterion：模型到底优化什么？

基础：

```yaml
criterion: DEIMCriterion
```

D-FINE baseline 中：

```yaml
DEIMCriterion:
    weight_dict:
        loss_vfl: 1
        loss_bbox: 5
        loss_giou: 2
        loss_fgl: 0.15
        loss_ddf: 1.5

    losses:
        ['vfl', 'boxes', 'local']
```

DEIM base 覆盖为：

```yaml
DEIMCriterion:
    weight_dict:
        loss_mal: 1
        loss_bbox: 5
        loss_giou: 2
        loss_fgl: 0.15
        loss_ddf: 1.5

    losses:
        ['mal', 'boxes', 'local']

    gamma: 1.5
```

观察最关键变化：

```text
D-FINE baseline
loss_vfl

        ↓

DEIM
loss_mal
```

也就是：

```text
Varifocal-style classification supervision
        ↓
Matchability-Aware Loss
```

这就是 DEIM 核心训练改进真正进入代码配置的位置之一。

第 16 课会完整拆 MAL。

---

# 43. Criterion 与 Matcher 是嵌套依赖

基础配置：

```yaml
DEIMCriterion:

    matcher:
        type: HungarianMatcher

        weight_dict:
            cost_class: 2
            cost_bbox: 5
            cost_giou: 2

        alpha: 0.25
        gamma: 2.0
```

对象图：

```text
DEIMCriterion
      │
      └── HungarianMatcher
```

也就是说：

```python
criterion = DEIMCriterion(
    matcher=HungarianMatcher(...)
)
```

这正是第 6 课学的内容进入真实仓库的位置。

---

# 44. 再次区分 matcher cost 与 loss weight

Matcher：

```yaml
cost_class: 2
cost_bbox: 5
cost_giou: 2
```

用于：

```text
prediction ↔ GT assignment
```

Criterion：

```yaml
loss_mal: 1
loss_bbox: 5
loss_giou: 2
...
```

用于：

```text
backpropagation
```

所以：

```text
matcher weight
≠
loss weight
```

即使数字碰巧类似，也不能混为一谈。

---

# 45. 进入 Optimizer

X 配置：

```yaml
optimizer:

    type: AdamW

    params:

        -
            params: '^(?=.*backbone)(?!.*norm|bn).*$'
            lr: 0.000005

        -
            params: '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$'
            weight_decay: 0.

    lr: 0.0005

    betas: [0.9, 0.999]

    weight_decay: 0.000125
```

这一段初学者通常最容易直接跳过。

但它非常重要。

---

# 46. 默认 LR

```yaml
lr: 0.0005
```

也就是：

\[
5\times10^{-4}
\]

这是默认 optimizer learning rate。

没有被特殊 param group 捕获的 trainable parameters，会落入默认组。

---

# 47. Backbone LR

```yaml
params:
    '^(?=.*backbone)(?!.*norm|bn).*$'

lr:
    0.000005
```

也就是：

\[
5\times10^{-6}
\]

与默认：

\[
5\times10^{-4}
\]

相比：

\[
\frac}
}
=

0.01
\]

所以：

```text
backbone LR
=
detector default LR 的 1%
```

为什么？

因为 backbone 通常来自预训练：

```text
已经有较好的视觉特征
```

所以 fine-tuning 时不希望它被大 LR 快速破坏。

---

# 48. 逐段拆正则表达式

```regex
^(?=.*backbone)(?!.*norm|bn).*$
```

先看：

```regex
^
```

字符串开始。

```regex
(?=.*backbone)
```

要求参数名中：

```text
必须包含 backbone
```

```regex
(?!.*norm|bn)
```

要求：

```text
不能匹配指定 norm / bn 模式
```

最后：

```regex
.*$
```

匹配整个参数名。

所以它的目标大致是：

```text
backbone 中
非 normalization 参数
```

---

# 49. 为什么用参数名正则，而不是手工传参数？

如果手写：

```python
optimizer = AdamW([
    {
        "params": model.backbone.parameters(),
        "lr": ...
    },
    {
        "params": model.encoder.parameters(),
        "lr": ...
    }
])
```

对于复杂模型：

```text
norm
bias
backbone
encoder
decoder
```

需要很多特殊规则。

用 regex 可以按：

```python
model.named_parameters()
```

中的名字自动筛选。

例如：

```text
backbone.stages.2.blocks.0.conv.weight

encoder.fpn_blocks.0.norm.weight

decoder.layers.3.cross_attn.sampling_offsets.bias
```

配置系统就能按名字分组。

---

# 50. `YAMLConfig.get_optim_params()`

官方实现的核心逻辑：

```python
for pg in cfg['params']:

    pattern = pg['params']

    params = {
        k: v
        for k, v in model.named_parameters()
        if v.requires_grad
        and len(re.findall(pattern, k)) > 0
    }

    pg['params'] = params.values()

    param_groups.append(pg)
```

逐行解释。

---

## 50.1 遍历 param group

```python
for pg in cfg['params']:
```

每个 YAML：

```yaml
-
    params: regex
    lr: ...
```

就是一个 group。

---

## 50.2 取 regex

```python
pattern = pg['params']
```

例如：

```text
^(?=.*backbone)(?!.*norm|bn).*$
```

---

## 50.3 遍历全部 trainable parameters

```python
model.named_parameters()
```

返回：

```text
(name, parameter)
```

---

## 50.4 regex 筛选

```python
re.findall(pattern, k)
```

如果参数名匹配：

```text
加入这个 group
```

---

## 50.5 未匹配参数怎么办？

源码还会计算：

```python
unseen = set(names) - set(visited)
```

然后把未匹配参数放进额外默认 group。

所以不是：

```text
regex 没写到
→ 参数就完全不训练
```

而是：

```text
未匹配 trainable parameters
→ default optimizer settings
```

这是一个非常重要的源码细节。

---

# 51. Norm 为什么常设 `weight_decay: 0`？

第二组：

```yaml
params:
    '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$'

weight_decay: 0.
```

意思大致是：

```text
encoder / decoder 中的 normalization 参数
```

不使用 weight decay。

这是常见训练策略：

```text
Conv / Linear weights
→ weight decay

Norm parameters
→ often no weight decay
```

后面第 18 课会专门讲 optimizer recipe。

---

# 52. AdamW 的数学直觉

AdamW 可以粗略写成：

\[
\theta_
=======

\theta_t
--------

\eta
\frac
+\epsilon}
----------

\eta\lambda\theta_t
\]

其中：

```text
η
= learning rate

m
= first moment

v
= second moment

λ
= weight decay
```

X 配置：

```yaml
betas: [0.9, 0.999]
weight_decay: 0.000125
```

这些不是网络结构参数。

它们属于：

```text
optimization dynamics
```

---

# 53. 为什么 X 的 backbone LR 如此小？

X 使用：

```text
HGNetv2-B5
```

backbone 是大型预训练视觉特征提取器。

如果 detector head：

```text
随机初始化 / 需要大幅适应检测任务
```

而 backbone：

```text
已经学到较稳定特征
```

那么合理策略是：

```text
backbone
→ 小步更新

encoder / decoder
→ 大步学习
```

所以：

```text
differential learning rate
```

本质是：

```text
不同模块处于不同训练阶段
```

---

# 54. `epoches: 58`

DEIM-X：

```yaml
epoches: 58
```

注意仓库这里字段拼写就是：

```text
epoches
```

而不是标准英文：

```text
epochs
```

读源码时：

> **配置 key 必须尊重仓库实际拼写。**

不能因为觉得拼写奇怪就自己改。

---

# 55. 为什么只有 58 epochs？

原始 D-FINE-X baseline 的训练周期更长。

DEIM 的核心目标之一就是：

```text
fast convergence
```

所以 DEIM recipe 会通过：

```text
Dense O2O
MAL
strong augmentation
training schedule
```

提高训练效率。

因此：

```text
DEIM
```

不是单纯追求最终 AP。

它非常强调：

```text
更快收敛
```

---

# 56. `flat_epoch: 29`

```yaml
flat_epoch: 29
```

DEIM 使用自定义的：

```text
flat-cosine style LR schedule
```

可以先画成：

```text
LR

│       ─────────────
│      /             \
│     /               \
│____/                 \____
│
└──────────────────────────── epoch
   warmup      flat      cosine
```

具体 scheduler 数学形式到第 18 课再展开。

本课只需要知道：

```text
29
```

不是网络层数。

它控制：

```text
learning-rate timeline
```

---

# 57. `no_aug_epoch: 8`

```yaml
no_aug_epoch: 8
```

表示训练最后阶段会保留一段：

```text
reduced/no strong augmentation phase
```

总 epoch：

```text
58
```

最后：

```text
8 epochs
```

大致对应：

```text
50 → 58
```

这与：

```yaml
stop_epoch: 50
```

形成呼应。

所以训练 schedule 大致可以理解为：

```text
0 ─────────────── 50 ───── 58
      strong / staged        final refinement
      augmentation           phase
```

这不是巧合。

---

# 58. 数据集配置从哪里来？

`dfine_hgnetv2_x_coco.yml` include：

```yaml
../dataset/coco_detection.yml
```

这里定义：

```text
task: detection

num_classes: 80

train dataset:
CocoDetection

val dataset:
CocoDetection

evaluator:
CocoEvaluator
```

以及：

```text
COCO train image path
COCO annotation path
COCO val image path
COCO val annotation path
```

所以：

```text
模型配置
```

和：

```text
数据集配置
```

也是分离的。

---

# 59. 为什么 `num_classes` 不在 X 模型文件里？

因为：

```text
num_classes
```

属于：

```text
dataset/task-level shared configuration
```

而不是：

```text
X architecture-specific configuration
```

所以放在：

```text
coco_detection.yml
```

更合理。

如果换自己的数据集：

```text
80 classes
→ 5 classes
```

主要应该修改 dataset/task config，而不是复制整个 X architecture。

这就是模块化配置的价值。

---

# 60. `__share__` 的思想

框架中的一些组件需要共享同一个配置变量。

例如：

```text
num_classes
```

可能同时被：

```text
decoder
criterion
postprocessor
```

使用。

如果每个模块都独立写：

```yaml
num_classes: 80
```

很容易出现：

```text
decoder = 80
criterion = 80
postprocessor = 5
```

这种灾难。

所以配置系统会支持：

```text
shared values
```

即：

```text
一个 global config value
↓
多个模块共同使用
```

后面读具体类的：

```python
__share__
```

时，你就知道它为什么存在。

---

# 61. `__inject__` 与 `__share__` 的区别

非常重要。

## `__inject__`

表示：

```text
这个参数本身是另一个对象
```

例如：

```text
DEIM.backbone
→ HGNetv2 object
```

---

## `__share__`

表示：

```text
这个参数值从全局配置共享
```

例如：

```text
num_classes = 80
```

多个模块都使用同一个值。

所以：

```text
inject
→ object dependency

share
→ configuration value dependency
```

---

# 62. PostProcessor

基础配置：

```yaml
postprocessor: PostProcessor
```

以及：

```yaml
PostProcessor:
    num_top_queries: 300
```

它负责把模型输出转换为最终可评估/推理的检测结果。

概念上：

```text
normalized model outputs
↓
scores / labels / boxes
↓
select predictions
↓
scale boxes back to original image size
```

所以：

```text
PostProcessor
```

不参与 backbone feature extraction。

也不是 decoder layer。

它属于：

```text
inference / evaluation output conversion
```

---

# 63. Solver 是怎么来的？

`train.py`：

```python
solver = TASKS[
    cfg.yaml_cfg['task']
](cfg)
```

而：

```yaml
task: detection
```

所以：

```text
TASKS["detection"]
```

会选择 detection solver。

概念上：

```text
task = detection
↓
DetSolver
↓
fit()
```

于是：

```python
solver.fit()
```

才真正进入训练生命周期。

---

# 64. `solver.fit()` 后发生什么？

高层逻辑：

```text
solver.fit()
│
├── setup model
├── setup criterion
├── setup optimizer
├── setup dataloader
├── setup EMA
├── setup AMP
├── setup scheduler
│
└── training loop
     │
     ├── load batch
     ├── model forward
     ├── criterion
     ├── backward
     ├── optimizer.step
     ├── EMA update
     ├── scheduler update
     ├── validation
     └── checkpoint
```

第 23 课会真正从 `fit()` 逐行走到：

```python
optimizer.step()
```

本课只建立地图。

---

# 65. 一份 YAML 最终控制了多少东西？

现在列出来。

```text
deim_hgnetv2_x_coco.yml
```

最终间接决定：

```text
① task

② dataset
③ augmentation
④ dataloader
⑤ collate function

⑥ model container
⑦ backbone
⑧ encoder
⑨ decoder

⑩ query number
⑪ decoder layers
⑫ deformable attention points
⑬ denoising
⑭ D-FINE localization settings

⑮ matcher
⑯ criterion
⑰ loss types
⑱ loss weights

⑲ optimizer
⑳ parameter groups
㉑ module-specific LR
㉒ weight decay

㉓ LR scheduler
㉔ warmup
㉕ training epochs
㉖ no-augmentation stage

㉗ EMA
㉘ AMP

㉙ evaluator
㉚ postprocessor
㉛ output directory
```

所以：

> **学会读配置，其实就是学会读整个训练系统的 wiring diagram。**

---

# 66. 完整配置合并后的“心理模型”

以后不要脑中保存 6 个 YAML 文件。

应该把它们 merge 成一棵最终树：

```text
FINAL CONFIG

task: detection
num_classes: 80

model: DEIM
│
└── DEIM
    ├── backbone: HGNetv2
    ├── encoder: HybridEncoder
    └── decoder: DFINETransformer

HGNetv2
├── name: B5
├── return_idx: [1,2,3]
├── freeze_at: ...
└── freeze_norm: ...

HybridEncoder
├── in_channels: [512,1024,2048]
├── feat_strides: [8,16,32]
├── hidden_dim: 384
├── nhead: 8
└── dim_feedforward: 2048

DFINETransformer
├── feat_channels: [384,384,384]
├── feat_strides: [8,16,32]
├── num_levels: 3
├── num_layers: 6
├── num_queries: 300
├── num_denoising: 100
├── reg_max: 32
├── reg_scale: 8
└── num_points: [3,6,3]

criterion: DEIMCriterion
│
├── MAL
├── bbox
├── GIoU
├── FGL
├── DDF
└── matcher: HungarianMatcher

optimizer: AdamW
├── backbone LR
├── detector LR
└── special no-decay groups

train_dataloader
├── COCO
├── Mosaic
├── MixUp
└── staged augmentation

scheduler
├── warmup
├── flat phase
└── cosine phase

postprocessor
└── top queries
```

这才是最终模型/训练系统。

---

# 67. 一个很重要的源码阅读原则：不要把“配置层”和“实现层”混在一起

例如 YAML：

```yaml
HybridEncoder:
    hidden_dim: 384
```

它告诉我们：

```text
encoder hidden dimension = 384
```

但它没有告诉我们：

```text
384 在 forward 的哪一行出现？
input projection 用什么卷积？
FPN 如何融合？
PAN 如何融合？
Transformer encoder 如何处理 P5？
```

这些必须去：

```text
hybrid_encoder.py
```

看实现。

同理：

```yaml
reg_scale: 8
```

只告诉我们：

```text
reg_scale = 8
```

但：

```text
它参与哪条公式？
在哪里乘？
在哪里除？
怎样影响 bbox？
```

必须进入：

```text
DFINETransformer implementation
```

所以：

```text
YAML
=
告诉你“有什么”和“取什么值”

Python source
=
告诉你“具体怎么做”
```

---

# 68. 反过来也一样：只看 Python 源码不看 YAML 也会读错

例如源码：

```python
class DFINETransformer(...):

    def __init__(
        self,
        hidden_dim=256,
        num_layers=6,
        num_queries=300,
        ...
    ):
```

你可能以为 X：

```text
hidden_dim = 256
```

但 YAML 可能通过：

```text
feat_channels
```

以及其他 X-specific override 改变实际接口设置。

所以：

> **真实运行行为 = Python defaults + merged YAML + CLI override**

不能只看其中一层。

---

# 69. 小型实验 1：自己实现 YAML include merge

下面写一个极简思想实验。

```python
base = {
    "model": "DEIM",

    "HybridEncoder": {
        "hidden_dim": 256,
        "dim_feedforward": 1024,
    },

    "DFINETransformer": {
        "num_layers": 6,
        "reg_scale": 4,
    }
}

x_cfg = {
    "HybridEncoder": {
        "hidden_dim": 384,
        "dim_feedforward": 2048,
    },

    "DFINETransformer": {
        "reg_scale": 8,
    }
}
```

自己实现：

```python
def deep_merge(base, override):

    out = dict(base)

    for k, v in override.items():

        if (
            k in out
            and isinstance(out[k], dict)
            and isinstance(v, dict)
        ):
            out[k] = deep_merge(
                out[k],
                v
            )

        else:
            out[k] = v

    return out
```

然后：

```python
final_cfg = deep_merge(
    base,
    x_cfg
)

print(final_cfg)
```

你应该得到：

```text
HybridEncoder.hidden_dim
→ 384

HybridEncoder.dim_feedforward
→ 2048

DFINETransformer.num_layers
→ 6

DFINETransformer.reg_scale
→ 8
```

这个实验就是：

```text
配置继承
```

的最小模型。

---

# 70. 小型实验 2：自己写 Registry

```python
REGISTRY = {}

def register(cls):
    REGISTRY[cls.__name__] = cls
    return cls


@register
class HGNetv2:

    def __init__(self, name):
        self.name = name


@register
class HybridEncoder:

    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim
```

测试：

```python
print(REGISTRY)
```

然后：

```python
cls = REGISTRY["HGNetv2"]

backbone = cls(
    name="B5"
)

print(backbone.name)
```

输出：

```text
B5
```

这就是：

```text
YAML string
↓
Python class
```

的核心机制。

---

# 71. 小型实验 3：自己写 `create()`

```python
def create(name, cfg):

    cls = REGISTRY[name]

    kwargs = cfg[name]

    return cls(**kwargs)
```

配置：

```python
cfg = {
    "HGNetv2": {
        "name": "B5"
    },

    "HybridEncoder": {
        "hidden_dim": 384
    }
}
```

运行：

```python
backbone = create(
    "HGNetv2",
    cfg
)

encoder = create(
    "HybridEncoder",
    cfg
)
```

这时你已经实现了一个极简 DEIM workspace factory。

---

# 72. 小型实验 4：模拟 Dependency Injection

```python
@register
class DEIM:

    def __init__(
        self,
        backbone,
        encoder
    ):
        self.backbone = backbone
        self.encoder = encoder
```

配置：

```python
cfg = {
    "DEIM": {
        "backbone": "HGNetv2",
        "encoder": "HybridEncoder"
    },

    "HGNetv2": {
        "name": "B5"
    },

    "HybridEncoder": {
        "hidden_dim": 384
    }
}
```

可以写：

```python
def create_deim(cfg):

    backbone = create(
        cfg["DEIM"]["backbone"],
        cfg
    )

    encoder = create(
        cfg["DEIM"]["encoder"],
        cfg
    )

    return DEIM(
        backbone=backbone,
        encoder=encoder
    )
```

于是：

```text
字符串配置
↓
递归实例化
↓
完整 object graph
```

这就是框架思想。

---

# 73. 小型实验 5：真正理解 optimizer regex

```python
import re

names = [
    "backbone.stage2.conv.weight",
    "backbone.stage2.bn.weight",
    "encoder.fpn.conv.weight",
    "encoder.fpn.norm.weight",
    "decoder.layers.0.linear.weight",
    "decoder.layers.0.norm.weight",
]
```

Backbone pattern：

```python
pattern = r'^(?=.*backbone)(?!.*norm|bn).*$'
```

测试：

```python
for name in names:

    if re.findall(pattern, name):
        print(name)
```

观察哪些参数被选中。

然后测试：

```python
pattern = \
r'^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$'
```

这样你会真正理解 optimizer param groups，而不是把 regex 当魔法字符串。

---

# 74. 小型实验 6：算 X 的三个 feature level

```python
H = W = 640

strides = [8, 16, 32]

for s in strides:

    print(
        s,
        H // s,
        W // s
    )
```

得到：

```text
8  → 80 × 80
16 → 40 × 40
32 → 20 × 20
```

总 spatial positions：

```python
total = sum(
    (H // s) * (W // s)
    for s in strides
)

print(total)
```

得到：

```text
8400
```

这就是后面 decoder flatten feature 时非常关键的数字。

---

# 75. 小型实验 7：算 attention head dimension

X HybridEncoder：

```text
hidden_dim = 384
nhead = 8
```

所以：

```python
hidden_dim = 384
nhead = 8

head_dim = hidden_dim // nhead

print(head_dim)
```

得到：

```text
48
```

于是：

```text
[B,N,384]
↓
[B,8,N,48]
```

第 4 课的知识现在正式落到 X 配置上。

---

# 76. 小型实验 8：算 Backbone LR 比例

```python
backbone_lr = 0.000005
default_lr = 0.0005

print(
    backbone_lr / default_lr
)
```

输出：

```text
0.01
```

所以：

```text
backbone LR
=
default detector LR × 0.01
```

也就是：

```text
1%
```

这个数字非常值得记住。

---

# 77. 第一次完整模拟 `cfg.model`

现在把整个过程串起来：

```text
train.py

cfg =
YAMLConfig(
    "deim_hgnetv2_x_coco.yml"
)

        │
        ▼

load_config()

        │
        ├── coco_detection.yml
        ├── runtime.yml
        ├── dataloader.yml
        ├── optimizer.yml
        ├── dfine_hgnetv2.yml
        ├── dfine_hgnetv2_x_coco.yml
        ├── deim.yml
        └── deim_hgnetv2_x_coco.yml

        │
        ▼

deep merged config

        │
        ▼

cfg.model

        │
        ▼

create("DEIM")

        │
        ├── create("HGNetv2")
        │
        ├── create("HybridEncoder")
        │
        └── create("DFINETransformer")
        │
        ▼

DEIM(
    backbone=...,
    encoder=...,
    decoder=...
)

        │
        ▼

PyTorch nn.Module
```

这就是本课最核心的一张图。

---

# 78. 再模拟 `cfg.criterion`

```text
cfg.criterion

        │
        ▼

create("DEIMCriterion")

        │
        ▼

DEIMCriterion config

        │
        └── matcher
             │
             ▼
        create("HungarianMatcher")

        │
        ▼

DEIMCriterion(
    matcher=HungarianMatcher(...),
    weight_dict=...,
    losses=...
)
```

这就是：

```text
nested dependency injection
```

---

# 79. 再模拟 `cfg.optimizer`

```text
cfg.optimizer

        │
        ▼

get_optim_params(
    optimizer_cfg,
    cfg.model
)

        │
        ▼

model.named_parameters()

        │
        ├── regex group 1
        ├── regex group 2
        └── unseen/default group

        │
        ▼

create(
    "optimizer",
    params=param_groups
)

        │
        ▼

AdamW(...)
```

所以 optimizer 依赖：

```text
model 必须先存在
```

这也是为什么 property/lazy build 的依赖顺序很重要。

---

# 80. 一个非常值得掌握的调试技巧：打印最终 merged config

以后如果你怀疑：

```text
到底哪个 YAML 覆盖了哪个？
```

不要猜。

直接在：

```python
cfg = YAMLConfig(...)
```

后打印：

```python
print(cfg.yaml_cfg)
```

或者更漂亮地：

```python
import pprint

pprint.pp(
    cfg.yaml_cfg
)
```

你真正应该调试的是：

```text
merged config
```

而不是单个 YAML 文件。

---

# 81. 第二个调试技巧：打印真正创建出来的 model

```python
model = cfg.model

print(model)
```

这可以验证：

```text
到底创建了 HGNetv2 还是别的 backbone？

encoder 是什么？

decoder 是什么？
```

YAML 只是声明。

真正创建出的：

```python
nn.Module
```

才是运行事实。

---

# 82. 第三个调试技巧：打印 parameter groups

```python
optimizer = cfg.optimizer

for i, group in enumerate(
    optimizer.param_groups
):

    print(
        i,
        group["lr"],
        group["weight_decay"],
        len(group["params"])
    )
```

这能验证：

```text
backbone 是否真的用了小 LR？
norm 是否真的 no weight decay？
```

比只看 YAML 靠谱得多。

---

# 83. 第四个调试技巧：打印 parameter name → group

可以自己构造：

```python
name_by_id = {
    id(p): name
    for name, p
    in model.named_parameters()
}
```

再：

```python
for i, group in enumerate(
    optimizer.param_groups
):

    print(
        f"\nGROUP {i}"
    )

    for p in group["params"]:

        print(
            name_by_id[id(p)]
        )
```

以后第 18 课我们会用这种方法真正分析 X optimizer。

---

# 84. 第五个调试技巧：不要看到 `None` 就以为配置坏了

YAML 中常见：

```yaml
something: ~
```

YAML 的：

```text
~
```

就是：

```python
None
```

例如：

```yaml
base_size_repeat: ~
```

表示：

```python
base_size_repeat = None
```

这不是字符串 `"~"`。

---

# 85. 第六个调试技巧：区分 YAML list 与 Tensor shape

例如：

```yaml
feat_channels:
    [384,384,384]
```

它只是：

```python
list[int]
```

不是 Tensor：

```text
[3,384]
```

同样：

```yaml
feat_strides:
    [8,16,32]
```

也不是：

```text
tensor shape
```

这是配置列表。

只有 forward 运行时才产生真正 Tensor。

---

# 86. 第七个调试技巧：不要把 `num_levels=3` 理解成“3 层 decoder”

这是三个不同概念：

```text
num_levels = 3
→ 三个 feature scales

num_layers = 6
→ 六层 decoder

num_queries = 300
→ 三百个 object queries
```

它们完全不同。

以后看到：

```text
3 / 6 / 300
```

必须立即知道各自属于哪个维度。

---

# 87. 第八个调试技巧：同名 `hidden_dim` 不一定代表同一 tensor

例如：

```yaml
HybridEncoder:
    hidden_dim: 384
```

和：

```yaml
DFINETransformer:
    hidden_dim: ...
```

即使都叫：

```text
hidden_dim
```

也属于不同 class 的 constructor argument。

不能看到名字相同就假定：

```text
它们必然是同一个变量
```

是否共享，要看：

```text
__share__
```

以及实际 class implementation。

---

# 88. 本课第一次完整还原 `deim_dfine_x`

现在我们把它写成一张最终结构图。

```text
DEIM-D-FINE-X
│
├── Dataset
│   └── COCO
│       └── 80 classes
│
├── Data Pipeline
│   ├── Mosaic
│   ├── Photometric Distort
│   ├── ZoomOut
│   ├── IoU Crop
│   ├── Horizontal Flip
│   ├── Resize 640×640
│   └── staged MixUp / augmentation
│
├── Model: DEIM
│   │
│   ├── Backbone: HGNetv2-B5
│   │   ├── C3: stride 8
│   │   ├── C4: stride 16
│   │   └── C5: stride 32
│   │
│   ├── Encoder: HybridEncoder
│   │   ├── input: [512,1024,2048]
│   │   ├── hidden_dim: 384
│   │   ├── nhead: 8
│   │   └── FFN: 2048
│   │
│   └── Decoder: DFINETransformer
│       ├── 3 feature levels
│       ├── 6 decoder layers
│       ├── 300 queries
│       ├── 100 denoising config
│       ├── reg_max: 32
│       ├── reg_scale: 8
│       └── deformable points: [3,6,3]
│
├── Criterion: DEIMCriterion
│   ├── MAL
│   ├── bbox
│   ├── GIoU
│   ├── FGL
│   └── DDF
│
├── Matcher
│   └── HungarianMatcher
│       ├── class cost
│       ├── bbox cost
│       └── GIoU cost
│
├── Optimizer: AdamW
│   ├── backbone LR: 5e-6
│   ├── default LR: 5e-4
│   └── weight decay: 1.25e-4
│
├── Scheduler
│   ├── warmup
│   ├── flat-cosine
│   └── no-aug final stage
│
└── PostProcessor
    └── top queries
```

如果你能从 YAML 自己画出这张图，本课就已经成功了一大半。

---

# 89. 用 640×640 再做一次端到端 shape 地图

输入：

```text
Images
[B,3,640,640]
```

Backbone：

```text
C3
[B,512,80,80]

C4
[B,1024,40,40]

C5
[B,2048,20,20]
```

HybridEncoder：

```text
P3
[B,384,80,80]

P4
[B,384,40,40]

P5
[B,384,20,20]
```

spatial positions：

```text
80×80 = 6400
40×40 = 1600
20×20 = 400
```

总：

```text
8400
```

decoder：

```text
300 matching queries
```

典型最终检测维度：

```text
300 class predictions

300 box predictions
```

训练：

```text
prediction set
↓
HungarianMatcher
↓
GT
↓
DEIMCriterion
↓
loss
↓
backward
```

---

# 90. 这里哪些 shape 是“已由配置直接确定”的？

确定：

```text
input spatial size
640×640

strides
8 / 16 / 32

feature spatial sizes
80 / 40 / 20

backbone output channels
512 / 1024 / 2048

encoder target channels
384 / 384 / 384

num levels
3

num queries
300

decoder layers
6

sampling points
3 / 6 / 3
```

---

# 91. 哪些东西不能只靠 YAML 推断？

例如：

```text
HybridEncoder 内部每一次 concat 后 channel

DFINETransformer 内部每个 tensor 的 exact shape

D-FINE distribution logits 的 exact layout

denoising query 的实际数量

reference point 的 exact tensor shape

decoder intermediate output layout
```

这些必须：

```text
打开源码
+
实际 forward trace
```

所以我们后面第 9～14 课才会继续拆。

---

# 92. 为什么这一课必须放在第 9 课 HGNetv2 之前？

因为如果直接打开：

```text
hgnetv2.py
```

你会看到很多：

```text
Stem
HGBlock
EffectiveSE
stage_config
return_idx
freeze_at
freeze_norm
```

如果不知道：

```text
最终 X 配置给它传了什么
```

就很容易迷路。

现在我们已经知道：

```text
name = B5
return_idx = [1,2,3]
...
```

所以下一课读 HGNetv2 时，我们不是研究：

```text
所有可能的 HGNetv2
```

而是研究：

# X 配置真正走到的那一条执行路径

这会大幅降低源码阅读难度。

---

# 93. 同样，这一课也为后面的 `deimv2_dinov3_x` 做准备

以后看到：

```yaml
DEIM:
    backbone: DINOv3STAs
```

你会立刻知道：

```text
不是 train.py 改了
```

而是：

```text
配置依赖注入
把 backbone 从 HGNetv2 换成 DINOv3STAs
```

所以：

```text
DEIM container
HybridEncoder
decoder
criterion
solver
```

整个框架可以大量复用。

这就是配置驱动 architecture 的强大之处。

---

# 94. 本课必须掌握的 15 个结论

## 1

`deim_hgnetv2_x_coco.yml` 不是完整配置。

它通过：

```yaml
__include__
```

继承多份 YAML。

---

## 2

最终配置是：

```text
recursive merge
```

后的结果。

---

## 3

DEIM-D-FINE-X 可以理解成：

```text
D-FINE-X detector
+
DEIM training recipe
```

---

## 4

顶层：

```yaml
model: DEIM
```

会通过配置系统创建：

```python
DEIM(...)
```

---

## 5

DEIM 的三个主要依赖：

```text
HGNetv2
HybridEncoder
DFINETransformer
```

---

## 6

Registry 负责：

```text
name → Python class
```

---

## 7

`create()` 负责：

```text
config → Python object
```

---

## 8

`__inject__` 用于：

```text
object dependency
```

---

## 9

`__share__` 用于：

```text
shared config value
```

---

## 10

X backbone：

```text
HGNetv2-B5
```

返回 stride：

```text
8 / 16 / 32
```

特征尺度：

```text
80 / 40 / 20
```

---

## 11

X HybridEncoder：

```text
hidden_dim = 384
```

---

## 12

DFINETransformer：

```text
3 feature levels
6 decoder layers
300 queries
reg_max=32
reg_scale=8
num_points=[3,6,3]
```

---

## 13

DEIMCriterion 使用：

```text
MAL
```

替代 baseline 中相应的分类监督配置，并继续结合 localization losses。

---

## 14

Optimizer 使用：

```text
regex parameter groups
```

让 backbone 与 detector 使用不同 LR。

---

## 15

真正运行行为不是单个 YAML 决定的，而是：

```text
Python defaults
+
included YAMLs
+
override YAML
+
CLI overrides
```

共同决定。

---

# 95. 本课练习

## 练习 1

不看前文，画出：

```text
deim_hgnetv2_x_coco.yml
```

的 include tree。

---

## 练习 2

解释：

```yaml
model: DEIM

DEIM:
    backbone: HGNetv2
```

为什么不是同一个层级的含义。

---

## 练习 3

解释：

```text
register
create
inject
share
```

四个概念分别解决什么问题。

---

## 练习 4

640×640 输入：

```text
stride = [8,16,32]
```

手算三个 feature spatial sizes。

---

## 练习 5

计算：

\[
80^2+40^2+20^2
\]

并解释为什么：

```text
8400
```

会在后面的 decoder 源码中频繁出现。

---

## 练习 6

X HybridEncoder：

```text
hidden_dim = 384
nhead = 8
```

计算：

```text
head_dim
```

---

## 练习 7

解释：

```text
num_levels = 3

num_layers = 6

num_queries = 300
```

分别是什么。

---

## 练习 8

解释：

```yaml
num_points: [3,6,3]
```

与第 7 课 deformable attention 的关系。

---

## 练习 9

计算：

```text
backbone lr = 5e-6

default lr = 5e-4
```

backbone LR 是 default LR 的多少比例。

---

## 练习 10

为什么：

```text
matcher cost weight
```

和：

```text
criterion loss weight
```

不能混为一谈？

---

# 96. 练习答案

## 练习 4

```text
640 / 8  = 80
640 / 16 = 40
640 / 32 = 20
```

---

## 练习 5

\[
6400+1600+400
=============

8400
\]

表示三个 multi-scale feature map flatten 后的 spatial positions 总数。

---

## 练习 6

\[
384/8=48
\]

所以：

```text
head_dim = 48
```

---

## 练习 7

```text
num_levels
→ feature scale 数量

num_layers
→ decoder layer 数量

num_queries
→ object query 数量
```

---

## 练习 8

三个尺度分别使用：

```text
3
6
3
```

个采样点。

每个 head/query 合计：

\[
3+6+3=12
\]

个 multi-scale deformable sampling points。

---

## 练习 9

\[
5e-6 / 5e-4
===========

0.01
\]

所以：

```text
1%
```

---

# 97. 建议自己真正运行的检查脚本

以后环境准备好以后，建议创建：

```text
inspect_config.py
```

内容类似：

```python
from engine.core import YAMLConfig
import pprint

cfg = YAMLConfig(
    "configs/deim_dfine/"
    "deim_hgnetv2_x_coco.yml"
)

print("=" * 80)
print("MERGED CONFIG")
print("=" * 80)

pprint.pp(
    cfg.yaml_cfg
)


print("=" * 80)
print("MODEL")
print("=" * 80)

model = cfg.model

print(model)


print("=" * 80)
print("TRAINABLE PARAMETERS")
print("=" * 80)

for name, p in model.named_parameters():

    if p.requires_grad:

        print(
            name,
            tuple(p.shape)
        )
```

这会把：

```text
静态 YAML 阅读
```

升级为：

```text
runtime verification
```

---

# 98. 再进一步：检查 optimizer group

```python
optimizer = cfg.optimizer

name_map = {
    id(p): name
    for name, p
    in model.named_parameters()
}

for i, group in enumerate(
    optimizer.param_groups
):

    print(
        f"\n===== GROUP {i} ====="
    )

    print(
        "lr =",
        group["lr"]
    )

    print(
        "weight_decay =",
        group["weight_decay"]
    )

    for p in group["params"]:

        print(
            name_map.get(
                id(p),
                "UNKNOWN"
            )
        )
```

这个实验以后会非常有价值。

因为它能把：

```yaml
regex
```

真正转换成：

```text
具体参数名
```

---

# 99. 本课最终总图

```text
COMMAND

train.py
-c deim_hgnetv2_x_coco.yml
--use-amp

        │
        ▼

YAMLConfig

        │
        ▼

recursive YAML include + merge

        │
        ├── COCO dataset config
        ├── runtime config
        ├── dataloader config
        ├── optimizer base
        ├── D-FINE HGNetv2 base
        ├── D-FINE-X override
        ├── DEIM training recipe
        └── DEIM-X override

        │
        ▼

FINAL CONFIG TREE

        │
        ├───────────────┐
        ▼               ▼

      MODEL          TRAINING
        │               │
        │               ├── DEIMCriterion
        │               │    └── HungarianMatcher
        │               │
        │               ├── AdamW
        │               ├── param groups
        │               ├── LR schedule
        │               ├── Dense O2O augmentation
        │               ├── EMA
        │               └── AMP
        │
        ▼

DEIM
│
├── HGNetv2-B5
│
├── HybridEncoder
│
└── DFINETransformer

        │
        ▼

[B,3,640,640]

        │
        ▼

C3 / C4 / C5

        │
        ▼

P3 / P4 / P5

        │
        ▼

8400 spatial positions

        │
        ▼

300 object queries

        │
        ▼

classification + localization

        │
        ▼

Hungarian O2O

        │
        ▼

DEIM losses

        │
        ▼

backward()
```

---

# 100. 下一课之前你应该做到什么？

进入第 9 课《`deim_dfine_x` Backbone——HGNetv2 X》之前，建议你已经能独立回答：

```text
1. 为什么 X 配置只有几十行，却能创建完整模型？

2. __include__ 在干什么？

3. YAMLConfig 在干什么？

4. Registry 为什么存在？

5. create() 为什么存在？

6. __inject__ 是什么？

7. model: DEIM 最终创建了什么？

8. HGNetv2-B5 为什么输出三个尺度？

9. 640×640 为什么得到 80/40/20？

10. 为什么三个尺度 flatten 后是 8400？

11. HybridEncoder 的 384 是什么？

12. num_levels / num_layers / num_queries 有什么区别？

13. [3,6,3] 是什么？

14. 为什么 backbone LR 只有 detector LR 的 1%？

15. D-FINE-X 与 DEIM training recipe 是怎样组合起来的？
```

如果这些已经能回答，那么下一课打开 `hgnetv2.py` 时，你不会是在一个巨大 Python 文件里盲目寻找结构。

你会明确知道我们要追的是：

```text
HGNetv2(
    name="B5",
    return_idx=[1,2,3],
    ...
)
```

对应的那一条真实执行路径。

---

# 101. 下一课预告

## 第 9 课：`deim_dfine_x` Backbone——HGNetv2 X

下一课会真正进入：

```text
engine/backbone/hgnetv2.py
```

我们会从：

```text
HGNetv2-B5
```

开始逐层拆：

```text
Stem
↓
Stage 1
↓
Stage 2
↓
Stage 3
↓
Stage 4
```

并对 640×640 输入逐层计算：

```text
H × W
channels
stride
参数量
receptive field
```

重点研究：

```text
ConvBNAct
LightConvBNAct
HGBlock
EffectiveSE
aggregation
downsampling
return_idx
freeze_at
freeze_norm
```

最后用 forward hook 得到真实：

```text
[B,512,80,80]
[B,1024,40,40]
[B,2048,20,20]
```

并解释：

> **为什么 HGNetv2-B5 能作为 `deim_dfine_x` 的视觉 backbone，以及这些 feature 为什么正好能接进 HybridEncoder。**

---

## 本课使用的官方源码依据

本讲义按官方仓库当前配置结构核对，重点依据：

```text
Intellindust-AI-Lab/DEIM

train.py

engine/core/yaml_config.py

configs/deim_dfine/
    dfine_hgnetv2_x_coco.yml
    deim_hgnetv2_x_coco.yml

configs/base/
    dfine_hgnetv2.yml
    deim.yml

configs/dataset/
    coco_detection.yml
```

同时参考当前 DEIMv2 仓库的同系配置体系，以确保后续能够自然过渡到：

```text
deimv2_dinov3_x_coco.yml
```

> **阅读原则**：配置项会随仓库版本调整。后续涉及某个类内部 tensor 的精确 shape、默认值或公式时，以该版本实际 Python constructor / forward 为最终依据，而不是仅凭 YAML 名称推断。
