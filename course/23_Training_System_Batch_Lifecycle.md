# DEIMv2 源码课程 · 第 23 课

# 训练系统完整解剖------一条 batch 到底经历了什么

> **本课定位**：第 22 课已经把 `deimv2_dinov3_x`
> 的模型前向主链完整贯通。本课把视角再向外扩一层：从
> `python train.py ...` 开始，一直追到
> `DataLoader → augmentation → collate → model → matcher → criterion → backward → optimizer → EMA → scheduler → checkpoint → COCO evaluation`。
>
> 本课以 DEIMv2 官方仓库当前 `main` 分支为源码基准，重点跟踪
> `deimv2_dinov3_x_coco.yml`。仓库会更新，所以文中凡是与
> epoch、默认参数、具体 key 有关的细节，都应以你本地 commit
> 的**最终解析配置与运行时打印**为最高依据。

------------------------------------------------------------------------

## 0. 本课最终要回答的问题

当你执行：

``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --master_port=7777 --nproc_per_node=4 \
train.py \
-c configs/deimv2/deimv2_dinov3_x_coco.yml \
--use-amp \
--seed=0
```

到底发生了什么？

最终主链可以先压缩成：

``` text
Shell / torchrun
        ↓
train.py
        ↓
setup_distributed()
        ↓
YAMLConfig
        ↓
DetSolver.fit()
        ↓
BaseSolver.train() / _setup()
        ↓
model + criterion + optimizer + EMA + scaler
        ↓
COCO Dataset
        ↓
per-image transforms
        ↓
BatchImageCollateFunction
        ↓
samples + targets
        ↓
GPU
        ↓
model(samples, targets)
        ↓
DINOv3STAs
        ↓
HybridEncoder
        ↓
DEIMTransformer
        ↓
outputs
        ↓
DEIMCriterion
        ↓
HungarianMatcher
        ↓
MAL + bbox + GIoU + FGL + DDF + auxiliary/DN losses
        ↓
sum(loss_dict.values())
        ↓
backward()
        ↓
gradient clipping
        ↓
AdamW.step()
        ↓
EMA.update()
        ↓
LR scheduler
        ↓
下一 batch
        ↓
epoch 结束
        ↓
checkpoint
        ↓
EMA model evaluation
        ↓
PostProcessor
        ↓
COCO evaluator
        ↓
AP / best checkpoint
```

本课的目标是：以后训练报错时，你能准确判断错误属于哪一层，而不是把所有问题都归结为"模型有问题"。

------------------------------------------------------------------------

# 第一部分：从命令行进入 `train.py`

## 1. `torchrun` 不是"启动一个 Python"

4 卡训练：

``` text
torchrun --nproc_per_node=4
```

本质上会启动 4 个 Python worker process。

概念上：

``` text
rank 0 → GPU 0 → train.py
rank 1 → GPU 1 → train.py
rank 2 → GPU 2 → train.py
rank 3 → GPU 3 → train.py
```

每个进程都会运行：

``` python
main(args)
```

因此你必须建立一个很重要的认知：

> DDP 不是"一个 Python 程序同时操作四张
> GPU"，而是通常由多个进程各自负责一张 GPU，再通过 distributed
> collectives 同步梯度与统计量。

------------------------------------------------------------------------

## 2. `train.py` 做的事情其实不多

官方入口的核心逻辑可以概括为：

``` python
dist_utils.setup_distributed(...)

cfg = YAMLConfig(
    args.config,
    **update_dict
)

solver = TASKS[
    cfg.yaml_cfg['task']
](cfg)

if args.test_only:
    solver.val()
else:
    solver.fit()

dist_utils.cleanup()
```

所以 `train.py` 自己并不负责：

``` text
forward
loss
backward
optimizer
COCO evaluation
```

它更像一个总入口和调度器。

------------------------------------------------------------------------

## 3. CLI 参数怎样覆盖 YAML？

入口会先解析：

``` python
update_dict = yaml_utils.parse_cli(
    args.update
)
```

再把命令行参数合并进去：

``` python
update_dict.update(...)
```

最后：

``` python
cfg = YAMLConfig(
    args.config,
    **update_dict
)
```

因此：

``` text
YAML
+
CLI override
=
最终运行配置
```

这也是为什么调试时不能只盯着某一个 `.yml` 文件。

------------------------------------------------------------------------

# 第二部分：YAML 是怎样变成 Python 对象的？

## 4. X 配置不是孤立文件

`deimv2_dinov3_x_coco.yml` 会 include：

``` yaml
__include__:
  - ../dataset/coco_detection.yml
  - ../runtime.yml
  - ../base/dataloader.yml
  - ../base/optimizer.yml
  - ../base/deimv2.yml
```

所以最终配置来自多个层次：

``` text
dataset
runtime
dataloader
optimizer
base model
X-specific overrides
CLI overrides
```

------------------------------------------------------------------------

## 5. 为什么最终配置必须打印？

例如：

``` text
runtime.yml:
use_amp: False
use_ema: False
```

但：

``` text
base/optimizer.yml:
use_amp: True
use_ema: True
```

再加上：

``` bash
--use-amp
```

最终到底是什么，应该看 merge 后配置。

源码中：

``` python
self.yaml_cfg = copy.deepcopy(cfg)
```

所以建议训练前保存：

``` python
print(cfg)
print(cfg.yaml_cfg)
```

或者把最终配置写入日志。

------------------------------------------------------------------------

## 6. `YAMLConfig` 是"延迟实例化"

一个非常重要的设计：

``` python
@property
def model(self):
    if self._model is None:
        self._model = create(...)
    return super().model
```

也就是说：

``` text
cfg 创建
≠
所有模型对象立即创建
```

很多对象只有第一次访问：

``` text
cfg.model
cfg.optimizer
cfg.train_dataloader
```

时才真正实例化。

这叫：

``` text
lazy construction / lazy instantiation
```

------------------------------------------------------------------------

## 7. 主要对象由谁创建？

`YAMLConfig` 负责构建：

``` text
model
postprocessor
criterion
optimizer
lr_scheduler
warmup scheduler
train_dataloader
val_dataloader
EMA
GradScaler
evaluator
```

所以你可以把：

``` text
YAMLConfig
```

理解成整个训练系统的：

``` text
object factory + dependency resolver
```

------------------------------------------------------------------------

# 第三部分：Solver 是训练系统的总导演

## 8. `task: detection`

COCO 配置：

``` yaml
task: detection
```

入口：

``` python
solver = TASKS[
    cfg.yaml_cfg['task']
](cfg)
```

最终进入 detection solver：

``` text
DetSolver
```

------------------------------------------------------------------------

## 9. `solver.fit()` 第一件事

`DetSolver.fit()`：

``` python
self.train()
```

这里不要把：

``` text
self.train()
```

误解成：

``` text
训练一个 epoch
```

它调用的是 `BaseSolver.train()`，主要做**训练环境初始化**。

------------------------------------------------------------------------

## 10. `BaseSolver._setup()`

核心顺序：

``` text
选择 device
↓
cfg.model
↓
可选 tuning checkpoint
↓
model.to(device)
↓
DDP / SyncBN wrapping
↓
criterion.to(device)
↓
postprocessor.to(device)
↓
EMA
↓
GradScaler
↓
output_dir
↓
TensorBoard writer
```

源码主线：

``` python
self.model = cfg.model

self.model = dist_utils.warp_model(
    self.model.to(device),
    sync_bn=cfg.sync_bn,
    find_unused_parameters=
        cfg.find_unused_parameters
)

self.criterion = self.to(
    cfg.criterion,
    device
)

self.postprocessor = self.to(
    cfg.postprocessor,
    device
)

self.ema = self.to(
    cfg.ema,
    device
)

self.scaler = cfg.scaler
```

------------------------------------------------------------------------

# 第四部分：DDP 到底包住了什么？

## 11. 模型先 `.to(device)` 再 wrap

逻辑：

``` text
DEIM
↓
cuda:local_rank
↓
DistributedDataParallel(...)
```

训练时每个 rank 都有自己的模型 replica。

------------------------------------------------------------------------

## 12. DDP 的核心不是"把 batch 自动复制"

假设：

``` text
total_batch_size = 32
world_size = 4
```

`YAMLConfig.get_rank_batch_size()`：

$$
B_{rank} = \frac{B_{total}}{world\_size}
$$

所以：

$$
32/4=8
$$

每个 rank 的 DataLoader：

``` text
batch_size = 8
```

全局有效 batch：

``` text
8 × 4 = 32
```

------------------------------------------------------------------------

## 13. 为什么 total batch 必须能整除 world size？

源码：

``` python
assert (
    total_batch_size
    % dist_utils.get_world_size()
    == 0
)
```

否则每个 rank 无法得到一致的 local batch size。

------------------------------------------------------------------------

# 第五部分：Optimizer 是怎样拿到参数的？

## 14. X 的 AdamW 不是所有参数一个 LR

X config：

``` yaml
optimizer:
  type: AdamW

  params:
    - params: '^(?=.*.dinov3)(?!.*(?:norm|bn|bias)).*$'
      lr: 0.00001

    - params: '^(?=.*.dinov3)(?=.*(?:norm|bn|bias)).*$'
      lr: 0.00001
      weight_decay: 0.

    - params: '^(?=.*(?:sta|encoder|decoder))(?=.*(?:norm|bn|bias)).*$'
      weight_decay: 0.

  lr: 0.0005
  betas: [0.9, 0.999]
  weight_decay: 0.000125
```

------------------------------------------------------------------------

## 15. Regex 参数组怎样工作？

源码：

``` python
params = {
    k: v
    for k, v in model.named_parameters()
    if v.requires_grad
    and len(re.findall(pattern, k)) > 0
}
```

然后未匹配的 trainable parameters：

``` python
unseen = set(names) - set(visited)
```

自动进入默认 group。

所以：

``` text
DINOv3
→ 小 LR

DINOv3 norm/bias
→ 小 LR + no weight decay

STA/encoder/decoder norm/bias
→ default LR + no weight decay

其余 trainable params
→ default LR + default weight decay
```

------------------------------------------------------------------------

## 16. 为什么这一点必须实际打印？

因为 regex 是按：

``` text
parameter name
```

匹配。

模型一旦重构命名，参数可能落入不同 group。

建议：

``` python
for i, pg in enumerate(
    optimizer.param_groups
):
    print(
        i,
        pg["lr"],
        pg.get(
            "weight_decay"
        ),
        len(pg["params"])
    )
```

更进一步，建立：

``` text
parameter name → optimizer group
```

映射表。

------------------------------------------------------------------------

# 第六部分：Dataset 从 COCO JSON 得到什么？

## 17. COCO 配置

``` yaml
train_dataloader:
  dataset:
    type: CocoDetection
    img_folder: ...
    ann_file: ...
```

`CocoDetection.__getitem__()`：

``` python
img, target = self.load_item(idx)

if self._transforms is not None:
    img, target, _ = self._transforms(
        img,
        target,
        self
    )

return img, target
```

------------------------------------------------------------------------

## 18. 原始 COCO bbox 是什么格式？

COCO annotation：

``` text
[x, y, width, height]
```

源码先：

``` python
boxes[:, 2:] += boxes[:, :2]
```

变成：

``` text
[x1,y1,x2,y2]
```

并 clamp 到图像边界。

------------------------------------------------------------------------

## 19. `target` 初始包含什么？

转换后典型 target：

``` python
{
    "boxes": ...,
    "labels": ...,
    "image_id": ...,
    "area": ...,
    "iscrowd": ...,
    "orig_size": ...,
    "idx": ...
}
```

其中：

``` text
boxes:
[Ni,4]

labels:
[Ni]

Ni:
第 i 张图片当前目标数量
```

所以 batch 内：

``` text
每张图的 GT 数量可以不同
```

这就是为什么 targets 是：

``` text
list[dict]
```

而不是简单 `[B,N,4]`。

------------------------------------------------------------------------

## 20. `orig_size` 一个值得注意的源码细节

当前 COCO dataset 中：

``` python
target["orig_size"] = torch.as_tensor(
    [int(w), int(h)]
)
```

也就是：

``` text
[width,height]
```

后面 evaluation：

``` python
orig_target_sizes = torch.stack(
    [t["orig_size"] for t in targets],
    dim=0
)
```

传给 PostProcessor。

因此读坐标代码时不要凭经验假定一定是 `[h,w]`，要顺着仓库自己的约定追。

------------------------------------------------------------------------

# 第七部分：Per-image Transform Pipeline

## 21. X 当前训练 transforms

X 配置覆盖为：

``` text
Mosaic
RandomPhotometricDistort
RandomZoomOut
RandomIoUCrop
SanitizeBoundingBoxes
RandomHorizontalFlip
Resize 640×640
SanitizeBoundingBoxes
ConvertPILImage
Normalize
ConvertBoxes(cxcywh, normalize=True)
```

注意：

``` text
MixUp / CopyBlend
```

不在这里完成。

它们位于：

``` text
collate_fn
```

阶段。

------------------------------------------------------------------------

## 22. 为什么 augmentation 分成两层？

### Dataset-level

单个 sample：

``` text
image_i + target_i
```

适合：

``` text
crop
flip
photometric
Mosaic（可主动读取其他样本）
```

### Batch-level / collate-level

已经拿到：

``` text
B 张 image + B 个 targets
```

适合：

``` text
MixUp
CopyBlend
multi-scale batch resize
```

这个分层是理解第 17 课训练增强真正落在哪段代码里的关键。

------------------------------------------------------------------------

# 第八部分：Epoch Policy 如何进入 Dataset 和 Collate？

## 23. 每个 epoch 开始

`DetSolver.fit()`：

``` python
self.train_dataloader.set_epoch(
    epoch
)
```

DataLoader：

``` python
def set_epoch(self, epoch):
    self._epoch = epoch
    self.dataset.set_epoch(epoch)
    self.collate_fn.set_epoch(epoch)
```

因此 epoch 信息同时流向：

``` text
Dataset transforms
CollateFunction
```

这就是 augmentation schedule 能随 epoch 改变的基础。

------------------------------------------------------------------------

## 24. DDP sampler 也要 `set_epoch`

分布式时：

``` python
self.train_dataloader.sampler.set_epoch(
    epoch
)
```

这是为了让 DistributedSampler：

``` text
不同 epoch 使用不同 shuffle
```

同时让各 rank 的数据划分保持正确协调。

------------------------------------------------------------------------

# 第九部分：DataLoader 怎样形成一个 batch？

## 25. DataLoader workers

配置：

``` text
num_workers = 4
```

注意在 4-rank DDP 下，通常是：

``` text
每个 rank
4 workers
```

总 worker 数量可达到：

``` text
4 ranks × 4 = 16 workers
```

这也是为什么 DataLoader 的：

``` text
CPU
RAM
shared memory
file descriptors
```

问题经常与 GPU 模型本身无关。

------------------------------------------------------------------------

## 26. Collate 输入是什么？

假设 local batch = 8：

``` text
items = [
  (img0,target0),
  (img1,target1),
  ...
  (img7,target7)
]
```

每张图在 dataset transforms 后通常已经同尺寸。

------------------------------------------------------------------------

## 27. Collate 首先 stack images

源码：

``` python
images = torch.cat(
    [x[0][None] for x in items],
    dim=0
)

targets = [
    x[1]
    for x in items
]
```

所以：

``` text
images:
[B,3,H,W]

targets:
长度 B 的 list
```

例如：

``` text
images:
[8,3,640,640]

targets:
[
  {"boxes":[12,4], ...},
  {"boxes":[ 5,4], ...},
  {"boxes":[21,4], ...},
  ...
]
```

------------------------------------------------------------------------

# 第十部分：MixUp 在 batch 里发生什么？

## 28. MixUp

源码思想：

``` python
images.roll(
    shifts=1,
    dims=0
)
```

也就是把 batch：

``` text
[A,B,C,D]
```

变成：

``` text
[D,A,B,C]
```

再：

$$
I'_i = \beta I_i + (1-\beta)I_{i-1}
$$

当前 beta 随机范围：

``` text
0.45 ~ 0.55
```

------------------------------------------------------------------------

## 29. MixUp 不只是混图像

GT 也会 concat：

``` python
updated_targets[i]["boxes"] =
    cat(
        target_i boxes,
        shifted_target_i boxes
    )
```

labels、area 同样合并。

还加入：

``` text
target["mixup"]
```

用于告诉 loss 每个目标的混合权重。

这就是为什么 MixUp 能增加 O2O positive supervision。

------------------------------------------------------------------------

# 第十一部分：CopyBlend 在 batch 里发生什么？

## 30. CopyBlend 先建立 object pool

遍历 batch 中所有目标：

``` text
boxes
labels
areas
source image
```

过滤：

``` text
area >= area_threshold
```

的对象。

------------------------------------------------------------------------

## 31. 然后把对象复制到其他图像

对于目标框：

``` text
normalized cxcywh
```

先还原成 pixel crop：

``` text
x1_src
y1_src
x2_src
y2_src
```

再随机选择目标图位置。

------------------------------------------------------------------------

## 32. blend 模式

``` python
blended_patch =
    destination * beta
    +
    source * (1-beta)
```

然后把新 box / label / area 加入目标图 target。

所以 CopyBlend 同样会改变：

``` text
图像
+
GT 集合
```

------------------------------------------------------------------------

# 第十二部分：Multi-scale training

## 33. Collate 最后可能再次 resize 整个 batch

源码：

``` python
if (
    self.scales is not None
    and self.epoch < self.stop_epoch
):
    sz = random.choice(
        self.scales
    )

    images = F.interpolate(
        images,
        size=sz
    )
```

所以：

> 配置里有 `Resize 640×640`，不代表模型训练时每个 batch 永远是 640×640。

------------------------------------------------------------------------

## 34. X 的 `base_size_repeat=3`

最终实际 scale list 应由：

``` python
generate_scales(
    base_size,
    base_size_repeat
)
```

运行时打印确认。

源码围绕：

``` text
0.75 × base_size
到
1.25 × base_size
```

按 32 对齐生成候选尺寸，并重复 base size 若干次。

------------------------------------------------------------------------

## 35. 为什么 boxes 不需要在这里重新 scale？

因为 dataset transform 最后已经：

``` text
ConvertBoxes
fmt = cxcywh
normalize = True
```

boxes 是：

``` text
0~1 normalized coordinates
```

整张图等比例 resize 到新正方形尺寸时，normalized box 数值仍可保持。

这是一个非常漂亮的数据管线设计。

------------------------------------------------------------------------

# 第十三部分：Batch 正式进入训练循环

## 36. `train_one_epoch()`

最核心的循环：

``` python
for i, (
    samples,
    targets
) in enumerate(data_loader):
```

一个 iteration 就从这里开始。

------------------------------------------------------------------------

## 37. CPU → GPU

源码：

``` python
samples = samples.to(device)

targets = [
    {
        k: v.to(device)
        for k, v in t.items()
    }
    for t in targets
]
```

所以：

``` text
images tensor
+
target 中所有 tensor
```

都会搬到当前 rank 的 GPU。

------------------------------------------------------------------------

## 38. 一个 batch 的典型结构

例如 local B=8：

``` text
samples:
[8,3,S,S]

targets:
list length = 8
```

其中：

``` text
S
```

可能是当前 multi-scale 随机出的尺寸，而不一定是 640。

第 i 张：

``` text
targets[i]["boxes"]:
[Ni,4]

targets[i]["labels"]:
[Ni]

targets[i]["mixup"]:
可选 [Ni]
```

------------------------------------------------------------------------

# 第十四部分：Meta 信息

## 39. 每个 iteration 还构造

``` python
global_step = (
    epoch * len(data_loader)
    + i
)

metas = dict(
    epoch=epoch,
    step=i,
    global_step=global_step,
    epoch_step=len(data_loader)
)
```

随后：

``` python
criterion(
    outputs,
    targets,
    **metas
)
```

所以 criterion 知道：

``` text
当前 epoch
当前 iteration
全局 step
一个 epoch 有多少 step
```

这使 matcher/loss 能实现 schedule。

------------------------------------------------------------------------

# 第十五部分：AMP 路径

## 40. 如果 scaler 存在

``` python
with torch.autocast(
    device_type=str(device),
    cache_enabled=True
):
    outputs = model(
        samples,
        targets=targets
    )
```

模型 forward 在 autocast 下运行。

------------------------------------------------------------------------

## 41. 为什么 criterion 又关闭 autocast？

源码：

``` python
with torch.autocast(
    device_type=str(device),
    enabled=False
):
    loss_dict = criterion(
        outputs,
        targets,
        **metas
    )
```

也就是说：

``` text
model forward:
mixed precision

criterion:
autocast disabled
```

这有助于让：

``` text
matching
IoU
bbox loss
distribution loss
```

等数值敏感部分使用更稳定的精度路径。

------------------------------------------------------------------------

# 第十六部分：模型 Forward

## 42. 训练态调用

``` python
outputs = model(
    samples,
    targets=targets
)
```

顶层 DEIM：

``` text
samples
↓
backbone
↓
encoder
↓
decoder(features, targets)
↓
outputs
```

第 22 课已经完整追过 feature / query shape。

本课只强调一个新点：

> **训练时 targets 会传进 decoder。**

这是因为 decoder 需要生成：

``` text
denoising queries
```

------------------------------------------------------------------------

## 43. 训练态 outputs 比推理态复杂

典型包含：

``` text
pred_logits
pred_boxes

aux_outputs

enc_aux_outputs
enc_meta

pre_outputs

pred_corners
ref_points

dn_aux_outputs
dn_pre_outputs
dn_meta
```

具体 key 以当前 commit 的 runtime：

``` python
print(outputs.keys())
```

为准。

------------------------------------------------------------------------

# 第十七部分：为什么 Decoder 要看到 GT？

## 44. Denoising training

训练时利用 GT 构造带噪：

``` text
class labels
boxes
```

形成：

``` text
DN queries
```

与普通 matching queries 一起进入 decoder。

目的：

``` text
让 decoder 更快学会
“已知附近有一个 object 时怎样恢复正确 label/box”
```

------------------------------------------------------------------------

## 45. DN 与 Hungarian Matching 不要混淆

Matching queries：

``` text
prediction
↔
GT

通过 Hungarian matching
```

DN queries：

``` text
GT 直接构造 noisy query
```

它们是不同 supervision 路径。

------------------------------------------------------------------------

# 第十八部分：Criterion 才是"训练逻辑中心"

## 46. 模型只负责预测

这是必须建立的架构意识：

``` text
model
→ outputs
```

而：

``` text
prediction 与 GT 如何匹配
loss 怎样计算
哪些层接受监督
```

主要在：

``` text
DEIMCriterion
```

完成。

------------------------------------------------------------------------

## 47. X 的 loss 配置来自 base

``` yaml
DEIMCriterion:
  weight_dict:
    loss_mal: 1
    loss_bbox: 5
    loss_giou: 2
    loss_fgl: 0.15
    loss_ddf: 1.5

  losses:
    - mal
    - boxes
    - local

  gamma: 1.5
  alpha: 0.75
  reg_max: 32
```

------------------------------------------------------------------------

# 第十九部分：Hungarian Matching 在哪里发生？

## 48. Matcher 配置

``` yaml
matcher:
  type: HungarianMatcher

  weight_dict:
    cost_class: 2
    cost_bbox: 5
    cost_giou: 2

  alpha: 0.25
  gamma: 2.0

  change_matcher: True
  iou_order_alpha: 4.0
  matcher_change_epoch: 45
```

------------------------------------------------------------------------

## 49. Cost 与 Loss 再区分一次

Matching cost：

``` text
只决定：
哪个 query ↔ 哪个 GT
```

例如：

$$
C = 2C_{cls} + 5C_{L1} + 2C_{GIoU}
$$

真正训练 loss：

``` text
决定：
匹配后怎样反向传播
```

两者权重数值即使相似，也不是同一个东西。

------------------------------------------------------------------------

# 第二十部分：一张图的 O2O Matching

## 50. 假设

``` text
300 matching queries
7 GT
```

cost matrix：

``` text
[300,7]
```

Hungarian：

``` text
7 对 one-to-one matches
```

概念：

``` text
q12 ↔ gt0
q31 ↔ gt1
q48 ↔ gt2
...
```

其他 queries 不成为这 7 个 GT 的 O2O positive match。

------------------------------------------------------------------------

# 第二十一部分：DEIMv2 的 Dense O2O 在系统层面怎样出现？

## 51. 它不是单一函数

前几课已经看到 Dense O2O v2 是一整套训练系统：

``` text
Mosaic
MixUp
CopyBlend
+
更多有效 GT
+
O2O matcher
+
MAL
+
matcher schedule
```

本课现在可以把它放回真正的调用位置：

``` text
Dataset / Collate
    ↓
GT density 上升
    ↓
Model
    ↓
Matcher
    ↓
Criterion
    ↓
denser useful O2O supervision
```

------------------------------------------------------------------------

# 第二十二部分：Loss Dict

## 52. Criterion 返回的不是一个 loss

而是：

``` python
loss_dict
```

概念：

``` python
{
    "loss_mal": ...,
    "loss_bbox": ...,
    "loss_giou": ...,
    "loss_fgl": ...,
    "loss_ddf": ...,

    "...aux...": ...,
    "...dn...": ...,
    "...enc...": ...
}
```

具体 key 数量随：

``` text
decoder layers
auxiliary supervision
DN
encoder auxiliary
```

而变化。

------------------------------------------------------------------------

## 53. 总 Loss

训练 loop 非常直接：

``` python
loss = sum(
    loss_dict.values()
)
```

这说明 criterion 返回的各项通常已经应用了相应权重 / 组织逻辑。

不要在外部训练 loop 再随意乘一遍。

------------------------------------------------------------------------

# 第二十三部分：Backward 到底发生了什么？

## 54. 非 AMP 路径

``` python
optimizer.zero_grad()

loss.backward()

if max_norm > 0:
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm
    )

optimizer.step()
```

数学上：

$$
g_t = \nabla_\theta L(\theta_t)
$$

然后 AdamW 根据：

``` text
gradient
first moment
second moment
weight decay
learning rate
```

更新参数。

------------------------------------------------------------------------

## 55. 为什么先 `zero_grad()`？

PyTorch 默认：

``` text
gradient 会累加
```

如果不清零：

``` text
batch 2 gradient
=
batch1 gradient
+
batch2 gradient
```

这只有做 gradient accumulation 时才是有意行为。

当前普通训练不是这样。

------------------------------------------------------------------------

# 第二十四部分：AMP 的 backward

## 56. AMP 路径

``` python
loss = sum(
    loss_dict.values()
)

scaler.scale(
    loss
).backward()
```

核心是：

$$
L_{scaled} = sL
$$

所以：

$$
\nabla(sL) = s\nabla L
$$

把小梯度放大，降低 FP16 underflow 风险。

------------------------------------------------------------------------

## 57. Clip 前必须 unscale

源码：

``` python
scaler.unscale_(
    optimizer
)

torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm
)
```

否则你 clip 的是：

``` text
被 scale 后的假梯度大小
```

而不是实际梯度。

------------------------------------------------------------------------

## 58. AMP optimizer step

``` python
scaler.step(
    optimizer
)

scaler.update()

optimizer.zero_grad()
```

GradScaler 会检查：

``` text
Inf / NaN gradient
```

并动态调整 scale。

------------------------------------------------------------------------

# 第二十五部分：Gradient Clipping

## 59. base optimizer

``` yaml
clip_max_norm: 0.1
```

所以：

$$
\|g\|_2 > 0.1
$$

时会按比例缩放整体梯度，使 norm 不超过阈值。

------------------------------------------------------------------------

## 60. 为什么 DETR 系列常见 gradient clipping？

Transformer / matching / bbox refinement 训练早期可能出现：

``` text
大梯度
不稳定更新
```

clip 是训练稳定性保护之一。

但：

``` text
gradient clipping
≠
解决所有 NaN 的万能工具
```

如果输入、loss、学习率或 AMP 已经异常，仍要找根因。

------------------------------------------------------------------------

# 第二十六部分：AdamW 更新

## 61. AdamW 简化公式

一阶矩：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
$$

二阶矩：

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$

参数更新近似：

$$
\theta_{t+1} = \theta_t - \eta
\frac{\hat m_t} {\sqrt{\hat v_t}+\epsilon} -
\eta\lambda\theta_t
$$

最后一项体现 decoupled weight decay。

------------------------------------------------------------------------

# 第二十七部分：EMA

## 62. 每次 optimizer step 后

源码：

``` python
if ema is not None:
    ema.update(model)
```

EMA 参数：

$$
\theta^{EMA}_t = d\theta^{EMA}_{t-1} +
(1-d)\theta_t
$$

------------------------------------------------------------------------

## 63. 为什么评估 EMA 而不是 raw model？

epoch 结束：

``` python
module = (
    self.ema.module
    if self.ema
    else self.model
)
```

再：

``` python
evaluate(module, ...)
```

EMA 相当于参数轨迹的平滑版本，通常能提供更稳定的验证性能。

------------------------------------------------------------------------

# 第二十八部分：LR Scheduler

## 64. DEIMv2 当前 base 使用 self-defined flat-cosine

配置：

``` yaml
lrsheduler: flatcosine
lr_gamma: 0.5
warmup_iter: 2000
flat_epoch: 29
no_aug_epoch: 8
```

`DetSolver.fit()` 检测到：

``` python
if args.lrsheduler is not None:
```

则构造：

``` text
FlatCosineLRScheduler
```

并设置：

``` text
self_lr_scheduler = True
```

------------------------------------------------------------------------

## 65. 它按 iteration 更新

训练 loop：

``` python
optimizer = lr_scheduler.step(
    cur_iters + i,
    optimizer
)
```

而不是等 epoch 结束才更新。

------------------------------------------------------------------------

## 66. 另一条 scheduler 路径

如果没有 self-defined scheduler：

``` text
warmup scheduler
```

可以在 iteration 内 step。

epoch 结束再：

``` python
self.lr_scheduler.step()
```

所以读训练代码时必须先判断：

``` text
当前 config 到底走哪条 scheduler 分支？
```

------------------------------------------------------------------------

# 第二十九部分：Logging

## 67. DDP 下 loss 需要 reduce

每个 rank 有自己的 local batch 和 local loss。

源码：

``` python
loss_dict_reduced =
    dist_utils.reduce_dict(
        loss_dict
    )
```

用于日志统计。

注意：

> 用于 logging 的 reduced loss 与 DDP backward 中的 gradient
> synchronization 是两个不同机制。

------------------------------------------------------------------------

## 68. MetricLogger

记录：

``` text
loss
各 loss component
learning rate
iteration time
data time
memory
```

epoch 结束：

``` python
metric_logger.synchronize_between_processes()
```

再打印 averaged stats。

------------------------------------------------------------------------

## 69. TensorBoard

main process 每 10 global steps：

``` text
Loss/total
Lr/pg_0
Lr/pg_1
...
Loss/loss_mal
Loss/loss_bbox
...
```

因此如果你要研究训练是否健康，TensorBoard 是非常有价值的。

------------------------------------------------------------------------

# 第三十部分：一个 iteration 的精确顺序

## 70. 非 AMP

``` text
DataLoader next()
↓
samples.to(GPU)
targets.to(GPU)
↓
model(samples, targets)
↓
criterion(outputs, targets)
↓
loss = sum(loss_dict)
↓
optimizer.zero_grad()
↓
loss.backward()
↓
clip_grad_norm_
↓
optimizer.step()
↓
EMA.update()
↓
LR step / warmup step
↓
reduce loss for logging
↓
next batch
```

------------------------------------------------------------------------

## 71. AMP

``` text
DataLoader next()
↓
GPU transfer
↓
autocast
  model forward
↓
autocast disabled
  criterion
↓
sum loss
↓
scaler.scale(loss).backward()
↓
scaler.unscale_(optimizer)
↓
clip gradients
↓
scaler.step(optimizer)
↓
scaler.update()
↓
optimizer.zero_grad()
↓
EMA.update()
↓
scheduler
↓
logging
```

------------------------------------------------------------------------

# 第三十一部分：为什么 AMP 路径和非 AMP 的 zero_grad 位置不同？

源码中：

非 AMP：

``` text
zero_grad
→ backward
→ step
```

AMP：

``` text
backward
→ step
→ zero_grad
```

只要上一次 iteration 结束后梯度已经被清掉，两种组织都能工作。

但你自己改代码做：

``` text
gradient accumulation
multiple optimizers
manual backward
```

时，就必须重新审视这个顺序。

------------------------------------------------------------------------

# 第三十二部分：Epoch 结束后发生什么？

## 72. Scheduler

如果不是 self-defined iteration scheduler：

``` python
self.lr_scheduler.step()
```

------------------------------------------------------------------------

## 73. `last_epoch`

``` python
self.last_epoch += 1
```

用于：

``` text
resume
checkpoint
training state
```

------------------------------------------------------------------------

# 第三十三部分：Checkpoint

## 74. 普通 checkpoint

在指定阶段：

``` text
last.pth
checkpointXXXX.pth
```

保存的是：

``` python
self.state_dict()
```

而不是只有：

``` text
model.state_dict()
```

------------------------------------------------------------------------

## 75. Solver checkpoint 里有什么？

`BaseSolver.state_dict()` 遍历 solver 中所有拥有：

``` python
state_dict()
```

的对象。

典型包括：

``` text
model
optimizer
lr_scheduler
EMA
GradScaler
last_epoch
...
```

所以：

``` text
resume checkpoint
```

是"继续训练状态"。

------------------------------------------------------------------------

# 第三十四部分：Resume 与 Tuning 完全不同

## 76. Resume

``` bash
-r checkpoint.pth
```

目标：

``` text
继续原来的训练
```

需要恢复：

``` text
model
optimizer
scheduler
EMA
scaler
last_epoch
...
```

------------------------------------------------------------------------

## 77. Tuning

``` bash
-t pretrained.pth
```

目标：

``` text
只借用预训练模型参数
```

不会把旧 optimizer training trajectory 原封不动继续。

------------------------------------------------------------------------

## 78. 为什么 Tuning 先于 EMA 创建？

源码注释明确：

``` text
Must load_tuning_state before EMA instance building
```

顺序：

``` text
model
↓
load tuning weights
↓
DDP wrap
↓
EMA
```

否则 EMA 可能基于错误的初始参数创建。

------------------------------------------------------------------------

# 第三十五部分：跨数据集 head mismatch

## 79. Tuning 时分类头可能尺寸不同

例如：

``` text
pretrain classes ≠ custom dataset classes
```

源码会处理：

``` text
denoising_class_embed
enc_score_head
decoder score heads
```

的 shape mismatch，并允许：

``` text
strict=False
```

加载匹配参数。

这就是为什么：

``` text
tuning
```

与：

``` text
resume
```

必须严格区分。

------------------------------------------------------------------------

# 第三十六部分：DEIMv2 的两阶段训练控制

## 80. `stop_epoch`

X：

``` text
stop_epoch = 50
```

它不仅表示：

``` text
停止 multiscale
```

在 solver 中还被用作一个 stage boundary。

------------------------------------------------------------------------

## 81. 到 stop_epoch 时

源码：

``` python
if epoch ==
   self.train_dataloader.collate_fn.stop_epoch:

    barrier()

    self.load_resume_state(
        best_stg1.pth
    )

    self.ema.decay =
        ema_restart_decay
```

所以阶段切换时会：

``` text
重新加载 stage-1 best
+
刷新 EMA decay
```

这比"到第 50 epoch 关掉 augmentation"更复杂。

------------------------------------------------------------------------

## 82. Best checkpoints

阶段前：

``` text
best_stg1.pth
```

阶段后：

``` text
best_stg2.pth
```

所以 DEIMv2 当前训练 recipe 本身包含：

``` text
stage-aware best-model selection
```

------------------------------------------------------------------------

# 第三十七部分：为什么第 17/18 课必须与 Solver 连起来看？

你之前看到：

``` text
Mosaic schedule
MixUp schedule
CopyBlend schedule
EMA
FlatCosine LR
```

现在可以看到它们不是互相独立：

``` text
augmentation schedule
        │
        ▼
stop_epoch
        │
        ├── multiscale stop
        ├── stage boundary
        ├── load best_stg1
        └── EMA refresh

LR schedule
        │
        └── no_aug_epoch 等阶段参数

evaluation
        │
        └── best_stg1 / best_stg2
```

训练 recipe 是一个**时间系统**，不是参数列表。

------------------------------------------------------------------------

# 第三十八部分：Validation

## 83. 每个 epoch 训练后

``` python
module = (
    self.ema.module
    if self.ema
    else self.model
)

test_stats, coco_evaluator = evaluate(...)
```

------------------------------------------------------------------------

## 84. Evaluation 不传 targets 给 model

``` python
outputs = model(samples)
```

所以：

``` text
没有 DN queries
```

然后：

``` python
results = postprocessor(
    outputs,
    orig_target_sizes
)
```

------------------------------------------------------------------------

# 第三十九部分：PostProcessor

## 85. 第 22 课已经看到

``` text
pred_logits
[B,300,80]

pred_boxes
[B,300,4]
```

经过：

``` text
sigmoid
top-k
cxcywh → xyxy
normalized → pixel
```

得到每张图：

``` text
labels
boxes
scores
```

------------------------------------------------------------------------

# 第四十部分：COCO Evaluator

## 86. 预测按 image_id 组织

源码：

``` python
res = {
    target["image_id"].item():
        output

    for target, output
    in zip(targets, results)
}
```

然后：

``` python
coco_evaluator.update(res)
```

------------------------------------------------------------------------

## 87. DDP evaluation

所有 rank 评估不同数据后：

``` text
metric_logger synchronize
coco_evaluator synchronize
accumulate
summarize
```

最终得到 COCO bbox stats。

------------------------------------------------------------------------

# 第四十一部分：COCO AP 不是单一 IoU=0.5

最常见：

``` text
AP
AP50
AP75
AP_small
AP_medium
AP_large
AR...
```

其中主 AP：

$$
AP = \text{mean AP over IoU thresholds}
$$

通常从：

``` text
0.50
0.55
...
0.95
```

取平均。

所以：

``` text
AP50 高
```

并不等于定位质量在严格 IoU 下同样好。

------------------------------------------------------------------------

# 第四十二部分：Best Model 怎样决定？

## 88. `test_stats`

对于 bbox：

``` text
test_stats["coco_eval_bbox"][0]
```

通常就是 COCO 主 AP。

solver 维护：

``` text
best_stat
top1
```

并根据阶段保存：

``` text
best_stg1.pth
best_stg2.pth
```

------------------------------------------------------------------------

# 第四十三部分：日志文件

## 89. 每个 epoch

main process 写：

``` text
output_dir/log.txt
```

每行 JSON：

``` text
train_...
test_...
epoch
n_parameters
```

------------------------------------------------------------------------

## 90. Evaluator state

还会保存：

``` text
output_dir/eval/latest.pth
```

并周期保存 epoch evaluator snapshot。

这与：

``` text
模型 checkpoint
```

不是同一个文件。

------------------------------------------------------------------------

# 第四十四部分：完整系统调用图

``` text
torchrun
│
├─ rank0 train.py
├─ rank1 train.py
├─ rank2 train.py
└─ rank3 train.py
        │
        ▼
setup_distributed
        │
        ▼
YAMLConfig
        │
        ├─ model
        ├─ criterion
        ├─ optimizer
        ├─ EMA
        ├─ scaler
        ├─ train loader
        ├─ val loader
        └─ evaluator
        │
        ▼
DetSolver.fit
        │
        ▼
BaseSolver._setup
        │
        ├─ device
        ├─ model.to
        ├─ DDP
        ├─ criterion.to
        ├─ EMA
        └─ resume/tuning
        │
        ▼
for epoch
        │
        ├─ loader.set_epoch
        ├─ sampler.set_epoch
        │
        ▼
train_one_epoch
        │
        ▼
COCO Dataset
        │
        ├─ load image
        ├─ load annotations
        ├─ xywh → xyxy
        └─ category remap
        │
        ▼
per-image transforms
        │
        ├─ Mosaic
        ├─ photometric
        ├─ zoom-out
        ├─ IoU crop
        ├─ sanitize
        ├─ flip
        ├─ resize
        ├─ normalize
        └─ boxes → normalized cxcywh
        │
        ▼
collate
        │
        ├─ stack images
        ├─ MixUp
        ├─ CopyBlend
        └─ multiscale resize
        │
        ▼
samples, targets
        │
        ▼
GPU
        │
        ▼
autocast
        │
        ▼
DEIM
        │
        ├─ DINOv3STAs
        ├─ HybridEncoder
        └─ DEIMTransformer
              │
              ├─ matching queries
              └─ DN queries
        │
        ▼
outputs
        │
        ▼
DEIMCriterion
        │
        ├─ HungarianMatcher
        ├─ MAL
        ├─ L1 bbox
        ├─ GIoU
        ├─ FGL
        ├─ DDF
        ├─ auxiliary
        └─ denoising
        │
        ▼
loss_dict
        │
        ▼
sum
        │
        ▼
backward
        │
        ▼
DDP gradient synchronization
        │
        ▼
unscale / gradient clip
        │
        ▼
AdamW.step
        │
        ▼
EMA.update
        │
        ▼
LR scheduler
        │
        ▼
next batch
        │
        ▼
epoch end
        │
        ├─ checkpoint
        │
        ▼
EMA model
        │
        ▼
evaluate
        │
        ├─ model(samples)
        ├─ PostProcessor
        ├─ image_id mapping
        └─ CocoEvaluator
        │
        ▼
AP
        │
        ├─ best_stg1
        └─ best_stg2
```

------------------------------------------------------------------------

# 第四十五部分：一条 batch 的 Tensor 视角

假设：

``` text
world_size = 4
total_batch_size = 32
local B = 8
current size = 640
```

rank 0：

``` text
samples
[8,3,640,640]
```

targets：

``` text
8 dictionaries
```

例如：

``` text
GT counts:
[12,5,9,17,4,23,7,11]
```

总 GT：

$$
12+5+9+17+4+23+7+11=88
$$

如果 MixUp：

``` text
GT 数可能大幅增加
```

CopyBlend：

``` text
再增加若干 objects
```

然后每张图：

``` text
300 matching queries
+
DN queries
```

进入 decoder。

------------------------------------------------------------------------

# 第四十六部分：为什么 target 不能 stack？

如果 GT 数：

``` text
12
5
9
17
...
```

无法直接：

``` python
torch.stack(
    boxes
)
```

因为第一维不同。

所以：

``` text
targets = list[dict]
```

是目标检测训练里非常常见的设计。

------------------------------------------------------------------------

# 第四十七部分：DDP 的 Gradient 同步发生在哪里？

## 91. 你不会在训练 loop 里看到

``` python
all_reduce(
    parameter.grad
)
```

因为：

``` text
DistributedDataParallel
```

通过 autograd hooks 在 backward 过程中自动组织 gradient reduction。

所以：

``` python
loss.backward()
```

背后已经触发 distributed gradient synchronization。

------------------------------------------------------------------------

# 第四十八部分：Local loss 与 Global update

假设 4 ranks：

``` text
rank0 batch → g0
rank1 batch → g1
rank2 batch → g2
rank3 batch → g3
```

DDP 归约后概念：

$$
g = \frac{
g_0+g_1+g_2+g_3
}{4}
$$

然后每个 rank 都用同步后的 gradient 更新自己的 model replica。

因此 optimizer step 后：

``` text
各 rank 参数继续保持一致
```

------------------------------------------------------------------------

# 第四十九部分：为什么 Dataset shuffle 还要 DistributedSampler？

不能让四个 rank 都训练同一批数据。

理想：

``` text
rank0 → shard A
rank1 → shard B
rank2 → shard C
rank3 → shard D
```

然后每 epoch 重新 shuffle / partition。

------------------------------------------------------------------------

# 第五十部分：训练异常如何按系统层定位？

## 92. 数据读取报错

常见位置：

``` text
CocoDetection
worker
annotation parsing
file path
```

优先检查：

``` text
img_folder
ann_file
COCO JSON
num_workers
```

------------------------------------------------------------------------

## 93. Augmentation bbox 报错

位置：

``` text
Compose
Mosaic
IoU crop
SanitizeBoundingBoxes
CopyBlend
```

检查：

``` text
boxes format
normalized?
empty boxes?
out of bounds?
NaN?
```

------------------------------------------------------------------------

## 94. Collate shape 报错

检查：

``` text
每张图是否同尺寸
transform 是否遗漏 resize
multi-scale resize
masks
```

------------------------------------------------------------------------

## 95. CUDA OOM

先判断发生在哪：

``` text
model forward?
criterion?
backward?
evaluation?
```

再检查：

``` text
local batch
input scale
DN queries
mixed precision
DINOv3 trainability
```

------------------------------------------------------------------------

## 96. NaN loss

训练 loop 已显式检查：

``` text
pred_boxes NaN/Inf
```

以及：

``` text
loss_value finite
```

排查顺序建议：

``` text
input
targets
model outputs
individual loss
gradient norm
learning rate
AMP scale
```

------------------------------------------------------------------------

## 97. AP 为 0

不要先怀疑 Transformer。

优先检查：

``` text
class mapping
num_classes
COCO category remap
box format
box normalization
orig_size order
postprocessor
annotation IDs
```

工程上这些问题远比"attention 数学错了"常见。

------------------------------------------------------------------------

# 第五十一部分：最有价值的单 batch 调试法

## 98. 第一步：DataLoader

``` python
samples, targets = next(
    iter(
        solver.train_dataloader
    )
)

print(
    samples.shape
)

for i, t in enumerate(targets):
    print(
        i,
        t["boxes"].shape,
        t["labels"].shape,
        t["boxes"].min(),
        t["boxes"].max()
    )
```

------------------------------------------------------------------------

## 99. 第二步：GPU

``` python
samples = samples.to(
    solver.device
)

targets = [
    {
        k: v.to(
            solver.device
        )
        for k, v in t.items()
    }
    for t in targets
]
```

------------------------------------------------------------------------

## 100. 第三步：Forward

``` python
solver.model.train()

outputs = solver.model(
    samples,
    targets=targets
)

print(
    outputs.keys()
)

print(
    outputs[
        "pred_logits"
    ].shape
)

print(
    outputs[
        "pred_boxes"
    ].shape
)
```

------------------------------------------------------------------------

## 101. 第四步：Criterion

``` python
loss_dict = solver.criterion(
    outputs,
    targets,
    epoch=0,
    step=0,
    global_step=0,
    epoch_step=len(
        solver.train_dataloader
    )
)

for k, v in loss_dict.items():
    print(
        k,
        float(
            v.detach()
        )
    )
```

------------------------------------------------------------------------

## 102. 第五步：Backward

``` python
loss = sum(
    loss_dict.values()
)

solver.optimizer.zero_grad()

loss.backward()
```

------------------------------------------------------------------------

## 103. 第六步：Gradient

``` python
total_norm_sq = 0.0

for name, p in (
    solver.model
    .named_parameters()
):
    if p.grad is None:
        continue

    g = (
        p.grad
        .detach()
        .float()
    )

    total_norm_sq += (
        g.norm(2).item()
        ** 2
    )

print(
    "grad norm:",
    total_norm_sq ** 0.5
)
```

------------------------------------------------------------------------

## 104. 第七步：检查关键模块是否有梯度

``` python
keywords = [
    "dinov3",
    "sta",
    "encoder",
    "decoder"
]

for name, p in (
    solver.model
    .named_parameters()
):
    if any(
        k in name
        for k in keywords
    ):
        if p.grad is not None:
            print(
                name,
                p.grad.norm().item()
            )
```

这能发现：

``` text
某模块意外 frozen
某模块 disconnected
find_unused parameter
梯度爆炸
```

------------------------------------------------------------------------

# 第五十二部分：一个"训练流水线探针"

建议你写：

``` python
def inspect_batch(
    samples,
    targets
):
    print(
        "samples:",
        tuple(
            samples.shape
        ),
        samples.dtype,
        float(samples.min()),
        float(samples.max())
    )

    for i, t in enumerate(
        targets
    ):
        print(
            f"target[{i}]"
        )

        for k, v in t.items():
            if hasattr(
                v,
                "shape"
            ):
                print(
                    " ",
                    k,
                    tuple(v.shape),
                    v.dtype
                )
```

每次改 dataset 时先跑它。

------------------------------------------------------------------------

# 第五十三部分：Loss Probe

``` python
def inspect_losses(
    loss_dict
):
    total = 0.0

    for k, v in sorted(
        loss_dict.items()
    ):
        value = float(
            v.detach()
        )

        print(
            f"{k:40s}",
            f"{value:.6f}"
        )

        total += value

    print(
        "TOTAL",
        total
    )
```

这样你能立刻看到：

``` text
哪个 loss 占主导
哪个突然爆炸
哪个一直是 0
```

------------------------------------------------------------------------

# 第五十四部分：Optimizer Group Probe

``` python
def inspect_optimizer(
    optimizer
):
    for i, pg in enumerate(
        optimizer.param_groups
    ):
        n = sum(
            p.numel()
            for p in pg[
                "params"
            ]
        )

        print(
            f"group={i}",
            f"params={n:,}",
            f"lr={pg['lr']}",
            f"wd={pg.get('weight_decay')}"
        )
```

这对 DINOv3 differential LR 尤其重要。

------------------------------------------------------------------------

# 第五十五部分：更进一步------参数名到 group

``` python
param_to_name = {
    id(p): name
    for name, p
    in model.named_parameters()
}

for gi, pg in enumerate(
    optimizer.param_groups
):
    print(
        "\nGROUP",
        gi
    )

    for p in pg["params"]:
        print(
            param_to_name.get(
                id(p),
                "<unknown>"
            )
        )
```

正式训练前建议至少检查一次。

------------------------------------------------------------------------

# 第五十六部分：AMP Probe

训练时建议定期打印：

``` python
if scaler is not None:
    print(
        scaler.get_scale()
    )
```

如果 scale 持续快速下降，可能意味着：

``` text
频繁 overflow
```

应进一步检查：

``` text
loss
grad
learning rate
numerical instability
```

------------------------------------------------------------------------

# 第五十七部分：EMA Probe

可以比较 raw 与 EMA 某个参数：

``` python
raw = dict(
    model.named_parameters()
)

ema = dict(
    ema_model.named_parameters()
)

name = next(
    iter(raw)
)

diff = (
    raw[name].detach()
    -
    ema[name].detach()
).abs().mean()

print(
    name,
    diff.item()
)
```

这样 EMA 不再是一个抽象概念。

------------------------------------------------------------------------

# 第五十八部分：Checkpoint Probe

``` python
state = torch.load(
    "last.pth",
    map_location="cpu"
)

print(
    state.keys()
)
```

你应该亲自确认：

``` text
model
optimizer
EMA
scheduler
last_epoch
scaler
...
```

哪些真的存在于你当前 commit 的 checkpoint。

------------------------------------------------------------------------

# 第五十九部分：为什么不要只保存 `model.state_dict()`？

如果你要：

``` text
resume training
```

仅 model weights 不够。

你会丢失：

``` text
Adam moments
LR scheduler position
AMP scaler
EMA
epoch
```

训练轨迹会改变。

------------------------------------------------------------------------

# 第六十部分：为什么预训练模型又不需要 optimizer state？

因为 tuning 的目标不是：

``` text
继续原实验轨迹
```

而是：

``` text
用一个好的参数初始化
开始新的训练任务
```

所以：

``` text
resume
和
tuning
```

语义完全不同。

------------------------------------------------------------------------

# 第六十一部分：数据增强的真正时间轴

X 当前主 schedule 可概念画成：

``` text
epoch
0    4           29              50       58
|----|------------|---------------|--------|

early warm stage

4~29:
MixUp active window

4~50:
CopyBlend active window

Mosaic / photometric / crop:
由 transform policy 控制阶段变化

<50:
multi-scale batch resize

50:
stage boundary
reload best_stg1
EMA refresh

>=50:
late clean / fine-tune stage

58:
training end
```

具体 augmentation policy 的启停行为应结合 `Compose`
当前实现验证，而不能只根据 epoch 数字猜测。

------------------------------------------------------------------------

# 第六十二部分：为什么后期减少强增强？

训练早期：

``` text
需要丰富目标
需要密集 supervision
需要强 regularization
```

后期：

``` text
希望输入分布更接近真实验证数据
希望 box refinement 稳定
希望 EMA 在干净阶段收敛
```

因此：

``` text
strong augmentation → clean fine-tuning
```

是合理的训练阶段设计。

------------------------------------------------------------------------

# 第六十三部分：训练速度由哪些环节组成？

一个 iteration：

$$
T_{iter} = T_{data} + T_{H2D} + T_{forward} + T_{criterion} +
T_{backward} + T_{optimizer} + T_{communication}
$$

所以 GPU 利用率低，不一定是模型慢。

可能是：

``` text
DataLoader
Mosaic
CopyBlend
disk I/O
CPU
DDP communication
```

------------------------------------------------------------------------

# 第六十四部分：Data Time 很高怎么办？

依次检查：

``` text
storage speed
num_workers
CPU utilization
RAM
shared memory
Mosaic cache
image decoding
COCO JSON
network filesystem
```

不要直接增加 GPU。

------------------------------------------------------------------------

# 第六十五部分：为什么 `num_workers` 不是越大越好？

worker 太多会增加：

``` text
process overhead
RAM
shared memory
file descriptors
context switching
```

特别在多 GPU 下：

``` text
workers_per_rank × world_size
```

才是总 worker 数。

------------------------------------------------------------------------

# 第六十六部分：为什么训练时 Model Forward 不是全部开销？

DEIMv2 的 criterion 还包含：

``` text
matching
IoU
FDR/localization loss
auxiliary layers
DN
```

因此：

``` text
forward FPS
```

不能直接代表：

``` text
training iterations/sec
```

------------------------------------------------------------------------

# 第六十七部分：为什么 validation 可能突然很慢？

validation 包含：

``` text
model inference
PostProcessor
CPU/GPU transfer
COCO evaluator update
distributed synchronization
COCO accumulate
```

而且 val DataLoader 也可能成为瓶颈。

------------------------------------------------------------------------

# 第六十八部分：本课最重要的"职责分层"

``` text
train.py
负责启动

YAMLConfig
负责对象构建

Solver
负责训练生命周期

Dataset
负责单样本读取

Transforms
负责单样本增强

Collate
负责 batch 级增强与组 batch

Model
负责预测

Criterion
负责监督与 loss

Matcher
负责 O2O assignment

Optimizer
负责参数更新

EMA
负责参数平滑副本

Scheduler
负责 LR 时间变化

PostProcessor
负责模型输出 → 检测结果

Evaluator
负责指标
```

以后看到 bug，先问：

> 这个 bug 属于哪个职责层？

------------------------------------------------------------------------

# 第六十九部分：最容易混淆的 12 对概念

``` text
1.
Dataset transform
vs
Collate augmentation

2.
total batch size
vs
per-rank batch size

3.
model forward
vs
criterion

4.
matching cost
vs
training loss

5.
matching query
vs
DN query

6.
loss reduction for logging
vs
DDP gradient all-reduce

7.
optimizer step
vs
EMA update

8.
iteration scheduler
vs
epoch scheduler

9.
resume
vs
tuning

10.
raw model
vs
EMA model

11.
checkpoint
vs
evaluator snapshot

12.
model prediction
vs
PostProcessor result
```

------------------------------------------------------------------------

# 第七十部分：一个 Batch 的"生命故事"

现在用自然语言完整讲一遍。

假设：

``` text
4 GPUs
total batch = 32
```

每个 rank 的 DataLoader 得到：

``` text
8 张图
```

Dataset 对每张图：

``` text
读取 JPEG
读取 COCO annotation
过滤 crowd
xywh → xyxy
类别 remap
执行 Mosaic / crop / flip / resize / normalize
boxes → normalized cxcywh
```

Collate：

``` text
把 8 张图 stack
根据 epoch 可能做 MixUp
或者 CopyBlend
再根据 multi-scale schedule
随机 resize 整个 batch
```

得到：

``` text
samples [8,3,S,S]
targets list[8]
```

然后搬到当前 GPU。

------------------------------------------------------------------------

# 第七十一部分：Forward 的生命故事

``` text
samples
→ DINOv3STAs
→ HybridEncoder
→ DEIMTransformer
```

训练时 decoder 同时看到：

``` text
targets
```

用于生成：

``` text
DN queries
```

输出：

``` text
主预测
辅助层预测
encoder proposals
FDR distributions
DN predictions
```

------------------------------------------------------------------------

# 第七十二部分：Supervision 的生命故事

Criterion：

``` text
取 matching predictions
与 targets
```

构建：

``` text
Hungarian cost matrix
```

得到 O2O assignment。

然后计算：

``` text
MAL
bbox L1
GIoU
FGL
DDF
auxiliary losses
DN losses
```

组成：

``` text
loss_dict
```

------------------------------------------------------------------------

# 第七十三部分：Gradient 的生命故事

``` text
loss_dict
↓ sum
scalar loss
↓ backward
每个参数获得 local gradient
↓
DDP gradient synchronization
↓
unscale（AMP）
↓
clip
↓
AdamW
↓
新参数
```

------------------------------------------------------------------------

# 第七十四部分：参数更新后的生命故事

``` text
new raw model
↓
EMA.update
↓
new EMA model

同时：
LR scheduler 更新下一步 LR
```

然后开始下一 batch。

------------------------------------------------------------------------

# 第七十五部分：Epoch 的生命故事

所有 batch 训练完：

``` text
train stats aggregate
↓
checkpoint
↓
选择 EMA model
↓
val loader
↓
inference
↓
postprocess
↓
COCO evaluator
↓
AP
↓
更新 best model
↓
写 log
↓
下一个 epoch
```

------------------------------------------------------------------------

# 第七十六部分：你应该设置的断点

推荐顺序：

``` text
train.py::main

BaseSolver._setup

YAMLConfig.build_dataloader

CocoDetection.__getitem__

BatchImageCollateFunction.__call__

train_one_epoch

DEIM.forward

DEIMTransformer.forward

DEIMCriterion.forward

HungarianMatcher.forward

loss.backward 附近

optimizer.step 附近

ModelEMA.update

DetSolver.fit epoch end

evaluate

PostProcessor.forward

CocoEvaluator.update
```

------------------------------------------------------------------------

# 第七十七部分：第一次 Debug 不要直接 4 GPU

建议先：

``` text
1 GPU
B 小
num_workers=0
AMP off
一个 batch
```

这样：

``` text
异常堆栈最清楚
DataLoader worker 错误不会被包装
没有 DDP 干扰
```

确认单卡单 batch 正确后，再逐层恢复：

``` text
workers
AMP
DDP
完整 batch
```

------------------------------------------------------------------------

# 第七十八部分：`num_workers=0` 为什么特别适合 debug？

多 worker 时 Dataset 异常可能显示成：

``` text
Caught xxx in DataLoader worker process
```

而：

``` text
num_workers=0
```

数据代码就在主进程执行，断点和 traceback 都更直观。

------------------------------------------------------------------------

# 第七十九部分：单 batch Overfit Test

这是检测训练最有价值的 sanity check 之一。

选择：

``` text
1~4 张图片
```

关闭大部分随机增强。

反复训练它们。

预期：

``` text
loss 显著下降
预测逐渐贴近 GT
```

如果连极小数据都无法 overfit：

``` text
优先怀疑 pipeline / label / loss / optimizer
```

而不是数据规模。

------------------------------------------------------------------------

# 第八十部分：训练前 10 项检查

``` text
1. 最终 merged config 正确
2. num_classes 正确
3. image path / annotation path 正确
4. class remap 正确
5. boxes 在 transform 后合法
6. local batch size 正确
7. optimizer groups 正确
8. DINOv3 weights 成功加载
9. 一个 batch forward 无 NaN
10. 一个 batch backward 有合理梯度
```

------------------------------------------------------------------------

# 第八十一部分：训练中 10 项监控

``` text
1. total loss
2. loss_mal
3. loss_bbox
4. loss_giou
5. loss_fgl
6. loss_ddf
7. LR per param group
8. grad norm
9. AMP scale
10. validation AP
```

另外建议：

``` text
data time
iter time
GPU memory
GT count / image
```

------------------------------------------------------------------------

# 第八十二部分：为什么 GT count 也值得监控？

Dense O2O v2 的训练思想与：

``` text
每张训练图的有效目标密度
```

高度相关。

建议记录：

``` text
原始 GT count
Mosaic 后 GT count
MixUp 后 GT count
CopyBlend 后 GT count
Sanitize 后 GT count
```

这样你才能真正量化第 17 课讲的：

``` text
denser supervision
```

而不是只凭感觉。

------------------------------------------------------------------------

# 第八十三部分：一个 GT Density Probe

``` python
def gt_stats(
    targets
):
    counts = [
        len(t["labels"])
        for t in targets
    ]

    print(
        "GT/image:",
        counts
    )

    print(
        "mean:",
        sum(counts)
        / len(counts)
    )

    print(
        "total:",
        sum(counts)
    )
```

分别插在：

``` text
dataset transform 后
collate 后
```

观察 augmentation 如何改变 GT density。

------------------------------------------------------------------------

# 第八十四部分：一个 Matcher Probe

在 matcher 中临时记录：

``` text
num queries
num GT
cost_class range
cost_bbox range
cost_giou range
matched query indices
matched GT indices
```

例如：

``` python
print(
    "queries:",
    out_bbox.shape[0]
)

print(
    "gt:",
    tgt_bbox.shape[0]
)

print(
    "cost:",
    C.shape
)
```

这样第 6、15、16 课会真正连接到 runtime。

------------------------------------------------------------------------

# 第八十五部分：一个 Loss-to-Gradient 实验

只保留一个 loss：

``` python
loss = loss_dict[
    "loss_bbox"
]
```

backward 后观察：

``` text
backbone
STA
encoder
decoder bbox head
classification head
```

哪些模块有梯度。

再换：

``` text
loss_mal
```

比较 gradient path。

这能让你理解：

> loss 不只是一个数字，它定义了哪些参数被怎样监督。

------------------------------------------------------------------------

# 第八十六部分：冻结 DINOv3 实验

把 DINOv3：

``` text
requires_grad=False
```

然后检查：

``` text
optimizer group
gradient
GPU memory
training speed
AP
```

你会直接理解：

``` text
fine-tuning foundation backbone
```

到底给训练系统增加了什么成本。

------------------------------------------------------------------------

# 第八十七部分：关闭 AMP 实验

比较：

``` text
AMP on
AMP off
```

记录：

``` text
GPU memory
iter/s
loss trajectory
first 100 iterations
```

不要只看"能不能跑"。

------------------------------------------------------------------------

# 第八十八部分：关闭 EMA 实验

比较：

``` text
raw model AP
EMA model AP
```

同时保存：

``` text
相同 epoch
```

的两套权重进行评估。

这样 EMA 的价值会变成可观测实验。

------------------------------------------------------------------------

# 第八十九部分：关闭 Dense O2O Augmentation 实验

依次关闭：

``` text
Mosaic
MixUp
CopyBlend
```

记录：

``` text
mean GT/image
positive matches/image
loss
AP
convergence speed
```

这会成为理解 DEIMv2 训练贡献最有价值的 ablation。

------------------------------------------------------------------------

# 第九十部分：训练系统中的 Shape 总表

  位置                       典型结构
  -------------------------- -----------------------------------------
  Dataset single image       `[3,H,W]` after tensor conversion
  Single target boxes        `[Ni,4]`
  Single target labels       `[Ni]`
  Collated images            `[B,3,S,S]`
  Targets                    `list[dict]`, length `B`
  Backbone features          3 multi-scale tensors
  Decoder matching queries   `300`
  Pred logits                `[B,300,80]` main branch
  Pred boxes                 `[B,300,4]` main branch
  Matcher cost per image     `[300,Ni]` conceptually
  Loss                       scalar tensors
  Parameter gradient         same shape as parameter
  COCO final boxes           variable/top-k detection representation

训练态因为 DN / auxiliary branches 会额外增加中间 tensor。

------------------------------------------------------------------------

# 第九十一部分：一个重要的性能公式

如果：

``` text
total_batch_size = 32
world_size = 4
```

则：

$$
B_{rank}=8
$$

如果每个 rank 每秒处理：

``` text
r images/s
```

理想全局 throughput 约：

$$
4r
$$

但真实情况受：

``` text
all-reduce
data loading
load imbalance
evaluation
```

影响。

------------------------------------------------------------------------

# 第九十二部分：Global Step

``` python
global_step =
    epoch * len(data_loader)
    + i
```

例如：

``` text
len(loader)=1000
epoch=3
i=200
```

则：

$$
global_step = 3\times1000+200 = 3200
$$

TensorBoard、scheduler、warmup 常用这个统一时间坐标。

------------------------------------------------------------------------

# 第九十三部分：Epoch 与 Iteration 不要混

如果：

``` text
72 epochs
1000 iters/epoch
```

则：

``` text
72,000 iterations
```

`warmup_iter=2000`：

``` text
不是 2000 epochs
```

而约：

``` text
2 epochs
```

------------------------------------------------------------------------

# 第九十四部分：为什么 LR Recipe 必须知道 `len(data_loader)`？

FlatCosine scheduler 构造时：

``` python
iter_per_epoch = len(
    self.train_dataloader
)
```

所以：

``` text
dataset size
batch size
world size
drop_last
```

都会影响：

``` text
iterations per epoch
```

从而影响 iteration-based schedule。

------------------------------------------------------------------------

# 第九十五部分：修改 Batch Size 为什么不能只改一个数字？

因为它会改变：

``` text
iterations / epoch
optimization noise
effective samples / update
LR scaling
EMA dynamics
warmup relative duration
```

官方 README 也专门给出 batch-size scaling recipe。

所以：

``` text
batch size
```

是训练 recipe 的核心变量。

------------------------------------------------------------------------

# 第九十六部分：为什么 DDP 训练还要 `drop_last=True`？

训练配置：

``` text
drop_last=True
```

有助于：

``` text
每个训练 iteration batch size 稳定
各 rank 数据形状/数量一致
BN / optimization statistics 稳定
```

验证：

``` text
drop_last=False
```

因为不能漏掉验证图片。

------------------------------------------------------------------------

# 第九十七部分：为什么 Validation batch 可以更大？

base：

``` text
train total batch = 32
val total batch = 64
```

训练需要保存：

``` text
activations
autograd graph
gradients
optimizer states
```

验证使用：

``` text
no_grad()
```

显存需求低，因此常可使用更大 batch。

------------------------------------------------------------------------

# 第九十八部分：`evaluate()` 为什么有 `@torch.no_grad()`？

因为不需要：

``` text
backward
gradient graph
```

所以：

``` text
减少显存
减少开销
```

同时避免误建训练 graph。

------------------------------------------------------------------------

# 第九十九部分：为什么 Criterion 在 evaluate 中几乎没参与主 AP？

当前 evaluate 主线：

``` text
model
→ postprocessor
→ coco evaluator
```

criterion 被设置：

``` python
criterion.eval()
```

但当前这段 evaluation loop 并没有用它计算 validation loss。

所以当前日志重点是：

``` text
COCO metrics
```

而不是 val loss。

------------------------------------------------------------------------

# 第一百部分：训练系统真正的"状态"

一个完整训练实验的 state 不只是模型权重。

至少包括：

``` text
model parameters
optimizer moments
LR schedule position
EMA parameters
AMP scale
last epoch
augmentation epoch state
sampler epoch
random states（部分场景）
```

这就是为什么 reproducibility 比"seed=0"复杂得多。

------------------------------------------------------------------------

# 第一百零一部分：Seed 能保证完全一致吗？

不一定。

还受：

``` text
CUDA kernels
distributed order
DataLoader workers
random augmentation
library versions
hardware
non-deterministic ops
```

影响。

seed 是 reproducibility 的必要组成，但不是绝对保证。

------------------------------------------------------------------------

# 第一百零二部分：你现在应该怎样读 `train.py`？

不要从第一行开始逐字背。

按调用图：

``` text
train.py
↓
YAMLConfig
↓
TASKS
↓
DetSolver.fit
↓
BaseSolver.train
↓
train_one_epoch
↓
evaluate
```

先抓主链，再进入：

``` text
Dataset
Collate
Criterion
Matcher
```

效率高很多。

------------------------------------------------------------------------

# 第一百零三部分：本课源码阅读清单

建议按以下顺序打开：

``` text
train.py

engine/core/yaml_config.py
engine/core/workspace.py
engine/core/yaml_utils.py

engine/solver/_solver.py
engine/solver/det_solver.py
engine/solver/det_engine.py

engine/data/dataset/coco_dataset.py
engine/data/dataloader.py
engine/data/transforms/...

engine/deim/deim.py
engine/deim/deim_decoder.py
engine/deim/deim_criterion.py / criterion-related source
engine/deim/matcher.py / matcher-related source
engine/deim/postprocessor.py

engine/optim/...
engine/misc/dist_utils.py
engine/data/coco_eval.py / evaluator-related source
```

具体 criterion / matcher 文件名以当前仓库目录为准。

------------------------------------------------------------------------

# 第一百零四部分：本课必须能口述的训练主链

学完后，你应该能不看讲义说：

``` text
torchrun 会为每张 GPU 启动一个训练进程。
train.py 首先初始化 distributed 环境，然后加载并合并 YAML 与命令行参数，
根据 task=detection 创建 DetSolver。

DetSolver 通过 BaseSolver._setup 延迟实例化 model、criterion、postprocessor、
EMA 和 GradScaler，并把模型放到对应 GPU 后用 DDP 包装。
YAMLConfig 还会根据 regex parameter groups 创建 AdamW，
使 DINOv3 使用更小学习率，而 detector 部分使用更大的默认学习率。

训练 DataLoader 从 COCO JSON 读取图片和 annotations。
Dataset 先把 COCO xywh box 转成合法 xyxy，
再执行 Mosaic、crop、flip、resize、normalize 等单样本增强，
最终把 boxes 转成 normalized cxcywh。

Collate 把单张图片组成 batch，并根据 epoch schedule 执行 MixUp、
CopyBlend 和 multi-scale batch resize。
因此真正送入模型的 samples 是 [B,3,S,S]，
而 targets 是长度为 B 的 list，因为每张图片的 GT 数量不同。

train_one_epoch 把 samples 和 targets 搬到当前 rank GPU。
AMP 开启时，model forward 在 autocast 中执行，
而 criterion 在 autocast disabled 的上下文中计算。
训练态 decoder 会利用 targets 生成 denoising queries，
同时输出主预测、auxiliary predictions、FDR 信息和 DN outputs。

DEIMCriterion 使用 HungarianMatcher 为 matching queries 与 GT 建立 O2O assignment，
然后计算 MAL、L1 bbox、GIoU、FGL、DDF 以及 auxiliary / denoising losses。
训练 loop 把 loss_dict 的各项求和，执行 backward。

DDP 会在 backward 过程中同步各 rank 的 gradients。
AMP 路径先 scale loss，backward 后 unscale，
再做 gradient clipping，最后由 AdamW 更新参数。
随后 EMA 更新其平滑模型副本，LR scheduler 更新学习率。

一个 epoch 结束后保存训练状态 checkpoint，
然后优先使用 EMA model 在 val set 上推理。
PostProcessor 把 normalized predictions 转成原图像素检测结果，
CocoEvaluator 汇总所有 rank 的预测并计算 COCO AP。

DEIMv2 还利用 stop_epoch 把训练分成阶段：
强 augmentation / multi-scale 阶段结束时会加载 best_stg1，
刷新 EMA，并继续后期 fine-tuning，最终保存 stage-aware best checkpoints。
```

如果你能真正解释这段话，那么你已经不只是"会运行 DEIMv2"，而是开始具备：

``` text
训练框架级源码阅读能力
```

------------------------------------------------------------------------

# 第一百零五部分：检查题

1.  `torchrun --nproc_per_node=4` 启动几个训练进程？
2.  `train.py` 自己是否负责 backward？
3.  YAML 为什么不是只读 X 那一个文件？
4.  `YAMLConfig` 的 lazy construction 是什么？
5.  `total_batch_size=32`、world size=4 时 local batch 是多少？
6.  Dataset target 为什么是 list，而不是 `[B,N,4]`？
7.  COCO 原始 bbox 是什么格式？
8.  Dataset 中什么时候转成 normalized cxcywh？
9.  MixUp 在 dataset transform 还是 collate？
10. CopyBlend 在哪里执行？
11. multi-scale resize 在哪里执行？
12. 为什么 multi-scale resize 后 normalized box 可以不重新缩放？
13. epoch 怎样传给 augmentation？
14. DDP sampler 为什么也要 `set_epoch()`？
15. model forward 为什么训练时要传 targets？
16. DN query 与 matching query 有什么区别？
17. Matcher cost 与 loss 有什么区别？
18. Criterion 返回 scalar 还是 dict？
19. AMP 为什么要 scale loss？
20. clip gradient 前为什么要 unscale？
21. DDP gradient synchronization 发生在什么阶段？
22. EMA 在 optimizer step 前还是后更新？
23. 当前 self-defined LR scheduler 是按 iteration 还是 epoch 更新？
24. resume 与 tuning 的区别？
25. checkpoint 为什么不应该只有 model weights？
26. validation 默认评估 raw model 还是 EMA model？
27. `stop_epoch` 除了停止 multiscale 还参与什么？
28. `best_stg1` 与 `best_stg2` 表示什么？
29. COCO AP 的主 AP 是否只看 IoU=0.5？
30. AP=0 时为什么应该先检查 class/box pipeline？

------------------------------------------------------------------------

# 第一百零六部分：检查题答案

``` text
1. 4 个。
2. 不负责；主要交给 solver / train_one_epoch。
3. 因为通过 __include__ 合并 dataset、runtime、optimizer、base 等配置。
4. 第一次访问 cfg.model 等 property 时才真正 create 对象。
5. 8。
6. 每张图 GT 数量不同。
7. xywh。
8. per-image transform pipeline 的 ConvertBoxes。
9. collate。
10. BatchImageCollateFunction。
11. collate 的末尾。
12. boxes 已归一化为相对坐标。
13. train_dataloader.set_epoch 会传给 dataset 与 collate_fn。
14. 让每 epoch 的 distributed shuffle/partition 正确变化。
15. 用于构造 denoising queries。
16. DN 从 GT 加噪构造；matching queries 通过 Hungarian 与 GT 匹配。
17. cost 决定 assignment；loss 决定反向传播。
18. loss_dict。
19. 降低低精度小梯度 underflow 风险。
20. 否则 clip 的是被 scale 的梯度。
21. backward 过程中由 DDP hooks 自动完成。
22. optimizer step 后。
23. iteration。
24. resume 恢复完整训练状态；tuning 主要加载预训练模型参数开始新任务。
25. 还需要 optimizer、scheduler、EMA、scaler、epoch 等。
26. EMA model（若启用 EMA）。
27. stage boundary、reload best_stg1、EMA refresh 等。
28. 两个训练阶段各自的最佳 checkpoint。
29. 不是；主 AP 对多个 IoU 阈值平均。
30. 类别映射、坐标格式、归一化、orig_size、annotation 等错误更常见。
```

------------------------------------------------------------------------

# 第一百零七部分：课后实验

## 实验 1：单 Batch 全链 Debug

要求：

``` text
num_workers=0
single GPU
AMP off
```

断点走完：

``` text
Dataset
→ Transform
→ Collate
→ Model
→ Criterion
→ Backward
→ Optimizer
```

记录每一步的：

``` text
type
shape
dtype
device
min/max
```

------------------------------------------------------------------------

## 实验 2：GT Density Timeline

分别统计：

``` text
raw COCO
after Mosaic
after MixUp
after CopyBlend
after sanitize
```

的：

``` text
mean objects / image
median
P90
max
```

把第 17 课 Dense O2O v2 变成真实数据。

------------------------------------------------------------------------

## 实验 3：Optimizer Group Audit

输出 Markdown：

``` text
group id
lr
weight decay
parameter count
example parameter names
```

确认：

``` text
DINOv3 LR
STA LR
encoder LR
decoder LR
norm/bias weight decay
```

都符合预期。

------------------------------------------------------------------------

## 实验 4：Train vs Eval Outputs

同一个 batch：

``` python
model.train()
model.eval()
```

分别打印：

``` text
outputs.keys()
query count
aux outputs
DN outputs
```

解释差异。

------------------------------------------------------------------------

## 实验 5：Loss Audit

记录前 100 iterations：

``` text
loss_mal
loss_bbox
loss_giou
loss_fgl
loss_ddf
total
```

画曲线并检查异常值。

------------------------------------------------------------------------

## 实验 6：Gradient Audit

每 20 iterations 记录：

``` text
global grad norm
DINOv3 grad norm
STA grad norm
encoder grad norm
decoder grad norm
```

检查 gradient clipping 前后的差异。

------------------------------------------------------------------------

## 实验 7：AMP Audit

比较 AMP on/off：

``` text
peak GPU memory
iterations/s
loss
grad norm
scaler scale
```

------------------------------------------------------------------------

## 实验 8：EMA Audit

同一 epoch：

``` text
raw model
EMA model
```

分别跑 COCO validation，比较 AP。

------------------------------------------------------------------------

## 实验 9：Resume Test

训练若干 iterations / epochs：

``` text
save checkpoint
→ terminate
→ resume
```

确认：

``` text
last_epoch
LR
optimizer state
EMA
AMP scaler
```

是否连续。

------------------------------------------------------------------------

## 实验 10：Tuning Test

用 COCO pretrained：

``` text
换自定义类别数
```

观察：

``` text
哪些参数成功加载
哪些分类头 mismatch
哪些 head 被重新初始化/映射
```

------------------------------------------------------------------------

# 第一百零八部分：推荐制作 `TrainingTrace`

你可以在仓库外写一个调试器，自动生成：

``` text
epoch
step
input shape
GT count
loss components
LR groups
grad norm
AMP scale
GPU memory
iteration time
data time
```

保存为：

``` text
training_trace.csv
```

这会比只看 terminal 日志强很多。

------------------------------------------------------------------------

# 第一百零九部分：推荐的最小实验顺序

以后改任何 DEIMv2 代码，都按：

``` text
1. import 成功
2. config create 成功
3. dataset 单样本成功
4. collate 一个 batch 成功
5. forward 成功
6. criterion 成功
7. backward 成功
8. optimizer step 成功
9. 10 iterations 无 NaN
10. tiny dataset 可 overfit
11. single-GPU 正式训练
12. DDP
```

这样定位问题非常快。

------------------------------------------------------------------------

# 第一百一十部分：本课与前 22 课怎样合流？

``` text
第 1~7 课
→ 检测 / DETR 基础

第 8~14 课
→ deim_dfine_x architecture

第 15~18 课
→ DEIM / DEIMv2 training optimization

第 19~22 课
→ DINOv3 + STA + DEIMTransformer + tensor trace

第 23 课
→ 把所有东西放进真正的训练生命周期
```

你现在应该开始看到：

> DEIMv2 不是"一个模型类"，而是 **模型结构 + matching/loss +
> augmentation + optimizer recipe + distributed runtime +
> evaluation/checkpoint policy** 的完整系统。

------------------------------------------------------------------------

# 第一百一十一部分：下一课预告

第 24 课：

# 《推理、评估、部署、改模型与最终毕业项目》

将覆盖：

``` text
PyTorch inference
PostProcessor
COCO AP
FLOPs / MACs / Params
latency / FPS
ONNX export
TensorRT
FP16
部署验证
自定义数据集
模型改造实验
完整毕业项目
```

第 23 课回答：

> "模型是怎样被训练出来的？"

第 24 课将回答：

> "训练好的模型怎样被正确评估、部署，并且怎样安全地修改它？"

------------------------------------------------------------------------

# 本课源码核对清单

本课主要对应官方仓库：

``` text
train.py

configs/
  runtime.yml
  dataset/coco_detection.yml
  base/dataloader.yml
  base/optimizer.yml
  base/deimv2.yml
  deimv2/deimv2_dinov3_x_coco.yml

engine/core/
  yaml_config.py
  workspace.py
  yaml_utils.py

engine/solver/
  _solver.py
  det_solver.py
  det_engine.py

engine/data/
  dataloader.py
  dataset/coco_dataset.py
  transforms/...

engine/deim/
  model / decoder / criterion / matcher / postprocessor related source

engine/optim/
engine/misc/
```

------------------------------------------------------------------------

# 版本阅读原则

继续沿用第 22 课的优先级：

``` text
runtime behavior
>
你本地 commit 的源码
>
最终 merged config
>
checkpoint 对应配置
>
README / paper
>
课程讲义
```

尤其训练系统比模型结构更容易因 commit 更新而变化。

------------------------------------------------------------------------

# 本课一句话总结

``` text
DEIMv2 的一个训练 batch 并不是“图片进模型、loss.backward”这么简单：

COCO sample 先经过 epoch-aware 的单图增强，
collate 再执行 MixUp / CopyBlend / multi-scale，
DDP 为每个 rank 提供 local batch；
模型在训练态生成普通 queries 与 DN queries，
criterion 通过 Hungarian O2O matching、MAL、box/GIoU 与 FDR/localization supervision
形成多项 loss；
AMP/DDP/gradient clipping/AdamW 完成稳定的分布式参数更新，
EMA 与 flat-cosine scheduler维护训练时间状态；
epoch 结束后又通过 stage-aware checkpoint、EMA evaluation、
PostProcessor 和 COCO evaluator 决定 best_stg1 / best_stg2。

真正的 DEIMv2，是这一整条训练系统。
```
