#!/usr/bin/env bash
set -euo pipefail

# 在 deimv2_dev 容器内运行本脚本。
cd /workspace/DEIMv2

# ============================================================================
#
# 默认训练命令：
# RESUME=/workspace/DEIMv2/outputs/deimv2_dinov3_x_charging_gun_nc4/last.pth \
# NPROC_PER_NODE=8 TRAIN_BATCH_SIZE=8 MODEL=x \
# OUTPUT_DIR=outputs/deimv2_dinov3_x_charging_gun_nc4 \
# ./train_deimv2_dinov3.sh
#
# 脚本从标注 json 自动读出训练图数,算出 iters_per_epoch,按比例自动生成 warmup_iter/ema.warmups,
# 并自动防呆(EMA 关时禁止训练 epoches 超过配置的 stop_epoch,避免撞上 self.ema.decay 空指针)。
# 换数据时你只改 DATA_ROOT 和 batch。
#
# 用户开关 —— 换数据集时只需要动这些:
#
#   DATA_ROOT                             -> 数据集根目录(见下方)
#   TRAIN_BATCH_SIZE / NPROC_PER_NODE     -> 需保持整除关系
#   NUM_CLASSES                           -> 无需设置:自动从 TRAIN_ANN 的 categories
#                                            读取(= max(category_id)+1,要求 id 从 0 连续)
#   EPOCHS                                -> 留空(默认)用 yml 各配置自带的 epoches
#                                            (x=58、l=68、m=102、s=132);
#                                            显式设置时自动"单变量联动":
#                                            以 yml 设计为基准等比推导并注入
#                                              stop_epoch / flat_epoch / policy.epoch / mixup_epochs /
#                                              copyblend_epochs / matcher_change_epoch
#                                            公式:stop=N-no_aug;其余按 yml值×N/yml_epoches 四舍五入
#   USE_EMA=1(默认)                        -> 完整两段式训练(stage1 带增强 ->
#                                            stage2 去增强 + EMA 刷新,输出 best_stg2.pth)
#   USE_EMA=0                             -> 只跑 stage1;此时训练 epoches 必须 <= 各配置的
#                                            stop_epoch(x=50、l=60、m=90、s=120),否则 det_solver
#                                            在 epoch==stop_epoch 会访问 self.ema 崩溃
#
# 所有按迭代数(iteration)计的调度参数都会根据训练标注文件自动推导,
# 比例与 base 配置里的 COCO 参数一致:
#   iters_per_epoch = 训练图数 / total_batch
#   warmup_iter     ~ 全程迭代数的 2%   (COCO: 2000 / ~106k 步)
#   ema warmups     ~ 1 个 epoch 的迭代数(COCO: 1000 / ~1830 步)
# ============================================================================

MODEL="${MODEL:-x}"                         # s / m / l / x
DEVICE="${DEVICE:-cuda}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"   # 所有 rank 的总 batch
VAL_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"     # 所有 rank 的总 batch
NUM_WORKERS="${NUM_WORKERS:-4}"
# 留空则使用 yml 配置自带的 epoches(x=58、l=68、m=102、s=132);显式设置时覆盖
EPOCHS="${EPOCHS:-}"
USE_EMA="${USE_EMA:-1}"
SEED="${SEED:-0}"
USE_AMP="${USE_AMP:-1}"
# 手动覆盖自动推导值(通常保持为空即可):
WARMUP_ITER="${WARMUP_ITER:-}"
EMA_WARMUPS="${EMA_WARMUPS:-}"
# 断点续训:填入 checkpoint 路径(如 outputs/.../last.pth)后从该处继续;
# 为空则从头开始。注意:续训需保持数据/参数与上次一致。
RESUME="${RESUME:-}"

DATA_ROOT="${DATA_ROOT:-/workspace/dataset/充电枪落地/coco20260910_nc4}"
TRAIN_IMAGES="${TRAIN_IMAGES:-${DATA_ROOT}/train2017}"
TRAIN_ANN="${TRAIN_ANN:-${DATA_ROOT}/annotations/train_annotation.json}"
VAL_IMAGES="${VAL_IMAGES:-${DATA_ROOT}/val2017}"
VAL_ANN="${VAL_ANN:-${DATA_ROOT}/annotations/test_annotation.json}"

CONFIG="configs/deimv2/deimv2_dinov3_${MODEL}_coco.yml"
PRETRAIN="ckpts/deimv2_dinov3_${MODEL}_coco.pth"
BACKBONE_WEIGHTS="${BACKBONE_WEIGHTS:-ckpts/dinov3_vits16plus_from_deimv2_${MODEL}.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/deimv2_dinov3_${MODEL}_charging_gun_nc4}"

case "${MODEL}" in
    s|m|l|x) ;;
    *) echo "MODEL must be s/m/l/x, got: ${MODEL}" >&2; exit 2 ;;
esac

for path in "${CONFIG}" "${PRETRAIN}" "${TRAIN_IMAGES}" "${TRAIN_ANN}" "${VAL_IMAGES}" "${VAL_ANN}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing path: ${path}" >&2
        exit 1
    fi
done

# ----------------------------------------------------------------------------
# 前置防呆检查
# ----------------------------------------------------------------------------
if [[ "${TRAIN_BATCH_SIZE}" -le 0 || $((TRAIN_BATCH_SIZE % NPROC_PER_NODE)) -ne 0 ]]; then
    echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} must be a positive multiple of NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
    exit 1
fi
if [[ "${VAL_BATCH_SIZE}" -le 0 || $((VAL_BATCH_SIZE % NPROC_PER_NODE)) -ne 0 ]]; then
    echo "VAL_BATCH_SIZE=${VAL_BATCH_SIZE} must be a positive multiple of NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
    exit 1
fi
if [[ "${USE_EMA}" != "0" && "${USE_EMA}" != "1" ]]; then
    echo "USE_EMA must be 0 or 1, got: ${USE_EMA}" >&2
    exit 1
fi

# 读取:yml 自带的 epoches 与 collate 的 stop_epoch(x=50、l=60、m=90、s=120)、
#       训练图数、类别数(自动从 TRAIN_ANN 的 categories 读取 = max(id)+1)
read -r CFG_EPOCHS CFG_STOP N_TRAIN N_CLASSES CFG_NOAUG CFG_FLAT CFG_CB CFG_MATCH < <(python3 - "${CONFIG}" "${TRAIN_ANN}" <<'PY'
import json
import sys
from engine.core import YAMLConfig

y = YAMLConfig(sys.argv[1]).yaml_cfg
cfg_epochs = y.get("epoches", 0)
cc = y.get("train_dataloader", {}).get("collate_fn", {})
crit = y.get("DEIMCriterion", {}).get("matcher", {})
cfg_stop = cc.get("stop_epoch", 0)
cfg_noaug = y.get("no_aug_epoch", 8) or 8
cfg_flat = y.get("flat_epoch", 0) or max(1, cfg_epochs // 2)
cb = cc.get("copyblend_epochs")
cfg_cb = int(cb[-1]) if isinstance(cb, (list, tuple)) and len(cb) else -1      # -1 = 未启用
cfg_match = int(crit.get("matcher_change_epoch") or -1) if crit.get("change_matcher") else -1

data = json.load(open(sys.argv[2]))
n_train = len(data["images"])
cats = data.get("categories")
if not cats:
    raise RuntimeError(f"标注文件缺少 categories 字段: {sys.argv[2]}")
ids = sorted(int(c["id"]) for c in cats)
n_classes = ids[-1] + 1
if ids != list(range(n_classes)):
    print(f"[warn] 类别 id 不连续(需从 0 连续): ids={ids};num_classes 将取 max(id)+1={n_classes}",
          file=sys.stderr)

# 必须单行输出(read 只消费第一行)
print(cfg_epochs, cfg_stop, n_train, n_classes, cfg_noaug, cfg_flat, cfg_cb, cfg_match)
PY
)
# 有效训练轮数:显式 EPOCHS 优先,否则用 yml 自带值
EPOCHS_EFF="${EPOCHS:-${CFG_EPOCHS}}"

# ---- EPOCHS 单变量联动:显式设置 EPOCHS 时,按 yml 设计等比推导各 epoch 尺度窗口 ----
LINK=0
STOP_EFF="${CFG_STOP}"
FLAT_EFF="${CFG_FLAT}"
POLICY_EPOCH=""; MIXUP_EPOCH=""; CB_EPOCH=""; MATCH_EPOCH=""
# 把 yml 值按 N/yml_epoches 等比放大,四舍五入
scale_epoch() { echo $(( ($1 * EPOCHS_EFF + CFG_EPOCHS / 2) / CFG_EPOCHS )); }
if [[ -n "${EPOCHS}" ]]; then
    LINK=1
    NOAUG_EFF="${CFG_NOAUG}"; (( NOAUG_EFF > 0 )) || NOAUG_EFF=8
    if (( EPOCHS_EFF - NOAUG_EFF <= 4 )); then
        echo "EPOCHS=${EPOCHS_EFF} 太小(需 > no_aug_epoch(${NOAUG_EFF}) + 4)" >&2
        exit 1
    fi
    STOP_EFF=$(( EPOCHS_EFF - NOAUG_EFF ))          # no_aug 尾段长度保持不变
    if (( CFG_EPOCHS > 0 )); then
        FLAT_EFF=$(scale_epoch "${CFG_FLAT}")
    fi
    (( FLAT_EFF > 4 )) || FLAT_EFF=$(( EPOCHS_EFF / 2 ))    # 保护:至少大于 warmup 起点 4
    POLICY_EPOCH="[4, ${FLAT_EFF}, ${STOP_EFF}]"
    MIXUP_EPOCH="[4, ${FLAT_EFF}]"
    if (( CFG_CB >= 0 )); then
        CB_EPOCH="[4, ${STOP_EFF}]"
    fi
    if (( CFG_MATCH >= 0 )); then
        if (( CFG_EPOCHS > 0 )); then
            MATCH_EPOCH=$(scale_epoch "${CFG_MATCH}")
        else
            MATCH_EPOCH=$(( (77 * EPOCHS_EFF + 50) / 100 ))
        fi
    fi
fi

if [[ "${USE_EMA}" == "0" ]] && (( EPOCHS_EFF > STOP_EFF )); then
    echo "USE_EMA=0 runs stage-1 only: epoches=${EPOCHS_EFF} exceeds stop_epoch=${STOP_EFF} of ${CONFIG}" >&2
    echo "-> 用 USE_EMA=1(完整两段式),或把 epoches 压到 stop_epoch 以内" >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# 按真实数据集规模推导按迭代数计的调度参数
# ----------------------------------------------------------------------------
ITERS_PER_EPOCH=$(( (N_TRAIN + TRAIN_BATCH_SIZE - 1) / TRAIN_BATCH_SIZE ))   # 向上取整
TOTAL_ITERS=$(( ITERS_PER_EPOCH * EPOCHS_EFF ))
if [[ -z "${WARMUP_ITER}" ]]; then
    WARMUP_ITER=$(( (TOTAL_ITERS * 2 + 99) / 100 ))                           # 约全程的 2%
    [[ "${WARMUP_ITER}" -lt 1 ]] && WARMUP_ITER=1
fi
if [[ -z "${EMA_WARMUPS}" ]]; then
    EMA_WARMUPS="${ITERS_PER_EPOCH}"                                          # 约 1 个 epoch
fi

echo "--------------------------------------------------------------"
echo "[data]   train images: ${N_TRAIN} -> ${ITERS_PER_EPOCH} iters/epoch (total_batch=${TRAIN_BATCH_SIZE}) | num_classes=${N_CLASSES} (auto from annotations)"
echo "[sched]  epoches=${EPOCHS_EFF} (yml=${CFG_EPOCHS}${EPOCHS:+ / EPOCHS env override}) stop_epoch=${STOP_EFF} total_iters=${TOTAL_ITERS} warmup_iter=${WARMUP_ITER} (quadratic warmup ends at epoch $((WARMUP_ITER / ITERS_PER_EPOCH + 1)))"
echo "[ema]    use_ema=${USE_EMA} ema_warmups=${EMA_WARMUPS}"
if [[ "${LINK}" == "1" ]]; then
    echo "[link]   EPOCHS=${EPOCHS} 联动: stop_epoch=${STOP_EFF} flat_epoch=${FLAT_EFF} policy=${POLICY_EPOCH} mixup=${MIXUP_EPOCH}${CB_EPOCH:+ copyblend=${CB_EPOCH}}${MATCH_EPOCH:+ matcher_change=${MATCH_EPOCH}}"
fi
if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(ls -A "${OUTPUT_DIR}")" ]]; then
    echo "[warn]   ${OUTPUT_DIR} exists (log.txt is APPENDED, stale best/checkpoint pths may be overwritten)"
    echo "         -> use OUTPUT_DIR=outputs/<new-name> ... for a clean run"
fi
echo "--------------------------------------------------------------"

# ----------------------------------------------------------------------------
# 检测器 checkpoint 里是 backbone.dinov3.* 前缀的键,而模型构造函数期望的是
# 不带该前缀的原始 DINOv3 state dict,这里做一次键名前缀剥离并单独保存。
# ----------------------------------------------------------------------------
if [[ ! -f "${BACKBONE_WEIGHTS}" ]]; then
    PRETRAIN_PATH="${PRETRAIN}" BACKBONE_PATH="${BACKBONE_WEIGHTS}" python3 - <<'PY'
import os
from pathlib import Path
import torch

pretrain_path = Path(os.environ["PRETRAIN_PATH"])
backbone_path = Path(os.environ["BACKBONE_PATH"])
state = torch.load(pretrain_path, map_location="cpu")
model_state = state.get("model", state)
prefix = "backbone.dinov3."
backbone_state = {
    key[len(prefix):]: value
    for key, value in model_state.items()
    if key.startswith(prefix)
}
if not backbone_state:
    raise RuntimeError(f"No {prefix} parameters found in {pretrain_path}")
backbone_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(backbone_state, backbone_path)
print(f"Extracted {len(backbone_state)} DINOv3 backbone tensors to {backbone_path}")
PY
fi

ARGS=(
    -c "${CONFIG}"
    -d "${DEVICE}"
    --seed "${SEED}"
    --output-dir "${OUTPUT_DIR}"
)

# -t(微调起点)与 -r(断点续训)互斥(train.py 有断言),二选一注入。
# 注意:-r/-t 必须放在 -u 之前。train.py 的 -u 是 nargs='+' 贪婪参数,
# 会一直吞掉后续不以 '-' 开头的参数;若在 -u 列表中间插入其它选项,
# 会把 use_ema=... 等 update 项截断成游离参数(unrecognized arguments)。
if [[ -n "${RESUME}" ]]; then
    ARGS+=(-r "${RESUME}")
else
    ARGS+=(-t "${PRETRAIN}")
fi

UPDATES=(
    "num_classes=${N_CLASSES}"
    "remap_mscoco_category=False"
    "DINOv3STAs.weights_path=${BACKBONE_WEIGHTS}"
    "train_dataloader.dataset.img_folder=${TRAIN_IMAGES}"
    "train_dataloader.dataset.ann_file=${TRAIN_ANN}"
    "train_dataloader.total_batch_size=${TRAIN_BATCH_SIZE}"
    "train_dataloader.num_workers=${NUM_WORKERS}"
    "val_dataloader.dataset.img_folder=${VAL_IMAGES}"
    "val_dataloader.dataset.ann_file=${VAL_ANN}"
    "val_dataloader.total_batch_size=${VAL_BATCH_SIZE}"
    "val_dataloader.num_workers=${NUM_WORKERS}"
    "warmup_iter=${WARMUP_ITER}"
)
# epoches 默认取自 yml 配置;仅显式设置 EPOCHS 时才注入覆盖
if [[ -n "${EPOCHS}" ]]; then
    UPDATES=("epoches=${EPOCHS}" "${UPDATES[@]}")
fi
# EPOCHS 联动:注入等比推导出的各 epoch 尺度窗口(仅显式设置 EPOCHS 时)
if [[ "${LINK}" == "1" ]]; then
    UPDATES+=(
        "train_dataloader.collate_fn.stop_epoch=${STOP_EFF}"
        "flat_epoch=${FLAT_EFF}"
        "train_dataloader.dataset.transforms.policy.epoch=${POLICY_EPOCH}"
        "train_dataloader.collate_fn.mixup_epochs=${MIXUP_EPOCH}"
    )
    if [[ -n "${CB_EPOCH}" ]]; then
        UPDATES+=("train_dataloader.collate_fn.copyblend_epochs=${CB_EPOCH}")
    fi
    if [[ -n "${MATCH_EPOCH}" ]]; then
        UPDATES+=("DEIMCriterion.matcher.matcher_change_epoch=${MATCH_EPOCH}")
    fi
fi
ARGS+=(-u "${UPDATES[@]}")

# use_ema/ema.warmups 必须紧跟 -u 列表之后(中间不能插入其它选项),
# 这样才会被 -u 的 nargs='+' 吞进 update 列表:
if [[ "${USE_EMA}" == "1" ]]; then
    ARGS+=(
        "use_ema=True"
        "ema.warmups=${EMA_WARMUPS}"
    )
fi

if [[ "${USE_AMP}" == "1" ]]; then
    ARGS+=(--use-amp)
fi

# 干跑模式:PRINT_ONLY=1 只打印最终启动命令与解析后的关键参数,不启动训练
# (用于参数/联动校验,避免误拉起训练)
if [[ "${PRINT_ONLY:-0}" == "1" ]]; then
    echo "[dry] 仅打印,不启动训练。最终命令:"
    printf ' %q' "${ARGS[@]}"; echo
    exit 0
fi

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    torchrun --nproc_per_node="${NPROC_PER_NODE}" train.py "${ARGS[@]}"
else
    python3 train.py "${ARGS[@]}"
fi
