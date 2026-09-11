#!/usr/bin/env bash
set -euo pipefail

# 在 deimv2_dev 容器内运行本脚本。
cd /workspace/DEIMv2

# ============================================================================
#
# 默认训练命令：
# RESUME=/workspace/DEIMv2/outputs/deim_hgnetv2_x_charging_gun_nc4/last.pth \
# NPROC_PER_NODE=8 TRAIN_BATCH_SIZE=8 NUM_CLASSES=4 MODEL=x \
# OUTPUT_DIR=outputs/deim_hgnetv2_x_charging_gun_nc4 \
# ./train_deim_dfine.sh
#
# 脚本从标注 json 自动读出训练图数,算出 iters_per_epoch,按比例自动生成 warmup_iter/ema.warmups,
# 并自动防呆(EMA 关时禁止训练 epoches 超过配置的 stop_epoch,避免撞上 self.ema.decay 空指针)。
# 换数据时你只改 DATA_ROOT 和 batch。
#
# 用户开关 —— 换数据集时只需要动这些:
#
#   DATA_ROOT                             -> 数据集根目录(见下方)
#   TRAIN_BATCH_SIZE / NPROC_PER_NODE     -> 需保持整除关系
#   EPOCHS                                -> 留空(默认)用 yml 各配置自带的 epoches
#                                            (x/l=58、m=102、s=132,调度窗口与之一致);
#                                            仅需临时覆盖时再显式设置,如 EPOCHS=80。
#   USE_EMA=1(默认)                        -> 完整两段式训练(stage1 带增强 ->
#                                            stage2 去增强 + EMA 刷新,输出 best_stg2.pth)
#   USE_EMA=0                             -> 只跑 stage1;此时训练 epoches 必须 <= 各配置的
#                                            stop_epoch(x/l=50、m=90、s=120),否则 det_solver
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
NUM_CLASSES="${NUM_CLASSES:-7}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"   # 所有 rank 的总 batch
VAL_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"     # 所有 rank 的总 batch
NUM_WORKERS="${NUM_WORKERS:-4}"
# 留空则使用 yml 配置自带的 epoches(x/l=58、m=102、s=132);显式设置时覆盖
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

DATA_ROOT="${DATA_ROOT:-/workspace/dataset/充电枪落地/coco20260910_nc7}"
TRAIN_IMAGES="${TRAIN_IMAGES:-${DATA_ROOT}/train2017}"
TRAIN_ANN="${TRAIN_ANN:-${DATA_ROOT}/annotations/train_annotation.json}"
VAL_IMAGES="${VAL_IMAGES:-${DATA_ROOT}/val2017}"
VAL_ANN="${VAL_ANN:-${DATA_ROOT}/annotations/test_annotation.json}"

CONFIG="configs/deim_dfine/deim_hgnetv2_${MODEL}_coco.yml"
PRETRAIN="ckpts/deim_dfine_hgnetv2_${MODEL}_coco.pth"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/deim_hgnetv2_${MODEL}_charging_gun_nc4}"

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

# 读取 yml 合并配置自带的 epoches 与 collate 的 stop_epoch(x/l=50、m=90、s=120)
read -r CFG_EPOCHS CFG_STOP < <(python3 - "${CONFIG}" <<'PY'
import sys
from engine.core import YAMLConfig
y = YAMLConfig(sys.argv[1]).yaml_cfg
# 必须单行输出(read 只消费第一行)
print(y.get("epoches", 0), y.get("train_dataloader", {}).get("collate_fn", {}).get("stop_epoch", 0))
PY
)
# 有效训练轮数:显式 EPOCHS 优先,否则用 yml 自带值
EPOCHS_EFF="${EPOCHS:-${CFG_EPOCHS}}"
if [[ "${USE_EMA}" == "0" ]] && (( EPOCHS_EFF > CFG_STOP )); then
    echo "USE_EMA=0 runs stage-1 only: epoches=${EPOCHS_EFF} exceeds stop_epoch=${CFG_STOP} of ${CONFIG}" >&2
    echo "-> 用 USE_EMA=1(完整两段式),或把 epoches 压到 stop_epoch 以内" >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# 按真实数据集规模推导按迭代数计的调度参数
# ----------------------------------------------------------------------------
N_TRAIN=$(python3 -c 'import json,sys;print(len(json.load(open(sys.argv[1]))["images"]))' "${TRAIN_ANN}")
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
echo "[data]   train images: ${N_TRAIN} -> ${ITERS_PER_EPOCH} iters/epoch (total_batch=${TRAIN_BATCH_SIZE})"
echo "[sched]  epoches=${EPOCHS_EFF} (yml=${CFG_EPOCHS}${EPOCHS:+ / EPOCHS env override}) stop_epoch=${CFG_STOP} total_iters=${TOTAL_ITERS} warmup_iter=${WARMUP_ITER} (quadratic warmup ends at epoch $((WARMUP_ITER / ITERS_PER_EPOCH + 1)))"
echo "[ema]    use_ema=${USE_EMA} ema_warmups=${EMA_WARMUPS}"
if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(ls -A "${OUTPUT_DIR}")" ]]; then
    echo "[warn]   ${OUTPUT_DIR} exists (log.txt is APPENDED, stale best/checkpoint pths may be overwritten)"
    echo "         -> use OUTPUT_DIR=outputs/<new-name> ... for a clean run"
fi
echo "--------------------------------------------------------------"

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
    "num_classes=${NUM_CLASSES}"
    "remap_mscoco_category=False"
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

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    torchrun --nproc_per_node="${NPROC_PER_NODE}" train.py "${ARGS[@]}"
else
    python3 train.py "${ARGS[@]}"
fi
