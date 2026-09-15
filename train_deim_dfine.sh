#!/usr/bin/env bash
set -euo pipefail

# 在 deimv2_dev 容器内运行本脚本。
cd /workspace/DEIMv2

# ============================================================================
#
# 默认训练命令：
# RESUME=/workspace/DEIMv2/outputs/deim_hgnetv2_x_charging_gun_nc4/last.pth \
# NPROC_PER_NODE=8 TRAIN_BATCH_SIZE=8 MODEL=x \
# OUTPUT_DIR=outputs/deim_hgnetv2_x_charging_gun_nc4 \
# ./train_deim_dfine.sh
#
# 脚本从标注 json 自动读出训练图数,算出 iters_per_epoch,按比例自动生成 warmup_iter/ema.warmups。
# 换数据时你只改 DATA_ROOT 和 batch。
#
# 用户开关 —— 换数据集时只需要动这些:
#
#   DATA_ROOT                             -> 数据集根目录(见下方)
#   TRAIN_BATCH_SIZE / NPROC_PER_NODE     -> 需保持整除关系
#   NUM_CLASSES                           -> 无需设置:自动从 TRAIN_ANN 的 categories
#                                            读取(= max(category_id)+1,要求 id 从 0 连续)
#   EPOCHS                                -> 留空(默认)用 yml 各配置自带的 epoches
#                                            (x/l=58、m=102、s=132);
#                                            显式设置时自动"单变量联动":
#                                            以 yml 设计为基准等比推导并注入
#                                              stop_epoch / flat_epoch / policy.epoch / mixup_epochs
#                                            公式(以有效训练段 T=N-no_aug 为基准):
#                                              stop=T;flat=4+T//2;policy=[4,flat,T];
#                                              mixup=[4,flat];copyblend 终点=T;
#                                              matcher=yml_match×T/yml_stop(保留各模型原比例)
#   VAL_BATCH_SIZE                        -> 验证集总 batch(默认与 TRAIN_BATCH_SIZE 相同)
#   MASTER_PORT                           -> torchrun 端口(默认 29500;并发跑多个实验时分别指定)
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
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${TRAIN_BATCH_SIZE:-8}}"   # 所有 rank 的总 batch
NUM_WORKERS="${NUM_WORKERS:-6}"
MASTER_PORT="${MASTER_PORT:-}"              # torchrun 端口(留空用默认 29500)
# 留空则使用 yml 配置自带的 epoches(x/l=58、m=102、s=132);显式设置时覆盖
EPOCHS="${EPOCHS:-}"
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

if [[ -n "${RESUME}" && ! -e "${RESUME}" ]]; then
    echo "Missing RESUME: ${RESUME}" >&2
    exit 1
fi

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

# 读取:yml 自带的 epoches/stop_epoch(x/l=50、m=90、s=120)/no_aug/flat(及可选的
#       copyblend/matcher 窗口)、训练图数、类别数(自动从 TRAIN_ANN 读取)、
#       以及 RESUME ckpt 的类别数(用于一致性校验)
if ! read -r CFG_EPOCHS CFG_STOP N_TRAIN N_CLASSES CFG_NOAUG CFG_FLAT CFG_CB CFG_MATCH RESUME_NC \
    < <(python3 - "${CONFIG}" "${TRAIN_ANN}" "${RESUME}" <<'PY'
import json
import os
import re
import sys
import warnings

import torch
from engine.core import YAMLConfig

warnings.filterwarnings("ignore")

y = YAMLConfig(sys.argv[1]).yaml_cfg
cfg_epochs = y.get("epoches", 0)
cc = y.get("train_dataloader", {}).get("collate_fn", {})
crit = y.get("DEIMCriterion", {}).get("matcher", {})
cfg_stop = cc.get("stop_epoch", 0)
cfg_noaug = y.get("no_aug_epoch")
if cfg_noaug is None:                       # 仅在缺失/None 时 fallback,
    cfg_noaug = 8                           # 显式写的 no_aug_epoch: 0 予以保留
cfg_flat = y.get("flat_epoch")
if cfg_flat is None:
    cfg_flat = max(1, cfg_epochs // 2)
cb = cc.get("copyblend_epochs")
cfg_cb = int(cb[-1]) if isinstance(cb, (list, tuple)) and len(cb) else -1      # -1 = 未启用
cfg_match = -1
if crit.get("change_matcher"):
    v = crit.get("matcher_change_epoch")        # 仅在缺失/None 时视为未启用,
    cfg_match = -1 if v is None else int(v)     # 显式写的 0 予以保留

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

# RESUME ckpt 的类别数(取 decoder 分类头行数)
resume_path = sys.argv[3]
resume_nc = -1
if resume_path:
    if not os.path.exists(resume_path):
        print(f"[error] RESUME 不存在: {resume_path}", file=sys.stderr)
        sys.exit(1)
    ck = torch.load(resume_path, map_location="cpu")
    if isinstance(ck, dict) and isinstance(ck.get("ema"), dict):
        st = ck["ema"].get("module", ck["ema"])
    elif isinstance(ck, dict) and "model" in ck:
        st = ck["model"]
    else:
        st = ck
    for k, v in st.items():
        k2 = k[len("module."):] if k.startswith("module.") else k
        if re.search(r"decoder\.dec_score_head\.\d+\.weight$", k2) and getattr(v, "dim", lambda: 0)() == 2:
            resume_nc = int(v.shape[0])
            break

# 必须单行输出(read 只消费第一行)
print(cfg_epochs, cfg_stop, n_train, n_classes, cfg_noaug, cfg_flat, cfg_cb, cfg_match, resume_nc)
PY
); then
    echo "解析配置/标注失败:请检查上方 python 报错(常见原因:标注文件损坏/缺 categories、RESUME 不可读)" >&2
    exit 1
fi

# 解析结果校验:非空且为数字(python 段异常时明确报错,而不是后续算术炸出难懂信息)
for kv in "CFG_EPOCHS=${CFG_EPOCHS}" "CFG_STOP=${CFG_STOP}" "N_TRAIN=${N_TRAIN}" \
          "N_CLASSES=${N_CLASSES}" "CFG_NOAUG=${CFG_NOAUG}" "CFG_FLAT=${CFG_FLAT}"; do
    if ! [[ "${kv#*=}" =~ ^[0-9]+$ ]]; then
        echo "解析配置/标注失败(${kv%%=*}=${kv#*=}),请检查上方 python 报错" >&2
        exit 1
    fi
done
if ! [[ "${CFG_CB}" =~ ^-?[0-9]+$ && "${CFG_MATCH}" =~ ^-?[0-9]+$ && "${RESUME_NC}" =~ ^-?[0-9]+$ ]]; then
    echo "解析失败(CFG_CB=${CFG_CB} CFG_MATCH=${CFG_MATCH} RESUME_NC=${RESUME_NC})" >&2
    exit 1
fi

# 续训 ckpt 与数据类别数一致性(不一致时 strict 加载会崩,这里提前拦截)
if (( RESUME_NC >= 0 )) && (( RESUME_NC != N_CLASSES )); then
    echo "续训 ckpt 类别数(${RESUME_NC})与标注类别数(${N_CLASSES})不一致" >&2
    echo "-> RESUME=${RESUME};检查它是否与 DATA_ROOT/TRAIN_ANN 配对" >&2
    exit 1
fi

# 有效训练轮数:显式 EPOCHS 优先,否则用 yml 自带值
EPOCHS_EFF="${EPOCHS:-${CFG_EPOCHS}}"
if (( EPOCHS_EFF <= 0 )); then
    echo "epoches 未定义:请设置 EPOCHS 或检查 ${CONFIG}" >&2
    exit 1
fi

# ---- EPOCHS 单变量联动 ----
# 官方以"有效训练段" T = epoches - no_aug_epoch 为基准定义中后段策略:
#   stop_epoch = T;flat_epoch = 4 + T//2;policy = [4, flat, T];mixup = [4, flat];
#   copyblend 终点 = T;matcher_change = 各模型原比例(CFG_MATCH/CFG_STOP) × T
LINK=0
STOP_EFF="${CFG_STOP}"
FLAT_EFF="${CFG_FLAT}"
POLICY_EPOCH=""; MIXUP_EPOCH=""; CB_EPOCH=""; MATCH_EPOCH=""
if [[ -n "${EPOCHS}" ]]; then
    LINK=1
    T_EFF=$(( EPOCHS_EFF - CFG_NOAUG ))              # 有效训练段长度(no_aug 尾段保持不变)
    if (( T_EFF <= 4 )); then
        echo "EPOCHS=${EPOCHS_EFF} 太小(需 > no_aug_epoch(${CFG_NOAUG}) + 4)" >&2
        exit 1
    fi
    STOP_EFF=$T_EFF
    FLAT_EFF=$(( 4 + T_EFF / 2 ))                    # 官方公式:flat = 4 + T//2
    if (( FLAT_EFF >= STOP_EFF )); then
        FLAT_EFF=$(( STOP_EFF - 1 ))
    fi
    POLICY_EPOCH="[4, ${FLAT_EFF}, ${STOP_EFF}]"
    MIXUP_EPOCH="[4, ${FLAT_EFF}]"
    if (( CFG_CB >= 0 )); then
        CB_EPOCH="[4, ${STOP_EFF}]"
    fi
    if (( CFG_MATCH >= 0 )); then
        if (( CFG_STOP > 0 )); then
            # 保留各模型原本的 CFG_MATCH/CFG_STOP 比例(X=90%、L≈83%、M≈89%、S≈83%),四舍五入
            MATCH_EPOCH=$(( (CFG_MATCH * T_EFF + CFG_STOP / 2) / CFG_STOP ))
        else
            MATCH_EPOCH=$(( (77 * EPOCHS_EFF + 50) / 100 ))
        fi
    fi
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
echo "[ema]    ema_warmups=${EMA_WARMUPS}"
if [[ "${LINK}" == "1" ]]; then
    echo "[link]   EPOCHS=${EPOCHS} 联动: stop_epoch=${STOP_EFF} flat_epoch=${FLAT_EFF} policy=${POLICY_EPOCH} mixup=${MIXUP_EPOCH}${CB_EPOCH:+ copyblend=${CB_EPOCH}}${MATCH_EPOCH:+ matcher_change=${MATCH_EPOCH}}"
fi
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
    "num_classes=${N_CLASSES}"
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
# EPOCHS 联动:注入推导出的 epoch 尺度窗口
if [[ "${LINK}" == "1" ]]; then
    UPDATES+=(
        "train_dataloader.collate_fn.stop_epoch=${STOP_EFF}"
        "train_dataloader.dataset.transforms.policy.epoch=${POLICY_EPOCH}"
        "flat_epoch=${FLAT_EFF}"
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
ARGS+=(
    "use_ema=True"
    "ema.warmups=${EMA_WARMUPS}"
)

if [[ "${USE_AMP}" == "1" ]]; then
    ARGS+=(--use-amp)
fi

# 干跑模式:PRINT_ONLY=1 只打印最终启动命令与解析后的关键参数,不启动训练
# (用于参数/联动校验,避免误拉起训练)
if [[ "${PRINT_ONLY:-0}" == "1" ]]; then
    echo "[dry] 仅打印,不启动训练。最终命令:"
    if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
        echo "[dry] torchrun --nproc_per_node=${NPROC_PER_NODE}${MASTER_PORT:+ --master_port=${MASTER_PORT}} train.py <ARGS>"
    else
        echo "[dry] python3 train.py <ARGS>"
    fi
    printf ' %q' "${ARGS[@]}"; echo
    exit 0
fi

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    PORT_ARGS=()
    if [[ -n "${MASTER_PORT}" ]]; then
        PORT_ARGS=(--master_port="${MASTER_PORT}")
    fi
    torchrun --nproc_per_node="${NPROC_PER_NODE}" "${PORT_ARGS[@]}" train.py "${ARGS[@]}"
else
    python3 train.py "${ARGS[@]}"
fi
