#!/usr/bin/env bash
# 一次跑通 L / X 两个模型,把结果图和检测框文本写到 z_trt_build/ 下。
#
# 用法(在容器内、仓库根目录执行):
#   bash tools/trt_cpp/run_both.sh
#   IMG=your.jpg THRH=0.3 bash tools/trt_cpp/run_both.sh
#   FP16=1 REBUILD=1 bash tools/trt_cpp/run_both.sh
set -euo pipefail

cd "$(dirname "$0")/../.."          # 仓库根目录

BIN=./tools/trt_cpp/build/deimv2_trt_infer
OUT=z_trt_build
IMG=${IMG:-example.jpg}
THRH=${THRH:-0.25}
NMS_IOU=${NMS_IOU:-0.9}

EXTRA=()
[[ "${FP16:-0}" == "1" ]] && EXTRA+=(--fp16)
[[ "${REBUILD:-0}" == "1" ]] && EXTRA+=(--rebuild)
[[ "${NO_NORMALIZE:-0}" == "1" ]] && EXTRA+=(--no-normalize)
[[ -n "${BENCH:-}" ]] && EXTRA+=(--benchmark "$BENCH")

if [[ ! -x "$BIN" ]]; then
    echo "找不到 $BIN,请先构建:"
    echo "  cmake -S tools/trt_cpp -B tools/trt_cpp/build -DCMAKE_BUILD_TYPE=Release"
    echo "  cmake --build tools/trt_cpp/build -j\$(nproc)"
    exit 1
fi

mkdir -p "$OUT"

# 规模:ONNX 文件名
declare -A MODELS=(
    [l]=z_onnx/PersonCarAnimal_od-v5-2-l-d4-512-896_20260920.onnx
    [x]=z_onnx/PersonCarAnimal_od-v5-2-x-d4-416-768_20260918.onnx
)

for m in l x; do
    onnx=${MODELS[$m]}
    echo "==================== ${m^^} 规模: $onnx ===================="
    "$BIN" \
        --onnx "$onnx" \
        --image "$IMG" \
        --thrh "$THRH" \
        --nms-iou "$NMS_IOU" \
        --output "$OUT/result_${m}.jpg" \
        --save-txt "$OUT/result_${m}.txt" \
        --dump-input "$OUT/cpp_in_${m}.bin" \
        --dump-output "$OUT/cpp_out_${m}.bin" \
        "${EXTRA[@]}"
    echo
done

echo "结果图: $OUT/result_l.jpg, $OUT/result_x.jpg"
