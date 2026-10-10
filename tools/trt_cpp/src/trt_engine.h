// TensorRT 10.x 引擎封装:ONNX -> engine(带磁盘缓存)、单输入单输出推理。
//
// 约定:网络输入名为 images、输出名为 output,形状分别为 [B,3,H,W] 与 [B,N,4+C];
// 只有 batch 维是动态的,H/W 必须静态(构建 profile 时按 ONNX 声明值固定)。
#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace deimv2 {

// TensorRT 的日志转发到 stderr,按 severity 过滤
class TrtLogger : public nvinfer1::ILogger {
public:
    explicit TrtLogger(nvinfer1::ILogger::Severity level = nvinfer1::ILogger::Severity::kWARNING)
        : level_(level) {}

    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override;

    void setLevel(nvinfer1::ILogger::Severity level) { level_ = level; }

private:
    nvinfer1::ILogger::Severity level_;
};

// 逐张量找 NaN:靠 TensorRT 的 debug tensor 机制。
// 构建时把张量标记成 debug tensor,运行时挂上这个 listener,
// 每收到一个张量就拷回 host 检查一遍,记录第一个出现 NaN 的以及它前面几个正常的张量。
class NanFindingListener : public nvinfer1::IDebugListener {
public:
    bool processDebugTensor(void const* addr, nvinfer1::TensorLocation location,
                            nvinfer1::DataType type, nvinfer1::Dims const& shape,
                            char const* name, cudaStream_t stream) noexcept override;

    bool found = false;              // 是否已经找到第一个 NaN
    size_t checked = 0;              // 已检查的张量数
    size_t nan_count = 0;            // 含 NaN 的张量总数
    std::string first_nan;           // 第一个出现 NaN 的张量名
    std::vector<std::string> before; // 它之前最近的几个正常张量(用于定位边界)
    std::vector<float> host;         // 复用缓冲
    // 全部检查过的张量:(名字, 是否含 NaN),按回调顺序
    std::vector<std::pair<std::string, bool>> all;

private:
    void note(char const* name, bool has_nan);
};

struct EngineOptions {
    int device = 0;             // 用哪块 GPU(engine 本身与设备序号无关,只与 GPU 架构绑)
    int max_batch = 1;          // ONNX 的 batch 维是动态的,profile 取 [1, max_batch],
                                // opt 定在 1(本工具一次只跑一张图)
    bool fp16 = false;          // 默认 FP32,便于与 onnxruntime 参考结果对齐
    // 强类型网络:各层精度完全由 ONNX 里的 tensor dtype 决定,TRT 不再自行降精度。
    // 配合"混合精度 ONNX"(FP16 主体 + norm 归约段 FP32)使用,见 README。
    bool strongly_typed = false;
    // FP16 下把名字含 "norm" 的 Reduce/Pow/ElementWise 层钉回 FP32。
    // 背景:DEIM 的 decoder 用 RMSNorm(ONNX 无此算子,被拆成 Pow/ReduceMean),
    // 弱类型网络的 kFP16 会把这些层降成 FP16,而 x^2 在 FP16 里超过 256 就溢出。
    bool fp16_fp32_norm = false;
    size_t workspace_mb = 2048;
    bool verbose = false;
    // 找 NaN:markUnfusedTensorsAsDebugTensors 不会阻止算子融合,代价小;
    // debug_nan_all 用 markDebug 标记每一个张量,会阻止融合、engine 会大很多。
    bool debug_nan = false;
    bool debug_nan_all = false;
    std::string debug_nan_filter;  // 只标记名字含该子串的张量
    std::string list_tensors;      // 非空则只打印名字含该子串的张量后退出(不构建)
};

class TrtEngine {
public:
    TrtEngine() = default;
    ~TrtEngine();

    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

    // engine_path 存在且 !rebuild 时直接反序列化;否则用 onnx 构建并写回 engine_path。
    bool loadOrBuild(const std::string& onnx_path,
                     const std::string& engine_path,
                     const EngineOptions& opt,
                     bool rebuild);

    int inputH() const { return in_h_; }
    int inputW() const { return in_w_; }
    const std::string& inputName() const { return input_name_; }
    const std::string& outputName() const { return output_name_; }

    // 单次推理。input: host 端 [batch,3,H,W] float32(与预处理输出一致);
    // output: host 端 [batch,N,4+C] float32,行优先(与 numpy 的 pred 布局相同);
    // out_shape: {batch, N, 4+C}。
    bool infer(const float* input, int batch,
               std::vector<float>& output, std::vector<int64_t>& out_shape);

    // 仅 --debug-nan 时有效:上一次 infer 里逐张量找 NaN 的结果
    const NanFindingListener* nanReport() const {
        return debug_listener_ ? debug_listener_.get() : nullptr;
    }

private:
    bool build(const std::string& onnx_path, const std::string& engine_path,
               const EngineOptions& opt);
    bool deserialize(const std::string& engine_path);
    bool initFromEngine();
    bool ensureBuffer(const std::string& name, size_t bytes);
    void releaseBuffers();

    TrtLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::unique_ptr<NanFindingListener> debug_listener_;
    cudaStream_t stream_ = nullptr;

    std::string input_name_;
    std::string output_name_;
    int in_h_ = 0;
    int in_w_ = 0;

    struct Buffer {
        void* ptr = nullptr;
        size_t bytes = 0;
    };
    std::unordered_map<std::string, Buffer> buffers_;
};

}  // namespace deimv2
