#include "trt_engine.h"

#include <NvInferPlugin.h>
#include <NvOnnxParser.h>

#include <cuda_fp16.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

namespace deimv2 {
namespace {

#define TRT_CHECK(expr)                                                          \
    do {                                                                         \
        if (!(expr)) {                                                           \
            std::cerr << "[TRT] 失败: " #expr " (trt_engine.cpp:" << __LINE__     \
                      << ")" << std::endl;                                       \
            return false;                                                        \
        }                                                                        \
    } while (0)

#define CUDA_CHECK(expr)                                                         \
    do {                                                                         \
        const cudaError_t err_ = (expr);                                         \
        if (err_ != cudaSuccess) {                                               \
            std::cerr << "[CUDA] " << cudaGetErrorString(err_) << " (trt_engine.cpp:" \
                      << __LINE__ << ")" << std::endl;                           \
            return false;                                                        \
        }                                                                        \
    } while (0)

std::string dimsToStr(const nvinfer1::Dims& d) {
    std::string s = "[";
    for (int i = 0; i < d.nbDims; ++i) {
        if (i) s += ",";
        s += std::to_string(d.d[i]);
    }
    return s + "]";
}

bool hasDynamicDim(const nvinfer1::Dims& d) {
    for (int i = 0; i < d.nbDims; ++i) {
        if (d.d[i] < 0) return true;
    }
    return false;
}

size_t volume(const nvinfer1::Dims& d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) v *= static_cast<size_t>(d.d[i]);
    return v;
}

constexpr size_t kDtypeSize[] = {0, 1, 1, 2, 2, 4, 4, 8, 8, 1, 1, 1, 1};

}  // namespace

void TrtLogger::log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept {
    if (severity > level_) return;  // severity 数值越大越不严重
    switch (severity) {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
        case nvinfer1::ILogger::Severity::kERROR:
            std::cerr << "[TRT][E] " << msg << std::endl;
            break;
        case nvinfer1::ILogger::Severity::kWARNING:
            std::cerr << "[TRT][W] " << msg << std::endl;
            break;
        case nvinfer1::ILogger::Severity::kINFO:
            std::cerr << "[TRT][I] " << msg << std::endl;
            break;
        default:
            std::cerr << "[TRT][V] " << msg << std::endl;
            break;
    }
}

bool NanFindingListener::processDebugTensor(void const* addr, nvinfer1::TensorLocation location,
                                            nvinfer1::DataType type, nvinfer1::Dims const& shape,
                                            char const* name, cudaStream_t stream) noexcept {
    size_t n = 1;
    for (int i = 0; i < shape.nbDims; ++i) n *= static_cast<size_t>(shape.d[i]);

    bool has_nan = false;
    if (type == nvinfer1::DataType::kFLOAT) {
        host.resize(n);
        if (location == nvinfer1::TensorLocation::kDEVICE) {
            cudaMemcpyAsync(host.data(), addr, n * sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        } else {
            std::memcpy(host.data(), addr, n * sizeof(float));
        }
        for (size_t i = 0; i < n; ++i) {
            if (std::isnan(host[i])) { has_nan = true; break; }
        }
    } else if (type == nvinfer1::DataType::kHALF) {
        std::vector<__half> h(n);
        if (location == nvinfer1::TensorLocation::kDEVICE) {
            cudaMemcpyAsync(h.data(), addr, n * sizeof(__half), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        } else {
            std::memcpy(h.data(), addr, n * sizeof(__half));
        }
        for (size_t i = 0; i < n; ++i) {
            if (std::isnan(static_cast<float>(h[i]))) { has_nan = true; break; }
        }
    }

    ++checked;
    note(name, has_nan);
    return true;
}

void NanFindingListener::note(char const* name, bool has_nan) {
    all.emplace_back(name, has_nan);
    if (has_nan) {
        ++nan_count;
        if (!found) {
            found = true;
            first_nan = name;
            std::cout << "[NaN] 第一个出现 NaN 的张量: " << name << std::endl;
            if (!before.empty()) {
                std::cout << "[NaN] 它之前最后几个正常张量(逆序):" << std::endl;
                for (auto it = before.rbegin(); it != before.rend(); ++it) {
                    std::cout << "        " << *it << std::endl;
                }
            }
        }
    } else {
        before.push_back(name);
        if (before.size() > 6) before.erase(before.begin());
    }
}

TrtEngine::~TrtEngine() {
    releaseBuffers();
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
    // context_ / engine_ / runtime_ 由 unique_ptr 释放
}

bool TrtEngine::loadOrBuild(const std::string& onnx_path,
                            const std::string& engine_path,
                            const EngineOptions& opt,
                            bool rebuild) {
    // 必须在任何 CUDA / TensorRT 调用之前选设备:
    // 引擎构建、显存分配、context 都跟着当前 CUDA 设备走。
    // 注意 engine 绑定的是**GPU 架构**(SM 版本)而不是设备序号 ——
    // 同架构的另一块卡可以直接复用同一个 engine,换架构必须 --rebuild。
    if (cudaSetDevice(opt.device) != cudaSuccess) {
        std::cerr << "[CUDA] 没有设备 " << opt.device << ":"
                  << cudaGetErrorString(cudaGetLastError()) << std::endl;
        return false;
    }
    {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, opt.device) == cudaSuccess) {
            std::cout << "[CUDA] 使用设备 " << opt.device << ": " << prop.name
                      << " (sm_" << prop.major << prop.minor << ", "
                      << prop.totalGlobalMem / (1024 * 1024) << " MB)" << std::endl;
        }
    }

    logger_.setLevel(opt.verbose ? nvinfer1::ILogger::Severity::kVERBOSE
                                 : nvinfer1::ILogger::Severity::kWARNING);
    if (!initLibNvInferPlugins(&logger_, "")) {
        std::cerr << "[TRT][W] initLibNvInferPlugins 失败(通常不影响标准算子)" << std::endl;
    }

    if (opt.debug_nan || opt.debug_nan_all) {
        debug_listener_ = std::make_unique<NanFindingListener>();
    }

    if (!rebuild && !engine_path.empty()) {
        std::ifstream ifs(engine_path, std::ios::binary);
        if (ifs.good()) {
            ifs.close();
            if (deserialize(engine_path)) {
                std::cout << "[TRT] 反序列化已有 engine: " << engine_path << std::endl;
                return true;
            }
            std::cerr << "[TRT] engine 反序列化失败,改为从 ONNX 重建" << std::endl;
        }
    }
    return build(onnx_path, engine_path, opt);
}

bool TrtEngine::build(const std::string& onnx_path, const std::string& engine_path,
                      const EngineOptions& opt) {
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger_));
    TRT_CHECK(builder != nullptr);

    // 默认 0U:显式 batch 的弱类型网络,精度由 BuilderFlag 决定;
    // 强类型时各张量精度完全按 ONNX 的 dtype 走。
    const uint32_t net_flags = opt.strongly_typed
        ? (1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED))
        : 0U;
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(net_flags));
    TRT_CHECK(network != nullptr);
    if (opt.strongly_typed) {
        std::cout << "[TRT] 强类型网络:精度完全按 ONNX 的 dtype 决定" << std::endl;
    }

    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger_));
    TRT_CHECK(parser != nullptr);

    std::cout << "[TRT] 解析 ONNX: " << onnx_path << std::endl;
    if (!parser->parseFromFile(onnx_path.c_str(),
                               static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            std::cerr << "[ONNX][E] " << parser->getError(i)->desc() << std::endl;
        }
        return false;
    }

    if (!opt.list_tensors.empty()) {
        int shown = 0;
        for (int i = 0; i < network->getNbLayers(); ++i) {
            nvinfer1::ILayer* layer = network->getLayer(i);
            if (layer == nullptr) continue;
            for (int o = 0; o < layer->getNbOutputs(); ++o) {
                nvinfer1::ITensor* t = layer->getOutput(o);
                if (t == nullptr) continue;
                const std::string n = t->getName();
                if (n.find(opt.list_tensors) != std::string::npos) {
                    std::cout << "[tensor] " << n << "  " << dimsToStr(t->getDimensions())
                              << "  dtype=" << static_cast<int>(t->getType()) << std::endl;
                    if (++shown >= 400) return false;
                }
            }
        }
        std::cout << "[TRT] 共匹配 " << shown << " 个张量" << std::endl;
        return false;   // 只列名字,不构建
    }

    if (opt.debug_nan || opt.debug_nan_all) {
        if (opt.debug_nan_all) {
            // 逐个标记:不会漏,但会阻止算子融合,engine 会明显变大变慢
            int marked = 0;
            const bool filtered = !opt.debug_nan_filter.empty();
            for (int i = 0; i < network->getNbLayers(); ++i) {
                nvinfer1::ILayer* layer = network->getLayer(i);
                if (layer == nullptr) continue;
                for (int o = 0; o < layer->getNbOutputs(); ++o) {
                    nvinfer1::ITensor* t = layer->getOutput(o);
                    if (t == nullptr) continue;
                    if (filtered &&
                        std::string(t->getName()).find(opt.debug_nan_filter) == std::string::npos) {
                        continue;
                    }
                    if (network->markDebug(*t)) ++marked;
                }
            }
            std::cout << "[TRT] 调试模式:标记了 " << marked << " 个张量(阻止融合)"
                      << (filtered ? ",过滤: " + opt.debug_nan_filter : "") << std::endl;
            if (filtered && marked == 0) {
                std::cout << "[TRT] 过滤没命中,下面是样例张量名(最多 15 个):" << std::endl;
                int shown = 0;
                for (int i = 0; i < network->getNbLayers() && shown < 15; ++i) {
                    nvinfer1::ILayer* layer = network->getLayer(i);
                    if (layer == nullptr) continue;
                    for (int o = 0; o < layer->getNbOutputs() && shown < 15; ++o) {
                        nvinfer1::ITensor* t = layer->getOutput(o);
                        if (t == nullptr) continue;
                        std::string n = t->getName();
                        if (n.find("backbone") != std::string::npos || shown < 5) {
                            std::cout << "        " << n << std::endl;
                            ++shown;
                        }
                    }
                }
            }
        } else {
            // 只标记本来就没被融合的张量,不额外损失性能
            network->markUnfusedTensorsAsDebugTensors();
            std::cout << "[TRT] 调试模式:标记未融合的张量(不阻止融合)" << std::endl;
        }
    }

    nvinfer1::ITensor* in_tensor = network->getInput(0);
    TRT_CHECK(in_tensor != nullptr);
    const nvinfer1::Dims in_dims = in_tensor->getDimensions();
    std::cout << "[TRT] 网络输入 " << in_tensor->getName() << " " << dimsToStr(in_dims) << std::endl;

    if (in_dims.nbDims != 4) {
        std::cerr << "[TRT] 只支持 4 维输入 [B,3,H,W],实际 " << dimsToStr(in_dims) << std::endl;
        return false;
    }
    if (in_dims.d[2] <= 0 || in_dims.d[3] <= 0) {
        std::cerr << "[TRT] H/W 是动态维,当前实现要求静态 H/W" << std::endl;
        return false;
    }

    const int channels = in_dims.d[1];
    const int bh = in_dims.d[2];
    const int bw = in_dims.d[3];
    // batch 维动态时 profile 取 [1, max_batch],opt 定在 1(本工具一次只跑一张图);
    // batch 维静态时 TRT 要求 min/opt/max 三者都等于网络声明值,否则 is valid 检查不过。
    int batch_min = 1;
    int batch_max = opt.max_batch;
    if (in_dims.d[0] > 0) {
        batch_min = batch_max = in_dims.d[0];
        if (opt.max_batch != in_dims.d[0]) {
            std::cerr << "[TRT][W] ONNX 的 batch 维是静态的(" << in_dims.d[0]
                      << "),忽略 --max-batch " << opt.max_batch << std::endl;
        }
    }
    const nvinfer1::Dims4 shape_min{batch_min, channels, bh, bw};
    const nvinfer1::Dims4 shape_opt{batch_min, channels, bh, bw};
    const nvinfer1::Dims4 shape_max{batch_max, channels, bh, bw};

    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    TRT_CHECK(config != nullptr);
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                               opt.workspace_mb * 1024ULL * 1024ULL);
    // TRT 10.12 起 kFP16 标为 deprecated(官方建议改用 strong typing),但"弱类型网络 +
    // kFP16"仍然可用,且保持输入/输出为 FP32、只有内部计算用 FP16 —— 正是这里想要的语义。
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    if (opt.fp16 && !opt.strongly_typed) {
        if (builder->platformHasFastFp16()) {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
            std::cout << "[TRT] 启用 FP16" << std::endl;
        } else {
            std::cerr << "[TRT][W] 该 GPU 不支持快速 FP16,回退 FP32" << std::endl;
        }
    }
    if (opt.fp16 && opt.fp16_fp32_norm) {
        // TRT 的提示原文:"Forcing Reduce or Pow Layers in FP32 precision ... can help
        // preserving accuracy"。deim decoder 的 RMSNorm 被拆成 Pow/ReduceMean,
        // 这些层在 FP16 下会溢出,这里按名字把它们钉回 FP32。
        config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
        int pinned = 0;
        for (int i = 0; i < network->getNbLayers(); ++i) {
            nvinfer1::ILayer* layer = network->getLayer(i);
            if (layer == nullptr) continue;
            const std::string lname = layer->getName();
            if (lname.find("norm") == std::string::npos) continue;
            const nvinfer1::LayerType lt = layer->getType();
            if (lt == nvinfer1::LayerType::kREDUCE || lt == nvinfer1::LayerType::kUNARY ||
                lt == nvinfer1::LayerType::kELEMENTWISE) {
                layer->setPrecision(nvinfer1::DataType::kFLOAT);
                ++pinned;
            }
        }
        std::cout << "[TRT] 把 " << pinned << " 个 norm 相关层钉在 FP32" << std::endl;
    }
#pragma GCC diagnostic pop

    // createOptimizationProfile 返回的指针由 builder 持有,调用方不得 delete
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    TRT_CHECK(profile != nullptr);
    TRT_CHECK(profile->setDimensions(in_tensor->getName(), nvinfer1::OptProfileSelector::kMIN,
                                     shape_min));
    TRT_CHECK(profile->setDimensions(in_tensor->getName(), nvinfer1::OptProfileSelector::kOPT,
                                     shape_opt));
    TRT_CHECK(profile->setDimensions(in_tensor->getName(), nvinfer1::OptProfileSelector::kMAX,
                                     shape_max));
    TRT_CHECK(config->addOptimizationProfile(profile) >= 0);

    const char* precision = opt.strongly_typed ? "强类型(按图里的 dtype)"
                                              : (opt.fp16 ? "FP16" : "FP32");
    std::cout << "[TRT] 开始构建 engine(batch " << batch_min << "~" << batch_max
              << ", " << precision
              << ", workspace=" << opt.workspace_mb << "MB),首次构建可能需要几分钟…" << std::endl;
    std::unique_ptr<nvinfer1::IHostMemory> serialized(
        builder->buildSerializedNetwork(*network, *config));
    TRT_CHECK(serialized != nullptr);
    std::cout << "[TRT] engine 构建完成,大小 " << serialized->size() << " 字节" << std::endl;

    if (!engine_path.empty()) {
        std::ofstream ofs(engine_path, std::ios::binary);
        if (ofs.good()) {
            ofs.write(static_cast<const char*>(serialized->data()),
                      static_cast<std::streamsize>(serialized->size()));
            std::cout << "[TRT] engine 已缓存到: " << engine_path << std::endl;
        } else {
            std::cerr << "[TRT][W] 无法写入 engine 缓存: " << engine_path << std::endl;
        }
    }

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    TRT_CHECK(runtime_ != nullptr);
    engine_.reset(runtime_->deserializeCudaEngine(serialized->data(), serialized->size()));
    TRT_CHECK(engine_ != nullptr);
    return initFromEngine();
}

bool TrtEngine::deserialize(const std::string& engine_path) {
    std::ifstream ifs(engine_path, std::ios::binary);
    TRT_CHECK(ifs.good());
    std::vector<char> blob((std::istreambuf_iterator<char>(ifs)),
                           std::istreambuf_iterator<char>());
    if (blob.empty()) {
        std::cerr << "[TRT] engine 文件为空: " << engine_path << std::endl;
        return false;
    }
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    TRT_CHECK(runtime_ != nullptr);
    engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
    if (engine_ == nullptr) {
        std::cerr << "[TRT] engine 与当前 TensorRT/GPU 不兼容: " << engine_path << std::endl;
        return false;
    }
    return initFromEngine();
}

bool TrtEngine::initFromEngine() {
    context_.reset(engine_->createExecutionContext());
    TRT_CHECK(context_ != nullptr);
    CUDA_CHECK(cudaStreamCreate(&stream_));

    int n_in = 0, n_out = 0;
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* name = engine_->getIOTensorName(i);
        const nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            input_name_ = name;
            ++n_in;
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            output_name_ = name;
            ++n_out;
        }
        const nvinfer1::DataType dt = engine_->getTensorDataType(name);
        if (dt != nvinfer1::DataType::kFLOAT) {
            std::cerr << "[TRT] 张量 " << name << " 的 dtype 不是 FP32(枚举值 "
                      << static_cast<int>(dt) << "),当前实现只处理 FP32" << std::endl;
            return false;
        }
        std::cout << "[TRT] " << (mode == nvinfer1::TensorIOMode::kINPUT ? "输入" : "输出")
                  << " " << name << " " << dimsToStr(engine_->getTensorShape(name))
                  << " (engine 声明形状)" << std::endl;
    }
    if (n_in != 1 || n_out != 1) {
        std::cerr << "[TRT] 只支持单输入单输出,实际 输入=" << n_in << " 输出=" << n_out << std::endl;
        return false;
    }

    const nvinfer1::Dims d = engine_->getTensorShape(input_name_.c_str());
    if (d.nbDims != 4 || d.d[2] <= 0 || d.d[3] <= 0) {
        std::cerr << "[TRT] 输入形状异常: " << dimsToStr(d) << std::endl;
        return false;
    }
    in_h_ = d.d[2];
    in_w_ = d.d[3];
    return true;
}

bool TrtEngine::ensureBuffer(const std::string& name, size_t bytes) {
    Buffer& buf = buffers_[name];
    if (buf.ptr != nullptr && buf.bytes >= bytes) return true;
    if (buf.ptr != nullptr) {
        cudaFree(buf.ptr);
        buf.ptr = nullptr;
        buf.bytes = 0;
    }
    CUDA_CHECK(cudaMalloc(&buf.ptr, bytes));
    buf.bytes = bytes;
    return true;
}

void TrtEngine::releaseBuffers() {
    for (auto& kv : buffers_) {
        if (kv.second.ptr != nullptr) cudaFree(kv.second.ptr);
        kv.second.ptr = nullptr;
        kv.second.bytes = 0;
    }
    buffers_.clear();
}

bool TrtEngine::infer(const float* input, int batch,
                      std::vector<float>& output, std::vector<int64_t>& out_shape) {
    TRT_CHECK(context_ != nullptr);

    const nvinfer1::Dims4 in_dims{batch, 3, in_h_, in_w_};
    if (!context_->setInputShape(input_name_.c_str(), in_dims)) {
        std::cerr << "[TRT] setInputShape " << dimsToStr(in_dims)
                  << " 失败:batch 超出构建 engine 时的 profile 范围" << std::endl;
        return false;
    }

    // 输出形状里可能带动态维,先让 TRT 推断一次
    nvinfer1::Dims out_dims = context_->getTensorShape(output_name_.c_str());
    if (hasDynamicDim(out_dims)) {
        if (context_->inferShapes(0, nullptr) != 0) {
            std::cerr << "[TRT] inferShapes 失败" << std::endl;
            return false;
        }
        out_dims = context_->getTensorShape(output_name_.c_str());
    }
    if (hasDynamicDim(out_dims)) {
        std::cerr << "[TRT] 输出形状仍含动态维: " << dimsToStr(out_dims) << std::endl;
        return false;
    }

    const size_t in_bytes = static_cast<size_t>(batch) * 3 * in_h_ * in_w_ * sizeof(float);
    const size_t out_elems = volume(out_dims);
    const size_t out_bytes = out_elems * sizeof(float);

    TRT_CHECK(ensureBuffer(input_name_, in_bytes));
    TRT_CHECK(ensureBuffer(output_name_, out_bytes));
    TRT_CHECK(context_->setTensorAddress(input_name_.c_str(), buffers_[input_name_].ptr));
    TRT_CHECK(context_->setTensorAddress(output_name_.c_str(), buffers_[output_name_].ptr));

    if (debug_listener_) {
        // 打开调试张量并挂上 listener,必须在 enqueueV3 之前
        if (!context_->setUnfusedTensorsDebugState(true)) {
            std::cerr << "[TRT][W] setUnfusedTensorsDebugState 失败" << std::endl;
        }
        context_->setTensorDebugState(output_name_.c_str(), true);  // 输出也一并看
        if (!context_->setDebugListener(debug_listener_.get())) {
            std::cerr << "[TRT][W] setDebugListener 失败" << std::endl;
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(buffers_[input_name_].ptr, input, in_bytes,
                               cudaMemcpyHostToDevice, stream_));
    if (!context_->enqueueV3(stream_)) {
        std::cerr << "[TRT] enqueueV3 失败" << std::endl;
        return false;
    }
    output.resize(out_elems);
    CUDA_CHECK(cudaMemcpyAsync(output.data(), buffers_[output_name_].ptr, out_bytes,
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    out_shape.assign(out_dims.d, out_dims.d + out_dims.nbDims);
    return true;
}

}  // namespace deimv2
