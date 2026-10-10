// DEIMv2 检测模型 C++ TensorRT 推理验证工具
//
// 流程与 tools/inference/onnx_inf_v2.py 对齐:
//   预处理  直接拉伸(非 letterbox)到模型输入尺寸 -> /255 -> ImageNet 归一化(dinov3)
//   推理    TensorRT(engine 首次从 ONNX 构建并缓存)
//   后处理  sigmoid -> 每 query 取最高分类别 -> 阈值过滤 -> cxcywh×原图 -> xyxy -> 全类别 NMS
//
// 用法:
//   ./deimv2_trt_infer --onnx z_onnx/xxx.onnx --image example.jpg
//   ./deimv2_trt_infer --onnx z_onnx/xxx.onnx --image example.jpg --fp16 --rebuild

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "postprocess.h"
#include "preprocess.h"
#include "trt_engine.h"

namespace {

using deimv2::Detection;

// 与 onnx_inf_v2.py 的 _CLASS_COLORS 一致(注意这里是 RGB,PIL 用的顺序)
constexpr int kClassColors[][3] = {
    {230, 25, 75},   {60, 180, 75},  {255, 225, 25}, {0, 130, 200},  {245, 130, 48},
    {145, 30, 180},  {70, 240, 240}, {240, 50, 230}, {250, 190, 190}, {170, 110, 40},
};
constexpr int kNumClassColors = sizeof(kClassColors) / sizeof(kClassColors[0]);

struct Args {
    std::string onnx;
    std::string image;
    std::string engine;
    std::string output = "trt_result.jpg";
    std::string dump_input;
    std::string dump_output;
    std::string save_txt;
    std::string nan_dump;
    float thrh = 0.25f;
    float nms_iou = 0.9f;
    int device = 0;
    int max_batch = 1;
    int workspace_mb = 2048;
    int benchmark = 0;
    bool fp16 = false;
    bool rebuild = false;
    bool verbose = false;
    bool no_normalize = false;
    bool fp16_fp32_norm = false;
    bool strongly_typed = false;
    bool debug_nan = false;
    bool debug_nan_all = false;
    std::string debug_nan_filter;
    std::string list_tensors;
    deimv2::NormParams norm;
};

void printUsage(const char* prog) {
    std::cout <<
        "用法: " << prog << " --onnx <model.onnx> --image <img.jpg> [选项]\n"
        "\n"
        "必填:\n"
        "  --onnx <path>         ONNX 模型(export_onnx_v2.py 导出的裸 concat 图)\n"
        "  --image <path>        输入图片\n"
        "\n"
        "engine:\n"
        "  --device <n>          用哪块 GPU(默认 0;engine 只与 GPU 架构绑定,与序号无关)\n"
        "  --engine <path>       engine 缓存路径(默认 <onnx>.engine)\n"
        "  --rebuild             忽略已有缓存,强制从 ONNX 重新构建\n"
        "  --max-batch <n>       engine 支持的 batch 上限(默认 1,实际按 1 张图推理)\n"
        "  --fp16                弱类型 FP16 构建。精度不可接受,仅作对照\n"
        "  --fp16-fp32-norm      同上,但把 norm 相关的层钉在 FP32\n"
        "  --strongly-typed      强类型构建:精度按 ONNX 的 dtype 走。\n"
        "                        要用 FP16 请配 export_onnx_v2.py --fp16 导出的模型\n"
        "  --debug-nan           逐张量找 NaN(标记未融合的张量,不掉性能)\n"
        "  --debug-nan-all       同上,但标记每个张量(会阻止融合,engine 更大更慢)\n"
        "  --debug-nan-filter S  只标记名字含 S 的张量(配合上一个用,缩小范围)\n"
        "  --list-tensors S      只列出名字含 S 的张量名和 dtype,不做构建(用于查命名)\n"
        "  --workspace <mb>      workspace 上限(默认 2048)\n"
        "\n"
        "前处理:\n"
        "  --no-normalize        只做 /255,不做 ImageNet 归一化(对应 deim_dfine)\n"
        "  --mean r,g,b          自定义归一化均值(默认 0.485,0.456,0.406)\n"
        "  --std r,g,b           自定义归一化标准差(默认 0.229,0.224,0.225)\n"
        "\n"
        "后处理:\n"
        "  --thrh <f>            分数阈值(默认 0.25,与 onnx_inf_v2.py 一致)\n"
        "  --nms-iou <f>         全类别 NMS 的 IoU 阈值(默认 0.9;<=0 关闭)\n"
        "\n"
        "输出:\n"
        "  --output <path>       结果图(默认 trt_result.jpg)\n"
        "  --save-txt <path>     检测框文本(便于与 Python 结果 diff)\n"
        "  --dump-input <path>   预处理后的输入张量存为 f32 裸数据\n"
        "  --dump-output <path>  原始输出张量存为 f32 裸数据\n"
        "  --benchmark <n>       推理 n 次取平均耗时\n"
        "  --verbose             打开 TensorRT 详细日志\n";
}

bool parseFloats3(const std::string& s, float out[3]) {
    std::stringstream ss(s);
    std::string tok;
    int i = 0;
    while (std::getline(ss, tok, ',') && i < 3) {
        try {
            out[i++] = std::stof(tok);
        } catch (...) {
            return false;
        }
    }
    return i == 3;
}

bool parseArgs(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&](std::string& dst) -> bool {
            if (i + 1 >= argc) {
                std::cerr << "缺少参数值: " << k << std::endl;
                return false;
            }
            dst = argv[++i];
            return true;
        };
        if (k == "--help" || k == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        } else if (k == "--onnx") {
            if (!next(a.onnx)) return false;
        } else if (k == "--image") {
            if (!next(a.image)) return false;
        } else if (k == "--engine") {
            if (!next(a.engine)) return false;
        } else if (k == "--output") {
            if (!next(a.output)) return false;
        } else if (k == "--save-txt") {
            if (!next(a.save_txt)) return false;
        } else if (k == "--nan-dump") {
            if (!next(a.nan_dump)) return false;
        } else if (k == "--dump-input") {
            if (!next(a.dump_input)) return false;
        } else if (k == "--dump-output") {
            if (!next(a.dump_output)) return false;
        } else if (k == "--mean") {
            std::string v;
            if (!next(v) || !parseFloats3(v, a.norm.mean)) {
                std::cerr << "--mean 需要 3 个逗号分隔的数" << std::endl;
                return false;
            }
        } else if (k == "--std") {
            std::string v;
            if (!next(v) || !parseFloats3(v, a.norm.std)) {
                std::cerr << "--std 需要 3 个逗号分隔的数" << std::endl;
                return false;
            }
        } else if (k == "--thrh") {
            std::string v;
            if (!next(v)) return false;
            a.thrh = std::stof(v);
        } else if (k == "--nms-iou") {
            std::string v;
            if (!next(v)) return false;
            a.nms_iou = std::stof(v);
        } else if (k == "--device") {
            std::string v;
            if (!next(v)) return false;
            a.device = std::stoi(v);
        } else if (k == "--max-batch") {
            std::string v;
            if (!next(v)) return false;
            a.max_batch = std::stoi(v);
        } else if (k == "--workspace") {
            std::string v;
            if (!next(v)) return false;
            a.workspace_mb = std::stoi(v);
        } else if (k == "--benchmark") {
            std::string v;
            if (!next(v)) return false;
            a.benchmark = std::stoi(v);
        } else if (k == "--fp16") {
            a.fp16 = true;
        } else if (k == "--fp16-fp32-norm") {
            a.fp16 = true;
            a.fp16_fp32_norm = true;
        } else if (k == "--strongly-typed") {
            a.strongly_typed = true;
        } else if (k == "--debug-nan") {
            a.debug_nan = true;
        } else if (k == "--debug-nan-all") {
            a.debug_nan = true;
            a.debug_nan_all = true;
        } else if (k == "--list-tensors") {
            if (!next(a.list_tensors)) return false;
        } else if (k == "--debug-nan-filter") {
            a.debug_nan = true;
            a.debug_nan_all = true;
            if (!next(a.debug_nan_filter)) return false;
        } else if (k == "--rebuild") {
            a.rebuild = true;
        } else if (k == "--verbose") {
            a.verbose = true;
        } else if (k == "--no-normalize") {
            a.no_normalize = true;
        } else {
            std::cerr << "未知参数: " << k << std::endl;
            printUsage(argv[0]);
            return false;
        }
    }
    if (a.onnx.empty() || a.image.empty()) {
        std::cerr << "--onnx 与 --image 均为必填" << std::endl;
        printUsage(argv[0]);
        return false;
    }
    if (a.engine.empty()) a.engine = a.onnx + ".engine";
    a.norm.enable = !a.no_normalize;
    return true;
}

bool writeRawF32(const std::string& path, const float* data, size_t count) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.good()) {
        std::cerr << "无法写入: " << path << std::endl;
        return false;
    }
    ofs.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * sizeof(float)));
    std::cout << "  已导出 " << path << " (" << count << " 个 float32)" << std::endl;
    return true;
}

void drawDetections(cv::Mat& bgr, const std::vector<Detection>& dets) {
    for (const Detection& d : dets) {
        const int* c = kClassColors[d.label % kNumClassColors];
        const cv::Scalar color(c[2], c[1], c[0]);  // RGB(PIL) -> BGR(OpenCV)

        cv::rectangle(bgr, cv::Point(cvRound(d.x1), cvRound(d.y1)),
                      cv::Point(cvRound(d.x2), cvRound(d.y2)), color, 3);

        char label[64];
        std::snprintf(label, sizeof(label), "cls%d: %.2f", d.label, d.score);
        int baseline = 0;
        const cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        const int ty = std::max(cvRound(d.y1), ts.height + 4);
        cv::rectangle(bgr,
                      cv::Point(cvRound(d.x1), ty - ts.height - 4),
                      cv::Point(cvRound(d.x1) + ts.width + 4, ty + baseline),
                      color, cv::FILLED);
        cv::putText(bgr, label, cv::Point(cvRound(d.x1) + 2, ty - 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parseArgs(argc, argv, args)) return 1;

    using Clock = std::chrono::steady_clock;
    auto msSince = [](Clock::time_point t0) {
        return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    };

    // ---- 1. 图片 ----
    cv::Mat bgr = cv::imread(args.image, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        std::cerr << "无法读取图片: " << args.image << std::endl;
        return 1;
    }
    const int orig_w = bgr.cols;
    const int orig_h = bgr.rows;
    std::cout << "图片: " << args.image << " " << orig_w << "x" << orig_h << std::endl;

    // ---- 2. engine ----
    deimv2::EngineOptions opt;
    opt.device = args.device;
    opt.max_batch = args.max_batch;
    opt.fp16 = args.fp16;
    opt.fp16_fp32_norm = args.fp16_fp32_norm;
    opt.strongly_typed = args.strongly_typed;
    opt.debug_nan = args.debug_nan;
    opt.debug_nan_all = args.debug_nan_all;
    opt.debug_nan_filter = args.debug_nan_filter;
    opt.list_tensors = args.list_tensors;
    opt.workspace_mb = args.workspace_mb;
    opt.verbose = args.verbose;

    deimv2::TrtEngine engine;
    if (!engine.loadOrBuild(args.onnx, args.engine, opt, args.rebuild)) {
        std::cerr << "engine 准备失败" << std::endl;
        return 1;
    }

    const int in_h = engine.inputH();
    const int in_w = engine.inputW();
    std::cout << "模型输入: " << in_w << "x" << in_h
              << " | 归一化: " << (args.norm.enable ? "ImageNet (x/255-mean)/std" : "仅 /255")
              << std::endl;

    // ---- 3. 预处理 ----
    auto t0 = Clock::now();
    std::vector<unsigned char> rgb(static_cast<size_t>(orig_w) * orig_h * 3);
    cv::Mat rgb_view(orig_h, orig_w, CV_8UC3, rgb.data());
    cv::cvtColor(bgr, rgb_view, cv::COLOR_BGR2RGB);

    std::vector<float> input;
    deimv2::preprocess(rgb.data(), orig_h, orig_w, in_h, in_w, args.norm, input);
    const double t_pre = msSince(t0);

    if (!args.dump_input.empty()) {
        writeRawF32(args.dump_input, input.data(), input.size());
    }

    // ---- 4. 推理 ----
    std::vector<float> output;
    std::vector<int64_t> out_shape;
    t0 = Clock::now();
    if (!engine.infer(input.data(), 1, output, out_shape)) {
        std::cerr << "推理失败" << std::endl;
        return 1;
    }
    const double t_infer = msSince(t0);

    if (const auto* rep = engine.nanReport(); rep != nullptr) {
        std::cout << "[NaN] 检查了 " << rep->checked << " 个张量,其中 "
                  << rep->nan_count << " 个含 NaN" << std::endl;
        if (!rep->found) {
            std::cout << "[NaN] 没有发现 NaN" << std::endl;
        }
        if (!args.nan_dump.empty()) {
            std::ofstream ofs(args.nan_dump);
            ofs << "# name has_nan\n";
            for (const auto& [n, bad] : rep->all) ofs << n << " " << (bad ? 1 : 0) << "\n";
            std::cout << "[NaN] 全部张量状态已写入: " << args.nan_dump << std::endl;
        }
    }

    if (out_shape.size() != 3) {
        std::cerr << "输出维数异常: " << out_shape.size() << std::endl;
        return 1;
    }
    const int n_queries = static_cast<int>(out_shape[1]);
    const int num_classes = static_cast<int>(out_shape[2]) - 4;
    std::cout << "模型输出: [1," << n_queries << "," << out_shape[2] << "] -> query 数 "
              << n_queries << ",类别数 " << num_classes << std::endl;

    if (!args.dump_output.empty()) {
        writeRawF32(args.dump_output, output.data(), output.size());
    }

    if (args.benchmark > 0) {
        double total = 0.0;
        for (int i = 0; i < args.benchmark; ++i) {
            const auto tb = Clock::now();
            if (!engine.infer(input.data(), 1, output, out_shape)) {
                std::cerr << "benchmark 推理失败" << std::endl;
                return 1;
            }
            total += msSince(tb);
        }
        std::cout << "benchmark: " << args.benchmark << " 次平均 "
                  << (total / args.benchmark) << " ms" << std::endl;
    }

    // ---- 5. 后处理 ----
    t0 = Clock::now();
    const float scale[4] = {static_cast<float>(orig_w), static_cast<float>(orig_h),
                            static_cast<float>(orig_w), static_cast<float>(orig_h)};
    std::vector<Detection> dets =
        deimv2::decodeOutput(output.data(), n_queries, num_classes, scale, args.thrh);
    const int n_before = static_cast<int>(dets.size());
    deimv2::nmsClassAgnostic(dets, args.nms_iou);
    const double t_post = msSince(t0);

    if (args.nms_iou > 0.f) {
        std::cout << "检测到 " << dets.size() << " 个目标(score >= " << args.thrh
                  << ";全类别 NMS IoU=" << args.nms_iou << " 过滤 " << n_before << " -> "
                  << dets.size() << ")" << std::endl;
    } else {
        std::cout << "检测到 " << dets.size() << " 个目标(score >= " << args.thrh
                  << ";NMS 已关闭)" << std::endl;
    }
    for (const Detection& d : dets) {
        std::printf("  [%d] score=%.3f xyxy=(%.0f,%.0f,%.0f,%.0f)\n", d.label, d.score,
                    d.x1, d.y1, d.x2, d.y2);
    }

    std::cout << "耗时: 预处理 " << t_pre << " ms | 推理 " << t_infer << " ms | 后处理 "
              << t_post << " ms" << std::endl;

    // ---- 6. 保存 ----
    drawDetections(bgr, dets);
    if (!cv::imwrite(args.output, bgr)) {
        std::cerr << "结果图写入失败: " << args.output << std::endl;
        return 1;
    }
    std::cout << "结果已保存: " << args.output << std::endl;

    if (!args.save_txt.empty()) {
        std::ofstream ofs(args.save_txt);
        if (!ofs.good()) {
            std::cerr << "无法写入: " << args.save_txt << std::endl;
            return 1;
        }
        ofs << "# label score x1 y1 x2 y2\n";
        for (const Detection& d : dets) {
            ofs << d.label << " " << d.score << " " << d.x1 << " " << d.y1 << " " << d.x2 << " "
                << d.y2 << "\n";
        }
        std::cout << "检测框已写入: " << args.save_txt << std::endl;
    }
    return 0;
}
