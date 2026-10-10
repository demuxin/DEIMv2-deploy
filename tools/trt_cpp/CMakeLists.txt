cmake_minimum_required(VERSION 3.17)
project(deimv2_trt_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "构建类型" FORCE)
endif()

# TensorRT 安装根目录(容器内默认路径,可用 -DTENSORRT_ROOT=... 覆盖)
set(TENSORRT_ROOT "/usr/local/tensorrt/TensorRT-10.12.0.36" CACHE PATH "TensorRT 根目录")

find_package(CUDAToolkit REQUIRED)          # 只需要 cudart,没有 .cu 文件,不依赖 nvcc
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# TensorRT 10.x 未提供 CMake config,手工导入两个 shared lib
add_library(trt_nvinfer SHARED IMPORTED)
set_target_properties(trt_nvinfer PROPERTIES
  IMPORTED_LOCATION "${TENSORRT_ROOT}/lib/libnvinfer.so"
  INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_ROOT}/include")

add_library(trt_onnxparser SHARED IMPORTED)
set_target_properties(trt_onnxparser PROPERTIES
  IMPORTED_LOCATION "${TENSORRT_ROOT}/lib/libnvonnxparser.so"
  INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_ROOT}/include")

# initLibNvInferPlugins 在 plugin 库里(不在 libnvinfer)
add_library(trt_plugin SHARED IMPORTED)
set_target_properties(trt_plugin PROPERTIES
  IMPORTED_LOCATION "${TENSORRT_ROOT}/lib/libnvinfer_plugin.so"
  INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_ROOT}/include")

add_executable(deimv2_trt_infer
  src/main.cpp
  src/trt_engine.cpp
  src/preprocess.cpp
  src/postprocess.cpp)

target_include_directories(deimv2_trt_infer PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  ${OpenCV_INCLUDE_DIRS})

target_link_libraries(deimv2_trt_infer PRIVATE
  trt_nvinfer
  trt_onnxparser
  trt_plugin
  CUDA::cudart
  ${OpenCV_LIBS})

# 运行时从 TensorRT 目录找 .so,省去手工 export LD_LIBRARY_PATH
set_target_properties(deimv2_trt_infer PROPERTIES
  BUILD_RPATH "${TENSORRT_ROOT}/lib"
  INSTALL_RPATH "${TENSORRT_ROOT}/lib")

# 打开 -O3 与常用告警
target_compile_options(deimv2_trt_infer PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-O3 -Wall -Wextra -Wno-unused-parameter>)
