# docker build -t 192.168.2.78:1443/algolib/train/deimv2:1.0 -f Dockerfile .

FROM ubuntu:22.04

USER root

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt update && \
    apt-get install -y --no-install-recommends \
        python3-pip vim \
        tzdata \
        ca-certificates && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install onnx onnxruntime onnxsim torch==2.5.1 torchvision==0.20.1 \
    faster-coco-eval>=1.6.7 PyYAML tensorboard scipy calflops transformers

# docker run -it --gpus=all --user root --privileged=true \
# -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -e TZ=Asia/Shanghai \
# -v /data/50T/zxd:/workspace --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
# -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility --shm-size=64g --pid=host \
# --net=host --name="deimv2_dev" 192.168.2.78:1443/algolib/train/deimv2:1.0 /bin/bash
