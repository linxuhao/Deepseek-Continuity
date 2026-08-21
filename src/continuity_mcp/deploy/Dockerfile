# ==========================================
# 两个 ggml/Vulkan 引擎的构建 + 运行镜像。
#
# 从源码编译而不是拉预编译二进制: Vulkan 后端的 shader 是运行时编译的 SPIR-V,
# 但 host 端还是要针对本机的编译器/库编。这也让镜像与厂商无关 —— 里面没有一行
# CUDA 或 ROCm。换卡不用换镜像, 但接入方式分两种:
#   A 卡 / Intel   宿主机递 /dev/dri, ICD 用镜像里的 mesa      (A 卡实测过)
#   N 卡           nvidia-container-toolkit 注入宿主机驱动     (未实测, 见 nvidia 覆盖文件)
#
# 三个依赖不加就会构建失败, 每个都花过一次:
#   mesa-vulkan-drivers  Vulkan 的 ICD。A 卡 (RADV) 和 Intel (ANV) 的驱动都在 mesa 里,
#                        缺了容器里 vulkaninfo 报 "Found no drivers"。
#                        N 卡不走这条: 它的 ICD 在宿主机驱动里, 由 nvidia-container-toolkit
#                        注入, 见 docker-compose.nvidia.yml。
#   spirv-headers        ggml-vulkan.cpp 条件 include <spirv/unified1/spirv.hpp>,
#                        缺了报 "'spv' has not been declared"
#   glslc/glslang-tools  编译 shader
# libglslang-dev 在 Ubuntu 24.04 不存在, 别加。
# ==========================================
FROM ubuntu:24.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates pkg-config ninja-build \
      libvulkan-dev vulkan-tools mesa-vulkan-drivers \
      glslc glslang-tools spirv-tools spirv-headers \
      python3 curl \
 && rm -rf /var/lib/apt/lists/*

# 固定到实测跑通的提交。浮动到 master 意味着某天的上游改动会静默改变本机行为,
# 而这套东西的全部价值就是"跨次调用不变"。
ARG SDCPP_REF=97d2990
ARG AUDIOCPP_REF=aec444c

WORKDIR /src
RUN git clone --recursive https://github.com/leejet/stable-diffusion.cpp.git \
 && git -C stable-diffusion.cpp checkout ${SDCPP_REF} \
 && git -C stable-diffusion.cpp submodule update --init --recursive \
 && cmake -S stable-diffusion.cpp -B stable-diffusion.cpp/build \
      -DCMAKE_BUILD_TYPE=Release -DSD_VULKAN=ON -G Ninja \
 && cmake --build stable-diffusion.cpp/build -j"$(nproc)"

RUN git clone https://github.com/0xShug0/audio.cpp.git \
 && git -C audio.cpp checkout ${AUDIOCPP_REF} \
 && git -C audio.cpp submodule update --init --recursive \
 && cd audio.cpp \
 && bash scripts/build_linux.sh --backend vulkan \
      --target audiocpp_cli --target audiocpp_server

# ---- 运行镜像: 只要 Vulkan 运行时, 不要编译器 ----
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      libvulkan1 mesa-vulkan-drivers vulkan-tools libgomp1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/stable-diffusion.cpp/build/bin/ /opt/continuity/bin/
COPY --from=build /src/audio.cpp/build/linux-vulkan-release/bin/ /opt/continuity/bin/
# model_specs 不是可选的: /v1/audio/speech 要按 family 找 model contract spec, 找不到就
# 500 "model contract spec not found for family 'qwen3_tts'"。而那个端点是唯一支持
# 内联 base64 参考音的 —— 少了它, 克隆就只能靠文件路径, 跨机部署直接不成立。
# 解析器会从 cwd 逐级向上找 model_specs/<family>.json, 所以放在 WORKDIR 下即可。
COPY --from=build /src/audio.cpp/model_specs/ /opt/continuity/model_specs/
ENV PATH="/opt/continuity/bin:${PATH}"
WORKDIR /opt/continuity
