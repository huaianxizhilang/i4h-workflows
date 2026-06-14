# syntax=docker/dockerfile:1.4

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

# Regional apt mirror (China: mirrors.ustc.edu.cn, mirrors.aliyun.com, mirrors.tuna.tsinghua.edu.cn)
# Set APT_MIRROR= to keep archive.ubuntu.com
ARG APT_MIRROR=mirrors.ustc.edu.cn

# PyPI / Miniconda mirrors (China)
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_EXTRA_INDEX_URL=https://pypi.nvidia.com
ARG PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
ARG CONDA_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/anaconda

SHELL ["/bin/bash", "-c"]

# Install all packages in a single layer to avoid caching issues
RUN if [[ -n "${APT_MIRROR}" ]]; then \
        for f in /etc/apt/sources.list /etc/apt/sources.list.d/ubuntu.sources; do \
            [[ -f "${f}" ]] || continue; \
            sed -i \
                -e "s|http://archive.ubuntu.com/ubuntu/|http://${APT_MIRROR}/ubuntu/|g" \
                -e "s|http://security.ubuntu.com/ubuntu/|http://${APT_MIRROR}/ubuntu/|g" \
                -e "s|archive.ubuntu.com|${APT_MIRROR}|g" \
                -e "s|security.ubuntu.com|${APT_MIRROR}|g" \
                "${f}"; \
        done; \
        echo "Using apt mirror: ${APT_MIRROR}"; \
    fi && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        wget \
        curl \
        jq \
        vim \
        git \
        xvfb \
        build-essential \
        cmake \
        vulkan-tools \
        unzip \
        lsb-release \
        libglib2.0-0 \
        libdbus-1-3 \
        libopengl0 \
        libxcb-keysyms1 \
        libglu1-mesa && \
    rm -rf /var/lib/apt/lists/*

# all devices should be visible
ENV NVIDIA_VISIBLE_DEVICES=all
# set 'compute' driver cap to use Cuda
# set 'video' driver cap to use the video encoder
# set 'graphics' driver cap to use OpenGL/EGL
# set 'display' to allow use of virtual display
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility,display

# Install Everything in conda environment

WORKDIR /workspace/i4h-workflows

COPY tools /workspace/i4h-workflows/tools
COPY workflows /workspace/i4h-workflows/workflows
COPY i4h /workspace/i4h-workflows/i4h
COPY CMakeLists.txt /workspace/i4h-workflows/CMakeLists.txt

RUN mkdir -p /opt/miniconda3 && \
    if [[ -n "${CONDA_MIRROR}" ]]; then \
        MINICONDA_URL="${CONDA_MIRROR}/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    else \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    fi && \
    echo "Downloading Miniconda from ${MINICONDA_URL}" && \
    wget -q "${MINICONDA_URL}" -O /opt/miniconda3/miniconda.sh && \
    bash /opt/miniconda3/miniconda.sh -b -u -p /opt/miniconda3 && \
    rm /opt/miniconda3/miniconda.sh && \
    source /opt/miniconda3/bin/activate && \
    conda tos accept && \
    conda init --all && \
    conda create -n robotic_ultrasound python=3.11 -y && \
    chmod -R a+rX /opt/miniconda3

ENV PATH="/opt/miniconda3/envs/robotic_ultrasound/bin:/opt/miniconda3/bin:$PATH"
ENV CONDA_DEFAULT_ENV=robotic_ultrasound

RUN --mount=type=cache,target=/root/.cache/pip \
    source /opt/miniconda3/bin/activate && \
    conda activate robotic_ultrasound && \
    if [[ -n "${PIP_INDEX_URL}" ]]; then \
        pip config set global.index-url "${PIP_INDEX_URL}"; \
        [[ -n "${PIP_TRUSTED_HOST}" ]] && pip config set global.trusted-host "${PIP_TRUSTED_HOST}"; \
    fi && \
    if [[ -n "${PIP_EXTRA_INDEX_URL}" ]]; then \
        pip config set global.extra-index-url "${PIP_EXTRA_INDEX_URL}"; \
    fi && \
    cd /workspace/i4h-workflows && \
    BUILD_DOCKER_IMAGE=true bash tools/env_setup_robot_us.sh && \
    chmod -R a+rX /opt/miniconda3
# Fix: tensorstore 0.1.81 requires ml_dtypes.float8_e3m4 (added in 0.5.0),
# but openpi pins ml_dtypes 0.2.0 via tensorflow. --no-deps avoids pulling
# in newer numpy/jax/jaxlib that would break other packages.
RUN --mount=type=cache,target=/root/.cache/pip \
    source /opt/miniconda3/bin/activate && \
    conda activate robotic_ultrasound && \
    pip install --no-deps "ml_dtypes>=0.5.0,<1.0" && \
    chmod -R a+rX /opt/miniconda3

ENV PYTHONPATH=/workspace/i4h-workflows/workflows/robotic_ultrasound/scripts:/workspace/i4h-workflows/install/lib/clarius_solum:/workspace/i4h-workflows/install/lib/clarius_cast
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/i4h-workflows/install/lib/clarius_solum:/workspace/i4h-workflows/install/lib/clarius_cast
ENV RTI_LICENSE_FILE=/root/rti/rti_license.dat
