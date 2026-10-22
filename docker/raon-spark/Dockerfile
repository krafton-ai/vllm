# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
ARG BASE_IMAGE=vllm/vllm-openai@sha256:2a7cde230b59f3ce6cab33dd245ba6bee41aa87b38c9fe84f966ff24016813ce
ARG BUILD_IMAGE=vllm/vllm-openai@sha256:2a7cde230b59f3ce6cab33dd245ba6bee41aa87b38c9fe84f966ff24016813ce
FROM ${BUILD_IMAGE} AS build
RUN uv venv --allow-existing --python /usr/bin/python3 --system-site-packages /opt/venv
ENV VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:${PATH}"
COPY docker/raon-spark/overlay.lock /tmp/overlay.lock
RUN uv pip install --python /opt/venv/bin/python --no-deps -r /tmp/overlay.lock
# Download this pinned ARM64 wheel as described in README.md before building.
COPY docker/raon-spark/precompiled.whl /tmp/precompiled.whl
RUN echo '29cfdc2b966c72e7515807ad0fbdeda0c15db01599b58197b09bf8a095556ec6  /tmp/precompiled.whl' | sha256sum -c -
WORKDIR /opt/raon/source
COPY pyproject.toml setup.py MANIFEST.in CMakeLists.txt LICENSE README.md ./
COPY vllm ./vllm
COPY csrc ./csrc
COPY cmake ./cmake
COPY rust ./rust
COPY requirements ./requirements
COPY tools/build_rust.py ./tools/build_rust.py
ARG VLLM_REF
RUN test -n "$VLLM_REF" && \
    VLLM_USE_PRECOMPILED=1 VLLM_TARGET_DEVICE=cuda \
    VLLM_PRECOMPILED_WHEEL_LOCATION=/tmp/precompiled.whl \
    VLLM_VERSION_OVERRIDE="0.1.dev1+g${VLLM_REF}" \
    uv pip install --python /opt/venv/bin/python --no-deps --no-build-isolation .

FROM ${BASE_IMAGE} AS runtime
COPY --from=build /opt/venv /opt/venv
ARG VLLM_REF
LABEL org.opencontainers.image.title="Raon vLLM for NVIDIA DGX Spark" \
      org.opencontainers.image.source="https://github.com/krafton-ai/vllm" \
      org.opencontainers.image.revision="${VLLM_REF}" \
      org.opencontainers.image.licenses="Apache-2.0"
# CUDA runtime headers must precede the pip-distributed cuBLAS/cuSPARSE headers.
# CPATH would override nvcc's explicit CUDA include path and mix SDK versions.
ENV VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:${PATH}" \
    CPATH="" CPLUS_INCLUDE_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include \
    TORCH_CUDA_ARCH_LIST=12.1 MAX_JOBS=8 \
    VLLM_CACHE_ROOT=/cache/vllm TORCH_EXTENSIONS_DIR=/cache/torch
COPY docker/raon-spark/serve.sh /opt/raon/serve.sh
WORKDIR /opt/raon
ENTRYPOINT ["bash", "/opt/raon/serve.sh"]
