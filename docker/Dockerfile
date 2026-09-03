# TTT-MEMO development environment.
#
# No CUDA base image: e2e/uv.lock already pins the full CUDA stack as pip
# wheels via jax[cuda12] -> jax-cuda12-plugin[with-cuda], and those are what
# JAX loads at runtime:
#
#   nvidia-cuda-nvcc-cu12   12.8.93       (ptxas)
#   nvidia-cuda-runtime-cu12 12.8.90
#   nvidia-cudnn-cu12       9.8.0.87
#   nvidia-nccl-cu12        2.26.2.post1
#   nvidia-cublas/cufft/cusolver/cusparse/nvjitlink/cupti-cu12
#
# That covers the CUDA 12.8.1 / cuDNN 9.8.0 / NCCL 2.26.2 versions the official
# README recommends, so the container only needs the host NVIDIA driver, which
# the NVIDIA container runtime injects. This codebase has no custom CUDA
# kernels (attention goes through jax.nn.dot_product_attention).
#
# Trade-off: Nsight binaries (nsys, ncu) are not in the pip wheels. JAX's own
# profiler still works through cupti.
#
# Python 3.12 matches the cp312 wheels resolved in e2e/uv.lock.
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv is pinned, but it does not choose package versions here: `uv sync --locked`
# refuses to re-resolve and installs exactly what e2e/uv.lock records.
COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /uvx /usr/local/bin/

# The virtualenv lives in the image, never inside the bind-mounted submodule.
# e2e/.gitignore does not ignore .venv, so a venv there would leave the
# submodule permanently dirty.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=/usr/local/bin/python3.12 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy

# Only the dependency manifests are copied, so editing source code does not
# invalidate this layer.
WORKDIR /build
COPY e2e/pyproject.toml e2e/uv.lock ./

# --no-install-project: install dependencies but not the `ttt-e2e` package
#   itself. Building it would write egg-info into the mounted submodule, and we
#   import `ttt` via PYTHONPATH instead.
# --no-dev: skip the upstream dev group (ruff only).
RUN uv sync --locked --no-install-project --no-dev

COPY docker/requirements-research.txt ./
RUN uv pip install --python /opt/venv/bin/python -r requirements-research.txt

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Import the official implementation from the submodule without installing it.
ENV PYTHONPATH=/workspace/e2e

# A CUDA base image would set these; on a plain Python base they must be
# declared so the NVIDIA container runtime mounts the driver libraries.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# The GPU is shared with the Windows desktop; do not grab all VRAM at startup.
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

WORKDIR /workspace

CMD ["bash"]
