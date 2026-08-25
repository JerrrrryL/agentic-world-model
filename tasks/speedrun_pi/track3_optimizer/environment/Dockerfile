# NanoGPT optimizer speedrun (Prime Intellect track-3) — agent sandbox.
#
# PI ran the task in a bwrap sandbox on the host with a plain venv (Python 3.12.3,
# torch==2.11) rather than a CUDA image, so we mirror that: Ubuntu 24.04 ships
# Python 3.12, and the torch wheel brings its own CUDA runtime via the nvidia-*
# packages.  libcuda.so comes from the host driver, injected by the nvidia
# container runtime into /usr/lib/x86_64-linux-gnu (see TRITON_LIBCUDA_PATH).
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        build-essential ca-certificates curl git tmux procps vim less \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv on PATH, as in the upstream setup.
RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/venv

COPY requirements.txt /opt/requirements.txt
RUN pip install --upgrade pip && pip install -r /opt/requirements.txt

WORKDIR /app
COPY train_gpt_simple.py run.sh verify.py requirements.txt /app/
# `logs/` is where the frozen logging infra writes; `data/` is where the
# read-only FineWeb volume is mounted by environment/docker-compose.yaml.
# NOTE: data is never baked into the image.
RUN mkdir -p /app/logs /app/data /app/scratchpad && touch /app/scratchpad/.gitkeep

# Triton/Inductor needs libcuda without /sbin/ldconfig; nvidia-smi needs the
# `utility` driver capability (the compose device reservation only asks for `gpu`).
ENV TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TORCHINDUCTOR_CACHE_DIR=/app/.inductor-cache \
    OMP_NUM_THREADS=8

CMD ["sleep", "infinity"]
