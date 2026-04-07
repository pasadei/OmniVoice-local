FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        git \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create venv to avoid PEP 668 issues
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch with CUDA 12.8
RUN pip install --no-cache-dir \
    torch==2.8.0+cu128 \
    torchaudio==2.8.0+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Install OmniVoice from PyPI
RUN pip install --no-cache-dir omnivoice

# Install server dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
COPY app/server.py /app/server.py

# Defaults (override via environment or compose)
ENV OMNIVOICE_MODEL="k2-fsa/OmniVoice" \
    OMNIVOICE_DEVICE="cuda:0" \
    OMNIVOICE_DTYPE="float16" \
    OMNIVOICE_SAMPLES_DIR="/samples" \
    OMNIVOICE_HOST="0.0.0.0" \
    OMNIVOICE_PORT="8000" \
    OMNIVOICE_OUTPUT_FORMAT="wav" \
    OMNIVOICE_WYOMING_ENABLED="false" \
    OMNIVOICE_WYOMING_HOST="0.0.0.0" \
    OMNIVOICE_WYOMING_PORT="10200"

EXPOSE 8000 8001
EXPOSE 10200

CMD ["python3", "server.py"]
