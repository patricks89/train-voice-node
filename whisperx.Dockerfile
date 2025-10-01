FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_NO_TORCHVISION=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git ffmpeg tini && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Torch-Stack (ohne torchvision)
RUN pip install --upgrade pip && \
    pip install torch==2.4.1+cu121 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Audio/IO + sentencepiece
RUN pip install "numpy<1.27,>=1.23" "pyarrow<16.2" soundfile librosa==0.10.2.post1 sentencepiece "torchmetrics<1"

# Schnelles ASR-Backend
RUN pip install "ctranslate2>=4.3" "faster-whisper==1.0.0"

# Transformers + WhisperX
RUN pip install "transformers==4.40.2" accelerate && \
    pip install git+https://github.com/m-bain/whisperx.git

# Sanity-Check (leichte Imports)
RUN python - <<'PY'
import os, torch
print("torch", torch.__version__, "cuda:", torch.cuda.is_available())
print("TRANSFORMERS_NO_TORCHVISION =", os.environ.get("TRANSFORMERS_NO_TORCHVISION"))
import ctranslate2, faster_whisper, whisperx
print("deps import OK")
PY

ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["/bin/bash"]
