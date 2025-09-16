#!/usr/bin/env bash
set -euo pipefail

########################
# USER-VARIABLEN
########################
NAS_SRC="/mnt/nas/Allgemein/VoiceClone/GianGiachen"

# Lokale, schnelle NVMe-Ziele
LOCAL_ROOT="/mnt/nvme_pool"
LOCAL_DATA="${LOCAL_ROOT}/dataset"
LOCAL_WAVS="${LOCAL_DATA}/wavs"
LOCAL_TXT="${LOCAL_DATA}/transcripts"
LOCAL_CKPT="${LOCAL_ROOT}/checkpoints"

# Zielordner auf dem NAS für fertige Checkpoints/Modelle
NAS_OUT="${NAS_SRC}/checkpoints_out"

# WhisperX Modell & Sprache
WHISPER_MODEL="large-v3"
LANG="de"

# Hugging Face Token (optional)
HF_TOKEN="${HF_TOKEN:-}"

# Docker Image-Namen
IMG_WHISPER="whisper-tools"
IMG_XTTS="xtts-finetune:cu121"

# Für Subprozesse exportieren
export LOCAL_ROOT LOCAL_DATA LOCAL_WAVS LOCAL_TXT LOCAL_CKPT NAS_OUT HF_TOKEN WHISPER_MODEL LANG

########################
# PREP
########################
echo "==> Prüfe & lege lokale Ordner an …"
mkdir -p "${LOCAL_WAVS}" "${LOCAL_TXT}" "${LOCAL_CKPT}" "${NAS_OUT}"

########################
# DOCKERFILES
########################
echo "==> Erzeuge whisperx.Dockerfile …"
cat > whisperx.Dockerfile <<'DOCK'
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
DOCK

echo "==> Erzeuge xtts-finetune.Dockerfile …"
cat > xtts-finetune.Dockerfile <<'DOCK'
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg sox tini libsndfile1 build-essential && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip && \
    pip install "TTS@git+https://github.com/coqui-ai/TTS.git@main#egg=TTS"
RUN git clone https://github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages.git /opt/xtts-ft && \
    pip install -r /opt/xtts-ft/requirements.txt
RUN python - <<'PY'
from TTS.tts.configs.xtts_config import XttsConfig
print("XttsConfig import OK")
PY
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["/bin/bash"]
DOCK

########################
# IMAGES BAUEN
########################
echo "==> Baue Docker-Images …"
docker build -t "${IMG_WHISPER}" -f whisperx.Dockerfile .
docker build -t "${IMG_XTTS}"    -f xtts-finetune.Dockerfile .

########################
# DATEN VOM NAS SPIEGELN
########################
echo "==> Kopiere Rohdaten vom NAS lokal (rsync)…"
mkdir -p "${LOCAL_DATA}"
rsync -av --progress --delete \
  --include="*/" --include="*.mp4" --include="*.wav" --exclude="*" \
  "${NAS_SRC}/" "${LOCAL_DATA}/"

########################
# DATEINAMEN SÄUBERN
########################
echo "==> Säubere Dateinamen (nur a-zA-Z0-9._-) rekursiv …"
env LOCAL_DATA="${LOCAL_DATA}" python3 - <<'PY'
import os, re, shutil
root = os.environ["LOCAL_DATA"]
safe_re = re.compile(r'[^a-zA-Z0-9._-]')
def sanitize(name: str) -> str:
    base, ext = os.path.splitext(name)
    return safe_re.sub('_', base) + safe_re.sub('_', ext)
for dirpath, _, filenames in os.walk(root):
    for fn in filenames:
        src = os.path.join(dirpath, fn)
        new = sanitize(fn)
        if new != fn:
            dst = os.path.join(dirpath, new)
            base, ext = os.path.splitext(new); i=1
            while os.path.exists(dst):
                dst = os.path.join(dirpath, f"{base}_{i}{ext}"); i += 1
            print(f"rename: {src} -> {dst}")
            shutil.move(src, dst)
print(">> Filename-Sanitize fertig")
PY

########################
# MP4 -> WAV @22050 mono (parallel)
########################
echo "==> Konvertiere MP4 -> WAV (22050 Hz, mono) parallel …"
mkdir -p "${LOCAL_WAVS}"
export LOCAL_WAVS
find "${LOCAL_DATA}" -type f -iname '*.mp4' -print0 | \
  xargs -0 -I{} -P"$(nproc)" bash -c '
    mp4="$1"
    base="$(basename "$mp4")"
    stem="${base%.*}"
    out="${LOCAL_WAVS}/${stem}.wav"
    if [[ ! -f "$out" ]]; then
      ffmpeg -y -i "$mp4" -ar 22050 -ac 1 -vn "$out" >/dev/null 2>&1 || exit 0
    fi
  ' _ {}

echo ">> MP4->WAV done"

########################
# TRANSKRIPTION (GPU, fp16, batching, VAD aus)
########################
echo "==> Transkribiere WAVs mit WhisperX einmalig (GPU fp16, batching) …"
docker run --rm --gpus all \
  -e TRANSFORMERS_NO_TORCHVISION=1 \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e LANG_CODE="${LANG}" \
  -e MODEL_NAME="${WHISPER_MODEL}" \
  -v "${LOCAL_DATA}":/data \
  "${IMG_WHISPER}" bash -lc '
set -e
python - << "PY"
import os, glob, torch
import whisperx

root = "/data"
wav_dir = os.path.join(root, "wavs")
out_dir = os.path.join(root, "transcripts")
os.makedirs(out_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
# try fp16 first; if OOM happens, switch to int8_float16 (less VRAM, a bit slower)
preferred = "float16" if device == "cuda" else "int8"
model_name = os.environ.get("MODEL_NAME", "large-v3")
lang = os.environ.get("LANG_CODE", "de")

def load_model_safe(compute_type):
    print(f"load model: {model_name}  device: {device}  dtype: {compute_type}")
    return whisperx.load_model(
        model_name,
        device,
        compute_type=compute_type,
        # pass language directly to the ASR; helps stability
        asr_options={"initial_prompt": ""},
        # turn on VAD for cleaner segments; if pyannote issues occur, set method="none"
        vad_options={"method": "pyannote", "min_speech_duration_ms": 250},
    )

try:
    model = load_model_safe(preferred)
except Exception as e:
    if device == "cuda" and preferred != "int8_float16":
        print(f"Falling back to int8_float16 due to: {e}")
        model = load_model_safe("int8_float16")
    else:
        raise

wavs = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
print("files:", len(wavs))

# keep batch small to fit 11GB on 2080 Ti
BATCH = 6 if device == "cuda" else 2

for i, wav in enumerate(wavs, 1):
    base = os.path.splitext(os.path.basename(wav))[0]
    txt_path = os.path.join(out_dir, base + ".txt")
    if os.path.exists(txt_path):
        continue
    audio = whisperx.load_audio(wav)
    result = model.transcribe(audio, batch_size=BATCH, language=lang)
    with open(txt_path, "w", encoding="utf-8") as f:
        if "segments" in result:
            for seg in result["segments"]:
                t = (seg.get("text") or "").strip()
                if t:
                    f.write(t + "\n")
        else:
            t = (result.get("text") or "").strip()
            if t:
                f.write(t + "\n")
    if i % 10 == 0:
        print(f"[{i}/{len(wavs)}] {base}")
print(">> transcription done")
PY
'

########################
# MP4 AUFRÄUMEN
########################
echo "==> Lösche lokale MP4 (bereits in WAV umgewandelt) …"
find "${LOCAL_DATA}" -type f -name "*.mp4" -delete || true

########################
# METADATEN (Coqui: PIPE-getrennt "audio_file|text")
########################
echo "==> Erzeuge train/eval Metadaten (Coqui, audio_file|text) …"
env LOCAL_DATA="${LOCAL_DATA}" python3 - <<'PY'
import os, glob, random, csv

root    = os.environ["LOCAL_DATA"]
wav_dir = os.path.join(root, "wavs")
txt_dir = os.path.join(root, "transcripts")

MAX_LEN = 250            # harte Grenze pro Zeile (XTTS ~253)
EVAL_MIN = 3             # mindestens 3 Eval-Zeilen
EVAL_FRAC = 0.10         # 10% Eval
pairs = []

def chunk_text(t, max_len=MAX_LEN):
    t = " ".join(x.strip() for x in t.split())  # normalize whitespace
    if len(t) <= max_len:
        return [t]
    return [ t[i:i+max_len] for i in range(0, len(t), max_len) ]

for wav in sorted(glob.glob(os.path.join(wav_dir, "*.wav"))):
    stem = os.path.splitext(os.path.basename(wav))[0]
    txt_path = os.path.join(txt_dir, stem + ".txt")
    if not os.path.exists(txt_path):
        continue
    with open(txt_path, "r", encoding="utf-8") as f:
        text = " ".join(line.strip() for line in f if line.strip()).strip()
    if not text:
        continue
    # --- Container-Pfad direkt reinschreiben ---
    wav_ctr = "/workspace/dataset/wavs/" + os.path.basename(wav)
    # --- lange Texte in <=MAX_LEN-Stücke splitten ---
    for chunk in chunk_text(text):
        pairs.append((wav_ctr, chunk))

if len(pairs) < 2:
    raise SystemExit("Zu wenige Samples für train/eval!")

# Kürzere Zeilen bevorzugt für Eval auswählen (um Filter zu vermeiden)
pairs.sort(key=lambda r: len(r[1]))
eval_n = max(EVAL_MIN, int(EVAL_FRAC * len(pairs)))
eval_set  = pairs[:eval_n]
train_set = pairs[eval_n:]

def write_pipe(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="|", quoting=csv.QUOTE_ALL)
        w.writerow(["audio_file","text"])
        for r in rows:
            w.writerow(r)

train_csv = os.path.join(root, "metadata_train_coqui.csv")
eval_csv  = os.path.join(root, "metadata_eval_coqui.csv")
write_pipe(train_csv, train_set)
write_pipe(eval_csv,  eval_set)
print(f">> train rows={len(train_set)}  eval rows={len(eval_set)} (total after split={len(pairs)})")
PY

########################
# TRAINING
########################
echo "==> Starte XTTS GPT-Finetune (lokal, NVMe)…"
docker run --rm -it --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v "${LOCAL_DATA}":/workspace/dataset \
  -v "${LOCAL_CKPT}":/workspace/checkpoints \
  "${IMG_XTTS}" bash -lc '
set -e
cd /opt/xtts-ft

# 1) Normalisiere Metadata auf Container-Pfade und stelle Pipe-Delimiter sicher
python - << "PY"
import os, pandas as pd
root="/workspace/dataset"
for split in ("train","eval"):
    p=os.path.join(root, f"metadata_{split}_coqui.csv")
    # Korrekt: Quotes parsen lassen, Pipe als Trenner
    df=pd.read_csv(p, sep="|", engine="python")
    # Spalten prüfen
    need={"audio_file","text"}
    if not need.issubset(df.columns):
        cols=df.columns.tolist()
        if len(cols)>=2:
            df=df.rename(columns={cols[0]:"audio_file", cols[1]:"text"})
        else:
            raise RuntimeError(f"Unerwartetes Format in {p}")
    df=df[["audio_file","text"]]
    # Defensive: Whitespace strippen
    df["audio_file"]=df["audio_file"].astype(str).str.strip()
    df["text"]=df["text"].astype(str).str.strip()
    # Optional: Plausibilitätscheck der ersten 3 Zeilen
    print("Preview", p, ":\n", df.head(3).to_string(index=False))
    # Keine Pfad-Umschreibung mehr nötig, CSV enthält Container-Pfade
    df.to_csv(p, sep="|", index=False)
    print("normalized", p, "rows=", len(df))
PY

# 2) Sanity-Check
python - << "PY"
import pandas as pd
for p in ["/workspace/dataset/metadata_train_coqui.csv",
          "/workspace/dataset/metadata_eval_coqui.csv"]:
    df = pd.read_csv(p, sep="|")
    assert "audio_file" in df.columns and "text" in df.columns
    print("OK:", p, "rows:", len(df))
PY

# 3) Training mit Coqui-Metadaten (pipe)
python3 train_gpt_xtts.py \
  --output_path /workspace/checkpoints \
  --metadatas /workspace/dataset/metadata_train_coqui.csv,/workspace/dataset/metadata_eval_coqui.csv,de \
  --num_epochs 5 \
  --batch_size 2 \
  --grad_acumm 6 \
  --max_text_length 220 \
  --max_audio_length 220500 \
  --weight_decay 1e-2 \
  --lr 5e-6 \
  --save_step 25 \
'

########################
# EXPORT ZUM NAS
########################
echo "==> Exportiere Checkpoints zum NAS …"
rsync -av --progress "${LOCAL_CKPT}/" "${NAS_OUT}/"

echo "✅ Fertig."
