#!/usr/bin/env bash
set -euo pipefail

########################
# USER-VARIABLEN
########################
# Quellordner auf dem NAS mit Rohdaten (mp4/wav gemischt ok)
NAS_SRC="/mnt/nas/Allgemein/VoiceClone/GianGiachen"

# Lokale, schnelle NVMe-Ziele
LOCAL_ROOT="/mnt/nvme_pool/voiceclone"
LOCAL_DATA="${LOCAL_ROOT}/dataset"
LOCAL_WAVS="${LOCAL_DATA}/wavs"
LOCAL_TXT="${LOCAL_DATA}/transcripts"
LOCAL_CKPT="${LOCAL_ROOT}/checkpoints"

# Zielordner auf dem NAS für fertige Checkpoints/Modelle
NAS_OUT="${NAS_SRC}/checkpoints_out"

# WhisperX Modell & Sprache
WHISPER_MODEL="small"
LANG="de"   # WhisperX erwartet 'de'

# Hugging Face Token (optional)
HF_TOKEN="${HF_TOKEN:-}"

# Docker Image-Namen
IMG_WHISPER="whisper-tools"
IMG_XTTS="xtts-finetune:cu121"

# Diese Variablen in den ENV exportieren (für Python-Heredocs etc.)
export NAS_SRC LOCAL_ROOT LOCAL_DATA LOCAL_WAVS LOCAL_TXT LOCAL_CKPT NAS_OUT
export WHISPER_MODEL LANG HF_TOKEN IMG_WHISPER IMG_XTTS

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
    TRANSFORMERS_NO_TORCHVISION=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git ffmpeg tini && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Torch-Stack passend zu CUDA 12.1
RUN pip install --upgrade pip && \
    pip install \
      torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1 \
      --index-url https://download.pytorch.org/whl/cu121

# Audio/IO + Tools
RUN pip install "numpy<1.27,>=1.23" "pyarrow<16.2" soundfile librosa==0.10.2.post1 \
    sentencepiece "torchmetrics<1"

# Schnellere ASR-Backends und Abhängigkeiten, die WhisperX beim Import erwartet
RUN pip install "ctranslate2>=4.3" "faster-whisper==1.0.0" \
    "pyannote.audio==3.1.1"

# Transformers + WhisperX
RUN pip install "transformers==4.40.2" accelerate && \
    pip install git+https://github.com/m-bain/whisperx.git

# Sanity-Check (nur leichte Imports)
RUN python - <<'PY'
import torch
print("torch", torch.__version__, "cuda:", torch.cuda.is_available())
import ctranslate2, faster_whisper, whisperx
from pyannote.audio.core.model import Model as _Dummy  # stellt sicher, dass pyannote geladen werden kann
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
# LOCAL_DATA ist exportiert -> für Python verfügbar
python3 - <<'PY'
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
# TRANSKRIPTION (GPU fp16, batching, ohne VAD)
########################
echo "==> Transkribiere WAVs mit WhisperX einmalig (GPU fp16, batching, ohne VAD) …"
docker run --rm --gpus all \
  -e TRANSFORMERS_NO_TORCHVISION=1 \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e LANG_CODE="${LANG}" \
  -e MODEL_NAME="${WHISPER_MODEL}" \
  -v "${LOCAL_DATA}":/data \
  "${IMG_WHISPER}" bash -lc '
set -e
python3 - << "PY"
import os, glob, torch
import whisperx

root="/data"
wav_dir=os.path.join(root,"wavs")
out_dir=os.path.join(root,"transcripts")
os.makedirs(out_dir, exist_ok=True)

device="cuda" if torch.cuda.is_available() else "cpu"
compute_type="float16" if device=="cuda" else "int8"
model_name=os.environ.get("MODEL_NAME","small")
lang=os.environ.get("LANG_CODE","de")

print("load model:", model_name, "device:", device, "dtype:", compute_type)
model = whisperx.load_model(
    model_name,
    device,
    compute_type=compute_type,
    asr_options={"initial_prompt": ""},
    vad_options={"method": "none"}   # VAD/diarization aus
)

wavs = sorted(glob.glob(os.path.join(wav_dir,"*.wav")))
print("files:", len(wavs))

for i, wav in enumerate(wavs, 1):
    base=os.path.splitext(os.path.basename(wav))[0]
    txt_path=os.path.join(out_dir, base+".txt")
    if os.path.exists(txt_path):
        continue
    audio = whisperx.load_audio(wav)
    result = model.transcribe(audio, batch_size=16, language=lang)
    with open(txt_path,"w",encoding="utf-8") as f:
        if "segments" in result:
            for seg in result["segments"]:
                t = (seg.get("text") or "").strip()
                if t: f.write(t+"\n")
        else:
            t = (result.get("text") or "").strip()
            if t: f.write(t+"\n")
    if i % 10 == 0:
        print(f"[{i}/{len(wavs)}] {base}")
print(">> transcription done")
PY
'

########################
# MP4 AUFRÄUMEN (nicht mehr benötigt)
########################
echo "==> Lösche lokale MP4 (bereits in WAV umgewandelt) …"
find "${LOCAL_DATA}" -type f -name "*.mp4" -delete || true

########################
# METADATEN (ohne pandas)
########################
echo "==> Erzeuge train/eval Metadaten (pure Python) …"
python3 - <<'PY'
import os, glob, random, csv
root = os.environ["LOCAL_DATA"]
wav_dir = os.path.join(root, "wavs")
txt_dir = os.path.join(root, "transcripts")

pairs = []
for wav in sorted(glob.glob(os.path.join(wav_dir, "*.wav"))):
    stem = os.path.splitext(os.path.basename(wav))[0]
    txt_path = os.path.join(txt_dir, stem + ".txt")
    if not os.path.exists(txt_path):
        continue
    with open(txt_path, "r", encoding="utf-8") as f:
        text = " ".join(line.strip() for line in f if line.strip())
    text = text.strip()
    if text:
        pairs.append((wav, text))

if len(pairs) < 2:
    raise SystemExit("Zu wenige Samples für train/eval!")

random.seed(1337)
random.shuffle(pairs)
eval_n = max(1, int(0.05 * len(pairs)))
eval_set = pairs[:eval_n]
train_set = pairs[eval_n:]

def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audio_file","text"])
        for r in rows: w.writerow(r)

train_csv = os.path.join(root, "metadata_train_coqui.csv")
eval_csv  = os.path.join(root, "metadata_eval_coqui.csv")
write_csv(train_csv, train_set)
write_csv(eval_csv,  eval_set)
print(f">> train rows={len(train_set)}  eval rows={len(eval_set)}")
PY

########################
# TRAINING
########################
echo "==> Starte XTTS GPT-Finetune (lokal, NVMe)…"
docker run --rm -it --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:true" \
  -v "${LOCAL_DATA}":/workspace/dataset \
  -v "${LOCAL_CKPT}":/workspace/checkpoints \
  "${IMG_XTTS}" bash -lc '
set -e
cd /opt/xtts-ft

# Metadaten für das Training in Pipe-Format und Container-Pfade mappen
python3 - << "PY"
import os, csv
root="/workspace/dataset"
def convert(split):
    src=os.path.join(root, f"metadata_{split}_coqui.csv")
    dst=os.path.join(root, f"metadata_{split}_pipe.csv")
    rows=[]
    with open(src, encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r:
            wav=row["audio_file"]
            mapped=os.path.join("/workspace/dataset/wavs", os.path.basename(wav))
            rows.append((mapped, row["text"]))
    with open(dst,"w",encoding="utf-8",newline="") as f:
        w=csv.writer(f, delimiter="|")
        for a,t in rows:
            w.writerow([a,t])
    print("wrote", dst, "rows=",len(rows))
for s in ("train","eval"): convert(s)
PY

# Kurzer Check
python3 - << "PY"
import os, csv
for s in ("train","eval"):
    path=f"/workspace/dataset/metadata_{s}_pipe.csv"
    ok=0; total=0
    with open(path,encoding="utf-8") as f:
        r=csv.reader(f, delimiter="|")
        for row in r:
            total+=1
            ok+= os.path.exists(row[0])
    print(s, "rows=",total, "ok=",ok, "miss=", total-ok)
PY

# Training
python3 train_gpt_xtts.py \
  --output_path /workspace/checkpoints \
  --metadatas /workspace/dataset/metadata_train_pipe.csv,/workspace/dataset/metadata_eval_pipe.csv,de \
  --num_epochs 5 \
  --batch_size 2 \
  --grad_acumm 6 \
  --max_text_length 220 \
  --max_audio_length 220500 \
  --weight_decay 1e-2 \
  --lr 5e-6 \
  --save_step 2000
'

########################
# EXPORT ZUM NAS
########################
echo "==> Exportiere Checkpoints zum NAS …"
rsync -av --progress "${LOCAL_CKPT}/" "${NAS_OUT}/"

echo "✅ Fertig."
