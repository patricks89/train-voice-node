#!/usr/bin/env bash
set -euo pipefail

# ==============================
# Defaults (auf fast_store umgestellt)
# ==============================
LANG="${LANG:-de}"
CKPT_DIR="${CKPT_DIR:-/mnt/fast_store/checkpoints/latest_full}"
ORIG_DIR="${ORIG_DIR:-/mnt/fast_store/checkpoints/latest_full}"   # Fallback, falls tokenizer fehlt
DATA_DIR="${DATA_DIR:-/mnt/fast_store/dataset}"                   # ggf. auf /mnt/fast_scratch/dataset setzen
OUT_DIR="${OUT_DIR:-/mnt/fast_store/tts_out}"
IMG_XTTS="${IMG_XTTS:-xtts-finetune:cu121}"

TEXT=""
REF_WAV=""
OUT_WAV=""

usage() {
  cat <<USAGE
Usage: $(basename "$0") --text "Hallo Welt" [--lang de] [--ref "/path/ref.wav"] [--out "/path/out.wav"]
Optionen:
  --text, -t   Zu sprechender Text (Pflicht, oder via STDIN)
  --lang, -l   Sprachcode (default: de)
  --ref,  -r   Referenz-WAV (Voice-Conditioning). Falls leer: längstes Segment aus DATA_DIR/wavs/segments, dann /mnt/fast_store/tts_refs
  --out,  -o   Ausgabe-WAV (default: OUT_DIR/tts_YYYYmmdd_HHMMSS.wav)
  --ckpt       Pfad zu latest_full (default: ${CKPT_DIR})
  --orig       Pfad zu Original-/Tokenizer-Assets (default: ${ORIG_DIR})
  --data       Pfad zu Dataset (default: ${DATA_DIR})
  --img        Docker-Image (default: ${IMG_XTTS})
USAGE
  exit 1
}

# ------------------------------
# CLI-Args
# ------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --text|-t) TEXT="$2"; shift 2;;
    --lang|-l) LANG="$2"; shift 2;;
    --ref|-r)  REF_WAV="$2"; shift 2;;
    --out|-o)  OUT_WAV="$2"; shift 2;;
    --ckpt)    CKPT_DIR="$2"; shift 2;;
    --orig)    ORIG_DIR="$2"; shift 2;;
    --data)    DATA_DIR="$2"; shift 2;;
    --img)     IMG_XTTS="$2"; shift 2;;
    -h|--help) usage;;
    *) echo "Unbekannte Option: $1"; usage;;
  esac
done

# Text ggf. aus STDIN
if [[ -z "${TEXT}" && ! -t 0 ]]; then
  TEXT="$(cat)"
fi
[[ -z "${TEXT}" ]] && { echo "Fehler: --text fehlt (oder per STDIN übergeben)."; usage; }

# Pfade prüfen
MODEL_PTH="${CKPT_DIR}/model.pth"
CFG_JSON="${CKPT_DIR}/config.json"
[[ -f "${MODEL_PTH}" ]] || { echo "Fehlt: ${MODEL_PTH}"; exit 1; }
[[ -f "${CFG_JSON}" ]]  || { echo "Fehlt: ${CFG_JSON}"; exit 1; }
mkdir -p "${OUT_DIR}"

# Referenz automatisch wählen (längstes Segment), wenn nicht gesetzt
auto_ref=""
if [[ -z "${REF_WAV}" ]]; then
  SEG_DIR="${DATA_DIR}/wavs/segments"
  if [[ -d "${SEG_DIR}" ]]; then
    best=""; bestdur=0
    shopt -s nullglob
    for f in "${SEG_DIR}"/*.wav; do
      dur=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$f" || echo 0)
      dur=${dur%.*}
      if [[ "${dur:-0}" -gt "${bestdur:-0}" ]]; then best="$f"; bestdur="$dur"; fi
    done
    if [[ -n "${best}" ]]; then
      auto_ref="$best"
      echo ">> REF_WAV auto (Segments): ${auto_ref} (~${bestdur}s)"
    fi
  fi
fi

# Zweiter Fallback: persistente Referenzen
if [[ -z "${REF_WAV}" && -z "${auto_ref}" && -d "/mnt/fast_store/tts_refs" ]]; then
  auto_ref="$(ls -1 /mnt/fast_store/tts_refs/*.wav 2>/dev/null | head -n1 || true)"
  [[ -n "${auto_ref}" ]] && echo ">> REF_WAV auto (tts_refs): ${auto_ref}"
fi

# Final wählen
if [[ -z "${REF_WAV}" ]]; then
  REF_WAV="${auto_ref}"
fi
[[ -n "${REF_WAV}" ]] || { echo "Keine Referenz gefunden (weder Segmente noch /mnt/fast_store/tts_refs). Bitte --ref angeben."; exit 1; }

# Referenz-Datei validieren
if [[ -d "${REF_WAV}" ]]; then
  echo "Fehler: --ref zeigt auf ein VERZEICHNIS, erwartet WAV-Datei: ${REF_WAV}"
  exit 1
fi
[[ -f "${REF_WAV}" ]] || { echo "Fehler: Referenz-WAV existiert nicht: ${REF_WAV}"; exit 1; }

# Out-Dateiname default
if [[ -z "${OUT_WAV}" ]]; then
  ts="$(date +%F_%H%M%S)"
  OUT_WAV="${OUT_DIR}/tts_${ts}.wav"
fi

# Text temporär ablegen (sauber für Docker)
TEXT_FILE="$(mktemp)"
printf '%s' "${TEXT}" > "${TEXT_FILE}"

# ------------------------------
# Inference (Docker)
# ------------------------------
docker run --rm --gpus all \
  --shm-size=4g --ipc=host \
  -e LANG_CODE="${LANG}" \
  -e OUT_BASENAME="$(basename "${OUT_WAV}")" \
  -v "${CKPT_DIR}":/ckpt:ro \
  -v "${ORIG_DIR}":/orig:ro \
  -v "${DATA_DIR}":/data:ro \
  -v "${OUT_DIR}":/out \
  -v "${TEXT_FILE}":/tmp/text.txt:ro \
  -v "${REF_WAV}":/host_ref.wav:ro \
  "${IMG_XTTS}" bash -lc '
set -e
# 1) Referenz robust vorbereiten: 24kHz/mono, max 60s
ffmpeg -y -i "/host_ref.wav" -ar 24000 -ac 1 -t 60 "/tmp/ref_24k_mono_60s.wav" >/dev/null 2>&1

python - <<PY
import os, json, torch, soundfile as sf
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

ckpt_dir = "/ckpt"
orig_dir = "/orig"
text     = open("/tmp/text.txt","r",encoding="utf-8").read()
lang     = os.environ.get("LANG_CODE","de")
out_wav  = os.path.join("/out", os.path.basename(os.environ.get("OUT_BASENAME","tts.wav")))
ref_wav  = "/tmp/ref_24k_mono_60s.wav"

cfg_path = os.path.join(ckpt_dir, "config.json")
model_pth= os.path.join(ckpt_dir, "model.pth")
checkpoint_dir = ckpt_dir

# 1) Config & Modell
cfg = XttsConfig(); cfg.load_json(cfg_path)
model = Xtts.init_from_config(cfg)

# 2) Checkpoint laden (versch. Coqui-Versionen kompatibel)
#    Bevorzugt tokenizer.json im CKPT_DIR, sonst vocab/merges aus ORIG_DIR
vocab_path = None
tok_json   = os.path.join(ckpt_dir, "tokenizer.json")
if os.path.isfile(tok_json):
    vocab_path = None
else:
    v = os.path.join(orig_dir, "vocab.json")
    m = os.path.join(orig_dir, "merges.txt")
    if os.path.isfile(v) and os.path.isfile(m):
        vocab_path = v

loaded = False
try:
    model.load_checkpoint(cfg, checkpoint_path=model_pth, checkpoint_dir=checkpoint_dir, vocab_path=vocab_path, use_deepspeed=False)
    loaded = True; print("[load] pattern A ok")
except TypeError as e:
    print("[load] pattern A TypeError:", e)

if not loaded:
    try:
        if vocab_path:
            model.load_checkpoint(cfg, checkpoint_dir, model_pth, vocab_path)
        else:
            model.load_checkpoint(cfg, checkpoint_dir, model_pth)
        loaded = True; print("[load] pattern B ok")
    except Exception as e:
        print("[load] pattern B failed:", e)

if not loaded:
    model.load_checkpoint(cfg, checkpoint_dir, model_pth); print("[load] pattern C ok")

model.eval()
if torch.cuda.is_available():
    model = model.to("cuda")

# 3) Voice-Conditioning – Längen als INTs
gpt_cond_len   = int(30)   # Sekunden
max_ref_length = int(60)   # Sekunden
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
    audio_path=[ref_wav],
    gpt_cond_len=gpt_cond_len,
    max_ref_length=max_ref_length
)

# 4) Synthese
with torch.no_grad():
    wav = model.inference(
        text=text,
        language=lang,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=0.7
    )

sr = 24000
if isinstance(wav, dict) and "wav" in wav:
    audio = wav["wav"]; sr = wav.get("sample_rate", sr)
else:
    audio = wav

sf.write(out_wav, audio, sr)
print(">> wrote:", out_wav)
PY
'

echo "✅ Fertig: ${OUT_WAV}"
rm -f "${TEXT_FILE}"
