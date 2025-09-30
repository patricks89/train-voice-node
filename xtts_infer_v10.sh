#!/usr/bin/env bash
set -euo pipefail

# Beispiele für Aufrufe:
#   ./scripts/xtts_infer_v1.sh --text "Allegra zämä, willkommen bi üs." --ref /pfad/zur/stimme.wav
#   ./scripts/xtts_infer_v1.sh --text "Buongiorno." --lang it --ckpt /mnt/fast_store/checkpoints/latest_full --temperature 0.6 --top-p 0.8
#   ./scripts/xtts_infer_v1.sh --from-file prompts.txt --style-prompt "sprich warm und gelassen" --speed 0.92 --seed 1234
#   ./scripts/xtts_infer_v1.sh --text "Breaking news." --emotion excited --repetition-penalty 12 --length-penalty 1.2 --denoise 1

# ==============================
# Defaults (an xtts_finetune_whisperx_asr_v8.sh angepasst)
# ==============================
STORE_ROOT_DEFAULT="${STORE_ROOT_DEFAULT:-/mnt/fast_store}"
TARGET_LANG="${XTTS_LANG:-de}"
CKPT_DIR="${CKPT_DIR:-${STORE_ROOT_DEFAULT}/checkpoints/latest_full}"
ORIG_DIR="${ORIG_DIR:-${STORE_ROOT_DEFAULT}/checkpoints/XTTS_v2.0_original_model_files}"
DATA_DIR="${DATA_DIR:-${STORE_ROOT_DEFAULT}/dataset_ref}"
OUT_DIR="${OUT_DIR:-${STORE_ROOT_DEFAULT}/tts_out}"
IMG_XTTS="${IMG_XTTS:-xtts-finetune:cu121}"
CACHE_ROOT="${CACHE_ROOT:-${STORE_ROOT_DEFAULT}/cache}"

TEXT=""
TEXT_FILE_INPUT=""
REF_WAV=""
OUT_WAV=""
TEMPERATURE="${TEMPERATURE:-1.0}"
GPT_COND_SEC="${GPT_COND_SEC:-30}"
MAX_REF_SEC="${MAX_REF_SEC:-60}"
DENOISE="${DENOISE:-0}"
TOP_P="${TOP_P:-}"
TOP_K="${TOP_K:-}"
LENGTH_PENALTY="${LENGTH_PENALTY:-}"
REPETITION_PENALTY="${REPETITION_PENALTY:-}"
SPEED="${SPEED:-}"
SPEAKER_MIX="${SPEAKER_MIX:-}"
EMOTION="${EMOTION:-}"
STYLE_PROMPT="${STYLE_PROMPT:-}"
STYLE_POSITION="${STYLE_POSITION:-prepend}"
PROMPT_PREFIX="${PROMPT_PREFIX:-}"
PROMPT_SUFFIX="${PROMPT_SUFFIX:-}"
TEXT_SPLIT="${TEXT_SPLIT:-}"
TEXT_SPLIT_SET=0
SEED="${SEED:-}"
GTP_MIN_LEN="${GTP_MIN_LEN:-}"
GTP_MAX_LEN="${GTP_MAX_LEN:-}"
GTP_MIN_AUDIO="${GTP_MIN_AUDIO:-}"
GTP_MAX_AUDIO="${GTP_MAX_AUDIO:-}"
DEBUG_SIGNATURE="${DEBUG_SIGNATURE:-0}"

mkdir -p "${CACHE_ROOT}/"{hf,torch,pip,xdg,tts}
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CACHE_ROOT}/pip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export TTS_HOME="${TTS_HOME:-${CACHE_ROOT}/tts}"

usage() {
  cat <<USAGE
Usage: $(basename "$0") --text "Hallo Welt" [--lang de] [--ref "/path/ref.wav"] [--out "/path/out.wav"]
Optionen:
  --text, -t        Zu sprechender Text (Pflicht, oder via STDIN)
  --from-file, -f   Datei mit Zeilen ("text||/individuelle_out.wav"). Setzt Batch-Modus voraus.
  --lang, -l        Sprachcode (default: de)
  --ref,  -r        Referenz-WAV (Voice-Conditioning). Falls leer: längstes Segment aus DATA_DIR/wavs/segments
  --out,  -o        Ausgabe-WAV (default: OUT_DIR/tts_YYYYmmdd_HHMMSS.wav bzw. Batch-Ableitung)
  --ckpt            Pfad zu latest_full (default: ${CKPT_DIR})
  --orig            Pfad zu Original-Assets (default: ${ORIG_DIR})
  --data            Pfad zu Dataset (default: ${DATA_DIR})
  --img             Docker-Image (default: ${IMG_XTTS})
  --temperature     Sampling-Temperatur (default: ${TEMPERATURE})
  --gpt-seconds     GPT conditioning length Sekunden (default: ${GPT_COND_SEC})
  --max-ref-seconds Maximale Referenzlänge Sekunden (default: ${MAX_REF_SEC})
  --denoise         1=ffmpeg Hochpass/Breitband-Filter anwenden (default: ${DENOISE})
  --top-p           Stochastisches Sampling Top-P (z.B. 0.85)
  --top-k           Stochastisches Sampling Top-K (z.B. 50)
  --length-penalty  Längen-Penalty (>1 länger, <1 kürzer)
  --repetition-penalty Wiederholungs-Penalty (größer = weniger Loops)
  --speed           Wiedergaberate (z.B. 0.95 für langsam, 1.05 schneller)
  --speaker-mix     Zwischen Referenzstimmen mischen (0..1, falls unterstützt)
  --emotion         Emotions-Keyword für kompatible Modelle (z.B. calm, excited)
  --style-prompt    Stil-/Stimmungs-Prompt, wird vor den Text gesetzt
  --style-position  prepend|append – Position für style-prompt (default: prepend)
  --prefix          Text voranstellen (z.B. kurze Regieanweisung)
  --suffix          Text anhängen (z.B. Stillemarker)
  --text-split      Text in Sätze teilen bevor Modell läuft (toggle)
  --no-text-split   Textsplitting deaktivieren
  --seed            Zufalls-Seed für deterministische Runs
  --gpt-min-len     Minimale GPT-Länge (Tokens) falls unterstützt
  --gpt-max-len     Maximale GPT-Länge (Tokens)
  --gpt-min-audio   Minimale Audio-Token-Anzahl
  --gpt-max-audio   Maximale Audio-Token-Anzahl
  --debug-signature 1 = verfügbare Inference-Parameter ausgeben
USAGE
  exit 1
}

# ------------------------------
# CLI-Args
# ------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --text|-t) TEXT="$2"; shift 2;;
    --from-file|-f) TEXT_FILE_INPUT="$2"; shift 2;;
    --lang|-l) TARGET_LANG="$2"; shift 2;;
    --ref|-r)  REF_WAV="$2"; shift 2;;
    --out|-o)  OUT_WAV="$2"; shift 2;;
    --ckpt)    CKPT_DIR="$2"; shift 2;;
    --orig)    ORIG_DIR="$2"; shift 2;;
    --data)    DATA_DIR="$2"; shift 2;;
    --img)     IMG_XTTS="$2"; shift 2;;
    --temperature) TEMPERATURE="$2"; shift 2;;
    --gpt-seconds) GPT_COND_SEC="$2"; shift 2;;
    --max-ref-seconds) MAX_REF_SEC="$2"; shift 2;;
    --denoise) DENOISE="$2"; shift 2;;
    --top-p) TOP_P="$2"; shift 2;;
    --top-k) TOP_K="$2"; shift 2;;
    --length-penalty) LENGTH_PENALTY="$2"; shift 2;;
    --repetition-penalty) REPETITION_PENALTY="$2"; shift 2;;
    --speed) SPEED="$2"; shift 2;;
    --speaker-mix) SPEAKER_MIX="$2"; shift 2;;
    --emotion) EMOTION="$2"; shift 2;;
    --style-prompt) STYLE_PROMPT="$2"; shift 2;;
    --style-position) STYLE_POSITION="$2"; shift 2;;
    --prefix) PROMPT_PREFIX="$2"; shift 2;;
    --suffix) PROMPT_SUFFIX="$2"; shift 2;;
    --text-split) TEXT_SPLIT=1; TEXT_SPLIT_SET=1; shift;;
    --no-text-split) TEXT_SPLIT=0; TEXT_SPLIT_SET=1; shift;;
    --seed) SEED="$2"; shift 2;;
    --gpt-min-len) GTP_MIN_LEN="$2"; shift 2;;
    --gpt-max-len) GTP_MAX_LEN="$2"; shift 2;;
    --gpt-min-audio) GTP_MIN_AUDIO="$2"; shift 2;;
    --gpt-max-audio) GTP_MAX_AUDIO="$2"; shift 2;;
    --debug-signature)
      if [[ $# -gt 1 && "${2:-}" != -* ]]; then
        DEBUG_SIGNATURE="$2"; shift 2
      else
        DEBUG_SIGNATURE=1; shift
      fi;;
    -h|--help) usage;;
    *) echo "Unbekannte Option: $1"; usage;;
  esac
done

# Textquelle bestimmen
if [[ -n "${TEXT_FILE_INPUT}" ]]; then
  [[ -f "${TEXT_FILE_INPUT}" ]] || { echo "Fehler: batch-Datei fehlt: ${TEXT_FILE_INPUT}"; exit 1; }
elif [[ -z "${TEXT}" && ! -t 0 ]]; then
  TEXT="$(cat)"
fi
if [[ -z "${TEXT}" && -z "${TEXT_FILE_INPUT}" ]]; then
  echo "Fehler: --text oder --from-file oder STDIN erforderlich."; usage
fi

if [[ -n "${TARGET_LANG}" ]]; then
  lang_lower="${TARGET_LANG%%.*}"
  lang_lower="${lang_lower//-/_}"
  lang_lower="${lang_lower,,}"
  if [[ "${lang_lower}" == *_* ]]; then
    TARGET_LANG="${lang_lower%%_*}"
  else
    TARGET_LANG="${lang_lower}"
  fi
fi
[[ -n "${TARGET_LANG}" ]] || TARGET_LANG="de"

STYLE_POSITION="${STYLE_POSITION,,}"
case "${STYLE_POSITION}" in
  append|suffix) STYLE_POSITION="append" ;;
  prepend|prefix|"") STYLE_POSITION="prepend" ;;
  *) STYLE_POSITION="prepend" ;;
esac

case "${TEXT_SPLIT}" in
  1|true|yes|on) TEXT_SPLIT=1 ;;
  0|false|no|off|"") TEXT_SPLIT=0 ;;
  *) TEXT_SPLIT=1 ;;
esac

case "${DENOISE}" in
  1|true|yes|on) DENOISE=1 ;;
  0|false|no|off|"") DENOISE=0 ;;
  *) DENOISE=1 ;;
esac

case "${DEBUG_SIGNATURE}" in
  1|true|yes|on) DEBUG_SIGNATURE=1 ;;
  0|false|no|off|"") DEBUG_SIGNATURE=0 ;;
  *) DEBUG_SIGNATURE=1 ;;
esac

# Pfade prüfen
MODEL_PTH="${CKPT_DIR}/model.pth"
CFG_JSON="${CKPT_DIR}/config.json"
[[ -f "${MODEL_PTH}" ]] || { echo "Fehlt: ${MODEL_PTH}"; exit 1; }
[[ -f "${CFG_JSON}" ]]  || { echo "Fehlt: ${CFG_JSON}"; exit 1; }
mkdir -p "${OUT_DIR}"
[[ -d "${DATA_DIR}" ]] || mkdir -p "${DATA_DIR}"
[[ -d "${ORIG_DIR}" ]] || mkdir -p "${ORIG_DIR}"

# Referenz automatisch wählen (längstes Segment), wenn nicht gesetzt
if [[ -z "${REF_WAV}" ]]; then
  declare -a REF_DIRS=()
  [[ -d "${DATA_DIR}/wavs/segments" ]] && REF_DIRS+=("${DATA_DIR}/wavs/segments")
  [[ -d "${DATA_DIR}/reference_wavs" ]] && REF_DIRS+=("${DATA_DIR}/reference_wavs")
  [[ -d "${DATA_DIR}" ]] && REF_DIRS+=("${DATA_DIR}")
  [[ -d "/mnt/fast_scratch/dataset/wavs/segments" ]] && REF_DIRS+=("/mnt/fast_scratch/dataset/wavs/segments")
  [[ -d "${CKPT_DIR}/reference_wavs" ]] && REF_DIRS+=("${CKPT_DIR}/reference_wavs")

  best=""; bestdur=0
  shopt -s nullglob
  for dir in "${REF_DIRS[@]}"; do
    for f in "${dir}"/*.wav; do
      dur=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$f" || echo 0)
      dur=${dur%.*}
      if [[ "${dur:-0}" -gt "${bestdur:-0}" ]]; then
        best="$f"; bestdur="$dur"
      fi
    done
  done
  [[ -n "${best}" ]] || { echo "Keine Referenz gefunden und --ref nicht gesetzt."; exit 1; }
  REF_WAV="${best}"
  echo ">> REF_WAV auto: ${REF_WAV} (~${bestdur}s)"
fi

# Referenz-Datei validieren
if [[ -d "${REF_WAV}" ]]; then
  echo "Fehler: --ref zeigt auf ein VERZEICHNIS, erwartet WAV-Datei: ${REF_WAV}"
  exit 1
fi
[[ -f "${REF_WAV}" ]] || { echo "Fehler: Referenz-WAV existiert nicht: ${REF_WAV}"; exit 1; }

# Out-Dateiname default (nur Single-Run)
if [[ -z "${OUT_WAV}" && -z "${TEXT_FILE_INPUT}" ]]; then
  ts="$(date +%F_%H%M%S)"
  OUT_WAV="${OUT_DIR}/tts_${ts}.wav"
fi

run_inference_job() {
  local job_text="$1"
  local target_path="$2"
  [[ -n "${job_text}" ]] || { echo "Skip: leerer Text" >&2; return 0; }

  local job_dir job_base tmp_text
  job_dir="$(dirname "${target_path}")"
  job_base="$(basename "${target_path}")"
  mkdir -p "${job_dir}"

  tmp_text="$(mktemp)"
  printf '%s' "${job_text}" > "${tmp_text}"

  set +e
  docker run --rm --gpus all \
    --shm-size=4g --ipc=host \
    -e LANG_CODE="${TARGET_LANG}" \
    -e OUT_BASENAME="${job_base}" \
    -e OUT_DIR_CONTAINER="/job_out" \
    -e INFER_TEMPERATURE="${TEMPERATURE}" \
    -e GPT_COND_SEC="${GPT_COND_SEC}" \
    -e MAX_REF_SEC="${MAX_REF_SEC}" \
    -e INFER_TOP_P="${TOP_P}" \
    -e INFER_TOP_K="${TOP_K}" \
    -e INFER_LENGTH_PENALTY="${LENGTH_PENALTY}" \
    -e INFER_REPETITION_PENALTY="${REPETITION_PENALTY}" \
    -e INFER_SPEED="${SPEED}" \
    -e INFER_SPEAKER_MIX="${SPEAKER_MIX}" \
    -e INFER_EMOTION="${EMOTION}" \
    -e STYLE_PROMPT_ENV="${STYLE_PROMPT}" \
    -e STYLE_POSITION_ENV="${STYLE_POSITION}" \
    -e STYLE_PREFIX_ENV="${PROMPT_PREFIX}" \
    -e STYLE_SUFFIX_ENV="${PROMPT_SUFFIX}" \
    -e INFER_TEXT_SPLIT="${TEXT_SPLIT}" \
    -e INFER_TEXT_SPLIT_SET="${TEXT_SPLIT_SET}" \
    -e INFER_SEED="${SEED}" \
    -e INFER_GPT_MIN_LEN="${GTP_MIN_LEN}" \
    -e INFER_GPT_MAX_LEN="${GTP_MAX_LEN}" \
    -e INFER_GPT_MIN_AUDIO="${GTP_MIN_AUDIO}" \
    -e INFER_GPT_MAX_AUDIO="${GTP_MAX_AUDIO}" \
    -e INFER_DEBUG_SIGNATURE="${DEBUG_SIGNATURE}" \
    -e APPLY_DENOISE="${DENOISE}" \
    -e HF_HOME=/store_cache/hf \
    -e TRANSFORMERS_CACHE=/store_cache/hf/transformers \
    -e TORCH_HOME=/store_cache/torch \
    -e PIP_CACHE_DIR=/store_cache/pip \
    -e XDG_CACHE_HOME=/store_cache/xdg \
    -e TTS_HOME=/store_cache/tts \
    -v "${CKPT_DIR}":/ckpt:ro \
    -v "${ORIG_DIR}":/orig:ro \
    -v "${DATA_DIR}":/data:ro \
    -v "${CACHE_ROOT}":/store_cache \
    -v "${job_dir}":/job_out \
    -v "${tmp_text}":/tmp/text.txt:ro \
    -v "${REF_WAV}":/host_ref.wav:ro \
    "${IMG_XTTS}" bash -lc '
set -e
ffmpeg -y -i "/host_ref.wav" -ar 24000 -ac 1 -t "${MAX_REF_SEC}" "/tmp/ref_24k_mono_60s.wav" >/dev/null 2>&1

python - <<PY
import os, json, torch, soundfile as sf, inspect, random
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

ckpt_dir = "/ckpt"
orig_dir = "/orig"


def env_float(name, default=None):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_int(name, default=None):
    val = env_float(name, default)
    if val is None:
        return default
    try:
        return int(round(val))
    except (TypeError, ValueError):
        return default


text     = open("/tmp/text.txt","r",encoding="utf-8").read().strip()
lang_raw = os.environ.get("LANG_CODE","de")
lang     = (lang_raw.split(".")[0].replace("-", "_").split("_")[0] or "de").lower()
out_dir  = os.environ.get("OUT_DIR_CONTAINER","/out")
out_wav  = os.path.join(out_dir, os.path.basename(os.environ.get("OUT_BASENAME","tts.wav")))
ref_wav  = "/tmp/ref_24k_mono_60s.wav"

style_prompt = os.environ.get("STYLE_PROMPT_ENV", "").strip()
style_position = os.environ.get("STYLE_POSITION_ENV", "prepend").strip().lower()
prefix = os.environ.get("STYLE_PREFIX_ENV", "").strip()
suffix = os.environ.get("STYLE_SUFFIX_ENV", "").strip()

if style_prompt:
    if style_position == "append":
        text = f"{text} {style_prompt}".strip()
    else:
        text = f"{style_prompt}. {text}".strip()
if prefix:
    text = f"{prefix} {text}".strip()
if suffix:
    text = f"{text} {suffix}".strip()

seed_env = os.environ.get("INFER_SEED")
if seed_env:
    try:
        seed_val = int(seed_env)
    except ValueError:
        seed_val = None
    if seed_val is not None:
        random.seed(seed_val)
        try:
            import numpy as np
        except ImportError:
            np = None
        else:
            np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_val)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except AttributeError:
            pass

temperature = env_float("INFER_TEMPERATURE", 0.7)
gpt_cond_len = env_int("GPT_COND_SEC", 30)
max_ref_length = env_int("MAX_REF_SEC", 60)
text_split_flag = os.environ.get("INFER_TEXT_SPLIT", "0").lower() in {"1","true","yes","on"}
text_split_requested = os.environ.get("INFER_TEXT_SPLIT_SET", "0").lower() in {"1","true","yes","on"}
top_p = env_float("INFER_TOP_P")
top_k = env_int("INFER_TOP_K")
length_penalty = env_float("INFER_LENGTH_PENALTY")
repetition_penalty = env_float("INFER_REPETITION_PENALTY")
speed = env_float("INFER_SPEED")
speaker_mix = env_float("INFER_SPEAKER_MIX")
emotion = os.environ.get("INFER_EMOTION", "").strip()
gpt_min_len = env_int("INFER_GPT_MIN_LEN")
gpt_max_len = env_int("INFER_GPT_MAX_LEN")
gpt_min_audio = env_int("INFER_GPT_MIN_AUDIO")
gpt_max_audio = env_int("INFER_GPT_MAX_AUDIO")
debug_signature = os.environ.get("INFER_DEBUG_SIGNATURE", "0").lower() in {"1","true","yes","on"}

cfg_path = os.path.join(ckpt_dir, "config.json")
model_pth= os.path.join(ckpt_dir, "model.pth")
checkpoint_dir = ckpt_dir

cfg = XttsConfig(); cfg.load_json(cfg_path)
model_args = getattr(cfg, "model_args", {}) or {}

def localize(key, fallback_name):
    value = model_args.get(key)
    candidates = []
    if value:
        candidates.append(value)
        candidates.append(os.path.join(ckpt_dir, os.path.basename(value)))
    candidates.append(os.path.join(ckpt_dir, fallback_name))
    candidates.append(os.path.join(orig_dir, fallback_name))
    for cand in candidates:
        if cand and os.path.isfile(cand):
            model_args[key] = cand
            return
    model_args.pop(key, None)

localize("tokenizer_file", "tokenizer.json")
localize("merges_file", "merges.txt")
cfg.model_args = model_args

model = Xtts.init_from_config(cfg)

vocab_path = None
for cand in (
    model_args.get("tokenizer_file"),
    os.path.join(ckpt_dir, "vocab.json"),
    os.path.join(orig_dir, "vocab.json"),
):
    if cand and os.path.isfile(cand) and cand.endswith("vocab.json"):
        vocab_path = cand
        break

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

available_params = set(inspect.signature(model.inference).parameters.keys())
if debug_signature:
    print(">> inference accepts:", ", ".join(sorted(available_params)))

infer_kwargs = {}

def assign(value, *names):
    if value is None:
        return
    for name in names:
        if name in available_params:
            infer_kwargs[name] = value
            return


def assign_bool(flag, *names):
    if not names:
        return
    if flag is None:
        return
    for name in names:
        if name in available_params:
            infer_kwargs[name] = bool(flag)
            return


def assign_str(value, *names):
    if not value:
        return
    for name in names:
        if name in available_params:
            infer_kwargs[name] = value
            return


assign(temperature, "temperature", "gpt_sampling_temperature")
assign(top_p, "gpt_sampling_topp", "gpt_sampling_top_p", "top_p")
assign(top_k, "gpt_sampling_topk", "gpt_sampling_top_k", "top_k")
assign(length_penalty, "length_penalty")
assign(repetition_penalty, "repetition_penalty")
assign(speed, "speed")
assign(speaker_mix, "gpt_speaker_mix", "speaker_mix", "speaker_weight")
assign(gpt_min_len, "gpt_min_length", "gpt_min_len")
assign(gpt_max_len, "gpt_max_length", "gpt_max_len")
assign(gpt_min_audio, "gpt_min_audio_tokens", "gpt_min_audio_length", "min_audio_tokens")
assign(gpt_max_audio, "gpt_max_audio_tokens", "gpt_max_audio_length", "max_audio_tokens")
if text_split_requested:
    assign_bool(text_split_flag, "enable_text_splitting")
assign_str(emotion, "emotion", "style", "speaker_style", "speaker_emotion")

if debug_signature and infer_kwargs:
    print(">> using kwargs:", infer_kwargs)

gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
    audio_path=[ref_wav],
    gpt_cond_len=gpt_cond_len,
    max_ref_length=max_ref_length
)

with torch.no_grad():
    wav = model.inference(
        text=text,
        language=lang,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        **infer_kwargs
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
  local status=$?
  rm -f "${tmp_text}"
  set -e
  if [[ ${status} -ne 0 ]]; then
    echo "❌ Inference fehlgeschlagen für ${target_path}" >&2
    return ${status}
  fi

  if [[ "${DENOISE}" == "1" ]]; then
    local denoise_tmp="${target_path%.wav}.denoise_tmp.wav"
    set +e
    ffmpeg -y -i "${target_path}" -af "highpass=f=60,afftdn=nr=12:nt=w" "${denoise_tmp}" >/dev/null 2>&1
    local denoise_status=$?
    set -e
    if [[ ${denoise_status} -eq 0 && -f "${denoise_tmp}" ]]; then
      mv "${denoise_tmp}" "${target_path}"
    else
      rm -f "${denoise_tmp}" 2>/dev/null || true
      echo "WARN: Denoise fehlgeschlagen für ${target_path}" >&2
    fi
  fi

  echo "✅ Output: ${target_path}"
  return 0
}

if [[ -n "${TEXT_FILE_INPUT}" ]]; then
  batch_ts="$(date +%F_%H%M%S)"
  count=0
  while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
    line="${raw_line%%$'\r'}"
    line="${line#${line%%[![:space:]]*}}"
    line="${line%${line##*[![:space:]]}}"
    [[ -z "${line}" || "${line}" == \#* ]] && continue

    text_part="${line}"
    custom_out=""
    if [[ "${line}" == *"||"* ]]; then
      text_part="${line%%||*}"
      custom_out="${line#*||}"
    fi
    text_part="${text_part#${text_part%%[![:space:]]*}}"
    text_part="${text_part%${text_part##*[![:space:]]}}"
    custom_out="${custom_out#${custom_out%%[![:space:]]*}}"
    custom_out="${custom_out%${custom_out##*[![:space:]]}}"
    [[ -z "${text_part}" ]] && continue

    ((count+=1))
    out_path=""
    if [[ -n "${custom_out}" ]]; then
      if [[ "${custom_out}" != /* ]]; then
        out_path="${OUT_DIR}/${custom_out}"
      else
        out_path="${custom_out}"
      fi
    elif [[ -n "${OUT_WAV}" ]]; then
      if [[ ${count} -eq 1 ]]; then
        out_path="${OUT_WAV}"
      else
        if [[ "${OUT_WAV}" == *.* ]]; then
          base="${OUT_WAV%.*}"
          ext="${OUT_WAV##*.}"
          out_path="${base}_${count}.${ext}"
        else
          out_path="${OUT_WAV}_${count}"
        fi
      fi
    else
      out_path="${OUT_DIR}/tts_batch_${batch_ts}_${count}.wav"
    fi

    [[ "${out_path}" == *.wav ]] || out_path+=".wav"

    run_inference_job "${text_part}" "${out_path}"
  done < "${TEXT_FILE_INPUT}"

  echo "✅ Batch fertig: ${count} Dateien"
  exit 0
fi

run_inference_job "${TEXT}" "${OUT_WAV}"
