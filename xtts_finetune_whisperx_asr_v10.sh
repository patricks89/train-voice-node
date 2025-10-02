#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# XTTS Finetune + WhisperX ASR — v8-kompatible Pipeline (mit Verbesserungen)
#
# Was das Script tut (Schritt für Schritt):
#  1) Umfeld prüfen (docker/ffmpeg/… vorhanden?), Mounts checken.
#  2) **RAM-Checkpoint (/mnt/ram_ckpt) leeren** (immer) und **Fast-Scratch
#     (/mnt/fast_scratch) leeren** (immer) – danach benötigte Ordner neu anlegen.
#  3) Docker-Images „wie v8“ handhaben:
#       - Wenn `whisper-tools` / `xtts-finetune:cu121` fehlen → lokal bauen
#         aus `whisperx.Dockerfile` / `xtts-finetune.Dockerfile`.
#       - Kein Registry/Pull nötig.
#  4) Rohdaten (mp4/wav) vom NAS nach Scratch rsyncen.
#  5) MP4 → WAV transkodieren (parallel, SR konfigurierbar).
#  6) WhisperX (im Container) transkribiert WAVs, optional Dialekt-Rewrite
#     schon „raw“. Resultat: JSON-Segmente.
#  7) „Smart Chunking“ (Längenlimits), dann Segment-Cutting (ffmpeg) in WAV-Schnipsel.
#  8) Metadata-CSV (coqui-Format: `audio_file|text`) erstellen (Train/Eval).
#  9) Training (im Container) – Checkpoints nach **RAM** (tmpfs).
# 10) Letzten Lauf **aus RAM nach STORE spiegeln**, „latest_full“ exportieren
#     (model.pth + config.json + Tokenizer von HF).
# 11) Optionaler Cleanup am Ende (Scratch + RAM leeren).
#
# Beispiele (Aufruf):
#   # Standard:
#   ./xtts_finetune_whisperx_asr_v8plus.sh
#
#   # Andere Quelle & SR, 5 Epochen, größere Batch:
#   NAS_SRC=/mnt/nas/Allgemein/VoiceClone/ABC \
#   SR=24000 EPOCHS=5 BATCH_SIZE=2 GRAD_ACCUM=8 \
#   ./xtts_finetune_whisperx_asr_v8plus.sh
#
#   # Force Sprache auf 'de', Whisper large-v3, 300er SAVE_STEP:
#   WHISPER_FORCE_LANG=de WHISPER_MODEL=large-v3 SAVE_STEP=300 \
#   ./xtts_finetune_whisperx_asr_v8plus.sh
#
#   # Mit HF-Token (für private Repos/Coqui-Tokenizer):
#   HF_TOKEN=hf_xxx ./xtts_finetune_whisperx_asr_v8plus.sh
#
# Wichtige Variablen (alle per ENV überschreibbar):
#   SCRATCH_ROOT   : Flüchtiges Arbeitsverz. (mp4/wav/tmp)       (Default: /mnt/fast_scratch)
#   STORE_ROOT     : Persistenter Bereich (Checkpoints, Caches)   (Default: /mnt/fast_store)
#   RAM_CKPT_ROOT  : tmpfs für Checkpoints während Trainings      (Default: /mnt/ram_ckpt)
#   NAS_SRC        : Quelle mit mp4/wav (z.B. CIFS-Automount)
#   HF_TOKEN       : HuggingFace-Token (für priv. Modelle/Coqui)
#   WHISPER_MODEL  : z.B. large-v3
#   WHISPER_FORCE_LANG : ""=Auto, sonst z.B. "de"
#   VAD_METHOD     : none|silero (nur falls WhisperX-Version VAD kennt)
#   DIALECT_HINT   : Prompt-Bias (wird mit LEXICON_FILE erweitert)
#   LEXICON_FILE   : Persistente Dialekt-Liste (optional)
#   LANG_TAG       : Sprach-Tag fürs Training (z.B. "de")
#   SR             : WAV Sample Rate (z.B. 22050/24000)
#   MAX_TXT_CHARS  : max. Text-Länge pro Sample
#   MIN_DUR/TARGET_DUR/MAX_DUR : Segmentlängen (Sek.)
#   MAX_AUDIO_SAMPLES : max. Audiosamples fürs Training
#   DIALECT_REWRITE : 1 = Rewrite beim CSV-Bau (post)
#   RAW_DIALECT_REWRITE : 1 = Rewrite schon in RAW-JSON (pre)
#   REWRITE_FILE   : Zusätzliche Regeln (TSV regex|repl)
#   EPOCHS         : #Epochen (Default 10)
#   BATCH_SIZE     : Batch-Größe (Default 1)
#   GRAD_ACCUM     : Gradient Accumulation (Default 16)
#   LR             : Lernrate (Default 5e-6)
#   WEIGHT_DECAY   : Weight Decay (Default 1e-2)
#   SAVE_STEP      : Speichern alle N Steps
#   USE_STORE_CACHE: 1 = persistente Caches in STORE_ROOT (empfohlen)
#   CLEAN_AFTER_TRAIN : 1 = Am Ende Scratch & RAM leeren
#
# Docker:
#   whisper-tools        → gebaut aus ./whisperx.Dockerfile (falls fehlt)
#   xtts-finetune:cu121  → gebaut aus ./xtts-finetune.Dockerfile (falls fehlt)
#
###############################################################################

# ======= Konfiguration (Defaults) =======
SCRATCH_ROOT="${SCRATCH_ROOT:-/mnt/fast_scratch}"
STORE_ROOT="${STORE_ROOT:-/mnt/fast_store}"
RAM_CKPT_ROOT="${RAM_CKPT_ROOT:-/mnt/ram_ckpt}"

SCRATCH_DATA="${SCRATCH_ROOT}/dataset"
SCRATCH_WAVS="${SCRATCH_DATA}/wavs"
SCRATCH_TMP="${SCRATCH_ROOT}/tmp"

export SCRATCH_DATA SCRATCH_WAVS SCRATCH_TMP

REF_EXPORT_BASE="${REF_EXPORT_BASE:-/mnt/fast_store/dataset_ref/reference_wavs}"

STORE_CKPT="${STORE_ROOT}/checkpoints"
STORE_CACHE="${STORE_ROOT}/cache"

export DOCKER_TMPDIR="${DOCKER_TMPDIR:-${SCRATCH_ROOT}/docker-tmp}"

# User-vars / Tokens
HF_TOKEN="${HF_TOKEN:-hf_}"
NAS_SRC="${NAS_SRC:-/mnt/nas/Allgemein/VoiceClone/standpunkte}"

# WhisperX / ASR
WHISPER_MODEL="${WHISPER_MODEL:-large-v3}"
WHISPER_FORCE_LANG="${WHISPER_FORCE_LANG:-}"
VAD_METHOD="${VAD_METHOD:-none}"

DIALECT_HINT="${DIALECT_HINT:-Allegra, Grüezi, Chur, Bündner, nöd, nüüt, öppis, chli, lueg, tuet, wos, dä, d’, miar, bisch, goht, gits, schaffa, gäge, nid, üs, äbe, gaht, gäll.}"
LEXICON_FILE="${LEXICON_FILE:-${STORE_ROOT}/dialect_lexicon.txt}"

# Segmentierung / Limits
LANG_TAG="${LANG_TAG:-de}"
SR="${SR:-22050}"
MAX_TXT_CHARS="${MAX_TXT_CHARS:-180}"
MIN_DUR="${MIN_DUR:-0.6}"
TARGET_DUR="${TARGET_DUR:-5.0}"
MAX_DUR="${MAX_DUR:-8.0}"
MAX_AUDIO_SAMPLES="${MAX_AUDIO_SAMPLES:-240000}"

DIALECT_REWRITE="${DIALECT_REWRITE:-1}"
RAW_DIALECT_REWRITE="${RAW_DIALECT_REWRITE:-0}"
REWRITE_FILE="${REWRITE_FILE:-${STORE_ROOT}/dialect_rewrite.tsv}"

# Training-HP
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
LR="${LR:-5e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
SAVE_STEP="${SAVE_STEP:-300}"

# Caches & Cleanup
USE_STORE_CACHE="${USE_STORE_CACHE:-1}"
CLEAN_AFTER_TRAIN="${CLEAN_AFTER_TRAIN:-1}"

# Docker-Images / Dockerfiles (wie v8 – lokal bauen)
IMG_WHISPER="${IMG_WHISPER:-whisper-tools}"
IMG_XTTS="${IMG_XTTS:-xtts-finetune:cu121}"
DOCKERFILE_WHISPER="${DOCKERFILE_WHISPER:-whisperx.Dockerfile}"
DOCKERFILE_XTTS="${DOCKERFILE_XTTS:-xtts-finetune.Dockerfile}"

# ======= Utilities =======
say(){ echo "==> $*"; }
die(){ echo "ERROR: $*" >&2; exit 1; }
on_err(){ echo; echo "💥 Abbruch – siehe Meldung oben."; echo; }

trap on_err ERR

# Optionales Debug
if [[ "${DEBUG:-}" == "1" ]]; then set -x; fi

ensure_bins(){
  local miss=()
  for b in docker ffmpeg ffprobe python3 rsync awk sed grep stat findmnt timeout xargs df; do
    command -v "$b" >/dev/null 2>&1 || miss+=("$b")
  done
  if (( ${#miss[@]} > 0 )); then
    die "Benötigte Tools fehlen: ${miss[*]}"
  fi
}

ensure_dir(){ mkdir -p -- "$1"; }

ensure_mount(){
  # Stumpfer Check: ist der Pfad ein Mountpoint? Falls nicht, versuche mount -a
  local path="$1"
  if ! mountpoint -q -- "$path"; then
    say "$path ist nicht gemountet – versuche 'mount -a' …"
    mount -a || true
    mountpoint -q -- "$path" || die "$path ist nicht gemountet. Bitte fstab/Mount prüfen."
  fi
}

safe_empty_dir(){
  # Löscht *Inhalte* eines Verzeichnisses sicher (nicht das Verzeichnis selbst)
  local dir="$1"; shift
  [[ -d "$dir" ]] || return 0
  shopt -s dotglob nullglob
  for p in "$dir"/* "$dir"/.[!.]* "$dir"/..?*; do
    [[ -e "$p" ]] || continue
    rm -rf --one-file-system -- "$p"
  done
  shopt -u dotglob nullglob
}

ensure_docker_image_build_or_use(){
  # Wenn Image fehlt → lokal bauen (wie v8). Keine Registry erforderlich.
  local tag="$1"
  local dockerfile="$2"
  if ! docker image inspect "$tag" >/dev/null 2>&1; then
    [[ -f "$dockerfile" ]] || die "Dockerfile '$dockerfile' fehlt (benötigt für $tag)."
    say "Baue Docker-Image $tag aus $dockerfile …"
    DOCKER_BUILDKIT=1 docker build -t "$tag" -f "$dockerfile" . > /dev/null
  fi
}

bytes_to_human(){
  awk -v b="$1" 'BEGIN{
    hum[1024^4]="T";hum[1024^3]="G";hum[1024^2]="M";hum[1024]="K";
    for (x=1024^4; x>=1024; x/=1024) if (b>=x){ printf "%.1f%s\n", b/x, hum[x]; exit }
    print b "B"
  }'
}

# ======= PREP =======
ensure_bins

# Basisverzeichnisse
for p in "$SCRATCH_ROOT" "$STORE_ROOT" "$RAM_CKPT_ROOT" \
         "$SCRATCH_WAVS" "$SCRATCH_TMP" "$STORE_CKPT" \
         "$DOCKER_TMPDIR" "$STORE_CACHE"; do
  ensure_dir "$p"
done

# Mounts prüfen (hilfreich bei systemd automount)
ensure_mount "$SCRATCH_ROOT"
ensure_mount "$STORE_ROOT"
ensure_mount "$RAM_CKPT_ROOT"

# **Immer** zu Beginn leeren:
say "Leere RAM-Checkpoints ($RAM_CKPT_ROOT) …"
safe_empty_dir "$RAM_CKPT_ROOT"

say "Leere Fast-Scratch ($SCRATCH_ROOT) …"
safe_empty_dir "$SCRATCH_ROOT"
# danach Mindeststruktur wieder anlegen
ensure_dir "$SCRATCH_WAVS"
ensure_dir "$SCRATCH_TMP"
ensure_dir "$DOCKER_TMPDIR"

# Persistente Caches auf STORE (empfohlen)
if [[ "$USE_STORE_CACHE" -eq 1 ]]; then
  ensure_dir "${STORE_CACHE}/hf"
  ensure_dir "${STORE_CACHE}/torch"
  ensure_dir "${STORE_CACHE}/pip"
  ensure_dir "${STORE_CACHE}/xdg"
  ensure_dir "${STORE_CACHE}/tts"
  export HF_HOME="${HF_HOME:-${STORE_CACHE}/hf}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
  export TORCH_HOME="${TORCH_HOME:-${STORE_CACHE}/torch}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${STORE_CACHE}/pip}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${STORE_CACHE}/xdg}"
  export TTS_HOME="${TTS_HOME:-${STORE_CACHE}/tts}"
else
  ensure_dir "${SCRATCH_ROOT}/cache/hf"
  ensure_dir "${SCRATCH_ROOT}/cache/torch"
  ensure_dir "${SCRATCH_ROOT}/cache/pip"
  ensure_dir "${SCRATCH_ROOT}/cache/xdg"
  ensure_dir "${SCRATCH_ROOT}/cache/tts"
  export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/cache/hf}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
  export TORCH_HOME="${TORCH_HOME:-${SCRATCH_ROOT}/cache/torch}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SCRATCH_ROOT}/cache/pip}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_ROOT}/cache/xdg}"
  export TTS_HOME="${TTS_HOME:-${SCRATCH_ROOT}/cache/tts}"
fi

# Docker-Images wie v8 behandeln (lokal bauen falls fehlen)
ensure_docker_image_build_or_use "$IMG_WHISPER" "$DOCKERFILE_WHISPER"
ensure_docker_image_build_or_use "$IMG_XTTS"    "$DOCKERFILE_XTTS"

# NAS kurz „anticken“ (triggert systemd.automount, meckert aber nicht wenn leer)
if [[ -n "${NAS_SRC}" ]]; then
  say "Prüfe NAS-Quelle (${NAS_SRC}) …"
  timeout 8s bash -lc "stat -t '${NAS_SRC}' >/dev/null 2>&1 || ls -ld '${NAS_SRC}' >/dev/null 2>&1" || true
fi

# ======= ROHDATEN SYNC (mp4/wav) → SCRATCH =======
say "Synchronisiere Rohdaten (Audio/Video) → Scratch …"
ensure_dir "$SCRATCH_DATA"
rsync -av --delete --include="*/" \
  --include="*.[Mm][Pp]4" --include="*.[Mm][Kk][Vv]" --include="*.[Mm][Oo][Vv]" --include="*.[Aa][Vv][Ii]" --include="*.[Ww][Ee][Bb][Mm]" \
  --include="*.[Mm][Pp]3" --include="*.[Ff][Ll][Aa][Cc]" --include="*.[Mm]4[Aa]" --include="*.[Oo][Gg][Gg]" --include="*.[Aa][Aa][Cc]" \
  --include="*.[Ww][Aa][Vv]" \
  --exclude="*" \
  "${NAS_SRC}/" "${SCRATCH_DATA}/"

# Frischer Reset der Segment-Outputs
rm -rf "${SCRATCH_DATA}/segments_json_raw" "${SCRATCH_DATA}/segments_json" "${SCRATCH_WAVS}/segments"
ensure_dir "${SCRATCH_DATA}/segments_json_raw"
ensure_dir "${SCRATCH_DATA}/segments_json"
ensure_dir "${SCRATCH_WAVS}/segments"

# ======= AUDIO/VIDEO -> WAV @ SR mono =======
say "Transkodieren Audio/Video → WAV (SR=${SR}) …"
python3 - <<'PY'
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

scr_data = os.environ["SCRATCH_DATA"]
dest = os.environ["SCRATCH_WAVS"]
sr = os.environ.get("SR", "22050")

allowed_exts = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm",
    ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wav"
}

dest_abs = os.path.abspath(dest)
scr_abs = os.path.abspath(scr_data)

def sanitize(stem: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    cleaned = cleaned.strip("._-") or "audio"
    return cleaned[:255]

def make_name(rel_path: str, used: set) -> str:
    stem = os.path.splitext(rel_path)[0]
    stem = stem.replace(os.sep, "__")
    base = sanitize(stem)
    candidate = base
    idx = 1
    lowered = candidate.lower()
    while lowered in used:
        idx += 1
        candidate = f"{base}_{idx}"
        lowered = candidate.lower()
    used.add(lowered)
    return candidate

used_names = set()
jobs = []

if os.path.isdir(dest_abs):
    for entry in os.listdir(dest_abs):
        if entry.lower().endswith(".wav"):
            used_names.add(os.path.splitext(entry)[0].lower())

for root, dirs, files in os.walk(scr_abs):
    root_abs = os.path.abspath(root)
    if root_abs == dest_abs or root_abs.startswith(dest_abs + os.sep):
        dirs[:] = []
        continue
    for name in files:
        ext = os.path.splitext(name)[1].lower()
        if ext not in allowed_exts:
            continue
        src = os.path.join(root_abs, name)
        rel = os.path.relpath(src, scr_abs)
        wav_name = make_name(rel, used_names)
        out_path = os.path.join(dest_abs, wav_name + ".wav")
        jobs.append((src, out_path))

if not jobs:
    print(">> keine Audio/Video-Dateien gefunden.")
    raise SystemExit(0)

os.makedirs(dest_abs, exist_ok=True)

workers = min(8, max(1, os.cpu_count() or 2))
failures = []

def convert(args):
    src, out_path = args
    cmd = ["ffmpeg", "-y", "-i", src, "-ar", sr, "-ac", "1", "-vn", out_path]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode != 0 or not os.path.isfile(out_path):
        raise RuntimeError(src)

with ThreadPoolExecutor(max_workers=workers) as pool:
    futures = [pool.submit(convert, job) for job in jobs]
    for fut in futures:
        try:
            fut.result()
        except Exception as exc:
            src = getattr(exc, "args", [None])[0]
            failures.append(str(src) if src else "unknown")

if failures:
    raise SystemExit(f"ffmpeg-Fehler bei {len(failures)} Dateien: {failures[:5]}")

print(f">> transkodiert: {len(jobs)} Dateien → {dest_abs}")
PY

# ======= TRANSKRIPTION (WhisperX) =======
if [[ -f "${LEXICON_FILE}" ]]; then
  DIALECT_HINT="${DIALECT_HINT} $(tr $'\n' ' ' < "${LEXICON_FILE}")"
fi

CACHE_MNT="-v ${STORE_CACHE}:/store_cache"
if [[ "$USE_STORE_CACHE" -eq 0 ]]; then
  CACHE_MNT="-v ${SCRATCH_ROOT}/cache:/store_cache"
fi

say "Starte WhisperX (ASR) …"
docker run --rm --gpus all \
  --shm-size=8g --ipc=host \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e WHISPER_FORCE_LANG="${WHISPER_FORCE_LANG}" \
  -e WHISPER_MODEL="${WHISPER_MODEL}" \
  -e DIALECT_HINT="${DIALECT_HINT}" \
  -e VAD_METHOD="${VAD_METHOD}" \
  -e RAW_DIALECT_REWRITE="${RAW_DIALECT_REWRITE}" \
  -e HF_HOME=/store_cache/hf \
  -e TRANSFORMERS_CACHE=/store_cache/hf/transformers \
  -e TORCH_HOME=/store_cache/torch \
  -e PIP_CACHE_DIR=/store_cache/pip \
  -e XDG_CACHE_HOME=/store_cache/xdg \
  -v "${SCRATCH_DATA}":/data \
  ${CACHE_MNT} \
  -v "${SCRATCH_TMP}":/tmp \
  "${IMG_WHISPER}" bash -lc '
set -e
python - <<PY
import os, glob, json, torch, inspect, re
import whisperx

root="/data"
wavs=sorted(glob.glob(os.path.join(root,"wavs","*.wav")))
seg_dir=os.path.join(root,"segments_json_raw"); os.makedirs(seg_dir, exist_ok=True)

device="cuda" if torch.cuda.is_available() else "cpu"
force_lang=os.environ.get("WHISPER_FORCE_LANG","").strip() or None
model_name=os.environ.get("WHISPER_MODEL","large-v3")
dialect_hint=os.environ.get("DIALECT_HINT","")
vad_method=os.environ.get("VAD_METHOD","none")
raw_rw = int(os.environ.get("RAW_DIALECT_REWRITE","0"))

dialect_instr=("Schreibe konsequent im Schweizerdeutsch (Bündnerdialekt), "
               "verwende Dialektorthographie und KEIN Hochdeutsch.")
full_hint=(dialect_instr+" "+dialect_hint).strip()

try:
    from whisperx.asr import TranscriptionOptions
    base_asr_opts={"initial_prompt": full_hint, "temperature": 0.5, "beam_size": 1, "best_of": 1,
                   "condition_on_previous_text": False, "suppress_tokens": [-1]}
    sig=inspect.signature(TranscriptionOptions); allowed=set(sig.parameters.keys())
    asr_opts={k:v for k,v in base_asr_opts.items() if k in allowed and v is not None}
except Exception:
    asr_opts={"initial_prompt": full_hint, "condition_on_previous_text": False}

def load(compute):
    m = whisperx.load_model(model_name, device, compute_type=compute,
                            asr_options=asr_opts, vad_options={"method": vad_method})
    for attr in ("vad","vad_pipeline","vad_options"):
        try:
            if hasattr(m, attr): setattr(m, attr, None)
        except Exception: pass
    return m

try: model=load("float16")
except Exception: model=load("int8_float16")

rules=[(r"\\bund\\b","u"),(r"\\bUnd\\b","U"),(r"\\bich\\b","i"),(r"\\bIch\\b","I"),
       (r"\\bnicht\\b","nid"),(r"\\bNicht\\b","Nid"),(r"\\bklein\\b","chli"),(r"\\bKlein\\b","Chli"),
       (r"\\bwirklich\\b","würkli"),(r"\\bWirklich\\b","Würkli"),(r"\\bauch\\b","au"),
       (r"\\bgeht\\b","goht"),(r"\\bgehts\\b","gohts"),(r"ß","ss"),(r"Grüß","Grüez")]
def rewrite_dialect(t:str)->str:
    for pat,repl in rules: t=re.sub(pat,repl,t)
    return re.sub(r"\\s+"," ",t).strip()

total=0
for w in wavs:
    out=os.path.join(seg_dir, os.path.basename(w)+".json")
    audio=whisperx.load_audio(w)
    res=model.transcribe(audio, batch_size=6, language=force_lang or "de", task="transcribe")
    segs=[]
    for s in (res.get("segments") or []):
        txt=(s.get("text") or "").strip()
        st=float(s.get("start",0.0)); et=float(s.get("end",0.0))
        if not txt or not (et>st): continue
        if raw_rw: txt=rewrite_dialect(txt)
        segs.append({"start":st,"end":et,"text":txt})
    json.dump({"audio":os.path.basename(w),"segments":segs}, open(out,"w",encoding="utf-8"), ensure_ascii=False)
    total+=len(segs)
print(">> transcription raw segs:", total)
PY
'

# ======= SMART CHUNKING =======
python3 - <<'PY'
import os, json, glob, re
root=os.environ["SCRATCH_DATA"]
raw_dir=os.path.join(root,"segments_json_raw")
out_dir=os.path.join(root,"segments_json"); os.makedirs(out_dir, exist_ok=True)
MAX_CHARS=int(os.environ.get("MAX_TXT_CHARS","180"))
MIN_DUR=float(os.environ.get("MIN_DUR","0.6"))
TARGET=float(os.environ.get("TARGET_DUR","5.0"))
MAX_DUR=float(os.environ.get("MAX_DUR","8.0"))

def cut_points(text):
    pts=[m.end() for m in re.finditer(r'[\.!\?,;:]\s', text)]
    return pts or [len(text)//2]

def flush(buf,out_list):
    if not buf: return
    st=buf[0]["start"]; et=buf[-1]["end"]
    txt=" ".join(x["text"] for x in buf).strip()
    if not txt: return
    if len(txt)>MAX_CHARS or (et-st)>MAX_DUR:
        cps=cut_points(txt); mid=cps[len(cps)//2]
        left=txt[:mid].strip(); right=txt[mid:].strip()
        dur=et-st; etm=st + dur*(len(left)/max(1,len(txt)))
        if len(left)>=5:  out_list.append({"start":st,"end":etm,"text":left})
        if len(right)>=5: out_list.append({"start":etm,"end":et,"text":right})
    else:
        out_list.append({"start":st,"end":et,"text":txt})

def process(segs):
    acc=[]; out=[]; cur_len=0.0; cur_chars=0
    for s in segs:
        st=float(s["start"]); et=float(s["end"]); txt=(s["text"] or "").strip()
        dur=et-st
        if dur<=0: continue
        if not acc:
            acc=[s]; cur_len=dur; cur_chars=len(txt); continue
        if (cur_len+dur)<=MAX_DUR and (cur_chars+len(txt))<=MAX_CHARS:
            acc.append(s); cur_len+=dur; cur_chars+=len(txt)
            if cur_len>=TARGET or cur_chars>=MAX_CHARS*0.9:
                flush(acc,out); acc=[]; cur_len=0; cur_chars=0
        else:
            flush(acc,out); acc=[s]; cur_len=dur; cur_chars=len(txt)
    flush(acc,out)
    return [x for x in out if (x["end"]-x["start"])>=MIN_DUR]

tin=tout=0
for j in sorted(glob.glob(os.path.join(raw_dir,"*.json"))):
    d=json.load(open(j,"r",encoding="utf-8"))
    segs=d.get("segments",[]); tin+=len(segs)
    new=process(segs); tout+=len(new)
    json.dump({"audio":d["audio"],"segments":new}, open(os.path.join(out_dir, os.path.basename(j)),"w",encoding="utf-8"), ensure_ascii=False)
print(f">> smart chunking | in={tin} -> out={tout}")
PY

# ======= SEGMENT CUTTING =======
python3 - <<'PY'
import os, json, subprocess, shlex, re
root=os.path.abspath(os.environ["SCRATCH_DATA"])
jdir=os.path.join(root,"segments_json")
wdir=os.path.join(root,"wavs")
out=os.path.join(wdir,"segments"); os.makedirs(out,exist_ok=True)
sr=os.environ.get("SR","22050")
def safe(stem): return re.sub(r'[^a-zA-Z0-9._-]+','_', stem)
cnt=0; miss=0
for fn in sorted(os.listdir(jdir)):
    if not fn.endswith(".json"): continue
    d=json.load(open(os.path.join(jdir,fn),"r",encoding="utf-8"))
    audio=d["audio"]
    orig=os.path.join(wdir, audio)
    alt=os.path.join(wdir, safe(os.path.splitext(audio)[0]) + ".wav")
    wav = orig if os.path.isfile(orig) else (alt if os.path.isfile(alt) else None)
    if not wav:
        print("[CUT] skip, missing wav:", orig); miss+=1; continue
    stem=safe(os.path.splitext(audio)[0])
    for i,s in enumerate(d.get("segments",[]),1):
        st=float(s["start"]); et=float(s["end"]); dur=et-st
        if dur<=0: continue
        out_wav=os.path.join(out,f"{stem}_seg{i:04d}.wav")
        cmd=f'ffmpeg -y -ss {st:.3f} -t {dur:.3f} -i {shlex.quote(wav)} -ar {sr} -ac 1 -vn {shlex.quote(out_wav)}'
        subprocess.run(cmd,shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        cnt+=1
print(f">> segment cutting | files: {cnt} | missing wavs: {miss}")
PY

# ======= METADATA CSV =======
python3 - <<'PY'
import os, json, csv, glob, random, re, sys
root=os.path.abspath(os.environ["SCRATCH_DATA"])
seg_json=os.path.join(root,"segments_json")
seg_wavs=os.path.join(root,"wavs","segments")
max_chars=int(os.environ.get("MAX_TXT_CHARS","180"))
do_rw=int(os.environ.get("DIALECT_REWRITE","1"))
rw_file=os.environ.get("REWRITE_FILE","")
def safe(stem): return re.sub(r'[^a-zA-Z0-9._-]+','_', stem)
rules=[(r"\bund\b","u"),(r"\bUnd\b","U"),(r"\bich\b","i"),(r"\bIch\b","I"),
       (r"\bnicht\b","nid"),(r"\bNicht\b","Nid"),(r"\bklein\b","chli"),(r"\bKlein\b","Chli"),
       (r"\bwirklich\b","würkli"),(r"\bWirklich\b","Würkli"),(r"\bauch\b","au"),
       (r"\bgeht\b","goht"),(r"\bgehts\b","gohts"),(r"ß","ss"),(r"Grüß","Grüez")]
ext=[]
if rw_file and os.path.isfile(rw_file):
    for line in open(rw_file,"r",encoding="utf-8"):
        line=line.strip()
        if not line or line.startswith("#") or "|" not in line: continue
        pat,repl=line.split("|",1); ext.append((pat,repl))
def rewrite(t):
    for pat,repl in rules: t=re.sub(pat,repl,t)
    for pat,repl in ext:   t=re.sub(pat,repl,t)
    return re.sub(r"\s+"," ", t).strip()
pairs=[]
for j in sorted(glob.glob(os.path.join(seg_json,"*.json"))):
    d=json.load(open(j,"r",encoding="utf-8"))
    base=os.path.splitext(d["audio"])[0]
    base_safe=safe(base)
    for i,s in enumerate(d.get("segments",[]),1):
        text=(s.get("text") or "").strip()
        if len(text)<3: continue
        seg=os.path.join(seg_wavs,f"{base_safe}_seg{i:04d}.wav")
        if not os.path.exists(seg): continue
        t=rewrite(text) if do_rw else text
        if len(t)>max_chars: t=t[:max_chars].rsplit(" ",1)[0]
        if len(t)<3: continue
        pairs.append(("/workspace/dataset/wavs/segments/"+os.path.basename(seg), t))
random.shuffle(pairs)
n=len(pairs)
if n<20:
    print(f"Zu wenige Segmente nach Filterung: {n}"); sys.exit(1)
eval_n=min(50,max(5,int(0.1*n)))
train=pairs[eval_n:]; evals=pairs[:eval_n]
def write_csv(p,rows):
    with open(p,"w",encoding="utf-8",newline="") as f:
        w=csv.writer(f,delimiter="|",quoting=csv.QUOTE_ALL)
        w.writerow(["audio_file","text"])
        for r in rows: w.writerow(r)
write_csv(os.path.join(root,"metadata_train_coqui.csv"),train)
write_csv(os.path.join(root,"metadata_eval_coqui.csv"),evals)
print(f">> train={len(train)} eval={len(evals)} total={n}")
PY

# ======= TRAINING =======
say "Starte Training …"
docker run --rm --gpus all \
  --shm-size=8g --ipc=host \
  -e HF_HOME=/store_cache/hf \
  -e TRANSFORMERS_CACHE=/store_cache/hf/transformers \
  -e TORCH_HOME=/store_cache/torch \
  -e PIP_CACHE_DIR=/store_cache/pip \
  -e XDG_CACHE_HOME=/store_cache/xdg \
  -e TTS_HOME=/store_cache/tts \
  -v "${SCRATCH_DATA}":/workspace/dataset \
  -v "${RAM_CKPT_ROOT}":/workspace/ram_ckpt \
  -v "${STORE_CKPT}":/workspace/store_ckpt \
  ${CACHE_MNT} \
  -v "${SCRATCH_TMP}":/tmp \
  "${IMG_XTTS}" bash -lc '
set -e
cd /opt/xtts-ft
python3 train_gpt_xtts.py \
  --output_path /workspace/ram_ckpt \
  --metadatas /workspace/dataset/metadata_train_coqui.csv,/workspace/dataset/metadata_eval_coqui.csv,'"$LANG_TAG"' \
  --num_epochs '"$EPOCHS"' \
  --batch_size '"$BATCH_SIZE"' \
  --grad_acumm '"$GRAD_ACCUM"' \
  --max_text_length '"$MAX_TXT_CHARS"' \
  --max_audio_length '"$MAX_AUDIO_SAMPLES"' \
  --weight_decay '"$WEIGHT_DECAY"' \
  --lr '"$LR"' \
  --save_step '"$SAVE_STEP"'
'

# RAM → STORE spiegeln (letzter Lauf)
say "Spiegle letzten Lauf aus RAM → STORE …"
RAM_LAST="$(ls -dt "${RAM_CKPT_ROOT}"/GPT_XTTS_FT-* 2>/dev/null | head -n1 || true)"
SYNCED_DIR=""
if [[ -n "${RAM_LAST}" && -d "${RAM_LAST}" ]]; then
  cp -a "${RAM_LAST}" "${STORE_CKPT}/"
  SYNCED_DIR="${STORE_CKPT}/$(basename "${RAM_LAST}")"
  say "Gespiegelt: ${SYNCED_DIR}"
else
  echo "WARN: Kein Trainings-Output in ${RAM_CKPT_ROOT} gefunden."
fi

# ======= EXPORT latest_full (+ HF-Tokenizer) =======
LATEST="${SYNCED_DIR:-$(ls -dt "${STORE_CKPT}"/GPT_XTTS_FT-* 2>/dev/null | head -n1 || true)}"
if [[ -n "$LATEST" ]]; then
  ln -sfn "$LATEST" "${STORE_CKPT}/latest_run"
  ensure_dir "${STORE_CKPT}/latest_full"
  cp -f "$(ls -t "$LATEST"/checkpoint_*.pth | head -n1)" "${STORE_CKPT}/latest_full/model.pth"
  cp -f "$LATEST/config.json" "${STORE_CKPT}/latest_full/config.json" || true

  docker run --rm \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e HF_HOME=/tmp/hf -e TRANSFORMERS_CACHE=/tmp/hf/transformers \
    -v "${STORE_CKPT}/latest_full:/out" \
    python:3.11-slim bash -lc '
set -e
python - <<PY
import os, json, shutil, sys, subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub>=0.20"])
from huggingface_hub import hf_hub_download
token=os.environ.get("HF_TOKEN") or None
dst="/out"
repos=["coqui/XTTS-v2","coqui/XTTS-v2.0","coqui/XTTS-v2-original"]
want_primary=["tokenizer.json"]
want_fallback=["vocab.json","merges.txt"]
want_optional=["tokenizer_config.json","special_tokens_map.json","added_tokens.json"]
def try_dl(repo,fn):
    try:
        p=hf_hub_download(repo_id=repo, filename=fn, revision="main", token=token)
        shutil.copy2(p, os.path.join(dst, os.path.basename(fn))); return True
    except Exception:
        return False
have_tok=False
for r in repos:
    if any(try_dl(r,f) for f in want_primary):
        have_tok=True; break
have_vm=False
if not have_tok:
    for r in repos:
        got=[try_dl(r,f) for f in want_fallback]
        if all(got):
            have_vm=True
            for f in want_optional: try_dl(r,f)
            break
cfg_path=os.path.join(dst,"config.json")
if os.path.isfile(cfg_path):
    cfg=json.load(open(cfg_path,"r",encoding="utf-8")); ma=cfg.get("model_args",{}) or {}; ch=False
    if have_tok:
        ma["tokenizer_file"]=os.path.join(dst,"tokenizer.json"); ma.pop("merges_file",None); ch=True
    elif have_vm:
        ma["tokenizer_file"]=os.path.join(dst,"vocab.json"); ma["merges_file"]=os.path.join(dst,"merges.txt"); ch=True
    if ch:
        cfg["model_args"]=ma
        json.dump(cfg, open(cfg_path,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
else:
    print("WARN: config.json fehlt in latest_full")
PY
'
  if [[ -s "${STORE_CKPT}/latest_full/tokenizer.json" || ( -s "${STORE_CKPT}/latest_full/vocab.json" && -s "${STORE_CKPT}/latest_full/merges.txt" ) ]]; then
    echo "✅ Inference-Export bereit: ${STORE_CKPT}/latest_full"
  else
    echo "WARN: Tokenizer-Assets konnten nicht geladen werden – ggf. HF_TOKEN setzen und erneut exportieren."
  fi
else
  echo "WARN: Kein Checkpoint für Export gefunden."
fi

# ======= SEGMENT-EXPORT → STORE (Reference WAVs) =======
SRC_TAG_RAW="${NAS_SRC%/}"
SRC_TAG_RAW="${SRC_TAG_RAW##*/}"
SRC_TAG="${SRC_TAG_RAW:-unknown_source}"
SRC_TAG="${SRC_TAG//[^A-Za-z0-9._-]/_}"
if [[ -z "${SRC_TAG//_/}" ]]; then
  SRC_TAG="unknown_source"
fi
REF_EXPORT_DIR="${REF_EXPORT_BASE}/${SRC_TAG}"

if [[ -d "${SCRATCH_WAVS}/segments" ]]; then
  say "Exportiere Segmente nach ${REF_EXPORT_DIR} …"
  ensure_dir "${REF_EXPORT_DIR}/segments"
  rsync -av --delete -- "${SCRATCH_WAVS}/segments/" "${REF_EXPORT_DIR}/segments/"
  if [[ -f "${SCRATCH_DATA}/metadata_train_coqui.csv" ]]; then
    cp -f "${SCRATCH_DATA}/metadata_train_coqui.csv" "${REF_EXPORT_DIR}/metadata_train_coqui.csv"
  fi
  if [[ -f "${SCRATCH_DATA}/metadata_eval_coqui.csv" ]]; then
    cp -f "${SCRATCH_DATA}/metadata_eval_coqui.csv" "${REF_EXPORT_DIR}/metadata_eval_coqui.csv"
  fi
fi

# ======= CLEANUP am Ende (optional) =======
if [[ "${CLEAN_AFTER_TRAIN}" -eq 1 ]]; then
  say "Cleanup: ${SCRATCH_ROOT} und ${RAM_CKPT_ROOT} leeren …"
  [[ -d "${SCRATCH_ROOT}" && "${SCRATCH_ROOT}" != "/" ]] && find "${SCRATCH_ROOT}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  [[ -d "${RAM_CKPT_ROOT}" && "${RAM_CKPT_ROOT}" != "/" ]] && find "${RAM_CKPT_ROOT}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
fi

echo
echo "  Scratch : ${SCRATCH_ROOT}"
echo "  Store   : ${STORE_ROOT}   (Caches: ${STORE_CACHE})"
echo "  RAM     : ${RAM_CKPT_ROOT}"
