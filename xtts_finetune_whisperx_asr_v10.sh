#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# XTTS Finetune + WhisperX ASR — v10 (fixed)
#
# Speicher-Layout:
#   SCRATCH  : /mnt/fast_scratch  (flüchtig; Transcodierung, WAV-Segmente, tmp)
#   STORE    : /mnt/fast_store    (persistent; Checkpoints, Caches)
#   RAM-CKPT : /mnt/ram_ckpt      (tmpfs; Checkpoints während des Trainings)
#
# Wichtige Parameter (anpassen bei Bedarf):
#   HF_TOKEN                : HuggingFace Token (für private Modelle/Coqui)
#   NAS_SRC                 : Quelle der Rohdaten (mp4/wav)
#   WHISPER_MODEL           : z.B. large-v3 (faster-whisper Backend)
#   WHISPER_FORCE_LANG      : "" (Auto) oder "de"
#   VAD_METHOD              : "none" | "silero"  (wir setzen default "none")
#   DIALECT_HINT            : Dialektvokabular/Leitwörter (Prompt-Bias)
#   LEXICON_FILE            : optionale Ergänzungsliste (persistent)
#   LANG_TAG                : Sprach-Tag fürs Training (z.B. "de")
#   SR                      : Ziel-Samplerate WAV (**24000**)
#   MAX_TXT_CHARS           : max. Textlänge pro Sample (CSV)
#   MIN_DUR / TARGET_DUR /
#   MAX_DUR                 : Chunking (Sekunden)
#   MAX_AUDIO_SAMPLES       : Audiosamples-Grenze im Training
#   DIALECT_REWRITE         : 1 = Dialekt-Rewrite beim CSV-Bau
#   RAW_DIALECT_REWRITE     : 1 = Rewrite schon in raw-JSON von WhisperX
#   REWRITE_FILE            : zusätzliche Rewrite-Regeln (TSV, persistent)
#   SAVE_STEP               : Checkpoint-Intervall (Steps) — direkt in RAM
#   CLEAN_AFTER_RUN         : 1 = RAM + Scratch nach Export aufräumen
#   RAMCKPT_MIN_FREE_GB     : Abbruch, wenn RAM-ckpt weniger frei hat
#
# Caches (persistieren Downloads!):
#   STORE/cache/{hf,torch,pip}      werden in Container nach /cache gemountet
#   HF_HOME, TRANSFORMERS_CACHE,
#   TORCH_HOME, XDG_CACHE_HOME      zeigen alle auf /cache (persistent)
#
# STORAGE-LAYOUT (wie bei dir):
#   SCRATCH  : /mnt/fast_scratch  (flüchtig; Transcodierung, WAV-Segmente, tmp)
#   STORE    : /mnt/fast_store    (persistent; Checkpoints, Caches)
#   RAM-CKPT : /mnt/ram_ckpt      (tmpfs; Checkpoints während des Trainings)
#
# Das Skript:
#   1) Sorgt dafür, dass Mounts vorhanden sind und leer **zu Beginn**:
#        - /mnt/ram_ckpt wird geleert (tmpfs).
#        - /mnt/fast_scratch wird geleert.
#   2) Zieht benötigte Docker-Images automatisch (WhisperX & Python slim).
#        - IMG_XTTS wird versucht zu pullen; falls kein Registry-Pfad, Hinweis.
#   3) Synchronisiert Rohdaten (mp4/wav) vom NAS → SCRATCH.
#   4) Transcodiert mp4 → wav @SR mono.
#   5) WhisperX-Transkription (+Dialekt-Prompt), segmentiert & chunkt.
#   6) Schneidet WAV-Segmente, baut CSVs.
#   7) Startet XTTS-Training (Checkpoints in RAM) und spiegelt letztes Run → STORE.
#   8) Exportiert "latest_full" + besorgt Tokenizer-Assets (HF) via python:3.11-slim.
#
# DEBUG:
#   DEBUG=1 ./xtts_finetune_whisperx_asr_v10.sh   # zeigt set -x
###############################################################################

########################
# KONFIG (überschreibbar via Env)
########################
SCRATCH_ROOT="${SCRATCH_ROOT:-/mnt/fast_scratch}"    # flüchtig
STORE_ROOT="${STORE_ROOT:-/mnt/fast_store}"          # persistent
RAM_CKPT_ROOT="${RAM_CKPT_ROOT:-/mnt/ram_ckpt}"      # tmpfs (z.B. 15G)

# Unterordner
SCRATCH_DATA="${SCRATCH_ROOT}/dataset"
SCRATCH_WAVS="${SCRATCH_DATA}/wavs"
SCRATCH_TMP="${SCRATCH_ROOT}/tmp"

STORE_CKPT="${STORE_ROOT}/checkpoints"
PERSISTENT_CACHE="${STORE_ROOT}/cache"   # persistente Caches (HF/Torch/Pip)

# Optional: Docker-Temp auf Scratch
export DOCKER_TMPDIR="${SCRATCH_ROOT}/docker-tmp"

# HF-Token (leer lassen, wenn nur public Modelle)
HF_TOKEN="${HF_TOKEN:-hf_xyz}"

# Rohdaten-Quelle
NAS_SRC="${NAS_SRC:-/mnt/nas/Allgemein/VoiceClone/*}"

# WhisperX / ASR
WHISPER_MODEL="${WHISPER_MODEL:-large-v3}"
WHISPER_FORCE_LANG="${WHISPER_FORCE_LANG:-}"    # ""=Auto sonst "de"
VAD_METHOD="${VAD_METHOD:-none}"                # none|silero

# Dialekt-Prompt
DIALECT_HINT="${DIALECT_HINT:-Allegra, Grüezi, Chur, Bündner, nöd, nüüt, öppis, chli, lueg, tuet, wos, dä, d’, miar, bisch, goht, gits, schaffa, gäge, nid, üs, äbe, gaht, gäll.}"
LEXICON_FILE="${LEXICON_FILE:-${STORE_ROOT}/dialect_lexicon.txt}"      # persistent

# Dataset / Training
LANG_TAG="${LANG_TAG:-de}"
SR="${SR:-24000}"                     # 24k für XTTS v2 ist üblich
MAX_TXT_CHARS="${MAX_TXT_CHARS:-180}"
MIN_DUR="${MIN_DUR:-0.6}"
TARGET_DUR="${TARGET_DUR:-5.0}"
MAX_DUR="${MAX_DUR:-8.0}"
MAX_AUDIO_SAMPLES="${MAX_AUDIO_SAMPLES:-240000}"

# Dialekt-Rewrite
DIALECT_REWRITE="${DIALECT_REWRITE:-1}"          # 1 = Rewrite beim CSV-Bau
RAW_DIALECT_REWRITE="${RAW_DIALECT_REWRITE:-0}"  # 1 = Rewrite schon in raw JSON
REWRITE_FILE="${REWRITE_FILE:-${STORE_ROOT}/dialect_rewrite.tsv}"

# Training / Checkpointing
SAVE_STEP="${SAVE_STEP:-300}"                     # seltener speichern = weniger I/O
CLEAN_AFTER_RUN="${CLEAN_AFTER_RUN:-1}"          # 1 = nach Export RAM+Scratch leeren
RAMCKPT_MIN_FREE_GB="${RAMCKPT_MIN_FREE_GB:-5}"  # Mindestfrei für RAM-ckpt

# Docker Images
# WhisperX: öffentlich verfügbar → wir ziehen automatisch, falls fehlt.
IMG_WHISPER="${IMG_WHISPER:-ghcr.io/m-bain/whisperx:latest}"
# XTTS: Dein custom Image (enthält /opt/xtts-ft/train_gpt_xtts.py)
# Wenn das Bild keinen Registry-Pfad enthält (kein "/"), versuchen wir trotzdem zu pullen.
# Scheitert das, bekommst du eine klare Meldung (siehe ensure_docker_image_or_pull).
IMG_XTTS="${IMG_XTTS:-xtts-finetune:cu121}"
# Für Tokenizer-Export:
IMG_PY_SLIM="${IMG_PY_SLIM:-python:3.11-slim}"

# Export-Schalter:
DO_EXPORT="${DO_EXPORT:-1}"   # 1 = latest_full exportieren + Tokenizer ziehen

########################
# ENV für Container-Caches
########################
docker_env_persistent_cache=(
  -e HF_HOME=/cache/hf
  -e TRANSFORMERS_CACHE=/cache/hf/transformers
  -e TORCH_HOME=/cache/torch
  -e XDG_CACHE_HOME=/cache
)
docker_env_hf_token=()
if [[ -n "${HF_TOKEN}" ]]; then
  docker_env_hf_token+=(
    -e "HF_TOKEN=${HF_TOKEN}"
    -e "HUGGINGFACE_HUB_TOKEN=${HF_TOKEN}"
    -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}"
  )
fi

########################
# UTILs
########################
[[ "${DEBUG:-0}" = "1" ]] && set -x

say() { echo "==> $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

bytes_to_human() {
  awk -v b="$1" 'BEGIN{
    hum[1024^4]="T";hum[1024^3]="G";hum[1024^2]="M";hum[1024]="K";
    for (x=1024^4; x>=1024; x/=1024) if (b>=x){ printf "%.1f%s\n", b/x, hum[x]; exit }
    print b "B"
  }'
}

safe_empty_dir() {
  # Leert ein Verzeichnis (optionales "keep"-Listing von Basenamen)
  local dir="$1"; shift || true
  local keep=("$@")
  [[ -d "$dir" ]] || return 0
  shopt -s dotglob nullglob
  for p in "$dir"/* "$dir"/.[!.]* "$dir"/..?*; do
    [[ -e "$p" ]] || continue
    local base="$(basename "$p")"
    local skip=0
    for k in "${keep[@]:-}"; do
      [[ "$base" == "$k" ]] && { skip=1; break; }
    done
    (( skip )) && continue
    rm -rf --one-file-system -- "$p"
  done
  shopt -u dotglob nullglob
}

ensure_dir() { mkdir -p -- "$1"; }

ensure_mount() {
  # ensure_mount <pfad> [erwarteter_fstype]
  local path="$1"; local want_fstype="${2:-}"
  if ! mountpoint -q "$path"; then
    say "WARN: $path ist nicht gemountet – versuche mount -a …"
    mount -a || true
  fi
  if ! mountpoint -q "$path"; then
    die "$path ist nicht gemountet. Bitte fstab prüfen."
  fi
  if [[ -n "$want_fstype" ]]; then
    local got; got="$(findmnt -no FSTYPE -- "$path" || true)"
    [[ "$got" == "$want_fstype" ]] || die "$path ist kein $want_fstype (ist: ${got:-unbekannt})."
  fi
}

ensure_bins() {
  local miss=()
  for b in docker ffmpeg python3 rsync awk sed grep stat; do
    command -v "$b" >/dev/null 2>&1 || miss+=("$b")
  done
  (( ${#miss[@]} )) && die "Fehlende Binaries: ${miss[*]}"
}

ensure_docker_image_or_pull() {
  # ensure_docker_image_or_pull <image> [fallback_image]
  local img="$1"; local fallback="${2:-}"
  if docker image inspect "$img" >/dev/null 2>&1; then
    return 0
  fi
  say "Docker-Image fehlt: ${img} → versuche docker pull …"
  if docker pull "$img"; then
    return 0
  fi
  if [[ -n "$fallback" ]]; then
    say "Pull für ${img} fehlgeschlagen. Probiere Fallback: ${fallback} …"
    if docker image inspect "$fallback" >/dev/null 2>&1 || docker pull "$fallback"; then
      export IMG_WHISPER="$fallback"
      say "Nutze Fallback-Image: ${IMG_WHISPER}"
      return 0
    fi
  fi

  # Zusätzlicher Hinweis für Custom-Images ohne Registry-Pfad
  if [[ "$img" != */* ]]; then
    cat >&2 <<EOF
ERROR: Konnte Docker-Image "${img}" nicht finden/pullen.
- Das sieht nach einem lokalen/custom Image ohne Registry-Pfad aus.
- Bitte:
  a) baue es lokal (docker build -t ${img} <DEIN_DOCKERFILE_VERZEICHNIS>)
  b) oder setze IMG_XTTS auf ein pullbares Image (mit Registry, z.B. ghcr.io/…/xtts-finetune:cu121)
EOF
  fi
  return 1
}

trap 'echo; echo "💥 Abbruch – siehe Meldung oben. (Zeile $LINENO)"; echo' ERR

########################
# PREP: Binaries, Mounts, Pre-Clean
########################
ensure_bins

# Verzeichnis-Skelett (wird ggf. nach dem Leeren wieder angelegt)
ensure_dir "$SCRATCH_ROOT"
ensure_dir "$STORE_ROOT"
ensure_dir "$RAM_CKPT_ROOT"

# Mounts sicherstellen
ensure_mount "$SCRATCH_ROOT" "xfs"     # dein fast_scratch
ensure_mount "$STORE_ROOT"   "xfs"     # dein fast_store (persistent)
ensure_mount "$RAM_CKPT_ROOT" "tmpfs"  # RAM-ckpt muss tmpfs sein

# *** WICHTIG: Am Anfang IMMER leeren ***
say "Leere RAM-Checkpoints (RAM_CKPT_ROOT) …"
safe_empty_dir "$RAM_CKPT_ROOT"     # alles raus

say "Leere Fast-Scratch (SCRATCH_ROOT) …"
safe_empty_dir "$SCRATCH_ROOT"      # alles raus (wir legen danach neu an)

# Ordnerstruktur erneut anlegen (sauber, frisch)
ensure_dir "$SCRATCH_WAVS"
ensure_dir "$SCRATCH_TMP"
ensure_dir "$STORE_CKPT"
ensure_dir "$PERSISTENT_CACHE/hf" "$PERSISTENT_CACHE/torch" "$PERSISTENT_CACHE/pip"
ensure_dir "$SCRATCH_ROOT/docker-tmp"

# RAM-Kapazität prüfen
free_kb="$(df -k --output=avail "$RAM_CKPT_ROOT" | awk 'NR==2{print $1}')"
free_b=$((free_kb*1024))
free_h="$(bytes_to_human "$free_b")"
say "RAM-ckpt frei: $free_h"
need=$(( RAMCKPT_MIN_FREE_GB * 1024 * 1024 * 1024 ))
(( free_b < need )) && die "Zu wenig freier RAM-ckpt (benötigt >= ${RAMCKPT_MIN_FREE_GB}G)."

########################
# DOCKER-IMAGES SICHERN (auto-pull)
########################
# WhisperX: Wenn dein IMG_WHISPER unbrauchbar ist, fallbacke auf ghcr.io/m-bain/whisperx:latest.
ensure_docker_image_or_pull "$IMG_WHISPER" "ghcr.io/m-bain/whisperx:latest"

# Python slim für Tokenizer-Export (klein & öffentlich)
ensure_docker_image_or_pull "$IMG_PY_SLIM"

# XTTS-Image (custom). Wir versuchen zu pullen, ansonsten klare Anleitung.
ensure_docker_image_or_pull "$IMG_XTTS"

########################
# NAS erreichbar?
########################
if [[ -n "${NAS_SRC}" ]]; then
  say "Prüfe NAS-Quelle (${NAS_SRC}) …"
  # autofs/cifs sauber anstupsen, aber nicht ewig hängen:
  timeout 10s bash -lc "stat -t '${NAS_SRC}' >/dev/null 2>&1 || ls -ld '${NAS_SRC}' >/dev/null 2>&1" \
    || die "NAS-Quelle ${NAS_SRC} nicht erreichbar."
fi

########################
# ROHDATEN SYNC (mp4/wav) → SCRATCH
########################
say "Synchronisiere Rohdaten (mp4/wav) → Scratch …"
ensure_dir "$SCRATCH_DATA"
# Nur mp4/wav; .wav im Ziel sauber halten
rsync -av --delete --timeout=30 \
  --include="*/" --include="*.mp4" --include="*.wav" --exclude="*" \
  "${NAS_SRC}/" "${SCRATCH_DATA}/"

########################
# Reset Segment-Outputs
########################
rm -rf "${SCRATCH_DATA}/segments_json_raw" "${SCRATCH_DATA}/segments_json" "${SCRATCH_WAVS}/segments"
ensure_dir "${SCRATCH_DATA}/segments_json_raw" "${SCRATCH_DATA}/segments_json" "${SCRATCH_WAVS}/segments"

########################
# MP4 -> WAV @SR mono (SCRATCH)
########################
say "Transcodieren MP4 → WAV …"
find "${SCRATCH_DATA}" -type f -iname '*.mp4' -print0 | \
  xargs -0 -I{} -P"$(nproc)" bash -c '
    in="$1"; base="$(basename "$in" .mp4)"
    out="'"${SCRATCH_WAVS}"'/${base}.wav"
    [[ -f "$out" ]] || ffmpeg -y -i "$in" -ar '"$SR"' -ac 1 -vn "$out" >/dev/null 2>&1
  ' _ {}

########################
# TRANSKRIPTION (WhisperX)
########################
if [[ -f "${LEXICON_FILE}" ]]; then
  # Lexikon an Dialekt-Hinweis anhängen (einzeilig)
  DIALECT_HINT="${DIALECT_HINT} $(tr "\n" " " < "${LEXICON_FILE}")"
fi

say "Starte WhisperX (ASR) …"
docker run --rm --gpus all \
  --shm-size=8g --ipc=host \
  -v "${SCRATCH_DATA}":/data \
  -v "${PERSISTENT_CACHE}":/cache \
  -v "${SCRATCH_TMP}":/tmp \
  "${docker_env_persistent_cache[@]}" \
  "${docker_env_hf_token[@]}" \
  -e "WHISPER_FORCE_LANG=${WHISPER_FORCE_LANG}" \
  -e "WHISPER_MODEL=${WHISPER_MODEL}" \
  -e "DIALECT_HINT=${DIALECT_HINT}" \
  -e "VAD_METHOD=${VAD_METHOD}" \
  -e "RAW_DIALECT_REWRITE=${RAW_DIALECT_REWRITE}" \
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

########################
# SMART CHUNKING
########################
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

########################
# SEGMENT CUTTING
########################
python3 - <<'PY'
import os, json, subprocess, shlex, re
root=os.path.abspath(os.environ["SCRATCH_DATA"])
jdir=os.path.join(root,"segments_json")
wdir=os.path.join(root,"wavs")
out=os.path.join(wdir,"segments"); os.makedirs(out,exist_ok=True)
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
        cmd=f'ffmpeg -y -ss {st:.3f} -t {dur:.3f} -i {shlex.quote(wav)} -ar {os.environ["SR"]} -ac 1 -vn {shlex.quote(out_wav)}'
        subprocess.run(cmd,shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
        cnt+=1
print(f">> segment cutting | files: {cnt} | missing wavs: {miss}")
PY

########################
# METADATA CSV
########################
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

########################
# TRAINING (Checkpoints → RAM; Export → STORE)
########################
say "Starte Training …"
docker run --rm --gpus all \
  --shm-size=8g --ipc=host \
  -v "${SCRATCH_DATA}":/workspace/dataset \
  -v "${RAM_CKPT_ROOT}":/workspace/ram_ckpt \
  -v "${STORE_CKPT}":/workspace/store_ckpt \
  -v "${PERSISTENT_CACHE}":/cache \
  -v "${SCRATCH_TMP}":/tmp \
  "${docker_env_persistent_cache[@]}" \
  "${docker_env_hf_token[@]}" \
  "${IMG_XTTS}" bash -lc '
set -e
cd /opt/xtts-ft
python3 train_gpt_xtts.py \
  --output_path /workspace/ram_ckpt \
  --metadatas /workspace/dataset/metadata_train_coqui.csv,/workspace/dataset/metadata_eval_coqui.csv,'"$LANG_TAG"' \
  --num_epochs 10 \
  --batch_size 1 \
  --grad_acumm 16 \
  --max_text_length '"$MAX_TXT_CHARS"' \
  --max_audio_length '"$MAX_AUDIO_SAMPLES"' \
  --weight_decay 1e-2 \
  --lr 5e-6 \
  --save_step '"$SAVE_STEP"'
'

# Nach dem Training: letzten Lauf aus RAM auf STORE spiegeln
say "Spiegle letzten Lauf aus RAM → STORE …"
RAM_LAST="$(ls -dt "${RAM_CKPT_ROOT}"/GPT_XTTS_FT-* 2>/dev/null | head -n1 || true)"
if [[ -n "${RAM_LAST}" && -d "${RAM_LAST}" ]]; then
  cp -a "${RAM_LAST}" "${STORE_CKPT}/"
  SYNCED_DIR="${STORE_CKPT}/$(basename "${RAM_LAST}")"
  say "Gespiegelt: ${SYNCED_DIR}"
else
  echo "WARN: Kein Trainings-Output in ${RAM_CKPT_ROOT} gefunden."
  SYNCED_DIR=""
fi

########################
# EXPORT für Inference (latest_full) + Tokenizer
########################
if [[ "${DO_EXPORT}" = "1" ]]; then
  LATEST="${SYNCED_DIR:-$(ls -dt "${STORE_CKPT}"/GPT_XTTS_FT-* 2>/dev/null | head -n1 || true)}"
  if [[ -n "$LATEST" ]]; then
    ln -sfn "$LATEST" "${STORE_CKPT}/latest_run"
    ensure_dir "${STORE_CKPT}/latest_full"
    cp -f "$(ls -t "$LATEST"/checkpoint_*.pth | head -n1)" "${STORE_CKPT}/latest_full/model.pth"
    cp -f "$LATEST/config.json" "${STORE_CKPT}/latest_full/config.json" || true

    say "Tokenizers aus HF besorgen (falls nötig) …"
    docker run --rm \
      -v "${STORE_CKPT}/latest_full:/out" \
      "${docker_env_hf_token[@]}" \
      "${IMG_PY_SLIM}" bash -lc '
set -e
python - <<PY
import os, json, shutil, sys, subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub>=0.20"])
from huggingface_hub import hf_hub_download
token=os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN") or None
dst="/out"
repos=["coqui/XTTS-v2","coqui/XTTS-v2.0","coqui/XTTS-v2-original"]
want_primary=["tokenizer.json"]
want_fallback=["vocab.json","merges.txt"]
want_optional=["tokenizer_config.json","special_tokens_map.json","added_tokens.json"]
def try_dl(repo,fn):
    try:
        p=hf_hub_download(repo_id=repo, filename=fn, revision="main", token=token)
        shutil.copy2(p, os.path.join(dst, os.path.basename(fn))); return True
    except Exception: return False
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
      echo "WARN: Tokenizer-Assets konnten nicht geladen werden – bitte manuell nach ${STORE_CKPT}/latest_full/ legen."
    fi
  else
    echo "WARN: Kein Checkpoint gefunden."
  fi
fi

########################
# Nachlauf-Cleanup
########################
if [[ "${CLEAN_AFTER_RUN}" == "1" ]]; then
  say "Bereinige nach Export: RAM-ckpt und Scratch …"
  safe_empty_dir "$RAM_CKPT_ROOT"
  safe_empty_dir "$SCRATCH_ROOT"
fi

echo
echo "  Scratch : ${SCRATCH_ROOT}"
echo "  Store   : ${STORE_ROOT}"
echo "  RAM     : ${RAM_CKPT_ROOT}"
