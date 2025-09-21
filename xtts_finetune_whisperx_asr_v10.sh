#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# TRAINING-PIPELINE – Speicherlayout & Parameter (für KI-Training optimiert)
#
# Speicher:
#   - SCRATCH_ROOT  : flüchtig (mp4/wav, transcodings, temp) -> /mnt/fast_scratch
#   - STORE_ROOT    : persistent (Checkpoints, *Caches*)       -> /mnt/fast_store
#   - RAM_CKPT_ROOT : tmpfs (Zwischen-Checkpoints)             -> /mnt/ram_ckpt
#
# WICHTIG: Caches (HF/Transformers, Torch, Pip, XDG) zeigen auf STORE_ROOT,
#          damit Modelle/Tokenizer nur einmal geladen werden (WhisperX etc.).
#
# TUNABLES (oben gesammelt, alle per ENV überschreibbar):
#   – Daten/Quelle:
#       NAS_SRC                Pfad zu deinen Rohdaten (mp4/wav)
#   – WhisperX/ASR:
#       WHISPER_MODEL          Whisper/WhisperX-Model (z. B. large-v3)
#       WHISPER_FORCE_LANG     "" = Auto (empfohlen), oder "de" erzwingen
#       VAD_METHOD             none|silero (falls WhisperX-Version VAD kennt)
#       DIALECT_HINT           Prompt-Bias; wird um LEXICON_FILE ergänzt
#       LEXICON_FILE           zusätzliche Dialekt-Wortliste (persistent)
#   – Segmentierung & Limits:
#       SR                     Ziel-Samplerate (Hz) für WAV
#       MAX_TXT_CHARS          max. Textlänge pro Sample (Zeichen)
#       MIN_DUR / TARGET_DUR / MAX_DUR   Segments (Sek.)
#       MAX_AUDIO_SAMPLES      max. Audiosamples pro Beispiel
#       DIALECT_REWRITE        1=Rewrite bei CSV (post), 0=aus
#       RAW_DIALECT_REWRITE    1=Rewrite schon in RAW-JSON, 0=aus
#       REWRITE_FILE           eigene Regex-Rewrite-Regeln (TSV, persistent)
#   – Training (XTTS):
#       EPOCHS                 Anzahl Epochen
#       BATCH_SIZE             Batch-Größe
#       GRAD_ACCUM             Gradient Accumulation (eff. Batch = BATCH_SIZE*GRAD_ACCUM)
#       LR                     Lernrate
#       WEIGHT_DECAY           Weight Decay
#       SAVE_STEP              Speicherschritt (Iteration)
#   – Caches & Docker:
#       USE_STORE_CACHE        1=persistent (empfohlen), 0=Scratch
#       DOCKER_TMPDIR          temp für Docker-Builds/Pulls -> Scratch
#       (Optional) Docker data-root in /etc/docker/daemon.json setzen, siehe unten.
#   – Cleanup:
#       CLEAN_AFTER_TRAIN      1=Scratch & RAM-CKPT leeren nach Export, 0=behalten
###############################################################################

########################
# STORAGE-LAYOUT
########################
SCRATCH_ROOT="${SCRATCH_ROOT:-/mnt/fast_scratch}"     # flüchtig
STORE_ROOT="${STORE_ROOT:-/mnt/fast_store}"           # persistent
RAM_CKPT_ROOT="${RAM_CKPT_ROOT:-/mnt/ram_ckpt}"       # tmpfs (z. B. 15G)

# Unterordner
SCRATCH_DATA="${SCRATCH_ROOT}/dataset"
SCRATCH_WAVS="${SCRATCH_DATA}/wavs"
SCRATCH_TMP="${SCRATCH_ROOT}/tmp"

STORE_CKPT="${STORE_ROOT}/checkpoints"
STORE_CACHE="${STORE_ROOT}/cache"                     # HF/Torch/Pip/XDG

# Docker Temp (schnelle, flüchtige I/O)
export DOCKER_TMPDIR="${DOCKER_TMPDIR:-${SCRATCH_ROOT}/docker-tmp}"

########################
# USER-VARS (Quellen & Tokens)
########################
export RAW_DIALECT_REWRITE="${RAW_DIALECT_REWRITE:-1}"      # vor-ASR-Rewrite (0/1)
HF_TOKEN="${HF_TOKEN:-*}"   # <— besser per ENV setzen
NAS_SRC="${NAS_SRC:-/mnt/nas/Allgemein/VoiceClone/*}"

########################
# WHISPERX / ASR
########################
WHISPER_MODEL="${WHISPER_MODEL:-large-v3}"
WHISPER_FORCE_LANG="${WHISPER_FORCE_LANG:-}"    # ""=Auto, sonst "de"
VAD_METHOD="${VAD_METHOD:-none}"                # none|silero

DIALECT_HINT="${DIALECT_HINT:-Allegra, Grüezi, Chur, Bündner, nöd, nüüt, öppis, chli, lueg, tuet, wos, dä, d’, miar, bisch, goht, gits, schaffa, gäge, nid, üs, äbe, gaht, gäll.}"
LEXICON_FILE="${LEXICON_FILE:-${STORE_ROOT}/dialect_lexicon.txt}"

########################
# SEGMENTIERUNG / LIMITS
########################
LANG_TAG="${LANG_TAG:-de}"
SR="${SR:-22050}"
MAX_TXT_CHARS="${MAX_TXT_CHARS:-180}"
MIN_DUR="${MIN_DUR:-0.6}"
TARGET_DUR="${TARGET_DUR:-5.0}"
MAX_DUR="${MAX_DUR:-8.0}"
MAX_AUDIO_SAMPLES="${MAX_AUDIO_SAMPLES:-240000}"

DIALECT_REWRITE="${DIALECT_REWRITE:-1}"
REWRITE_FILE="${REWRITE_FILE:-${STORE_ROOT}/dialect_rewrite.tsv}"

########################
# TRAINING-HYPERPARAMETER
########################
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
LR="${LR:-5e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
SAVE_STEP="${SAVE_STEP:-100}"

########################
# CACHES & CLEANUP
########################
USE_STORE_CACHE="${USE_STORE_CACHE:-1}"   # 1=persistent Caches im STORE
CLEAN_AFTER_TRAIN="${CLEAN_AFTER_TRAIN:-1}"

# Host-Env für Caches (wir mounten identische Pfade in Container)
if [[ "$USE_STORE_CACHE" -eq 1 ]]; then
  mkdir -p "${STORE_CACHE}"/{hf,torch,pip,xdg}
  export HF_HOME="${HF_HOME:-${STORE_CACHE}/hf}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
  export TORCH_HOME="${TORCH_HOME:-${STORE_CACHE}/torch}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${STORE_CACHE}/pip}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${STORE_CACHE}/xdg}"
else
  mkdir -p "${SCRATCH_ROOT}/cache"/{hf,torch,pip,xdg}
  export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/cache/hf}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
  export TORCH_HOME="${TORCH_HOME:-${SCRATCH_ROOT}/cache/torch}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SCRATCH_ROOT}/cache/pip}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_ROOT}/cache/xdg}"
fi

########################
# DOCKER-IMAGES
########################
IMG_WHISPER="${IMG_WHISPER:-whisper-tools}"
IMG_XTTS="${IMG_XTTS:-xtts-finetune:cu121}"

export \
  SCRATCH_ROOT SCRATCH_DATA SCRATCH_WAVS STORE_CKPT STORE_CACHE SCRATCH_TMP RAM_CKPT_ROOT \
  HF_TOKEN WHISPER_MODEL LANG_TAG IMG_XTTS WHISPER_FORCE_LANG DIALECT_HINT \
  MAX_TXT_CHARS MIN_DUR MAX_DUR TARGET_DUR SR DIALECT_REWRITE RAW_DIALECT_REWRITE VAD_METHOD \
  REWRITE_FILE LEXICON_FILE MAX_AUDIO_SAMPLES EPOCHS BATCH_SIZE GRAD_ACCUM LR WEIGHT_DECAY SAVE_STEP

###############################################################################
# PREP & CHECKS
###############################################################################
for p in "$SCRATCH_ROOT" "$STORE_ROOT" "$RAM_CKPT_ROOT" "$SCRATCH_WAVS" "$STORE_CKPT" "$SCRATCH_TMP" \
         "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$DOCKER_TMPDIR"; do
  [[ -d "$p" ]] || mkdir -p "$p"
done

# Mount-Check – falls System gerade gebootet hat
for m in "$SCRATCH_ROOT" "$STORE_ROOT" "$RAM_CKPT_ROOT"; do
  if ! mountpoint -q "$m"; then
    echo "WARN: $m ist nicht gemountet – versuche mount -a …"
    mount -a || true
  fi
done

# Images bauen (nur wenn fehlen)
docker image inspect "${IMG_WHISPER}" >/dev/null 2>&1 || docker build -t "${IMG_WHISPER}" -f whisperx.Dockerfile .
docker image inspect "${IMG_XTTS}"    >/div/null 2>&1 || docker build -t "${IMG_XTTS}"    -f xtts-finetune.Dockerfile .

###############################################################################
# ROHDATEN SYNC (mp4/wav) → SCRATCH
###############################################################################
rsync -av --delete --include="*/" --include="*.mp4" --include="*.wav" --exclude="*" \
  "${NAS_SRC}/" "${SCRATCH_DATA}/"

# Frischer Start
rm -rf "${SCRATCH_DATA}/segments_json_raw" "${SCRATCH_DATA}/segments_json" "${SCRATCH_WAVS}/segments"
mkdir -p "${SCRATCH_DATA}/segments_json_raw" "${SCRATCH_DATA}/segments_json" "${SCRATCH_WAVS}/segments"

###############################################################################
# MP4 -> WAV @ ${SR} mono (auf SCRATCH)
###############################################################################
find "${SCRATCH_DATA}" -type f -iname '*.mp4' -print0 | \
  xargs -0 -I{} -P"$(nproc)" bash -c '
    in="$1"; base="$(basename "$in" .mp4)"
    out="'"${SCRATCH_WAVS}"'/${base}.wav"
    [[ -f "$out" ]] || ffmpeg -y -i "$in" -ar '"$SR"' -ac 1 -vn "$out" >/dev/null 2>&1
  ' _ {}

###############################################################################
# TRANSKRIPTION (WhisperX) – Dialekt-Bias, VAD off
###############################################################################
if [[ -f "${LEXICON_FILE}" ]]; then
  DIALECT_HINT="${DIALECT_HINT} $(tr '\n' ' ' < "${LEXICON_FILE}")"
fi

# Container-Cache (persist.) + Temp (Scratch)
CACHE_MNT="-v ${STORE_CACHE}:/store_cache"
[[ "$USE_STORE_CACHE" -eq 0 ]] && CACHE_MNT="-v ${SCRATCH_ROOT}/cache:/store_cache"

docker run --rm --gpus all \
  --shm-size=8g --ipc=host \
  -e HF_TOKEN -e WHISPER_FORCE_LANG -e WHISPER_MODEL \
  -e DIALECT_HINT -e VAD_METHOD -e RAW_DIALECT_REWRITE \
  -e HF_HOME=/store_cache/hf \
  -e TRANSFORMERS_CACHE=/store_cache/hf/transformers \
  -e TORCH_HOME=/store_cache/torch \
  -e PIP_CACHE_DIR=/store_cache/pip \
  -e XDG_CACHE_HOME=/store_cache/xdg \
  -v "${SCRATCH_DATA}":/data $CACHE_MNT -v "${SCRATCH_TMP}":/tmp \
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

from whisperx.asr import TranscriptionOptions
base_asr_opts={"initial_prompt": full_hint, "temperature": 0.5, "beam_size": 1, "best_of": 1,
               "condition_on_previous_text": False, "suppress_tokens": [-1]}
try:
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

###############################################################################
# SMART CHUNKING
###############################################################################
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

###############################################################################
# SEGMENT CUTTING (ffmpeg)
###############################################################################
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
        subprocess.run(cmd,shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        cnt+=1
print(f">> segment cutting | files: {cnt} | missing wavs: {miss}")
PY

###############################################################################
# METADATA (pipe: audio_file|text)
###############################################################################
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

###############################################################################
# TRAINING (Checkpoints -> RAM; Export -> STORE)
###############################################################################
docker run --rm --gpus all \
  --shm-size=8g --ipc=host \
  -e HF_HOME=/store_cache/hf \
  -e TRANSFORMERS_CACHE=/store_cache/hf/transformers \
  -e TORCH_HOME=/store_cache/torch \
  -e PIP_CACHE_DIR=/store_cache/pip \
  -e XDG_CACHE_HOME=/store_cache/xdg \
  -v "${SCRATCH_DATA}":/workspace/dataset \
  -v "${RAM_CKPT_ROOT}":/workspace/ram_ckpt \
  -v "${STORE_CKPT}":/workspace/store_ckpt \
  $CACHE_MNT -v "${SCRATCH_TMP}":/tmp \
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

# RAM -> STORE spiegeln (letzter Lauf)
echo ">> Spiegle letzten Lauf aus RAM → STORE …"
RAM_LAST="$(ls -dt "${RAM_CKPT_ROOT}"/GPT_XTTS_FT-* 2>/dev/null | head -n1 || true)"
if [[ -n "${RAM_LAST}" && -d "${RAM_LAST}" ]]; then
  cp -a "${RAM_LAST}" "${STORE_CKPT}/"
  SYNCED_DIR="${STORE_CKPT}/$(basename "${RAM_LAST}")"
  echo ">> Gespiegelt: ${SYNCED_DIR}"
else
  echo "WARN: Kein Trainings-Output in ${RAM_CKPT_ROOT} gefunden."
  SYNCED_DIR=""
fi

###############################################################################
# EXPORT für Inference (latest_full) + Tokenizer via Docker
###############################################################################
LATEST="${SYNCED_DIR:-$(ls -dt "${STORE_CKPT}"/GPT_XTTS_FT-* 2>/dev/null | head -n1 || true)}"
if [[ -n "$LATEST" ]]; then
  ln -sfn "$LATEST" "${STORE_CKPT}/latest_run"
  mkdir -p "${STORE_CKPT}/latest_full"
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
    except Exception: return False
have_tok=False
for r in repos:
    if any(try_dl(r,f) for f in want_primary): have_tok=True; break
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

###############################################################################
# CLEANUP – Scratch & RAM-Checkpoint leeren (abschaltbar)
###############################################################################
if [[ "${CLEAN_AFTER_TRAIN}" -eq 1 ]]; then
  echo ">> Cleanup: ${SCRATCH_ROOT} und ${RAM_CKPT_ROOT} leeren …"
  # Safety-Guards gegen "rm -rf /"
  [[ -d "${SCRATCH_ROOT}" && "${SCRATCH_ROOT}" != "/" ]] && find "${SCRATCH_ROOT}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  [[ -d "${RAM_CKPT_ROOT}" && "${RAM_CKPT_ROOT}" != "/" ]] && find "${RAM_CKPT_ROOT}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
fi

echo
echo "  Scratch : ${SCRATCH_ROOT}"
echo "  Store   : ${STORE_ROOT}   (Caches: ${STORE_CACHE})"
echo "  RAM     : ${RAM_CKPT_ROOT}"
