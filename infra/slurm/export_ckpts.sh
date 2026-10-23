#!/bin/bash
# Rolling HF export of an FSDP run's checkpoints on a Slurm site (run it on a login node, detached).
#   SITE=infra/sites/<site>.env setsid nohup bash infra/slurm/export_ckpts.sh <EXP> <slurm job-name prefix> >> <log> 2>&1 &
# Every 20 min: each complete global_step_N older than latest_checkpointed_iteration is merged on CPU (fla grabs CUDA
# otherwise) into /tmp, uploaded to $HF_CKPT_REPO/$HF_CKPT_PREFIX/<EXP>/step_N, size-verified on HF, then DELETED
# locally. The latest ckpt always stays (the chain resumes from it). When no job of the chain is left (two successful
# squeue calls in a row -- one failed squeue during a Slurm outage once ended an exporter early), the final ckpt is
# exported but kept, and the token file is removed unless another exporter is still running.
# Megatron (dist_checkpoint) runs need a different merge; this script handles FSDP shards only.
set -u
RUN=$1; JOBPFX=$2
set -a; . "${SITE:?SITE=infra/sites/<site>.env}"; set +a
: "${HF_CKPT_REPO:=amphora/rlmath-agentic-grpo-checkpoints}" "${HF_CKPT_PREFIX:=qwen3.5-9b}" "${HF_TOKEN_FILE:=$HOME/.hf_amphora_token}"
: "${EXPORT_STATE_DIR:=$HOME/rlmath/export_state}"; mkdir -p "$EXPORT_STATE_DIR"
C=$CKPT_ROOT/$RUN E=/tmp/${USER}_hf_export_$RUN P=$HF_CKPT_PREFIX/$RUN VERL=$W/verl
WS=$(( ${TRAIN_NODES:-1} * ${TRAIN_GPUS_PER_NODE:-4} ))   # FSDP world size = number of model shards per ckpt
BASE=$(ls -d $HF_HOME_OVERRIDE/hub/models--${MODEL_HF_ID:-Qwen--Qwen3.5-9B}/snapshots/*/ 2>/dev/null | head -1)
DONE=$EXPORT_STATE_DIR/${RUN}_done; RUNNING=$EXPORT_STATE_DIR/$RUN.running; touch $DONE $RUNNING
cd $VERL; export PYTHONPATH=$VERL CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 HYDRA_FULL_ERROR=1
log() { echo "[$(date '+%m-%d %H:%M %Z')] $*"; }

export_step() {  # $1 = step, $2 = delete after verify (1/0)
  local s=$1 S=$C/global_step_$1 T=$E/$P/step_$1
  grep -qx "$s" $DONE && { [ "$2" = 1 ] && rm -rf $S && log "step $s already on HF -> deleted local"; return 0; }
  [ "$(ls $S/actor/model_world_size_* 2>/dev/null | wc -l)" -ge $WS ] && [ -d $S/actor/huggingface ] || { log "step $s incomplete, skip"; return 1; }
  log "merging step $s"; rm -rf $E; mkdir -p $T
  $VENV/bin/python -m verl.model_merger merge --backend fsdp --local_dir $S/actor --target_dir $T 2>&1 | grep -vE "^\s*$|^config:" | tail -1
  for f in $S/actor/huggingface/*; do case "$f" in *.safetensors|*.bin) ;; *) cp -n "$f" $T/ ;; esac; done
  [ -n "$BASE" ] && cp -nL $BASE/preprocessor_config.json $BASE/video_preprocessor_config.json $T/ 2>/dev/null
  ls $T/model*.safetensors >/dev/null 2>&1 || { log "step $s MERGE FAILED"; return 1; }
  log "uploading step $s ($(du -sh $T | cut -f1), $(ls $T | wc -l) files)"
  HF_TOKEN=$(cat $HF_TOKEN_FILE) $VENV/bin/python - "$E" "$P/step_$s" "$T" "$HF_CKPT_REPO" <<'PY' || { log "step $s UPLOAD/VERIFY FAILED"; return 1; }
import os, sys
from huggingface_hub import HfApi
E, prefix, T, repo = sys.argv[1:5]; api = HfApi()
api.upload_large_folder(repo_id=repo, folder_path=E, repo_type="model", num_workers=8)
local = {f: os.path.getsize(f"{T}/{f}") for f in os.listdir(T)}
remote = {x.path.split("/")[-1]: x.size for x in api.list_repo_tree(repo, path_in_repo=prefix) if hasattr(x, "size")}
bad = [f for f in local if remote.get(f) != local[f]]
assert not bad, f"mismatch {bad}"
print(f"verified {len(local)} files, {sum(local.values())/1e9:.2f} GB")
PY
  echo $s >> $DONE; rm -rf $E
  [ "$2" = 1 ] && { rm -rf $S; log "step $s on HF, local ckpt deleted"; } || log "step $s on HF (local kept)"
}

while true; do
  latest=$(cat $C/latest_checkpointed_iteration.txt 2>/dev/null || echo 0)
  for s in $(ls $C 2>/dev/null | sed -n 's/^global_step_//p' | sort -n); do
    [ "$s" -lt "$latest" ] && export_step $s 1
  done
  q=$(squeue --me -h -o %j 2>/dev/null); qrc=$?
  if [ $qrc -eq 0 ] && ! grep -q "^$JOBPFX" <<< "$q"; then gone=$((${gone:-0}+1)); else gone=0; fi
  if [ "${gone:-0}" -ge 2 ]; then
    latest=$(cat $C/latest_checkpointed_iteration.txt); export_step $latest 0
    rm -f $RUNNING; [ -n "$(ls $EXPORT_STATE_DIR/*.running 2>/dev/null)" ] || rm -f $HF_TOKEN_FILE
    log "chain over; final step $latest exported (kept locally), exiting"; exit 0
  fi
  sleep 1200
done
