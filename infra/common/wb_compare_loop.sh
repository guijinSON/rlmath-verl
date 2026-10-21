#!/bin/bash
# refresh math-problem-gen/rlmath-compare mirrors every 30 min (W&B auth from ~/.wandb_key). Moved here from KT 10-12.
W=${W:-$HOME/rlmath-rl}
cd $W/wbcmp
[ -f ~/.wandb_key ] && export WANDB_API_KEY=$(cat ~/.wandb_key)   # 10-17: per-pod key file (no .netrc)
while true; do
  echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M') KST]"
  env -u PYTHONPATH WANDB_DIR=$PWD timeout 1500 $W/venv-q36/bin/python -s $W/bin/wb_compare.py 2>&1 | grep -v "Warning\|Loaded credentials\|(+0 new)\|^SKIPPED"
  sleep 1800
done
