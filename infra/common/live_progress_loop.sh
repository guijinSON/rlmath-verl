#!/bin/bash
# live_progress_loop.sh <W> <EXP> <source id> <prompts per step> <python>: every 5 min push rollout progress to W&B
W=$1; EXP=$2; SRC=$3; PER=$4; PY=$5
cd $W/wbcmp 2>/dev/null || { mkdir -p $W/wbcmp; cd $W/wbcmp; }
while true; do
  LOGS="$W/logs/$EXP.steps1-2.log $(ls -tr $W/logs/$EXP.before_step*.log 2>/dev/null | tr '\n' ' ') $W/logs/$EXP.log"
  [ -f ~/.wandb_key ] && export WANDB_API_KEY=$(cat ~/.wandb_key)
  WANDB_DIR=$PWD timeout 600 $PY -s $W/verl/infra/common/live_progress.py $SRC $EXP $PER $LOGS 2>&1 | grep -v "Warning\|Loaded credentials" | tail -1
  sleep 300
done
