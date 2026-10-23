#!/bin/bash
# W&B mirror for a Slurm site whose compute nodes are offline (WANDB_MODE=offline). Run ONE per site on a login node:
#   SITE=infra/sites/<site>.env setsid nohup bash infra/slurm/wandb_offline_sync.sh > /dev/null 2>&1 < /dev/null &
# Every 5 min, for each offline-run dir under $WANDB_DIR/wandb: written in the last 3 h -> keep a `wandb beta sync
# --live` running for it; otherwise (job over) -> stop the live sync, do one final full sync, backfill the steps a
# time-limit kill truncated from the job's Slurm log, mark it done. If the offline writer died while the job log keeps
# growing, backfill from the log hourly. Chained runs listed in $ST/merge_groups also get one stitched run.
# Login nodes reboot: check $ST/heartbeat and restart this when it goes stale. Single instance across login nodes:
# the newest launch owns $ST/supervisor.pid, older ones exit.
set -a; . "${SITE:?SITE=infra/sites/<site>.env}"; set +a
export WANDB_API_KEY=$(cat ${WANDB_KEY_FILE:-$HOME/.wandb_key}); unset WANDB_MODE
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); V=$VENV/bin
WDS="$WANDB_DIR/wandb ${WANDB_EXTRA_DIRS:-}"; LOGS="${SLURM_LOG_DIR:?}/*.out"
ST=${WANDB_SYNC_STATE:-$HOME/rlmath/wandb_sync_state}; mkdir -p $ST/live $ST/done $ST/logs
ME=$(hostname):$$; echo $ME > $ST/supervisor.pid
while true; do
  [ "$(cat $ST/supervisor.pid)" = "$ME" ] || { echo "[$(date)] $ME superseded by $(cat $ST/supervisor.pid), exiting" >> $ST/logs/supervisor.log; exit 0; }
  date +%s > $ST/heartbeat; echo "$(hostname) $(date)" >> $ST/heartbeat
  for D in $(for w in $WDS; do ls -d $w/offline-run-* 2>/dev/null; done); do
    id=${D##*-}
    if [ -f $ST/done/$id ]; then   # done but the .wandb grew afterwards (long first step after a resume) -> reopen
      [ -n "$(find $D -maxdepth 2 -name "*.wandb" -newer $ST/done/$id | head -1)" ] || continue
      rm -f $ST/done/$id; echo "[$(date)] reopened $id (grew after done)" >> $ST/logs/supervisor.log
    fi
    recent=$(find $D -maxdepth 2 -name "*.wandb" -mmin -180 | head -1)
    JL=$(grep -l "offline-run-[0-9_]*-$id" $LOGS 2>/dev/null | head -1)
    if [ -n "$JL" ] && [ -n "$(find $JL -mmin -20)" ] && [ -z "$(find $D -maxdepth 2 -name "*.wandb" -mmin -60 | head -1)" ] \
       && [ -z "$(find $ST/fallback_$id -mmin -60 2>/dev/null)" ]; then
      touch $ST/fallback_$id; echo "[$(date)] log fallback $id <- $JL" >> $ST/logs/supervisor.log
      (cd /tmp && timeout 600 $V/python $HERE/wandb_backfill_from_log.py $id $JL >> $ST/logs/$id.log 2>&1)
      continue
    fi
    pidf=$ST/live/$id
    if [ -n "$recent" ]; then
      if ! { [ -f $pidf ] && kill -0 $(cat $pidf) 2>/dev/null; }; then
        echo "[$(date)] live sync start $id" >> $ST/logs/supervisor.log
        setsid $V/wandb beta sync --live --yes "$D" >> $ST/logs/$id.log 2>&1 < /dev/null & echo $! > $pidf
      fi
    else
      [ -f $pidf ] && { kill $(cat $pidf) 2>/dev/null; sleep 5; rm -f $pidf; }
      echo "[$(date)] final sync $id" >> $ST/logs/supervisor.log
      timeout 1800 $V/wandb beta sync --yes --no-skip-synced "$D" >> $ST/logs/$id.log 2>&1
      [ -n "$JL" ] && (cd /tmp && timeout 600 $V/python $HERE/wandb_backfill_from_log.py $id $JL --final >> $ST/logs/$id.log 2>&1)
      touch $ST/done/$id; echo "[$(date)] done $id (backfill log: ${JL:-none})" >> $ST/logs/supervisor.log
    fi
  done
  for G in $(cat $ST/merge_groups 2>/dev/null); do
    mkdir -p $ST/merge_$G
    (cd /tmp && timeout 900 $V/python $HERE/wandb_merge_parts.py $G $ST/merge_$G >> $ST/logs/merge.log 2>&1)
  done
  sleep 300
done
