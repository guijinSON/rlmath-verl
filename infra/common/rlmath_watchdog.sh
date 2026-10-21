#!/bin/bash
# rlmath_watchdog.sh -- keep one rlmath training run alive (resume from the latest checkpoint after a crash or hang).
#
# Settings come from the environment (see the per-machine wrapper):
#   W            workspace root                     EXP        experiment name (ckpt/$EXP, logs/$EXP.log)
#   KILL         cleanup script (stops Ray, vLLM, driver; never the tunnels)
#   GPUS_CHECK   comma list of GPU indices the run uses (nvidia-smi order)
#   HOSTS_FILE   RLMATH_DOCKER_HOSTS_FILE of the run (sandbox tunnels to verify)   DEFAULT_SOCK  run_rlmath_async.sh's DOCKER_HOST socket
#   RELAUNCH     command (run in $W) that resumes the run (ALLOW_RESUME=1, WANDB_RUN_ID=..., WANDB_RESUME=allow, launcher)
#   TOTAL_STEPS  final step (stop watching when reached)
# Every CHECK s: restart if the driver is gone, or if the log has not changed for STALL_MIN minutes (hang). A restart:
# stop everything, wait for the run's GPUs to be free, wait for every sandbox tunnel to answer, keep the dead log as
# logs/$EXP.before_step<N>_<reason><time>.log (live_progress picks it up), relaunch, and verify the resume actually started.
# Gives up if it crashes twice in a row after the same step (repeat = config problem).
# Stops the run + alerts if the last 30 groups contain no completed trial (broken sandboxes).
# Gives up (logs/WATCHDOG_ALERT_$EXP) after MAX_RESTARTS restarts within 6 h. Disable: touch logs/WATCHDOG_DISABLE_$EXP.
set -u
CHECK=${CHECK:-300}; STALL_MIN=${STALL_MIN:-90}; MAX_RESTARTS=${MAX_RESTARTS:-3}
L=$W/logs/$EXP.log; C=$W/ckpt/$EXP; ALERT=$W/logs/WATCHDOG_ALERT_$EXP; OFF=$W/logs/WATCHDOG_DISABLE_$EXP
log() { echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M KST')] $*"; }
driver_alive() { ps -eo args | grep -c '^python3 -m recipe\.rlmath\.async_main'; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk -v g="$GPUS_CHECK" 'BEGIN{n=split(g,a,",");for(i=1;i<=n;i++)w[a[i]]=1} {if((NR-1) in w) s+=$1} END{print s+0}'; }
hosts() { { grep -v '^#' "$HOSTS_FILE" 2>/dev/null | awk '$2 > 0 {print $1}'; echo "unix://$DEFAULT_SOCK"; } | sort -u; }
tunnels_ok() {
  local bad=""
  for h in $(hosts); do timeout 30 "$W/bin/docker" -H "$h" info --format '{{.NCPU}}' >/dev/null 2>&1 || bad="$bad $h"; done
  [ -z "$bad" ] && return 0; log "sandbox host(s) not answering:$bad"; return 1
}
restarts=()

log "watchdog for $EXP started (check ${CHECK}s, stall ${STALL_MIN}min, max $MAX_RESTARTS restarts/6h)"
while true; do
  sleep "$CHECK"
  [ -f "$OFF" ] && { log "disabled ($OFF exists)"; continue; }
  done_step=$(cat "$C/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
  if [ "${done_step:-0}" -ge "$TOTAL_STEPS" ] || grep -q "Training Progress: 100%" "$L" 2>/dev/null; then
    log "run finished (checkpoint $done_step / $TOTAL_STEPS); watchdog exits"; exit 0
  fi
  # broken sandboxes: the driver is alive and the log moves, but no trial completes (e.g. missing compose plugin) -> the run
  # would train on all-zero groups and checkpoint them. Stop it and alert; do not relaunch (needs a human fix).
  recent=$(grep -a "\[RLMathRollouter\]\[stream\]" "$L" 2>/dev/null | tail -30)
  if [ "$(echo "$recent" | grep -c .)" -ge 30 ] && ! echo "$recent" | grep -q "'completed'"; then
    log "=== BROKEN SANDBOXES: last 30 groups have no completed trial ($(echo "$recent" | tail -1 | grep -oE "harbor=\{[^}]*\}")); stopping the run, NOT relaunching"
    bash "$KILL" > /dev/null 2>&1; echo "$(date) broken sandboxes (no completed trials); run stopped at ckpt $done_step" >> "$ALERT"; exit 1
  fi
  alive=$(driver_alive); age=$(( ($(date +%s) - $(stat -c %Y "$L" 2>/dev/null || echo 0)) / 60 ))
  reason=""
  [ "$alive" -lt 1 ] && reason="crash"
  [ -z "$reason" ] && [ "$age" -ge "$STALL_MIN" ] && reason="hang"
  [ -z "$reason" ] && continue

  now=$(date +%s); keep=(); for t in "${restarts[@]:-}"; do [ -n "$t" ] && [ $((now - t)) -lt 21600 ] && keep+=("$t"); done; restarts=("${keep[@]:-}")
  if [ "${#keep[@]}" -ge "$MAX_RESTARTS" ]; then
    log "GIVING UP: $reason after ${#keep[@]} restarts in 6 h (last checkpoint $done_step)"; echo "$(date) $reason after ${#keep[@]} restarts; ckpt $done_step" >> "$ALERT"; exit 1
  fi
  # same crash twice at the same step (no step completed in between) = config problem, not bad luck -> alert, don't loop
  crash_at=$(grep -aoE 'training/global_step:[0-9]+' "$L" 2>/dev/null | tail -1 | cut -d: -f2); crash_at=${crash_at:-$done_step}
  if [ "$reason" = crash ] && [ "$crash_at" = "${last_crash_at:-none}" ]; then
    log "GIVING UP: crashed again right after step $crash_at (same place as the previous crash); not relaunching"
    grep -E "OutOfMemoryError|Error executing|RayActorError|ActorDiedError|tunnel down|EngineCore failed" "$L" 2>/dev/null | tail -2 | cut -c1-220 | sed 's/^/    /'
    echo "$(date) repeated crash after step $crash_at; ckpt $done_step" >> "$ALERT"; exit 1
  fi
  [ "$reason" = crash ] && last_crash_at=$crash_at
  log "=== $reason detected (driver alive=$alive, log age ${age} min, last checkpoint $done_step); restarting"
  grep -E "OutOfMemoryError|Error executing|RayActorError|ActorDiedError|tunnel down|EngineCore failed" "$L" 2>/dev/null | tail -3 | cut -c1-220 | sed 's/^/    /'
  bash "$KILL" > /dev/null 2>&1
  for i in $(seq 60); do u=$(gpu_used); [ "$u" -lt 2000 ] && break; sleep 10; done
  [ "$u" -ge 2000 ] && { log "GPUs still hold ${u} MiB after stop; running cleanup again"; bash "$KILL" > /dev/null 2>&1; sleep 30; }
  for i in $(seq 30); do tunnels_ok && break; sleep 30; done
  tunnels_ok || { log "GIVING UP: sandbox tunnels down for 15 min; not relaunching"; echo "$(date) tunnels down" >> "$ALERT"; exit 1; }
  [ -f "$L" ] && mv "$L" "$W/logs/$EXP.before_step$((done_step + 1))_${reason}$(date +%m%d%H%M).log"
  (cd "$W" && eval "$RELAUNCH" < /dev/null > /dev/null 2>&1)
  restarts+=("$(date +%s)")
  ok=0
  for i in $(seq 90); do
    sleep 20
    if grep -qE "tunnel down|refusing|less than desired|EngineCore failed|OutOfMemoryError|preflight failed" "$L" 2>/dev/null; then
      log "relaunch FAILED: $(grep -oE "tunnel down|refusing|less than desired|EngineCore failed|OutOfMemoryError|preflight failed" "$L" | head -1)"; break
    fi
    grep -q "Setting global step" "$L" 2>/dev/null && [ "$(grep -c 'Sending keys' "$L")" -gt 20 ] && { ok=1; break; }
  done
  [ "$ok" = 1 ] && log "resumed: $(grep -oE 'Setting global step to [0-9]+' "$L" | tail -1), rollouts flowing" \
                || log "resume not confirmed within 30 min (next check will retry if the driver is gone)"
done
