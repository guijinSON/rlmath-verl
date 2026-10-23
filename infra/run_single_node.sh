#!/bin/bash
# Launch an rlmath run on one GPU machine (Kubernetes reservation pod or SSH box) from a fixed recipe + a site profile.
#
#   SITE=infra/sites/mlxp.env RECIPE=recipe/rlmath/recipes/q35_9b_64x16_lr2e-6.env \
#   EXP=rlmath_q35_9b_<data>_<site>_v1 TRAIN_DIR=<dir> VAL_DIR=<dir> bash infra/run_single_node.sh [extra hydra overrides]
#
# Options (STALENESS, BACKEND, MCORE_OPTIM, FSDP_OPTIM, TOTAL_STEPS, VAL_BEFORE, LR) go in front of the command.
# Site first (hardware, paths, sandboxes), then the recipe (algorithm); the run is detached and logs to $LOG_DIR/$EXP.log.
set -u
: "${SITE:?SITE=infra/sites/<site>.env}" "${RECIPE:?RECIPE=recipe/rlmath/recipes/<recipe>.env}" "${EXP:?EXP=<run name>}"
: "${TRAIN_DIR:?TRAIN_DIR=<Harbor task dir>}" "${VAL_DIR:?VAL_DIR=<Harbor val task dir>}"
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SITE=$(cd "$REPO" && realpath "$SITE"); RECIPE=$(cd "$REPO" && realpath "$RECIPE")
set -a; . "$SITE"; . "$RECIPE"; set +a
export EXP TRAIN_DIR VAL_DIR REMOTE_AGENT_TASK_PATH_TEMPLATE=${REMOTE_AGENT_TASK_PATH_TEMPLATE:-$TRAIN_DIR/\{instance_id\}}
if [ "$(cd "$REPO" && pwd -P)" != "$(cd "$W/verl" && pwd -P)" ]; then echo "warning: running $W/verl, not this checkout ($REPO)"; fi
echo "drivers alive: $(pgrep -fc '^python3 -m recipe.rlmath.async_main')"
# PRINT_ARGS=1: print the resolved Hydra overrides and stop -- no prelaunch side effects, nothing started
if [ "${PRINT_ARGS:-0}" = 1 ]; then cd "$W/verl" 2>/dev/null || cd "$REPO"; exec bash recipe/rlmath/run_rlmath_async.sh "$@"; fi
if declare -F site_prelaunch > /dev/null; then site_prelaunch || { echo "ABORT: site_prelaunch failed"; exit 1; }; fi
LOG_DIR=${LOG_DIR:-$W/logs}; mkdir -p "$LOG_DIR"; LOG=$LOG_DIR/${EXP}$([ "${SMOKE:-0}" = 1 ] && echo _smoke).log
cd "$W/verl" && setsid nohup bash recipe/rlmath/run_rlmath_async.sh "$@" > "$LOG" 2>&1 < /dev/null &
echo "launched $EXP (recipe $(basename "$RECIPE"), site $(basename "$SITE"), backend ${TRAIN_BACKEND}, lr ${LR}, staleness ${STALENESS}) -> $LOG"
