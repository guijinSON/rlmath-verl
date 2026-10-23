#!/bin/bash
# Multi-node rlmath run on a Slurm GH200 cluster: one Ray cluster over all nodes (ROLLOUT_NODES vLLM nodes +
# TRAIN_NODES trainer nodes, from the site profile), Apptainer sandboxes on the nodes, driver on the head node.
#
#   SITE=infra/sites/jupiter.env RECIPE=recipe/rlmath/recipes/q35_9b_64x16_lr2e-6.env EXP=<run> TRAIN_DIR=<dir> VAL_DIR=<dir> \
#     sbatch -A <account> -p <partition> -N 8 -J <name>[-cN] -o <log dir>/%x-%j.out infra/slurm/rlmath.sbatch
#
# Continuations: resubmit with the same EXP (ALLOW_RESUME=1 in the site profile) and a job name ending in -cN, chained
# with --dependency=afterany:<prev>. SITE/RECIPE are re-read at job start, so edits reach already-queued jobs.
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@300
set -u
: "${SITE:?}" "${RECIPE:?}" "${EXP:?}" "${TRAIN_DIR:?}" "${VAL_DIR:?}"
set -a; . "$SITE"; . "$RECIPE"; set +a
export EXP TRAIN_DIR VAL_DIR REMOTE_AGENT_TASK_PATH_TEMPLATE=${REMOTE_AGENT_TASK_PATH_TEMPLATE:-$TRAIN_DIR/\{instance_id\}}
if declare -F site_preflight > /dev/null; then site_preflight || exit 3; fi
ulimit -c 0                                   # crashed workers left multi-GB core.* files in the code dir
LOGD=${SLURM_LOG_DIR:-$PWD}; IB_IF=${IB_IF:-ib0}
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST)); HEAD=${NODES[0]}
IBIP="ip -o -4 addr show $IB_IF | awk '{print \$4}' | cut -d/ -f1"
HEAD_IP=$(srun --overlap -N1 -n1 -w $HEAD bash -c "$IBIP")
echo "[$(date)] $EXP: recipe $(basename $RECIPE) backend $TRAIN_BACKEND lr $LR staleness $STALENESS; nodes=${NODES[*]} head=$HEAD ($HEAD_IP)"
RAYCOMMON="--temp-dir /tmp/ray_rlmath --num-gpus ${GPUS_PER_NODE:-4} --min-worker-port 30000 --max-worker-port 39999 --disable-usage-stats --block"
cd $W/verl
# every node sources the launcher with ENV_ONLY=1 (exports only) before its own `ray start`, so Ray workers share the driver's env
srun --overlap -N1 -n1 -w $HEAD bash -c "ENV_ONLY=1 source recipe/rlmath/run_rlmath_async.sh && ray start --head --node-ip-address=$HEAD_IP --port 6391 --dashboard-port 8291 --ray-client-server-port 10091 $RAYCOMMON" > $LOGD/ray_head_$SLURM_JOB_ID.log 2>&1 &
sleep 25
for n in "${NODES[@]:1}"; do
  srun --overlap -N1 -n1 -w $n bash -c "ENV_ONLY=1 source recipe/rlmath/run_rlmath_async.sh && ray start --address=$HEAD_IP:6391 --node-ip-address=\$($IBIP) $RAYCOMMON" > $LOGD/ray_worker_${n}_$SLURM_JOB_ID.log 2>&1 &
done
sleep 30
srun --overlap -N1 -n1 -w $HEAD bash -c "source $VENV/bin/activate && ray status --address=$HEAD_IP:6391 | head -25"
srun --overlap -N1 -n1 -w $HEAD bash -c "RAY_EXTERNAL=1 RAY_ADDRESS=$HEAD_IP:6391 bash recipe/rlmath/run_rlmath_async.sh $*"
rc=$?; echo "[$(date)] driver exited rc=$rc"
