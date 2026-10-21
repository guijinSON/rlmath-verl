source <(sed -n '/^log()/,/^restarts=()/p' $W/verl/infra/common/rlmath_watchdog.sh)
echo "driver_alive=$(driver_alive)  gpu_used=$(gpu_used) MiB  log age=$(( ($(date +%s) - $(stat -c %Y $W/logs/$EXP.log)) / 60 )) min  ckpt=$(cat $W/ckpt/$EXP/latest_checkpointed_iteration.txt)"
echo "hosts: $(hosts | tr '\n' ' ')"; tunnels_ok && echo "tunnels OK"
