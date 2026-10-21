W=${W:-$HOME/rlmath-rl} EXP=rlmath_q35_9b_aav_co1500_4bu_mlxp_v2 KILL=$W/verl/infra/k8s-pod/cleanup.sh GPUS_CHECK=$(seq -s, 0 $(($(nvidia-smi -L | grep -c "^GPU")-1))) \
HOSTS_FILE=$W/docker_hosts DEFAULT_SOCK=/tmp/sbx7-docker.sock TOTAL_STEPS=160 \
RELAUNCH="RLMATH_DEFAULT_SOCK=/tmp/sbx7-docker.sock ALLOW_RESUME=1 VAL_BEFORE=false MAX_CONCURRENT_PROMPTS=null bash verl/infra/k8s-pod/launch_9b_construct1500.sh" \
  bash $W/verl/infra/common/rlmath_watchdog.sh
