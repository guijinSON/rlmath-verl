#!/bin/bash
# fresh MLXP pod (home dir is not persistent): restore the sandbox-VM ssh setup from the node-local nvme copy
W=${W:-$HOME/rlmath-rl}
mkdir -p /root/.ssh && chmod 700 /root/.ssh
cp $W/keys/id_rlmath_sbx /root/.ssh/id_rlmath_sbx && chmod 600 /root/.ssh/id_rlmath_sbx
cp $W/ssh_config.rlmath /root/.ssh/config.rlmath
grep -q "Include /root/.ssh/config.rlmath" /root/.ssh/config 2>/dev/null || echo "Include /root/.ssh/config.rlmath" >> /root/.ssh/config
chmod 600 /root/.ssh/config /root/.ssh/config.rlmath
cp $W/keys/wandb_key /root/.wandb_key 2>/dev/null && chmod 600 /root/.wandb_key   # W&B key (from KT, 10-20)
