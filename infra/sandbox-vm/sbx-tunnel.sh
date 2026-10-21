#!/bin/bash
# Persistent SSH unix-socket tunnel: local /tmp/<alias>-docker.sock -> the sandbox VM's /var/run/docker.sock.
# One long-lived ssh per VM; DOCKER_HOST=ssh://... would spawn an ssh per docker call and flood the VM's sshd at
# 100+ concurrent trials. Restarts itself if the connection drops.
#   setsid nohup bash infra/sandbox-vm/sbx-tunnel.sh sbx5 >/dev/null 2>&1 </dev/null &
# Start it from a separate script file: a `pgrep -f sbx-tunnel.sh || start` guard inside `bash -c '...'` matches its
# own command line and never starts the tunnel.
ALIAS=${1:?usage: sbx-tunnel.sh <ssh-alias>}; W=${W:-$HOME/rlmath-rl}; mkdir -p $W/logs
while true; do
  ssh -N -o ControlMaster=no -o ControlPath=none -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o StreamLocalBindUnlink=yes \
      -L /tmp/$ALIAS-docker.sock:/var/run/docker.sock $ALIAS
  echo "$(date -u) tunnel exited ($?), restarting" >> $W/logs/$ALIAS-tunnel.log
  sleep 3
done
