#!/bin/bash
# Prepare a fresh sandbox VM (Ubuntu + Docker, e.g. a vast.ai KVM instance). Run ON the VM as root.
# Afterwards: authorize the GPU node's sandbox key in ~/.ssh/authorized_keys (vast.ai rewrites that file on every
# instance restart, so re-add it after a stop/start), add an ssh alias, start sbx-tunnel.sh, list it in docker_hosts.
set -eu
# 1) many small bridge networks: one compose project (= network) per trial
mkdir -p /etc/docker
cat > /etc/docker/daemon.json <<'JSON'
{"default-address-pools": [{"base": "10.96.0.0/12", "size": 24}], "max-concurrent-downloads": 10}
JSON
systemctl restart docker
# 2) every docker call through the tunnel is an ssh channel; hundreds run concurrently
sed -i -E '/^#?(MaxStartups|MaxSessions)/d' /etc/ssh/sshd_config
printf 'MaxStartups 200:30:500\nMaxSessions 500\n' >> /etc/ssh/sshd_config
systemctl restart ssh || systemctl restart sshd
# 3) images every trial needs. The egress sidecar is built locally by Harbor; if it is missing every trial fails with
#    "pull access denied for harbor-prebuilt". Copy both from a working VM:
#      ssh old-vm 'docker save rlmath-base:0.3-tools harbor-prebuilt:<egress-sidecar-tag> | zstd' | ssh new-vm 'zstd -d | docker load'
docker images --format '{{.Repository}}:{{.Tag}}' | grep -E 'rlmath-base|egress' || echo "WARNING: base/egress images missing"
