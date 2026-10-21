#!/bin/bash
# Warm the BuildKit cache for a task family's Dockerfile on a sandbox VM (run ON the VM).
# Harbor builds one image per TRIAL; the per-task layers are tiny once the shared layers are cached, but a cold VM
# spends minutes on apt/pip for the first trials and they time out. Usage: warm_build.sh <task>/environment/Dockerfile
set -eu
DF=${1:?usage: warm_build.sh <Dockerfile>}; d=$(mktemp -d)
grep -vE '^(COPY|ADD) ' "$DF" > $d/Dockerfile      # shared layers only
t0=$(date +%s); DOCKER_BUILDKIT=1 docker build -q -t rlmath-warm:latest $d >/dev/null
echo "warm build $(( $(date +%s) - t0 ))s"
