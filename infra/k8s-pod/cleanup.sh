#!/bin/bash
# Stop this pod's rlmath run: driver, launcher shell, Ray daemons/workers, vLLM engines. Never the ssh tunnels, watchdog or Jupyter.
# Every PID is matched on its own /proc/<pid>/cmdline (no pkill -f, no `ray stop`).
for d in /proc/[0-9]*; do
  pid=${d#/proc/}; [ "$pid" = "$$" ] && continue
  c=$(tr '\0' ' ' < $d/cmdline 2>/dev/null) || continue
  case "$c" in
    "python3 -m recipe.rlmath.async_main"*|"bash recipe/rlmath/run_rlmath_async.sh"*|ray::*|VLLM::*|*/raylet/raylet*|*/gcs/gcs_server*|*ray/dashboard/*|*ray/_private/*|*ray/autoscaler/*)
      kill -9 "$pid" 2>/dev/null && echo "killed $pid ${c:0:90}";;
  esac
done
