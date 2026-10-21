# infra/: how the rlmath runs are wired together

The training code is in `recipe/rlmath/`. This directory holds what it takes to run it: the GPU machines, the CPU
machines that host the agent sandboxes, the tunnels between them, launchers, watchdogs and checkpoint export.
Every script reads a workspace root `W` (default `$HOME/rlmath-rl`) with this layout:

```
$W/verl/            this repository (checked out on the GPU machine)
$W/venv*/           python env (torch 2.11 / cu130, vLLM 0.20.2, transformers 5.3 for Qwen3.5; see below)
$W/bin/docker       static docker CLI + compose plugin ($W/.docker); the GPU machine runs no Docker daemon
$W/data/<set>/      Harbor task dirs (train, val)
$W/ckpt/<EXP>/      checkpoints        $W/logs/<EXP>.log   run log (the watchdog reads it)
$W/docker_hosts     sandbox routing    $W/rlmath_limits    live concurrency    (templates in sandbox-vm/)
```

## 1. The picture

```
                       ┌──────────────── GPU machine ────────────────────────────────┐
                       │  rollout GPUs: vLLM servers ◀── LLM proxy (per-trial session) │
                       │       ▲ NCCL weight sync            ▲ OpenAI API calls        │
                       │  trainer GPUs: FSDP / Megatron      │                         │
                       │                       Terminus-2 agents (one per trial, CPU)  │
                       │                            │ docker / compose CLI              │
                       │   /tmp/sbx5-docker.sock  /tmp/sbx7-docker.sock  ...           │
                       └──────────┬───────────────────────┬──────────────────────────┘
                     ssh -L unix socket            ssh -L unix socket      (sandbox-vm/sbx-tunnel.sh)
                       ┌──────────▼─────────┐  ┌──────────▼─────────┐
                       │ sandbox VM 5        │  │ sandbox VM 7        │   CPU-only Docker hosts
                       │ dockerd, 245 cores  │  │ dockerd, 122 cores  │   (vast.ai KVM VMs)
                       │ per trial: task     │  │                     │
                       │ container + egress  │  │                     │
                       │ sidecar (no net)    │  │                     │
                       └─────────────────────┘  └─────────────────────┘
```

* The **agent runs on the GPU machine**: Terminus-2 talks to the policy through the LLM proxy on localhost. Only the
  sandbox (the task's container, where the agent's shell commands execute) is remote.
* Each **trial** = one Harbor task attempt: Harbor builds an image from the task's Dockerfile (cached layers +
  the task files), starts the task container plus an egress-control sidecar (no network access), drives a tmux
  session in it, copies `/logs` back, and runs the verifier. A GRPO step of 64 prompts × 16 rollouts is 1,024 trials.
* On a **Slurm supercomputer** (JUPITER, GH200) there are no remote VMs: sandboxes are Apptainer instances on the
  compute nodes themselves (see §4).

## 2. GPU machines

| Kind | How we reach it | Layout used | Notes |
|---|---|---|---|
| Kubernetes reservation pod (8× H200) | `kubectl --kubeconfig <cfg> exec <pod> -- bash -s < script` | 4 vLLM GPUs + 4 FSDP trainer GPUs | Pod home is wiped when the pod changes; node-local nvme persists on the same node, so `W` lives there and `k8s-pod/pod_restore_ssh.sh` restores `~/.ssh` + the W&B key from it. Shared machine: the run starts its own Ray head on non-default ports and **never** calls `ray stop`; `k8s-pod/cleanup.sh` kills only this run's processes by checking each `/proc/<pid>/cmdline`. |
| Single SSH box (8× H200 behind an SSH proxy) | `ssh -i <key> -p <port> <user>@<proxy>` | same as above | Same Docker-VM sandboxes. Keep data off small home volumes. |
| Slurm GH200 cluster (4 GPUs/node, aarch64) | SSH to login nodes; `sbatch` | 8 nodes: 4 vLLM nodes + 4 trainer nodes | Apptainer sandboxes on the nodes; see §4. |

Long jobs are always detached (`setsid nohup … < /dev/null &`) so they survive the SSH/exec session.

## 3. Sandbox VMs (CPU)

**Why remote VMs.** The GPU pods cannot run a Docker daemon, and one GRPO step needs hundreds of concurrent
containers. We rent CPU-only KVM VMs (vast.ai, 61–245 cores each) that run plain `dockerd`.

**Connection.** One persistent SSH tunnel per VM forwards the VM's Docker socket to a local unix socket:

```bash
# ~/.ssh/config.rlmath has one alias per VM (sandbox-vm/ssh_config.example); key used only for the VMs
setsid nohup bash $W/verl/infra/sandbox-vm/sbx-tunnel.sh sbx5 >/dev/null 2>&1 </dev/null &
$W/bin/docker -H unix:///tmp/sbx5-docker.sock info --format '{{.NCPU}}'     # 245
```

`DOCKER_HOST=ssh://vm` is not usable at this scale: it opens one SSH connection per docker call and floods the VM's
sshd. The tunnel keeps a single connection open and restarts itself if it drops.

**Routing.** `recipe/rlmath/remote_docker.py` reads `RLMATH_DOCKER_HOSTS_FILE` (`<docker host> <weight>` lines,
re-read every 20 s). Each new trial goes to the host with the fewest live trials per unit weight, so VMs can be
added, drained (weight 0) or reweighted while a run is going. `RLMATH_DEFAULT_SOCK` names the socket the launch
script sanity-checks.

**Capacity.** `RLMATH_LIMITS_FILE` (`recipe/rlmath/limits.py`, re-read every 30 s) sets prompts in flight and trials
per agent worker. Each trial is two containers; above roughly 0.75 trials per sandbox core the VM saturates and Harbor's
600 s environment start times out (`EnvironmentStartTimeoutError`). Such failures are classified as infrastructure
failures (`recipe/rlmath/overlong.py`), excluded from their GRPO group, and a circuit breaker pauses new trials
when they cluster.

**Fresh VM.** `sandbox-vm/setup_vm.sh` (Docker address pools, sshd session limits, required images), then
`sandbox-vm/warm_build.sh` with one task's Dockerfile so the first trials do not all pay the apt/pip layers.

**vast.ai specifics we hit.**
* When the account credit reaches $0, vast stops *every* instance. Tunnels then loop on "connection refused"; the
  watchdog stops the run once 30 groups finish with no completed trial.
* Restarting an instance rewrites `~/.ssh/authorized_keys`: re-add the GPU machine's sandbox key afterwards.
  The BuildKit cache can also be gone (re-run `warm_build.sh`).
* A stopped instance may not restart if its host's resources were rented out meanwhile ("state change queued");
  its disk is kept, but plan to route around it.
* A VM must belong to one run at a time: launchers delete every container on each host they list with weight > 0.

## 4. Slurm cluster (JUPITER GH200)

* One 8-node job: Ray head on the first node; 4 nodes of vLLM rollout, 4 nodes of trainer (FSDP, or Megatron TP4).
* Sandboxes are Apptainer instances of the task image on the compute nodes, with trial directories on node-local
  storage (the shared scratch filesystem has a tight inode quota).
* Jobs are chained (`--dependency=afterany`) because the time limit is 12 h. Each link resumes from the newest
  checkpoint.
* A rolling exporter on a login node uploads every checkpoint except the newest to the Hugging Face checkpoint repo,
  verifies the uploaded sizes, then deletes the local copy (`export_*_run.sh`).
* Login access is restricted to source IP ranges registered in the account portal. Scratch filesystems are cleaned
  automatically (30/90-day retention), so keep venvs and code reproducible from this repo.

## 5. Run lifecycle

```
k8s-pod/launch_*.sh          sets all knobs as env vars, clears containers on owned VMs, starts:
  recipe/rlmath/run_rlmath_async.sh   env checks, Ray head on private ports, docker sanity check, then
    python3 -m recipe.rlmath.async_main   Hydra config; rollouter + trainer actors (verl fully_async_policy)
common/rlmath_watchdog.sh    every 5 min: driver alive? log stalled > 90 min? → cleanup, wait for GPUs and
                             tunnels, relaunch with ALLOW_RESUME=1; gives up after 3 restarts in 6 h or a
                             repeat crash at the same step; stops the run if sandboxes are broken
```

* **Checkpoints** every `SAVE_FREQ` steps under `$W/ckpt/$EXP` (FSDP shards, or Megatron dist_checkpointing with
  `RLMATH_MCORE_DIST_CKPT=1` plus an HF copy). Merged weights are published as `<model>/<EXP>/step_N`.
* **W&B**: the key is read from a 0600 file into `WANDB_API_KEY` at launch, never put in argv or the repo. Without
  a key the run logs offline (`WANDB_MODE=offline`) and can be synced later. `common/wb_compare.py` mirrors runs
  into a comparison project with a trajectories-consumed x-axis.
* **Monitoring helpers**: `common/live_progress.py` (per-step progress from the log), `common/tool_outcomes.py`
  (what the agent ran), `common/vllm_metrics.py` (vLLM Prometheus metrics).

## 6. Software stack (Qwen3.5 hybrid GDN models)

* torch 2.11 + cu130, vLLM 0.20.2, transformers 5.3, flash-attn 2, torchao (bf16 stochastic-rounding AdamW).
* `RLMATH_TEXT_ONLY=1`: Qwen3.5 checkpoints are vision-language; we train text-only.
* No sequence packing for Qwen3.5 on FSDP (`DYNAMIC_BSZ=false MICRO_BS=1`): the GDN layers would leak recurrent
  state across packed trajectories unless `recipe/rlmath/patch_q35_packing.py` is applied.
* vLLM 0.20 JIT-compiles GDN kernels with nvcc and Triton needs gcc/g++: fresh containers need a compiler
  (the launchers install one) and the `nvidia-cuda-nvcc` wheel (`CUDA_HOME` inside the venv).
* Megatron path (`TRAIN_BACKEND=megatron`, `k8s-pod/install_megatron.sh`): TransformerEngine 2.15 +
  Megatron-Core 0.16.1 + mbridge. GDN has no context parallelism or packed sequences in mcore 0.16.1, so it runs
  `use_remove_padding=false`, and long trajectories need the chunked LM head (`RLMATH_LOGITS_CHUNK`).

## 7. Operational rules that saved us

* Kill by explicit PID after checking `/proc/<pid>/cmdline`; never `pkill -f`. A `pgrep -f PATTERN` inside the same
  `bash -c '...'` matches its own command line.
* A dropped SSH/exec session is not proof that the command did not run. Check state before re-running anything.
* Secrets (SSH keys, HF/W&B tokens) live in 0600 files and are read into env vars. They never go in argv, logs or
  this repository; only `*.example` templates are committed.
