#!/bin/bash
# Megatron test env on MLXP (10-19): venv-mcore = copy of venv-q36 (torch 2.11/cu130, vllm 0.20.2, tf 5.3.0) +
# TransformerEngine v2.15 (source) + Megatron-Core core_v0.16.1 + mbridge@main + Megatron-Bridge, matching verl's
# docker/Dockerfile.stable.vllm. torch/vllm/transformers/flash-attn are pinned by a constraints file so nothing upgrades.
set -euxo pipefail
W=${W:-$HOME/rlmath-rl}; V=$W/venv-mcore; PY=$V/bin/python; SP=$V/lib/python3.12/site-packages
[ -d $V ] || cp -a $W/venv-q36 $V
$PY -m ensurepip --upgrade 2>/dev/null || { curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && $PY /tmp/get-pip.py; }
$PY -m pip freeze | grep -iE "^(torch|torchvision|torchaudio|vllm|transformers|flash.attn|flash_attn|triton|numpy|flashinfer.python)==" > $W/mcore_constraints.txt
cat $W/mcore_constraints.txt
$PY -m pip install -c $W/mcore_constraints.txt -U wheel setuptools cmake ninja pybind11
# full CUDA 13.0.2 toolkit (the pip cu13 wheels lack nvml.h / cuda_profiler_api.h / nvtx; the -dev wheels are 0.0.1 stubs)
CT=$W/../cuda-13.0.2
if [ ! -x $CT/bin/nvcc ]; then
  mkdir -p $W/../dl $W/../dl/tmp
  R=$W/../dl/cuda_13.0.2_580.95.05_linux.run
  [ -s $R ] || curl -sSL -o $R https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/cuda_13.0.2_580.95.05_linux.run
  sh $R --silent --toolkit --toolkitpath=$CT --no-man-page --override --tmpdir=$W/../dl/tmp
fi
export CUDA_HOME=$CT CUDNN_PATH=$SP/nvidia/cudnn NCCL_HOME=$SP/nvidia/nccl
[ -e $NCCL_HOME/lib/libnccl.so ] || ln -s libnccl.so.2 $NCCL_HOME/lib/libnccl.so
export PATH=$CUDA_HOME/bin:$V/bin:$PATH
export CPATH=$CUDA_HOME/include:$CUDA_HOME/include/cccl:$CUDNN_PATH/include:$NCCL_HOME/include:${CPATH:-}
export LIBRARY_PATH=$CUDA_HOME/lib64:$CUDNN_PATH/lib:$NCCL_HOME/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDNN_PATH/lib:$NCCL_HOME/lib:${LD_LIBRARY_PATH:-}
export NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS=90 MAX_JOBS=96 NVTE_BUILD_THREADS_PER_JOB=1
nvcc --version | tail -1
$PY -m pip install -c $W/mcore_constraints.txt --no-build-isolation "git+https://github.com/NVIDIA/TransformerEngine.git@v2.15"
$PY -m pip install --no-deps "git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.16.1"
$PY -m pip install -c $W/mcore_constraints.txt -U --no-deps "git+https://github.com/ISEEKYAN/mbridge.git@main"
$PY -m pip install --no-deps megatron-bridge
$PY -m pip show megatron-bridge | grep -iE "^(version|requires)"
$PY - <<'PY'
import torch, transformer_engine, megatron.core, mbridge, megatron.bridge, vllm, transformers
print("OK torch", torch.__version__, "TE", transformer_engine.__version__, "mcore", megatron.core.__version__,
      "bridge", getattr(megatron.bridge, "__version__", "?"), "vllm", vllm.__version__, "tf", transformers.__version__)
PY
echo INSTALL_DONE
