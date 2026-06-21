#!/usr/bin/env bash
# Set up a project venv on lxplus using the LCG CUDA Python/PyTorch stack.
# Usage from repo root:
#   bash scripts/setup_lxplus_venv.sh
# Then for every new shell:
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_110_cuda/x86_64-el9-gcc13-opt/setup.sh
#   source .venv/lcg110-cuda/bin/activate

set -euo pipefail

LCG_VIEW="/cvmfs/sft.cern.ch/lcg/views/LCG_110_cuda/x86_64-el9-gcc13-opt/setup.sh"
VENV_DIR=".venv/lcg110-cuda"

if [[ ! -f "${LCG_VIEW}" ]]; then
  echo "ERROR: LCG view not found: ${LCG_VIEW}" >&2
  exit 1
fi

source "${LCG_VIEW}"

python3 -m venv --system-site-packages "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

# Keep the LCG CUDA/PyTorch stack. Do not pip-install torch/numpy/pandas/pyarrow here.
# The venv is created with --system-site-packages so the LCG packages remain visible.
python3 -m pip install --upgrade pip

python3 - <<'PY'
import sys
import torch
import numpy as np
import pandas as pd
import pyarrow as pa
import yaml
import matplotlib
print('python:', sys.version)
print('torch:', torch.__version__)
print('torch cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
print('numpy:', np.__version__)
print('pandas:', pd.__version__)
print('pyarrow:', pa.__version__)
print('yaml ok')
print('matplotlib:', matplotlib.__version__)
PY

echo ""
echo "Venv ready: ${VENV_DIR}"
echo "For each new shell, run:"
echo "  source ${LCG_VIEW}"
echo "  source ${VENV_DIR}/bin/activate"
