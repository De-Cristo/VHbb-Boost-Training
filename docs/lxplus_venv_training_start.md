# lxplus venv start guide for VHbb boosted DNN smoke training

This guide is based on the local-agent environment report from `lxplus901.cern.ch` on 2026-06-21.

## Key environment finding

The lxplus GPU environment already provides the required ML stack through the LCG CUDA view:

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_110_cuda/x86_64-el9-gcc13-opt/setup.sh
```

Observed packages from the report:

```text
Python:   3.13.11
torch:    2.11.0
CUDA:     12.5 through PyTorch
numpy:    2.4.4
pandas:   2.2.3
pyarrow:  24.0.0
matplotlib and PyYAML available
GPU:      NVIDIA Tesla T4, 15.6 GB VRAM
```

Therefore, do **not** pip-install `torch` for this smoke test. Use the LCG-provided CUDA PyTorch.

## Recommended venv policy

Use a venv only for small project-local packages or future additions. The venv should inherit LCG packages:

```bash
python3 -m venv --system-site-packages .venv/lcg110-cuda
```

This keeps `torch`, `numpy`, `pandas`, `pyarrow`, and CUDA-related libraries from the LCG view visible inside the venv.

Important rule for every new shell/session:

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_110_cuda/x86_64-el9-gcc13-opt/setup.sh
source .venv/lcg110-cuda/bin/activate
```

The LCG view must be sourced first so that Python, CUDA libraries, and runtime library paths are consistent.

## One-time setup

From the repository root on lxplus:

```bash
cd /afs/cern.ch/work/l/lichengz/private/VHbb-Training/VHbb-Boost-Training

source /cvmfs/sft.cern.ch/lcg/views/LCG_110_cuda/x86_64-el9-gcc13-opt/setup.sh

python3 -m venv --system-site-packages .venv/lcg110-cuda
source .venv/lcg110-cuda/bin/activate

python3 -m pip install --upgrade pip
```

Do not run `pip install -r requirements-smoke.txt` on lxplus. That file is now documentation only. The LCG CUDA view already provides the required core stack, and reinstalling `torch`, `numpy`, `pandas`, or `pyarrow` with pip can break the CUDA/runtime consistency.

## Verify environment

Run:

```bash
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
```

Expected for GPU node:

```text
cuda available: True
gpu: NVIDIA Tesla T4
```

## Inspect-only first run

Use inspect-only before training:

```bash
python3 scripts/smoke_one_sr_dnn.py \
  --config configs/smoke_one_sr_dnn.yaml \
  --region SR_Wenu_250_400_boosted_0J \
  --fraction 0.01 \
  --max-events-per-class 20000 \
  --inspect-only
```

Check:

```text
outputs/smoke_one_sr_dnn/<timestamp>_SR_Wenu_250_400_boosted_0J/manifest.json
outputs/smoke_one_sr_dnn/<timestamp>_SR_Wenu_250_400_boosted_0J/branch_report.json
outputs/smoke_one_sr_dnn/<timestamp>_SR_Wenu_250_400_boosted_0J/selected_branches.json
outputs/smoke_one_sr_dnn/<timestamp>_SR_Wenu_250_400_boosted_0J/sample_summary.json
outputs/smoke_one_sr_dnn/<timestamp>_SR_Wenu_250_400_boosted_0J/plots/feature_svb_raw/*.png
```

Before training, confirm:

```text
1. both signal and background files are found;
2. selected weight column is sensible;
3. selected event column or hash fallback is sensible;
4. feature list has no truth/gen/weight/sample leakage;
5. SvB feature plots look physical.
```

## First DNN smoke training

After inspect-only passes:

```bash
python3 scripts/smoke_one_sr_dnn.py \
  --config configs/smoke_one_sr_dnn.yaml \
  --region SR_Wenu_250_400_boosted_0J \
  --fraction 0.02 \
  --max-events-per-class 50000 \
  --epochs 20 \
  --batch-size 8192 \
  --device auto
```

`--device auto` should use CUDA if the venv was activated after the LCG CUDA view and the session is on a GPU node.

## Expected output files

```text
outputs/smoke_one_sr_dnn/<timestamp>_<region>/preprocess_stats.json
outputs/smoke_one_sr_dnn/<timestamp>_<region>/training_history.json
outputs/smoke_one_sr_dnn/<timestamp>_<region>/metrics.json
outputs/smoke_one_sr_dnn/<timestamp>_<region>/checkpoints/model.pt
outputs/smoke_one_sr_dnn/<timestamp>_<region>/plots/training_curves.png
outputs/smoke_one_sr_dnn/<timestamp>_<region>/plots/scores/score_distribution_validation.png
outputs/smoke_one_sr_dnn/<timestamp>_<region>/plots/scores/roc_validation.png
outputs/smoke_one_sr_dnn/<timestamp>_<region>/plots/scores/sic_validation.png
outputs/smoke_one_sr_dnn/<timestamp>_<region>/scored_validation.parquet
```

## What the local agent should report back

```text
exact command used
number of discovered files and events
selected weight column
selected event/fold column
number of features before preprocessing
number of features after preprocessing
validation loss and AUC from metrics.json
whether SvB feature plots look reasonable
whether score plots look reasonable
any traceback or warnings
```

## Common fixes

If the script cannot find files:

```text
Check actual region names under the EOS path and rerun with --region <existing_region>.
```

If the weight column is wrong:

```text
Update features.weight_candidates in configs/smoke_one_sr_dnn.yaml.
```

If truth/gen/leakage columns are selected:

```text
Tighten features.forbidden_patterns in configs/smoke_one_sr_dnn.yaml.
```

If too many useless columns survive:

```text
Add a temporary explicit allowlist, or tighten preferred_patterns.
```

If CUDA is not visible:

```text
Check that the LCG CUDA view was sourced before activating the venv.
Check that the session is on a GPU node.
Run nvidia-smi and the Python torch.cuda.is_available() test.
```
