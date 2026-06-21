# VHbb-Boost-Training

Training framework for boosted VHbb signal-vs-background discriminants.

This repository is the working area for code, configs, visualization outputs, and coordination plans. The parquet column inputs are mounted read-only outside the repo or accessed directly from EOS on lxplus.

## Immediate start: one-SR DNN smoke test

The current executable first step is a one-SR smoke workflow:

```text
config:  configs/smoke_one_sr_dnn.yaml
script:  scripts/smoke_one_sr_dnn.py
venv:    docs/lxplus_venv_training_start.md
```

It reads a small fraction of one boosted SR, inspects parquet branches, makes signal-vs-background feature plots, preprocesses features, and trains a weighted plain DNN using CUDA if PyTorch sees a GPU.

Default one-SR test region:

```text
SR_Wenu_250_400_boosted_0J
```

Default EOS input path:

```text
/eos/cms/store/group/phys_higgs/hbb/VHbbResults/Run3VHbbResults/ntuples/VHBB_Parquets/output_VHbb_STXS_boosted_svb_2024_0619_condor@lxplus/columns_final
```

## lxplus venv policy

The local-agent report confirmed that `LCG_110_cuda` already provides Python 3.13, GPU PyTorch, numpy, pandas, pyarrow, matplotlib, and PyYAML.

Use a venv on top of the LCG CUDA view:

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_110_cuda/x86_64-el9-gcc13-opt/setup.sh
python3 -m venv --system-site-packages .venv/lcg110-cuda
source .venv/lcg110-cuda/bin/activate
```

Do not pip-install `torch` on lxplus. See:

```text
docs/lxplus_venv_training_start.md
scripts/setup_lxplus_venv.sh
```

## Inspect-only command

```bash
python3 scripts/smoke_one_sr_dnn.py \
  --config configs/smoke_one_sr_dnn.yaml \
  --region SR_Wenu_250_400_boosted_0J \
  --fraction 0.01 \
  --max-events-per-class 20000 \
  --inspect-only
```

## First DNN smoke training command

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

Outputs are written under:

```text
outputs/smoke_one_sr_dnn/<timestamp>_<region>/
```

## Current v1 plan after smoke test

The full v1 training target is:

```text
phase space: one merged boosted-SR phase space
models:      plain DNN baseline vs ParT-Lite+MHA target NN
folding:     two-fold even/odd out-of-fold production
outputs:     score_dnn_v1 and score_partlite_mha_v1
```

Important details:

- All `SR_*_boosted*` nominal parquet files are merged into one inclusive boosted training phase space.
- The plain DNN is required as a transparent comparison baseline.
- The ParT-Lite+MHA model is the target improved neural network.
- Final production uses even/odd two-fold application, not a final held-out test dataset.
- Visualization plots are mandatory at every step.

## Important paths

```text
repo visible to ChatGPT:
/home/lzhang/lxplus/columns_final/VHbb-Boost-Training

mounted training space visible to ChatGPT:
/home/lzhang/lxplus/columns_final/lxplus-taining-space

real lxplus/AFS repo reported by local agent:
/afs/cern.ch/work/l/lichengz/private/VHbb-Training/VHbb-Boost-Training

EOS parquet path:
/eos/cms/store/group/phys_higgs/hbb/VHbbResults/Run3VHbbResults/ntuples/VHBB_Parquets/output_VHbb_STXS_boosted_svb_2024_0619_condor@lxplus/columns_final
```

ChatGPT should only read/write plans, configs, and code through the mounted repo. The user/local lxplus agent handles GPU jobs, Condor, execution, and git push/pull.

## Planning files

Start here:

```text
docs/lxplus_venv_training_start.md
.ai-bridge/boosted_training_handoff.md
configs/smoke_one_sr_dnn.yaml
scripts/smoke_one_sr_dnn.py
```

Full v1 design:

```text
docs/boosted_vhbb_training_model_inputs.md
configs/boosted_vhbb_training_inputs.yaml
```
