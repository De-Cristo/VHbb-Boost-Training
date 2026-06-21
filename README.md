# VHbb-Boost-Training

Training framework for boosted VHbb signal-vs-background discriminants.

This repository is the working area for code, configs, visualization outputs, and coordination plans. The parquet column inputs are mounted read-only outside the repo.

## Current v1 plan

The first training iteration is:

```text
phase space: one merged boosted-SR phase space
models:      plain DNN baseline vs ParT-Lite+MHA target NN
folding:     two-fold even/odd out-of-fold production
output:      score_dnn_v1 and score_partlite_mha_v1
```

Important details:

- All `SR_*_boosted*` nominal parquet files are merged into one inclusive boosted training phase space.
- The plain DNN is required as a transparent comparison baseline.
- The ParT-Lite+MHA model is the target improved neural network.
- Final production uses even/odd two-fold application, not a final held-out test dataset.
- Visualization plots are mandatory at every step.

## Important paths

```text
repo:
/home/lzhang/lxplus/columns_final/VHbb-Boost-Training

read-only parquet mount:
/home/lzhang/lxplus/columns_final/columns_mount

reference repo only:
/home/lzhang/lxplus/columns_final/tch_2026
```

ChatGPT should only read/write plans, configs, and code through the mounted repo. The user/local lxplus agent handles GPU jobs, Condor, execution, and git push/pull.

## Planning files

Start here:

```text
docs/boosted_vhbb_training_model_inputs.md
```

Machine-readable training contract:

```text
configs/boosted_vhbb_training_inputs.yaml
```

Local-agent handoff:

```text
.ai-bridge/boosted_training_handoff.md
```

## Intended implementation order

```text
M0: manifest and parquet column report
M1: feature map and pretraining visualization
M2: two-fold dataset and dataloader
M3: plain DNN training and out-of-fold application
M4: ParT-Lite+MHA training and out-of-fold application
M5: DNN vs ParT-Lite+MHA comparison and physics validation
```

See `.ai-bridge/boosted_training_handoff.md` for the concrete task list.
