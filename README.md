# VHbb-Boost-Training

Training framework for boosted VHbb signal-vs-background discriminants.

This repository is the working area for code, configs, and coordination plans. The parquet column inputs are mounted read-only outside the repo.

## Current starting point

The first target is a **high-level boosted VHbb SvB baseline**:

```text
signal:     VH, H → bb, boosted AK8 topology
background: V+jets, top, diboson, optional QCD
model:      high-level MLP baseline
output:     score_svb_boosted_v0
```

The first milestone is deliberately not a full ParT-lite particle-cloud model. The first milestone is a robust high-level training pipeline from the existing parquet columns. ParT-lite is kept as a later milestone if AK8 constituent and secondary-vertex jagged arrays are available.

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
M0: manifest and parquet column inspection
M1: high-level dataloader and feature mapping
M2: high-level MLP baseline training
M3: physics validation plots
M4: lxplus wrapper scripts for local agent execution
```

See `.ai-bridge/boosted_training_handoff.md` for the concrete task list.
