# Handoff: Boosted VHbb Training v1

This file is for the local lxplus agent and the user. ChatGPT is not executing lxplus/GPU/Condor commands. ChatGPT only reads/writes plans, configs, and code through the mounted repository.

## Immediate first step: one-SR DNN smoke test

New files for the first executable smoke workflow:

```text
configs/smoke_one_sr_dnn.yaml
scripts/smoke_one_sr_dnn.py
requirements-smoke.txt
```

Purpose:

```text
1. Read a small fraction of parquet files from one boosted SR.
2. Inspect parquet branches / columns.
3. Select safe numeric training features.
4. Make signal-vs-background plots for each selected parameter.
5. Preprocess features: finite filtering, clipping, median imputation, standardization.
6. Train a weighted plain DNN using GPU if available.
7. Save metrics, plots, checkpoint, and scored validation parquet.
```

Default one-SR region:

```text
SR_Wenu_250_400_boosted_0J
```

Default real lxplus/EOS input path:

```text
/eos/cms/store/group/phys_higgs/hbb/VHbbResults/Run3VHbbResults/ntuples/VHBB_Parquets/output_VHbb_STXS_boosted_svb_2024_0619_condor@lxplus/columns_final
```

### Inspect-only command

Use this first to check file discovery, branches, feature selection, and SvB plots without training:

```bash
cd /path/to/VHbb-Boost-Training
python3 scripts/smoke_one_sr_dnn.py \
  --config configs/smoke_one_sr_dnn.yaml \
  --region SR_Wenu_250_400_boosted_0J \
  --fraction 0.01 \
  --max-events-per-class 20000 \
  --inspect-only
```

Expected outputs:

```text
outputs/smoke_one_sr_dnn/<timestamp>_<region>/manifest.json
outputs/smoke_one_sr_dnn/<timestamp>_<region>/branch_report.json
outputs/smoke_one_sr_dnn/<timestamp>_<region>/selected_branches.json
outputs/smoke_one_sr_dnn/<timestamp>_<region>/sample_summary.json
outputs/smoke_one_sr_dnn/<timestamp>_<region>/plots/feature_svb_raw/*.png
```

### GPU DNN smoke-training command

After inspect-only looks reasonable:

```bash
cd /path/to/VHbb-Boost-Training
python3 scripts/smoke_one_sr_dnn.py \
  --config configs/smoke_one_sr_dnn.yaml \
  --region SR_Wenu_250_400_boosted_0J \
  --fraction 0.02 \
  --max-events-per-class 50000 \
  --epochs 20 \
  --batch-size 8192 \
  --device auto
```

`--device auto` uses CUDA if PyTorch sees a GPU, otherwise CPU. For actual lxplus GPU training, run this command from an interactive GPU node or a batch job prepared by the local agent.

Expected training outputs:

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

## Current v1 objective

Build the first transparent boosted VHbb training workflow with:

```text
one merged boosted-SR phase space
plain DNN baseline
ParT-Lite+MHA target model
two-fold even/odd out-of-fold production
visualization at every step
```

Main design document:

```text
docs/boosted_vhbb_training_model_inputs.md
```

Machine-readable contract:

```text
configs/boosted_vhbb_training_inputs.yaml
```

## Working directories

```text
main repo:
/home/lzhang/lxplus/columns_final/VHbb-Boost-Training

read-only parquet mount:
/home/lzhang/lxplus/columns_final/columns_mount

reference repo only:
/home/lzhang/lxplus/columns_final/tch_2026
```

## Four hard requirements for v1

### 1. DNN and ParT-Lite+MHA both required

Do not implement only one model. v1 requires:

```text
src/vhbb_boost/models/dnn.py
src/vhbb_boost/models/partlite_mha.py
```

The DNN is the required comparison baseline. The ParT-Lite+MHA model is the target improved NN.

### 2. Merge all boosted SRs into one phase space

Use:

```text
/home/lzhang/lxplus/columns_final/columns_mount/*/SR_*_boosted*/nominal.parquet
```

Do not make separate v1 trainings for 0L/1L/2L, pT(V), or jet-bin regions. Save those labels for plotting and diagnostics only.

### 3. Use two-fold even/odd production

Do not create a final train/val/test split for production. Instead:

```text
fold = event % 2
```

or fallback:

```text
fold = stable_hash(sample_name, region_name, row_index) % 2
```

Production logic:

```text
model for even events: train on odd, apply to even
model for odd events:  train on even, apply to odd
```

Within each training side, make a small internal validation split for early stopping and plots.

### 4. Visualization is mandatory

Every step should produce plots or JSON summaries. At minimum:

```text
input composition plots
feature distribution plots
fold balance plots
training curves
DNN vs ParT-Lite+MHA ROC/SIC comparison
score shape by channel/pT/jet-bin
AK8 mass sculpting checks
signed-yield checks
fold closure checks
```

## Later v1 implementation order after smoke test

```text
M0: manifest and column report for all merged boosted SRs
M1: feature map and pretraining plots
M2: two-fold dataset layer
M3: plain DNN two-fold training
M4: ParT-Lite+MHA two-fold training
M5: model comparison and physics validation
```

## Non-goals for v1

Do not spend v1 time on:

```text
separate per-channel training
separate per-STXS-bin training
final test split
negative-weight reweighting
systematic-variation training
κλ/STXS-aware loss
full production Condor workflow inside model code
```

The local agent may add thin lxplus wrappers later, but the core training code should remain normal Python entrypoints.
