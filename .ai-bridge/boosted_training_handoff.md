# Handoff: Boosted VHbb Training v1

This file is for the local lxplus agent and the user. ChatGPT is not executing lxplus/GPU/Condor commands. ChatGPT only reads/writes plans, configs, and code through the mounted repository.

## v1 objective

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

## Implementation order

### M0: manifest and column report

Implement:

```text
scripts/build_manifest.py
scripts/inspect_columns.py
```

Outputs:

```text
outputs/manifests/boosted_v1_manifest.json
outputs/manifests/column_report.json
outputs/plots/inputs/n_files_by_process_group.png
outputs/plots/inputs/n_events_by_process_group.png
outputs/plots/inputs/sum_abs_weight_by_process_group.png
outputs/plots/inputs/sum_signed_weight_by_process_group.png
outputs/plots/inputs/n_events_by_region.png
outputs/plots/inputs/fold_balance_even_odd.png
```

### M1: feature map and pretraining plots

Update:

```text
configs/boosted_vhbb_training_inputs.yaml
```

Add actual parquet column mappings:

```yaml
column_map:
  ak8_pt: <actual_column_name>
  ak8_eta: <actual_column_name>
  ak8_msoftdrop: <actual_column_name>
  ak8_bb_tag_score: <actual_column_name>
  V_pt: <actual_column_name>
  event_weight: <actual_column_name>
```

Implement:

```text
src/vhbb_boost/features.py
scripts/plot_input_features.py
```

Plots:

```text
outputs/plots/features/feature_distribution_signal_vs_background/*.png
outputs/plots/features/feature_distribution_by_process_group/*.png
outputs/plots/features/feature_correlation_matrix_dnn_inputs.png
outputs/plots/features/missing_value_fraction_by_feature.png
```

### M2: two-fold dataset layer

Implement:

```text
src/vhbb_boost/data.py
src/vhbb_boost/weights.py
src/vhbb_boost/folds.py
```

Requirements:

```text
read all merged SR files
assign process groups and labels
build fold = even/odd
use abs(weight) for loss
preserve signed weight for physics plots
make internal validation split inside training side only
share same dataset API for DNN and ParT-Lite+MHA
```

### M3: plain DNN

Implement:

```text
src/vhbb_boost/models/dnn.py
scripts/train_dnn.py
scripts/apply_dnn_oof.py
```

Outputs:

```text
outputs/runs/dnn_v1/<run_id>/fold_even_application/...
outputs/runs/dnn_v1/<run_id>/fold_odd_application/...
```

### M4: ParT-Lite+MHA

Implement:

```text
src/vhbb_boost/models/partlite_mha.py
scripts/train_partlite_mha.py
scripts/apply_partlite_mha_oof.py
```

The model should support:

```text
high-level-token MHA mode, if only flat columns exist
AK8 particle-cloud ParT-lite mode, if constituent/SV arrays exist
```

Outputs:

```text
outputs/runs/partlite_mha_v1/<run_id>/fold_even_application/...
outputs/runs/partlite_mha_v1/<run_id>/fold_odd_application/...
```

### M5: model comparison and physics validation

Implement:

```text
scripts/compare_models.py
scripts/plot_score_shapes.py
scripts/plot_mass_sculpting.py
```

Outputs:

```text
outputs/comparisons/dnn_vs_partlite_mha_v1/roc_dnn_vs_partlite_mha.png
outputs/comparisons/dnn_vs_partlite_mha_v1/sic_dnn_vs_partlite_mha.png
outputs/comparisons/dnn_vs_partlite_mha_v1/score_dnn_vs_partlite_mha_2d.png
outputs/comparisons/dnn_vs_partlite_mha_v1/score_by_channel.png
outputs/comparisons/dnn_vs_partlite_mha_v1/score_by_reco_pTV_bin.png
outputs/comparisons/dnn_vs_partlite_mha_v1/score_by_reco_jet_bin.png
outputs/comparisons/dnn_vs_partlite_mha_v1/ak8_msoftdrop_in_score_quantiles.png
outputs/comparisons/dnn_vs_partlite_mha_v1/fold_closure_summary.json
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
