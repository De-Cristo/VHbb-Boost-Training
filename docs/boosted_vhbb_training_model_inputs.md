# Boosted VHbb Training v1 Plan: Merged-SR DNN vs ParT-Lite+MHA

This document defines the v1 training plan for `VHbb-Boost-Training`. It supersedes the earlier one-model high-level baseline plan.

ChatGPT's role is to read/write plans, configs, and code through the mounted repository. The user and local lxplus agent handle execution, GPU jobs, Condor submission, and git push/pull.

---

## 1. v1 guiding decisions

### Decision 1: train both a plain DNN and a ParT-Lite+MHA model

We need two model families from the beginning:

```text
Model A: plain high-level DNN
Model B: ParT-Lite + MHA neural network
```

The plain DNN is not just a toy baseline. It is the required comparison point for the ParT-Lite+MHA model. The two models should share the same event manifest, same merged-SR phase space, same weights, same folds, and same validation plots.

### Decision 2: merge all boosted SRs into one v1 phase space

For v1, all boosted SRs are treated as one common training phase space:

```text
all boosted SRs → one inclusive boosted training dataset
```

Region labels such as channel, pT(V) bin, and jet multiplicity are still saved for monitoring and plotting, but v1 should not train separate models per SR, per channel, or per pT bin.

Recommended v1 policy:

```text
use all SR_*_boosted* nominal parquet files together
train one inclusive score per model family
validate the score separately in each channel / pT / jet bin
```

### Decision 3: final training uses a two-fold even/odd method, not train/val/test

For final model production, we should use a two-fold strategy:

```text
even events: evaluated by model trained on odd events
odd events:  evaluated by model trained on even events
```

This avoids applying a model to events it was trained on, while also avoiding the need to reserve a final test dataset that reduces MC statistics.

The validation set should be internal to each training fold, split from that fold's training side only.

### Decision 4: visualization is a first-class deliverable

Every step should produce transparent plots. The goal is that we can always answer:

```text
What events entered?
What features were used?
How were the weights handled?
How did each model train?
How do DNN and ParT-Lite+MHA compare?
Does the score behave reasonably in every SR slice?
Does the score sculpt AK8 mass or other critical shapes?
```

---

## 2. Scope of v1

### Physics task

```text
Task: boosted VHbb signal-vs-background classification
Signal: VH, H → bb boosted topology
Background: V+jets, top, diboson, optional QCD if present
Input files: all nominal boosted SR parquet files
Phase space: one merged boosted-SR phase space
Output scores:
  - score_dnn_v1
  - score_partlite_mha_v1
```

### Initial file glob

```text
/home/lzhang/lxplus/columns_final/columns_mount/*/SR_*_boosted*/nominal.parquet
```

### Region metadata retained for plotting

```text
channel:          Znn, Wenu, Wmnu, Zee, Zmm
lepton_category:  0lep, 1lep, 2lep
reco_pTV_bin:     250_400, 400_600, GT400, GT600, etc.
reco_jet_bin:     0J, 1J, GE2J, GT1J, inclusive
region_name:      original SR name
sample_name:      original sample name
```

These are used for diagnostics and plotting. They should not define separate v1 training jobs.

---

## 3. Model A: plain high-level DNN

### Purpose

The plain DNN is the direct comparison baseline for ParT-Lite+MHA.

It answers:

```text
How far can we get using only flat, high-level analysis columns?
How much additional value does ParT-Lite+MHA provide?
Is the ParT-Lite+MHA improvement stable across SR slices?
```

### Input type

Flat, reconstructed, analysis-level features from parquet.

Primary feature groups:

```text
AK8 / Higgs candidate:
  ak8_pt
  ak8_eta
  ak8_phi
  ak8_mass
  ak8_msoftdrop
  ak8_bb_tag_score
  optional substructure and subjet variables

Vector boson and leptons/MET:
  V_pt
  V_eta
  V_phi
  MET_pt
  MET_phi
  lepton kinematics if present

Topology:
  n_ak4_jets
  n_additional_jets
  n_btagged_ak4_jets
  ht
  soft activity if present

Derived angular variables:
  dphi_V_AK8
  deta_V_AK8
  dR_V_AK8
  dphi_MET_AK8
```

### Suggested architecture

```text
model_name: boosted_plain_dnn_v1
input: standardized flat feature vector
output: binary logit / sigmoid score_dnn_v1
loss: weighted binary cross entropy using abs(event_weight)
```

A good starting architecture:

```text
InputNorm
Dense(256) + LayerNorm + GELU + Dropout(0.05)
Dense(256) + LayerNorm + GELU + Dropout(0.05)
Dense(128) + LayerNorm + GELU + Dropout(0.05)
Dense(64)  + LayerNorm + GELU + Dropout(0.05)
Dense(32)  + GELU
Dense(1)
```

Use `BCEWithLogitsLoss` rather than applying sigmoid before the loss.

---

## 4. Model B: ParT-Lite+MHA neural network

### Purpose

The ParT-Lite+MHA model is the target improved NN.

It should test whether a lightweight particle/sequence-aware architecture plus event-level multi-head attention gives better boosted VHbb separation than the plain DNN.

### v1 implementation policy

The model should support two modes:

```text
mode 1: high-level-token MHA, if only flat parquet columns exist
mode 2: AK8 particle-cloud ParT-lite, if jagged AK8 constituent/SV arrays exist
```

This is important because we do not want to block development if the current parquet files only contain high-level columns.

### Mode 1: high-level-token MHA

If low-level AK8 constituents are not available, build tokens from feature groups:

```text
Token 1: AK8/Higgs candidate features
Token 2: vector-boson features
Token 3: MET/lepton features
Token 4: event-topology features
Token 5: resolved-overlap features, if present
```

Each token is produced by a small embedding MLP. The token sequence is passed to a small MHA encoder. The pooled output is sent to a classifier head.

This gives a meaningful MHA comparison even before full particle-cloud inputs exist.

### Mode 2: AK8 ParT-lite

If low-level AK8 arrays exist, use:

```text
AK8 PF candidates: max 64
AK8 secondary vertices: max 8
optional subjet tokens
global high-level features
```

Recommended lightweight architecture:

```text
particle embed dim: 64 or 128
MHA blocks: 2-4
attention heads: 4 or 8
classifier head: small MLP
output: score_partlite_mha_v1
```

The first ParT-Lite+MHA implementation should be intentionally smaller than full JetClass ParT.

---

## 5. Feature policy

### Allowed model inputs

Use reconstructed analysis-level variables that can exist in data:

```text
AK8 kinematics
AK8 soft-drop mass
AK8 boosted b-tag / ParticleNet / DeepAK8-like scores
vector-boson reconstructed kinematics
MET and lepton kinematics
additional jet and b-tag multiplicities
resolved-overlap variables if present
```

### Metadata for monitoring only

Use these for slicing plots, not as default physics inputs:

```text
region_name
sample_name
channel
lepton_category
reco_pTV_bin
reco_jet_bin
year / era
```

For v1, since all SRs are merged into one phase space, the default should be:

```text
channel / pT / jet-bin metadata are monitoring variables, not input features
```

If the local agent finds that missing-value handling requires channel-aware masks, this should be documented explicitly.

### Forbidden final model inputs

Do not feed these to the classifier:

```text
truth labels
truth STXS bin
generator-level pT(V) or pT(H)
sample name
raw event weight
fold assignment
train/validation flag
```

---

## 6. Weighting and negative weights

### Training loss

Use:

```text
loss_weight = abs(signed_event_weight)
```

Do not use signed weights directly in the loss.

### Validation histograms

Use both:

```text
abs weights:    ML metric stability
signed weights: physics-shape and yield validation
```

### Class/process balancing

For training stability:

```text
first normalize per process group
then normalize signal vs background class weight
```

This prevents the largest V+jets background from completely dominating the training.

### Negative-weight reweighting

Do not merge negative-weight reweighting into the v1 SvB training. It is a separate future workstream.

---

## 7. Two-fold method

### Fold definition

Preferred fold key:

```text
fold = event % 2
```

If the `event` column is not available, use a stable hash:

```text
fold = stable_hash(sample_name, region_name, row_index) % 2
```

### Final two-fold production

For each model family, train two models:

```text
DNN fold-even application model:
  train on odd events
  apply to even events

DNN fold-odd application model:
  train on even events
  apply to odd events

ParT-Lite+MHA fold-even application model:
  train on odd events
  apply to even events

ParT-Lite+MHA fold-odd application model:
  train on even events
  apply to odd events
```

Final scored output combines:

```text
score(event) = score from model that did not train on this event
```

### Internal validation inside each fold

Within each training side, reserve a small validation subset for early stopping and training monitoring:

```text
outer fold: even/odd for final application
inner validation: 10-20% of the training side only
```

No final test dataset is required for v1 production. Instead, evaluate final out-of-fold scores on the full merged sample.

---

## 8. Visualization-first workflow

Every major script should write plots and JSON summaries. The repo should make bad behavior visible early.

### Step A: manifest and input plots

Produce:

```text
n_files_by_process_group.png
n_events_by_process_group.png
sum_abs_weight_by_process_group.png
sum_signed_weight_by_process_group.png
n_events_by_region.png
sum_weight_by_region.png
fold_balance_even_odd.png
missing_columns_report.json
```

### Step B: feature plots before training

For all selected features:

```text
feature_distribution_signal_vs_background/<feature>.png
feature_distribution_by_process_group/<feature>.png
feature_correlation_matrix_dnn_inputs.png
missing_value_fraction_by_feature.png
```

Priority physics features:

```text
ak8_pt
ak8_msoftdrop
ak8_bb_tag_score
V_pt
n_additional_jets
n_btagged_ak4_jets
```

### Step C: training plots

For each model family and fold:

```text
loss_curve.png
auc_curve.png
learning_rate_curve.png
gradient_norm_curve_optional.png
score_train_vs_internal_val.png
overtraining_ks.json
```

### Step D: out-of-fold comparison plots

For the final out-of-fold scores:

```text
roc_dnn_vs_partlite_mha.png
sic_dnn_vs_partlite_mha.png
score_dnn_vs_partlite_mha_2d.png
score_distribution_by_class_dnn.png
score_distribution_by_class_partlite_mha.png
score_by_process_group_comparison.png
```

### Step E: physics transparency plots

For each model score:

```text
score_by_channel.png
score_by_reco_pTV_bin.png
score_by_reco_jet_bin.png
background_composition_by_score_bin.png
signed_yield_by_score_bin.png
ak8_msoftdrop_in_score_quantiles.png
ak8_bb_tag_score_in_score_quantiles.png
V_pt_in_score_quantiles.png
```

### Step F: fold-closure plots

Compare even-applied and odd-applied outputs:

```text
score_even_vs_odd_by_class.png
score_even_vs_odd_by_region.png
auc_even_vs_odd.json
fold_yield_closure.json
```

These plots are mandatory because two-fold training can otherwise hide fold-dependent behavior.

---

## 9. Implementation milestones

### M0: manifest + column report

Deliverables:

```text
scripts/build_manifest.py
scripts/inspect_columns.py
outputs/manifests/boosted_v1_manifest.json
outputs/manifests/column_report.json
outputs/plots/inputs/*.png
```

### M1: feature map + pretraining visualization

Deliverables:

```text
configs/boosted_vhbb_training_inputs.yaml
src/vhbb_boost/features.py
scripts/plot_input_features.py
outputs/plots/features/*.png
```

### M2: two-fold dataset and dataloader

Deliverables:

```text
src/vhbb_boost/data.py
src/vhbb_boost/weights.py
src/vhbb_boost/folds.py
```

Requirements:

```text
merged-SR loading
even/odd outer fold assignment
internal validation split
shared dataset interface for DNN and ParT-Lite+MHA
```

### M3: plain DNN training

Deliverables:

```text
src/vhbb_boost/models/dnn.py
scripts/train_dnn.py
scripts/apply_dnn_oof.py
outputs/runs/dnn_v1/<run_id>/...
```

### M4: ParT-Lite+MHA training

Deliverables:

```text
src/vhbb_boost/models/partlite_mha.py
scripts/train_partlite_mha.py
scripts/apply_partlite_mha_oof.py
outputs/runs/partlite_mha_v1/<run_id>/...
```

### M5: model comparison and physics validation

Deliverables:

```text
scripts/compare_models.py
scripts/plot_score_shapes.py
scripts/plot_mass_sculpting.py
outputs/comparisons/dnn_vs_partlite_mha_v1/...
```

---

## 10. v1 success criteria

v1 is successful when we have:

```text
1. one merged-SR manifest covering all nominal boosted SR parquets
2. confirmed feature map from actual parquet columns
3. two-fold out-of-fold DNN score for every selected event
4. two-fold out-of-fold ParT-Lite+MHA score for every selected event
5. DNN vs ParT-Lite+MHA comparison plots
6. fold-closure plots
7. score-shape plots in every major SR slice
8. mass-sculpting and signed-yield checks
```

Do not judge v1 only by inclusive AUC. The model must be transparent and stable across the merged SR phase space.
