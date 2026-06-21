# Channel-aware 250-400 boosted DNN run

Updated: 2026-06-21T22:09:01.659Z
Workspace: /home/lzhang/lxplus/columns_final/VHbb-Boost-Training
Target agent: lxplus local agent (custom)

## Plan

You are the lxplus local agent working in the `VHbb-Boost-Training` repository.

Goal: run the next meaningful boosted VHbb DNN smoke/benchmark training. The previous one-SR run proved the pipeline works, but one SR is not enough statistics and the background composition must include real V+jets.

Read these files first:

```text
docs/region_selection_and_channel_rules.md
configs/smoke_one_sr_dnn.yaml
scripts/smoke_one_sr_dnn.py
configs/boosted_vhbb_training_inputs.yaml
```

Physics/training rules:

1. A single SR such as `SR_Wenu_250_400_boosted_0J` is only a technical smoke test. Do not use it as the meaningful benchmark.
2. For the next DNN benchmark, use the merged all-channel/all-jet-bin 250--400 pT(V) region family:

```text
SR_*_250_400_boosted_*J
```

3. The DNN must be channel-aware. Reconstructed channel indicators are allowed model inputs, for example:

```text
events_LeptonCategory
events_isWLNuFlag
events_isZLLFlag
events_isZNuNuFlag
```

Derived one-hot channel columns from `region_name` are also acceptable if implemented:

```text
channel_Znn
channel_Wenu
channel_Wmnu
channel_Zee
channel_Zmm
lep_category_0L
lep_category_1L
lep_category_2L
```

4. Do not use `sample_name`, process labels, truth/gen/STXS labels, or event weights as model inputs.
5. Do not mix overlapping region families without event-level de-duplication. In particular, do not combine `SR_*_GT400_boosted*` with `SR_*_400_600_boosted*` or `SR_*_GT600_boosted*` until de-duplication is implemented and checked.
6. For this run, stay only in the 250--400 family. Do not include GT400 in the same training.
7. Check that V+jets backgrounds are actually included. The previous run only had diboson background, so its AUC was only a technical check, not a real SvB result.

Use the LCG-backed venv:

```bash
cd /afs/cern.ch/work/l/lichengz/private/VHbb-Training/VHbb-Boost-Training
source /cvmfs/sft.cern.ch/lcg/views/LCG_110_cuda/x86_64-el9-gcc13-opt/setup.sh
source .venv/lcg110-cuda/bin/activate
```

First run inspect-only:

```bash
python3 scripts/smoke_one_sr_dnn.py \
  --config configs/smoke_one_sr_dnn.yaml \
  --region 'SR_*_250_400_boosted_*J' \
  --fraction 0.05 \
  --max-events-per-class 200000 \
  --inspect-only
```

Inspect the output directory under:

```text
outputs/smoke_one_sr_dnn/<timestamp>_SR___250_400_boosted__J/
```

Check these files:

```text
manifest.json
sample_summary.json
selected_branches.json
preprocess_stats.json, if created
plots/feature_svb_raw/*.png
```

Before training, confirm:

```text
sample_summary.json includes signal_VH_Hbb
sample_summary.json includes background_Vjets
sample_summary.json includes background_diboson if present
selected_branches.json includes channel-aware reco indicators
selected_branches.json does not include truth/gen/STXS/sample/process/weight leakage
```

If `background_Vjets` is missing, do not proceed to training. Instead, inspect actual W/Z+jets sample directory names and region names, then update the sample patterns or region matching.

If inspect-only is good, run training:

```bash
python3 scripts/smoke_one_sr_dnn.py \
  --config configs/smoke_one_sr_dnn.yaml \
  --region 'SR_*_250_400_boosted_*J' \
  --fraction 0.20 \
  --max-events-per-class 500000 \
  --epochs 20 \
  --batch-size 8192 \
  --device auto
```

Report back with:

```text
1. exact commands used
2. torch.cuda.is_available() and GPU name
3. output directory path
4. number of parquet files and sampled events
5. process groups in sample_summary.json
6. signal/background event counts
7. signed and absolute weight sums by label/process group
8. selected weight column
9. selected event/fold column
10. selected channel-aware columns
11. number of features before preprocessing
12. number of kept features after preprocessing
13. whether pfcand/SV arrays are present and dropped by flat preprocessing
14. validation loss and AUC
15. whether feature plots and score plots look reasonable
16. any errors, warnings, or signs of duplicate-event/region-overlap issues
```

Interpretation rule: only treat the DNN AUC/score plots as meaningful if real V+jets backgrounds are present. Otherwise, mark it as a technical pipeline test only.

## Implementation contract

- Work from this plan in small, reviewable steps.
- Keep edits scoped to the requested task and existing project conventions.
- Run focused verification before handing work back.
- Update .ai-bridge/agent-status.md with files touched, checks run, results, blockers, and review notes.
- Save the final review diff to .ai-bridge/implementation-diff.patch when practical.
- Append notable execution events to .ai-bridge/execution-log.jsonl when the implementation agent supports logging.
