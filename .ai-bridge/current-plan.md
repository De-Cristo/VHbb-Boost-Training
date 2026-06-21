# Rerun smoke with real V+jets backgrounds

Updated: 2026-06-21T21:56:53.785Z
Workspace: /home/lzhang/lxplus/columns_final/VHbb-Boost-Training
Target agent: lxplus local agent (custom)

## Plan

The previous smoke runs proved that the venv, CUDA, parquet reading, plotting, preprocessing, and DNN training work. However, the output summaries show that the selected backgrounds were only `background_diboson`, not V+jets/top. Therefore the high AUC is a technical smoke-test result, not yet a real SvB benchmark.

I updated `configs/smoke_one_sr_dnn.yaml` to:

1. include real W/Z+jets sample-name patterns seen in the mounted sample list, especially `WtoENu`, `WtoMuNu`, `WtoTauNu`, `WtoLNu-2Jets`, and `Zto2Nu-2Jets`;
2. forbid channel/process-like flags from default flat-DNN inputs: `LeptonCategory`, `Hcc_flag`, `isWLNuFlag`, `isZLLFlag`, `isZNuNuFlag`, `isBoostedTopology`, and `era`.

Next task: rerun the wildcard smoke test and check that `background_Vjets` is now present in `sample_summary.json`.

Use the same LCG-backed venv:

```bash
cd /afs/cern.ch/work/l/lichengz/private/VHbb-Training/VHbb-Boost-Training
source /cvmfs/sft.cern.ch/lcg/views/LCG_110_cuda/x86_64-el9-gcc13-opt/setup.sh
source .venv/lcg110-cuda/bin/activate
```

First inspect-only:

```bash
python3 scripts/smoke_one_sr_dnn.py \
  --config configs/smoke_one_sr_dnn.yaml \
  --region 'SR_*_250_400_boosted_*J' \
  --fraction 0.05 \
  --max-events-per-class 200000 \
  --inspect-only
```

Check `sample_summary.json`. We need to see at least:

```text
signal_VH_Hbb
background_Vjets
background_diboson
```

If V+jets still does not appear, inspect actual region names inside W/Z+jets sample directories and update sample/region matching.

If inspect-only is good, run a corrected DNN smoke training:

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

Report back:

- process groups in `sample_summary.json`;
- signal/background event counts and signed/abs weight sums;
- selected features before preprocessing;
- kept features after preprocessing;
- whether pfcand/SV arrays are present but dropped by the flat DNN preprocessor;
- validation loss/AUC;
- whether the feature and score plots still look reasonable.

Important interpretation: pfcand/SV columns appearing in selected branches is good news for the future ParT-Lite model. The current flat DNN smoke script cannot consume jagged/sequence arrays, so it drops them during preprocessing. Do not remove those branches from the ntuples.

## Implementation contract

- Work from this plan in small, reviewable steps.
- Keep edits scoped to the requested task and existing project conventions.
- Run focused verification before handing work back.
- Update .ai-bridge/agent-status.md with files touched, checks run, results, blockers, and review notes.
- Save the final review diff to .ai-bridge/implementation-diff.patch when practical.
- Append notable execution events to .ai-bridge/execution-log.jsonl when the implementation agent supports logging.
