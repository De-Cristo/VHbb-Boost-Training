# Start one-SR boosted VHbb DNN smoke test

Updated: 2026-06-21T21:17:03.032Z
Workspace: /home/lzhang/lxplus/columns_final/VHbb-Boost-Training
Target agent: lxplus local agent (custom)

## Plan

You are the lxplus local implementation agent for the `VHbb-Boost-Training` repository.

Your first job is not to implement the full v1 framework. Your first job is to run and debug the one-SR DNN smoke workflow that ChatGPT has already written.

Start by reading these files, in this order:

1. `.ai-bridge/boosted_training_handoff.md`
2. `configs/smoke_one_sr_dnn.yaml`
3. `scripts/smoke_one_sr_dnn.py`
4. `README.md`
5. `docs/boosted_vhbb_training_model_inputs.md`
6. `configs/boosted_vhbb_training_inputs.yaml`

Immediate goal:

Run a small one-SR smoke test using real parquet files from EOS:

`/eos/cms/store/group/phys_higgs/hbb/VHbbResults/Run3VHbbResults/ntuples/VHBB_Parquets/output_VHbb_STXS_boosted_svb_2024_0619_condor@lxplus/columns_final`

Default region:

`SR_Wenu_250_400_boosted_0J`

Step 1: inspect-only branch/feature/plot test. Run from the repo root:

```bash
python3 scripts/smoke_one_sr_dnn.py \
  --config configs/smoke_one_sr_dnn.yaml \
  --region SR_Wenu_250_400_boosted_0J \
  --fraction 0.01 \
  --max-events-per-class 20000 \
  --inspect-only
```

Then inspect these outputs:

- `outputs/smoke_one_sr_dnn/<timestamp>_<region>/manifest.json`
- `outputs/smoke_one_sr_dnn/<timestamp>_<region>/branch_report.json`
- `outputs/smoke_one_sr_dnn/<timestamp>_<region>/selected_branches.json`
- `outputs/smoke_one_sr_dnn/<timestamp>_<region>/sample_summary.json`
- `outputs/smoke_one_sr_dnn/<timestamp>_<region>/plots/feature_svb_raw/*.png`

Check whether:

- signal and background files are both discovered;
- the selected region exists for enough samples;
- the selected weight column is sensible;
- the selected event column is sensible, or the stable-hash fallback is used;
- feature plots are produced and not obviously broken;
- selected features do not include truth/gen/sample/weight/leakage columns.

Step 2: if inspect-only is reasonable, run DNN smoke training:

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

Use a GPU-capable lxplus environment if available. `--device auto` should use CUDA if PyTorch sees a GPU, otherwise CPU.

Inspect these training outputs:

- `preprocess_stats.json`
- `training_history.json`
- `metrics.json`
- `checkpoints/model.pt`
- `plots/training_curves.png`
- `plots/scores/score_distribution_validation.png`
- `plots/scores/roc_validation.png`
- `plots/scores/sic_validation.png`
- `scored_validation.parquet`

Important debugging priorities:

1. If no files are found, check actual region names under the EOS path and update `configs/smoke_one_sr_dnn.yaml` or rerun with `--region <existing_region>`.
2. If the selected weight column is wrong, update `features.weight_candidates` in `configs/smoke_one_sr_dnn.yaml`.
3. If feature selection includes bad columns, update `features.forbidden_patterns`.
4. If too many useless columns survive, tighten `features.preferred_patterns` or add an explicit feature allowlist.
5. If training is unstable, first reduce `fraction`, lower `batch_size`, or inspect `preprocess_stats.json` for dropped/pathological columns.
6. Do not implement ParT-Lite yet. First make this one-SR DNN smoke workflow run cleanly.

After running, report back:

- exact command used;
- number of files/events discovered;
- selected weight column;
- selected event/fold column;
- number of selected features before and after preprocessing;
- validation AUC/loss from `metrics.json`;
- whether the score and feature plots look reasonable;
- any errors or traceback.

## Implementation contract

- Work from this plan in small, reviewable steps.
- Keep edits scoped to the requested task and existing project conventions.
- Run focused verification before handing work back.
- Update .ai-bridge/agent-status.md with files touched, checks run, results, blockers, and review notes.
- Save the final review diff to .ai-bridge/implementation-diff.patch when practical.
- Append notable execution events to .ai-bridge/execution-log.jsonl when the implementation agent supports logging.
