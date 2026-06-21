# Audit populated V+jets path

Updated: 2026-06-21T22:27:08.307Z
Workspace: /home/lzhang/lxplus/columns_final/VHbb-Boost-Training
Target agent: lxplus local agent (custom)

## Plan

The recursive region discovery fix is correct and should be kept. The current blocker is data availability: the discovered V+jets parquet files under the current EOS columns root have schemas but zero rows, so they cannot be used for SvB training.

Do not proceed to a meaningful DNN training using only signal + diboson. That would be a technical test only and not a VHbb SvB benchmark.

Current EOS path with empty V+jets:

```text
/eos/cms/store/group/phys_higgs/hbb/VHbbResults/Run3VHbbResults/ntuples/VHBB_Parquets/output_VHbb_STXS_boosted_svb_2024_0619_condor@lxplus/columns_final
```

Please do the following next:

1. Keep the `sample_dir.rglob("*")` nested-region discovery fix in `scripts/smoke_one_sr_dnn.py`.
2. Write a short data-audit report under `docs/`, e.g.

```text
docs/YYYY-MM-DD_vjets_empty_data_audit.md
```

3. In that report, include:

```text
- EOS columns_root checked
- number of sample directories
- number of discovered parquet files by process group
- total metadata row count by process group
- total loaded pandas row count by process group for the 250_400 boosted pattern
- examples of populated signal/diboson files
- examples of empty V+jets files
- conclusion that V+jets files are schema-only / empty in this path
```

4. Search only for candidate alternative `columns_final` directories if they are nearby and easy to inspect, for example under:

```text
/eos/cms/store/group/phys_higgs/hbb/VHbbResults/Run3VHbbResults/ntuples/VHBB_Parquets/
```

For each candidate, do not run training. First perform a lightweight row-count audit for the region family:

```text
SR_*_250_400_boosted_*J
```

The candidate is usable only if `background_Vjets` has nonzero rows in `sample_summary`/audit results.

5. If no populated V+jets path is found quickly, stop and ask the user/ntupling owner for the correct production path. Do not try to compensate by training against diboson-only.

6. Once a populated path is provided, update `configs/smoke_one_sr_dnn.yaml` `paths.eos_columns_root` or rerun with `--columns-root <new_path>`, then repeat:

```bash
python3 scripts/smoke_one_sr_dnn.py \
  --config configs/smoke_one_sr_dnn.yaml \
  --columns-root <new_populated_columns_final_path> \
  --region 'SR_*_250_400_boosted_*J' \
  --fraction 0.05 \
  --max-events-per-class 200000 \
  --inspect-only
```

Proceed to training only after the inspect-only output confirms:

```text
signal_VH_Hbb: nonzero rows
background_Vjets: nonzero rows
background_diboson: nonzero rows if present
selected channel-aware columns are included
no truth/gen/STXS/sample/process/weight leakage
```

Then train:

```bash
python3 scripts/smoke_one_sr_dnn.py \
  --config configs/smoke_one_sr_dnn.yaml \
  --columns-root <new_populated_columns_final_path> \
  --region 'SR_*_250_400_boosted_*J' \
  --fraction 0.20 \
  --max-events-per-class 500000 \
  --epochs 20 \
  --batch-size 8192 \
  --device auto
```

Report back with the audit report path, any candidate populated paths found, and whether V+jets has nonzero rows.

## Implementation contract

- Work from this plan in small, reviewable steps.
- Keep edits scoped to the requested task and existing project conventions.
- Run focused verification before handing work back.
- Update .ai-bridge/agent-status.md with files touched, checks run, results, blockers, and review notes.
- Save the final review diff to .ai-bridge/implementation-diff.patch when practical.
- Append notable execution events to .ai-bridge/execution-log.jsonl when the implementation agent supports logging.
