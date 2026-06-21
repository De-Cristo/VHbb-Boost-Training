# Channel-aware merged SR training

Updated: 2026-06-21T22:07:40.437Z
Workspace: /home/lzhang/lxplus/columns_final/VHbb-Boost-Training
Target agent: lxplus local agent (custom)

## Plan

Please read the new rules file first:

```text
docs/region_selection_and_channel_rules.md
```

User corrections to apply:

- One small SR is only a technical smoke test. It does not have enough statistics for a useful benchmark.
- For the next DNN run, use the merged all-channel/all-jet-bin family:

```text
SR_*_250_400_boosted_*J
```

- The DNN should be channel-aware. Reconstructed channel indicators are allowed inputs, for example:

```text
events_LeptonCategory
events_isWLNuFlag
events_isZLLFlag
events_isZNuNuFlag
```

or one-hot columns derived from region name:

```text
channel_Znn, channel_Wenu, channel_Wmnu, channel_Zee, channel_Zmm
lep_category_0L, lep_category_1L, lep_category_2L
```

- Do not combine overlapping region families without event-level de-duplication.

Safe region families:

```text
250 < pT(V) < 400:
  SR_*_250_400_boosted_*J

pT(V) > 400:
  SR_*_GT400_boosted*
```

Do not combine the high-pT inclusive family with narrower high-pT families until duplicate-event handling is implemented.

Next run:

```bash
cd /afs/cern.ch/work/l/lichengz/private/VHbb-Training/VHbb-Boost-Training
source /cvmfs/sft.cern.ch/lcg/views/LCG_110_cuda/x86_64-el9-gcc13-opt/setup.sh
source .venv/lcg110-cuda/bin/activate

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

- process groups in `sample_summary.json`, especially whether `background_Vjets` appears;
- event counts and signed/absolute weight sums;
- selected channel-aware columns;
- kept features after preprocessing;
- validation loss/AUC;
- feature and score plot quality;
- any signs of duplicate events or region overlap.

## Implementation contract

- Work from this plan in small, reviewable steps.
- Keep edits scoped to the requested task and existing project conventions.
- Run focused verification before handing work back.
- Update .ai-bridge/agent-status.md with files touched, checks run, results, blockers, and review notes.
- Save the final review diff to .ai-bridge/implementation-diff.patch when practical.
- Append notable execution events to .ai-bridge/execution-log.jsonl when the implementation agent supports logging.
