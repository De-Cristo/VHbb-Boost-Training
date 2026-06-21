# Use LCG-backed venv and start one-SR DNN smoke training

Updated: 2026-06-21T21:33:37.612Z
Workspace: /home/lzhang/lxplus/columns_final/VHbb-Boost-Training
Target agent: lxplus local agent (custom)

## Plan

Read the local-agent environment report and use the venv policy now added to the repo.

Start by reading:

1. `docs/lxplus_venv_training_start.md`
2. `README.md`
3. `configs/smoke_one_sr_dnn.yaml`
4. `scripts/smoke_one_sr_dnn.py`
5. `.ai-bridge/boosted_training_handoff.md`

Environment conclusion from your report:

- Host `lxplus901.cern.ch` has a Tesla T4 GPU with 15.6 GB VRAM.
- `nvidia-smi` works and GPU matmul passed.
- `LCG_110_cuda` provides Python 3.13.11, torch 2.11.0, CUDA 12.5, numpy, pandas, pyarrow, matplotlib, and PyYAML.
- Do not use bare system Python 3.9.
- Do not pip-install torch.

Use this venv policy:

```bash
cd /afs/cern.ch/work/l/lichengz/private/VHbb-Training/VHbb-Boost-Training

source /cvmfs/sft.cern.ch/lcg/views/LCG_110_cuda/x86_64-el9-gcc13-opt/setup.sh
python3 -m venv --system-site-packages .venv/lcg110-cuda
source .venv/lcg110-cuda/bin/activate
python3 -m pip install --upgrade pip
```

For every new shell after setup:

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_110_cuda/x86_64-el9-gcc13-opt/setup.sh
source .venv/lcg110-cuda/bin/activate
```

Verify:

```bash
python3 - <<'PY'
import torch, sys
print(sys.version)
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

Then run inspect-only first:

```bash
python3 scripts/smoke_one_sr_dnn.py \
  --config configs/smoke_one_sr_dnn.yaml \
  --region SR_Wenu_250_400_boosted_0J \
  --fraction 0.01 \
  --max-events-per-class 20000 \
  --inspect-only
```

Check output JSONs and feature plots. If the branch/feature/weight selection looks good, run:

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

Report back:

- whether venv setup succeeded;
- `torch.cuda.is_available()` and GPU name;
- exact command used;
- number of files/events discovered;
- selected weight column;
- selected event/fold column;
- selected features before and after preprocessing;
- validation loss/AUC from `metrics.json`;
- whether feature/score plots look reasonable;
- errors, warnings, or traceback.

## Implementation contract

- Work from this plan in small, reviewable steps.
- Keep edits scoped to the requested task and existing project conventions.
- Run focused verification before handing work back.
- Update .ai-bridge/agent-status.md with files touched, checks run, results, blockers, and review notes.
- Save the final review diff to .ai-bridge/implementation-diff.patch when practical.
- Append notable execution events to .ai-bridge/execution-log.jsonl when the implementation agent supports logging.
