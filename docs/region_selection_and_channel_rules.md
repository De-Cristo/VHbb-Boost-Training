# Region selection and channel-awareness rules

These rules are part of the boosted VHbb v1 training design. They are meant to avoid two common mistakes:

1. training on too little statistics by using only one small SR;
2. accidentally double-counting the same event by mixing overlapping SR region families.

## 1. Use merged SR families for sufficient statistics

A single SR such as:

```text
SR_Wenu_250_400_boosted_0J
```

is useful for a first technical smoke test, but it is not sufficient for a meaningful DNN benchmark. For the first realistic DNN smoke training, use all channels and jet bins in one pT family:

```text
SR_*_250_400_boosted_*J
```

This merges, for example, Wenu, Wmnu, Zee, Zmm, Znn and all available boosted jet-bin categories in the 250--400 pT(V) family.

## 2. The merged DNN should be channel-aware

When several channels are merged into one training phase space, the DNN should not be forced to infer the channel only from missing or zero-filled physics variables. It should receive explicit reconstructed-channel information.

Allowed channel inputs for the flat DNN include columns such as:

```text
events_LeptonCategory
events_isWLNuFlag
events_isZLLFlag
events_isZNuNuFlag
```

or derived one-hot columns from `region_name`, for example:

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

This is not considered truth leakage: these are reconstructed analysis categories that are also known in data. They should still be monitored carefully because they can dominate if the sample composition is inconsistent across channels.

For v1, the recommended policy is:

```text
channel metadata: allowed as DNN inputs
pT-bin metadata: monitoring only by default
jet-bin metadata: monitoring only by default
sample/process name: forbidden as model input
truth/gen/STXS labels: forbidden as model input
```

## 3. Do not mix overlapping pT-region families without de-duplication

Wildcard region matching is convenient, but it can accidentally select overlapping regions. The script/config must treat region families as physics selections, not just strings.

Recommended mutually exclusive starting families:

```text
250 < pT(V) < 400:
  SR_*_250_400_boosted_*J

pT(V) > 400:
  SR_*_GT400_boosted*
```

Do not combine these broad/inclusive patterns with narrower patterns unless explicit event-level de-duplication is implemented:

```text
SR_*_boosted*              + any narrower boosted pattern
SR_*_GT400_boosted*        + SR_*_400_600_boosted*
SR_*_GT400_boosted*        + SR_*_GT600_boosted*
inclusive boosted region   + exclusive pT or jet-bin region
```

The safe first training choice is therefore one of:

```text
--region 'SR_*_250_400_boosted_*J'
```

or:

```text
--region 'SR_*_GT400_boosted*'
```

but not a mixture such as:

```text
--region 'SR_*_boosted*'
```

unless the code explicitly proves event-level uniqueness.

## 4. Event de-duplication rule

If multiple region families must be combined later, the manifest builder must de-duplicate events using a stable event key. Preferred key:

```text
sample_name
events_run or run
events_luminosityBlock or luminosityBlock
events_EventNr or event
```

If only `events_EventNr` is available, use:

```text
sample_name
events_EventNr
```

with a warning that run/lumi are missing. The fallback key:

```text
sample_name
region_name
local_row_index
```

is acceptable for fold assignment only. It does not prove that overlapping region families have been de-duplicated.

## 5. Practical next training step

After the smoke script has verified CUDA and parquet access, the next useful training should be:

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

Check that `sample_summary.json` includes real V+jets backgrounds, not only diboson:

```text
signal_VH_Hbb
background_Vjets
background_diboson
```

Top backgrounds should also be included if their samples and matching SR directories are present.
