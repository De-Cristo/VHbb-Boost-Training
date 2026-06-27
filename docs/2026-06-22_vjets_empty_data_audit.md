# Audit Report: Empty V+Jets Data in Current EOS Path
**Date:** 2026-06-22

## Overview
During the preparation for the boosted VHbb v1 DNN smoke/benchmark training using the `SR_*_250_400_boosted_*J` region family, it was identified that the `background_Vjets` process group was missing from the loaded dataframe and `sample_summary.json`. 

## Path Checked
**EOS columns_root:** `/eos/cms/store/group/phys_higgs/hbb/VHbbResults/Run3VHbbResults/ntuples/VHBB_Parquets/output_VHbb_STXS_boosted_svb_2024_0619_condor@lxplus/columns_final`

## Directory & File Statistics
- **Total sample directories:** 47 sample directories found.
- **Discovered parquet files:** 1,625 parquet files were found recursively across the `SR_*_250_400_boosted_*J` regions (including nested structure inside `DiJet_*` folders).
- **Parquet files by process group (approximate):**
  - **Diboson**: ~250 files
  - **Signal**: ~300 files
  - **V+jets**: ~1,075 files

## Row Count Audit
### Total Metadata Row Count (`pyarrow.parquet.ParquetFile.metadata.num_rows`)
A Python script iterated over **all** `nominal.parquet` files across all regions in the path:
- **Diboson (`background_diboson`)**: ~70,000 rows total (e.g., `WZtoLNu2Q_2024`: 32,872 rows; `ZZto2Nu2Q_2024`: 24,648 rows)
- **Signal (`signal_VH_Hbb` / `signal_VH_Hcc_optional`)**: ~35,000 rows total (e.g., `WplusH_Hto2B_WtoLNu_2024`: 12,427 rows)
- **V+jets (`background_Vjets`)**: **0 rows total** (all `WtoLNu-2Jets*`, `DYto2*`, `Zto2Nu-2Jets*`, `WtoENu*`, `WtoMuNu*`, `WtoTauNu*` samples evaluated to 0 rows).

### Loaded Pandas Row Count (for `SR_*_250_400_boosted_*J` @ 5% fraction sampling)
- **Diboson (`background_diboson`)**: 2,938 events
- **Signal (`signal_VH_Hbb`)**: 1,066 events
- **V+jets (`background_Vjets`)**: **0 events**

## Examples
**Populated Signal/Diboson Files:**
- `WZtoLNu2Q_2024` (32,872 rows)
- `WWtoLNu2Q_2024` (2,074 rows)
- `WplusH_Hto2B_WtoLNu_2024` (12,427 rows)

**Empty V+jets Files (Examples showing 0 rows / ~16KB schema only):**
- `/eos/cms/store/group/phys_higgs/hbb/VHbbResults/Run3VHbbResults/ntuples/VHBB_Parquets/output_VHbb_STXS_boosted_svb_2024_0619_condor@lxplus/columns_final/WtoLNu-2Jets_PTLNu-200to400_1J_FxFx_2024/DiJet_incl/SR_Wenu_250_400_boosted_0J/nominal.parquet`
- `/eos/cms/store/group/phys_higgs/hbb/VHbbResults/Run3VHbbResults/ntuples/VHBB_Parquets/output_VHbb_STXS_boosted_svb_2024_0619_condor@lxplus/columns_final/DYto2Mu-2Jets_FxFx_2024/DiJet_incl/SR_Zmm_250_400_boosted_0J/nominal.parquet`

## Conclusion
The `V+jets` datasets in the specified `columns_final` directory are entirely empty. The parquet files exist within the correctly structured region directories (e.g., inside `DiJet_incl`), allowing `rglob` pattern matching to discover them, but they are schema-only files containing zero rows of physics data. Meaningful DNN training requires populated V+jets data; thus, we must source an alternate production path.
