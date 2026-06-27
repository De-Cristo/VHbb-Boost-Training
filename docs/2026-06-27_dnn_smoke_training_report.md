# Boosted VHbb DNN Benchmark Training Report
**Date:** 2026-06-27
**Regions:** `SR_*_250_400_boosted_*J` AND `SR_*_GT400_boosted*`
**Fraction:** 100% (Full Dataset)

## 15-Point Summary of Results

1. **Process Groups Present:** 4 distinct process groups successfully loaded.
   - `background_Vjets`
   - `background_diboson`
   - `background_top`
   - `signal_VH_Hbb`

2. **Total Events Processed:** 366,467 events (Full 100% dataset).

3. **Event Breakdown by Process Group:**
   - **Top:** 261,950
   - **Diboson:** 58,469
   - **V+jets:** 24,515
   - **Signal (VHbb):** 21,533

4. **Weight Sums (Absolute):**
   - **Background (Label 0):** 8,703,187,402,777.8
   - **Signal (Label 1):** 236,621,093.1

5. **Weight Sums (Signed):**
   - **Background (Label 0):** 8,442,547,248,764.1
   - **Signal (Label 1):** 226,113,089.7

6. **Channel Indicators Included:** Yes, the model is channel-aware. The following flags were retained as features:
   - `events_LeptonCategory`
   - `events_isWLNuFlag`
   - `events_isZLLFlag`
   - `events_isZNuNuFlag`

7. **Feature Counts (Before):** 65 initial columns extracted from the parquet files.

8. **Feature Counts (After Preprocessing):** 33 features kept after dropping 1D arrays/constants.

9. **Fold Splitting:** Two-fold validation strategy successfully applied.
   - **Fold 0:** 182,974 events
   - **Fold 1:** 183,493 events

10. **Training Statistics:** 
    - `n_train`: 183,493
    - `n_val`: 182,974

11. **Model Performance - Best Validation AUC:** **0.9348** (Epoch 7)

12. **Model Performance - Best Validation Loss:** **0.2199** (Epoch 18)

13. **Model Performance - Best Training Loss:** 0.2519 (Epoch 20)

14. **Output Directory:** `outputs/smoke_one_sr_dnn/20260627_153256_SR___250_400_boosted__J_SR___GT400_boosted_`

15. **Conclusion & Next Steps:** 
    - Combining `250_400` and `GT400` pT families into a fully inclusive 100% training was a major success.
    - Massive increase in dataset size up to 366,467 events!
    - Peak validation AUC reached **0.9348**!
    - V+jets events climbed to a very healthy 24,515 events.
    - All plots and distributions are successfully generated.
