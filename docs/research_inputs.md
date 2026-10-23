# Research inputs and rerun prerequisites

The code retains its original project-relative paths. This handover includes
selected source and result evidence but does not duplicate the large source data
or all intermediate caches. Restore the full original project workspace for reruns.

## Historical LOS

- `EDA/READMIT_FE.xlsx`, sheet `READMIT_LOS` (2,976 profiles).
- `Model/los_ridge_export/split_assignments.csv`.
- `Model/los_ridge_export/model_manifest.json` for source fingerprint and prior metrics.
- Helper `Model/los_ridge_export/los_features.py`.

## CAPSTONE LOS

- `Data/raw/table_CAPSTONE_2026_TRAIN_SET.csv`.
- `Model/external_los_preprocessing_outputs/20261014_224830_037913/`:
  `preprocessing_manifest.json`, `combined_los_model_table.csv`,
  `combined_split_metadata.csv`.
- `Model/external_los_comparison_outputs/20261014_231637_044250/`:
  modelling manifest and original comparison table.
- HistGB evidence tables in the supplied evidence directory for comparison.
- The current CAPSTONE Ridge notebook validates the CAPSTONE source hash and
  analysis/split snapshot hashes. Old historical workbook changes do not prevent
  this CAPSTONE-only loader from running. Other original scripts keep stricter checks.

## CSIRO

- `Data/csiro_synthea_csv/`: original encounters, procedures, medications and
  other tables required by `csiro_readmission_preparation.py`.
- Patient-group split assignments in
  `Model/csiro_readmission_model_outputs/20261015_014411/`.
- Intermediate expanded episode and journey feature caches used by the original
  scripts must be restored or regenerated in their documented dependency order.
- Read each original script's entry point and source checks before rerunning;
  do not substitute a new split or vocabulary while claiming exact reproduction.

Python training outputs do not automatically update the final SAS analytic store.
If a new deployment is pursued later, verify inputs, target definition, model
identity, scoring parity and evaluation in that deployment environment separately.
