# Hospital Analytics and Discharge Review Handover

This package records the final SAS patient-discharge interface, its server-side
scoring program, the Visual Analytics report package, and selected modelling
experiments. The deployed patient interface uses the existing SAS Gradient
Boosting analytic store. The CAPSTONE LOS and CSIRO readmission experiments
provide analytical evidence and recommendations; they are separate from the
model called by the deployed dashboard.

## Start here

1. Open the [patient-discharge form](https://vfl-030.engage.sas.com/SASJobExecution/?_program=%2FUsers%2Fuc_capstone_2026_s2_08%40uni.canberra.edu.au%2FMy%20Folder%2FPatient_Discharge_Workspace&_action=form)
   with a SAS account that can access this personal-folder Job and its data/model.
2. Search a patient number, open an admission, review the inputs and run a prediction.
3. Use the education checklist and review the resource reference summaries.
4. A successful prediction records the workspace discharge snapshot in the same
   browser tab. This is not a write to a hospital discharge system or SAS source table.

The report URI is `/reports/reports/4717e92d-7502-436f-99ff-5bb03122cf5a`.
The report package includes Overview Dashboard and Patient Discharge pages.
The Overview is authored with native Visual Analytics objects. The patient page
embeds the HTML form through Data-driven content (DDC).

## Files

| Path | Purpose |
| --- | --- |
| `Model/sas_readmission_live/Patient_Discharge_Workspace_Blue.html` | Final blue HTML form; original two-column dashboard arrangement |
| `Model/sas_readmission_live/Patient_Discharge_API_Taylor_Scenario.sas` | Matching SAS search/load/score implementation |
| `Deliverable/Final_Dashboard.sasreportpkg.zip` | Exported report package, including report definition and exported report data |
| `Model/CAPSTONE_LOS_Ridge_Feature_Comparison.ipynb` | CAPSTONE Ridge feature comparison |
| `Model/LOS_Historical_Linear_Ridge_HistGB.ipynb` | Historical LOS fixed-model comparison |
| `Model/csiro_patient_journey_features.py` | CSIRO treatment and medication feature/model experiment |
| `docs/architecture.md` | Interaction, model-experiment and repository diagrams |
| `docs/deployment_and_test_checklist.md` | Deployment settings, tests and acceptance evidence still to collect |
| `docs/report_evidence.md` | Report claims, supporting results and limitations |
| `source_manifest.json` | File origins and SHA256 hashes for this handover snapshot |

## Deployment

1. Create or open the `Patient_Discharge_Workspace` SAS Job.
2. Paste the matching `.sas` file into the Job code editor, and attach the final
   blue `.html` file as the Job's HTML form.
3. Keep `_OUTPUT_TYPE=json` (Character) and the working Compute context setting
   (`SAS Job Execution compute context` in the tested account).
4. Use `_action=form` to display the form. JavaScript uses `_action=execute` for
   search, load and score requests and expects JSON responses.
5. Ensure the Job can access `TUNDATA.CAPSTONE_2026_TRAIN_SET` and the model
   `Models/READMISSION_GB.sashdat`. These server resources are not included here.
6. Point the patient report's DDC URL at the form URL. For VA table selection,
   send `PATIENT_NUMBER` and `ENCOUNTER_KEY` and filter to exactly one admission.

This handover was checked against local source files and the supplied report
package. The recipient must verify SAS permissions, published file versions,
and working report access in their own account. A personal-folder URL may
require sharing or relocation for another account.

## Data and research prerequisites

Full source datasets and fitted Python binary models are not bundled. Restore
the project's `Data/` and `EDA/` inputs and required preprocessing outputs before
rerunning the original experiments. `docs/research_inputs.md` lists the required
project locations. The notebooks are not standalone dataset distributions.
The Python module search paths used in the notebooks must remain available when
loading any separately supplied fitted pipelines.

`requirements-research.txt` lists required libraries without claiming a newly
tested clean installation. Original run manifests record actual versions.

## Important behaviour

- Probability >13% is orange; probability <=13% is blue. Actual model warnings
  may take precedence. The existing threshold is a review rule, not a newly
  clinically validated cutoff.
- Taylor (patient `500000616`, encounter `300069926`) is included using the
  original source values, with Procedure ICD `88.72` supplied as an explicit
  scenario because the original procedure value is missing. Editable scenario
  changes do not update the source. It must not be reported as an untouched case.
- Resource values are historical group medians and middle-50% ranges, not
  predictions of the patient's next admission or staffing orders.
- GitHub stores versioned source and documentation. No automatic SAS deployment
  or GitHub-hosted scoring service is configured.
- Model Manager accepted the Python model registration, but CAS external-language
  execution restrictions prevented successful scoring validation. Separate
  PROC PYTHON trials are not part of this final delivery architecture.

## Evidence status

The user confirmed GitHub authentication/push setup and SAS dashboard execution.
Repository content upload and recipient acceptance must be checked after this
package is uploaded. A report package is useful for exported presentation but
does not embed the Job code, live analytic store, or all live SAS dependencies.
Do not assume it is a complete server backup or an importable content migration.
