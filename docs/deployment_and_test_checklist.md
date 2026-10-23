# Deployment and acceptance checklist

The checklists below are an evidence collection plan, not a claim that all
recipient tests have passed.

| Check | Expected result | Evidence to retain |
| --- | --- | --- |
| Open Job form | Correct final blue interface appears | URL and screenshot |
| Patient search | Intended admission and identifiers appear | Input ID and returned record |
| Score valid input | One request returns one finite probability between 0 and 1 | SAS JSON/log and matching screen |
| Input change | Old probability is cleared before new prediction | Before/after screenshots |
| 13% boundary | <=13% blue; >13% orange | Browser test record or representative outputs |
| Missing/invalid input | Prediction request is blocked or rejected clearly | Error example |
| Discharge snapshot | Last successful values reopen in same browser tab | Completion/reopen screenshots |
| Native Overview | Charts load and selected filters work | Representative screenshot and walkthrough |
| Another user's access | Intended recipient opens Report and Job | Name, date and access outcome |
| Repository delivery | Final files can be downloaded and README paths resolve | GitHub commit SHA and repository link |

Record test date, environment, tester, expected result, actual result, pass/fail,
and evidence path. Separate local browser checks using simulated SAS responses
from actual SAS execution. No clinical hospital chart is required for prototype
functional testing; use documented synthetic records and known input/output pairs.

Model location: `Models/READMISSION_GB.sashdat`.
Source table: `TUNDATA.CAPSTONE_2026_TRAIN_SET`.
Job: `/Users/uc_capstone_2026_s2_08@uni.canberra.edu.au/My Folder/Patient_Discharge_Workspace`.
Report URI: `/reports/reports/4717e92d-7502-436f-99ff-5bb03122cf5a`.

Report export status notes limited interactivity for the automated explanation
object: composite-object actions are not supported in the export. The live Job
remains dependent on SAS login, permissions, data, model and execution services.

The owner must confirm the final live report sharing URL and intended recipient.
Screenshots and `/reportImages/directImage` links are not interactive report links.
