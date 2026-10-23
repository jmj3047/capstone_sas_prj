# Evidence for report claims

## Model performance

| Experiment | Finding | Evidence directory |
| --- | --- | --- |
| CAPSTONE HistGB regression | MAE 4.4093 to 3.0491 days with all three variables; 30.85% reduction | `Model/los_three_feature_evidence_outputs/20261015_010113_208147` |
| CAPSTONE Ridge | MAE 4.7444 to 3.4245 days with all three variables; 27.82% reduction | `Model/capstone_ridge_feature_outputs/20261021_005852_898217` |
| CSIRO readmission | AP 0.889874 to 0.893296 after treatment/medication extension; about 0.34 percentage points | `Model/csiro_patient_journey_model_outputs/20261015_104338` |

These are research models. The final SAS dashboard uses the existing SAS GB;
its validation metrics have not been established by these Python experiment tables.
AP is average precision, not classification accuracy. LOS improvements describe
prediction-error reductions, not actual reductions in hospital stay.

The historical follow-up comparison found Ridge test MAE 5.5683 and HistGB 5.7572
days, while validation MAE favoured HistGB. Its Linear regression predictions were
numerically unstable and require diagnosis. Do not include its enormous error as
a valid model performance benchmark or treat the test winner as prospectively selected.
The original notebook and unmodified output table are retained for traceability.

Three CAPSTONE extra variables have unconfirmed timing/provenance. Report their
association with improved retrospective prediction without claiming all are
available at admission or required for every LOS model. The LOS target in that
experiment is ADMIT_LOS + ICU_DAYS, a project assumption needing confirmation.

## Delivery and assessment

The report rubric specifies a maximum 10-page final report and assigns 50/100
points to deliverable quality, including comprehensive evidence and organised
handover materials. It explicitly requests KPI/QA alignment and a Completed /
Modified / Not Delivered comparison with the Proposal and Plan.

The 20 October mentor meeting emphasises requirement-to-deliverable-to-test-to-
handover traceability, acceptance criteria, and functional and non-functional
quality. It does not require a newly invented RACI matrix. Reference actual
sprints, Planner, meeting minutes, risk and issue registers succinctly.

Relevant source locations in the project (not copied into this code package):
- `Deliverable/report rubric.pdf`
- `Meeting log/1020 [UC Capstone] Mentor session with SAS team.docx`
- `Capstone_drive1023/2. Sprint 2 - Proposal/Project Proposal and Plan Report.docx`

Do not claim retrospective tools were used prospectively or invent sponsor
acceptance, clinical benefit, data availability or successful advanced deployment.

## Remaining completion evidence

1. Repository content uploaded and final commit SHA recorded.
2. Final live Report and Job are accessible to the intended recipient.
3. Deployment files match the final local source hashes.
4. Representative SAS end-to-end tests and recipient walkthrough recorded.
5. Final report's citations and GenAI acknowledgement completed under UC requirements.
