# Discharge education checklist: evidence and implementation

Reviewed 19 October 2026. This document describes the checklist in `Patient_Discharge_Workspace.html`.

## What a tick means

A tick means the selected discharge nurse has marked that topic as completed. The title is struck through and the nurse ID, local date/time and completion count are displayed. Unticking removes that completion entry. The third topic includes teach-back and support checks. At the user's request, a successful model prediction now completes the discharge workflow in this workspace; this is a UI workflow rule rather than evidence of a hospital discharge event.

Draft checklist completion is held in page memory, keyed by patient number and encounter key. A change to diagnosis, comorbidity index, discharge destination or chronic-condition count resets the draft checklist. Changing the selected nurse does not reattribute items already completed by another nurse. When scoring succeeds, the inputs, staff, score, checklist, education wording, resource display and completion time are captured in a read-only discharge snapshot. Snapshots use sessionStorage for the same browser tab and survive refreshes, subject to available storage. They are not written to SAS or a clinical record. Editing and unsuccessful rescoring do not replace the last successful snapshot.

## Selection logic

The recorded diagnosis ICD code selects the verified historical diagnosis group: CHF, PNU, AMI or COPD. The dashboard does not trust the previously inconsistent Capstone diagnosis-group label. The checklist covers the group rather than claiming different evidence-based instructions for every individual ICD code. Unsupported codes receive general education prompts, and model scoring remains subject to mapping validation.

The screen presents three checkboxes: (1) condition-specific precautions and worsening symptoms, (2) medicines and follow-up, and (3) other conditions, home support and understanding. Each topic has a collapsed **Explanation & sources** section. The underlying content is retained: two disease-specific explanations, three common discharge explanations and one history/coordination explanation. Medication doses, fluid targets, activity restrictions and treatment changes must come from the patient's agreed discharge plan.

| Group | Education topics | Source |
|---|---|---|
| CHF | Recognising worsening symptoms; following the agreed weight/symptom monitoring and individual diet/fluid plan | [NHLBI: Living with heart failure](https://www.nhlbi.nih.gov/health/heart-failure/living-with) |
| PNU | Prescribed treatment and gradual recovery; worsening symptoms and follow-up | [NHLBI: Pneumonia recovery](https://www.nhlbi.nih.gov/health/pneumonia/recovery) |
| AMI | Emergency response to suspected recurrence; medicines and cardiac rehabilitation | [NHLBI: Heart attack recovery](https://www.nhlbi.nih.gov/health/heart-attack/recovery) |
| COPD | Inhaler demonstration; flare-up plan and avoiding triggers | [NHLBI: Living with COPD](https://www.nhlbi.nih.gov/health/copd/living-with) |
| All groups | Medicine explanation, follow-up/results/contact arrangements, teach-back and practical support | [AHRQ IDEAL](https://www.ahrq.gov/patient-safety/patients-families/engagingfamilies/strategy4/index.html), [AHRQ RED Tool 3](https://www.ahrq.gov/patient-safety/settings/hospital/red/toolkit/redtool3.html) |
| History/coordination | Confirm actual conditions, treatment burden, preferences and care coordination | [NICE NG56 recommendations](https://www.nice.org.uk/guidance/ng56/chapter/recommendations) |

The dashboard contains paraphrased conversation suggestions, with a source link beside each item. They are materials for staff to adapt to the actual discharge plan, not a new validated clinical protocol.

## How the comorbidity index is used

The source does not establish that `COMORBIDITY_INDEX` is Charlson or another particular validated scale. It must not be treated as a diagnosis list, a count of conditions, or a basis for medication recommendations. It is separate from the model input `NUM_CHRONIC_COND`.

- Missing index: ask staff to confirm the other conditions and missing information.
- Zero index: still confirm the complete history.
- Positive index: show a coordination discussion covering actual conditions, conflicting advice, treatment difficulties, patient priorities and a contact responsible for coordination.

This numerical branch is a local interface rule. NICE supports individualised assessment and coordination, but does not validate this dataset's index or the dashboard's positive-index trigger. The former 1–2 / 3–4 / 5+ education tiers were removed. No essential common education is withheld for a lower index. Historical resource-reference buckets are a separate descriptive feature.

## Why some edits blocked scoring

The old diagnosis/ICD change handler cleared APR DRG. A later prediction attempt then stopped at input validation with `Complete or correct: drg`. Changing discharge destination alone did not clear DRG. The screenshots show the diagnosis changing from 428.21 to 428.31 while the discharge destination stays the same.

The current UI displays diagnosis, ICD and APR DRG as recorded, read-only values because there is no verified rule here for deriving a new APR DRG from a selected diagnosis alone. Procedure ICD code, APR DRG severity, operation count and order set used are also read-only source values. Discharge destination remains editable and uses the existing 11-category training mapping. Other inputs retain validation; for example ICU days cannot exceed length of stay, and a missing procedure code cannot be silently replaced by zero. Editing an input clears the previous score until Predict is run again.

The patient list labels the probability column **Readmission probability** and the status column **Discharge status**. At the user's request, the latest UI uses **Not discharged** until a successful prediction creates a workspace completion snapshot, then shows **Discharged** in the status and action. The Discharged button opens that snapshot. These labels describe this workspace's workflow; the source still has no actual discharge completion flag or timestamp. The prediction rule does not check or establish clinical readiness for discharge.

The local browser test uses a simulated SAS endpoint. It verifies checklist behaviour and request payloads, including all mapped discharge destinations. It does not execute the remote SAS model or establish clinical validity.
