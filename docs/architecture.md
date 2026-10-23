# Architecture and interactions

## Deployed patient interface

```mermaid
sequenceDiagram
    actor User as Nurse or reviewer
    participant VA as SAS Visual Analytics DDC
    participant HTML as HTML discharge form
    participant Job as SAS Job Execution
    participant Source as CAPSTONE CAS table
    participant Model as SAS GB analytic store
    User->>VA: Open Patient Discharge page
    VA->>Job: Request _action=form
    Job-->>HTML: Serve associated HTML form
    User->>HTML: Search patient or open admission
    HTML->>Job: POST _action=execute, ui_action=search/load
    Job->>Source: Query eligible records
    Job-->>HTML: JSON patient records
    Note over HTML: Taylor scenario is prefilled locally
    User->>HTML: Review or edit model inputs
    HTML->>Job: POST ui_action=score, p1 through p21
    Job->>Job: Validate input and mapping
    Job->>Model: PROC ASTORE with submitted inputs
    Model-->>Job: Probability and warnings
    Job-->>HTML: JSON scoring response
    HTML-->>User: Probability and review status
    HTML->>HTML: Save completion snapshot in sessionStorage
```

The landing/input page in this diagram is the attached HTML form. No separate
externally hosted landing page has been verified. VA-to-HTML single-admission
messages are optional when the user searches within the HTML form.

## Native report and custom interface

```mermaid
flowchart LR
    Data[Prepared SAS data] --> Overview[Native drag and drop Overview Dashboard]
    Report[SAS Visual Analytics report] --> Overview
    Report --> DDC[Patient Discharge DDC object]
    DDC --> Form[HTML form]
    Form --> API[SAS Job search load score]
    API --> Data
    API --> Store[Existing READMISSION_GB.sashdat]
    REF[Embedded lookup and historical resource summaries] --> Form
```

The Overview's chart-specific data bindings should be read from the report
definition rather than assumed identical to the patient Job's table. Displayed
resource summaries are an HTML build-time snapshot. Doctor selection is a model
input; nurse selection supports workflow attribution but is not a model input.

## Research and deployment decision

```mermaid
flowchart TD
    Historical[Historical LOS profiles] --> LOS1[Linear Ridge HistGB comparison]
    Capstone[CAPSTONE records] --> LOS2[Ridge and HistGB feature experiments]
    Extra[Comorbidity monitoring care team variables] --> LOS2
    CSIRO[CSIRO synthetic episodes and history] --> Readmit[GB diagnosis history and treatment comparisons]
    LOS1 --> Evidence[Research results and recommendations]
    LOS2 --> Evidence
    Readmit --> Evidence
    Python[Python model package] --> MM[Model Manager registration]
    MM --> Limit[CAS Python scoring blocked by configuration]
    SASGB[Existing SAS GB] --> Final[Final discharge dashboard scoring]
```

Local experimental models have not been established as the model inside the
final SAS analytic store. Their metrics must not be attached to that store as
deployment validation. The observed CAS restriction concerns external-language
execution, not a proven universal prohibition on Python under the license.

## Source control and handover

```mermaid
flowchart LR
    Files[Final SAS HTML research code and report export] --> Commit[Review stage and commit]
    Commit --> Push[Push or GitHub upload]
    Push --> GitHub[GitHub capstone_sas_prj]
    GitHub --> Recipient[Recipient downloads source]
    Recipient --> Setup[Restore dependencies and configure SAS]
    Setup --> Test[Run acceptance checklist]
```

Changes committed through GitHub Upload files are already on GitHub; a separate
SAS Push is unnecessary for that upload. This is distinct from automatically
deploying the updated files to SAS, which remains a manual operation.
