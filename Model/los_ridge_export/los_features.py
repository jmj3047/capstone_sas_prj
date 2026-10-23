"""Input preparation and engineered features from TOTAL_LOS_FE_tuning.ipynb."""
import numpy as np
import pandas as pd

NUMERIC = ['PATIENT_AGE', 'NUM_CHRONIC_COND', 'NUM_VISITS']
CATEGORICAL = ['DOCTOR', 'DEPARTMENT', 'GENDER', 'Patient location',
               'RACE_CD', 'ICD9_TARGET', 'DIAGNOSIS_ICD_CODE']
FEATURES = NUMERIC + CATEGORICAL
NUMERIC_FE = ['LOG1P_VISITS', 'AGE_CENTRED_SQUARED', 'AGE_X_CHRONIC']
DIAGNOSIS_FE = ['ICD_FAMILY', 'CHRONIC_COUNT_BAND']

def prepare(frame):
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing model inputs: {sorted(missing)}')
    out = frame[FEATURES].copy()
    for col in NUMERIC:
        out[col] = pd.to_numeric(out[col], errors='coerce')
    for col in CATEGORICAL:
        out[col] = out[col].map(lambda v: '<missing>' if pd.isna(v) else str(v).strip())
    age, chronic, visits = (out[c] for c in NUMERIC)
    out['LOG1P_VISITS'] = np.log1p(visits.clip(lower=0))
    out['AGE_CENTRED_SQUARED'] = ((age - 75) / 10) ** 2
    out['AGE_X_CHRONIC'] = (age / 10) * chronic
    out['ICD_FAMILY'] = out['DIAGNOSIS_ICD_CODE'].map(
        lambda v: '<missing>' if v == '<missing>' else v.split('.')[0])
    out['CHRONIC_COUNT_BAND'] = chronic.map(
        lambda v: '<missing>' if pd.isna(v) else '0' if v == 0 else '1' if v == 1 else '2+')
    return out
