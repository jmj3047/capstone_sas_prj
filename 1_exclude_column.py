import os
import pandas as pd
import numpy as np

# ---------------------------------------------------
# 1. Load original data
# ---------------------------------------------------
path = "/mnt/viya-share/data/courses/TUNDATA/readmit_historical.sas7bdat"

df = pd.read_sas(
    path,
    format="sas7bdat",
    encoding="utf-8"
)

print("Original:", df.shape)


# ---------------------------------------------------
# 2. Basic cleaning
# ---------------------------------------------------

# Clean column names
df.columns = (
    df.columns
    .str.strip()
    .str.upper()
)

# Trim character columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

# Blank strings -> missing
df = df.replace(r"^\s*$", np.nan, regex=True)

# Infinite values -> missing
df = df.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------
# 3. Define clinical profile
# These five fields are NOT used to determine
# whether two rows represent the same clinical profile
# ---------------------------------------------------

exclude_cols = [
    "ENCOUNTER_KEY",
    "PATIENT_NUMBER",
    "HOSPITAL",
    "REGION",
    "DISCH_NURSE_ID"
]

clean_df = df.drop(
    columns=exclude_cols,
    errors="ignore"
).copy()

print("Original shape:", df.shape)
print("After dropping columns:", clean_df.shape)

print("\nRemaining columns:")
print(clean_df.columns.tolist())

print("\nPreview:")
print(clean_df.head())
