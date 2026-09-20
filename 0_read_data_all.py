import os
import pandas as pd

path = "/mnt/viya-share/data/courses/TUNDATA/readmit_historical.sas7bdat"

# print("Exists:", os.path.exists(path))
# print("File size MB:", os.path.getsize(path) / 1024**2)

df = pd.read_sas(
    path,
    format="sas7bdat",
    encoding="utf-8"
)

print("Shape:", df.shape)
print(df.head())
print(df.columns.tolist())