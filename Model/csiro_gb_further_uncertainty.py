"""Paired patient bootstrap for already-frozen additional candidate comparisons."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def run(root, draws=300):
    root=Path(root)
    f=sorted((root/'Model'/'csiro_gb_further_outputs').glob('*/COMPLETE.json'))[-1].parent
    d=sorted((root/'Model'/'csiro_gb_dynamics_outputs').glob('*/COMPLETE.json'))[-1].parent
    a=pd.read_csv(f/'test_predictions.csv.gz',dtype={'PATIENT':str})
    b=pd.read_csv(d/'test_predictions.csv.gz',dtype={'PATIENT':str})
    assert np.array_equal(a[['PATIENT','EPISODE_NUMBER','actual']].values,b[['PATIENT','EPISODE_NUMBER','actual']].values)
    codes,patients=pd.factorize(a.PATIENT)
    rng=np.random.default_rng(42)
    predictions={c:a[c].to_numpy() for c in a.columns if c not in ['PATIENT','EPISODE_NUMBER','actual','Previous_expanded_GB']}
    predictions['Native GB diagnosis dynamics']=b.Diagnosis_dynamics_GB.to_numpy()
    baseline=a.Previous_expanded_GB.to_numpy()
    y=a.actual.to_numpy()
    intervals={c:[] for c in predictions}
    for _ in range(draws):
        weights=np.bincount(rng.integers(0,len(patients),len(patients)),minlength=len(patients))[codes]
        before=average_precision_score(y,baseline,sample_weight=weights)
        for c,p in predictions.items():
            intervals[c].append(average_precision_score(y,p,sample_weight=weights)-before)
    rows=[]
    old=average_precision_score(y,baseline)
    for c,p in predictions.items():
        lower,upper=np.quantile(intervals[c],[.025,.975])
        rows.append(dict(Candidate=c,Test_AP=average_precision_score(y,p),AP_gain=average_precision_score(y,p)-old,
            Patient_bootstrap_gain_lower_95=float(lower),Patient_bootstrap_gain_upper_95=float(upper),
            Bootstrap_draws=draws,Test_patients=len(patients)))
    pd.DataFrame(rows).to_csv(f/'paired_patient_AP_gain_intervals.csv',index=False)
    print(pd.DataFrame(rows).round(5).to_string(index=False))


if __name__=='__main__': run(Path(__file__).resolve().parents[1])
