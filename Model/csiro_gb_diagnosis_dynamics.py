"""Past-only diagnosis-set changes and first-observed recency for native GB."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from joblib import dump
from csiro_gb_further_search import build, add_cadence, CADENCE
from csiro_gb_feature_expansion import FEATURE_GROUPS
from csiro_readmission_models import probability_scores, threshold_scores, f1_threshold

DYNAMICS=['NEW_CODES_VS_PREVIOUS','REMOVED_CODES_VS_PREVIOUS','CODE_JACCARD_PREVIOUS',
    'PRIOR_IDENTICAL_CODE_SET_EPISODES','DAYS_SINCE_NEWEST_CODE_FIRST_OBSERVED',
    'CODES_FIRST_OBSERVED_WITHIN_30D','DAYS_SINCE_CURRENT_REASON_FIRST_OBSERVED']


def add_dynamics(e):
    e=e.copy()
    assert e.sort_values(['PATIENT','EPISODE_NUMBER']).index.equals(e.index)
    stop=pd.to_datetime(e.STOP,utc=True)
    result=np.full((len(e),len(DYNAMICS)),np.nan)
    for _,idx in e.groupby('PATIENT',sort=False).groups.items():
        previous=None
        seen_codes={}
        seen_reasons={}
        identical=Counter()
        for i in idx:
            text=e.at[i,'ACTIVE_CODES_TEXT']
            codes=set(text.split()) if isinstance(text,str) else set()
            t=stop.iloc[i]
            if previous is not None:
                result[i,0]=len(codes-previous)
                result[i,1]=len(previous-codes)
                union=codes|previous
                result[i,2]=len(codes&previous)/len(union) if union else 1.0
            key=tuple(sorted(codes))
            result[i,3]=identical[key]  # excludes this episode
            ages=[]
            for code in codes:
                if code not in seen_codes:
                    seen_codes[code]=t
                ages.append((t-seen_codes[code]).total_seconds()/86400)
            if ages:
                result[i,4]=min(ages)
                result[i,5]=sum(d<=30 for d in ages)
            else:
                result[i,5]=0
            reason=str(e.at[i,'FIRST_REASONCODE'])
            if reason not in seen_reasons:
                seen_reasons[reason]=t
            result[i,6]=(t-seen_reasons[reason]).total_seconds()/86400
            identical[key]+=1
            previous=codes
    for i,c in enumerate(DYNAMICS): e[c]=result[:,i]
    return e


def model(params):
    p=build(params)
    numeric=p.named_steps['preprocess'].transformers[0][2]
    p.named_steps['preprocess'].transformers[0]=(p.named_steps['preprocess'].transformers[0][0],p.named_steps['preprocess'].transformers[0][1],numeric+DYNAMICS)
    return p


def run(root):
    root=Path(root)
    prior=sorted((root/'Model'/'csiro_gb_expansion_outputs').glob('*/COMPLETE.json'))[-1].parent
    info=json.loads((prior/'selection_before_test.json').read_text())
    old=json.loads((prior/'summary.json').read_text())
    original=Path(info['previous_run'])
    output=root/'Model'/'csiro_gb_dynamics_outputs'/datetime.now(ZoneInfo('Australia/Sydney')).strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True,exist_ok=False)
    print('Building diagnosis-set changes from observed past episodes',flush=True)
    e=pd.read_csv(root/'Model'/'csiro_gb_feature_cache'/'expanded_episodes.csv.gz',dtype={'PATIENT':str,'FIRST_REASONCODE':str,'PREVIOUS_REASONCODE':str,'ACTIVE_CODES_TEXT':str})
    e=add_dynamics(add_cadence(e))
    e=e.loc[e.READMIT_30D_OBSERVED.notna()].reset_index(drop=True)
    y=e.READMIT_30D_OBSERVED.astype(int)
    a=pd.read_csv(original/'csiro_split_assignment.csv.gz',dtype={'group_id':str}).sort_values('row_index').reset_index(drop=True)
    assert e.PATIENT.equals(a.group_id)
    train=np.flatnonzero(a['split'].eq('Train')); val=np.flatnonzero(a['split'].eq('Validation')); test=np.flatnonzero(a['split'].eq('Test'))
    num,cat,_=FEATURE_GROUPS['Expanded without current LOS']
    X=e[num+CADENCE+DYNAMICS+cat+['ACTIVE_CODES_TEXT']].copy()
    X['ACTIVE_CODES_TEXT']=X.ACTIVE_CODES_TEXT.fillna('')
    for c in cat: X[c]=X[c].fillna('<missing>').astype(str).str.replace(r'\.0$','',regex=True)
    rng=np.random.default_rng(42)
    ids=rng.choice(e.PATIENT.iloc[train].unique(),5000,replace=False)
    tune=train[e.PATIENT.iloc[train].isin(ids).to_numpy()]
    folds=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=42).split(X.iloc[tune],y.iloc[tune],e.PATIENT.iloc[tune]))
    for u,v in folds: assert not set(e.PATIENT.iloc[tune].iloc[u])&set(e.PATIENT.iloc[tune].iloc[v])
    assert not (set(e.PATIENT.iloc[train])&set(e.PATIENT.iloc[val]) or set(e.PATIENT.iloc[train])&set(e.PATIENT.iloc[test]) or set(e.PATIENT.iloc[val])&set(e.PATIENT.iloc[test]))
    specs=[dict(n_estimators=300,learning_rate=.05,max_depth=4,min_samples_leaf=50,subsample=.8,max_features=.8),
        dict(n_estimators=400,learning_rate=.05,max_depth=5,min_samples_leaf=50,subsample=.8,max_features=.8)]
    cv_rows=[]
    for params in specs:
        print('Native GB diagnosis dynamics grouped CV:',params,flush=True)
        scores=cross_validate(model(params),X.iloc[tune],y.iloc[tune],cv=folds,scoring='average_precision',n_jobs=1,error_score='raise')
        cv_rows.append(dict(Parameters=json.dumps(params),CV_AP=float(scores['test_score'].mean()),CV_AP_SD=float(scores['test_score'].std())))
        pd.DataFrame(cv_rows).to_csv(output/'training_group_cv.csv',index=False)
    best=max(cv_rows,key=lambda r:r['CV_AP'])
    selected=model(json.loads(best['Parameters']))
    print('Refitting native GB diagnosis dynamics on full training',flush=True)
    selected.fit(X.iloc[train],y.iloc[train])
    val_prob=selected.predict_proba(X.iloc[val])[:,1]
    validation=probability_scores(y.iloc[val],val_prob)
    cutoff=f1_threshold(y.iloc[val],val_prob)
    (output/'selection_before_test.json').write_text(json.dumps(dict(Parameters=best['Parameters'],validation=validation,
        selection='patient-group training CV',previous_validation_AP=info['validation_AP'],
        source_previous_run=str(prior),dynamics_features=DYNAMICS,test_used_for_search=False,
        definition='time first observed in known admission snapshots, not true disease onset'),indent=2),encoding='utf-8')
    print('Selection frozen before test; validation AP:',validation['AP'],flush=True)
    prob=selected.predict_proba(X.iloc[test])[:,1]
    reference=pd.read_csv(prior/'test_predictions.csv.gz',dtype={'PATIENT':str})
    assert np.array_equal(reference.actual,y.iloc[test]) and np.array_equal(reference.PATIENT,e.PATIENT.iloc[test])
    old_cut=float(pd.read_csv(prior/'gb_before_after_test.csv').loc[lambda d:d.Threshold_rule.eq('Validation-F1')&~d.Experiment.str.startswith('Original'),'Threshold'].iloc[0])
    tests=[]
    for name,p,c in [('Previous expanded GB',reference.expanded_GB_probability,old_cut),('Native GB with diagnosis dynamics',prob,cutoff)]:
        for rule,t in [('0.5',.5),('Validation-F1',c)]:
            tests.append(dict(Candidate=name,Threshold_rule=rule,**probability_scores(y.iloc[test],p),**threshold_scores(y.iloc[test],p,t)))
    pd.DataFrame(tests).to_csv(output/'test_comparison.csv',index=False)
    pd.DataFrame({'PATIENT':e.PATIENT.iloc[test].values,'EPISODE_NUMBER':e.EPISODE_NUMBER.iloc[test].values,'actual':y.iloc[test].values,
        'Previous_expanded_GB':reference.expanded_GB_probability.values,'Diagnosis_dynamics_GB':prob}).to_csv(output/'test_predictions.csv.gz',index=False,compression='gzip')
    dump(selected,output/'diagnosis_dynamics_gb_model.joblib',compress=3)
    summary=dict(previous_test_AP=old['expanded_test']['AP'],validation=validation,test=probability_scores(y.iloc[test],prob),
        validation_improved=bool(validation['AP']>info['validation_AP']),patient_overlap=0,
        clinical_validation=False,caveat='Observed synthetic all-cause readmission; previously reported test reused.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (output/'COMPLETE.json').write_text(json.dumps({'completed':True}),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    print('OUTPUT:',output,flush=True)
    return output


if __name__=='__main__': run(Path(__file__).resolve().parents[1])
