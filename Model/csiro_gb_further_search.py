"""Additional CSIRO-only cadence FE and two clearly named boosting families."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from joblib import dump
from csiro_gb_feature_expansion import feature_pipeline, FEATURE_GROUPS, DAY_NS
from csiro_readmission_models import probability_scores, threshold_scores, f1_threshold

CADENCE = ['PRIOR_GAP_MEAN_3', 'PRIOR_GAP_SD_3', 'PRIOR_GAP_MEAN_5', 'PRIOR_GAP_SD_5',
           'PRIOR_SHORT_RETURN_RATE_5', 'PRIOR_LOS_MEAN_3', 'CURRENT_VS_PRIOR_GAP_RATIO',
           'CURRENT_REASON_PREVIOUS_COUNT']


def add_cadence(e):
    """Uses only gaps ending in an admission before the current admission."""
    e=e.copy()
    assert e.sort_values(['PATIENT','EPISODE_NUMBER']).index.equals(e.index)
    g=e.groupby('PATIENT',sort=False)
    past_gap=g.DAYS_SINCE_PREV_DISCHARGE.shift()
    for width in [3,5]:
        e[f'PRIOR_GAP_MEAN_{width}']=past_gap.groupby(e.PATIENT,sort=False).transform(lambda x:x.rolling(width,min_periods=1).mean())
        e[f'PRIOR_GAP_SD_{width}']=past_gap.groupby(e.PATIENT,sort=False).transform(lambda x:x.rolling(width,min_periods=2).std())
    past_short=(past_gap<=30).astype(float).where(past_gap.notna())
    e['PRIOR_SHORT_RETURN_RATE_5']=past_short.groupby(e.PATIENT,sort=False).transform(lambda x:x.rolling(5,min_periods=1).mean())
    past_los=g.CSIRO_EPISODE_LOS_DAYS.shift()
    e['PRIOR_LOS_MEAN_3']=past_los.groupby(e.PATIENT,sort=False).transform(lambda x:x.rolling(3,min_periods=1).mean())
    e['CURRENT_VS_PRIOR_GAP_RATIO']=e.DAYS_SINCE_PREV_DISCHARGE/(1+e.PRIOR_GAP_MEAN_3)
    e['CURRENT_REASON_PREVIOUS_COUNT']=e.groupby(['PATIENT','FIRST_REASONCODE'],dropna=False,sort=False).cumcount()
    return e


def build(params, family='GradientBoostingClassifier', vocabulary=200):
    num,cat,codes=FEATURE_GROUPS['Expanded without current LOS']
    p=feature_pipeline(num+CADENCE,cat,codes)
    p.named_steps['preprocess'].transformers[2][1].named_steps['vocabulary'].set_params(max_features=vocabulary)
    if family=='Histogram Gradient Boosting':
        p.set_params(model=HistGradientBoostingClassifier(random_state=42,early_stopping=False,**params))
    else:
        p.named_steps['model'].set_params(**params)
    return p


def run(root, jobs=2):
    root=Path(root)
    previous=sorted((root/'Model'/'csiro_gb_expansion_outputs').glob('*/COMPLETE.json'))[-1].parent
    selection=json.loads((previous/'selection_before_test.json').read_text())
    old=json.loads((previous/'summary.json').read_text())
    original=Path(selection['previous_run'])
    stamp=datetime.now(ZoneInfo('Australia/Sydney')).strftime('%Y%m%d_%H%M%S')
    output=root/'Model'/'csiro_gb_further_outputs'/stamp
    output.mkdir(parents=True,exist_ok=False)
    print('Building past-only admission cadence features',flush=True)
    e=pd.read_csv(root/'Model'/'csiro_gb_feature_cache'/'expanded_episodes.csv.gz',
        dtype={'PATIENT':str,'FIRST_REASONCODE':str,'PREVIOUS_REASONCODE':str,'ACTIVE_CODES_TEXT':str})
    e=add_cadence(e)
    e=e.loc[e.READMIT_30D_OBSERVED.notna()].reset_index(drop=True)
    y=e.READMIT_30D_OBSERVED.astype(int)
    assignments=pd.read_csv(original/'csiro_split_assignment.csv.gz',dtype={'group_id':str}).sort_values('row_index').reset_index(drop=True)
    assert e.PATIENT.equals(assignments.group_id)
    train=np.flatnonzero(assignments['split'].eq('Train'))
    val=np.flatnonzero(assignments['split'].eq('Validation'))
    test=np.flatnonzero(assignments['split'].eq('Test'))
    assert not (set(e.PATIENT.iloc[train])&set(e.PATIENT.iloc[val]) or set(e.PATIENT.iloc[train])&set(e.PATIENT.iloc[test]) or set(e.PATIENT.iloc[val])&set(e.PATIENT.iloc[test]))
    num,cat,_=FEATURE_GROUPS['Expanded without current LOS']
    X=e[num+CADENCE+cat+['ACTIVE_CODES_TEXT']].copy()
    X['ACTIVE_CODES_TEXT']=X.ACTIVE_CODES_TEXT.fillna('')
    for c in cat:
        X[c]=X[c].fillna('<missing>').astype(str).str.replace(r'\.0$','',regex=True)
    assert not {'NEXT_START','GAP_TO_NEXT_DAYS','READMIT_30D_OBSERVED','DEATHDATE','LAST_RECORDED_CONTACT'}&set(X.columns)
    rng=np.random.default_rng(42)
    ids=rng.choice(e.PATIENT.iloc[train].unique(),size=5000,replace=False)
    tune=train[e.PATIENT.iloc[train].isin(ids).to_numpy()]
    folds=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=42).split(X.iloc[tune],y.iloc[tune],e.PATIENT.iloc[tune]))
    for a,b in folds:
        assert not set(e.PATIENT.iloc[tune].iloc[a])&set(e.PATIENT.iloc[tune].iloc[b])
    specs=[
        ('GB cadence/code200 - 300 trees','GradientBoostingClassifier',dict(n_estimators=300,learning_rate=.05,max_depth=4,min_samples_leaf=50,subsample=.8,max_features=.8)),
        ('GB cadence/code200 - 500 trees','GradientBoostingClassifier',dict(n_estimators=500,learning_rate=.03,max_depth=4,min_samples_leaf=50,subsample=.8,max_features=None)),
        ('Histogram GB - 300 / 15 leaves','Histogram Gradient Boosting',dict(max_iter=300,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=50,l2_regularization=1.0)),
        ('Histogram GB - 500 / 31 leaves','Histogram Gradient Boosting',dict(max_iter=500,learning_rate=.05,max_leaf_nodes=31,min_samples_leaf=50,l2_regularization=10.0)),
    ]
    rows=[]
    for name,family,params in specs:
        print('Train-only grouped CV:',name,flush=True)
        start=time.time()
        cv=cross_validate(build(params,family),X.iloc[tune],y.iloc[tune],cv=folds,
            scoring={'AP':'average_precision','ROC_AUC':'roc_auc'},n_jobs=jobs,error_score='raise')
        rows.append(dict(Candidate=name,Family=family,CV_AP=float(cv['test_AP'].mean()),
            CV_AP_SD=float(cv['test_AP'].std()),CV_ROC_AUC=float(cv['test_ROC_AUC'].mean()),
            Seconds=time.time()-start,Parameters=json.dumps(params)))
        pd.DataFrame(rows).to_csv(output/'training_group_cv.csv',index=False)
        print('CV AP:',round(rows[-1]['CV_AP'],4),flush=True)
    cv_table=pd.DataFrame(rows)
    selected_cv=cv_table.sort_values('CV_AP',ascending=False).groupby('Family',sort=False).head(1)
    fits,valid_rows={},[]
    for r in selected_cv.itertuples():
        print('Full training refit:',r.Candidate,flush=True)
        p=build(json.loads(r.Parameters),r.Family)
        p.fit(X.iloc[train],y.iloc[train])
        prob=p.predict_proba(X.iloc[val])[:,1]
        valid_rows.append(dict(Candidate=r.Candidate,Family=r.Family,**probability_scores(y.iloc[val],prob),Threshold=f1_threshold(y.iloc[val],prob)))
        fits[r.Candidate]=p
        pd.DataFrame(valid_rows).to_csv(output/'validation_candidates.csv',index=False)
    valid=pd.DataFrame(valid_rows)
    best=valid.sort_values('AP',ascending=False).iloc[0]
    native=valid.loc[valid.Family.eq('GradientBoostingClassifier')].iloc[0]
    frozen=dict(previous_model_validation_AP=selection['validation_AP'],previous_model_test_AP=old['expanded_test']['AP'],
        selected_additional_candidate=best.Candidate,selected_validation_AP=float(best.AP),
        improved_validation_over_previous=bool(best.AP>selection['validation_AP']),
        primary_original_GB_validation_AP=float(native.AP),tuning_rows=len(tune),
        target_unchanged=True,test_used_for_search=False,test_previously_reported=True,
        source_fingerprint=json.loads((root/'Model'/'csiro_gb_feature_cache'/'fingerprint.json').read_text()))
    (output/'selection_before_test.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    print('Settings frozen before final test',flush=True)
    reference=pd.read_csv(previous/'test_predictions.csv.gz',dtype={'PATIENT':str})
    assert np.array_equal(reference.actual,y.iloc[test]) and np.array_equal(reference.PATIENT,e.PATIENT.iloc[test])
    test_rows=[]
    predictions=pd.DataFrame({'PATIENT':e.PATIENT.iloc[test].values,'EPISODE_NUMBER':e.EPISODE_NUMBER.iloc[test].values,'actual':y.iloc[test].values,
        'Previous_expanded_GB':reference.expanded_GB_probability.values})
    prior_threshold=float(pd.read_csv(previous/'gb_before_after_test.csv').loc[lambda t:t.Threshold_rule.eq('Validation-F1') & ~t.Experiment.str.startswith('Original'),'Threshold'].iloc[0])
    for cut_name,cutoff in [('0.5',.5),('Validation-F1',prior_threshold)]:
        test_rows.append(dict(Candidate='Previous expanded GB',Family='GradientBoostingClassifier',Threshold_rule=cut_name,
            **probability_scores(y.iloc[test],reference.expanded_GB_probability),**threshold_scores(y.iloc[test],reference.expanded_GB_probability,cutoff)))
    for r in valid.itertuples():
        prob=fits[r.Candidate].predict_proba(X.iloc[test])[:,1]
        predictions[r.Candidate]=prob
        for cut_name,cutoff in [('0.5',.5),('Validation-F1',r.Threshold)]:
            test_rows.append(dict(Candidate=r.Candidate,Family=r.Family,Threshold_rule=cut_name,
                **probability_scores(y.iloc[test],prob),**threshold_scores(y.iloc[test],prob,cutoff)))
        dump(fits[r.Candidate],output/('original_gb_model.joblib' if r.Family=='GradientBoostingClassifier' else 'histogram_gb_model.joblib'),compress=3)
    tests=pd.DataFrame(test_rows)
    tests.to_csv(output/'test_comparison.csv',index=False)
    predictions.to_csv(output/'test_predictions.csv.gz',index=False,compression='gzip')
    tests=tests.loc[tests.Threshold_rule.eq('Validation-F1')]
    summary=dict(previous_GB_test_AP=old['expanded_test']['AP'],previous_GB_validation_AP=selection['validation_AP'],
        results=tests[['Candidate','Family','AP','ROC_AUC','F1']].to_dict('records'),
        patient_overlap=0,clinical_validation=False,selection=frozen,
        caveat='CSIRO all-cause target differs from historical readmitted; test is reused, not pristine external validation.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (output/'COMPLETE.json').write_text(json.dumps({'completed':True}),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    print('OUTPUT:',output,flush=True)
    return output


if __name__=='__main__':
    run(Path(__file__).resolve().parents[1])
