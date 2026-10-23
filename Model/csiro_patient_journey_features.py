"""Discharge-time readmission predictors grouped by their availability in a journey."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from joblib import load, dump
from csiro_gb_diagnosis_dynamics import model as base_model, add_dynamics, DYNAMICS
from csiro_gb_further_search import add_cadence, CADENCE
from csiro_gb_feature_expansion import dense, FEATURE_GROUPS, DAY_NS
from csiro_readmission_models import probability_scores, threshold_scores, f1_threshold

PRIOR_NUM=['PRIOR_COMPLETED_PROC_30D','PRIOR_COMPLETED_PROC_365D','PRIOR_RX_STARTS_30D','PRIOR_RX_STARTS_365D','ACTIVE_RX_CODES_AT_ADMISSION']
PROC_NUM=['CURRENT_IP_COMPLETED_PROC_RECORDS','CURRENT_IP_UNIQUE_PROC_CODES']
MED_NUM=['CURRENT_IP_RX_RECORDS','CURRENT_IP_UNIQUE_RX_CODES','CURRENT_IP_RX_ENDED_BY_DISCHARGE',
    'ACTIVE_RX_CODES_AT_DISCHARGE','NEW_ACTIVE_RX_CODES_VS_ADMISSION','RX_STARTS_LAST_30D_AT_DISCHARGE']
TEXTS=['CURRENT_IP_PROC_TEXT','CURRENT_IP_RX_TEXT','ACTIVE_RX_AT_DISCHARGE_TEXT']
GROUPS={
    'Existing history and diagnosis':([],[]),
    '+ Pre-admission treatment history':(PRIOR_NUM,[]),
    '+ Inpatient procedures':(PRIOR_NUM+PROC_NUM,['CURRENT_IP_PROC_TEXT']),
    '+ Inpatient medications and discharge state':(PRIOR_NUM+MED_NUM,['CURRENT_IP_RX_TEXT','ACTIVE_RX_AT_DISCHARGE_TEXT']),
    '+ Full treatment journey':(PRIOR_NUM+PROC_NUM+MED_NUM,TEXTS),
}


class SafeCodeVocabulary(BaseEstimator, TransformerMixin):
    """A fold with no sufficiently frequent codes contributes zero columns."""
    def __init__(self,min_df=10,max_features=40):
        self.min_df=min_df; self.max_features=max_features
    def fit(self,X,y=None):
        vector=CountVectorizer(binary=True,token_pattern=r'(?u)\b\d+\b',min_df=self.min_df,max_features=self.max_features)
        try:
            vector.fit(X); self.vectorizer_=vector
        except ValueError as exc:
            if 'empty vocabulary' not in str(exc) and 'no terms remain' not in str(exc): raise
            self.vectorizer_=None
        return self
    def transform(self,X):
        return self.vectorizer_.transform(X) if self.vectorizer_ is not None else csr_matrix((len(X),0))
    def get_feature_names_out(self,input_features=None):
        return self.vectorizer_.get_feature_names_out() if self.vectorizer_ is not None else np.array([],dtype=object)


def active_rx_codes(rows,when,strict_start=False):
    """Future end values affect neither the input nor the state beyond 'not ended yet'."""
    known=(rows.START<when) if strict_start else (rows.START<=when)
    return set(rows.loc[known & (rows.STOP.isna() | (rows.STOP>when)),'CODE'].dropna())


def inpatient_map(source,e):
    pieces=[]
    for c in pd.read_csv(source/'encounters.csv',usecols=['Id','PATIENT','START','STOP','ENCOUNTERCLASS'],dtype=str,chunksize=250000):
        pieces.append(c.loc[c.ENCOUNTERCLASS.eq('inpatient')].copy())
    x=pd.concat(pieces,ignore_index=True)
    for c in ['START','STOP']: x[c]=pd.to_datetime(x[c],utc=True,errors='coerce')
    x=x.loc[x.START.notna() & x.STOP.notna() & (x.STOP>=x.START)].sort_values(['PATIENT','START','STOP','Id'])
    assert not x.Id.duplicated().any()
    running=x.groupby('PATIENT',sort=False).STOP.cummax()
    last=running.groupby(x.PATIENT,sort=False).shift()
    new=last.isna() | (x.START>last)
    x['EPISODE_NUMBER']=new.groupby(x.PATIENT,sort=False).cumsum().astype(int)
    keys=x[['PATIENT','EPISODE_NUMBER']].drop_duplicates().reset_index(drop=True)
    assert np.array_equal(keys.to_numpy(),e[['PATIENT','EPISODE_NUMBER']].to_numpy())
    return x.set_index('Id')[['PATIENT','EPISODE_NUMBER']]


def read_events(path,patients,ip_map):
    parts=[]; scanned=0
    for c in pd.read_csv(path,usecols=['PATIENT','ENCOUNTER','START','STOP','CODE'],dtype=str,chunksize=250000):
        scanned+=len(c)
        c=c.loc[c.PATIENT.isin(patients)].copy()
        for col in ['START','STOP']: c[col]=pd.to_datetime(c[col],utc=True,errors='coerce')
        parts.append(c)
    x=pd.concat(parts,ignore_index=True)
    bad=x.START.isna() | (x.STOP.notna() & (x.STOP<x.START))
    audit=dict(file=path.name,all_records=scanned,patient_records=len(x),invalid_time_records=int(bad.sum()),missing_stop=int(x.STOP.isna().sum()))
    x=x.loc[~bad].copy()
    x['IP_EPISODE']=x.ENCOUNTER.map(ip_map.EPISODE_NUMBER)
    owner=x.ENCOUNTER.map(ip_map.PATIENT)
    x.loc[owner.notna() & owner.ne(x.PATIENT),'IP_EPISODE']=np.nan
    return x,audit


def prepare(root):
    root=Path(root); source=root/'Data'/'csiro_synthea_csv'
    prior=root/'Model'/'csiro_gb_feature_cache'/'expanded_episodes.csv.gz'
    cache=root/'Model'/'csiro_patient_journey_feature_cache'; cache.mkdir(exist_ok=True)
    fingerprint={str(p.relative_to(root)):{'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in [prior,source/'encounters.csv',source/'procedures.csv',source/'medications.csv']}
    cached=cache/'journey_episodes.csv.gz'
    if cached.exists() and (cache/'fingerprint.json').exists() and json.loads((cache/'fingerprint.json').read_text())==fingerprint:
        print('Using patient journey feature cache',flush=True)
        return pd.read_csv(cached,dtype={'PATIENT':str,'FIRST_REASONCODE':str,'PREVIOUS_REASONCODE':str,**{c:str for c in TEXTS+['ACTIVE_CODES_TEXT']}}),cache
    e=pd.read_csv(prior,dtype={'PATIENT':str,'FIRST_REASONCODE':str,'PREVIOUS_REASONCODE':str,'ACTIVE_CODES_TEXT':str})
    e=add_dynamics(add_cadence(e))
    for c in ['START','STOP']: e[c]=pd.to_datetime(e[c],utc=True)
    for c in PRIOR_NUM+PROC_NUM+MED_NUM: e[c]=0
    for c in TEXTS: e[c]=''
    print('Mapping raw inpatient encounters to merged episodes',flush=True)
    ip=inpatient_map(source,e); patients=set(e.PATIENT)
    indices=e.groupby('PATIENT',sort=False).groups
    audits=[]
    print('Streaming procedures; only completed records are eligible',flush=True)
    procedures,audit=read_events(source/'procedures.csv',patients,ip); audits.append(audit)
    procedures=procedures.loc[procedures.STOP.notna()]
    for patient,p in procedures.groupby('PATIENT',sort=False):
        idx=indices[patient]
        ends=np.sort(p.STOP.astype('int64').to_numpy())
        for i in idx:
            admission=e.at[i,'START']; discharge=e.at[i,'STOP']
            ns=admission.value
            for days in [30,365]:
                e.at[i,f'PRIOR_COMPLETED_PROC_{days}D']=np.searchsorted(ends,ns,'left')-np.searchsorted(ends,ns-days*DAY_NS,'left')
            current=p.loc[p.IP_EPISODE.eq(e.at[i,'EPISODE_NUMBER']) & (p.START>=admission) & (p.STOP<=discharge)]
            codes=set(current.CODE.dropna())
            e.at[i,'CURRENT_IP_COMPLETED_PROC_RECORDS']=len(current)
            e.at[i,'CURRENT_IP_UNIQUE_PROC_CODES']=len(codes)
            e.at[i,'CURRENT_IP_PROC_TEXT']=' '.join(sorted(codes))
    del procedures
    print('Streaming medication prescribing starts and point-in-time states',flush=True)
    medications,audit=read_events(source/'medications.csv',patients,ip); audits.append(audit)
    for patient,m in medications.groupby('PATIENT',sort=False):
        idx=indices[patient]; starts=np.sort(m.START.astype('int64').to_numpy())
        for i in idx:
            admission=e.at[i,'START']; discharge=e.at[i,'STOP']; a=admission.value; d=discharge.value
            for days in [30,365]:
                e.at[i,f'PRIOR_RX_STARTS_{days}D']=np.searchsorted(starts,a,'left')-np.searchsorted(starts,a-days*DAY_NS,'left')
            before=active_rx_codes(m,admission,strict_start=True); after=active_rx_codes(m,discharge)
            e.at[i,'ACTIVE_RX_CODES_AT_ADMISSION']=len(before)
            e.at[i,'ACTIVE_RX_CODES_AT_DISCHARGE']=len(after)
            e.at[i,'NEW_ACTIVE_RX_CODES_VS_ADMISSION']=len(after-before)
            e.at[i,'ACTIVE_RX_AT_DISCHARGE_TEXT']=' '.join(sorted(after))
            e.at[i,'RX_STARTS_LAST_30D_AT_DISCHARGE']=np.searchsorted(starts,d,'right')-np.searchsorted(starts,d-30*DAY_NS,'left')
            current=m.loc[m.IP_EPISODE.eq(e.at[i,'EPISODE_NUMBER']) & (m.START>=admission) & (m.START<=discharge)]
            codes=set(current.CODE.dropna())
            e.at[i,'CURRENT_IP_RX_RECORDS']=len(current)
            e.at[i,'CURRENT_IP_UNIQUE_RX_CODES']=len(codes)
            e.at[i,'CURRENT_IP_RX_ENDED_BY_DISCHARGE']=int((current.STOP.notna() & (current.STOP<=discharge)).sum())
            e.at[i,'CURRENT_IP_RX_TEXT']=' '.join(sorted(codes))
    e.to_csv(cached,index=False,compression='gzip')
    (cache/'fingerprint.json').write_text(json.dumps(fingerprint,indent=2),encoding='utf-8')
    pd.DataFrame(audits).to_csv(cache/'source_event_audit.csv',index=False)
    print('Treatment feature preparation completed',flush=True)
    return e,cache


def pipeline(params,extra_num,texts):
    p=base_model(params)
    pre=p.named_steps['preprocess']
    name,num,cols=pre.transformers[0]; pre.transformers[0]=(name,num,cols+extra_num)
    for text in texts:
        pre.transformers.append((text,Pipeline([('vocabulary',SafeCodeVocabulary(min_df=10,max_features=40)),
            ('dense',FunctionTransformer(dense,feature_names_out='one-to-one'))]),text))
    return p


def journey_codebook():
    num,cat,_=FEATURE_GROUPS['Expanded without current LOS']; existing=num+CADENCE+DYNAMICS+cat+['ACTIVE_CODES_TEXT']
    diagnosis=set(DYNAMICS+['FIRST_REASONCODE','ACTIVE_CODES_TEXT','CONFIRMED_CONDITION_CODE_COUNT_AT_DISCHARGE','CURRENT_REASON_PREVIOUS_COUNT'])
    rows=[]
    for c in existing:
        stage='2 Admission' if c in ['AGE_AT_ADMISSION','GENDER','RACE','FIRST_DESCRIPTION'] else ('3 Diagnosis updates' if c in diagnosis else '1 Pre-admission history')
        rows.append(dict(Feature=c,Journey_stage=stage,Availability='by index discharge' if stage.startswith('3') else 'before / at admission',Source='existing timing-reviewed features'))
    for c in PRIOR_NUM: rows.append(dict(Feature=c,Journey_stage='1 Pre-admission history',Availability='strictly before admission',Source='procedures / medications'))
    for c in PROC_NUM+['CURRENT_IP_PROC_TEXT']: rows.append(dict(Feature=c,Journey_stage='4 Inpatient procedures',Availability='procedure STOP <= discharge; linked inpatient encounter',Source='procedures.csv'))
    for c in MED_NUM+['CURRENT_IP_RX_TEXT','ACTIVE_RX_AT_DISCHARGE_TEXT']:
        stage='5 Discharge medication state' if c in ['ACTIVE_RX_CODES_AT_DISCHARGE','NEW_ACTIVE_RX_CODES_VS_ADMISSION','ACTIVE_RX_AT_DISCHARGE_TEXT','RX_STARTS_LAST_30D_AT_DISCHARGE'] else '4 Inpatient prescribing'
        rows.append(dict(Feature=c,Journey_stage=stage,Availability='prescribed by discharge; only prior end events used',Source='medications.csv'))
    return pd.DataFrame(rows).sort_values(['Journey_stage','Feature'])


def run(root,resume_dir=None):
    root=Path(root).resolve(); e,cache=prepare(root)
    prior=sorted((root/'Model'/'csiro_gb_dynamics_outputs').glob('*/COMPLETE.json'))[-1].parent
    info=json.loads((prior/'selection_before_test.json').read_text())
    expanded=Path(info['source_previous_run']); old_info=json.loads((expanded/'selection_before_test.json').read_text())
    original=Path(old_info['previous_run'])
    outputs=(root/'Model'/'csiro_patient_journey_model_outputs').resolve()
    output=Path(resume_dir) if resume_dir is not None else outputs/datetime.now(ZoneInfo('Australia/Sydney')).strftime('%Y%m%d_%H%M%S')
    if not output.is_absolute(): output=root/output
    output=output.resolve()
    if output.parent!=outputs: raise ValueError('Resume folder must belong to this experiment output directory.')
    if (output/'COMPLETE.json').exists():
        print('Already completed:',output,flush=True)
        return output
    output.mkdir(parents=True,exist_ok=resume_dir is not None)
    journey_codebook().to_csv(output/'patient_journey_codebook.csv',index=False)
    e=e.loc[e.READMIT_30D_OBSERVED.notna()].reset_index(drop=True); y=e.READMIT_30D_OBSERVED.astype(int)
    assignments=pd.read_csv(original/'csiro_split_assignment.csv.gz',dtype={'group_id':str}).sort_values('row_index').reset_index(drop=True)
    assert e.PATIENT.equals(assignments.group_id)
    train=np.flatnonzero(assignments['split'].eq('Train')); val=np.flatnonzero(assignments['split'].eq('Validation')); test=np.flatnonzero(assignments['split'].eq('Test'))
    num,cat,_=FEATURE_GROUPS['Expanded without current LOS']; base_num=num+CADENCE+DYNAMICS
    all_features=base_num+PRIOR_NUM+PROC_NUM+MED_NUM+cat+['ACTIVE_CODES_TEXT']+TEXTS
    X=e[all_features].copy()
    for c in cat: X[c]=X[c].fillna('<missing>').astype(str).str.replace(r'\.0$','',regex=True)
    for c in TEXTS+['ACTIVE_CODES_TEXT']: X[c]=X[c].fillna('')
    assert not {'NEXT_START','GAP_TO_NEXT_DAYS','DEATHDATE','LAST_RECORDED_CONTACT','TOTALCOST','DISPENSES'}&set(X.columns)
    rng=np.random.default_rng(42); ids=rng.choice(e.PATIENT.iloc[train].unique(),5000,replace=False)
    tune=train[e.PATIENT.iloc[train].isin(ids).to_numpy()]
    folds=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=42).split(X.iloc[tune],y.iloc[tune],e.PATIENT.iloc[tune]))
    for a,b in folds: assert not set(e.PATIENT.iloc[tune].iloc[a])&set(e.PATIENT.iloc[tune].iloc[b])
    assert not (set(e.PATIENT.iloc[train])&set(e.PATIENT.iloc[val]) or set(e.PATIENT.iloc[train])&set(e.PATIENT.iloc[test]) or set(e.PATIENT.iloc[val])&set(e.PATIENT.iloc[test]))
    params=json.loads(info['Parameters'])
    configuration=dict(parameters=params,features=all_features,baseline_run=str(prior),
        split_assignment=str(original/'csiro_split_assignment.csv.gz'),seed=42,cv_folds=3,
        tuning_patients=5000,code_vocabulary_max=40,code_min_df=10,
        source_fingerprint=json.loads((cache/'fingerprint.json').read_text()))
    config_path=output/'run_configuration.json'
    if config_path.exists() and json.loads(config_path.read_text())!=configuration:
        raise ValueError('Saved run configuration differs; do not mix experiments.')
    cv_path=output/'journey_group_cv.csv'
    rows=[]
    if resume_dir is not None and cv_path.exists():
        if (cache/'journey_episodes.csv.gz').stat().st_mtime_ns>cv_path.stat().st_mtime_ns:
            raise ValueError('Feature cache changed after saved CV; rerun CV in a new experiment.')
        if (original/'csiro_split_assignment.csv.gz').stat().st_mtime_ns>cv_path.stat().st_mtime_ns:
            raise ValueError('Patient split changed after saved CV.')
        prior_cv=pd.read_csv(cv_path)
        assert prior_cv.Experiment.tolist()==list(GROUPS)
        assert np.isfinite(prior_cv[['CV_AP','CV_AP_SD']].to_numpy()).all()
        for row,(extra,texts) in zip(prior_cv.to_dict('records'),GROUPS.values()):
            assert row['Added_numeric']==len(extra) and row['Added_code_sets']==len(texts)
        rows=prior_cv.to_dict('records')
        print('Reusing all five completed patient-group CV comparisons; no new CV search.',flush=True)
    else:
        for name,(extra,texts) in GROUPS.items():
            print('Patient-group CV journey ablation:',name,flush=True)
            cv=cross_validate(pipeline(params,extra,texts),X.iloc[tune],y.iloc[tune],cv=folds,scoring='average_precision',n_jobs=2,error_score='raise')
            rows.append(dict(Experiment=name,CV_AP=float(cv['test_score'].mean()),CV_AP_SD=float(cv['test_score'].std()),Added_numeric=len(extra),Added_code_sets=len(texts)))
            pd.DataFrame(rows).to_csv(cv_path,index=False)
            print('CV AP:',round(rows[-1]['CV_AP'],4),flush=True)
    config_path.write_text(json.dumps(configuration,indent=2),encoding='utf-8')
    winner=max(rows[1:],key=lambda r:r['CV_AP'])['Experiment']
    extra,texts=GROUPS[winner]
    print('Full training refit of CV-selected new treatment set:',winner,flush=True)
    fitted_path=output/'patient_journey_gb.joblib'
    fitted_metadata=output/'training_complete.json'
    if fitted_path.exists() and fitted_metadata.exists():
        assert json.loads(fitted_metadata.read_text())==configuration
        selected=load(fitted_path)
        print('Loaded completed training checkpoint.',flush=True)
    else:
        selected=pipeline(params,extra,texts)
        selected.named_steps['model'].set_params(verbose=1)
        selected.fit(X.iloc[train],y.iloc[train])
        dump(selected,fitted_path,compress=3)
        fitted_metadata.write_text(json.dumps(configuration,indent=2),encoding='utf-8')
        print('Full-training checkpoint saved before holdout evaluation.',flush=True)
    probability=selected.predict_proba(X.iloc[val])[:,1]; val_metrics=probability_scores(y.iloc[val],probability)
    threshold=f1_threshold(y.iloc[val],probability)
    frozen=dict(selected_by='patient-group training CV; best additional treatment set',winner=winner,Parameters=params,
        validation=val_metrics,previous_validation=info['validation'],retain_previous_if_validation_not_improved=bool(val_metrics['AP']<=info['validation']['AP']),
        added_numeric=extra,added_code_sets=texts,features_selected=base_num+extra+cat+['ACTIVE_CODES_TEXT']+texts,
        target='unchanged exploratory 30-day observed readmission',prediction_time='discharge',test_used_for_search=False,
        previous_test_already_reported=True,treatment_sources=['procedures.csv','medications.csv'],
        vocabulary_fitted_in_training_folds=True,source_fingerprint=json.loads((cache/'fingerprint.json').read_text()))
    (output/'selection_before_test.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    print('Selection frozen; validation AP:',val_metrics['AP'],flush=True)
    old=pd.read_csv(prior/'test_predictions.csv.gz',dtype={'PATIENT':str})
    assert np.array_equal(old.actual,y.iloc[test]) and np.array_equal(old.PATIENT,e.PATIENT.iloc[test])
    before=old.Diagnosis_dynamics_GB.to_numpy(); after=selected.predict_proba(X.iloc[test])[:,1]
    old_cut=float(pd.read_csv(prior/'test_comparison.csv').loc[lambda d:d.Threshold_rule.eq('Validation-F1') & d.Candidate.ne('Previous expanded GB'),'Threshold'].iloc[0])
    tests=[]
    for name,p,cut in [('Existing diagnosis-dynamics GB',before,old_cut),(winner,after,threshold)]:
        for rule,c in [('0.5',.5),('Validation-F1',cut)]: tests.append(dict(Experiment=name,Threshold_rule=rule,**probability_scores(y.iloc[test],p),**threshold_scores(y.iloc[test],p,c)))
    pd.DataFrame(tests).to_csv(output/'test_comparison.csv',index=False)
    pd.DataFrame({'PATIENT':e.PATIENT.iloc[test].values,'EPISODE_NUMBER':e.EPISODE_NUMBER.iloc[test].values,'actual':y.iloc[test].values,
        'baseline_GB_probability':before,'journey_GB_probability':after}).to_csv(output/'test_predictions.csv.gz',index=False,compression='gzip')
    group_codes,patients=pd.factorize(e.PATIENT.iloc[test]); gains=[]
    for _ in range(200):
        weights=np.bincount(rng.integers(0,len(patients),len(patients)),minlength=len(patients))[group_codes]
        gains.append(average_precision_score(y.iloc[test],after,sample_weight=weights)-average_precision_score(y.iloc[test],before,sample_weight=weights))
    gain=probability_scores(y.iloc[test],after)['AP']-probability_scores(y.iloc[test],before)['AP']
    importance_rows=rng.choice(val,min(5000,len(val)),replace=False)
    features=frozen['features_selected']
    perm=permutation_importance(selected,X.iloc[importance_rows][features],y.iloc[importance_rows],scoring='average_precision',n_repeats=2,random_state=42,n_jobs=1)
    pd.DataFrame({'Feature':features,'Validation_AP_drop':perm.importances_mean,'Repeat_SD':perm.importances_std}).sort_values('Validation_AP_drop',ascending=False).to_csv(output/'validation_importance.csv',index=False)
    dump(selected,output/'patient_journey_gb.joblib',compress=3)
    e[PRIOR_NUM+PROC_NUM+MED_NUM].describe().to_csv(output/'added_numeric_summary.csv')
    summary=dict(winner=winner,validation=val_metrics,baseline_test=probability_scores(y.iloc[test],before),journey_test=probability_scores(y.iloc[test],after),
        test_AP_gain=gain,paired_patient_gain_95=np.quantile(gains,[.025,.975]).tolist(),patient_overlap=0,
        added_features=extra+texts,clinical_validation=False,retain_previous_if_validation_not_improved=frozen['retain_previous_if_validation_not_improved'],
        warning='Procedure records are not surgery counts; RxNorm product counts are not ingredient counts or adherence. Planned admissions included; follow-up proxy unverified; old holdout reused.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (output/'COMPLETE.json').write_text(json.dumps({'completed':True}),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True); print('OUTPUT:',output,flush=True)
    return output


if __name__=='__main__': run(Path(__file__).resolve().parents[1])
