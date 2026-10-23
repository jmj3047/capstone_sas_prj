"""CSIRO-only GB feature ablations with a frozen, patient-disjoint holdout.

All histories are computed before the index admission. Conditions are evaluated
as of discharge. No billing, future encounters or next-admission data enter X.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from csiro_readmission_models import C_NUM, C_CAT, probability_scores, threshold_scores, f1_threshold

ADMISSION_CAT = ['FIRST_DESCRIPTION', 'FIRST_REASONCODE', 'RACE']
HISTORY_NUM = ['DAYS_SINCE_PREV_DISCHARGE', 'PREVIOUS_IP_LOS_DAYS', 'PREV_GAP_BETWEEN_ADMISSIONS',
    'PRIOR_IP_30D', 'PRIOR_IP_90D', 'PRIOR_IP_365D', 'PRIOR_ED_30D', 'PRIOR_ED_365D',
    'PRIOR_URGENTCARE_30D', 'PRIOR_OUTPATIENT_30D', 'PRIOR_AMBULATORY_30D']
HISTORY_CAT = ['PREVIOUS_REASONCODE']
DAY_NS = 86400 * 10**9
SAFE_C_NUM = ['CONFIRMED_CONDITION_CODE_COUNT_AT_DISCHARGE' if x=='ACTIVE_CONDITION_CODES_AT_DISCHARGE' else x for x in C_NUM]
FEATURE_GROUPS = {
    'Baseline 5 (timing-reviewed count)': (SAFE_C_NUM, C_CAT, False),
    '+ Admission type and reason': (SAFE_C_NUM, C_CAT + ADMISSION_CAT, False),
    '+ Prior utilisation history': (SAFE_C_NUM + HISTORY_NUM, C_CAT + ADMISSION_CAT + HISTORY_CAT, False),
    '+ Active diagnosis codes': (SAFE_C_NUM + HISTORY_NUM, C_CAT + ADMISSION_CAT + HISTORY_CAT, True),
    'Expanded without current LOS': ([x for x in SAFE_C_NUM if x != 'CSIRO_EPISODE_LOS_DAYS'] + HISTORY_NUM,
        C_CAT + ADMISSION_CAT + HISTORY_CAT, True),
}


def lookback_counts(event_ns, index_ns, days):
    """[admission - days, admission), no index/future visit counted."""
    event_ns = np.sort(np.asarray(event_ns, dtype=np.int64))
    return np.searchsorted(event_ns, index_ns, side='left') - np.searchsorted(event_ns, index_ns-days*DAY_NS, side='left')


def dense(x):
    return x.toarray()


def known_condition_mask(c, discharge):
    """Require the documenting encounter to have completed by discharge."""
    day = discharge.tz_convert('Australia/Brisbane').tz_localize(None).normalize()
    return ((c.DOCUMENTED_AT <= discharge) & (c.DOCUMENTATION_VISIT_COMPLETED_AT <= discharge) &
        (c.START <= day) & (c.STOP.isna() | (c.STOP >= day)))


def feature_pipeline(numeric, categorical, with_codes, params=None):
    transforms = [
        ('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric),
        ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=10, sparse_output=False), categorical),
    ]
    if with_codes:
        transforms.append(('codes', Pipeline([
            ('vocabulary', CountVectorizer(binary=True, token_pattern=r'(?u)\b\d+\b', max_features=60, min_df=20)),
            ('dense', FunctionTransformer(dense, feature_names_out='one-to-one')),
        ]), 'ACTIVE_CODES_TEXT'))
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=.1, max_depth=2,
        min_samples_leaf=20, random_state=42)
    if params:
        model.set_params(**params)
    return Pipeline([('preprocess', ColumnTransformer(transforms, sparse_threshold=0)), ('model', model)])


def prepare_features(root, cache):
    root, cache = Path(root), Path(cache)
    source = root/'Data'/'csiro_synthea_csv'
    episode_path = root/'Model'/'csiro_readmission_preparation_outputs'/'csiro_inpatient_episodes.csv.gz'
    fingerprint = {str(f.relative_to(root)): {'size': f.stat().st_size, 'mtime_ns': f.stat().st_mtime_ns}
        for f in [episode_path, source/'encounters.csv', source/'conditions.csv']}
    fingerprint['feature_version'] = 2
    cached = cache/'expanded_episodes.csv.gz'
    if cached.exists() and (cache/'fingerprint.json').exists():
        if json.loads((cache/'fingerprint.json').read_text()) == fingerprint:
            print('Using matching feature cache', flush=True)
            return pd.read_csv(cached, dtype={'PATIENT':str, 'FIRST_REASONCODE':str, 'PREVIOUS_REASONCODE':str, 'ACTIVE_CODES_TEXT':str}).fillna({'ACTIVE_CODES_TEXT':''})
    cache.mkdir(parents=True, exist_ok=True)
    legacy_cache = cache/'expanded_episodes.csv.gz'
    reusable_history = legacy_cache.exists() and (cache/'fingerprint.json').exists()
    if reusable_history:
        old_fingerprint = json.loads((cache/'fingerprint.json').read_text())
        reusable_history = all(old_fingerprint.get(k)==v for k,v in fingerprint.items() if k!='feature_version')
    if reusable_history:
        e = pd.read_csv(legacy_cache, dtype={'PATIENT':str,'FIRST_REASONCODE':str,'PREVIOUS_REASONCODE':str})
        for col in ['START','STOP']:
            e[col] = pd.to_datetime(e[col],utc=True)
        e = attach_confirmed_conditions(e, source)
        e.to_csv(cached,index=False,compression='gzip')
        (cache/'fingerprint.json').write_text(json.dumps(fingerprint,indent=2),encoding='utf-8')
        return e
    e = pd.read_csv(episode_path, dtype={'PATIENT':str, 'FIRST_REASONCODE':str})
    for col in ['START', 'STOP']:
        e[col] = pd.to_datetime(e[col], utc=True)
    assert e.sort_values(['PATIENT','EPISODE_NUMBER']).index.equals(e.index)
    groups = e.groupby('PATIENT', sort=False)
    previous_stop = groups.STOP.shift()
    e['DAYS_SINCE_PREV_DISCHARGE'] = (e.START-previous_stop).dt.total_seconds()/86400
    e['PREVIOUS_IP_LOS_DAYS'] = groups.CSIRO_EPISODE_LOS_DAYS.shift()
    e['PREV_GAP_BETWEEN_ADMISSIONS'] = e.DAYS_SINCE_PREV_DISCHARGE.groupby(e.PATIENT, sort=False).shift()
    e['PREVIOUS_REASONCODE'] = groups.FIRST_REASONCODE.shift()
    for col in ['PRIOR_IP_30D','PRIOR_IP_90D','PRIOR_IP_365D']:
        e[col] = 0
    for patient, idx in groups.groups.items():
        index_ns = e.loc[idx,'START'].astype('int64').to_numpy()
        for days in [30,90,365]:
            e.loc[idx,f'PRIOR_IP_{days}D'] = lookback_counts(index_ns,index_ns,days)
    patient_ids = set(e.PATIENT)
    classes = {'emergency': [('PRIOR_ED_30D',30),('PRIOR_ED_365D',365)],
        'urgentcare':[('PRIOR_URGENTCARE_30D',30)], 'outpatient':[('PRIOR_OUTPATIENT_30D',30)],
        'ambulatory':[('PRIOR_AMBULATORY_30D',30)]}
    parts = []
    print('Scanning prior ED/urgent/outpatient/ambulatory contacts', flush=True)
    for c in pd.read_csv(source/'encounters.csv', usecols=['PATIENT','START','ENCOUNTERCLASS'], dtype=str, chunksize=250000):
        c = c.loc[c.PATIENT.isin(patient_ids) & c.ENCOUNTERCLASS.isin(classes)].copy()
        c['START'] = pd.to_datetime(c.START, utc=True, errors='coerce')
        parts.append(c.dropna(subset=['START']))
    contacts = pd.concat(parts, ignore_index=True)
    for fields in classes.values():
        for col,_ in fields:
            e[col] = 0
    index_groups = groups.groups
    for (patient, kind), contact in contacts.groupby(['PATIENT','ENCOUNTERCLASS'], sort=False):
        idx = index_groups[patient]
        index_ns = e.loc[idx,'START'].astype('int64').to_numpy()
        event_ns = contact.START.astype('int64').to_numpy()
        for col,days in classes[kind]:
            e.loc[idx,col] = lookback_counts(event_ns,index_ns,days)
    del contacts, parts
    e = attach_confirmed_conditions(e, source)
    e.to_csv(cached, index=False, compression='gzip')
    (cache/'fingerprint.json').write_text(json.dumps(fingerprint,indent=2), encoding='utf-8')
    print('Feature preparation completed', flush=True)
    return e


def attach_confirmed_conditions(e, source):
    print('Scanning diagnosis documentation timestamps', flush=True)
    patient_ids = set(e.PATIENT)
    index_groups = e.groupby('PATIENT',sort=False).groups
    parts = []
    for c in pd.read_csv(source/'conditions.csv', usecols=['PATIENT','START','STOP','CODE','ENCOUNTER'], dtype=str, chunksize=250000):
        c = c.loc[c.PATIENT.isin(patient_ids)].copy()
        for col in ['START','STOP']:
            c[col] = pd.to_datetime(c[col], errors='coerce')
        parts.append(c)
    conditions = pd.concat(parts, ignore_index=True)
    encounter_ids = set(conditions.ENCOUNTER.dropna())
    times = []
    for c in pd.read_csv(source/'encounters.csv',usecols=['Id','START','STOP'],dtype=str,chunksize=250000):
        c = c.loc[c.Id.isin(encounter_ids)].copy()
        c['DOCUMENTED_AT'] = pd.to_datetime(c.START,utc=True,errors='coerce')
        c['DOCUMENTATION_VISIT_COMPLETED_AT'] = pd.to_datetime(c.STOP,utc=True,errors='coerce')
        times.append(c[['Id','DOCUMENTED_AT','DOCUMENTATION_VISIT_COMPLETED_AT']])
    documented = pd.concat(times,ignore_index=True)
    assert not documented.Id.duplicated().any()
    conditions = conditions.merge(documented,left_on='ENCOUNTER',right_on='Id',how='left',validate='many_to_one')
    condition_groups = {p:c for p,c in conditions.groupby('PATIENT',sort=False)}
    day = e.STOP.dt.tz_convert('Australia/Brisbane').dt.tz_localize(None).dt.normalize()
    code_text = np.empty(len(e), dtype=object)
    code_text[:] = ''
    for patient,idx in index_groups.items():
        c = condition_groups.get(patient)
        if c is None:
            continue
        for i in idx:
            # Same-day resolution time is unknown: retain as possibly active,
            # rather than using a later-on-the-same-day resolution to exclude it.
            active = c.loc[known_condition_mask(c,e.at[i,'STOP']),'CODE'].dropna().unique()
            code_text[i] = ' '.join(sorted(active))
    e['ACTIVE_CODES_TEXT'] = code_text
    counts = e.ACTIVE_CODES_TEXT.map(lambda x: len(x.split()))
    e['CONFIRMED_CONDITION_CODE_COUNT_AT_DISCHARGE'] = counts
    print('Timing-reviewed condition-count rows differing from legacy date-only count:',
        counts.ne(e.ACTIVE_CONDITION_CODES_AT_DISCHARGE).sum(),flush=True)
    return e


def run(root, cv_patients=5000, jobs=2):
    root = Path(root)
    completed = sorted((root/'Model'/'csiro_readmission_model_outputs').glob('*/COMPLETE.json'))
    previous = completed[-1].parent
    prior_manifest=json.loads((previous/'manifest.json').read_text())
    episode_source=root/'Model'/'csiro_readmission_preparation_outputs'/'csiro_inpatient_episodes.csv.gz'
    expected_hash=prior_manifest['source_hashes'][str(episode_source.relative_to(root))]
    assert hashlib.sha256(episode_source.read_bytes()).hexdigest()==expected_hash, 'Prepared source changed since the baseline. Rebuild the baseline first.'
    stamp = datetime.now(ZoneInfo('Australia/Sydney')).strftime('%Y%m%d_%H%M%S')
    output = root/'Model'/'csiro_gb_expansion_outputs'/stamp
    output.mkdir(parents=True, exist_ok=False)
    e = prepare_features(root, root/'Model'/'csiro_gb_feature_cache')
    e = e.loc[e.READMIT_30D_OBSERVED.notna()].reset_index(drop=True)
    y = e.READMIT_30D_OBSERVED.astype(int)
    split = pd.read_csv(previous/'csiro_split_assignment.csv.gz',dtype={'group_id':str})
    split = split.sort_values('row_index').reset_index(drop=True)
    assert len(e) == len(split) and e.PATIENT.equals(split.group_id)
    idx = {s: np.flatnonzero(split['split'].eq(s)) for s in ['Train','Validation','Test']}
    group_sets = [set(e.PATIENT.iloc[idx[s]]) for s in idx]
    assert not (group_sets[0]&group_sets[1] or group_sets[0]&group_sets[2] or group_sets[1]&group_sets[2])
    allowed = set(SAFE_C_NUM+C_CAT+ADMISSION_CAT+HISTORY_NUM+HISTORY_CAT+['ACTIVE_CODES_TEXT'])
    X = e[list(sorted(allowed))].copy()
    assert 'ACTIVE_CONDITION_CODES_AT_DISCHARGE' not in X.columns
    assert not any(c in X.columns for c in ['NEXT_START','GAP_TO_NEXT_DAYS','READMIT_30D_OBSERVED','LAST_RECORDED_CONTACT','DEATHDATE'])
    for col in set(SAFE_C_NUM+HISTORY_NUM):
        X[col] = pd.to_numeric(X[col],errors='coerce')
    for col in set(C_CAT+ADMISSION_CAT+HISTORY_CAT):
        X[col] = X[col].fillna('<missing>').astype(str).str.replace(r'\.0$', '', regex=True)
    X['ACTIVE_CODES_TEXT'] = X.ACTIVE_CODES_TEXT.fillna('')
    train,val,test = idx['Train'],idx['Validation'],idx['Test']
    training_patterns = pd.DataFrame({'Admission_type':e.FIRST_DESCRIPTION.iloc[train], 'readmitted':y.iloc[train]}).groupby('Admission_type',dropna=False).readmitted.agg(['size','sum','mean'])
    training_patterns.columns=['Training_episodes','Readmissions','Observed_readmission_rate']
    training_patterns.sort_values('Training_episodes',ascending=False).to_csv(output/'training_admission_type_patterns.csv')
    rows, models = [], {}
    for name,(numeric,categorical,codes) in FEATURE_GROUPS.items():
        print(f'Full-training GB ablation: {name}',flush=True)
        model = feature_pipeline(numeric,categorical,codes)
        start=time.time()
        model.fit(X.iloc[train],y.iloc[train])
        p = model.predict_proba(X.iloc[val])[:,1]
        models[name] = model
        rows.append(dict(Experiment=name, **probability_scores(y.iloc[val],p),
            Numeric_inputs=len(numeric), Categorical_inputs=len(categorical), Active_code_vocabulary=codes,
            Encoded_inputs=len(model.named_steps['preprocess'].get_feature_names_out()),
            Fit_seconds=time.time()-start, Parameters=json.dumps(model.named_steps['model'].get_params())))
        pd.DataFrame(rows).to_csv(output/'validation_ablations.csv',index=False)
        print(f'Validation AP: {rows[-1]["AP"]:.4f}',flush=True)
    ablation=pd.DataFrame(rows)
    expanded=ablation.loc[~ablation.Experiment.str.startswith('Baseline 5')].sort_values('AP',ascending=False).iloc[0].Experiment
    numeric,categorical,codes=FEATURE_GROUPS[expanded]
    rng=np.random.default_rng(42)
    ids=e.PATIENT.iloc[train].unique()
    if cv_patients and len(ids)>cv_patients:
        ids=rng.choice(ids,size=cv_patients,replace=False)
    tune=train[e.PATIENT.iloc[train].isin(ids).to_numpy()]
    cv=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=42).split(X.iloc[tune],y.iloc[tune],e.PATIENT.iloc[tune]))
    for a,b in cv:
        assert not set(e.PATIENT.iloc[tune].iloc[a]) & set(e.PATIENT.iloc[tune].iloc[b])
    grids=[
        {'model__n_estimators':[200], 'model__max_depth':[2], 'model__learning_rate':[.1], 'model__min_samples_leaf':[20]},
        {'model__n_estimators':[300], 'model__max_depth':[3], 'model__learning_rate':[.05], 'model__min_samples_leaf':[30], 'model__subsample':[.8]},
        {'model__n_estimators':[300], 'model__max_depth':[4], 'model__learning_rate':[.05], 'model__min_samples_leaf':[50], 'model__subsample':[.8], 'model__max_features':[.8]},
        {'model__n_estimators':[400], 'model__max_depth':[3], 'model__learning_rate':[.05], 'model__min_samples_leaf':[100], 'model__subsample':[.8]},
    ]
    print(f'Group-CV tuning best feature family: {expanded}; {len(tune):,} training-only rows',flush=True)
    search=GridSearchCV(feature_pipeline(numeric,categorical,codes),grids,cv=cv,scoring='average_precision',
        refit=False,n_jobs=jobs,error_score='raise')
    search.fit(X.iloc[tune],y.iloc[tune])
    pd.DataFrame(search.cv_results_).drop(columns=[c for c in search.cv_results_ if c.startswith('param_')]).to_csv(output/'gb_group_cv_search.csv',index=False)
    best=int(np.argmax(search.cv_results_['mean_test_score']))
    selected_params=search.cv_results_['params'][best]
    print('Refitting tuned GB on full training episodes',selected_params,flush=True)
    tuned=feature_pipeline(numeric,categorical,codes).set_params(**selected_params)
    tuned.fit(X.iloc[train],y.iloc[train])
    p_val=tuned.predict_proba(X.iloc[val])[:,1]
    tuned_name='Tuned: '+expanded
    models[tuned_name]=tuned
    rows.append(dict(Experiment=tuned_name,**probability_scores(y.iloc[val],p_val),
        Numeric_inputs=len(numeric),Categorical_inputs=len(categorical),Active_code_vocabulary=codes,
        Encoded_inputs=len(tuned.named_steps['preprocess'].get_feature_names_out()),
        Fit_seconds=np.nan,Parameters=json.dumps(selected_params)))
    validation=pd.DataFrame(rows).sort_values('AP',ascending=False)
    validation.to_csv(output/'validation_candidates.csv',index=False)
    winner=validation.iloc[0].Experiment
    selected=models[winner]
    (output/'selection_before_test.json').write_text(json.dumps({'selected_by':'validation AP', 'winner':winner,
        'validation_AP':float(validation.iloc[0].AP),'previous_run':str(previous),
        'training_rows':len(train),'validation_rows':len(val),'test_rows':len(test),
        'tuning_rows':len(tune),'cv_patients':cv_patients,'prediction_time':'discharge',
        'target':'unchanged exploratory 30-day observed all-cause readmission',
        'old_test_previously_reported':True,'test_used_for_search':False,
        'condition_timing_rule':'diagnosis encounter START and STOP <= index discharge; same-day resolution retained as uncertain',
        'legacy_date_only_count_used_as_input':False},indent=2),encoding='utf-8')
    threshold=f1_threshold(y.iloc[val],selected.predict_proba(X.iloc[val])[:,1])
    print('Selection frozen before final test:',winner,flush=True)
    p_test=selected.predict_proba(X.iloc[test])[:,1]
    old=pd.read_csv(previous/'csiro_test_predictions.csv.gz',dtype={'group_id':str})
    assert np.array_equal(old.row_index.to_numpy(),test)
    assert np.array_equal(old.actual.to_numpy(),y.iloc[test].to_numpy())
    assert np.array_equal(old.group_id.to_numpy(),e.PATIENT.iloc[test].to_numpy())
    baseline_probability=old['Gradient boosting'].to_numpy()
    before=probability_scores(y.iloc[test],baseline_probability)
    after=probability_scores(y.iloc[test],p_test)
    tests=[]
    for name,p,t in [('Original GB (5 inputs)',baseline_probability,
            float(pd.read_csv(previous/'csiro_validation_comparison.csv').set_index('Model').loc['Gradient boosting','Validation_F1_threshold'])),
            (winner,p_test,threshold)]:
        for rule,cutoff in [('0.5',.5),('Validation-F1',t)]:
            tests.append(dict(Experiment=name,Threshold_rule=rule,**probability_scores(y.iloc[test],p),
                **threshold_scores(y.iloc[test],p,cutoff)))
    pd.DataFrame(tests).to_csv(output/'gb_before_after_test.csv',index=False)
    pd.DataFrame({'PATIENT':e.PATIENT.iloc[test].values,'EPISODE_NUMBER':e.EPISODE_NUMBER.iloc[test].values,
        'actual':y.iloc[test].values,'baseline_GB_probability':baseline_probability,'expanded_GB_probability':p_test}).to_csv(output/'test_predictions.csv.gz',index=False,compression='gzip')
    # Paired patient cluster bootstrap, same resampled patients for both models.
    group_codes,unique=pd.factorize(e.PATIENT.iloc[test])
    deltas=[]
    for _ in range(200):
        mult=np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))
        weights=mult[group_codes]
        deltas.append(average_precision_score(y.iloc[test],p_test,sample_weight=weights)-
            average_precision_score(y.iloc[test],baseline_probability,sample_weight=weights))
    interval=np.quantile(deltas,[.025,.975])
    summary=dict(winner=winner,baseline_test=before,expanded_test=after,AP_gain=float(after['AP']-before['AP']),
        paired_patient_bootstrap_AP_gain_95_percent=[float(x) for x in interval],test_patients=len(unique),
        unchanged_target=True,patient_overlap=0,clinical_validation=False,
        limitations='Planned admissions included; negative follow-up proxy unverified; synthetic schedule/template effects possible; prior test was already reported.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    numeric,categorical,codes=FEATURE_GROUPS[expanded if winner==tuned_name else winner]
    raw_features=numeric+categorical+(['ACTIVE_CODES_TEXT'] if codes else [])
    importance=permutation_importance(selected,X.iloc[val][raw_features],y.iloc[val],
        scoring='average_precision',n_repeats=3,random_state=42,n_jobs=1)
    pd.DataFrame({'Feature':raw_features,'Validation_AP_drop':importance.importances_mean,
        'Repeat_SD':importance.importances_std}).sort_values('Validation_AP_drop',ascending=False).to_csv(output/'validation_importance.csv',index=False)
    if codes:
        vector=selected.named_steps['preprocess'].named_transformers_['codes'].named_steps['vocabulary']
        pd.DataFrame({'SNOMED_code':vector.get_feature_names_out()}).to_csv(output/'training_selected_code_vocabulary.csv',index=False)
    (output/'COMPLETE.json').write_text(json.dumps({'completed':True,'output':str(output)}),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    print('OUTPUT:',output,flush=True)
    return output


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--cv-patients',type=int,default=5000)
    parser.add_argument('--jobs',type=int,default=2)
    args=parser.parse_args()
    run(args.root,cv_patients=args.cv_patients or None,jobs=args.jobs)
